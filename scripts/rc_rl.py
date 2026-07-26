"""Experimental reverse-curriculum RL for one replay-verified SMB1 level.

This is a research prototype, not a reproduced result.  It combines exact
emulator resets along a demonstration with clipped double-Q learning and
demo/online replay.  Its design is loosely motivated by reverse curriculum
generation, Go-Explore, and RLPD, but this script does not establish the
convergence, robustness, or sample-efficiency claims of those methods.

There is no preserved report or checkpoint supporting the historical
``+391 / 0.22`` observation.  Every new run therefore requires explicit input
and output paths and records enough provenance to assess it independently.

Example:

    ./venv/bin/python scripts/rc_rl.py \
        --level 1-1 \
        --solution data/solutions/1-1.json \
        --seed 0 \
        --max-env-steps 250000 \
        --checkpoint-out runs/20260725-rc-rl-1-1/checkpoint.pt \
        --report runs/20260725-rc-rl-1-1/report.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from collections import deque
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mario.entity import OBS_DIM_ENTITY, entity_obs
from mario.env import MarioSim, N_ACTIONS
from mario.io import (
    env_fingerprint,
    git_rev,
    utc_now_iso,
    write_json_atomic,
)
from mario.ram import mario_level_x
from mario.reward import is_death, is_success
from mario.solution_verification import (
    SolutionVerification,
    verify_stock_solution,
)

DEVICE = "cpu"


@dataclass(frozen=True)
class FrontierState:
    """Emulator state plus the wrapper-side observation caches at that state."""

    emulator_state: object
    info: dict
    obs: object


class Replay:
    def __init__(self, rng: np.random.Generator, cap: int = 120_000):
        self.rng = rng
        self.s = deque(maxlen=cap)
        self.a = deque(maxlen=cap)
        self.r = deque(maxlen=cap)
        self.s2 = deque(maxlen=cap)
        self.d = deque(maxlen=cap)

    def add(self, s, a, r, s2, d) -> None:
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s2.append(s2)
        self.d.append(d)

    def __len__(self) -> int:
        return len(self.s)

    def sample(self, n: int):
        idx = self.rng.integers(0, len(self.s), size=n)
        states = np.asarray([self.s[i] for i in idx], np.float32)
        actions = np.asarray([self.a[i] for i in idx], np.int64)
        rewards = np.asarray([self.r[i] for i in idx], np.float32)
        next_states = np.asarray([self.s2[i] for i in idx], np.float32)
        terminals = np.asarray([self.d[i] for i in idx], np.float32)
        return states, actions, rewards, next_states, terminals


class QNet(nn.Module):
    def __init__(self, dim: int = OBS_DIM_ENTITY, hidden: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an experimental reverse-curriculum learner from a "
            "replay-verified stock SMB1 solution."
        ))
    parser.add_argument("--level", required=True, help="stock SMB1 level, e.g. 1-1")
    parser.add_argument(
        "--solution",
        default="",
        help="solution manifest (default: data/solutions/<level>.json)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-env-steps", type=int, default=250_000)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing explicitly named report/checkpoint artifacts")

    parser.add_argument("--curriculum-stages", type=int, default=24)
    parser.add_argument("--advance-window", type=int, default=20)
    parser.add_argument("--advance-successes", type=int, default=14)
    parser.add_argument("--max-episode-chunks", type=int, default=420)
    parser.add_argument("--horizon-scale", type=float, default=2.0)
    parser.add_argument("--horizon-slack", type=int, default=25)
    parser.add_argument("--eval-horizon-scale", type=float, default=3.0)
    parser.add_argument("--eval-horizon-slack", type=int, default=50)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--critics", type=int, default=2,
        help="number of Q critics; clipped double-Q requires at least 2")
    parser.add_argument("--replay-capacity", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.5)
    return parser


def parse_level(level: str) -> tuple[int, int]:
    try:
        world, stage = (int(part) for part in level.split("-"))
    except (TypeError, ValueError):
        raise ValueError(f"invalid stock SMB1 level {level!r}; expected W-S") from None
    if not 1 <= world <= 8 or not 1 <= stage <= 4:
        raise ValueError(f"stock SMB1 level out of range: {level!r}")
    return world, stage


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    world, stage = parse_level(args.level)
    positive_ints = (
        "max_env_steps",
        "curriculum_stages",
        "advance_window",
        "advance_successes",
        "max_episode_chunks",
        "horizon_slack",
        "eval_horizon_slack",
        "hidden",
        "critics",
        "replay_capacity",
        "batch_size",
        "utd",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.critics < 2:
        raise ValueError("--critics must be at least 2 for clipped double-Q")
    if args.advance_successes > args.advance_window:
        raise ValueError("--advance-successes cannot exceed --advance-window")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError("require 0 <= epsilon-end <= epsilon-start <= 1")
    if args.epsilon_decay_fraction <= 0:
        raise ValueError("--epsilon-decay-fraction must be positive")
    if args.horizon_scale <= 0 or args.eval_horizon_scale <= 0:
        raise ValueError("horizon scales must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if not 0 <= args.gamma <= 1:
        raise ValueError("--gamma must be in [0, 1]")
    if not 0 < args.polyak_tau <= 1:
        raise ValueError("--polyak-tau must be in (0, 1]")

    report = Path(args.report).expanduser().resolve()
    checkpoint = Path(args.checkpoint_out).expanduser().resolve()
    if report == checkpoint:
        raise ValueError("--report and --checkpoint-out must be distinct paths")
    if not args.overwrite:
        existing = [str(path) for path in (report, checkpoint) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing artifacts: " + ", ".join(existing))
    return world, stage


def set_all_seeds(seed: int) -> np.random.Generator:
    """Seed every PRNG used by this CPU-only prototype."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)


def resolve_solution_path(args: argparse.Namespace) -> Path:
    if args.solution:
        return Path(args.solution).expanduser().resolve()
    return (ROOT / "data" / "solutions" / f"{args.level}.json").resolve()


def file_ref(path: str | Path) -> dict:
    artifact = Path(path).expanduser().resolve()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    display_path = path_label(artifact)
    return {
        "path": display_path,
        "sha256": digest,
        "bytes": artifact.stat().st_size,
    }


def path_label(path: str | Path) -> str:
    artifact = Path(path).expanduser().resolve()
    try:
        return str(artifact.relative_to(ROOT))
    except ValueError:
        return str(artifact)


def load_verified_demo(
        solution_path: str | Path,
        expected_level: str,
        seed: int,
        verifier: Callable[..., SolutionVerification] = verify_stock_solution,
) -> tuple[list[int], int, SolutionVerification, dict]:
    """Load a manifest only after the repository's independent replay gate."""
    path = Path(solution_path)
    check = verifier(path, seed=seed)
    if check.level != expected_level:
        raise ValueError(
            f"solution level {check.level} does not match --level {expected_level}")
    if not check.replay_verified:
        raise ValueError(
            "refusing unverified demonstration: "
            f"status={check.status} reason={check.reason}")

    manifest = json.loads(path.read_text())
    actions = [int(action) for action in manifest["path"]]
    # Legacy stock manifests may omit this field; the shared verifier defines
    # their repository-wide compatibility default as eight frames.
    chunk_frames = int(manifest.get("chunk_frames", check.chunk_frames))
    if chunk_frames != check.chunk_frames:
        raise ValueError("manifest changed between verification and load")
    return actions, chunk_frames, check, manifest


def step_reward(x_prev: int, x_now: int, flag: bool) -> float:
    """Progress shaping plus a sparse flag bonus; no theorem is claimed."""
    reward = 0.02 * float(np.clip(x_now - x_prev, -16, 16))
    if flag:
        reward += 10.0
    return reward


def curriculum_frontiers(path_length: int, stages: int) -> tuple[list[int], int]:
    if path_length <= 0:
        raise ValueError("verified demonstration unexpectedly has no actions")
    delta = max(1, path_length // stages)
    frontiers = list(range(path_length - delta, -1, -delta))
    if not frontiers or frontiers[-1] != 0:
        frontiers.append(0)
    return frontiers, delta


def capture_frontier_states(
        sim,
        path: list[int],
        chunk_frames: int,
        frontiers: list[int],
        seed: int,
) -> dict[int, FrontierState]:
    """Replay one prefix and retain emulator plus Python-side cache state."""
    wanted = set(frontiers)
    max_frontier = max(wanted)
    states: dict[int, FrontierState] = {}
    sim.reset(seed)
    for t in range(max_frontier + 1):
        if t in wanted:
            obs = sim.last_obs
            states[t] = FrontierState(
                emulator_state=sim.snapshot(),
                info=dict(sim.last_info),
                obs=obs.copy() if hasattr(obs, "copy") else obs,
            )
        if t == max_frontier:
            break
        _info, done = sim.run_chunk(path[t], chunk_frames)
        if done:
            raise RuntimeError(
                f"demonstration terminated before curriculum frontier t={t + 1}")
    if states.keys() != wanted:
        raise RuntimeError("failed to capture every curriculum frontier")
    return states


def restore_frontier(sim, frontier: FrontierState) -> None:
    sim.restore(
        frontier.emulator_state,
        cached_info=frontier.info,
        cached_obs=frontier.obs,
    )


def seed_demo_buffer(
        sim,
        path: list[int],
        chunk_frames: int,
        seed: int,
        demo: Replay,
) -> None:
    sim.reset(seed)
    state = entity_obs(sim.ram, sim.last_info)
    x_prev = mario_level_x(sim.ram)
    for action in path:
        info, done = sim.run_chunk(action, chunk_frames)
        x_now = mario_level_x(sim.ram)
        flag = is_success(info)
        dead = is_death(info, done)
        next_state = entity_obs(sim.ram, info)
        demo.add(
            state,
            action,
            step_reward(x_prev, x_now, flag),
            next_state,
            float(flag or dead or done),
        )
        state = next_state
        x_prev = x_now
        if done:
            break


def verify_candidate_path(
        world: int,
        stage: int,
        path: list[int],
        chunk_frames: int,
        seed: int,
        verifier: Callable[..., SolutionVerification] = verify_stock_solution,
) -> SolutionVerification:
    """Independently replay a learned trajectory through the shared gate."""
    with tempfile.TemporaryDirectory(prefix="mario-rc-rl-verify-") as tmp:
        manifest = Path(tmp) / f"{world}-{stage}.json"
        write_json_atomic(manifest, {
            "solved": True,
            "path": [int(action) for action in path],
            "chunk_frames": int(chunk_frames),
        })
        return verifier(manifest, seed=seed)


def config_from_args(args: argparse.Namespace, chunk_frames: int) -> dict:
    return {
        "max_env_steps": args.max_env_steps,
        "chunk_frames": chunk_frames,
        "curriculum_stages": args.curriculum_stages,
        "advance_window": args.advance_window,
        "advance_successes": args.advance_successes,
        "max_episode_chunks": args.max_episode_chunks,
        "horizon_scale": args.horizon_scale,
        "horizon_slack": args.horizon_slack,
        "eval_horizon_scale": args.eval_horizon_scale,
        "eval_horizon_slack": args.eval_horizon_slack,
        "hidden": args.hidden,
        "critics": args.critics,
        "replay_capacity": args.replay_capacity,
        "batch_size": args.batch_size,
        "utd": args.utd,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "polyak_tau": args.polyak_tau,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_fraction": args.epsilon_decay_fraction,
        "device": DEVICE,
    }


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run_experiment(
        args: argparse.Namespace,
        world: int,
        stage: int,
        path: list[int],
        chunk_frames: int,
        solution_ref: dict,
        solution_check: SolutionVerification,
        rng: np.random.Generator,
        sim_factory: Callable[..., MarioSim] = MarioSim,
        candidate_verifier: Callable[..., SolutionVerification] = verify_stock_solution,
) -> tuple[dict, dict | None]:
    """Train once and return the truthful result plus optional checkpoint ref."""
    level = f"{world}-{stage}"
    path_length = len(path)
    frontiers, delta = curriculum_frontiers(
        path_length, args.curriculum_stages)
    print(
        f"[rc_rl] {level} cf={chunk_frames} verified solution len={path_length}; "
        f"{len(frontiers)} stages delta={delta}",
        flush=True,
    )

    sim = sim_factory(world, stage)
    started = time.monotonic()
    last_candidate: dict | None = None
    checkpoint_ref: dict | None = None
    try:
        frontier_states = capture_frontier_states(
            sim, path, chunk_frames, frontiers, args.seed)

        online = Replay(rng, args.replay_capacity)
        demo = Replay(rng, args.replay_capacity)
        successes = Replay(rng, args.replay_capacity)
        seed_demo_buffer(sim, path, chunk_frames, args.seed, demo)
        print(f"[rc_rl] demo buffer: {len(demo)} transitions", flush=True)

        qs = [QNet(hidden=args.hidden).to(DEVICE) for _ in range(args.critics)]
        targets = [QNet(hidden=args.hidden).to(DEVICE) for _ in range(args.critics)]
        for q, target in zip(qs, targets):
            target.load_state_dict(q.state_dict())
        parameters = [parameter for q in qs for parameter in q.parameters()]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        def q_mean(batch):
            return sum(q(batch) for q in qs) / args.critics

        def act(obs, epsilon: float) -> int:
            if random.random() < epsilon:
                return random.randrange(N_ACTIONS)
            tensor = torch.as_tensor(
                obs[None], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                return int(q_mean(tensor).argmax(1))

        def train_step() -> float:
            parts = []
            if len(successes) > 0:
                n_demo = args.batch_size // 3
                n_success = args.batch_size // 3
                n_online = args.batch_size - n_demo - n_success
                parts.extend([
                    demo.sample(n_demo),
                    successes.sample(n_success),
                    online.sample(n_online),
                ])
            else:
                n_demo = args.batch_size // 2
                parts.extend([
                    demo.sample(n_demo),
                    online.sample(args.batch_size - n_demo),
                ])
            states = torch.as_tensor(
                np.concatenate([part[0] for part in parts]),
                dtype=torch.float32,
                device=DEVICE,
            )
            actions = torch.as_tensor(
                np.concatenate([part[1] for part in parts]),
                dtype=torch.int64,
                device=DEVICE,
            )
            rewards = torch.as_tensor(
                np.concatenate([part[2] for part in parts]),
                dtype=torch.float32,
                device=DEVICE,
            )
            next_states = torch.as_tensor(
                np.concatenate([part[3] for part in parts]),
                dtype=torch.float32,
                device=DEVICE,
            )
            terminals = torch.as_tensor(
                np.concatenate([part[4] for part in parts]),
                dtype=torch.float32,
                device=DEVICE,
            )
            with torch.no_grad():
                next_actions = q_mean(next_states).argmax(1)
                target_values = torch.stack([
                    target(next_states).gather(
                        1, next_actions[:, None]).squeeze(1)
                    for target in targets
                ]).min(0).values
                targets_y = (
                    rewards
                    + args.gamma * (1 - terminals) * target_values
                )
            loss = sum(
                F.smooth_l1_loss(
                    q(states).gather(1, actions[:, None]).squeeze(1),
                    targets_y,
                )
                for q in qs
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            for q, target in zip(qs, targets):
                for parameter, target_parameter in zip(
                        q.parameters(), target.parameters()):
                    target_parameter.data.mul_(1 - args.polyak_tau).add_(
                        args.polyak_tau * parameter.data)
            return float(loss.detach())

        frontier_index = 0
        frontier_t = frontiers[frontier_index]
        recent = deque(maxlen=args.advance_window)
        steps = 0
        episodes = 0
        solved = False
        frontier_history = [{
            "event": "start",
            "frontier_t": frontier_t,
            "steps": steps,
            "episodes": episodes,
        }]
        success_curve: list[dict] = []

        while steps < args.max_env_steps and not solved:
            episodes += 1
            restore_frontier(sim, frontier_states[frontier_t])
            x_prev = mario_level_x(sim.ram)
            horizon = min(
                args.max_episode_chunks,
                int((path_length - frontier_t) * args.horizon_scale)
                + args.horizon_slack,
            )
            decay_steps = (
                args.max_env_steps * args.epsilon_decay_fraction)
            epsilon = max(
                args.epsilon_end,
                args.epsilon_start
                - steps
                * (args.epsilon_start - args.epsilon_end)
                / decay_steps,
            )
            observation = entity_obs(sim.ram, sim.last_info)
            episode_transitions = []
            episode_success = False

            for _ in range(horizon):
                action = act(observation, epsilon)
                info, done = sim.run_chunk(action, chunk_frames)
                steps += 1
                x_now = mario_level_x(sim.ram)
                flag = is_success(info)
                dead = is_death(info, done)
                next_observation = entity_obs(sim.ram, info)
                transition = (
                    observation,
                    action,
                    step_reward(x_prev, x_now, flag),
                    next_observation,
                    float(flag or dead or done),
                )
                online.add(*transition)
                episode_transitions.append(transition)
                observation = next_observation
                x_prev = x_now
                for _ in range(args.utd):
                    train_step()
                if flag:
                    episode_success = True
                    break
                if dead or done or steps >= args.max_env_steps:
                    break

            if episode_success:
                for transition in episode_transitions:
                    successes.add(*transition)
            recent.append(int(episode_success))

            eligible = (
                len(recent) == recent.maxlen
                and sum(recent) >= args.advance_successes
            )
            if eligible and frontier_t != 0:
                previous_frontier = frontier_t
                frontier_index += 1
                frontier_t = frontiers[frontier_index]
                frontier_history.append({
                    "event": "advance",
                    "from_frontier_t": previous_frontier,
                    "frontier_t": frontier_t,
                    "steps": steps,
                    "episodes": episodes,
                    "window_success_rate": float(np.mean(recent)),
                })
                recent.clear()
                print(
                    f"[rc_rl] advance frontier={frontier_t} "
                    f"steps={steps} episodes={episodes} epsilon={epsilon:.3f}",
                    flush=True,
                )
            elif eligible:
                sim.reset(args.seed)
                greedy_observation = entity_obs(sim.ram, sim.last_info)
                greedy_path: list[int] = []
                max_x = 0
                live_success = False
                eval_horizon = (
                    int(path_length * args.eval_horizon_scale)
                    + args.eval_horizon_slack
                )
                for _ in range(eval_horizon):
                    with torch.no_grad():
                        tensor = torch.as_tensor(
                            greedy_observation[None],
                            dtype=torch.float32,
                            device=DEVICE,
                        )
                        action = int(q_mean(tensor).argmax(1))
                    greedy_path.append(action)
                    info, done = sim.run_chunk(action, chunk_frames)
                    max_x = max(max_x, mario_level_x(sim.ram))
                    greedy_observation = entity_obs(sim.ram, info)
                    if is_success(info):
                        live_success = True
                        break
                    if is_death(info, done) or done:
                        break

                candidate_check = verify_candidate_path(
                    world,
                    stage,
                    greedy_path,
                    chunk_frames,
                    args.seed,
                    verifier=candidate_verifier,
                ) if live_success else None
                last_candidate = {
                    "live_success": live_success,
                    "independent_replay_verified": bool(
                        candidate_check and candidate_check.replay_verified),
                    "verification": (
                        candidate_check.to_json() if candidate_check else None),
                    "path": greedy_path,
                    "path_length": len(greedy_path),
                    "x_max": int(max_x),
                }
                solved = bool(
                    candidate_check and candidate_check.replay_verified)
                frontier_history.append({
                    "event": (
                        "verified_start_policy"
                        if solved else "failed_start_evaluation"),
                    "frontier_t": frontier_t,
                    "steps": steps,
                    "episodes": episodes,
                    "live_success": live_success,
                    "independent_replay_verified": solved,
                    "x_max": int(max_x),
                })
                print(
                    f"[rc_rl] start evaluation live={live_success} "
                    f"replay_verified={solved} x_max={max_x}",
                    flush=True,
                )
                if not solved:
                    recent.clear()

            if episodes == 1 or episodes % 25 == 0 or episode_success:
                success_curve.append({
                    "episodes": episodes,
                    "steps": steps,
                    "frontier_t": frontier_t,
                    "episode_success": episode_success,
                    "window_size": len(recent),
                    "window_success_rate": (
                        float(np.mean(recent)) if recent else 0.0),
                    "epsilon": float(epsilon),
                })

            if episodes % 50 == 0:
                rate = float(np.mean(recent)) if recent else 0.0
                print(
                    f"[rc_rl] episodes={episodes} steps={steps} "
                    f"frontier={frontier_t} success_rate={rate:.2f} "
                    f"epsilon={epsilon:.3f}",
                    flush=True,
                )

        elapsed = time.monotonic() - started
        result = {
            "status": (
                "verified_policy" if solved else "budget_exhausted"),
            "solved": solved,
            "independent_replay_verified": solved,
            "steps": steps,
            "episodes": episodes,
            "elapsed_seconds": round(elapsed, 3),
            "final_frontier_t": frontier_t,
            "frontier_count": len(frontiers),
            "frontier_delta": delta,
            "demo_transitions": len(demo),
            "online_transitions": len(online),
            "successful_transitions": len(successes),
            "frontier_history": frontier_history,
            "success_curve": success_curve,
            "last_candidate": last_candidate,
        }
        if solved:
            checkpoint_path = Path(args.checkpoint_out).expanduser().resolve()
            save_checkpoint_atomic(checkpoint_path, {
                "schema_version": 1,
                "kind": "experimental_reverse_curriculum_double_q",
                "level": level,
                "seed": args.seed,
                "solution": solution_ref,
                "solution_verification": solution_check.to_json(),
                "driver": file_ref(Path(__file__)),
                "report_target": path_label(args.report),
                "config": config_from_args(args, chunk_frames),
                "result": result,
                "model": {
                    "input_dim": OBS_DIM_ENTITY,
                    "hidden": args.hidden,
                    "n_actions": N_ACTIONS,
                    "n_critics": args.critics,
                    "states": [q.state_dict() for q in qs],
                },
            })
            checkpoint_ref = file_ref(checkpoint_path)
        return result, checkpoint_ref
    finally:
        sim.close()


def base_report(args: argparse.Namespace, world: int, stage: int) -> dict:
    return {
        "schema_version": 1,
        "experiment": "experimental_reverse_curriculum_double_q",
        "generated_at": utc_now_iso(),
        "git_revision": git_rev(),
        "environment": env_fingerprint(),
        "level": f"{world}-{stage}",
        "seed": args.seed,
        "source": {
            "driver": file_ref(Path(__file__)),
        },
        "inputs": {},
        "outputs": {
            "report": path_label(args.report),
            "checkpoint_target": path_label(args.checkpoint_out),
        },
        "config": {},
        "result": None,
        "checkpoint": None,
        "claim_boundary": (
            "This artifact reports one experimental run. It does not preserve "
            "or validate the historical +391 / 0.22 observation, and it does "
            "not establish convergence or generalization."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        world, stage = validate_args(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    report = base_report(args, world, stage)
    report_path = Path(args.report).expanduser().resolve()
    try:
        rng = set_all_seeds(args.seed)
        solution_path = resolve_solution_path(args)
        actions, chunk_frames, solution_check, manifest = load_verified_demo(
            solution_path, args.level, args.seed)
        solution_ref = file_ref(solution_path)
        report["inputs"] = {
            "solution": solution_ref,
            "solution_verification": solution_check.to_json(),
            "manifest_declared_chunk_frames": manifest.get("chunk_frames"),
            "effective_chunk_frames": chunk_frames,
            "manifest_declared_path_length": len(actions),
            "manifest_extra_fields": sorted(
                set(manifest) - {"path", "chunk_frames"}),
        }
        report["config"] = config_from_args(args, chunk_frames)
        result, checkpoint_ref = run_experiment(
            args,
            world,
            stage,
            actions,
            chunk_frames,
            solution_ref,
            solution_check,
            rng,
        )
        report["result"] = result
        report["checkpoint"] = checkpoint_ref
        exit_code = 0
    except Exception as exc:
        report["result"] = {
            "status": "error",
            "solved": False,
            "independent_replay_verified": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 2
        print(f"[rc_rl] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

    report["completed_at"] = utc_now_iso()
    write_json_atomic(report_path, report)
    print(f"[rc_rl] wrote report {report_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
