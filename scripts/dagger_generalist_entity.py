"""Closed-loop DAgger for the entity generalist (the decisive loop).

Each round: roll the current policy out across ALL TRAIN levels (balanced — every level
every round, the fix for V4's single-level oscillation), find where it dies/stalls, query the
search oracle (`beam_search(start_prefix=)`) for the recovery, and relabel those recovered
states with entity tokens + procedure-cloning soft/value targets (`label_at_state`). The
corrections (source=2, up-weighted in the loss) are APPENDED into each train level's own
shard, then the policy is retrained warm-start. Held-out levels are never rolled out or
relabeled — integrity is preserved by construction.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/dagger_generalist_entity.py \
        --config configs/heldout_split.json --ckpt data/generalist.pt --rounds 4 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mario.entity import entity_obs
from mario.entity_policy import EntityController, load_policy_checkpoint
from mario.env import MarioSim
from mario.label import label_at_state
from mario.reward import is_death, is_success
from mario.search import beam_search

SHARDS = ROOT / "data" / "entity_shards"
PY = str(ROOT / "venv" / "bin" / "python")
CF = 8
PREDEATH_OFFSETS = (3, 6, 10, 14)
RECOVER_BEAM, RECOVER_DEPTH, RECOVER_KEEP = 24, 60, 10


def collect_level(ctrl: EntityController, w: int, s: int, seeds: int) -> list[tuple]:
    """Roll out, find failures, oracle-relabel recovered states. Returns rows for source=2."""
    rows = []
    for seed in range(seeds):
        sim = MarioSim(w, s); sim.reset(seed); acts = []; failure_at = None
        beat = False
        for _ in range(700):
            a = ctrl.act(sim.ram, sim.last_info); acts.append(a)
            info, done = sim.run_chunk(a, CF)
            if is_success(info):
                beat = True
                break
            if is_death(info, done) or done:
                failure_at = len(acts); break
        sim.close()
        # A policy that survives but makes no terminal progress is still a
        # DAgger failure.  Anchor at the rollout cap just as we do at death.
        if not beat and failure_at is None:
            failure_at = len(acts)
        if failure_at is None:
            continue
        for off in PREDEATH_OFFSETS:
            if failure_at - off < 1:
                continue
            prefix = acts[:failure_at - off]
            rec = beam_search(w, s, beam_width=RECOVER_BEAM, max_depth=RECOVER_DEPTH,
                              seed=seed, start_prefix=prefix)
            if not rec.path:
                continue
            sim2 = MarioSim(w, s); sim2.reset(seed); dead = False
            for a in prefix:
                if sim2.run_chunk(a, CF)[1]:
                    dead = True; break
            if not dead:
                for j in range(min(RECOVER_KEEP, len(rec.path))):
                    x0 = int(sim2.last_info.get("x_pos", 0))
                    obs = entity_obs(sim2.ram, sim2.last_info)
                    soft, value, _b, doomed, _q = label_at_state(sim2, x0, chunk_frames=CF)
                    if not doomed:
                        rows.append((obs, int(rec.path[j]), soft, float(value), w * 10 + s))
                    if sim2.run_chunk(rec.path[j], CF)[1]:
                        break
            sim2.close()
    return rows


def append_corrections(w: int, s: int, rows: list[tuple], rnd: int) -> int:
    """Append source=2 correction rows into level-{w}-{s}.npz (preserving the allowlist)."""
    p = SHARDS / f"level-{w}-{s}.npz"
    d = {k: np.array(v) for k, v in np.load(p).items()} if p.exists() else None
    if not rows:
        return 0
    obs = np.asarray([r[0] for r in rows], np.float32)
    hard = np.asarray([r[1] for r in rows], np.int64)
    soft = np.asarray([r[2] for r in rows], np.float32)
    value = np.asarray([r[3] for r in rows], np.float32)
    lid = np.asarray([r[4] for r in rows], np.int64)
    src = np.full(len(rows), 2, np.int8)
    traj = np.full(len(rows), 900000 + rnd, np.int64)
    if d is None:
        merged = {"obs": obs, "hard_action": hard, "soft_targets": soft, "value": value,
                  "level_id": lid, "source": src, "traj": traj}
    else:
        merged = {
            "obs": np.concatenate([d["obs"], obs]),
            "hard_action": np.concatenate([d["hard_action"], hard]),
            "soft_targets": np.concatenate([d["soft_targets"], soft]),
            "value": np.concatenate([d["value"], value]),
            "level_id": np.concatenate([d["level_id"], lid]),
            "source": np.concatenate([d["source"], src]),
            "traj": np.concatenate([d["traj"], traj]),
        }
    np.savez_compressed(p, **merged)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "generalist.pt"))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    train = [tuple(int(x) for x in p.split("-")) for p in cfg["train_levels"]]
    holdout = set(cfg["holdout_levels"])
    for (w, s) in train:
        assert f"{w}-{s}" not in holdout, "INTEGRITY: train level in holdout"

    _, ckpt_meta = load_policy_checkpoint(args.ckpt, device="cpu")
    if sorted(ckpt_meta.get("train_levels", [])) != sorted(cfg["train_levels"]) \
            or sorted(ckpt_meta.get("holdout_levels", [])) != sorted(holdout):
        raise SystemExit(
            "checkpoint train/holdout metadata does not match the DAgger config")

    for rnd in range(1, args.rounds + 1):
        net, _ = load_policy_checkpoint(args.ckpt, device="cpu")
        ctrl = EntityController(net, device="cpu")
        total = 0
        for (w, s) in train:
            rows = collect_level(ctrl, w, s, args.seeds)
            n = append_corrections(w, s, rows, rnd)
            total += n
            if n:
                print(f"  r{rnd} {w}-{s}: +{n} corrections", flush=True)
        print(f"[dagger r{rnd}] {total} corrections total; retraining warm-start ...", flush=True)
        if total == 0:
            print("  no corrections — stopping early"); break
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        subprocess.run([PY, "scripts/train_generalist.py", "--config", args.config,
                        "--init", args.ckpt, "--out", args.ckpt, "--epochs", "20"],
                       cwd=str(ROOT), check=True, env=env)
        print(f"[dagger r{rnd}] retrained -> {args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
