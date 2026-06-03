"""Teacher-labeling primitive — the bridge from search to a learnable dataset.

`beam_search` only returns the single winning path. For distillation (V2) and DAgger
(V3) we need, at an ARBITRARY state, a soft distribution over the 7 actions plus a
value. Snapshots aren't picklable across processes, but action-path PREFIXES are — and
the emulator is deterministic — so we reconstruct any state by replaying its prefix,
then evaluate each action by a short lookahead.

label_state(prefix):
  replay prefix -> snapshot s0
  for each action a: restore s0, run chunk a, then
     flag  -> q[a] = +flag(+progress)
     death -> q[a] = -death           (annihilated by softmax)
     else  -> q[a] = best state_score reachable in a shallow beam rollout (depth chunks)
  soft_targets = softmax(q / tau);  best_action = argmax(q);  value = max(q)

This is the ONLY way to get soft targets out of this teacher without modifying search.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mario.env import MarioSim, N_ACTIONS
from mario.observation import observe
from mario.reward import DEFAULT, RewardWeights, is_death, is_success, state_score
from mario.search import _dedup_key  # reuse the spatial-bucket dedup

_NEG = -1e18


@dataclass
class LabelResult:
    best_action: int
    soft_targets: np.ndarray      # float32[7], sums to 1
    value: float                  # max_a q[a]  (expert value of this state)
    per_action_value: np.ndarray  # float32[7], raw q
    reachable: np.ndarray         # bool[7], False = action leads to immediate death
    obs: np.ndarray               # float32[OBS_DIM], observe() at the labeled state
    all_doomed: bool              # True iff every action is fatal/doomed (exclude these)


def _shallow_best(sim: MarioSim, start_snap, x_start: int, init_info: dict, *,
                  depth: int, beam_width: int, chunk_frames: int,
                  weights: RewardWeights, stuck_cap: int = 12) -> float:
    """Best state_score reachable within `depth` chunks from a given snapshot.

    A mini beam search (no reset) used to estimate an action's *future potential*.
    - reaches the flag        -> +flag (great)
    - every line dies/stalls  -> -death (this state is doomed within the horizon)
    - otherwise               -> best progress score reached

    Returning -death on a wiped-out beam is what lets us catch actions that don't kill
    Mario THIS chunk but doom him a couple of chunks later (e.g. running off a pit edge).
    """
    start_x = int(init_info.get("x_pos", 0))
    best = state_score(init_info, x_start, chunk_frames, died=False, stuck=0, w=weights)
    beam = [(start_snap, max(start_x, x_start), 0, chunk_frames)]  # (snap,x_max,stuck,frames)

    for _ in range(depth):
        cands = []  # (score, snap, x_max, stuck, frames, info)
        for snap, x_max, stuck, frames in beam:
            for a in range(N_ACTIONS):
                sim.restore(snap)
                info, done = sim.run_chunk(a, chunk_frames)
                if is_success(info):
                    return weights.flag + weights.progress * (int(info.get("x_pos", 0)) - x_start)
                if is_death(info, done):
                    continue
                x = int(info.get("x_pos", 0))
                nstuck = stuck + (0 if x > x_max else 1)
                if nstuck > stuck_cap:
                    continue
                nframes = frames + chunk_frames
                sc = state_score(info, x_start, nframes, died=False, stuck=nstuck, w=weights)
                cands.append((sc, sim.snapshot(), max(x_max, x), nstuck, nframes, info))
        if not cands:
            return -weights.death  # no survivable continuation -> doomed
        by_key: dict = {}
        for c in cands:
            k = _dedup_key(c[5])
            if k not in by_key or c[0] > by_key[k][0]:
                by_key[k] = c
        top = sorted(by_key.values(), key=lambda c: c[0], reverse=True)[:beam_width]
        beam = [(c[1], c[2], c[3], c[4]) for c in top]
        if top[0][0] > best:
            best = top[0][0]
    return best


def label_state(world: int, stage: int, action_path_prefix, *, chunk_frames: int = 8,
                depth: int = 5, beam_width: int = 4, tau: float = 40.0, gamma: float = 0.5,
                weights: RewardWeights = DEFAULT, seed: int = 0,
                sim: MarioSim | None = None) -> LabelResult:
    own = sim is None
    if own:
        sim = MarioSim(world, stage)
    sim.reset(seed=seed)
    done = False
    for a in action_path_prefix:
        _info, done = sim.run_chunk(a, chunk_frames)
        if done:
            break
    x_start = int(sim.last_info.get("x_pos", 0))
    obs = observe(sim.ram, sim.last_info)
    s0 = sim.snapshot()

    q = np.full(N_ACTIONS, _NEG, dtype=np.float64)
    reachable = np.zeros(N_ACTIONS, dtype=bool)
    for a in range(N_ACTIONS):
        sim.restore(s0)
        info, done = sim.run_chunk(a, chunk_frames)
        if is_success(info):
            q[a] = weights.flag + weights.progress * (int(info.get("x_pos", 0)) - x_start)
            reachable[a] = True
            continue
        if is_death(info, done):
            q[a] = -weights.death
            reachable[a] = False
            continue
        reachable[a] = True
        immediate = state_score(info, x_start, chunk_frames, died=False, stuck=0, w=weights)
        snap = sim.snapshot()
        look = _shallow_best(sim, snap, x_start, info, depth=depth,
                             beam_width=beam_width, chunk_frames=chunk_frames, weights=weights)
        if look <= -weights.death / 2:
            q[a] = look  # doomed within the horizon
        else:
            # Blend immediate progress (so running right beats idling / going left on
            # flat ground) with future potential (so hazards/jumps are still valued).
            q[a] = (1.0 - gamma) * immediate + gamma * look
    if own:
        sim.close()

    z = (q - q.max()) / tau
    ex = np.exp(z)
    soft = (ex / ex.sum()).astype(np.float32)
    all_doomed = bool((q <= -weights.death / 2).all())
    return LabelResult(best_action=int(np.argmax(q)), soft_targets=soft,
                       value=float(q.max()), per_action_value=q.astype(np.float32),
                       reachable=reachable, obs=obs.astype(np.float32),
                       all_doomed=all_doomed)


def soft_entropy(soft: np.ndarray) -> float:
    """Shannon entropy (nats) of a soft-target distribution — sanity metric."""
    p = soft[soft > 0]
    return float(-(p * np.log(p)).sum())
