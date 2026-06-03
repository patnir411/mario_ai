"""Beam-search teacher: use the emulator as a forward model to find a trajectory
that beats a level.

In-process only: snapshots (NativeStateSnapshot) are not picklable, so one search runs
on a single MarioSim. Parallelism happens ACROSS searches (levels/states), not within.

Core loop (DESIGN.md §6): from each beam node, try all 7 action-chunks via snapshot
restore; score the resulting state with the death-aware reward; drop dead nodes; dedup
by spatial bucket; keep the top-k. Return as soon as a node reaches the flag.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from mario.env import MarioSim, N_ACTIONS
from mario.reward import DEFAULT, RewardWeights, is_death, is_success, state_score


@dataclass
class Node:
    snap: object
    score: float
    info: dict
    path: list[int]
    x_max: int
    stuck: int
    frames: int


@dataclass
class SearchResult:
    solved: bool
    path: list[int]              # action index per chunk
    chunk_frames: int
    final_info: dict
    x_max: int
    frames: int
    depth_reached: int
    nodes_expanded: int
    wall_clock_s: float
    beam_width: int
    weights: RewardWeights
    stuck_log: list = field(default_factory=list)


def _dedup_key(info: dict, bucket: int = 16) -> tuple:
    """Collapse near-identical states so the beam doesn't fill with duplicates."""
    return (int(info.get("x_pos", 0)) // bucket,
            int(info.get("y_pos", 0)) // bucket,
            info.get("status", "small"))


def beam_search(world: int = 1, stage: int = 1, *, beam_width: int = 40,
                chunk_frames: int = 8, max_depth: int = 400,
                weights: RewardWeights = DEFAULT, seed: int = 0,
                stuck_cap: int = 12, progress_every: int = 0,
                start_prefix: list[int] | None = None) -> SearchResult:
    """Beam search to beat a level. With `start_prefix`, replay those chunks after reset
    and search the CONTINUATION from there (used for recovery trajectories in DAgger-lite
    dataset generation). The returned `path` is the continuation only; progress/x_start
    are measured from the post-prefix state."""
    sim = MarioSim(world, stage)
    sim.reset(seed=seed)
    if start_prefix:
        for a in start_prefix:
            _info, done = sim.run_chunk(a, chunk_frames)
            if done:
                break
    root_info = sim.last_info
    x_start = int(root_info.get("x_pos", 0))
    beam = [Node(sim.snapshot(), 0.0, root_info, [], x_start, 0, 0)]

    best = beam[0]
    nodes = 0
    t0 = time.perf_counter()
    depth = 0

    for depth in range(1, max_depth + 1):
        candidates: list[Node] = []
        for node in beam:
            for a in range(N_ACTIONS):
                sim.restore(node.snap)
                info, done = sim.run_chunk(a, chunk_frames)
                nodes += 1
                if is_success(info):
                    sim.close()
                    return SearchResult(
                        True, node.path + [a], chunk_frames, info,
                        int(info.get("x_pos", 0)), node.frames + chunk_frames,
                        depth, nodes, time.perf_counter() - t0, beam_width, weights)
                if is_death(info, done):
                    continue
                x = int(info.get("x_pos", 0))
                stuck = node.stuck + (0 if x > node.x_max else 1)
                if stuck > stuck_cap:
                    continue  # abandon nodes that stop making forward progress
                frames = node.frames + chunk_frames
                score = state_score(info, x_start, frames, died=False,
                                    stuck=stuck, w=weights)
                candidates.append(Node(sim.snapshot(), score, info,
                                       node.path + [a], max(node.x_max, x),
                                       stuck, frames))
        if not candidates:
            break

        # dedup: keep best-scoring node per spatial bucket
        best_by_key: dict[tuple, Node] = {}
        for c in candidates:
            k = _dedup_key(c.info)
            if k not in best_by_key or c.score > best_by_key[k].score:
                best_by_key[k] = c
        deduped = sorted(best_by_key.values(), key=lambda c: c.score, reverse=True)
        beam = deduped[:beam_width]

        if beam[0].x_max > best.x_max:
            best = beam[0]
        if progress_every and depth % progress_every == 0:
            print(f"  depth {depth:4d} | beam {len(beam):3d} | best_x {best.x_max:5d} "
                  f"| nodes {nodes:7d} | {nodes/(time.perf_counter()-t0):6.0f} n/s",
                  flush=True)

    sim.close()
    return SearchResult(False, best.path, chunk_frames, best.info, best.x_max,
                        best.frames, depth, nodes, time.perf_counter() - t0,
                        beam_width, weights)
