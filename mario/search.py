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
    wp_idx: int = 0   # current waypoint index (non-linear levels); 0 for pure-x search


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


def search_from_state(sim: MarioSim, root_snap, *, world: int, stage: int,
                      beam_width: int = 32, depth: int = 400, chunk_frames: int = 8,
                      weights: RewardWeights = DEFAULT, stuck_cap: int = 16,
                      waypoints=None, max_seconds: float = 90.0) -> list[int]:
    """Beam search on the GIVEN sim, starting from root_snap (the live state).

    Returns the action path that clears the level (flag_get OR a (world,stage) change —
    works in both single- and multi-stage envs) or, if not cleared within depth/time, the
    best-progress path. Uses the sim's own snapshot/restore so there's no cross-env
    transfer error. The CALLER must restore root_snap afterward. `waypoints` (optional) is
    a WaypointTracker spec for non-linear levels (maze/warp); pure-x reward if None.
    """
    from mario.waypoints import WaypointTracker
    sim.restore(root_snap)
    start_info = dict(sim.last_info)
    x_start = int(start_info.get("x_pos", 0))
    wp = WaypointTracker(waypoints) if waypoints else None

    root = Node(root_snap, 0.0, start_info, [], x_start, 0, 0)
    if wp:
        root.wp_idx = 0
    beam = [root]
    best = root
    t0 = time.perf_counter()

    for _ in range(1, depth + 1):
        if time.perf_counter() - t0 > max_seconds:
            break
        candidates = []
        for node in beam:
            wp_idx = getattr(node, "wp_idx", 0)
            for a in range(N_ACTIONS):
                sim.restore(node.snap)
                info, done = sim.run_chunk(a, chunk_frames)
                cleared = is_success(info) or \
                    (int(info.get("world", world)), int(info.get("stage", stage))) != (world, stage)
                if cleared:
                    sim.restore(root_snap)
                    return node.path + [a]
                if is_death(info, done):
                    continue
                x = int(info.get("x_pos", 0))
                nstuck = node.stuck + (0 if x > node.x_max else 1)
                if nstuck > stuck_cap:
                    continue
                if wp:
                    nidx, score = wp.score(info, wp_idx, x_start, weights)
                else:
                    nidx, score = wp_idx, state_score(info, x_start, 0, False, nstuck, weights)
                child = Node(sim.snapshot(), score, info, node.path + [a],
                             max(node.x_max, x), nstuck, node.frames + chunk_frames)
                child.wp_idx = nidx
                candidates.append(child)
        if not candidates:
            break
        by_key = {}
        for c in candidates:
            k = _dedup_key(c.info) + (getattr(c, "wp_idx", 0),)
            if k not in by_key or c.score > by_key[k].score:
                by_key[k] = c
        beam = sorted(by_key.values(), key=lambda c: c.score, reverse=True)[:beam_width]
        if beam[0].x_max > best.x_max or getattr(beam[0], "wp_idx", 0) > getattr(best, "wp_idx", 0):
            best = beam[0]

    sim.restore(root_snap)
    return best.path
