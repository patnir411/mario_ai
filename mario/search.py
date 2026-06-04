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
    x_max: int        # AREA-LOCAL running max x (resets on area change; used for stuck)
    stuck: int
    frames: int
    wp_idx: int = 0   # current waypoint index (non-linear levels); 0 for pure-x search
    area: int = 0          # current area byte ($0760)
    area_seq: int = 0      # # of area transitions this lineage has made (level-global progress)
    area_entry_x: int = 0  # x_pos when the current area was entered


# Level-global progress STRIDE: an area transition is worth more than any within-area x,
# so completing an area always beats fractional progress inside one. (max within-area
# x ~4000, so 10000 is a safe separation.)
AREA_STRIDE = 10000


def global_progress(area_seq: int, x: int, area_entry_x: int) -> float:
    """Φ — a monotone, level-global progress coordinate that does NOT reset on area change."""
    return area_seq * AREA_STRIDE + (x - area_entry_x)


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


def _dedup_key(info: dict, bucket: int = 16, area: int = 0) -> tuple:
    """Collapse near-identical states so the beam doesn't fill with duplicates.

    `area` ($0760) is part of the key: two states at the same (x,y) in DIFFERENT areas
    (e.g. a pipe's two ends, or a maze loop-room vs the room past it) are physically
    distinct and must not be merged — that merge is what made the beam collapse on warp/
    maze levels.
    """
    return (int(info.get("x_pos", 0)) // bucket,
            int(info.get("y_pos", 0)) // bucket,
            info.get("status", "small"), area)


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
    AREA = 0x0760
    root_info = sim.last_info
    x_start = int(root_info.get("x_pos", 0))
    root_area = int(sim.ram[AREA])
    beam = [Node(sim.snapshot(), 0.0, root_info, [], x_start, 0, 0,
                 area=root_area, area_seq=0, area_entry_x=x_start)]

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
                na = int(sim.ram[AREA])
                if na != node.area:                 # area transition = big forward progress
                    area_seq = node.area_seq + 1
                    area_entry_x, area_x_max, stuck = x, x, 0
                else:
                    area_seq = node.area_seq
                    area_entry_x = node.area_entry_x
                    area_x_max = max(node.x_max, x)
                    stuck = node.stuck + (0 if x > node.x_max else 1)
                if stuck > stuck_cap:
                    continue  # abandon nodes that stop making forward progress
                frames = node.frames + chunk_frames
                phi = global_progress(area_seq, x, area_entry_x)
                score = state_score(info, x_start, frames, died=False,
                                    stuck=stuck, w=weights, progress=phi)
                candidates.append(Node(sim.snapshot(), score, info,
                                       node.path + [a], area_x_max, stuck, frames,
                                       area=na, area_seq=area_seq, area_entry_x=area_entry_x))
        if not candidates:
            break

        # dedup: keep best-scoring node per spatial bucket (area-aware)
        best_by_key: dict[tuple, Node] = {}
        for c in candidates:
            k = _dedup_key(c.info, area=c.area)
            if k not in best_by_key or c.score > best_by_key[k].score:
                best_by_key[k] = c
        deduped = sorted(best_by_key.values(), key=lambda c: c.score, reverse=True)
        beam = deduped[:beam_width]

        # best = furthest level-global progress (area-first, then within-area x)
        if (beam[0].area_seq, beam[0].x_max) > (best.area_seq, best.x_max):
            best = beam[0]
        if progress_every and depth % progress_every == 0:
            print(f"  depth {depth:4d} | beam {len(beam):3d} | best_x {best.x_max:5d} "
                  f"| nodes {nodes:7d} | {nodes/(time.perf_counter()-t0):6.0f} n/s",
                  flush=True)

    sim.close()
    return SearchResult(False, best.path, chunk_frames, best.info, best.x_max,
                        best.frames, depth, nodes, time.perf_counter() - t0,
                        beam_width, weights)


def coverage_search(world: int = 1, stage: int = 1, *, beam_width: int = 48,
                    chunk_frames: int = 8, max_depth: int = 2000,
                    time_budget_s: float = 1800.0, tile: int = 16, seed: int = 0,
                    stuck_cap: int = 48, cov_bonus: float = 50.0,
                    area_bonus: float = 0.0, ground_bonus: float = 0.0,
                    loop_back_px: int = 0, pipe_macro_chunks: int = 0,
                    checkpoint_path: str | None = None,
                    progress_every: int = 200) -> SearchResult:
    """Beam search + Go-Explore coverage — the general solver for ALL level types.

    Plain x-greedy beam loops forever on the maze castles (4-4/7-4/8-4): its dedup key is
    pure (x,y) and its reward gives no credit for entering the correct pipe (an AREA change
    that resets x), so it plateaus at the loop boundary while the beam empties. This fixes
    both: the dedup key is the Go-Explore CELL (area $0760, x-tile, y-tile) so different
    maze rooms/heights never collapse, and every NEW cell a lineage opens earns a one-time
    novelty bonus — so the search actively values reaching new rooms/heights (pipe entry,
    the non-looping route) instead of only rightward x. On linear levels novelty saturates
    immediately and x-progress dominates, so it behaves like the proven beam (which already
    cleared the Hammer-Bros gauntlet to x=3861 on 8-4). chunk_frames default 8 matches the
    teacher; keep stuck_cap generous since maze detours legitimately stall x for a while.
    """
    from mario.ram import mario_level_x
    AREA, Y_ADDR, FLOAT = 0x0760, 0x00CE, 0x001D
    sim = MarioSim(world, stage)
    sim.reset(seed=seed)

    def cell(ram) -> tuple:
        return (int(ram[AREA]), mario_level_x(ram) // tile, int(ram[Y_ADDR]) // tile)

    root_info = sim.last_info
    x_start = int(root_info.get("x_pos", 0))
    visited = {cell(sim.ram)}
    root = Node(sim.snapshot(), 0.0, root_info, [], x_start, 0, 0)
    root.wp_idx = 0   # reuse wp_idx as cumulative novelty count along the lineage
    root.area = int(sim.ram[AREA])
    root.ground_xmax = x_start   # furthest x reached while GROUNDED (float_state==0)
    beam = [root]
    best = root

    def prog(n):   # area-FIRST (right pipe), then grounded-x (8-4 loop gate), then x
        return (getattr(n, "area", 0), getattr(n, "ground_xmax", 0), n.x_max)
    nodes = 0
    t0 = time.perf_counter()
    depth = 0

    for depth in range(1, max_depth + 1):
        if time.perf_counter() - t0 > time_budget_s:
            break
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
                        depth, nodes, time.perf_counter() - t0, beam_width, DEFAULT)
                if is_death(info, done):
                    continue
                c = cell(sim.ram)
                x = int(info.get("x_pos", 0))
                # loop-prune: a sharp BACKWARD x-jump within the same area is a maze
                # loop-checkpoint teleport (8-4 page 16 -> 12). Kill it like a death so the
                # beam stops re-running the loop and must find a clean (non-looping) crossing.
                if loop_back_px and c[0] == node.area and x < node.x_max - loop_back_px:
                    continue
                novel = c not in visited
                if novel:
                    visited.add(c)
                # a new cell counts as progress: reset the stall counter for detours
                stuck = 0 if (x > node.x_max or novel) else node.stuck + 1
                if stuck > stuck_cap:
                    continue
                novelty = node.wp_idx + (1 if novel else 0)
                grounded = int(sim.ram[FLOAT]) == 0
                ground_xmax = max(node.ground_xmax, x) if grounded else node.ground_xmax
                score = (state_score(info, x_start, 0, False, stuck, DEFAULT)
                         + cov_bonus * novelty
                         + area_bonus * int(sim.ram[AREA])
                         + ground_bonus * (ground_xmax - x_start))
                child = Node(sim.snapshot(), score, info, node.path + [a],
                             max(node.x_max, x), stuck, node.frames + chunk_frames)
                child.wp_idx = novelty
                child.area = c[0]
                child.ground_xmax = ground_xmax
                candidates.append((c, child))

            # pipe-entry macro: sustained DOWN in one atomic step, so a multi-chunk pipe
            # entry completes without its mid-crouch states being pruned. Only kept if it
            # actually changes AREA (a real pipe entry) or reaches the flag.
            if pipe_macro_chunks:
                sim.restore(node.snap)
                m_done = False
                for _ in range(pipe_macro_chunks):
                    info, m_done = sim.run_chunk(7, chunk_frames)
                    if is_success(info) or int(sim.ram[AREA]) != node.area or m_done:
                        break
                if is_success(info):
                    sim.close()
                    return SearchResult(
                        True, node.path + [7] * pipe_macro_chunks, chunk_frames, info,
                        int(info.get("x_pos", 0)), node.frames, depth, nodes,
                        time.perf_counter() - t0, beam_width, DEFAULT)
                na = int(sim.ram[AREA])
                if not m_done and na != node.area:        # entered a pipe → new area!
                    c = cell(sim.ram); x = int(info.get("x_pos", 0))
                    visited.add(c)
                    mc = Node(sim.snapshot(),
                              state_score(info, x_start, 0, False, 0, DEFAULT)
                              + cov_bonus * (node.wp_idx + 1) + area_bonus * na
                              + ground_bonus * (node.ground_xmax - x_start),
                              info, node.path + [7] * pipe_macro_chunks,
                              max(node.x_max, x), 0, node.frames)
                    mc.wp_idx = node.wp_idx + 1; mc.area = na; mc.ground_xmax = node.ground_xmax
                    candidates.append((c, mc))
        if not candidates:
            break
        best_by_key: dict[tuple, Node] = {}
        for c, ch in candidates:
            if c not in best_by_key or ch.score > best_by_key[c].score:
                best_by_key[c] = ch
        beam = sorted(best_by_key.values(), key=lambda n: n.score, reverse=True)[:beam_width]
        cand_best = max(beam, key=prog)
        if prog(cand_best) > prog(best):
            best = cand_best
        if progress_every and depth % progress_every == 0:
            max_area = max(getattr(n, "area", 0) for n in beam)
            print(f"  depth {depth:4d} | beam {len(beam):3d} | best area{best.area} x{best.x_max:5d} "
                  f"gnd{getattr(best,'ground_xmax',0):5d} | beam_max_area {max_area} "
                  f"| cells {len(visited):5d} | nodes {nodes:8d} "
                  f"| {nodes/(time.perf_counter()-t0):6.0f} n/s", flush=True)
            if checkpoint_path:
                import json as _json
                with open(checkpoint_path, "w") as _f:
                    _json.dump({"path": best.path, "x_max": best.x_max,
                                "area": getattr(best, "area", 0), "depth": depth}, _f)

    sim.close()
    return SearchResult(False, best.path, chunk_frames, best.info, best.x_max,
                        best.frames, depth, nodes, time.perf_counter() - t0,
                        beam_width, DEFAULT)


def area_search(world: int, stage: int, *, start_prefix: list[int] | None = None,
                beam_width: int = 96, chunk_frames: int = 8, prefix_cf: int | None = None,
                max_depth: int = 4000, max_x: int | None = None, actions=None,
                forbid_keys=None,
                time_budget_s: float = 600.0, tile: int = 16, seed: int = 0,
                stuck_cap: int = 70, cov_bonus: float = 50.0,
                checkpoint_path: str | None = None, progress_every: int = 50):
    """Per-AREA curriculum solver (the 8-4 / maze-castle unblock).

    The flat beam keys progress on x_pos, which is AREA-LOCAL (resets ~40 on a pipe/area
    transition), so entering the correct pipe reads as a ~-1000px catastrophe and is pruned.
    This solver instead treats reaching a NEW AREA ($0760 change) as SUCCESS: it explores
    the current area with coverage (so it reaches off-greedy affordances — left-backtrack
    pipes, lift crossings — not just the rightmost x), does NOT prune backtracks, and
    terminates the moment the area byte changes (pipe entered) or the flag is hit. Chain it
    with start_prefix across areas to traverse a multi-area level (8-4: overworld -> water
    -> Bowser). Returns (changed: bool, path: list[int], info: dict).

    Cell key includes status + a coarse x-speed bucket (admissibility: two states at the
    same tile but different powerup/velocity are physically different — don't merge them).
    """
    from mario.ram import mario_level_x, signed, AREA_NUMBER, AREA_POINTER
    Y_ADDR, XSPD = 0x00CE, 0x0057
    sim = MarioSim(world, stage, actions=actions) if actions is not None else MarioSim(world, stage)
    n_actions = len(actions) if actions is not None else N_ACTIONS
    sim.reset(seed=seed)
    if start_prefix:                       # prefix may use a coarser cf than exploration
        pcf = prefix_cf if prefix_cf is not None else chunk_frames
        for a in start_prefix:
            _i, d = sim.run_chunk(a, pcf)
            if d:
                break
    # Area identity = (raw area $0760, area pointer $0750). $0750 is the byte that actually
    # changes on a pipe/sub-area transition in this env ($0760 stays put) — so SUCCESS = the
    # area KEY changes (a new room entered) or the flag.
    start_key = (int(sim.ram[AREA_NUMBER]), int(sim.ram[AREA_POINTER]))
    # forbid = rooms already visited along the chain (incl. start) so we don't "succeed" by
    # walking BACK into a prior room (the 8-4 loop returns 229->101); success needs a NEW room.
    forbid = {tuple(k) for k in (forbid_keys or [])} | {start_key}
    entry_x = mario_level_x(sim.ram)

    def cell(ram) -> tuple:
        vx = signed(int(ram[XSPD]))
        vb = 0 if vx == 0 else (1 if vx > 0 else -1)
        return (int(ram[AREA_POINTER]), mario_level_x(ram) // tile, int(ram[Y_ADDR]) // tile,
                int(ram[0x0756]), vb)   # area-pointer, x-tile, y-tile, powerup, vx-sign

    visited = {cell(sim.ram)}
    # node tuple: (snap, path, x_max_in_area, stuck, novelty)
    beam = [(sim.snapshot(), [], entry_x, 0, 0)]
    best_path, best_x = [], entry_x
    nodes = 0
    t0 = time.perf_counter()

    for depth in range(1, max_depth + 1):
        if time.perf_counter() - t0 > time_budget_s:
            break
        candidates = []
        for snap, path, x_max, stuck, nov in beam:
            for a in range(n_actions):
                sim.restore(snap)
                info, done = sim.run_chunk(a, chunk_frames)
                nodes += 1
                cur_key = (int(sim.ram[AREA_NUMBER]), int(sim.ram[AREA_POINTER]))
                if is_success(info) or cur_key not in forbid:   # flag OR entered a NEW room
                    sim.close()
                    return True, path + [a], dict(info)
                if is_death(info, done):
                    continue
                c = cell(sim.ram)
                x = mario_level_x(sim.ram)
                if max_x is not None and x > max_x:   # local exploration cap (probe pipes here)
                    continue
                novel = c not in visited
                if novel:
                    visited.add(c)
                nstuck = 0 if (x > x_max or novel) else stuck + 1
                if nstuck > stuck_cap:
                    continue
                nnov = nov + (1 if novel else 0)
                score = (x - entry_x) + cov_bonus * nnov   # within-area progress + coverage
                candidates.append((c, sim.snapshot(), path + [a],
                                   max(x_max, x), nstuck, nnov, score, x))
        if not candidates:
            break
        by_key: dict[tuple, tuple] = {}
        for c, snp, pth, xm, st, nv, sc, x in candidates:
            if c not in by_key or sc > by_key[c][6]:
                by_key[c] = (c, snp, pth, xm, st, nv, sc, x)
        ranked = sorted(by_key.values(), key=lambda t: t[6], reverse=True)[:beam_width]
        beam = [(t[1], t[2], t[3], t[4], t[5]) for t in ranked]
        if ranked[0][7] > best_x:
            best_x, best_path = ranked[0][7], ranked[0][2]
        if checkpoint_path:
            import json as _json
            with open(checkpoint_path, "w") as _f:
                _json.dump({"path": best_path, "x_max": best_x, "depth": depth}, _f)
        if progress_every and depth % progress_every == 0:
            mx = max(t[7] for t in ranked)
            print(f"  [area {start_key}] depth {depth:4d} | beam {len(beam):3d} "
                  f"| best_x {best_x:5d} | frontier_x {mx:5d} | cells {len(visited):5d} "
                  f"| nodes {nodes:8d} | {nodes/(time.perf_counter()-t0):6.0f} n/s", flush=True)

    sim.close()
    return False, best_path, {}


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
