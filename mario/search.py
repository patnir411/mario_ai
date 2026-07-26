"""Beam-search teacher: use the emulator as a forward model to find a trajectory
that beats a level.

In-process only: snapshots (NativeStateSnapshot) are not picklable, so one search runs
on a single MarioSim. Parallelism happens ACROSS searches (levels/states), not within.

Core loop (DESIGN.md §6): from each beam node, try every configured action chunk via snapshot
restore; score the resulting state with the death-aware reward; drop dead nodes; dedup
by spatial bucket; keep the top-k. Return as soon as a node reaches the flag.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mario.actions import SMB1_N_ACTIONS
from mario.reward import DEFAULT, RewardWeights, is_death, is_success, state_score


# The vocabulary is emulator-free; the emulator class itself remains lazy so
# adapter-backed GBA/GB search imports do not require NES packages.
N_ACTIONS = SMB1_N_ACTIONS
MarioSim = None


def _new_nes_sim(*args, **kwargs):
    """Construct MarioSim, loading the optional NES stack on first use."""
    global MarioSim, N_ACTIONS
    if MarioSim is None:
        try:
            from mario.env import MarioSim as _MarioSim, N_ACTIONS as _N_ACTIONS
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "NES search requires the NES emulator stack. Install this project "
                "with `uv pip install -e '.[nes]'` in a dedicated NES environment."
            ) from exc
        MarioSim = _MarioSim
        N_ACTIONS = _N_ACTIONS
    return MarioSim(*args, **kwargs)


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
    obs_hist: object = None  # rolling list of last-K entity obs (only set for a temporal prior)


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


def _json_safe(obj):
    """Best-effort JSON normalization for trace rows built from emulator info dicts."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _adapter_physics_cell(info: dict, tile: int = 16) -> tuple:
    """Physics-aware adapter cell for SMB3-style movement affordances.

    Coarse `(x,y)` novelty collapses states that are physically very different:
    small vs powered-up, slow vs P-speed, grounded vs airborne, or different
    subpixel phase.  Default missing values to zero so generic fake adapters and
    other games keep working without extra fields.
    """
    x_fixed = int(info.get("x_fixed", int(info.get("x_pos", 0)) << 8))
    y = int(info.get("y_pos", 0))
    speed = int(info.get("speed_signed", info.get("speed", 0)))
    speed_bucket = max(-8, min(8, speed // 16))
    pspeed_bucket = int(info.get("pspeed", 0)) // 16
    powerup = int(info.get("powerup", 0))
    airborne = int(info.get("airborne", info.get("in_air", info.get("spin_jump", 0)))) > 0
    flight_bucket = int(info.get("flight", info.get("fly_time", 0))) // 16
    return (
        int(info.get("x_pos", 0)) // tile,
        y // tile,
        powerup,
        pspeed_bucket,
        speed_bucket,
        int(airborne),
        flight_bucket,
        (x_fixed & 0xff) // 32,
    )


def beam_search_adapter(adapter, *, beam_width: int = 40, chunk_frames: int = 4,
                        max_depth: int = 800, seed: int = 0, stuck_cap: int = 24,
                        tile: int = 16, progress_every: int = 0,
                        trace_path: str | Path | None = None) -> SearchResult:
    """Game-neutral beam search over a `GameAdapter`.

    This is the portability seam for SML/SMB3/SMW.  The adapter owns action count,
    snapshot semantics, terminal checks, progress, and dedup cells; the search only
    ranks alive states and returns a chunk-action path.
    """
    import json

    root_info = adapter.reset(seed=seed)
    x_start = float(adapter.progress(root_info))
    root = Node(adapter.snapshot(), 0.0, root_info, [], int(x_start), 0, 0)
    root.node_id = 0
    beam = [root]
    best = root
    nodes = 0
    next_node_id = 1
    t0 = time.perf_counter()
    depth = 0

    trace_f = None
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_f = trace_path.open("w")

    def write_trace(row: dict) -> None:
        if trace_f is not None:
            trace_f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")

    try:
        for depth in range(1, max_depth + 1):
            candidates: list[Node] = []
            for node in beam:
                for a in range(adapter.n_actions):
                    adapter.restore(node.snap)
                    info, done = adapter.run_chunk(a, chunk_frames)
                    nodes += 1
                    child_id = next_node_id
                    next_node_id += 1
                    success = adapter.is_success(info)
                    death = adapter.is_death(info, done)
                    progress = float(adapter.progress(info))
                    stuck = node.stuck + (0 if progress > node.x_max else 1)
                    score = (progress - x_start) - DEFAULT.stuck * stuck
                    row = {
                        "game_id": getattr(adapter, "game_id", ""),
                        "level_id": getattr(adapter, "level_id", ""),
                        "seed": seed,
                        "depth": depth,
                        "node_id": child_id,
                        "parent_id": getattr(node, "node_id", None),
                        "action_idx": a,
                        "action_name": adapter.action_names[a] if a < len(adapter.action_names) else str(a),
                        "score": score,
                        "progress": progress,
                        "terminal": bool(done or success or death),
                        "death": bool(death),
                        "success": bool(success),
                        "cell": adapter.cell(tile),
                        "info": info,
                    }
                    write_trace(row)
                    if success:
                        return SearchResult(
                            True, node.path + [a], chunk_frames, info, int(progress),
                            node.frames + chunk_frames, depth, nodes,
                            time.perf_counter() - t0, beam_width, DEFAULT)
                    if death or stuck > stuck_cap:
                        continue
                    child = Node(adapter.snapshot(), score, info, node.path + [a],
                                 int(max(node.x_max, progress)), stuck,
                                 node.frames + chunk_frames)
                    child.node_id = child_id
                    candidates.append(child)
            if not candidates:
                break

            best_by_key: dict[tuple, Node] = {}
            for c in candidates:
                adapter.restore(c.snap)
                k = adapter.cell(tile)
                if k not in best_by_key or c.score > best_by_key[k].score:
                    best_by_key[k] = c
            beam = sorted(best_by_key.values(), key=lambda c: c.score, reverse=True)[:beam_width]
            if beam and beam[0].x_max > best.x_max:
                best = beam[0]
            if progress_every and depth % progress_every == 0:
                print(f"  depth {depth:4d} | beam {len(beam):3d} | best_x {best.x_max:5d} "
                      f"| nodes {nodes:7d} | {nodes/(time.perf_counter()-t0):6.0f} n/s",
                      flush=True)
    finally:
        if trace_f is not None:
            trace_f.close()

    return SearchResult(False, best.path, chunk_frames, best.info, best.x_max,
                        best.frames, depth, nodes, time.perf_counter() - t0,
                        beam_width, DEFAULT)


def goal_suffix_search(adapter, path, chunk_frames, *,
                       band: int = 360, max_snapshots: int = 16,
                       run_action: str = "RIGHT+B",
                       jump_actions: tuple[str, ...] = (
                           "RIGHT+A", "RIGHT+A+B", "LEFT+RIGHT+A+B",
                           "LEFT+A", "LEFT+A+B", "A", "A+B",
                       ),
                       delays=range(0, 36, 3), holds=range(20, 64, 4),
                       tail: int = 48,
                       brake_actions: tuple[str, ...] = ("NOOP", "RIGHT", "LEFT", "LEFT+B", "RIGHT+B"),
                       brake_delays=range(0, 73, 8),
                       tail_actions: tuple[str, ...] = ("RIGHT+B", "RIGHT", "NOOP"),
                       post_settle_frames: int = 36,
                       card_return_action: str = "LEFT+B",
                       card_return_frames: tuple[int, ...] = (50, 40, 60, 30, 70, 80),
                       card_return_holds: tuple[int, ...] = (14, 10, 18, 22, 28, 34),
                       card_return_tail_actions: tuple[str, ...] = ("LEFT", "NOOP", "RIGHT", "RIGHT+B", "LEFT+B"),
                       time_budget_s: float | None = 30.0):
    """Level-agnostic end-of-level finisher for adapter games whose progress
    coordinate caps near the goal.

    Many SMA4/SMB3 levels end at a goal panel/roulette card after the in-level x
    coordinate has plateaued at the boundary, so a pure-x beam collapses every
    final-approach state into one bucket and never triggers the terminal.  This
    replays the beam path frame-by-frame, snapshots the handful of states closest
    to the run's observed ``x_max`` (within ``band`` px) plus late goal-card
    state changes, then frame-searches short finish templates.  It tries both the
    old run -> jump -> run suffix and a coast/brake -> jump -> tail suffix; the
    latter matters when a P-speed run overshoots the card hit-window.  Success is
    still the adapter's own ``is_success`` predicate.

    Returns ``{"path","chunk_frames","final_info","metadata"}`` (frame-level,
    chunk_frames=1) or ``None`` if no suffix reaches the goal.
    """
    deadline = None if time_budget_s is None else time.perf_counter() + float(time_budget_s)

    def expired() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    names = adapter.action_names
    if run_action not in names:
        return None
    run = names.index(run_action)
    jumps = [names.index(j) for j in jump_actions if j in names]
    if not jumps:
        return None
    brake_idxs = [names.index(a) for a in brake_actions if a in names]
    tail_idxs = [names.index(a) for a in tail_actions if a in names]
    noop = names.index("NOOP") if "NOOP" in names else 0
    card_return = names.index(card_return_action) if card_return_action in names else None
    card_return_tail_idxs = [names.index(a) for a in card_return_tail_actions if a in names]
    if not brake_idxs:
        brake_idxs = [run]
    if not tail_idxs:
        tail_idxs = [run]
    if not card_return_tail_idxs:
        card_return_tail_idxs = tail_idxs

    # Pass 1: find the run's max progress.
    adapter.reset(seed=0)
    x_max = 0
    for action in path:
        info, done = adapter.run_chunk(action, chunk_frames)
        x_max = max(x_max, int(info.get("x_pos", 0)))
        if adapter.is_death(info, done):
            break

    # Pass 2: snapshot frame-level cuts across the approach window (within `band`
    # px below x_max). Frame-level cuts matter because chunk boundaries can miss
    # the narrow card-hit window. Goal-card state changes are kept as extra
    # candidates when they occur near the end; the value is noisy earlier in a
    # level, so x-window filtering remains the main guardrail.
    adapter.reset(seed=0)
    snapshots: list[tuple[int, object, dict, str, tuple[int, ...]]] = []
    prefix: list[int] = []
    prev_goal = int(adapter.last_info.get("goalcard", 0))
    for action in path:
        for _ in range(chunk_frames):
            _obs, info, done = adapter.step(action)
            prefix.append(action)
            x = int(info.get("x_pos", 0))
            goal = int(info.get("goalcard", 0))
            reason = None
            if x >= x_max - band:
                reason = "x_band"
            if goal != prev_goal and x >= x_max - max(band * 2, band + 240):
                reason = "goalcard_change" if reason is None else reason + "+goalcard_change"
            if reason is not None:
                snapshots.append((len(prefix), adapter.snapshot(), dict(info), reason, tuple(prefix)))
            prev_goal = goal
            if adapter.is_death(info, done) or adapter.is_success(info):
                break
        if adapter.is_death(adapter.last_info, False) or adapter.is_success(adapter.last_info):
            break
    if len(snapshots) > max_snapshots:
        step = len(snapshots) / max_snapshots
        snapshots = [snapshots[int(i * step)] for i in range(max_snapshots)]

    post_snapshots: list[tuple[int, object, dict, str, tuple[int, ...]]] = []
    if post_settle_frames > 0 and not adapter.is_death(adapter.last_info, False) and not adapter.is_success(adapter.last_info):
        for frame in range(1, post_settle_frames + 1):
            _obs, info, done = adapter.step(noop)
            prefix.append(noop)
            if frame == 1 or frame % 4 == 0 or frame == post_settle_frames:
                post_snapshots.append((
                    len(prefix), adapter.snapshot(), dict(info), "post_settle", tuple(prefix),
                ))
            if adapter.is_death(info, done) or adapter.is_success(info):
                break
    snapshots.extend(post_snapshots)

    def try_suffix(cut_frames: int, snap, prefix_frames: tuple[int, ...], suffix: list[int], meta: dict):
        adapter.restore(snap)
        info = adapter.last_info
        done = False
        for frame, action in enumerate(suffix, start=1):
            if expired():
                return None
            _obs, info, done = adapter.step(action)
            if adapter.is_success(info):
                out_meta = dict(meta)
                out_meta.update({
                    "cut_frames": cut_frames,
                    "success_frame": frame,
                    "x_max": x_max,
                })
                return {
                    "path": list(prefix_frames) + suffix[:frame],
                    "chunk_frames": 1,
                    "final_info": dict(info),
                    "metadata": out_meta,
                }
            if adapter.is_death(info, done):
                break
        return None

    # Fast path for SMB3 roulette-card endings after high-speed overshoot: wait
    # for the cap/landing state, walk back left under the card, jump, then let
    # the end-walk routine take over. Keep this before the broader template scan
    # because it is far cheaper and addresses the common capped-progress failure.
    if card_return is not None:
        ordered = sorted(
            snapshots,
            key=lambda row: (
                row[3] != "post_settle",
                -int(row[2].get("y_pos", 0)),
                abs(int(row[2].get("speed_signed", row[2].get("speed", 0)))),
                -int(row[2].get("x_pos", 0)),
            ),
        )
        for cut, snap, snap_info, reason, prefix_frames in ordered[:max_snapshots]:
            for left_frames in card_return_frames:
                for jump in jumps:
                    for hold in card_return_holds:
                        for tail_action in card_return_tail_idxs:
                            if expired():
                                return None
                            suffix = [card_return] * left_frames + [jump] * hold + [tail_action] * max(tail, 96)
                            found = try_suffix(cut, snap, prefix_frames, suffix, {
                                "template": "card_return_jump",
                                "return_action": names[card_return],
                                "return_frames": left_frames,
                                "jump_action": names[jump],
                                "hold_frames": hold,
                                "tail_action": names[tail_action],
                                "snapshot_reason": reason,
                                "snapshot_info": snap_info,
                            })
                            if found is not None:
                                return found

    for cut, snap, snap_info, reason, prefix_frames in reversed(snapshots):
        # Original finish template: keep running, jump, keep running.
        for delay in delays:
            for jump in jumps:
                for hold in holds:
                    if expired():
                        return None
                    suffix = [run] * delay + [jump] * hold + [run] * tail
                    found = try_suffix(cut, snap, prefix_frames, suffix, {
                        "template": "run_jump_run",
                        "delay_frames": delay,
                        "jump_action": names[jump],
                        "hold_frames": hold,
                        "tail_action": names[run],
                        "snapshot_reason": reason,
                        "snapshot_info": snap_info,
                    })
                    if found is not None:
                        return found
        # P-speed/card levels can need braking before the jump.
        for brake in brake_idxs:
            for delay in brake_delays:
                for jump in jumps:
                    for hold in holds:
                        for tail_action in tail_idxs:
                            if expired():
                                return None
                            suffix = [brake] * delay + [jump] * hold + [tail_action] * tail
                            found = try_suffix(cut, snap, prefix_frames, suffix, {
                                "template": "brake_jump_tail",
                                "brake_action": names[brake],
                                "brake_frames": delay,
                                "jump_action": names[jump],
                                "hold_frames": hold,
                                "tail_action": names[tail_action],
                                "snapshot_reason": reason,
                                "snapshot_info": snap_info,
                            })
                            if found is not None:
                                return found
    return None


def coverage_search_adapter(adapter, *, beam_width: int = 64, chunk_frames: int = 6,
                            max_depth: int = 1500, seed: int = 0, stuck_cap: int = 80,
                            tile: int = 16, cov_bonus: float = 40.0,
                            stuck_penalty: float = 2.0, time_budget_s: float = 1500.0,
                            progress_every: int = 0,
                            physics_cell: bool = True,
                            trace_path: str | Path | None = None) -> SearchResult:
    """Adapter-generic Go-Explore: beam + a cell archive with a one-time novelty
    bonus per newly reached `(x_tile, y_tile)` cell.

    Plain `beam_search_adapter` plateaus where an obstacle (a pit/pipe needing a
    jump) gives no x-gain, so the frontier empties before clearing it.  Rewarding
    lineages that open NEW cells — crucially new *y* tiles, i.e. the airborne arc
    over the obstacle — makes the search value getting over/around it instead of
    only rightward x.  On open stretches novelty saturates and x-progress
    dominates, so it degrades to the proven beam.  A coarse `(x//tile, y//tile)`
    archive (not the adapter's fine cell) keeps novelty meaningful.
    """
    import json

    def coarse(info):
        if physics_cell:
            return _adapter_physics_cell(info, tile)
        return (int(info.get("x_pos", 0)) // tile, int(info.get("y_pos", 0)) // tile)

    root_info = adapter.reset(seed=seed)
    x_start = float(adapter.progress(root_info))
    visited = {coarse(root_info)}
    root = Node(adapter.snapshot(), 0.0, root_info, [], int(x_start), 0, 0)
    root.wp_idx = 0  # cumulative novelty count along the lineage
    beam = [root]
    best = root
    nodes = 0
    t0 = time.perf_counter()
    depth = 0
    trace_f = None
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_f = trace_path.open("w")

    def write_trace(row: dict) -> None:
        if trace_f is not None:
            trace_f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")

    try:
        for depth in range(1, max_depth + 1):
            if time.perf_counter() - t0 > time_budget_s:
                break
            scored: list[tuple[tuple, Node]] = []
            for node in beam:
                for a in range(adapter.n_actions):
                    adapter.restore(node.snap)
                    info, done = adapter.run_chunk(a, chunk_frames)
                    nodes += 1
                    progress = float(adapter.progress(info))
                    success = adapter.is_success(info)
                    death = adapter.is_death(info, done)
                    cl = coarse(info)
                    stuck = node.stuck + (0 if progress > node.x_max else 1)
                    novel_count = node.wp_idx + (0 if cl in visited else 1)
                    xmax = int(max(node.x_max, progress))
                    score = (xmax - x_start) + cov_bonus * novel_count - stuck_penalty * stuck
                    write_trace({
                        "game_id": getattr(adapter, "game_id", ""),
                        "level_id": getattr(adapter, "level_id", ""),
                        "solver": "coverage_search_adapter",
                        "seed": seed,
                        "depth": depth,
                        "action_idx": a,
                        "action_name": adapter.action_names[a] if a < len(adapter.action_names) else str(a),
                        "score": score,
                        "progress": progress,
                        "x_max": xmax,
                        "stuck": stuck,
                        "cell": cl,
                        "terminal": bool(done or success or death),
                        "death": bool(death),
                        "success": bool(success),
                        "info": info,
                    })
                    if success:
                        return SearchResult(
                            True, node.path + [a], chunk_frames, info,
                            int(adapter.progress(info)), node.frames + chunk_frames,
                            depth, nodes, time.perf_counter() - t0, beam_width, DEFAULT)
                    if death or stuck > stuck_cap:
                        continue
                    child = Node(adapter.snapshot(), score, info, node.path + [a],
                                 xmax, stuck, node.frames + chunk_frames)
                    child.wp_idx = novel_count
                    scored.append((cl, child))
            if not scored:
                break
            best_by: dict[tuple, Node] = {}
            for cl, c in scored:
                visited.add(cl)
                if cl not in best_by or c.score > best_by[cl].score:
                    best_by[cl] = c
            beam = sorted(best_by.values(), key=lambda c: c.score, reverse=True)[:beam_width]
            if beam[0].x_max > best.x_max:
                best = beam[0]
            if progress_every and depth % progress_every == 0:
                print(f"  cov depth {depth:4d} | beam {len(beam):3d} | best_x {best.x_max:5d} "
                      f"| cells {len(visited):6d} | nodes {nodes:7d} "
                      f"| {nodes / (time.perf_counter() - t0):6.0f} n/s", flush=True)
    finally:
        if trace_f is not None:
            trace_f.close()
    return SearchResult(False, best.path, chunk_frames, best.info, best.x_max,
                        best.frames, depth, nodes, time.perf_counter() - t0,
                        beam_width, DEFAULT)


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


def _mixed_policy_logp(logp, mix_eps: float = 0.0):
    """Return log((1-eps) * pi + eps * uniform) for a policy-prior vector.

    Weak learned priors are useful for ranking, but raw log-softmax outputs can be too
    confident off distribution. A small uniform mix keeps every action recoverable when
    coverage_search is using the prior as a soft score term.
    """
    if mix_eps <= 0.0:
        return logp
    import numpy as np

    arr = np.asarray(logp, dtype=np.float64)
    eps = min(max(float(mix_eps), 0.0), 1.0)
    if arr.size == 0:
        return arr
    p = np.exp(arr - np.max(arr))
    p /= max(float(p.sum()), 1e-12)
    p = (1.0 - eps) * p + eps / arr.size
    return np.log(np.clip(p, 1e-12, 1.0)).astype(np.float32)


def beam_search(world: int = 1, stage: int = 1, *, beam_width: int = 40,
                chunk_frames: int = 8, max_depth: int = 400,
                weights: RewardWeights = DEFAULT, seed: int = 0,
                stuck_cap: int = 12, progress_every: int = 0,
                start_prefix: list[int] | None = None,
                value_guide=None, value_weight: float = 600.0,
                policy_prior=None, policy_weight: float = 80.0,
                policy_topk: int | None = None) -> SearchResult:
    """Beam search to beat a level. With `start_prefix`, replay those chunks after reset
    and search the CONTINUATION from there (used for recovery trajectories in DAgger-lite
    dataset generation). The returned `path` is the continuation only; progress/x_start
    are measured from the post-prefix state.

    VALUE-GUIDED (DESIGN.md §10): pass a `value.ValueGuide` to add value_weight * V(s) to
    each node's score, where V≈P(a flag-reaching continuation exists). This makes a NARROW
    beam keep the solvable lineage and discard doomed-but-high-progress nodes at the frontier
    edge — a shallow/narrow search then behaves like a deeper/wider one. value_weight is in
    px-equivalent units (V in [0,1]); 600 ≈ "a doomed node is worth ~600px less progress"."""
    if policy_topk is not None and policy_topk <= 0:
        raise ValueError("policy_topk must be positive")
    sim = _new_nes_sim(world, stage)
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

    # TEMPORAL prior: maintain a rolling K-frame entity-obs history PER NODE so the prior can
    # condition on recent frames (a single-frame prior just uses .logp(ram,info)).
    pk = getattr(policy_prior, "ctx_k", None) if policy_prior is not None else None
    if pk:
        from mario.entity import entity_obs
        beam[0].obs_hist = [entity_obs(sim.ram, root_info)]

    best = beam[0]
    nodes = 0
    t0 = time.perf_counter()
    depth = 0

    for depth in range(1, max_depth + 1):
        candidates: list[Node] = []
        for node in beam:
            # POLICY PRIOR ("policy proposes, search disposes"): score log π(a|s) at this node,
            # optionally expand only its top-k actions — narrows branching without dropping the
            # solvable lineage (DESIGN.md §10 hybrid). Needs the node's live RAM.
            lp = None
            actions = range(N_ACTIONS)
            if policy_prior is not None:
                if pk:                              # temporal prior: stack the node's K history
                    stack = list(node.obs_hist)[-pk:]
                    while len(stack) < pk:
                        stack = [stack[0]] + stack
                    lp = policy_prior.logp_stack(stack)
                else:                               # single-frame prior
                    sim.restore(node.snap)
                    lp = policy_prior.logp(sim.ram, node.info)
                if len(lp) != N_ACTIONS:
                    raise ValueError(
                        f"policy_prior returned {len(lp)} actions, expected {N_ACTIONS}")
                if policy_topk is not None and policy_topk < N_ACTIONS:
                    actions = sorted(range(N_ACTIONS), key=lambda a: lp[a], reverse=True)[:policy_topk]
            for a in actions:
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
                if value_guide is not None:        # learned solvability guidance
                    from mario.observation import observe
                    score += value_weight * value_guide.p(observe(sim.ram, info))
                if lp is not None:                 # learned policy-prior bonus
                    score += policy_weight * float(lp[a])
                child_hist = None
                if pk:                              # extend the rolling K-frame history
                    child_hist = (list(node.obs_hist) + [entity_obs(sim.ram, info)])[-pk:]
                candidates.append(Node(sim.snapshot(), score, info,
                                       node.path + [a], area_x_max, stuck, frames,
                                       area=na, area_seq=area_seq, area_entry_x=area_entry_x,
                                       obs_hist=child_hist))
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
                    loop_needs_visited: bool = True, page_aware: bool = False, actions=None,
                    start_prefix: list[int] | None = None, prefix_cf: int | None = None,
                    checkpoint_path: str | None = None,
                    value_guide=None, value_weight: float = 600.0,
                    policy_prior=None, policy_weight: float = 80.0,
                    policy_topk: int | None = None, policy_mix_eps: float = 0.05,
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
    from mario.ram import mario_level_x, pipe_entering
    if policy_topk is not None and policy_topk <= 0:
        raise ValueError("policy_topk must be positive")
    AREA, APTR, Y_ADDR, FLOAT = 0x0760, 0x0750, 0x00CE, 0x001D
    PAGE_W = 1_000_000   # page-progress dominates x so ENTERING a pipe beats walking the surface
    if policy_prior is not None and actions is not None:
        raise ValueError("coverage_search policy_prior requires the default action vocabulary")
    observe_fn = None
    if value_guide is not None:
        from mario.observation import observe as observe_fn
    pk = getattr(policy_prior, "ctx_k", None) if policy_prior is not None else None
    entity_obs_fn = None
    if pk:
        from mario.entity import entity_obs as entity_obs_fn
    sim = (_new_nes_sim(world, stage, actions=actions) if actions is not None
           else _new_nes_sim(world, stage))
    sim.reset(seed=seed)
    prefix_actions = list(start_prefix) if start_prefix else []
    if prefix_actions:                 # replay a verified prefix (chain legs) before searching
        pcf = prefix_cf if prefix_cf is not None else chunk_frames
        for a in prefix_actions:
            _i, d = sim.run_chunk(a, pcf)
            if d:
                break

    def cell(ram) -> tuple:
        # page_aware: include $0750 so sub-area/page transitions (8-4 down-pipes keep $0760=3 but
        # advance $0750) are DISTINCT cells — lets the search chain THROUGH pipe entries instead of
        # collapsing the post-pipe pages together. Off by default (linear levels don't need it).
        base = (int(ram[AREA]), mario_level_x(ram) // tile, int(ram[Y_ADDR]) // tile)
        return base + (int(ram[APTR]),) if page_aware else base

    root_info = sim.last_info
    x_start = int(root_info.get("x_pos", 0))
    visited = {cell(sim.ram)}
    root = Node(sim.snapshot(), 0.0, root_info, [], x_start, 0, 0)
    root.wp_idx = 0   # reuse wp_idx as cumulative novelty count along the lineage
    root.area = int(sim.ram[AREA])
    root.ground_xmax = x_start   # furthest x reached while GROUNDED (float_state==0)
    root.aptr = int(sim.ram[APTR])   # $0750 page marker
    root.page_seq = 0                # count of REAL pipe/page entries along the lineage
    if entity_obs_fn is not None:
        root.obs_hist = [entity_obs_fn(sim.ram, root_info)]
    beam = [root]
    best = root

    def prog(n):   # page-progress FIRST (entered pipes), then grounded-x, then x
        return (getattr(n, "page_seq", 0), getattr(n, "ground_xmax", 0), n.x_max)
    nodes = 0
    n_actions = len(actions) if actions is not None else N_ACTIONS
    t0 = time.perf_counter()
    depth = 0

    for depth in range(1, max_depth + 1):
        if time.perf_counter() - t0 > time_budget_s:
            break
        candidates: list[Node] = []
        for node in beam:
            lp = None
            action_iter = range(n_actions)
            if policy_prior is not None:
                if pk:
                    stack = list(node.obs_hist)[-pk:]
                    while len(stack) < pk:
                        stack = [stack[0]] + stack
                    lp = policy_prior.logp_stack(stack)
                else:
                    sim.restore(node.snap)
                    lp = policy_prior.logp(sim.ram, node.info)
                if len(lp) != n_actions:
                    raise ValueError(f"policy_prior returned {len(lp)} actions, expected {n_actions}")
                lp = _mixed_policy_logp(lp, policy_mix_eps)
                if policy_topk is not None and policy_topk < n_actions:
                    k = max(1, int(policy_topk))
                    action_iter = sorted(range(n_actions), key=lambda i: lp[i], reverse=True)[:k]

            for a in action_iter:
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
                # page-progress: a REAL pipe/page entry = $0750 changed AND the gym in-step skip
                # teleported Mario (info.x_pos lags live RAM x) OR a live pipe state. This is the
                # 8-4 down-pipe signal ($0760 stays 3). A bare $0750 marker flip (no teleport) and
                # the page-LOOPBACK (no $0750 change, just a backward x jump) are NOT entries.
                live_x = mario_level_x(sim.ram)
                aptr = int(sim.ram[APTR])
                real_entry = (page_aware and aptr != node.aptr
                              and (abs(x - live_x) > 100 or pipe_entering(sim.ram)))
                page_seq = node.page_seq + (1 if real_entry else 0)
                # loop-prune: a sharp BACKWARD x-jump within the same PAGE is a maze loop teleport.
                # A real_entry resets the page (x is a fresh page coord), so never prune it.
                if (not real_entry and loop_back_px and c[0] == node.area
                        and x < node.x_max - loop_back_px
                        and (c in visited or not loop_needs_visited)):
                    continue
                novel = c not in visited
                if novel:
                    visited.add(c)
                stuck = 0 if (x > node.x_max or novel or real_entry) else node.stuck + 1
                if stuck > stuck_cap:
                    continue
                novelty = node.wp_idx + (1 if novel else 0)
                grounded = int(sim.ram[FLOAT]) == 0
                ground_xmax = max(node.ground_xmax, x) if grounded else node.ground_xmax
                # page-progress DOMINATES x so entering a pipe (page+1) beats walking the surface.
                score = (page_seq * PAGE_W
                         + state_score(info, x_start, 0, False, stuck, DEFAULT)
                         + cov_bonus * novelty
                         + area_bonus * int(sim.ram[AREA])
                         + ground_bonus * (ground_xmax - x_start))
                if value_guide is not None:
                    score += value_weight * value_guide.p(observe_fn(sim.ram, info))
                if lp is not None:
                    score += policy_weight * float(lp[a])
                # reset x_max on a real page entry (the new page has fresh x coords)
                new_xmax = live_x if real_entry else max(node.x_max, x)
                child = Node(sim.snapshot(), score, info, node.path + [a],
                             new_xmax, stuck, node.frames + chunk_frames)
                child.wp_idx = novelty
                child.area = c[0]
                child.ground_xmax = ground_xmax
                child.aptr = aptr
                child.page_seq = page_seq
                if entity_obs_fn is not None:
                    child.obs_hist = (list(node.obs_hist) + [entity_obs_fn(sim.ram, info)])[-pk:]
                candidates.append((c, child))

            # pipe-entry macro: sustained DOWN in one atomic step (the down-pipe descent), so a
            # multi-chunk entry completes without its mid-crouch states being pruned. Kept on a
            # real AREA change OR (page_aware) a real page entry ($0750 change + gym-skip teleport).
            if pipe_macro_chunks:
                sim.restore(node.snap)
                m_done = False
                for _ in range(pipe_macro_chunks):
                    info, m_done = sim.run_chunk(7, chunk_frames)
                    live_x = mario_level_x(sim.ram)
                    entered = (int(sim.ram[AREA]) != node.area
                               or (page_aware and int(sim.ram[APTR]) != node.aptr
                                   and (abs(int(info.get("x_pos", live_x)) - live_x) > 100
                                        or pipe_entering(sim.ram))))
                    if is_success(info) or entered or m_done:
                        break
                if is_success(info):
                    sim.close()
                    return SearchResult(
                        True, node.path + [7] * pipe_macro_chunks, chunk_frames, info,
                        int(info.get("x_pos", 0)), node.frames, depth, nodes,
                        time.perf_counter() - t0, beam_width, DEFAULT)
                na = int(sim.ram[AREA]); na_ptr = int(sim.ram[APTR]); live_x = mario_level_x(sim.ram)
                real = (na != node.area or (page_aware and na_ptr != node.aptr
                        and (abs(int(info.get("x_pos", live_x)) - live_x) > 100 or pipe_entering(sim.ram))))
                if not m_done and real:                   # entered a pipe → new area/page!
                    c = cell(sim.ram); x = int(info.get("x_pos", 0))
                    visited.add(c)
                    ps = node.page_seq + 1
                    score = (ps * PAGE_W + state_score(info, x_start, 0, False, 0, DEFAULT)
                             + cov_bonus * (node.wp_idx + 1) + area_bonus * na
                             + ground_bonus * (node.ground_xmax - x_start))
                    if value_guide is not None:
                        score += value_weight * value_guide.p(observe_fn(sim.ram, info))
                    if lp is not None and len(lp) > 7:
                        score += policy_weight * float(lp[7])
                    mc = Node(sim.snapshot(), score, info, node.path + [7] * pipe_macro_chunks,
                              live_x, 0, node.frames)
                    mc.wp_idx = node.wp_idx + 1; mc.area = na; mc.ground_xmax = node.ground_xmax
                    mc.aptr = na_ptr; mc.page_seq = ps
                    if entity_obs_fn is not None:
                        mc.obs_hist = (list(node.obs_hist) + [entity_obs_fn(sim.ram, info)])[-pk:]
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
                stuck_cap: int = 70, cov_bonus: float = 50.0, x_jump_px: int = 0,
                strict: bool = True,
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
    from mario.ram import (mario_level_x, signed, AREA_NUMBER, AREA_POINTER,
                           pipe_entering, stage_key)
    Y_ADDR, XSPD = 0x00CE, 0x0057
    sim = (_new_nes_sim(world, stage, actions=actions) if actions is not None
           else _new_nes_sim(world, stage))
    n_actions = len(actions) if actions is not None else N_ACTIONS
    sim.reset(seed=seed)
    if start_prefix:                       # prefix may use a coarser cf than exploration
        pcf = prefix_cf if prefix_cf is not None else chunk_frames
        for a in start_prefix:
            _i, d = sim.run_chunk(a, pcf)
            if d:
                break
    # SUCCESS = a REAL room/level transition. Codex-verified semantics (vs the old $0750-only
    # test, which false-positives because $0750 is a PENDING-destination marker, not entry):
    #   (a) flag_get; (b) a real pipe/area entry underway (pipe_entering: $06DE!=0 / $000E in
    #   {2,3}) — the decisive signal for flagpole-less water side-pipes (2-2/7-2); (c) the
    #   STAGE byte advanced; (d) fallback: the ($0760,$0750) area key left the visited set
    #   (still useful for maze castles where the pointer genuinely flips).
    start_key = (int(sim.ram[AREA_NUMBER]), int(sim.ram[AREA_POINTER]))
    start_stage = stage_key(sim.ram)
    # forbid = rooms already visited along the chain (incl. start) so we don't "succeed" by
    # walking BACK into a prior room (the 8-4 loop returns 229->101); success needs a NEW room.
    forbid = {tuple(k) for k in (forbid_keys or [])} | {start_key}
    entry_x = mario_level_x(sim.ram)

    def transitioned(info, ram) -> bool:
        if is_success(info) or pipe_entering(ram) or stage_key(ram) != start_stage:
            return True
        if strict:
            # STRICT: a bare ($0760,$0750) flip is a PENDING marker (false positive), NOT entry.
            # A real entry surfaces as the gym in-step skip teleporting Mario => info.x_pos
            # (pre-skip) vs live-RAM x (post-skip) mismatch (down-pipe forward teleport, etc.).
            ix = info.get("x_pos")
            return ix is not None and abs(int(ix) - mario_level_x(ram)) > 100
        return (int(ram[AREA_NUMBER]), int(ram[AREA_POINTER])) not in forbid

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
                prev_x = mario_level_x(sim.ram)   # for the x_jump (hidden-pipe) signal
                info, done = sim.run_chunk(a, chunk_frames)
                nodes += 1
                # x_jump_px>0: a large BACKWARD jump landing in an UNVISITED cell is a real
                # pipe/area entry whose destination shares the area bytes (2-2-class), which the
                # plain transitioned() (area-key/$06DE/stage) cannot see (gym fast-forwards it).
                # A jump back into a VISITED cell is the maze LOOP (8-4 page16->12) — NOT success.
                jumped = (x_jump_px and (prev_x - mario_level_x(sim.ram)) > x_jump_px
                          and cell(sim.ram) not in visited)
                if jumped or transitioned(info, sim.ram):   # flag / pipe / stage / new room / hidden-pipe jump
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


def search_from_state(sim: Any, root_snap, *, world: int, stage: int,
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
