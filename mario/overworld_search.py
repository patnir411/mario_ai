"""Two-tier overworld meta-search for SMB3 (SMA4).

The high level is a planner over the SMB3 overworld graph; the low level is the
existing per-level beam search invoked as an option.  Both tiers run over one
shared emulator core (Stable-Retro is single-instance per process), so a map-node
snapshot flows into the level solver and the cleared level's snapshot flows back.

This module provides:
  * map-node discovery (BFS over cursor moves + a robust enterability test),
  * the post-level-clear "settle to a controllable map" wait,
  * (M2.2) the meta-search that sequences level solves to beat a world.

Discovery and the meta-search take an `SMA4OverworldAdapter` (which owns the core);
the level tier uses the sibling `SMA4Adapter` level view over the same core.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path

_DIRS = [("LEFT",), ("RIGHT",), ("UP",), ("DOWN",)]


def _cursor(core) -> tuple[int, int]:
    return tuple(core.last_info.get("cursor", (0, 0)))


def _cursor_responds(core) -> bool:
    """Non-destructively test whether any map direction moves the cursor."""
    snap = core.snapshot()
    before = _cursor(core)
    moved = False
    for btn in _DIRS:
        core.restore(snap)
        for _ in range(12):
            core._step_buttons(btn)
        for _ in range(6):
            core._step_buttons(())
        if _cursor(core) != before:
            moved = True
            break
    core.restore(snap)
    return moved


def settle_to_map(core, *, max_frames: int = 1400, probe_every: int = 60) -> bool:
    """Advance (NOOP) until the overworld cursor is input-responsive again.

    After a level clear the game shows a results/panel screen for ~hundreds of
    frames that reads as ``overworld`` (timer 0, stale cursor) but ignores input.
    Wait it out until a direction actually moves the cursor.
    """
    if core.last_info.get("mode") == "overworld" and _cursor_responds(core):
        return True
    for i in range(1, max_frames + 1):
        core._step_buttons(())
        if i % probe_every == 0 and core.last_info.get("mode") == "overworld" \
                and _cursor_responds(core):
            return True
    return core.last_info.get("mode") == "overworld" and _cursor_responds(core)


def _test_enterable(core) -> bool:
    """Non-destructively test whether pressing A here enters a level."""
    snap = core.snapshot()
    info = core.enter_level()
    ok = info.get("mode") == "level" and int(info.get("x_pos", 0)) > 5
    core.restore(snap)
    return ok


@dataclass
class MapGraph:
    snaps: dict = field(default_factory=dict)       # cursor -> core snapshot
    edges: dict = field(default_factory=dict)       # cursor -> {direction: cursor}
    enterable: dict = field(default_factory=dict)   # cursor -> bool

    def levels(self) -> list[tuple[int, int]]:
        return sorted(c for c, ok in self.enterable.items() if ok)


def discover_nodes(ow, *, max_nodes: int = 48, hold: int = 16, settle: int = 12,
                   enter_test: bool = True) -> MapGraph:
    """BFS the reachable overworld nodes from the adapter's current map state.

    `ow` is an `SMA4OverworldAdapter`; uses its shared core for stepping and the
    proven `enter_level` for the (non-destructive) enterability test.  Returns a
    `MapGraph` with a snapshot per cursor node, the directed edges between nodes,
    and which nodes are enterable levels.
    """
    core = ow.core
    g = MapGraph()
    start = core.snapshot()
    sc = _cursor(core)
    g.snaps[sc] = start
    g.edges[sc] = {}
    if enter_test:
        core.restore(start)
        g.enterable[sc] = _test_enterable(core)
    q = deque([sc])
    while q and len(g.snaps) < max_nodes:
        c = q.popleft()
        for btn in _DIRS:
            core.restore(g.snaps[c])
            for _ in range(hold):
                core._step_buttons(btn)
            for _ in range(settle):
                core._step_buttons(())
            nc = _cursor(core)
            if nc == c:
                continue
            g.edges[c][btn[0]] = nc
            if nc not in g.snaps:
                g.snaps[nc] = core.snapshot()
                g.edges.setdefault(nc, {})
                if enter_test:
                    core.restore(g.snaps[nc])
                    g.enterable[nc] = _test_enterable(core)
                q.append(nc)
    return g


@dataclass
class WorldResult:
    cleared: list = field(default_factory=list)       # node cursors cleared, in order
    solutions: dict = field(default_factory=dict)     # node -> {entry_snap, path, chunk_frames}
    failures: list = field(default_factory=list)      # nodes that could not be solved
    attempts: list = field(default_factory=list)       # per-level telemetry summaries
    beat_world: bool = False


def _load_snapshot(path: str | Path):
    obj = pickle.loads(Path(path).read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def remap_solution_path(solution: dict, target_action_names: list[str]) -> tuple[list[int], dict]:
    """Map a cached solution path into the currently configured action set.

    Cached SMA4 paths store action *indices*, so replaying them under a richer
    action set is only safe after translating through the source `action_names`.
    Older artifacts may have stale `action_names`; when that happens and the
    target action set already contains every referenced index, keep the indices
    and report the fallback in metadata.
    """
    path = [int(a) for a in solution["path"]]
    source_names = list(solution.get("action_names") or [])
    max_idx = max(path) if path else -1
    if source_names and max_idx < len(source_names):
        target = {name: i for i, name in enumerate(target_action_names)}
        mapped = []
        missing = []
        for idx in path:
            name = source_names[idx]
            if name not in target:
                missing.append(name)
            else:
                mapped.append(target[name])
        if missing:
            raise KeyError(f"cached solution uses actions not in target set: {sorted(set(missing))}")
        return mapped, {"remapped": True, "source_action_names_valid": True}
    if max_idx < len(target_action_names):
        return path, {
            "remapped": False,
            "source_action_names_valid": False,
            "reason": "source action_names missing/stale; indices fit target action set",
        }
    raise ValueError(
        f"cached path max action index {max_idx} exceeds target action set "
        f"size {len(target_action_names)} and source action_names are unavailable/stale")


def replay_cached_solution(core, solution: dict, *, entry_snap=None,
                           settle: bool = True, max_settle_frames: int = 1400) -> dict:
    """Restore a cached entry snapshot, replay a saved level solution, and settle.

    This is the fast-forward path for world sweeps: it advances the overworld via
    exact replay from a known entry snapshot instead of spending minutes
    re-solving already-verified levels.  It returns a telemetry summary and
    leaves `core` on the post-clear map when `settle=True`.
    """
    if entry_snap is not None:
        core.restore(entry_snap)
    path, mapping = remap_solution_path(solution, list(core.action_names))
    chunk_frames = int(solution.get("chunk_frames", 1))
    x_max = int(core.last_info.get("x_pos", 0))
    pspeed_max = int(core.last_info.get("pspeed", 0))
    speed_max = abs(int(core.last_info.get("speed_signed", core.last_info.get("speed", 0))))
    final_info = dict(core.last_info)
    done = False
    success_at = None
    for i, action in enumerate(path, start=1):
        final_info, done = core.run_chunk(action, chunk_frames)
        x_max = max(x_max, int(final_info.get("x_pos", 0)))
        pspeed_max = max(pspeed_max, int(final_info.get("pspeed", 0)))
        speed_max = max(speed_max, abs(int(final_info.get("speed_signed", final_info.get("speed", 0)))))
        if core.is_success(final_info):
            success_at = i
            break
        if core.is_death(final_info, done):
            break
    settled = False
    if success_at is not None and settle:
        settled = settle_to_map(core, max_frames=max_settle_frames)
        final_info = dict(core.last_info)
    return {
        "level_id": solution.get("level_id"),
        "solved": success_at is not None,
        "success_at": success_at,
        "path_len": len(path),
        "chunk_frames": chunk_frames,
        "x_max": x_max,
        "pspeed_max": pspeed_max,
        "speed_abs_max": speed_max,
        "final_info": final_info,
        "settled_to_map": settled,
        "path_mapping": mapping,
    }


def fast_forward_cached_solutions(ow, solution_paths: list[str | Path], *,
                                  cache_dir: str | Path = "runs/sma4_cache",
                                  artifact_dir: str | Path | None = None,
                                  verbose: bool = True) -> list[dict]:
    """Replay solved levels from cached entry snapshots to advance the map.

    For a solution with a `snapshot` field, that snapshot is restored directly.
    For early levels without a persisted entry snapshot, the current map is
    discovered, the first enterable node is entered, and that entry snapshot is
    saved to `cache_dir/<level_id>_entry.pkl`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    core = ow.core
    out = []
    for solution_path in solution_paths:
        solution_path = Path(solution_path)
        solution = json.loads(solution_path.read_text())
        level_id = str(solution.get("level_id") or solution_path.stem)
        entry_source = None
        snap_path = solution.get("snapshot")
        if snap_path and Path(snap_path).exists():
            entry_snap = _load_snapshot(snap_path)
            entry_source = str(snap_path)
        elif level_id == "1-1" and hasattr(core, "env") and hasattr(core, "boot_to_level"):
            core.env.reset()
            core.boot_to_level(1, 1)
            entry_snap = core.snapshot()
            entry_source = str(cache_dir / f"{level_id}_entry.pkl")
            with open(entry_source, "wb") as f:
                pickle.dump({"entry_snap": entry_snap, "level_id": level_id,
                             "node": [64, 32], "entry_info": core.last_info,
                             "source": "boot_to_level"}, f)
        else:
            graph = discover_nodes(ow)
            levels = graph.levels()
            if not levels:
                raise RuntimeError(f"no enterable node found before cached replay of {level_id}")
            node = tuple(solution.get("node") or levels[0])
            if node not in graph.snaps:
                node = levels[0]
            core.restore(graph.snaps[node])
            entry_info = core.enter_level()
            if entry_info.get("mode") != "level" or int(entry_info.get("x_pos", 0)) <= 5:
                raise RuntimeError(f"failed to enter cached level {level_id} at node {node}: {entry_info}")
            entry_snap = core.snapshot()
            entry_source = str(cache_dir / f"{level_id}_entry.pkl")
            with open(entry_source, "wb") as f:
                pickle.dump({"entry_snap": entry_snap, "level_id": level_id,
                             "node": list(node), "entry_info": entry_info}, f)
        summary = replay_cached_solution(core, solution, entry_snap=entry_snap)
        summary.update({"solution_path": str(solution_path), "entry_snapshot": entry_source})
        if verbose:
            print(f"[ow-cache] {level_id}: solved={summary['solved']} "
                  f"settled={summary['settled_to_map']} x_max={summary['x_max']} "
                  f"cursor={summary['final_info'].get('cursor')}", flush=True)
        if artifact_dir is not None:
            (artifact_dir / f"{level_id}_cached_replay.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
        out.append(summary)
        if not summary["solved"] or not summary["settled_to_map"]:
            break
    return out


def overworld_solve(ow, *, max_levels: int = 8, beam_width: int = 64,
                    chunk_frames: int = 6, depth: int = 600, verbose: bool = True,
                    artifact_dir: str | Path | None = None,
                    target_cursor: tuple[int, int] | None = None,
                    target_label: str | None = None,
                    cache_entries_dir: str | Path | None = None):
    """Beat levels in the current world by discovering + solving enterable nodes.

    Two-tier loop over one shared core (`ow.core` is the level view): discover map
    nodes, pick an uncleared enterable level, enter it, solve it with
    `beam_search_adapter` (+ `goal_suffix_search` for the goal panel), then replay
    the solution to reach the post-clear map and repeat.  Each node's solution is
    cached as `(entry_snapshot, path, chunk_frames)` so it replays deterministically
    from the saved entry state (cached *paths* desync if re-entered with different
    timing, but a restored entry snapshot is exact).
    """
    from mario.search import (beam_search_adapter, coverage_search_adapter,
                              goal_suffix_search)
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
    if cache_entries_dir is not None:
        cache_entries_dir = Path(cache_entries_dir)
        cache_entries_dir.mkdir(parents=True, exist_ok=True)

    core = ow.core
    saved_initial = core._initial_state
    result = WorldResult()
    try:
        while len(result.cleared) < max_levels:
            g = discover_nodes(ow)
            targets = [n for n in g.levels() if n not in result.cleared]
            if target_cursor is not None:
                targets = [target_cursor] if target_cursor in targets else []
            if verbose:
                print(f"[ow] nodes={sorted(g.snaps)} enterable={g.levels()} "
                      f"cleared={result.cleared} -> targets={targets}", flush=True)
            if not targets:
                result.beat_world = bool(result.cleared) and not result.failures
                break
            node = targets[0]
            node_tag = f"{node[0]}_{node[1]}"
            node_dir = None
            if artifact_dir is not None:
                node_dir = artifact_dir / f"node_{node_tag}"
                node_dir.mkdir(parents=True, exist_ok=True)
            core.restore(g.snaps[node])
            if target_label:
                core.level_id = target_label
                parts = target_label.split("-", 1)
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    core.world, core.stage = int(parts[0]), int(parts[1])
            entry_info = core.enter_level()
            if entry_info.get("mode") != "level" or int(entry_info.get("x_pos", 0)) <= 5:
                result.failures.append(node)
                break
            entry_snap = core.snapshot()
            entry_snapshot_path = None
            if cache_entries_dir is not None:
                label = target_label or f"node_{node_tag}"
                entry_snapshot_path = cache_entries_dir / f"{label}_entry.pkl"
                with open(entry_snapshot_path, "wb") as f:
                    pickle.dump({"entry_snap": entry_snap, "level_id": label,
                                 "node": list(node), "entry_info": entry_info}, f)
            core._initial_state = entry_snap  # search/finisher reset() -> level entry
            # Escalation ladder: plain beam -> Go-Explore coverage (novelty escape
            # for obstacle stalls) -> goal-panel finisher.
            beam_trace = node_dir / "beam_trace.jsonl" if node_dir is not None else None
            res = beam_search_adapter(core, beam_width=beam_width,
                                      chunk_frames=chunk_frames, max_depth=depth,
                                      stuck_cap=36, trace_path=beam_trace)
            path, cf, solved = res.path, res.chunk_frames, res.solved
            solver = "beam_search_adapter"
            cov = None
            if not solved:
                cov_trace = node_dir / "coverage_trace.jsonl" if node_dir is not None else None
                cov = coverage_search_adapter(core, beam_width=beam_width,
                                              chunk_frames=chunk_frames,
                                              max_depth=depth * 2, stuck_cap=80,
                                              trace_path=cov_trace)
                if cov.solved:
                    res = cov
                    path, cf, solved = cov.path, cov.chunk_frames, True
                    solver = "coverage_search_adapter"
                elif cov.x_max >= res.x_max:
                    res, path, cf = cov, cov.path, cov.chunk_frames
                    solver = "coverage_search_adapter"
            if not solved and res.x_max > 0:
                suffix = goal_suffix_search(core, path, cf)
                if suffix is not None:
                    path, cf, solved = suffix["path"], suffix["chunk_frames"], True
                    suffix_meta = suffix["metadata"]
                    final_info = dict(suffix["final_info"])
                    solver = f"{solver}+goal_suffix"
                else:
                    suffix_meta = None
                    final_info = dict(res.final_info)
            else:
                suffix_meta = None
                final_info = dict(res.final_info)
            if verbose:
                print(f"[ow] node {node}: solved={solved} x_max={res.x_max} "
                      f"path_len={len(path)}", flush=True)
            attempt = {
                "node": list(node),
                "solved": bool(solved),
                "solver": solver,
                "x_max": int(res.x_max),
                "depth_reached": int(res.depth_reached),
                "nodes_expanded": int(res.nodes_expanded),
                "chunk_frames": int(cf),
                "path_len": len(path),
                "path": path,
                "final_info": final_info,
                "goal_suffix": suffix_meta,
                "entry_snapshot": str(entry_snapshot_path) if entry_snapshot_path is not None else None,
                "beam_trace": str(beam_trace) if node_dir is not None else None,
                "coverage_trace": str(node_dir / "coverage_trace.jsonl") if cov is not None and node_dir is not None else None,
            }
            if node_dir is not None and path:
                import json
                from mario.render import make_contact_sheet_adapter

                sheet = node_dir / ("solved_contact.png" if solved else "partial_contact.png")
                core.restore(entry_snap)
                attempt["contact_sheet"] = make_contact_sheet_adapter(
                    core, path, cf, sheet, cols=5, rows=5, scale=1)
                (node_dir / "attempt_summary.json").write_text(
                    json.dumps(attempt, indent=2, sort_keys=True) + "\n")
            result.attempts.append(attempt)
            if not solved:
                result.failures.append(node)
                break
            # Replay the solution to land back on the (advanced) map.
            core.restore(entry_snap)
            for a in path:
                _info, done = core.run_chunk(a, cf)
                if core.is_success(core.last_info) or done:
                    break
            settle_to_map(core)
            result.cleared.append(node)
            result.solutions[node] = {"entry_snap": entry_snap, "path": path,
                                      "chunk_frames": cf}
        if len(result.cleared) >= max_levels and not result.failures:
            result.beat_world = True
    finally:
        core._initial_state = saved_initial
    return result
