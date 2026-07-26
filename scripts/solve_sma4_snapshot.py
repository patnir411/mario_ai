"""Run an adapter search from a saved SMA4 entry snapshot.

This is the fast iteration path for hard levels discovered by the overworld
planner.  Example:

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_sma4_snapshot.py \
      runs/sma4_cache/1-2_entry.pkl 1-2 64 450 6 pspeed
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PHYSICS_ACTIONS, SMA4_PSPEED_ACTIONS  # noqa: E402
from mario.render import make_contact_sheet_adapter  # noqa: E402
from mario.search import beam_search_adapter, coverage_search_adapter, goal_suffix_search  # noqa: E402


def _load_snapshot(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def replay_from_snapshot(
    adapter: SMA4Adapter,
    root_snapshot,
    path: list[int],
    chunk_frames: int,
) -> tuple[bool, dict]:
    """Independently replay a candidate from its declared entry snapshot."""
    adapter.restore(root_snapshot)
    info = adapter.last_info
    done = False
    for action in path:
        info, done = adapter.run_chunk(action, chunk_frames)
        if adapter.is_success(info) or adapter.is_death(info, done):
            break
    return bool(adapter.is_success(info)), dict(info)


def _publish_attempt(
    payload: dict,
    run_dir: Path,
    canonical_path: Path,
    *,
    replay_verified: bool,
) -> dict:
    """Persist every attempt and promote only a replay-verified solution."""
    artifact = dict(payload)
    declared_root = artifact.get("replay_root")
    promoted = bool(replay_verified and isinstance(declared_root, dict) and declared_root)
    artifact["solved"] = promoted
    artifact["replay_verified"] = promoted
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "attempt_summary.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if promoted:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        raise SystemExit(__doc__)
    snap_path = Path(argv[1])
    level_id = argv[2]
    beam = int(argv[3]) if len(argv) > 3 else 64
    depth = int(argv[4]) if len(argv) > 4 else 450
    chunk_frames = int(argv[5]) if len(argv) > 5 else 6
    action_set = argv[6] if len(argv) > 6 else os.environ.get("MARIO_AI_SMA4_ACTIONS", "physics")
    time_budget = float(argv[7]) if len(argv) > 7 else 900.0

    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")
    if not snap_path.exists():
        raise SystemExit(f"snapshot not found: {snap_path}")
    if action_set == "physics":
        actions = SMA4_PHYSICS_ACTIONS
    elif action_set == "pspeed":
        actions = SMA4_PSPEED_ACTIONS
    elif action_set == "surface":
        actions = None
    else:
        raise SystemExit(f"unknown action set {action_set!r}; use surface, physics, or pspeed")

    run_id = time.strftime(f"%Y%m%d-%H%M%S-sma4_snapshot_{level_id.replace('-', '_')}_{action_set}")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = _load_snapshot(snap_path)
    snapshot_sha256 = hashlib.sha256(snap_path.read_bytes()).hexdigest()

    adapter = SMA4Adapter(actions=actions, boot_target="overworld")
    try:
        adapter.level_id = level_id
        adapter.restore(snap)
        adapter._initial_state = adapter.snapshot()
        adapter._start_lives = int(adapter.last_info.get("lives", adapter._start_lives))

        beam_trace = run_dir / "beam_trace.jsonl"
        result = beam_search_adapter(
            adapter, beam_width=beam, chunk_frames=chunk_frames, max_depth=depth,
            stuck_cap=48, trace_path=beam_trace, progress_every=25)
        solver = "beam_search_adapter"
        path, cf, solved = result.path, result.chunk_frames, result.solved
        cov = None
        if not solved:
            cov_trace = run_dir / "coverage_trace.jsonl"
            cov = coverage_search_adapter(
                adapter, beam_width=beam, chunk_frames=chunk_frames,
                max_depth=depth * 2, stuck_cap=120, time_budget_s=time_budget,
                trace_path=cov_trace, progress_every=25)
            if cov.solved or cov.x_max >= result.x_max:
                result = cov
                path, cf, solved = cov.path, cov.chunk_frames, cov.solved
                solver = "coverage_search_adapter"
        suffix_meta = None
        if not solved and result.x_max > 0:
            suffix = goal_suffix_search(adapter, path, cf)
            if suffix is not None:
                path, cf, solved = suffix["path"], suffix["chunk_frames"], True
                suffix_meta = suffix["metadata"]
                solver = f"{solver}+goal_suffix"

        solver_reported_solved = bool(solved)
        replay_verified, replay_final_info = replay_from_snapshot(
            adapter, adapter._initial_state, path, cf)
        sheet = run_dir / ("solved_contact.png" if replay_verified else "partial_contact.png")
        adapter.restore(adapter._initial_state)
        sheet_meta = make_contact_sheet_adapter(
            adapter, path, cf, sheet, post_clear_frames=120 if replay_verified else 0,
            cols=5, rows=5)
        summary = {
            "game_id": "sma4",
            "level_id": level_id,
            "emulator": "stable-retro/mgba",
            "rom_sha1": adapter.rom_sha1,
            "snapshot": str(snap_path),
            "replay_root": {
                "kind": "snapshot",
                "game_id": "sma4",
                "level_id": level_id,
                "path": str(snap_path),
                "sha256": snapshot_sha256,
                "rom_sha1": adapter.rom_sha1,
            },
            "action_set": action_set,
            "action_names": adapter.action_names,
            "solver": solver,
            "search_solved": solver_reported_solved,
            "solved": False,
            "replay_verified": False,
            "x_max": int(result.x_max),
            "chunk_frames": int(cf),
            "path": path,
            "path_len": len(path),
            "depth_reached": int(result.depth_reached),
            "nodes_expanded": int(result.nodes_expanded),
            "wall_clock_s": float(result.wall_clock_s),
            "final_info": replay_final_info,
            "search_final_info": dict(result.final_info),
            "goal_suffix": suffix_meta,
            "beam_trace": str(beam_trace),
            "coverage_trace": str(run_dir / "coverage_trace.jsonl") if cov is not None else None,
            "contact_sheet": sheet_meta,
        }
        out_path = Path("data") / "solutions" / "sma4" / f"{level_id}.json"
        artifact = _publish_attempt(
            summary, run_dir, out_path, replay_verified=replay_verified)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        print("run dir:", run_dir)
        return 0 if replay_verified else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
