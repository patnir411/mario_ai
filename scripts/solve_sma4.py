"""Solve a Super Mario Advance 4 / Super Mario Bros. 3 level.

Requires:
    MARIO_AI_SMA4_ROM=/path/to/SuperMarioAdvance4.gba

Usage:
    solve_sma4.py [world] [stage] [seed] [beam] [depth] [chunk_frames] [surface|physics|pspeed]

Example (World 1-1):
    MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' \
      ./venv/bin/python scripts/solve_sma4.py 1 1
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from mario.adapters import SMA4Adapter, SMA4_PHYSICS_ACTIONS, SMA4_PSPEED_ACTIONS
from mario.render import make_contact_sheet_adapter
from mario.search import beam_search_adapter, goal_suffix_search


def replay(adapter: SMA4Adapter, path: list[int], chunk_frames: int, seed: int,
           post_clear_frames: int = 900) -> tuple[bool, dict, int | None]:
    """Independently replay a candidate from the same declared fresh-boot root."""
    info = adapter.reset(seed=seed)
    done = False
    for action in path:
        info, done = adapter.run_chunk(action, chunk_frames)
        if adapter.is_success(info) or adapter.is_death(info, done):
            break
    if not adapter.is_success(info):
        return False, info, None
    for frame in range(1, post_clear_frames + 1):
        info, done = adapter.run_chunk(0, 1)
        if int(info.get("lives", adapter._start_lives)) < adapter._start_lives:
            return False, info, None
        if int(info.get("x_pos", 0)) < 16 and int(info.get("time", 0)) == 0:
            return True, info, frame
    return True, info, None


def _publish_attempt(
    payload: dict,
    run_dir: Path,
    canonical_path: Path,
    *,
    replay_verified: bool,
) -> dict:
    """Always persist the attempt, but promote only an independently replayed win.

    The normalization here is deliberately downstream of search: a solver may
    report a terminal candidate that is not reproducible.  Such a candidate is
    evidence, not a canonical solution, and must not replace an older verified
    artifact.
    """
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
    world = int(argv[1]) if len(argv) > 1 else 1
    stage = int(argv[2]) if len(argv) > 2 else 1
    seed = int(argv[3]) if len(argv) > 3 else 0
    beam = int(argv[4]) if len(argv) > 4 else 96
    depth = int(argv[5]) if len(argv) > 5 else 500
    chunk_frames = int(argv[6]) if len(argv) > 6 else 6
    action_set = argv[7] if len(argv) > 7 else os.environ.get("MARIO_AI_SMA4_ACTIONS", "surface")

    rom = os.environ.get("MARIO_AI_SMA4_ROM")
    if not rom:
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")

    run_id = time.strftime(f"%Y%m%d-%H%M%S-sma4_{world}_{stage}")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "beam_trace.jsonl"
    out_path = Path("data") / "solutions" / "sma4" / f"{world}-{stage}.json"

    if action_set == "physics":
        actions = SMA4_PHYSICS_ACTIONS
    elif action_set == "pspeed":
        actions = SMA4_PSPEED_ACTIONS
    elif action_set == "surface":
        actions = None
    else:
        raise SystemExit(f"unknown action set {action_set!r}; use surface, physics, or pspeed")

    adapter = SMA4Adapter(rom, world=world, stage=stage, actions=actions)
    try:
        result = beam_search_adapter(
            adapter, beam_width=beam, chunk_frames=chunk_frames, max_depth=depth,
            seed=seed, stuck_cap=36, trace_path=trace_path, progress_every=20)
        solution_path = result.path
        solution_chunk_frames = result.chunk_frames
        suffix = None
        solver = "beam_search_adapter"
        solved = result.solved
        # The in-level x coordinate caps at the goal boundary, so a pure-x beam
        # plateaus before the goal panel/card; a level-agnostic finisher hits it.
        if not solved and result.x_max > 0:
            suffix = goal_suffix_search(adapter, result.path, result.chunk_frames)
            if suffix is not None:
                solution_path = suffix["path"]
                solution_chunk_frames = suffix["chunk_frames"]
                solver = "beam_search_adapter+goal_suffix"
                solved = True
        solver_reported_solved = bool(solved)
        verified, final_info, post_clear_advance_frames = replay(
            adapter, solution_path, solution_chunk_frames, seed)
        payload = {
            "game_id": "sma4",
            "level_id": f"{world}-{stage}",
            "emulator": "stable-retro/mgba",
            "rom_sha1": adapter.rom_sha1,
            "action_names": adapter.action_names,
            "chunk_frames": solution_chunk_frames,
            "seed": seed,
            "path": solution_path,
            "search_solved": solver_reported_solved,
            "solved": False,
            "replay_verified": False,
            "replay_root": {
                "kind": "fresh_boot",
                "game_id": "sma4",
                "level_id": f"{world}-{stage}",
                "seed": seed,
                "rom_sha1": adapter.rom_sha1,
            },
            "post_clear_advance_frames": post_clear_advance_frames,
            "final_info": final_info,
            "beam_solved": bool(result.solved),
            "goal_suffix": suffix["metadata"] if suffix is not None else None,
            "solver": solver,
            "action_set": action_set,
            "x_max": result.x_max,
            "nodes_expanded": result.nodes_expanded,
            "depth_reached": result.depth_reached,
            "wall_clock_s": result.wall_clock_s,
            "trace": str(trace_path) if trace_path is not None else None,
        }
        sheet = run_dir / ("solved_contact.png" if verified else "partial_contact.png")
        payload["contact_sheet"] = make_contact_sheet_adapter(
            adapter, solution_path, solution_chunk_frames, sheet,
            seed=seed, post_clear_frames=120 if verified else 0, cols=5, rows=5)
        artifact = _publish_attempt(
            payload, run_dir, out_path, replay_verified=verified)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 0 if verified else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
