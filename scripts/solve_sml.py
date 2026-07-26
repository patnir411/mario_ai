"""Solve a Super Mario Land level through the generic adapter search.

Requires:
    MARIO_AI_SML_ROM=/path/to/SuperMarioLand.gb

Example:
    PYTHONPATH=. ./venv/bin/python scripts/solve_sml.py 1 1
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from mario.adapters import SMLAdapter, SML_SURFACE_ACTIONS
from mario.search import beam_search_adapter


def replay(adapter: SMLAdapter, path: list[int], chunk_frames: int, seed: int,
           post_clear_frames: int = 1200) -> tuple[bool, dict, int | None]:
    """Independently replay a candidate from the same declared fresh-boot root."""
    info = adapter.reset(seed=seed)
    start = (int(info.get("world", 1)), int(info.get("stage", 1)))
    done = False
    for action in path:
        info, done = adapter.run_chunk(action, chunk_frames)
        if adapter.is_success(info) or adapter.is_death(info, done):
            break
    if not adapter.is_success(info):
        return False, info, None
    for frame in range(1, post_clear_frames + 1):
        info, done = adapter.run_chunk(0, 1)
        if (int(info.get("world", start[0])), int(info.get("stage", start[1]))) != start:
            return True, info, frame
        if done and adapter.is_death(info, done):
            return False, info, None
    return True, info, None


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
    world = int(argv[1]) if len(argv) > 1 else 1
    stage = int(argv[2]) if len(argv) > 2 else 1
    seed = int(argv[3]) if len(argv) > 3 else 0
    beam = int(argv[4]) if len(argv) > 4 else 80
    depth = int(argv[5]) if len(argv) > 5 else 500
    chunk_frames = int(argv[6]) if len(argv) > 6 else 8

    rom = os.environ.get("MARIO_AI_SML_ROM")
    if not rom:
        raise SystemExit("MARIO_AI_SML_ROM must point to a legally obtained Super Mario Land ROM")

    run_id = time.strftime("%Y%m%d-%H%M%S-sml")
    run_dir = Path("runs") / run_id
    trace_path = run_dir / "trace.jsonl" if os.environ.get("MARIO_AI_TRACE") == "1" else None
    out_path = Path("data") / "solutions" / "sml" / f"{world}-{stage}.json"

    adapter = SMLAdapter(rom, world=world, stage=stage, actions=SML_SURFACE_ACTIONS)
    try:
        result = beam_search_adapter(
            adapter, beam_width=beam, chunk_frames=chunk_frames, max_depth=depth,
            seed=seed, stuck_cap=36, trace_path=trace_path, progress_every=10)
        verified, final_info, post_clear_advance_frames = replay(
            adapter, result.path, result.chunk_frames, seed)
        payload = {
            "game_id": "sml",
            "level_id": f"{world}-{stage}",
            "emulator": "pyboy",
            "rom_sha1": adapter.rom_sha1,
            "action_names": adapter.action_names,
            "chunk_frames": result.chunk_frames,
            "seed": seed,
            "path": result.path,
            "search_solved": bool(result.solved),
            "solved": False,
            "replay_verified": False,
            "replay_root": {
                "kind": "fresh_boot",
                "game_id": "sml",
                "level_id": f"{world}-{stage}",
                "seed": seed,
                "rom_sha1": adapter.rom_sha1,
            },
            "final_info": final_info,
            "post_clear_advance_frames": post_clear_advance_frames,
            "nodes_expanded": result.nodes_expanded,
            "depth_reached": result.depth_reached,
            "wall_clock_s": result.wall_clock_s,
            "trace": str(trace_path) if trace_path is not None else None,
        }
        artifact = _publish_attempt(
            payload, run_dir, out_path, replay_verified=verified)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 0 if verified else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
