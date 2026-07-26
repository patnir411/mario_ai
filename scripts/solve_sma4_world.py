"""Drive the SMB3 (SMA4) two-tier overworld meta-search for one world.

Discovers map nodes, solves enterable levels with per-level search (the M2 loop),
and reports which levels were cleared.  This is the first end-to-end exercise of
the two-tier agent.

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_sma4_world.py \
      [max_levels] [beam] [depth] [chunk_frames] [surface|physics|pspeed]

Add `--fast-forward-solved` to replay cached `data/solutions/sma4/{1-1,1-2}.json`
first, then start searching from the post-1-2 map.

Use `--target-cursor=X,Y --target-label=1-3` to force a specific discovered
map node and persist `runs/sma4_cache/<label>_entry.pkl`.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import (  # noqa: E402
    SMA4Adapter,
    SMA4OverworldAdapter,
    SMA4_PHYSICS_ACTIONS,
    SMA4_PSPEED_ACTIONS,
)
from mario.overworld_search import fast_forward_cached_solutions, overworld_solve  # noqa: E402


def main(argv: list[str]) -> int:
    fast_forward = "--fast-forward-solved" in argv
    target_cursor = None
    target_label = None
    filtered = []
    for arg in argv:
        if arg == "--fast-forward-solved":
            continue
        if arg.startswith("--target-cursor="):
            raw = arg.split("=", 1)[1]
            x, y = raw.split(",", 1)
            target_cursor = (int(x), int(y))
            continue
        if arg.startswith("--target-label="):
            target_label = arg.split("=", 1)[1]
            continue
        filtered.append(arg)
    argv = filtered
    max_levels = int(argv[1]) if len(argv) > 1 else 2
    beam = int(argv[2]) if len(argv) > 2 else 80
    depth = int(argv[3]) if len(argv) > 3 else 700
    chunk_frames = int(argv[4]) if len(argv) > 4 else 6
    action_set = argv[5] if len(argv) > 5 else os.environ.get("MARIO_AI_SMA4_ACTIONS", "surface")

    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")

    run_id = time.strftime("%Y%m%d-%H%M%S-sma4_world")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if action_set == "physics":
        core = SMA4Adapter(actions=SMA4_PHYSICS_ACTIONS, boot_target="overworld")
        ow = SMA4OverworldAdapter(core=core)
    elif action_set == "pspeed":
        core = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
        ow = SMA4OverworldAdapter(core=core)
    elif action_set == "surface":
        ow = SMA4OverworldAdapter()
    else:
        raise SystemExit(f"unknown action set {action_set!r}; use surface, physics, or pspeed")
    t0 = time.time()
    fast_forwarded = []
    try:
        ow.reset()
        if fast_forward:
            fast_forwarded = fast_forward_cached_solutions(
                ow,
                [Path("data/solutions/sma4/1-1.json"), Path("data/solutions/sma4/1-2.json")],
                cache_dir=Path("runs/sma4_cache"),
                artifact_dir=run_dir / "fast_forward",
                verbose=True,
            )
        result = overworld_solve(ow, max_levels=max_levels, beam_width=beam,
                                 chunk_frames=chunk_frames, depth=depth, verbose=True,
                                 artifact_dir=run_dir,
                                 target_cursor=target_cursor,
                                 target_label=target_label,
                                 cache_entries_dir=Path("runs/sma4_cache"))
    finally:
        ow.close()

    summary = {
        "world": 1,
        "cleared": [list(n) for n in result.cleared],
        "failures": [list(n) for n in result.failures],
        "beat_world": result.beat_world,
        "n_cleared": len(result.cleared),
        "path_lens": {f"{n[0]},{n[1]}": len(s["path"]) for n, s in result.solutions.items()},
        "attempts": result.attempts,
        "action_set": action_set,
        "fast_forwarded": fast_forwarded,
        "target_cursor": list(target_cursor) if target_cursor is not None else None,
        "target_label": target_label,
        "wall_clock_s": time.time() - t0,
    }
    (run_dir / "world_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Persist full solutions (entry snapshots + paths) for later stitching/replay.
    with open(run_dir / "world_solutions.pkl", "wb") as f:
        pickle.dump({"cleared": result.cleared, "solutions": result.solutions,
                     "failures": result.failures, "attempts": result.attempts}, f)
    print(json.dumps(summary, indent=2))
    print("run dir:", run_dir)
    return 0 if result.cleared else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
