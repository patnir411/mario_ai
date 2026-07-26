"""Solve ONE level to the flag with beam search; cache to data/solutions/{w}-{s}.json.

For linear levels (overworld/underground/water). Maze castles use coverage_search instead.
Idempotent: if a solved cache exists, exits immediately. Runs in-process (snapshots aren't
picklable) — launch several of these as separate OS processes to parallelize across cores.

    ./venv/bin/python scripts/solve_beam.py <world> <stage> [beam_width] [max_depth]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.search import beam_search
from mario.io import write_json_atomic
from mario.solution_verification import (
    replay_verify_stock_path,
    verify_stock_solution,
)

SOL = ROOT / "data" / "solutions"


def main() -> None:
    w = int(sys.argv[1]); s = int(sys.argv[2])
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    SOL.mkdir(parents=True, exist_ok=True)
    cache = SOL / f"{w}-{s}.json"
    if cache.exists():
        previous = verify_stock_solution(cache, seed=0)
        if previous.replay_verified:
            print(f"{w}-{s}: already replay-verified (cached)")
            return

    print(f"[beam] {w}-{s} beam={beam} max_depth={max_depth}", flush=True)
    r = beam_search(w, s, beam_width=beam, max_depth=max_depth, progress_every=50)
    replay_verified, replay_reason = (
        replay_verify_stock_path(w, s, list(r.path), r.chunk_frames, seed=0)
        if r.solved else (False, "search_did_not_reach_flag")
    )
    print(f"{w}-{s}: search_solved={r.solved} replay_verified={replay_verified} "
          f"x_max={r.x_max} path_len={len(r.path)} "
          f"nodes={r.nodes_expanded} wall={r.wall_clock_s:.0f}s", flush=True)
    write_json_atomic(cache, {
        "path": list(r.path),
        "solved": replay_verified,
        "replay_verified": replay_verified,
        "search_claimed_solved": bool(r.solved),
        "invalid_reason": None if replay_verified else f"replay_{replay_reason}",
        "replay_seed": 0,
        "by": "beam",
        "chunk_frames": r.chunk_frames,
        "x_max": r.x_max,
    })
    print(f"saved replay-gated result {cache}")


if __name__ == "__main__":
    main()
