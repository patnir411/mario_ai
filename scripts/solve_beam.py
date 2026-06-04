"""Solve ONE level to the flag with beam search; cache to data/solutions/{w}-{s}.json.

For linear levels (overworld/underground/water). Maze castles use coverage_search instead.
Idempotent: if a solved cache exists, exits immediately. Runs in-process (snapshots aren't
picklable) — launch several of these as separate OS processes to parallelize across cores.

    ./venv/bin/python scripts/solve_beam.py <world> <stage> [beam_width] [max_depth]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.search import beam_search

SOL = ROOT / "data" / "solutions"


def main() -> None:
    w = int(sys.argv[1]); s = int(sys.argv[2])
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    SOL.mkdir(parents=True, exist_ok=True)
    cache = SOL / f"{w}-{s}.json"
    if cache.exists() and json.loads(cache.read_text()).get("solved"):
        print(f"{w}-{s}: already solved (cached)"); return

    print(f"[beam] {w}-{s} beam={beam} max_depth={max_depth}", flush=True)
    r = beam_search(w, s, beam_width=beam, max_depth=max_depth, progress_every=50)
    print(f"{w}-{s}: solved={r.solved} x_max={r.x_max} path_len={len(r.path)} "
          f"nodes={r.nodes_expanded} wall={r.wall_clock_s:.0f}s", flush=True)
    cache.write_text(json.dumps(
        {"path": r.path, "solved": r.solved, "by": "beam",
         "chunk_frames": r.chunk_frames, "x_max": r.x_max}))
    print(f"saved {cache}")


if __name__ == "__main__":
    main()
