"""Solve ONE non-linear level (maze castle / vertical / warp-zone) with coverage_search.

Coverage_search keeps x-progress dominant (clears precision sections like the plain beam)
while a mild novelty bonus keeps loop-escape / vertical cells alive (the route past the
8-4 page-checkpoint, the 4-2 elevator climb). Use a BIG beam so coverage doesn't starve
the forward-precision lineage. Caches to data/solutions/{w}-{s}.json.

    ./venv/bin/python scripts/solve_coverage.py <world> <stage> [beam] [cov_bonus] \
        [stuck_cap] [budget_s] [chunk_frames]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.search import coverage_search
from mario.io import new_run_id, run_dir, write_json_atomic
from mario.render import make_contact_sheet

SOL = ROOT / "data" / "solutions"


def main() -> None:
    w = int(sys.argv[1]); s = int(sys.argv[2])
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 160
    cov = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    stuck = int(sys.argv[5]) if len(sys.argv) > 5 else 80
    budget = float(sys.argv[6]) if len(sys.argv) > 6 else 3600.0
    cf = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    area = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    ground = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    loopbk = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    pipemc = int(sys.argv[11]) if len(sys.argv) > 11 else 0
    SOL.mkdir(parents=True, exist_ok=True)
    cache = SOL / f"{w}-{s}.json"
    if cache.exists() and json.loads(cache.read_text()).get("solved"):
        print(f"{w}-{s}: already solved (cached)"); return

    print(f"[coverage] {w}-{s} beam={beam} cov_bonus={cov} stuck_cap={stuck} "
          f"budget={budget}s cf={cf}", flush=True)
    ckpt = str(SOL / f".ckpt_{w}_{s}.json")
    r = coverage_search(w, s, beam_width=beam, chunk_frames=cf, time_budget_s=budget,
                        cov_bonus=cov, stuck_cap=stuck, area_bonus=area, ground_bonus=ground,
                        loop_back_px=loopbk, pipe_macro_chunks=pipemc,
                        checkpoint_path=ckpt, progress_every=50)
    print(f"{w}-{s}: solved={r.solved} x_max={r.x_max} path_len={len(r.path)} "
          f"nodes={r.nodes_expanded} wall={r.wall_clock_s:.0f}s", flush=True)
    # always dump the best path + a contact sheet so a failure is inspectable / iterable
    rid = new_run_id(f"cov_{w}_{s}")
    d = run_dir(rid)
    write_json_atomic(d / "search_path.json",
                      {"path": r.path, "chunk_frames": cf, "solved": r.solved,
                       "x_max": r.x_max})
    try:
        make_contact_sheet(w, s, 0, r.path, cf, d / "contact_sheet.png", cols=6, rows=5)
        print(f"contact sheet: {d/'contact_sheet.png'}", flush=True)
    except Exception as e:
        print("WARN sheet:", e)
    if r.solved:
        cache.write_text(json.dumps(
            {"path": r.path, "solved": True, "by": "coverage_search",
             "chunk_frames": cf, "x_max": r.x_max}))
        print(f"saved {cache}")


if __name__ == "__main__":
    main()
