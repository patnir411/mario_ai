"""Phase 2 — reopen 8-4 with the corrected coverage_search (loop_needs_visited fix).

The old blanket loop-prune discarded EVERY same-area backward x-jump (incl. real hidden-pipe
entries). With loop_needs_visited=True, only jumps that return to an ALREADY-VISITED cell (the
page16->page12 maze loop) are pruned; a backward jump into a NEW cell (a real pipe/area entry that
keeps the area byte, 2-2-class) now survives and is explored. Success = flag.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84.py [budget_s]
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.search import coverage_search

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
CACHE = ROOT / "data/solutions/8-4.json"

print(f"[8-4] coverage_search loop_needs_visited=True budget={BUDGET}s", flush=True)
r = coverage_search(8, 4, beam_width=64, chunk_frames=8, max_depth=4000,
                    time_budget_s=BUDGET, stuck_cap=80, cov_bonus=80.0,
                    area_bonus=3000.0, ground_bonus=2.0,
                    loop_back_px=400, loop_needs_visited=True,
                    checkpoint_path="/tmp/8_4_ckpt.json", progress_every=100)
print(f"8-4: solved={r.solved} x_max={r.x_max} path_len={len(r.path)} "
      f"nodes={r.nodes_expanded} wall={r.wall_clock_s:.0f}s", flush=True)
if r.solved:
    CACHE.write_text(json.dumps({"path": r.path, "solved": True, "by": "coverage_loopfix",
                                 "chunk_frames": r.chunk_frames, "x_max": r.x_max}))
    print(f"saved {CACHE} solved=True")
else:
    print("not solved — checkpoint at /tmp/8_4_ckpt.json (best x reached)")
