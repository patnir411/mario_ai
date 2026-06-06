"""Phase 2 — solve 8-4 with PAGE-AWARE coverage_search (chains through the down-pipes).

8-4's route = 3 down-pipes (x1296/2432/3648, DOWN-held descent onto pipe-top 10/11) + 1 water
side-pipe -> Bowser -> axe. The pipes keep $0760=3 but advance $0750/page, so the cell key must be
page-aware (else post-pipe pages collapse and the search can't chain). pipe_macro_chunks tries a
sustained DOWN entry; loop_needs_visited prunes only the page-loop (visited cell). Success = flag.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84.py [budget_s]
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import ACTIONS
from mario.search import coverage_search

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
CACHE = ROOT / "data/solutions/8-4.json"
# default 9 actions + descent-control combos (indices 0-8 preserved so pipe_macro DOWN=7 works)
WIDE = ACTIONS + [["right", "down"], ["left", "down"], ["down", "A"]]

print(f"[8-4] page-aware coverage_search budget={BUDGET}s", flush=True)
r = coverage_search(8, 4, beam_width=56, chunk_frames=8, max_depth=6000,
                    time_budget_s=BUDGET, stuck_cap=90, cov_bonus=90.0,
                    ground_bonus=2.0, loop_back_px=400, loop_needs_visited=True,
                    pipe_macro_chunks=3, page_aware=True, actions=None,
                    checkpoint_path="/tmp/8_4_ckpt2.json", progress_every=60)
print(f"8-4: solved={r.solved} x_max={r.x_max} path_len={len(r.path)} "
      f"nodes={r.nodes_expanded} wall={r.wall_clock_s:.0f}s", flush=True)
if r.solved:
    CACHE.write_text(json.dumps({"path": r.path, "solved": True, "by": "coverage_pageaware",
                                 "chunk_frames": r.chunk_frames, "x_max": r.x_max,
                                 "actions": "wide_84"}))
    print(f"saved {CACHE} solved=True")
else:
    print("not solved — checkpoint /tmp/8_4_ckpt2.json")
