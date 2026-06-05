"""Phase 3 — solve 7-2 (underwater) reusing the 2-2 side-pipe recipe.

7-2 shares 2-2's geometry (exit side-pipe at level-x ~3024) but has different enemy timing, so it
needs its own swim path. Recipe: search a nav path to the mouth band, then from a truncation point
HOLD RIGHT to descend into the side-pipe (detected by the post-step x-jump, real_transition), then
beam surface area-2 to the flag.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_72.py [nav_budget_depth]
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x, real_transition, area_key, stage_key
from mario.search import beam_search, search_from_state
from mario.render import replay

CACHE = ROOT / "data/solutions/7-2.json"
MOUTH = 3010   # same exit column as 2-2

# 1) Nav: search a path through the water toward the mouth band.
print("[7-2] nav search ...", flush=True)
nav = beam_search(7, 2, beam_width=64, chunk_frames=8, max_depth=320, stuck_cap=16, progress_every=60)
print(f"  nav x_max={nav.x_max} path_len={len(nav.path)}", flush=True)
navpath = nav.path

# 2) Sweep truncation points: from each, HOLD RIGHT and look for the side-pipe entry (x-jump).
def reach(prefix):
    s = MarioSim(7, 2); s.reset(seed=0)
    for a in prefix:
        _, d = s.run_chunk(a, 8)
        if d: return None
    return s

best = None
for pn in range(len(navpath), max(0, len(navpath) - 90), -2):
    s = reach(navpath[:pn])
    if s is None: continue
    info_b = dict(s.last_info); ram_b = s.ram.copy(); ss = stage_key(s.ram)
    visited = set()
    entered = 0
    for f in range(110):
        info_a, d = s.run_chunk(1, 1); entered += 1
        fired, reason = real_transition(info_b, ram_b, info_a, s.ram, visited_cells=visited)
        if fired and reason in ("x_jump", "info_ram_mismatch", "area_key", "stage", "stage_ram"):
            best = (pn, entered, s.snapshot(), reason); break
        info_b, ram_b = dict(info_a), s.ram.copy()
        if d: break
    if best:
        print(f"  ENTRY found: prefix={best[0]} right_frames={best[1]} reason={best[3]}", flush=True)
        break

if not best:
    print("no side-pipe entry found from nav truncations — needs deeper nav/probe (fallback)."); sys.exit(1)

# 3) Beam surface area-2 to the flag from the post-entry snapshot.
pn, entered, post, _ = best
s = reach(navpath[:pn])
for _ in range(entered):
    s.run_chunk(1, 1)
post = s.snapshot()
tail = search_from_state(s, post, world=7, stage=2, beam_width=80, depth=500,
                         chunk_frames=8, max_seconds=180)

# 4) Normalize to cf=1, replay-verify, cache.
cf1 = []
for a in navpath[:pn]:
    cf1 += [a] * 8
cf1 += [1] * entered
for a in tail:
    cf1 += [a] * 8
recs = replay(7, 2, 0, cf1, 1)
beat = recs[-1]["flag"]; lastx = recs[-1]["info"].get("x_pos")
print(f"normalized cf1 len={len(cf1)} replay beat={beat} flag x_pos={lastx}", flush=True)
if beat:
    CACHE.write_text(json.dumps({"path": cf1, "solved": True, "by": "holdright_pipe+beam_area2",
                                 "chunk_frames": 1, "x_max": lastx}))
    print(f"saved {CACHE} solved=True")
else:
    print("entered pipe but area-2 beam did not reach flag — extend depth/budget"); sys.exit(1)
