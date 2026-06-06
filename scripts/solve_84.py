"""Phase 2 — solve 8-4 by LEG-BY-LEG chaining with page-aware coverage_search.

8-4 = 3 down-pipes (x1296/2432/3648) + 1 water side-pipe -> Bowser -> axe. A single coverage run
is too slow to chain all of it, so we chain: each run seeds from the prior best path (start_prefix)
and the page-progress score pushes it through the NEXT pipe. Repeat until flag_get.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84.py [per_leg_s] [max_legs]
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x, real_transition_strict
from mario.search import coverage_search
from mario.reward import is_success

PER_LEG = float(sys.argv[1]) if len(sys.argv) > 1 else 260.0
MAX_LEGS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CF = 8
CACHE = ROOT / "data/solutions/8-4.json"

def measure(path):
    """Replay path cf=8; return (n_real_transitions, final_x, flag)."""
    sim = MarioSim(8, 4); sim.reset(0)
    ib = dict(sim.last_info); rb = sim.ram.copy(); n = 0; flag = False
    for a in path:
        info, d = sim.run_chunk(a, CF)
        if real_transition_strict(ib, rb, info, sim.ram)[0]:
            n += 1
        ib, rb = dict(info), sim.ram.copy()
        if info.get("flag_get"):
            flag = True
        if d:
            break
    sim.close()
    return n, mario_level_x(sim.ram), flag

# Seed with the VERIFIED leg-1 entry (ckpt[:75] -> page 7) so we start past leg 1.
full = []
seed_ck = Path("/tmp/8_4_ckpt.json")
if seed_ck.exists():
    cand = json.loads(seed_ck.read_text()).get("path", [])[:75]
    np_, fx_, fl_ = measure(cand)
    if np_ >= 1:
        full = cand
        print(f"seeded from ckpt[:75]: pipes={np_} x={fx_}", flush=True)
prev = (-1, -1)
for leg in range(MAX_LEGS):
    print(f"\n=== LEG {leg} (prefix {len(full)} chunks) ===", flush=True)
    r = coverage_search(8, 4, beam_width=64, chunk_frames=CF, max_depth=4000,
                        time_budget_s=PER_LEG, stuck_cap=90, cov_bonus=90.0, ground_bonus=2.0,
                        loop_back_px=400, loop_needs_visited=True, pipe_macro_chunks=3,
                        page_aware=True, start_prefix=full or None, prefix_cf=CF,
                        progress_every=80)
    cand = full + list(r.path)
    npipe, fx, flag = measure(cand)
    print(f"leg {leg}: solved={r.solved} pipes={npipe} final_x={fx} flag={flag} cand_len={len(cand)}", flush=True)
    if r.solved or flag:
        npipe, fx, flag = measure(cand)
        if flag:
            CACHE.write_text(json.dumps({"path": cand, "solved": True, "by": "coverage_legchain",
                                         "chunk_frames": CF, "x_max": fx}))
            print(f"*** 8-4 SOLVED *** saved {CACHE} (pipes={npipe}, len={len(cand)})"); sys.exit(0)
    if (npipe, fx) <= prev:                       # no progress this leg -> stuck
        print(f"no progress (pipes,x)={(npipe,fx)} <= prev {prev}; stopping."); break
    prev = (npipe, fx); full = cand
    json.dump({"path": full}, open("/tmp/8_4_legchain.json", "w"))
print("did not reach flag; best chain saved /tmp/8_4_legchain.json")
