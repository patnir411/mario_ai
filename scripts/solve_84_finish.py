"""LEG 5 + ASSEMBLE: from the Bowser area (legs 1-4), beam to the axe, then concatenate
the whole cf1 chain, verify replay-from-reset beat=True, and cache data/solutions/8-4.json.

Requires (all verified REAL, not pending-marker):
  /tmp/8_4_prefix_leg123.json  (cf1, reaches water $02)
  /tmp/8_4_leg4.json           (leg4_cf1, reaches Bowser area $65 ep16)

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84_finish.py [max_s]
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.search import search_from_state
from mario.reward import is_success

MAXS = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
CACHE = ROOT / "data/solutions/8-4.json"

prefix = json.load(open("/tmp/8_4_prefix_leg123.json"))["path"]      # cf1
leg4 = json.load(open("/tmp/8_4_leg4.json"))["leg4_cf1"]             # cf1
chain = list(prefix) + list(leg4)

sim = MarioSim(8, 4); sim.reset(0)
for a in chain:
    if sim.run_chunk(a, 1)[1]: break
r = sim.ram
print(f"post-leg4 (Bowser area): x={mario_level_x(r)} page={r[0x006D]} area={r[0x0760]} "
      f"ptr={r[0x0750]}/{r[0x0751]} swim={r[0x0704]} grounded={r[0x001D]==0}", flush=True)
root = sim.snapshot()

# Leg 5: beam to the axe (flag_get / clear). cf8 for speed; expand to cf1 for the final path.
leg5_cf8 = search_from_state(sim, root, world=8, stage=4, beam_width=48, depth=400,
                             chunk_frames=8, max_seconds=MAXS, stuck_cap=24)
sim.restore(root)
# check whether leg5 actually cleared
sim2 = MarioSim(8, 4); sim2.reset(0)
for a in chain: sim2.run_chunk(a, 1)
cleared = False
for a in leg5_cf8:
    info, d = sim2.run_chunk(a, 8)
    if is_success(info): cleared = True; break
    if d: break
sim2.close()
print(f"leg5 len={len(leg5_cf8)} cleared(flag_get)={cleared}", flush=True)
if not cleared:
    r = sim.ram
    print(f"LEG-5 did NOT reach the axe/flag (best x={mario_level_x(sim.ram)}). Needs more search/waypoint.")
    json.dump({"leg5_cf8_partial": leg5_cf8}, open("/tmp/8_4_leg5.json", "w"))
    sim.close(); sys.exit(1)

full = list(chain) + [a for a in leg5_cf8 for _ in range(8)]
# VERIFY replay-from-reset
v = MarioSim(8, 4); v.reset(0); beat = False
for a in full:
    info, d = v.run_chunk(a, 1)
    if is_success(info): beat = True; break
    if d: break
v.close()
print(f"replay-from-reset beat={beat} full_len={len(full)}", flush=True)
if beat:
    CACHE.write_text(json.dumps({"path": full, "solved": True, "by": "downpipe_chain",
                                 "chunk_frames": 1, "world": 8, "stage": 4}))
    print(f"*** 8-4 SOLVED *** cached {CACHE}")
else:
    print("replay verification FAILED — not caching.")
sim.close()
