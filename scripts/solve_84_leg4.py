"""LEG 4 of 8-4: from the water area $02 (legs 1-3 = /tmp/8_4_prefix_leg123.json),
swim to the side-pipe and enter it (the 2-2/7-2 recipe: hold-RIGHT into the $6c sideways
pipe) -> Bowser area $65 ep16.

Success: AreaPointer $0750/$0751 -> ($65,$10)=(101,16), OR swim flag $0704 clears (left
water), OR a per-frame live-x teleport >100. Go-Explore the water with swim actions.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84_leg4.py [budget_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
PREFIX = json.load(open("/tmp/8_4_prefix_leg123.json"))["path"]   # cf1, in water $02
PAGE = lambda r: int(r[0x006D]); Y = lambda r: int(r[0x00CE])
SWIM = lambda r: int(r[0x0704]); PTR = lambda r: (int(r[0x0750]), int(r[0x0751]))

# REAL entry only: a per-frame live-x teleport (the gym-skip signal). The bare ptr=(101,16)
# is a PENDING destination marker (row-$0e object) and is NOT an entry — do not trust it.

sim = MarioSim(8, 4); sim.reset(0)
for a in PREFIX:
    if sim.run_chunk(a, 1)[1]: break
base = sim.snapshot(); r = sim.ram
start_swim = SWIM(r); start_ptr = PTR(r)
print(f"leg-4 seed: page={PAGE(r)} x={mario_level_x(r)} Y={Y(r)} swim={SWIM(r)} ptr={PTR(r)}", flush=True)

def cell(r): return (PAGE(r), mario_level_x(r) // 8, Y(r) // 12)
archive = {cell(sim.ram): (base, [])}
rng = random.Random(0); t0 = time.time(); iters = 0; best_x = mario_level_x(r)
# swim action set: right, right+A(stroke), A(up), down, right+down, noop
SWIMA = [1, 1, 2, 2, 5, 7, 0]
while time.time() - t0 < BUDGET:
    iters += 1
    snap, pre = rng.choice(list(archive.values()))
    sim.restore(snap); p = list(pre)
    for _ in range(rng.randint(2, 16)):
        a = rng.choice(SWIMA)
        xb = mario_level_x(sim.ram); _, d = sim.run_chunk(a, 4); xa = mario_level_x(sim.ram)
        p += [a] * 4
        if is_death(sim.last_info, d) or d: break
        r = sim.ram
        if abs(xa - xb) > 100:                       # REAL gym-skip teleport
            print(f"  *** LEG-4 SIDE-PIPE (teleport {xb}->{xa}): ptr={PTR(r)} swim={SWIM(r)} "
                  f"x={mario_level_x(r)} page={PAGE(r)} after {len(p)} cf1 ***", flush=True)
            json.dump({"leg4_cf1": p, "dest_ptr": PTR(r), "dest_swim": SWIM(r)},
                      open("/tmp/8_4_leg4.json", "w"))
            print("saved /tmp/8_4_leg4.json"); sim.close(); sys.exit(0)
        best_x = max(best_x, mario_level_x(r))
        c = cell(r)
        if c not in archive:
            archive[c] = (sim.snapshot(), list(p))
    if iters % 3000 == 0:
        print(f"  iters={iters} arch={len(archive)} best_x={best_x} t={time.time()-t0:.0f}s", flush=True)
print(f"\nNO LEG-4 side-pipe in {BUDGET}s (iters={iters}, archive={len(archive)}, best_x={best_x}).")
sim.close()
