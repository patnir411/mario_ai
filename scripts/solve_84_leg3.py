"""LEG 3 of 8-4: from the verified page-12 state (legs 1+2 = /tmp/8_4_prefix_leg12.json),
navigate to pipe-3 (p14c4r6, level_x 3648) and enter it (fall-onto-pipe-top + DOWN, same
mechanic as pipe-1) -> WATER area $02.

Success (robust, multi-signal): area_number $0760 changes from 3, OR swim flag $0704 set,
OR a per-frame live-RAM x jump >100 with a forward page advance. Verified later by raw
frame_advance. Navigation in cf8 (fast); entry attempts in cf1.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84_leg3.py [budget_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
PREFIX = json.load(open("/tmp/8_4_prefix_leg12.json"))["path"]   # cf1, reaches page12
PAGE = lambda r: int(r[0x006D]); Y = lambda r: int(r[0x00CE])
GROUNDED = lambda r: int(r[0x001D]) == 0; AREA = lambda r: int(r[0x0760]); SWIM = lambda r: int(r[0x0704])

def water(r): return AREA(r) != 3 or SWIM(r) != 0

# descent micro-sequences (cf1): approach (right/run) then sustained DOWN; some with a hop
def dseqs():
    out = []
    for nr in range(0, 18):
        for ra in (1, 3):
            out.append([ra] * nr + [7] * 24)
            out.append([ra] * nr + [2] + [7] * 24)
            out.append([ra] * nr + [4] + [7] * 24)
    return out
DSEQS = dseqs()

def try_enter(sim, snap):
    """From snap, try descent seqs; success = live-x teleport (>100) OR water. Return burst or None."""
    for seq in DSEQS:
        sim.restore(snap); burst = []
        for a in seq:
            xb = mario_level_x(sim.ram)
            _, d = sim.run_chunk(a, 1)
            xa = mario_level_x(sim.ram); burst.append(a)
            if water(sim.ram) or (abs(xa - xb) > 100 and PAGE(sim.ram) > 12):
                return burst, ("water" if water(sim.ram) else "xjump"), xa, AREA(sim.ram), SWIM(sim.ram)
            if d: break
    return None, None, None, None, None

sim = MarioSim(8, 4); sim.reset(0)
for a in PREFIX:
    if sim.run_chunk(a, 1)[1]: break
base = sim.snapshot(); r = sim.ram
print(f"leg-3 seed: page={PAGE(r)} x={mario_level_x(r)} Y={Y(r)} grounded={GROUNDED(r)} area={AREA(r)}", flush=True)

def cell(r): return (PAGE(r), mario_level_x(r) // 8, Y(r) // 16, GROUNDED(r))
archive = {cell(sim.ram): (base, [])}
rng = random.Random(0); t0 = time.time(); iters = 0; best_x = mario_level_x(r)
while time.time() - t0 < BUDGET:
    iters += 1
    snap, pre = rng.choice(list(archive.values()))
    sim.restore(snap); p = list(pre)
    for _ in range(rng.randint(2, 14)):
        a = rng.choice([1, 1, 3, 2, 2, 4, 0, 5, 6])    # right-biased + jumps
        _, d = sim.run_chunk(a, 8); p.append(a)
        if is_death(sim.last_info, d) or d: break
        r = sim.ram; x = mario_level_x(r)
        if water(r):                                    # stumbled straight into water
            burst_full = p
            print(f"  *** LEG-3 (explore) WATER area={AREA(r)} swim={SWIM(r)} x={x} after {len(p)} cf8 ***", flush=True)
            json.dump({"leg3_nav_cf8": p, "mode": "explore"}, open("/tmp/8_4_leg3.json", "w"))
            print("saved /tmp/8_4_leg3.json"); sim.close(); sys.exit(0)
        best_x = max(best_x, x)
        if 3540 <= x <= 3720 and GROUNDED(r):           # near pipe-3 -> attempt entry
            here = sim.snapshot()
            burst, why, dx, da, dsw = try_enter(sim, here)
            if burst is not None:
                print(f"  *** LEG-3 ENTER ({why}) area={da} swim={dsw} destx={dx}; nav={len(p)} cf8 + {len(burst)} cf1 ***", flush=True)
                json.dump({"leg3_nav_cf8": p, "leg3_entry_cf1": burst, "why": why,
                           "dest_area": da, "dest_swim": dsw}, open("/tmp/8_4_leg3.json", "w"))
                print("saved /tmp/8_4_leg3.json"); sim.close(); sys.exit(0)
            sim.restore(here)
        c = cell(r)
        if c not in archive:
            archive[c] = (sim.snapshot(), list(p))
    if iters % 3000 == 0:
        print(f"  iters={iters} arch={len(archive)} best_x={best_x} t={time.time()-t0:.0f}s", flush=True)
print(f"\nNO LEG-3 ENTRY in {BUDGET}s (iters={iters}, archive={len(archive)}, best_x={best_x}).")
print("pipe-3 is at x3648 row6; if best_x<3600 navigation is the blocker; else the descent timing is.")
sim.close()
