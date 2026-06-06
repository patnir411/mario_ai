"""4-4 descent-finder: from the post-page-5 state (Y64), can Mario reach a GROUNDED Y176
($00CE>=170) state ANYWHERE in pages 5-9?  Uses BALANCED actions (incl. left + down) so a
backward/edge descent isn't missed (prior right-biased searches never found Y176).
Reward = get low ($00CE high). Reports the deepest grounded Y reached + a path to Y176 if found.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_44_descent.py [budget_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_success

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
X = lambda r: mario_level_x(r); Y = lambda r: int(r[0x00CE]); PAGE = lambda r: int(r[0x006D])
ST = lambda r: int(r[0x001D]); ENG = lambda r: int(r[0x000E])

best = json.load(open("/tmp/4_4_best.json"))["path"]
sim = MarioSim(4, 4); sim.reset(0); seed = None; pre = None
for ci, a in enumerate(best):
    if sim.run_chunk(a, 8)[1]: break
    if X(sim.ram) >= 1536 and Y(sim.ram) == 0x40 and ST(sim.ram) == 0:
        seed = sim.snapshot(); pre = best[:ci + 1]; break
if seed is None: print("no seed"); sys.exit(1)
print(f"seed page={PAGE(sim.ram)} x={X(sim.ram)} Y={Y(sim.ram)}", flush=True)

def cell(r): return (PAGE(r), X(r) // 8, Y(r) // 8, ST(r) == 0)
# BALANCED action set: left/right/down/jumps equally — find ANY descent
ACTS = [1, 1, 2, 2, 6, 6, 7, 7, 4, 0, 5, 3]
archive = {cell(sim.ram): (seed, [])}
rng = random.Random(0); t0 = time.time(); it = 0
deepest = Y(sim.ram); deepest_path = None; found176 = None
while time.time() - t0 < BUDGET and found176 is None:
    it += 1
    snap, p0 = rng.choice(list(archive.values()))
    sim.restore(snap); p = list(p0)
    for _ in range(rng.randint(3, 24)):
        a = rng.choice(ACTS)
        info, d = sim.run_chunk(a, 4); p.append(a)
        if is_success(info): break
        r = sim.ram
        if d or ENG(r) not in (4, 8): break
        if ST(r) == 0 and Y(r) > deepest:
            deepest = Y(r); deepest_path = list(p)
            print(f"  deeper grounded Y={deepest} page={PAGE(r)} x={X(r)} t={time.time()-t0:.0f}s", flush=True)
        if ST(r) == 0 and Y(r) >= 170:
            found176 = list(p); print(f"  *** Y176 REACHED grounded! Y={Y(r)} page={PAGE(r)} x={X(r)} ***", flush=True); break
        c = cell(r)
        if c not in archive: archive[c] = (sim.snapshot(), list(p))
    if it % 5000 == 0:
        print(f"  it={it} arch={len(archive)} deepest_Y={deepest} t={time.time()-t0:.0f}s", flush=True)
if found176:
    json.dump({"pre_cf8": pre, "descent_cf8": found176}, open("/tmp/4_4_descent.json", "w"))
    print(f"DESCENT FOUND -> /tmp/4_4_descent.json (deepest Y176 reached)")
else:
    print(f"\nNO Y176 grounded state reachable from post-page-5 (deepest grounded Y={deepest}). "
          f"Confirms the upper route cannot reach the lower path.")
sim.close()
