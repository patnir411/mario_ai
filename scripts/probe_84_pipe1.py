"""DECISIVE: can Mario enter ANY pipe in 8-4 room 1?  Seed from the approach to the
first piranha-pipe (~x1296), Go-Explore with JUMPS (land on the pipe-top), and at every
grounded state hold DOWN, testing the CORRECT real-entry signal
(AREA_NUMBER flip / $06DE!=0 / $000E in {2,3}).  Explore a wide x window so it covers
several pipes, not just the first.
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x, AREA_NUMBER, MARIO_Y_ON_SCREEN, PLAYER_FLOAT_STATE
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
BASE = 3

def entry(r):
    return int(r[AREA_NUMBER]) != BASE or int(r[0x06DE]) != 0 or int(r[0x000E]) in (2, 3)

def down_test(sim, snap):
    sim.restore(snap)
    for f in range(44):
        _, d = sim.run_chunk(7, 1)
        if entry(sim.ram):
            return True, int(sim.ram[AREA_NUMBER]), mario_level_x(sim.ram)
        if d: break
    return False, BASE, 0

# Seed: replay prefix, grab the approach snapshot just before the first pipe (x~1230)
path = json.load(open(ROOT / "data/solutions/.prefix_8_4_3000.json"))["prefix"]
sim = MarioSim(8, 4); sim.reset(0)
seed = None
for a in path:
    _, d = sim.run_chunk(a, 8)
    if d: break
    if 1210 <= mario_level_x(sim.ram) <= 1240 and int(sim.ram[PLAYER_FLOAT_STATE]) == 0:
        seed = sim.snapshot()
if seed is None:
    print("no approach seed near x1230"); sys.exit(1)

def cell(r):
    return (mario_level_x(r) // 4, int(r[MARIO_Y_ON_SCREEN]) // 8, int(r[PLAYER_FLOAT_STATE]) == 0)

sim.restore(seed)
archive = {cell(sim.ram): seed}
probed = set(); rng = random.Random(0); t0 = time.time(); iters = 0; best_x = 0; hits = []
# action ids: 1=right, 2=right+A(jump), 3=right+B, 4=right+A+B(run-jump), 5=A, 0=noop, 6=left
JUMP_BIASED = [2, 2, 4, 4, 1, 1, 5, 3, 0, 6]
while time.time() - t0 < BUDGET:
    iters += 1
    snap = rng.choice(list(archive.values()))
    sim.restore(snap)
    for _ in range(rng.randint(2, 12)):
        a = rng.choice(JUMP_BIASED)
        _, d = sim.run_chunk(a, rng.choice([1, 2, 4]))
        if d or is_death(sim.last_info, d): break
        x = mario_level_x(sim.ram); best_x = max(best_x, x)
        if x > 1600:           # past room-1 first cluster; keep window tight to densify
            break
        c = cell(sim.ram)
        if c not in archive:
            archive[c] = sim.snapshot()
            if c[2] and c not in probed:
                probed.add(c)
                ent, na, ex = down_test(sim, archive[c])
                if ent:
                    hits.append((c[0] * 4, c[1] * 8, na))
                    print(f"  *** REAL ENTRY @ x~{c[0]*4} y~{c[1]*8} -> area {na} ***", flush=True)
                break
    if iters % 5000 == 0:
        ys = sorted({c[1] * 8 for c in probed})
        print(f"  iters={iters} arch={len(archive)} probed={len(probed)} best_x={best_x} "
              f"probed_ys={ys} hits={len(hits)} t={time.time()-t0:.0f}s", flush=True)
print(f"\nDONE iters={iters} archive={len(archive)} probed={len(probed)} best_x={best_x}")
if hits:
    print(f"REAL pipe entries: {sorted(set(hits))}")
    json.dump({"hits": hits}, open("/tmp/8_4_pipe1.json", "w"))
else:
    print("NO real pipe entry from any grounded state in room-1 pipe cluster (jump+DOWN, correct test).")
    print(f"grounded y-levels probed: {sorted({c[1]*8 for c in probed})}")
sim.close()
