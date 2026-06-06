"""7-4 (CastleArea5) — the 6-gate MULTI-loop. World-7 ProcLoopCommand requires hitting 3
consecutive checkpoints at the correct Player_Y grounded; MultiLoopCorrectCntr ($06d9) /
MultiLoopPassCntr ($06da) track it, resetting + looping back on a miss. KEY: put the counters
in the cell key (so 2-of-3-correct states stay alive) and reward MultiLoopCorrectCntr heavily
so the search ACCUMULATES correct checkpoints. Balanced actions (asc/descent). flag_get = win.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_74_seg2.py [budget_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_success

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
CACHE = ROOT / "data/solutions/7-4.json"
X = lambda r: mario_level_x(r); Y = lambda r: int(r[0x00CE]); PAGE = lambda r: int(r[0x006D])
ST = lambda r: int(r[0x001D]); ENG = lambda r: int(r[0x000E])
MLC = lambda r: int(r[0x06D9]); MLP = lambda r: int(r[0x06DA])
GATES = [(4, 0xb0), (5, 0x80), (6, 0x40), (8, 0x40), (9, 0x80), (0xa, 0x40)]

# seed: first grounded state at page>=7 (section 1 cleared) on the best path
best = json.load(open("/tmp/7_4_best.json"))["path"]
sim = MarioSim(7, 4); sim.reset(0); seed = None; pre = None
for ci, a in enumerate(best):
    if sim.run_chunk(a, 8)[1]: break
    if PAGE(sim.ram) >= 7 and ST(sim.ram) == 0:
        seed = sim.snapshot(); pre = best[:ci + 1]; break
if seed is None: print("no page>=7 seed"); sys.exit(1)
sim.restore(seed)
print(f"seed page={PAGE(sim.ram)} x={X(sim.ram)} Y={Y(sim.ram)} mlc={MLC(sim.ram)} mlp={MLP(sim.ram)} prefix={len(pre)}", flush=True)

def score(r):
    s = X(r) + 5000 * MLC(r)              # accumulate correct checkpoints dominates
    for gp, gy in GATES:
        if gp - 1 <= PAGE(r) <= gp:
            s += max(0, 100 - abs(Y(r) - gy))
            if ST(r) == 0 and abs(Y(r) - gy) <= 5: s += 300
    return s

def cell(r):
    return (PAGE(r), X(r) // 6, Y(r) // 6, ST(r) == 0, MLC(r), MLP(r))

ACTS = [1, 1, 2, 2, 6, 6, 7, 7, 4, 3, 0, 5]
archive = {cell(sim.ram): (seed, [], score(sim.ram))}
rng = random.Random(0); t0 = time.time(); it = 0; solved = None; bestmlc = MLC(sim.ram); maxx = X(sim.ram); best_path = []
while time.time() - t0 < BUDGET and solved is None:
    it += 1
    vals = list(archive.values())
    if rng.random() < 0.6: vals = sorted(vals, key=lambda v: v[2])[-100:]
    snap, p0, _ = rng.choice(vals)
    sim.restore(snap); p = list(p0)
    for _ in range(rng.randint(3, 26)):
        a = rng.choice(ACTS); xb = X(sim.ram)
        info, d = sim.run_chunk(a, 8)
        if is_success(info): solved = list(p) + [a]; print(f"  *** FLAG len={len(solved)} ***", flush=True); break
        if d or ENG(sim.ram) not in (4, 8) or (xb - X(sim.ram)) > 120: break
        p.append(a); r = sim.ram
        if MLC(r) > bestmlc or X(r) > maxx:
            if MLC(r) > bestmlc: bestmlc = MLC(r)
            if X(r) > maxx: maxx = X(r); best_path = list(p)
            print(f"  mlc={MLC(r)} mlp={MLP(r)} page={PAGE(r)} x={X(r)} Y={Y(r)} gnd={ST(r)==0} t={time.time()-t0:.0f}s", flush=True)
        sc = score(r); c = cell(r)
        if c not in archive or sc > archive[c][2]: archive[c] = (sim.snapshot(), list(p), sc)
    if it % 5000 == 0:
        print(f"  it={it} arch={len(archive)} bestmlc={bestmlc} maxx={maxx} t={time.time()-t0:.0f}s", flush=True)

if solved is None:
    print(f"\nNO FLAG in {BUDGET}s (bestmlc={bestmlc}, maxx={maxx}).")
    if best_path: json.dump({"path": list(pre) + best_path, "chunk_frames": 8}, open("/tmp/7_4_best.json", "w"))
    sys.exit(1)
full = list(pre) + list(solved)
v = MarioSim(7, 4); v.reset(0); beat = False
for a in full:
    info, d = v.run_chunk(a, 8)
    if is_success(info): beat = True; break
    if d: break
v.close()
print(f"replay-from-reset beat={beat} len={len(full)}")
if beat:
    CACHE.write_text(json.dumps({"path": full, "solved": True, "by": "castle_multiloop",
                                 "chunk_frames": 8, "world": 7, "stage": 4}))
    print(f"*** 7-4 SOLVED *** cached {CACHE}")
sim.close()
