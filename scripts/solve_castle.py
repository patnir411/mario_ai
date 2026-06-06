"""General warpless-CASTLE solver (4-4 / 7-4) via page-aware Go-Explore with the CORRECT
detector. Castles loop (ProcLoopCommand) unless Mario passes each gate page at the right
Player_Y grounded; the search rewards forward PAGE progress (loopbacks read as regress) and
tries every height at the gate columns. Down-pipes (if any) handled by the wide action set +
per-frame live-x teleport detection. Success = flag_get (axe).

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_castle.py W S [budget_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim, ACTIONS
from mario.ram import mario_level_x
from mario.reward import is_death, is_success

W = int(sys.argv[1]); S = int(sys.argv[2])
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
CACHE = ROOT / f"data/solutions/{W}-{S}.json"
PAGE = lambda r: int(r[0x006D]); Y = lambda r: int(r[0x00CE])
GROUNDED = lambda r: int(r[0x001D]) == 0; AREA = lambda r: int(r[0x0760])
# ProcLoopCommand gates per world (page -> required Player_Y grounded), from SMBDIS.ASM:7787.
GATES = {4: [(5, 0x40), (9, 0xb0)],
         7: [(4, 0xb0), (5, 0x80), (6, 0x40), (8, 0x40), (9, 0x80), (0xa, 0x40)],
         8: [(6, 0xf0), (0xb, 0xf0), (0x10, 0xf0)]}
WGATES = GATES.get(W, [])
def GPROG(r):
    base = PAGE(r) * 256 + mario_level_x(r) % 256
    # gate-aware: at a gate page, reward being near the required Y grounded (steers the frontier
    # toward the height that PASSES the loop gate instead of max-x dead-ends).
    for gp, gy in WGATES:
        if gp - 1 <= PAGE(r) <= gp:        # approaching OR at the gate -> get to the right height
            base += max(0, 80 - abs(Y(r) - gy))
    return base

sim = MarioSim(W, S); sim.reset(0)
root = sim.snapshot(); r = sim.ram
print(f"{W}-{S} start: page={PAGE(r)} x={mario_level_x(r)} Y={Y(r)} area={AREA(r)}", flush=True)

def valid(r):
    # reject garbage/transition frames (page byte 255, x underflow) that pollute the frontier
    return PAGE(r) <= 25 and 0 <= mario_level_x(r) <= 6500 and int(r[0x000E]) in (8, 4)

def cell(r):
    return (PAGE(r), mario_level_x(r) // 8, Y(r) // 12, GROUNDED(r))

# archive cell -> (snap, path, gprog)
archive = {cell(sim.ram): (root, [], GPROG(sim.ram))}
rng = random.Random(0); t0 = time.time(); iters = 0
best_prog = GPROG(sim.ram); best_page = PAGE(sim.ram); solved_path = None
best_path = []
# BALANCED action set incl. left+down — descents between alternating-height gates often
# require going LEFT off a platform edge (the 4-4 unlock); right-bias misses them.
ACTS = [1, 1, 2, 2, 6, 6, 7, 7, 4, 3, 0, 5]
while time.time() - t0 < BUDGET and solved_path is None:
    iters += 1
    # bias selection toward the highest-progress cells (push the frontier)
    if rng.random() < 0.55:
        # pick among the top-progress cells
        items = sorted(archive.values(), key=lambda v: v[2])[-40:]
        snap, pre, _ = rng.choice(items)
    else:
        snap, pre, _ = rng.choice(list(archive.values()))
    sim.restore(snap); p = list(pre)
    for _ in range(rng.randint(2, 16)):
        a = rng.choice(ACTS)
        xb = mario_level_x(sim.ram)
        info, d = sim.run_chunk(a, 8); p.append(a)
        if is_success(info):
            solved_path = list(p); print(f"  *** {W}-{S} FLAG x={mario_level_x(sim.ram)} len={len(p)} ***", flush=True)
            break
        if is_death(info, d) or d: break
        r = sim.ram
        if not valid(r): break          # garbage/transition frame -> abandon this rollout
        pr = GPROG(r)
        if PAGE(r) > best_page:
            best_page = PAGE(r); print(f"  new best page={best_page} x={mario_level_x(r)} Y={Y(r)} t={time.time()-t0:.0f}s", flush=True)
        if pr > best_prog:
            best_prog = pr; best_path = list(p)
        c = cell(r)
        if c not in archive or pr > archive[c][2]:
            archive[c] = (sim.snapshot(), list(p), pr)
    if iters % 4000 == 0:
        print(f"  iters={iters} arch={len(archive)} best_page={best_page} best_prog={best_prog} t={time.time()-t0:.0f}s", flush=True)

if solved_path is None:
    print(f"\nNO FLAG in {BUDGET}s (iters={iters}, archive={len(archive)}, best_page={best_page}).")
    json.dump({"path": best_path, "chunk_frames": 8}, open(f"/tmp/{W}_{S}_best.json", "w"))
    print(f"saved best-progress path (len={len(best_path)}, prog={best_prog}) to /tmp/{W}_{S}_best.json")
    sys.exit(1)
# verify replay-from-reset
v = MarioSim(W, S); v.reset(0); beat = False
for a in solved_path:
    info, d = v.run_chunk(a, 8)
    if is_success(info): beat = True; break
    if d: break
v.close()
print(f"replay-from-reset beat={beat} len={len(solved_path)}")
if beat:
    CACHE.write_text(json.dumps({"path": solved_path, "solved": True, "by": "castle_goexplore",
                                 "chunk_frames": 8, "world": W, "stage": S}))
    print(f"*** {W}-{S} SOLVED *** cached {CACHE}")
sim.close()
