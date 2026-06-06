"""4-4 targeted: from the post-page-5-gate state (Y64 upper, passed), aggressively find the
LOWER (Y=0xb0=176) path and pass the page-9 gate (needs $00CE=176 grounded at col0), then
reach the axe. 4-4 has NO enterable pipes (Codex-confirmed) -> pure height navigation.
Detector: per-frame live-x (loopback = x drops >100 -> abandon). frame_advance for raw stepping.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_44_lower.py [budget_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_success

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 400.0
CACHE = ROOT / "data/solutions/4-4.json"
X = lambda r: mario_level_x(r); Y = lambda r: int(r[0x00CE]); PAGE = lambda r: int(r[0x006D])
ST = lambda r: int(r[0x001D]); ENG = lambda r: int(r[0x000E])
AMAP = None

# Seed: replay /tmp/4_4_best.json up to first page>=6 @ Y64 grounded (page-5 gate passed).
best = json.load(open("/tmp/4_4_best.json"))["path"]
sim = MarioSim(4, 4); sim.reset(0); AMAP = sim.env._action_map
seed = None; seed_chunks = None
for ci, a in enumerate(best):
    d = sim.run_chunk(a, 8)[1]
    if d: break
    if X(sim.ram) >= 1536 and Y(sim.ram) == 0x40 and ST(sim.ram) == 0:
        seed = sim.snapshot(); seed_chunks = best[:ci + 1]; break
if seed is None:
    print("no post-page-5 seed found"); sys.exit(1)
print(f"seed: x={X(sim.ram)} page={PAGE(sim.ram)} Y={Y(sim.ram)} (page-5 gate passed), prefix={len(seed_chunks)} cf8", flush=True)

def score(r):
    x = X(r); y = Y(r); s = x
    # strongly reward being GROUNDED on the lower (Y176) path — the descent is via going
    # LEFT off the upper platform's edge (page 5), then right along Y176 to the page-9 gate.
    if ST(r) == 0 and y >= 168:
        s += 1500
    return s

def cell(r):
    return (PAGE(r), X(r) // 6, Y(r) // 6, ST(r) == 0)

rng = random.Random(0); root = seed
archive = {cell(sim.ram): (root, [], score(sim.ram))}
t0 = time.time(); it = 0; solved = None; maxx = X(sim.ram); best_y_at_p8 = 999
ACTS = [1, 1, 2, 2, 6, 6, 7, 7, 4, 3, 0, 5]   # BALANCED (incl. left+down) — the descent is backward
while time.time() - t0 < BUDGET and solved is None:
    it += 1
    vals = list(archive.values())
    if rng.random() < 0.6:
        vals = sorted(vals, key=lambda v: v[2])[-80:]
    snap, pre, _ = rng.choice(vals)
    sim.restore(snap); p = list(pre)
    for _ in range(rng.randint(3, 22)):
        a = rng.choice(ACTS)
        xb = X(sim.ram)
        info, d = sim.run_chunk(a, 8)
        if is_success(info):
            solved = list(p) + [a]; print(f"  *** FLAG len={len(solved)} ***", flush=True); break
        xa = X(sim.ram)
        if d or ENG(sim.ram) not in (4, 8) or (xb - xa) > 100:   # death or loopback
            break
        p.append(a); r = sim.ram
        if PAGE(r) == 8 and ST(r) == 0 and Y(r) < best_y_at_p8:
            pass
        if PAGE(r) >= 9 and X(r) >= 2304:                       # passed the page-9 gate!
            solved = list(p); print(f"  *** PAGE-9 GATE PASSED x={X(r)} page={PAGE(r)} Y={Y(r)} len={len(p)} ***", flush=True); break
        if X(r) > maxx:
            maxx = X(r); print(f"  new maxx={maxx} page={PAGE(r)} Y={Y(r)} grounded={ST(r)==0} t={time.time()-t0:.0f}s", flush=True)
        sc = score(r); c = cell(r)
        if c not in archive or sc > archive[c][2]:
            archive[c] = (sim.snapshot(), list(p), sc)
    if it % 4000 == 0:
        print(f"  it={it} arch={len(archive)} maxx={maxx} t={time.time()-t0:.0f}s", flush=True)

if solved is None:
    print(f"\nNO page-9 pass in {BUDGET}s (it={it}, maxx={maxx}).")
    sys.exit(1)
full = list(seed_chunks) + list(solved)
# continue to axe if only gate passed (not flag yet)
sim.reset(0)
beat = False
for a in full:
    info, d = sim.run_chunk(a, 8)
    if is_success(info): beat = True; break
    if d: break
if not beat:
    print("page-9 passed but not flag; extending to axe...", flush=True)
    from mario.search import search_from_state
    sim.reset(0)
    for a in full: sim.run_chunk(a, 8)
    snap = sim.snapshot()
    tail = search_from_state(sim, snap, world=4, stage=4, beam_width=48, depth=200, chunk_frames=8, max_seconds=120)
    full = full + tail
    v = MarioSim(4, 4); v.reset(0); beat = False
    for a in full:
        info, d = v.run_chunk(a, 8)
        if is_success(info): beat = True; break
        if d: break
    v.close()
print(f"replay-from-reset beat={beat} len={len(full)}")
if beat:
    CACHE.write_text(json.dumps({"path": full, "solved": True, "by": "castle_lowerpath",
                                 "chunk_frames": 8, "world": 4, "stage": 4}))
    print(f"*** 4-4 SOLVED *** cached {CACHE}")
sim.close()
