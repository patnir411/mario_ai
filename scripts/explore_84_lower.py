"""From the CONFIRMED post-pipe-1 state (ckpt[:75] -> page 7 lower path), Go-Explore
rightward and report how far 8-4 chains: max page, whether we hit the water area
(swim flag $0704) / a stage change / flag, and whether the page-11/16 loop gates
(need Y=240 grounded) send us back. Uses the CORRECT entry signal (info.x_pos vs live RAM).
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
PREFIX = json.load(open("/tmp/8_4_ckpt.json"))["path"][:75]

sim = MarioSim(8, 4); sim.reset(0)
for a in PREFIX:
    if sim.run_chunk(a, 8)[1]: break
root = sim.snapshot()
r = sim.ram
print(f"seed: page={r[0x006D]} x={mario_level_x(r)} Y={r[0x00CE]} grounded={r[0x001D]==0}", flush=True)

def cell(r):
    return (int(r[0x006D]), mario_level_x(r) // 8, int(r[0x00CE]) // 16, int(r[0x001D]) == 0)

archive = {cell(sim.ram): (root, list(PREFIX))}
rng = random.Random(0); t0 = time.time(); iters = 0
best_page = int(r[0x006D]); milestones = []
water_seen = flag_seen = False
best_path = None
while time.time() - t0 < BUDGET:
    iters += 1
    c0 = rng.choice(list(archive.values()))
    sim.restore(c0[0]); p = list(c0[1])
    for _ in range(rng.randint(2, 16)):
        a = rng.choice([1, 1, 2, 2, 3, 4, 7, 0, 5, 6])   # right-biased + down(7) for pipes
        ib = dict(sim.last_info); _, d = sim.run_chunk(a, 8); p.append(a)
        rr = sim.ram
        if is_death(sim.last_info, d) or d:
            if sim.last_info.get("flag_get"):
                flag_seen = True; best_path = list(p)
                print(f"  *** FLAG_GET *** len={len(p)}", flush=True)
            break
        page = int(rr[0x006D])
        if int(rr[0x0704]) and not water_seen:
            water_seen = True; milestones.append(("water", page, mario_level_x(rr))); best_path = list(p)
            print(f"  *** WATER reached page={page} x={mario_level_x(rr)} len={len(p)} ***", flush=True)
        if page > best_page:
            best_page = page; milestones.append(("page", page, mario_level_x(rr)))
            if best_path is None or len(p) < 400: best_path = list(p)
            print(f"  new best page={page} x={mario_level_x(rr)} Y={rr[0x00CE]} t={time.time()-t0:.0f}s", flush=True)
        c = cell(rr)
        if c not in archive:
            archive[c] = (sim.snapshot(), list(p))
    if iters % 4000 == 0:
        print(f"  iters={iters} arch={len(archive)} best_page={best_page} water={water_seen} t={time.time()-t0:.0f}s", flush=True)
print(f"\nDONE iters={iters} archive={len(archive)} best_page={best_page} water={water_seen} flag={flag_seen}")
print(f"milestones: {milestones[:20]}")
if best_path:
    json.dump({"path": best_path, "by": "explore_lower"}, open("/tmp/8_4_lower.json", "w"))
    print(f"saved best_path len={len(best_path)} to /tmp/8_4_lower.json")
sim.close()
