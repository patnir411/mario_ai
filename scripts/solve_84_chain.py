"""8-4 leg-by-leg down-pipe chain with the CORRECT entry detector.

Each down-pipe bypasses a ProcLoopCommand gate (pipe1 skips page6 ->page7,
pipe2 skips page11 ->page12, pipe3 skips page16 ->water $02). Entry = land on the
enterable pipe-top ($11/$10) with DOWN held; detected (through the gym wrapper) as
info.x_pos vs live-RAM-x mismatch with a FORWARD page advance.

Strategy per leg: Go-Explore forward from the leg-start; at every grounded state run a
DESCENT-DOWN micro-search (a few right/jump frames then hold DOWN); success = mismatch
firing with page advancing to >= the post-pipe target. Append the winning burst, repeat.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84_chain.py [per_leg_s]
"""
import sys, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_death

PER_LEG = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
CF = 8
PAGE = lambda r: int(r[0x006D])
Y = lambda r: int(r[0x00CE])
GROUNDED = lambda r: int(r[0x001D]) == 0
SWIM = lambda r: int(r[0x0704]) != 0

def replay(sim, path):
    sim.reset(0)
    for a in path:
        if sim.run_chunk(a, CF)[1]: return True
    return False

def mismatch(info_before, sim):
    ix = info_before.get("x_pos")
    return ix is not None and abs(int(ix) - mario_level_x(sim.ram)) > 100

# descent micro-sequences (frame-level, raw chunk=1): approach then sustained DOWN
def descent_seqs():
    seqs = []
    for nr in range(0, 14):          # walk/run right frames
        for ra in (1, 3):            # right, right+B(run)
            seqs.append([ra] * nr + [7] * 20)        # then hold DOWN 20f
            seqs.append([ra] * nr + [2] + [7] * 20)  # hop then DOWN (land-on-top)
    return seqs

DSEQS = descent_seqs()

def try_enter(sim, snap, target_page):
    """From snap, try each descent micro-seq; success = a per-frame live-x teleport (>100px,
    robust to stale last_info after restore) that lands on page >= target_page.
    Returns winning burst (list of cf1 actions) or None."""
    for seq in DSEQS:
        sim.restore(snap)
        burst = []
        for a in seq:
            xb = mario_level_x(sim.ram)
            _, d = sim.run_chunk(a, 1)
            xa = mario_level_x(sim.ram)
            burst.append(a)
            if abs(xa - xb) > 100 and PAGE(sim.ram) >= target_page:
                return burst, PAGE(sim.ram)
            if d: break
    return None, None

def go_leg(sim, start_path, start_cf_is_8, target_page, near_x, budget):
    """Explore forward from start_path; return (winning_full_burst_cf1, dest_page) or None.
    Builds cf1 bursts on top of the (cf8) start_path replay."""
    # replay start to leg-start, archive grounded states near the pipe
    sim.reset(0)
    for a in start_path:
        if sim.run_chunk(a, CF)[1]: break
    base = sim.snapshot()
    def cell(r): return (PAGE(r), mario_level_x(r) // 8, Y(r) // 16, GROUNDED(r))
    archive = {cell(sim.ram): (base, [])}
    rng = random.Random(0); t0 = time.time(); iters = 0; best_page = PAGE(sim.ram)
    while time.time() - t0 < budget:
        iters += 1
        snap, pre = rng.choice(list(archive.values()))
        sim.restore(snap); p = list(pre)
        for _ in range(rng.randint(2, 14)):
            a = rng.choice([1, 1, 3, 2, 0, 5, 6])
            ib = dict(sim.last_info); _, d = sim.run_chunk(a, CF); p.append(a)
            if is_death(sim.last_info, d) or d: break
            r = sim.ram
            # near the pipe -> attempt entry
            if abs(mario_level_x(r) - near_x) <= 96 and GROUNDED(r):
                here = sim.snapshot()
                burst, dp = try_enter(sim, here, target_page)
                if burst is not None:
                    print(f"  *** LEG ENTER: page->{dp} after {len(p)} cf8 + {len(burst)} cf1; x_at_attempt={mario_level_x(r)} ***", flush=True)
                    return ("cf8", p, "cf1", burst, dp)
                sim.restore(here)          # try_enter left sim elsewhere; resume explore
            c = cell(r)
            if c not in archive:
                archive[c] = (sim.snapshot(), list(p))
                if PAGE(r) > best_page:
                    best_page = PAGE(r); print(f"    leg explore new page={best_page} x={mario_level_x(r)} t={time.time()-t0:.0f}s", flush=True)
        if iters % 4000 == 0:
            print(f"    iters={iters} arch={len(archive)} best_page={best_page} t={time.time()-t0:.0f}s", flush=True)
    return None

sim = MarioSim(8, 4)
# Leg 1 is the confirmed canned entry (ckpt[:75], cf8)
full_cf8 = json.load(open("/tmp/8_4_ckpt.json"))["path"][:75]
replay(sim, full_cf8); print(f"after pipe-1: page={PAGE(sim.ram)} x={mario_level_x(sim.ram)} Y={Y(sim.ram)}", flush=True)

# Leg 2: enter pipe-2 (~x2432, page 9) -> target page >= 12
print("\n=== LEG 2 (pipe-2 ~x2432 -> page>=12) ===", flush=True)
res = go_leg(sim, full_cf8, True, 12, 2432, PER_LEG)
if res is None:
    print("LEG 2 FAILED: no pipe-2 entry found in budget. (pipe-2 location/target may differ.)")
    sys.exit(1)
_, leg2_cf8, _, leg2_cf1, dp = res
json.dump({"leg1_cf8": full_cf8, "leg2_cf8": leg2_cf8, "leg2_cf1": leg2_cf1, "dest_page": dp},
          open("/tmp/8_4_chain.json", "w"))
print(f"LEG 2 OK -> page {dp}; saved /tmp/8_4_chain.json")
sim.close()
