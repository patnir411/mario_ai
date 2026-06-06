"""Leg-3 (8-4 page-14 down-pipe at level-x 3648) entry search — Go-Explore from the verified
prefix+lift approach. The entry is a DOWN-held descent onto the pipe-top (10/11 at row6), like
leg 1. Success = a strict forward transition (info_ram_mismatch / pipe_live to a NEW page = water
$02). The hardened detector no longer false-fires on the page-loopback (no info/RAM mismatch).

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_84_leg3.py [budget_s]
"""
import json, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim, ACTIONS
from mario.ram import (mario_level_x, real_transition_strict, area_key, stage_key, pipe_entering,
                       signed)
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
prefix = json.load(open(ROOT / "data/solutions/.prefix_8_4_lift.json"))["prefix"]
lift = json.load(open(ROOT / "data/solutions/.lift_8_4.json"))["path"]
path = prefix + lift
# wide action set: default + right+down / left + down combos for descent control
WIDE = ACTIONS + [["right", "down"], ["left", "down"], ["down", "A"]]
def w(default_idx):  # map a default action index into WIDE (same prefix indices)
    return default_idx

sim = MarioSim(8, 4, actions=WIDE); sim.reset(0)
# replay prefix+lift up to the grounded approach just BEFORE the pipe-3 jump (collect seeds)
seeds = []
for i, a in enumerate(path):
    if sim.run_chunk(a, 8)[1]:
        break
    x = mario_level_x(sim.ram)
    if 3500 <= x <= 3620:
        seeds.append(sim.snapshot())
if not seeds:
    print("no approach seeds before pipe 3"); sys.exit(1)
print(f"approach seeds: {len(seeds)}; pipe-3 at x3648", flush=True)

def cell(r):
    vx = signed(int(r[0x0057]))
    return (mario_level_x(r) // 4, int(r[0x00CE]) // 4, int(r[0x001D]),
            1 if vx > 0 else (-1 if vx < 0 else 0))

import time
archive = {}
for sd in seeds:
    sim.restore(sd); archive[cell(sim.ram)] = (sd, [])
start_aks = set()
sim.restore(seeds[0]); ss = stage_key(sim.ram)
rng = random.Random(0); t0 = time.time(); iters = 0
while time.time() - t0 < BUDGET:
    iters += 1
    base_snap, base_path = rng.choice(list(archive.values()))
    sim.restore(base_snap)
    p = list(base_path)
    for _ in range(rng.randint(3, 18)):
        # bias toward right/jump/down (the descent-entry)
        a = rng.choice([1, 2, 2, 1, 7, 7, 0, 9, 9, 10, 11])
        a = a if a < len(WIDE) else 1
        ib = dict(sim.last_info); rb = sim.ram.copy()
        info, dn = sim.run_chunk(a, 1); r = sim.ram; p.append(a)
        f, reason = real_transition_strict(ib, rb, info, r)
        if (f and reason in ("info_ram_mismatch", "pipe_live", "stage", "stage_ram", "flag")) \
                or pipe_entering(r):
            full = path[:len(prefix) + len(lift)] + p  # NOTE: p is from a seed snapshot; see below
            print(f"*** LEG-3 ENTERED *** reason={reason} x={mario_level_x(r)} area={area_key(r)} "
                  f"stage={stage_key(r)} info_x={info.get('x_pos')} 0E={r[0x000E]} 06DE={r[0x06DE]}",
                  flush=True)
            json.dump({"note": "p is the burst from an approach snapshot; reconstruct via seed",
                       "burst": p, "reason": reason, "dest_x": mario_level_x(r)},
                      open("/tmp/8_4_leg3.json", "w"))
            print("saved /tmp/8_4_leg3.json"); sys.exit(0)
        if is_death(info, dn):
            break
        c = cell(r)
        if c not in archive:
            archive[c] = (sim.snapshot(), list(p))
    if iters % 4000 == 0:
        bx = max(k[0] * 4 for k in archive)
        print(f"  iters={iters} archive={len(archive)} best_x~{bx} t={time.time()-t0:.0f}s", flush=True)
print(f"NO LEG-3 ENTRY in {BUDGET}s (archive={len(archive)}). The descent entry was not stumbled; "
      f"needs a more targeted/longer search or a wired coverage_search.", flush=True)
