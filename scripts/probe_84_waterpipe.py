"""Phase 2 — targeted 8-4 water-pipe entry probe with the CORRECTED detector.

.lift_8_4.json reaches x3784 (past the decoded water pipe at page14 x3648-3679), so Mario
traverses that region ALIVE. Replay it, collect snapshots near the pipe, and from each try the
wide action set (down / down+A / right / sustained down) looking for a real_transition (x-jump to
a NEW cell, area/stage change) WITHOUT dying. Decides: enterable (solvable) vs lava/loop-gated.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/probe_84_waterpipe.py
"""
import json, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import (mario_level_x, real_transition, area_key, stage_key,
                       AREA_NUMBER, MARIO_Y_ON_SCREEN)
from mario.reward import is_death

WIDE = [["NOOP"], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"], ["A"],
        ["left"], ["left", "A"], ["down"], ["down", "A"], ["up"]]
lift = json.load(open(ROOT / "data/solutions/.lift_8_4.json"))["path"]

sim = MarioSim(8, 4, actions=WIDE); sim.reset(seed=0)
def bset(a): return frozenset(b for b in a if b not in ("B", "NOOP"))
from mario.env import ACTIONS as DA
wmap = {bset(a): i for i, a in enumerate(WIDE)}
TILE = 16
def vcell(ram):
    return (int(ram[AREA_NUMBER]), mario_level_x(ram)//TILE, int(ram[MARIO_Y_ON_SCREEN])//TILE)

# replay the lift path (default-action indices -> WIDE), collecting near-pipe snapshots + visited
visited = {vcell(sim.ram)}
seeds = []
for a in lift:
    info_b = dict(sim.last_info)
    i, d = sim.run_chunk(wmap.get(bset(DA[a]), 1), 8)
    visited.add(vcell(sim.ram))
    x = mario_level_x(sim.ram)
    if 3560 <= x <= 3720 and not d:
        seeds.append((sim.snapshot(), x, int(sim.ram[MARIO_Y_ON_SCREEN])))
    if d:
        break
print(f"reached x_max during lift replay; near-pipe seeds (x in 3560-3720): {len(seeds)}", flush=True)
if not seeds:
    print("lift path did not pass through the water-pipe band alive — cannot probe here."); sys.exit(1)

rng = random.Random(0); hits = 0; deaths = 0; tried = 0
for sid, (snap, sx, sy) in enumerate(seeds):
    ss = stage_key(sim.ram)
    for _ in range(400):
        sim.restore(snap); info_b = dict(sim.last_info); ram_b = sim.ram.copy(); ssk = stage_key(sim.ram)
        for f in range(20):
            a = rng.randrange(len(WIDE))
            info_a, d = sim.run_chunk(a, 4); tried += 1
            fired, reason = real_transition(info_b, ram_b, info_a, sim.ram, visited_cells=visited)
            if fired:
                hits += 1
                print(f"*** TRANSITION seed_x={sx} reason={reason} x={mario_level_x(sim.ram)} "
                      f"area={area_key(sim.ram)} stage={stage_key(sim.ram)} flag={info_a.get('flag_get')}",
                      flush=True)
                break
            if is_death(info_a, d):
                deaths += 1; break
            info_b, ram_b = dict(info_a), sim.ram.copy()
print(f"\nDONE: seeds={len(seeds)} rollouts_tried~{tried} transitions={hits} deaths={deaths}")
print("VERDICT:", "water pipe ENTERABLE (real transition found alive)" if hits
      else "no real transition at the water pipe (lava/geometry-gated — genuine, not detection)")
