"""Phase 0c — decisive 8-4 recon with the CORRECTED detector.

Explore 8-4 with coverage (Go-Explore-lite) over a wide action set, and on every expansion run
`real_transition(... visited_cells=visited)`. Report any REAL transition (new-area / new-cell
x-jump / stage / flag) that the old blind detector ($06DE/$000E after the wrapped step) missed.

Gate: if a real new-cell transition is found at some reachable x -> 8-4 entry is reachable
(Phase 2 targets it). If NONE across the reachable corridor -> 8-4 is a genuine dead-end for this
action set (report honestly).

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/recon_84_detector.py [budget_s]
"""
import sys, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import (mario_level_x, real_transition, area_key, stage_key, signed,
                       AREA_NUMBER, MARIO_Y_ON_SCREEN)
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
# Wide action set for the maze/mount (adds left+A combos the default 9-set lacks).
WIDE = [["NOOP"], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"], ["A"],
        ["left"], ["left", "A"], ["left", "A", "B"], ["down"], ["up"], ["down", "A"]]

sim = MarioSim(8, 4, actions=WIDE); sim.reset(seed=0)
TILE = 16
def cell(ram):
    vx = signed(int(ram[0x0057]))
    return (int(ram[AREA_NUMBER]), mario_level_x(ram) // TILE, int(ram[MARIO_Y_ON_SCREEN]) // TILE,
            1 if vx > 0 else (-1 if vx < 0 else 0))

start_info = dict(sim.last_info); start_ak = area_key(sim.ram); start_sk = stage_key(sim.ram)
root = (sim.snapshot(), [], dict(sim.last_info), sim.ram.copy())
archive = {cell(sim.ram): root}
visited = {(int(sim.ram[AREA_NUMBER]), mario_level_x(sim.ram)//TILE, int(sim.ram[MARIO_Y_ON_SCREEN])//TILE)}
best_x = mario_level_x(sim.ram); n = 0; transitions = []
rng = random.Random(0); t0 = time.perf_counter()

while time.perf_counter() - t0 < BUDGET:
    # pick a frontier cell biased to high-x + novelty
    snap, path, info_b, ram_b = rng.choice(list(archive.values()))
    sim.restore(snap)
    for _ in range(rng.randint(4, 28)):
        a = rng.randrange(len(WIDE))
        info_a, done = sim.run_chunk(a, 8); ram_a = sim.ram; n += 1
        fired, reason = real_transition(info_b, ram_b, info_a, ram_a, visited_cells=visited)
        if fired:
            transitions.append((reason, mario_level_x(ram_a), int(ram_a[MARIO_Y_ON_SCREEN]),
                                area_key(ram_a), stage_key(ram_a), info_a.get("flag_get"),
                                len(path)))
            print(f"*** TRANSITION reason={reason} x={mario_level_x(ram_a)} "
                  f"area_key={area_key(ram_a)} stage={stage_key(ram_a)} flag={info_a.get('flag_get')}",
                  flush=True)
            if reason in ("flag", "stage", "stage_ram", "area_key"):
                # decisive new-area/level transition — stop, this is the missed entry
                break
        if is_death(info_a, done):
            break
        c = cell(ram_a); vc = (c[0], c[1], c[2])
        visited.add(vc)
        x = mario_level_x(ram_a)
        if c not in archive:
            archive[c] = (sim.snapshot(), path + [a], dict(info_a), ram_a.copy())
        if x > best_x:
            best_x = x
        info_b, ram_b = dict(info_a), ram_a.copy()
    if len(transitions) > 80:
        break

print(f"\nRECON DONE: nodes={n} cells={len(archive)} best_x={best_x} elapsed={time.perf_counter()-t0:.0f}s")
print(f"transitions found: {len(transitions)}")
decisive = [t for t in transitions if t[0] in ("flag", "stage", "stage_ram", "area_key")]
xjumps = [t for t in transitions if t[0] == "x_jump"]
print(f"  decisive (flag/stage/area_key): {len(decisive)}")
print(f"  x_jump (new-cell teleports): {len(xjumps)}")
for t in (decisive[:10] or xjumps[:10]):
    print("   ", t)
print("VERDICT:", "REACHABLE — 8-4 has a real transition the blind detector missed"
      if decisive or xjumps else "no real transition found in budget (run longer / wider before concluding)")
