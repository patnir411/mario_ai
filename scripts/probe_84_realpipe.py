"""DECISIVE 8-4 test: is there ANY reachable grounded state in area 3 from which a
DOWN-hold produces a REAL pipe entry?  A real entry = AREA_NUMBER ($0760) changes
OR $06DE (ChangeAreaTimer) != 0 OR $000E (GameEngineSubroutine) in {2,3}.

This explicitly does NOT trust the info/RAM x-mismatch (that ALSO fires on the
in-area ProcLoop page-warp, which fooled the earlier "leg 1 -> page 7" claim).

Go-Explore over area 3 (full action set), archive grounded cells, DOWN-probe each.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/probe_84_realpipe.py [budget_s]
"""
import sys, time, random, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim, ACTIONS
from mario.ram import mario_level_x, AREA_NUMBER, MARIO_Y_ON_SCREEN, PLAYER_FLOAT_STATE, signed
from mario.reward import is_death

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0

def real_entry(ram, base_area):
    return (int(ram[AREA_NUMBER]) != base_area or int(ram[0x06DE]) != 0
            or int(ram[0x000E]) in (2, 3))

def down_probe(sim, snap, base_area):
    """From snap, nudge L/R to center then hold DOWN; report (entered, off, new_area)."""
    for off, act in ((0, None), (4, 1), (8, 1), (4, 6), (8, 6), (12, 1), (12, 6)):
        sim.restore(snap)
        ok = True
        if act is not None:
            for _ in range(off):
                _, d = sim.run_chunk(act, 1)
                if real_entry(sim.ram, base_area):
                    return True, off, int(sim.ram[AREA_NUMBER])
                if d: ok = False; break
        if not ok: continue
        for _ in range(40):
            _, d = sim.run_chunk(7, 1)            # hold DOWN
            if real_entry(sim.ram, base_area):
                return True, off, int(sim.ram[AREA_NUMBER])
            if d: break
    return False, 0, base_area

def cell(ram):
    return (int(ram[0x006D]), mario_level_x(ram) // 8, int(ram[PLAYER_FLOAT_STATE]) == 0)

sim = MarioSim(8, 4); sim.reset(0)
base_area = int(sim.ram[AREA_NUMBER])
print(f"base_area={base_area}; exploring + DOWN-probing every grounded cell for a REAL entry", flush=True)
root = sim.snapshot()
archive = {cell(sim.ram): (root, 0)}        # cell -> (snap, level_x)
probed = set()
rng = random.Random(0); t0 = time.time(); iters = 0; best_x = 0; hits = []
while time.time() - t0 < BUDGET:
    iters += 1
    snap, _ = rng.choice(list(archive.values()))
    sim.restore(snap)
    for _ in range(rng.randint(2, 14)):
        a = rng.choice([1, 1, 2, 2, 3, 0, 4, 6])     # right-biased, jumps
        _, d = sim.run_chunk(a, rng.choice([2, 4, 8]))
        if d or is_death(sim.last_info, d): break
        c = cell(sim.ram)
        if c not in archive:
            archive[c] = (sim.snapshot(), mario_level_x(sim.ram))
            best_x = max(best_x, mario_level_x(sim.ram))
            if c[2] and c not in probed:             # grounded -> DOWN-probe once
                probed.add(c)
                ent, off, na = down_probe(sim, archive[c][0], base_area)
                if ent:
                    hits.append((c[1] * 8, off, na))
                    print(f"  *** REAL ENTRY @ x~{c[1]*8} off={off} area {base_area}->{na} ***", flush=True)
                break        # down_probe left sim mid-restore; start a fresh rollout
    if iters % 3000 == 0:
        print(f"  iters={iters} archive={len(archive)} grounded_probed={len(probed)} "
              f"best_x={best_x} hits={len(hits)} t={time.time()-t0:.0f}s", flush=True)
print(f"\nDONE iters={iters} archive={len(archive)} grounded_probed={len(probed)} best_x={best_x}")
if hits:
    print(f"REAL down-pipe entries found: {sorted(set(hits))}")
    json.dump({"hits": hits}, open("/tmp/8_4_realpipe.json", "w"))
else:
    print("NO real down-pipe entry from any reachable grounded cell (AREA/$06DE/$000E test).")
sim.close()
