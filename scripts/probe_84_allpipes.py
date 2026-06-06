"""Probe the claimed down-pipes at x1296/2432/3648 using the navigation prefix that
actually crosses room 1 (.prefix_room2_8_4 reaches x3468). At every grounded snapshot
near each pipe, hold DOWN (and small L/R nudges) and test the CORRECT real-entry signal.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x, AREA_NUMBER, MARIO_Y_ON_SCREEN, PLAYER_FLOAT_STATE
BASE = 3
def entry(r):
    return int(r[AREA_NUMBER]) != BASE or int(r[0x06DE]) != 0 or int(r[0x000E]) in (2, 3)
def probe(sim, snap):
    for off, act in ((0, None), (2, 1), (4, 1), (6, 1), (2, 6), (4, 6), (8, 1)):
        sim.restore(snap); ok = True
        if act is not None:
            for _ in range(off):
                _, d = sim.run_chunk(act, 1)
                if entry(sim.ram): return True, int(sim.ram[AREA_NUMBER])
                if d: ok = False; break
        if not ok: continue
        for _ in range(44):
            _, d = sim.run_chunk(7, 1)
            if entry(sim.ram): return True, int(sim.ram[AREA_NUMBER])
            if d: break
    return False, BASE

for pf in [".prefix_room2_8_4", ".prefix_8_4_3000"]:
    d = json.load(open(ROOT / f"data/solutions/{pf}.json"))
    path = d.get("prefix") or d.get("path")
    sim = MarioSim(8, 4); sim.reset(0)
    snaps = {}     # x -> snapshot (grounded states only)
    for a in path:
        _, dn = sim.run_chunk(a, 8)
        if dn: break
        x = mario_level_x(sim.ram)
        if int(sim.ram[PLAYER_FLOAT_STATE]) == 0:
            snaps[x] = (sim.snapshot(), int(sim.ram[MARIO_Y_ON_SCREEN]))
    print(f"\n[{pf}] grounded xs: {sorted(snaps)[:40]}... total={len(snaps)}", flush=True)
    any_hit = False
    for pipe_x in (1296, 2432, 3648):
        cands = [(x, s, y) for x, (s, y) in snaps.items() if abs(x - pipe_x) <= 60]
        if not cands:
            print(f"  pipe x{pipe_x}: NO grounded approach snapshot within 60px"); continue
        hit = None
        for x, s, y in cands:
            ent, na = probe(sim, s)
            if ent: hit = (x, y, na); break
        if hit:
            print(f"  pipe x{pipe_x}: *** ENTER from x{hit[0]} y{hit[1]} -> area {hit[2]} ***"); any_hit = True
        else:
            print(f"  pipe x{pipe_x}: no entry (tried {len(cands)} grounded snaps, ys={sorted({c[2] for c in cands})})")
    sim.close()
