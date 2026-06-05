"""Targeted beam that drives Mario to the 2-2 side-pipe mouth (x~3016,row6=0x6c)
by minimizing distance to the mouth tile, then enters. Success = pipe_entering/stage/flag.
    ./venv/bin/python scripts/solve_22_entry.py [prefix_n] [beam] [depth] [cf]
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim, N_ACTIONS
from mario.ram import mario_level_x, pipe_entering, stage_key, signed
from mario.reward import is_death

path = json.load(open(ROOT/"data/solutions/2-2.json"))["path"]
PN = int(sys.argv[1]) if len(sys.argv) > 1 else 205
BW = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 220
CF = int(sys.argv[4]) if len(sys.argv) > 4 else 2
TX = 3016
sim = MarioSim(2, 2); sim.reset(seed=0)
for a in path[:PN]:
    sim.run_chunk(a, 8)
ss = stage_key(sim.ram)

from mario.observation import TILE_BASE, PAGE_BYTES
def probe_tile(r):
    """The engine's right-collision sample (Codex): (Player_X+13, Player_Y+24)."""
    lx = mario_level_x(r) + 13; sy = int(r[0x00CE]) + 24
    row = (sy - 32) // 16
    if row < 0 or row > 12:
        return -1
    page = (lx // 256) % 2; col = (lx % 256) // 16
    return int(r[TILE_BASE + page * PAGE_BYTES + row * 16 + col])

def score(r):
    # Drive the EXACT engine probe (X+13,Y+24) onto the 0x6c mouth tile.
    lx = mario_level_x(r) + 13; sy = int(r[0x00CE]) + 24
    s = -(abs(lx - 3026) + abs(sy - 136) * 1.0)   # want probe at col3024(=3024..3039), row6(sy~128..143)
    pt = probe_tile(r)
    if pt == 0x6c:
        s += 2000                                  # probe samples the mouth — the trigger cell
    elif pt == 0x6b:
        s += 400
    if int(r[0x001D]) == 0:
        s += 50
    return s

def cell(r):
    # Fine cell so cf=1 (sub-2px moves) doesn't collapse the beam to one cell:
    # exact x, exact y, signed vx bucket, player-state. Keeps grounded/swim/landing distinct.
    vx = signed(int(r[0x0057]))
    return (mario_level_x(r), int(r[0x00CE]),
            max(-3, min(3, vx // 4)), int(r[0x001D]), int(r[0x009F]) > 0)

start = sim.snapshot()
beam = [(start, [], score(sim.ram))]
best = (score(sim.ram), [])
best_snap = start
t0 = time.time()
for depth in range(DEPTH):
    by_cell = {}
    for snap, p, _ in beam:
        for a in range(N_ACTIONS):
            sim.restore(snap)
            info, done = sim.run_chunk(a, CF); r = sim.ram
            if pipe_entering(r) or stage_key(r) != ss or info.get("flag_get"):
                print(f"ENTERED depth={depth} x={mario_level_x(r)} Y={r[0x00CE]} "
                      f"000E={r[0x000E]} 06DE={r[0x06DE]} stage={stage_key(r)} flag={info.get('flag_get')}")
                json.dump({"prefix_n": PN, "prefix_cf": 8, "tail": p+[a], "tail_cf": CF},
                          open("/tmp/2_2_entry.json", "w"))
                print("saved /tmp/2_2_entry.json tail", len(p)+1); sys.exit()
            if is_death(info, done):
                continue
            c = cell(r); sc = score(r)
            if c not in by_cell or sc > by_cell[c][2]:
                by_cell[c] = (sim.snapshot(), p+[a], sc)
    if not by_cell:
        print(f"beam empty at depth {depth}"); break
    ranked = sorted(by_cell.values(), key=lambda t: t[2], reverse=True)
    beam = ranked[:BW]
    if beam[0][2] > best[0]:
        best = (beam[0][2], beam[0][1])
        best_snap = ranked[0][0]
    if depth % 20 == 0:
        print(f"depth {depth:3d} beam {len(beam):3d} bestscore {best[0]:.0f} "
              f"cells {len(by_cell)} {time.time()-t0:.0f}s", flush=True)
print("NO ENTRY. best score", best[0], "bestlen", len(best[1]))
sim.restore(best_snap)
r = sim.ram
print(f"best_snap x={mario_level_x(r)} sub86={r[0x0086]} Y={r[0x00CE]} 001D={r[0x001D]} "
      f"000E={r[0x000E]} 06DE={r[0x06DE]} vx={signed(int(r[0x0057]))}")
# Try several entry actions frame-by-frame from the exact best snapshot.
for label, act in [("RIGHT", 1), ("RIGHT+A", 2), ("RIGHT+B", 3), ("DOWN", 7), ("DOWN+RIGHT_then", 7)]:
    sim.restore(best_snap)
    out = []
    fired = False
    for f in range(30):
        info, done = sim.run_chunk(act, 1); r = sim.ram
        if f < 12:
            out.append(f"x{mario_level_x(r)} sub{r[0x0086]} Y{r[0x00CE]} s1D{r[0x001D]} e{r[0x000E]} d{r[0x06DE]}")
        if pipe_entering(r) or stage_key(r) != ss or info.get("flag_get"):
            print(f"[{label}] >>> TRIGGER at f{f} stage={stage_key(r)} flag={info.get('flag_get')}"); fired = True; break
        if done:
            out.append("DIED"); break
    if not fired:
        print(f"[{label}] no-trigger:", " | ".join(out[:8]))
