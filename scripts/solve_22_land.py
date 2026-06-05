"""Reverse-curriculum solver for 2-2's side-pipe entry (frame-precise, from-scratch).

Codex/disassembly facts that shape this:
  * In water, PlayerBGCollision forces Player_State $001D=1 every frame (SwimmingFlag $0704=1).
    $001D becomes 0 ONLY on a foot-landing frame (LandPlyr) within the same collision pass.
  * The side-pipe entry fires when, on such a frame: $000E==8, $001D==0, $0033==1,
    $0086&0x0f != 0, and the right side-probe tile (Player_X+13, row of Player_Y+24) is 0x6c.
  * col3024 is a solid wall (so grounded walking caps Mario at x=3010), BUT col3008 row6 is
    OPEN tunnel — Mario can DESCEND through it at x>=3011 (reachable airborne) and LAND on the
    col3008 row7 floor; on that landing frame, with Y in [104,119], the probe samples col3024's
    0x6c -> entry.

Strategy: (A) CF=2 probe-score beam to reach x>=3008 near-mouth states (airborne ok), collect a
pool. (B) From each pooled state, cf=1 sweep of sink/right/down patterns to hit the landing frame.
Saves the full action path to data/solutions/2-2.json on success.
"""
import json, sys, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim, N_ACTIONS
from mario.ram import mario_level_x, pipe_entering, stage_key, signed
from mario.observation import TILE_BASE, PAGE_BYTES
from mario.reward import is_death

PN = int(sys.argv[1]) if len(sys.argv) > 1 else 205
NAV_DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 120
POOL_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 800
path = json.load(open(ROOT/"data/solutions/2-2.json"))["path"]

def probe_tile(r):
    lx = mario_level_x(r) + 13; sy = int(r[0x00CE]) + 24; row = (sy - 32) // 16
    if row < 0 or row > 12:
        return -1
    page = (lx // 256) % 2; col = (lx % 256) // 16
    return int(r[TILE_BASE + page * PAGE_BYTES + row * 16 + col])

sim = MarioSim(2, 2); sim.reset(seed=0)
for a in path[:PN]:
    sim.run_chunk(a, 8)
ss = stage_key(sim.ram)

def score(r):
    lx = mario_level_x(r) + 13; sy = int(r[0x00CE]) + 24
    s = -(abs(lx - 3026) + abs(sy - 136))
    pt = probe_tile(r)
    if pt == 0x6c:
        s += 5000
    elif pt == 0x6b:
        s += 400
    return s

def cell(r):
    vx = signed(int(r[0x0057])); vy = signed(int(r[0x009F]))
    return (mario_level_x(r), int(r[0x00CE]), max(-2, min(2, vx // 8)),
            1 if vy > 0 else (-1 if vy < 0 else 0), int(r[0x001D]))

def finish(tail_cf, pre, tail, r, info):
    print(f"*** ENTERED *** x={mario_level_x(r)} Y={r[0x00CE]} 001D={r[0x001D]} "
          f"000E={r[0x000E]} 06DE={r[0x06DE]} stage={stage_key(r)} flag={info.get('flag_get')}")
    full = path[:PN] + pre + tail
    out = {"prefix_n": PN, "prefix_cf": 8, "pre_cf": 2, "tail": tail, "tail_cf": tail_cf,
           "pre": pre, "by": "reverse_curriculum_land"}
    json.dump(out, open("/tmp/2_2_entry.json", "w"))
    print("saved /tmp/2_2_entry.json  pre", len(pre), "tail", len(tail)); sys.exit(0)

# Phase A: CF=2 navigation, collect x>=3008 states.
beam = [(sim.snapshot(), [])]
pool = []
for depth in range(NAV_DEPTH):
    bc = {}
    for snp, p in beam:
        for a in range(N_ACTIONS):
            sim.restore(snp); info, dn = sim.run_chunk(a, 2); r = sim.ram
            if pipe_entering(r) or stage_key(r) != ss or info.get("flag_get"):
                finish(2, p + [a], [], r, info)
            if is_death(info, dn):
                continue
            c = cell(r); sc = score(r)
            if c not in bc or sc > bc[c][2]:
                bc[c] = (sim.snapshot(), p + [a], sc)
    if not bc:
        break
    ranked = sorted(bc.values(), key=lambda t: t[2], reverse=True)
    beam = [(t[0], t[1]) for t in ranked[:240]]
    for snp, pth, sc in ranked[:240]:
        sim.restore(snp)
        if mario_level_x(sim.ram) >= 3008:
            pool.append((snp, pth))
# dedup pool by (x,y,vx,001D), cap
uniq = {}
for snp, pth in pool:
    sim.restore(snp); r = sim.ram
    k = (mario_level_x(r), int(r[0x00CE]), signed(int(r[0x0057])), int(r[0x001D]))
    if k not in uniq:
        uniq[k] = (snp, pth)
pool = list(uniq.values())[:POOL_CAP]
print(f"navA done. pool of distinct x>=3008 states: {len(pool)}", flush=True)

# Phase B: from each pooled state, cf=1 landing sweep.
PATS = [list(p) for p in itertools.product([1, 0, 7], repeat=5)]   # right/noop/down, len5
tried = 0
for snp, pth in pool:
    for pat in PATS:
        sim.restore(snp); ss2 = stage_key(sim.ram); seq = []
        for a in pat + pat[:4]:   # up to 9 frames
            info, dn = sim.run_chunk(a, 1); r = sim.ram; seq.append(a)
            if pipe_entering(r) or stage_key(r) != ss2 or info.get("flag_get"):
                finish(1, pth, seq, r, info)
            if dn:
                break
        tried += 1
print(f"Phase B tried {tried} pattern-rollouts, no entry", flush=True)
