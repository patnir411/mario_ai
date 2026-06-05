"""Phase A — 2-2 mouth brute-forcer (de-risk the corrected side-pipe mechanics).

Binary question: does 2-2's side-exit fire AT ALL under true-random FULL-BUTTON input from
real near-mouth states? Detection is on engine-state bytes only (pipe_entering / stage_key),
NEVER the scroll-dependent $0500 tile buffer (which poisoned every prior attempt).

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/probe_22_mouth.py [pool] [rollouts] [horizon]

On the first fire it writes /tmp/2_2_fire.json (firing_state of the firing frame + the prior
frame, plus the exact action sequence) and stops — that's verified ground truth for Phase B.
"""
import json, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim, ACTIONS
from mario.ram import mario_level_x, pipe_entering, stage_key, signed, firing_state

# Full per-frame button vocabulary relevant underwater (incl. coast / down+right / A-tap).
WATER_ACTIONS = [
    ["NOOP"],              # 0 coast (release all) — A-tap timing comes from interleaving this
    ["A"],                 # 1 stroke in place
    ["right"],             # 2 drift right (face right, no stroke -> $001D coasts to 0)
    ["right", "A"],        # 3 stroke + right
    ["left"],              # 4
    ["left", "A"],         # 5
    ["down"],              # 6
    ["down", "A"],         # 7
    ["up"],                # 8
    ["right", "down"],     # 9  descend-right toward the mouth row
    ["right", "down", "A"],# 10
]

def _bset(a):  # button set with B stripped (B is irrelevant underwater)
    return frozenset(b for b in a if b not in ("B", "NOOP"))

# Map each default-ACTIONS index -> the WATER_ACTIONS index with the same button set.
_W_BY_SET = {_bset(a): i for i, a in enumerate(WATER_ACTIONS)}
def default_to_water(idx):
    return _W_BY_SET.get(_bset(ACTIONS[idx]), 2)  # fall back to 'right'

POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROLLOUTS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
HORIZON = int(sys.argv[3]) if len(sys.argv) > 3 else 90
BIAS = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 1 = weight toward swim-right-into-mouth
PREFIX_N = 224
FOCUS_LO, FOCUS_HI = 2900, 3045

# Weighted toward the human approach (mostly RIGHT, occasional stroke to hold height, coast
# frames give $001D==0 at the mouth). Indices per WATER_ACTIONS above.
_WEIGHTS = [3, 2, 30, 18, 1, 1, 4, 2, 1, 10, 6]
_CUM = []
_acc = 0
for w in _WEIGHTS:
    _acc += w; _CUM.append(_acc)
def sample_action(rng):
    if not BIAS:
        return rng.randrange(len(WATER_ACTIONS))
    t = rng.randrange(_CUM[-1])
    for i, c in enumerate(_CUM):
        if t < c:
            return i
    return 2

prefix = json.load(open(ROOT / "data/solutions/2-2.json"))["path"][:PREFIX_N]
wprefix = [default_to_water(a) for a in prefix]

def nav():
    s = MarioSim(2, 2, actions=WATER_ACTIONS); s.reset(seed=0)
    for a in wprefix:
        _, d = s.run_chunk(a, 8)
        if d:
            return None
    return s

def fine_cell(r):
    vx = signed(int(r[0x0057]))
    return (mario_level_x(r), int(r[0x00CE]) // 4, int(r[0x001D]),
            1 if vx > 0 else (-1 if vx < 0 else 0))

# --- Build a pool of distinct near-mouth save-states by random exploration in the band ---
base = nav()
if base is None:
    print("nav died during prefix replay"); sys.exit(1)
print(f"nav -> x={mario_level_x(base.ram)} Y={base.ram[0x00CE]} (band [{FOCUS_LO},{FOCUS_HI}])", flush=True)
base_snap = base.snapshot()
start_stage = stage_key(base.ram)
sim = base
pool = {}
rng = random.Random(0)
explore_iters = 0
while len(pool) < POOL and explore_iters < 60000:
    explore_iters += 1
    sim.restore(base_snap)
    for _ in range(rng.randint(4, 40)):
        a = rng.randrange(len(WATER_ACTIONS))
        info, d = sim.run_chunk(a, 1); r = sim.ram
        if pipe_entering(r) or stage_key(r) != start_stage or info.get("flag_get"):
            print("FIRE during pool-build!", firing_state(r));
            json.dump({"phase": "poolbuild", "firing_state": firing_state(r)}, open("/tmp/2_2_fire.json", "w"))
            sys.exit(0)
        if d:
            break
        x = mario_level_x(r)
        if FOCUS_LO <= x <= FOCUS_HI:
            c = fine_cell(r)
            if c not in pool:
                pool[c] = sim.snapshot()
pool = list(pool.values())
print(f"pool of distinct near-mouth states: {len(pool)} (explore_iters={explore_iters})", flush=True)

# --- True-random full-button rollouts from each pooled state; detect the trigger ---
tried = 0
for pid, snap in enumerate(pool):
    for _ in range(ROLLOUTS):
        sim.restore(snap); ss = stage_key(sim.ram); seq = []; prev = firing_state(sim.ram)
        for f in range(HORIZON):
            a = sample_action(rng)
            info, d = sim.run_chunk(a, 1); r = sim.ram; seq.append(a)
            if pipe_entering(r) or stage_key(r) != ss or info.get("flag_get"):
                fs = firing_state(r)
                print(f"*** FIRE *** pool={pid} frame={f} {fs}", flush=True)
                json.dump({"prefix_n": PREFIX_N, "pool_id": pid, "rollout_actions": seq,
                           "firing_state": fs, "prev_state": prev,
                           "flag": bool(info.get("flag_get")), "stage": list(stage_key(r))},
                          open("/tmp/2_2_fire.json", "w"))
                print("saved /tmp/2_2_fire.json"); sys.exit(0)
            if d:
                break
            prev = firing_state(r)
        tried += 1
    if pid % 25 == 0:
        print(f"  ...{pid}/{len(pool)} states, {tried} rollouts", flush=True)
print(f"NO FIRE across {tried} rollouts from {len(pool)} near-mouth states.", flush=True)
