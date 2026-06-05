"""Solve 2-2 (underwater) from scratch and cache data/solutions/2-2.json.

Mechanism (Codex/disassembly-verified; see 2-2_FINDINGS.md): the exit side-pipe is entered by
simply HOLDING RIGHT from the water-nav prefix — Mario sinks while drifting right and at
x=3011, Y=112 on a foot-landing frame ($001D=0) the right side-probe (X+13=3024) first reaches
the mouth metatile $6c, firing SideExitPipeEntry ($000E=2). gym fast-forwards this transition
INSIDE env.step(), so we DETECT it by the post-step backward x-jump (NOT pipe_entering, which the
wrapper has already cleaned up). Then a normal beam in surface area-2 reaches the flagpole.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_22.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.search import search_from_state
from mario.render import replay

CACHE = ROOT / "data/solutions/2-2.json"
PREFIX_N = 220  # water-nav prefix (cf=8) from the existing cache; reaches x~2916 near the mouth

prefix = json.loads(CACHE.read_text())["path"][:PREFIX_N]
sim = MarioSim(2, 2); sim.reset(seed=0)
for a in prefix:
    sim.run_chunk(a, 8)

# 1) Enter the side-pipe by holding RIGHT (cf=1); detect the transition by the x-jump.
prev_x = mario_level_x(sim.ram); entry_right = 0
for f in range(120):
    info, d = sim.run_chunk(1, 1); entry_right += 1
    if prev_x - mario_level_x(sim.ram) > 100 and f > 40:
        break
    prev_x = mario_level_x(sim.ram)
post = sim.snapshot()
print(f"entered side-pipe after {entry_right} right-frames; area-2 info "
      f"{ {k: info.get(k) for k in ('world','stage','x_pos','y_pos')} }", flush=True)

# 2) Beam from the post-entry state through surface area-2 to the flagpole.
tail = search_from_state(sim, post, world=2, stage=2, beam_width=80, depth=500,
                         chunk_frames=8, max_seconds=180)

# 3) Normalize the whole solution to a single chunk_frames (cf=1) and replay-verify.
cf1 = []
for a in prefix:
    cf1 += [a] * 8
cf1 += [1] * entry_right
for a in tail:
    cf1 += [a] * 8
recs = replay(2, 2, 0, cf1, 1)
beat = recs[-1]["flag"]; lastx = recs[-1]["info"].get("x_pos")
print(f"normalized cf1 len={len(cf1)}  replay beat={beat}  flag x_pos={lastx}", flush=True)
if beat:
    CACHE.write_text(json.dumps({"path": cf1, "solved": True,
                                 "by": "holdright_pipe+beam_area2",
                                 "chunk_frames": 1, "x_max": lastx}))
    print(f"saved {CACHE} solved=True")
else:
    print("NOT solved — did not reach flag"); sys.exit(1)
