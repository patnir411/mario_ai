"""Definitive test: can 8-4's page-16 loop-gate be crossed at FINE (cf=2) resolution?

Replays the cf=8 grounded approach to near the gate, then runs a frame-level beam that
explores every jump-height/crouch at the crossing, loop-pruned (a backward page-teleport
kills the node). Success = x pushes clearly past the gate (no snap-back). If even this
finds nothing, the surface crossing is impossible at any resolution -> 8-4 needs the pipe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.env import MarioSim, N_ACTIONS
from mario.ram import mario_level_x
from mario.reward import is_death, is_success

CF = 2
BACKOFF_X = 3500     # replay the coarse approach until just before the gate
TARGET_X = 4300      # clearly past page 16 (4096) without snapping back
AREA = 0x0760


def main() -> None:
    approach = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ckpt_8_4_approach.json"))
    apath = approach["path"]               # cf=8 chunks
    beam_w = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else 1200.0

    sim = MarioSim(8, 4); sim.reset(0)
    prefix = []
    for a in apath:
        info, done = sim.run_chunk(a, 8)
        prefix.append((a, 8))
        if mario_level_x(sim.ram) >= BACKOFF_X or done:
            break
    x0 = mario_level_x(sim.ram)
    print(f"approach -> x={x0} Y={sim.ram[0x00CE]} float={sim.ram[0x001D]} "
          f"prefix_chunks={len(prefix)}", flush=True)
    root_snap = sim.snapshot()

    # fine beam: nodes = (snap, path[cf2 actions], x_max, stuck, area)
    beam = [(root_snap, [], x0, 0, int(sim.ram[AREA]))]
    seen_best = x0
    best_path = []
    t0 = time.perf_counter()
    for d in range(depth):
        if time.perf_counter() - t0 > budget:
            print("budget hit", flush=True); break
        cand = []
        for snap, path, xmax, stuck, area in beam:
            for a in range(N_ACTIONS):
                sim.restore(snap)
                info, done = sim.run_chunk(a, CF)
                if is_success(info):
                    full = [(z, 8) for z in apath[:len(prefix)]] + [(b, CF) for b in path + [a]]
                    json.dump({"prefix": apath[:len(prefix)], "fine": path + [a],
                               "solved": True}, open("/tmp/gate_solved.json", "w"))
                    print(f">>> FLAG via gate at depth {d}! saved /tmp/gate_solved.json", flush=True)
                    sim.close(); return
                if is_death(info, done):
                    continue
                x = mario_level_x(sim.ram); na = int(sim.ram[AREA])
                if na != area:
                    print(f">>> AREA CHANGE {area}->{na} at x={x} depth={d}!", flush=True)
                if na == area and x < xmax - 250:    # loop teleport -> kill
                    continue
                ns = stuck + (0 if x > xmax else 1)
                if ns > 80:
                    continue
                cand.append((sim.snapshot(), path + [a], max(xmax, x), ns, na, x,
                             int(sim.ram[0x00CE])))
        if not cand:
            print("beam emptied", flush=True); break
        # dedup by fine cell (x//8, Y//8), keep max x
        bykey = {}
        for snap, path, xmax, ns, na, x, y in cand:
            k = (na, x // 8, y // 8)
            if k not in bykey or x > bykey[k][5]:
                bykey[k] = (snap, path, xmax, ns, na, x, y)
        ranked = sorted(bykey.values(), key=lambda t: t[5], reverse=True)[:beam_w]
        beam = [(s, p, xm, ns, na) for (s, p, xm, ns, na, x, y) in ranked]
        top = ranked[0]
        if top[5] > seen_best:
            seen_best = top[5]; best_path = top[1]
        if top[5] > TARGET_X:
            json.dump({"prefix": apath[:len(prefix)], "fine": top[1], "solved_gate": True},
                      open("/tmp/gate_crossed.json", "w"))
            print(f">>> CROSSED GATE: x={top[5]} depth={d}! saved /tmp/gate_crossed.json", flush=True)
            sim.close(); return
        if d % 10 == 0:
            print(f"  d{d:3d} beam{len(beam):3d} best_x{seen_best} top_x{top[5]} "
                  f"{d/(time.perf_counter()-t0):.1f}d/s", flush=True)
    print(f"DONE max_x={seen_best} (gate NOT crossed)", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
