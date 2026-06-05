"""Beat a multi-area level (8-4) by DFS over the ROOM GRAPH to the flag.

A pipe/sub-area transition flips $0750 (AREA_POINTER), not $0760. area_search now treats
reaching a NEW room-key (area $0760, pointer $0750) as success and forbids re-entering a
visited/dead room. We DFS the room graph: from each room try to reach an unvisited room;
recurse; on a dead-end backtrack and try a DIFFERENT exit (e.g. 101's near exit 229 is a
dead-end, so backtrack and take 101's far exit). Succeed on flag_get; save the path.

    ./venv/bin/python scripts/solve_area_chain.py [world] [stage] [per_room_budget_s] [max_depth_rooms]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mario.ram as R
from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.reward import is_success
from mario.search import area_search
from mario.render import make_contact_sheet

SOL = ROOT / "data" / "solutions"


def state_after(world, stage, path):
    sim = MarioSim(world, stage); sim.reset(0); info = sim.last_info; done = False
    for a in path:
        info, done = sim.run_chunk(a, 8)
        if done:
            break
    key = (int(sim.ram[R.AREA_NUMBER]), int(sim.ram[R.AREA_POINTER]))
    x = mario_level_x(sim.ram); flag = is_success(info); sim.close()
    return key, x, flag


def main() -> None:
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    sim = MarioSim(w, s); sim.reset(0)
    start_key = (int(sim.ram[R.AREA_NUMBER]), int(sim.ram[R.AREA_POINTER])); sim.close()
    print(f"{w}-{s}: start room {start_key}")

    dead: set = set()        # rooms with no unexplored forward exit
    best = {"path": [], "rooms": [start_key]}

    def dfs(prefix, path_keys, depth):
        """Return solved path (to flag) or None. path_keys = rooms on the current DFS stack."""
        if depth > max_depth:
            return None
        # try to reach unvisited rooms from here, one at a time
        while True:
            forbid = set(path_keys) | dead
            changed, sub, info = area_search(
                w, s, start_prefix=prefix or None, prefix_cf=8, forbid_keys=forbid,
                beam_width=96, chunk_frames=8, cov_bonus=50.0, stuck_cap=120,
                time_budget_s=budget, x_jump_px=120, progress_every=0)
            if not changed:
                return None                      # no more new rooms from here
            new_prefix = prefix + sub
            key, x, flag = state_after(w, s, new_prefix)
            print(f"  depth{depth} {path_keys[-1]} -> {key} x={x} chunks={len(new_prefix)} flag={flag}", flush=True)
            if len(new_prefix) > len(best["path"]):
                best.update(path=new_prefix, rooms=path_keys + [key])
            if flag:
                return new_prefix
            if key in path_keys or key in dead:
                dead.add(key); continue          # looped/dead; try another exit from here
            res = dfs(new_prefix, path_keys + [key], depth + 1)
            if res is not None:
                return res
            dead.add(key)                        # that room dead-ended; try another exit

    solved_path = dfs([], [start_key], 0)
    out = SOL / f"{w}-{s}.json"
    if solved_path is not None:
        out.write_text(json.dumps({"path": solved_path, "solved": True, "by": "area_chain_dfs",
                                   "chunk_frames": 8}))
        print(f"\nSOLVED {w}-{s}! saved {out} ({len(solved_path)} chunks)")
        prefix = solved_path
    else:
        print(f"\nNOT solved. deepest path {len(best['path'])} chunks rooms={best['rooms']}")
        prefix = best["path"]
    try:
        d = ROOT / "runs" / f"areachain_{w}_{s}"
        make_contact_sheet(w, s, 0, prefix, 8, d / "contact_sheet.png", cols=6, rows=5)
        print(f"contact sheet: {d/'contact_sheet.png'}")
    except Exception as e:
        print("WARN sheet:", e)


if __name__ == "__main__":
    main()
