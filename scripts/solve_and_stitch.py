"""Beat the game by solving each level offline (search, unlimited retries -> death-free),
then stitch every level's playthrough into one continuous full_run.mp4.

This sidesteps the multi-stage lives limit: each level is solved independently to the flag
(net for 1-1 to showcase the learned policy; beam search for the rest, with waypoints for
maze castles). Cached per-level so it's resumable. "Beat the game" = every route level cleared.

    ./venv/bin/python scripts/solve_and_stitch.py [any%|warpless] [beam] [max_depth]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.env import MarioSim
from mario.io import new_run_id, run_dir, write_json_atomic
from mario.policy import Controller, load_policy
from mario.render import replay
from mario.reward import is_success
from mario.search import search_from_state

DATA = ROOT / "data"
SOL = DATA / "solutions"
SCALE, FPS = 3, 60


def net_path_for_1_1():
    """Use the learned net to produce 1-1's path (showcase the trained policy)."""
    routing = json.loads((ROOT / "runs" / "routing.json").read_text())
    rid = routing.get("1-1")
    if not rid:
        return None
    net, meta = load_policy(ROOT / "runs" / rid / "checkpoint.pt", "cpu")
    ctrl = Controller(net, "cpu", meta["chunk_frames"])
    sim = MarioSim(1, 1); sim.reset(0); ctrl.reset()
    path = []
    for _ in range(400):
        a = ctrl.act(sim.ram, sim.last_info); path.append(a)
        info, done = sim.run_chunk(a, meta["chunk_frames"])
        if done:
            break
    sim.close()
    return path if is_success(info) else None


def solve_level(world, stage, beam, max_depth, waypoints):
    cache = SOL / f"{world}-{stage}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("solved"):
            return c["path"], True
    if (world, stage) == (1, 1):
        p = net_path_for_1_1()
        if p:
            write_json_atomic(cache, {"path": p, "solved": True, "by": "net"})
            return p, True
    sim = MarioSim(world, stage)
    sim.reset(0)
    snap = sim.snapshot()
    path = search_from_state(sim, snap, world=world, stage=stage, beam_width=beam,
                             depth=max_depth, waypoints=waypoints, max_seconds=240)
    sim.restore(snap)
    solved = False
    for a in path:
        info, done = sim.run_chunk(a, 8)
        if is_success(info):
            solved = True
            break
        if done:
            break
    sim.close()
    write_json_atomic(cache, {"path": path, "solved": solved, "by": "search"})
    return path, solved


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "warpless"
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 320
    route = [tuple(int(x) for x in k.split("-"))
             for k in json.loads((DATA / "route.json").read_text())[mode]]
    waypoints = json.loads((DATA / "waypoints.json").read_text()) \
        if (DATA / "waypoints.json").exists() else {}
    SOL.mkdir(parents=True, exist_ok=True)

    frames, results = [], []
    for (w, s) in route:
        wp = waypoints.get(f"{w}-{s}")
        path, solved = solve_level(w, s, beam, max_depth, wp)
        results.append((w, s, solved, len(path)))
        print(f"  {w}-{s}: solved={solved} chunks={len(path)}", flush=True)
        recs = replay(w, s, 0, path, 8)
        frames.extend(r["frame"] for r in recs)
        if not solved:
            print(f"  STOP: {w}-{s} not solved — rendering up to here.")
            break

    run_id = new_run_id(f"stitch_{mode.replace('%','pct')}")
    d = run_dir(run_id)
    out = d / "full_run.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            img = Image.fromarray(np.asarray(f).astype(np.uint8))
            img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
            img.save(td / f"f{i:06d}.png")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "f%06d.png"),
                        "-pix_fmt", "yuv420p", "-an", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n_solved = sum(1 for _, _, sv, _ in results if sv)
    write_json_atomic(d / "stitch_result.json",
                      {"mode": mode, "levels_solved": n_solved, "route_len": len(route),
                       "beat_game": n_solved == len(route),
                       "levels": [{"world": w, "stage": s, "solved": sv, "chunks": c}
                                  for w, s, sv, c in results]})
    print(f"\nlevels_solved={n_solved}/{len(route)} beat_game={n_solved==len(route)}")
    print(f"video: {out}")


if __name__ == "__main__":
    main()
