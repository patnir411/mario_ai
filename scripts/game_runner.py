"""Play a full-game route (net-primary + search-rescue) and render the playthrough.

Loads per-level specialist checkpoints from runs/routing.json (levels without an entry are
played by search), runs the multi-stage GameRunner, writes a game_result.json, and stitches
the whole run into one full_run.mp4.

    ./venv/bin/python scripts/game_runner.py [any%|warpless] [seed]
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

from mario.artifacts import validate_game_result
from mario.io import (env_fingerprint, git_rev, new_run_id, run_dir, utc_now_iso,
                      write_json_atomic)
from mario.policy import Controller, load_policy
from mario.runner import GameRunner

DATA = ROOT / "data"
SCALE, FPS = 3, 60


def load_controllers() -> dict:
    """{(w,s): Controller} from runs/routing.json ({'1-1': '<run_id>', ...})."""
    routing_path = ROOT / "runs" / "routing.json"
    routing = json.loads(routing_path.read_text()) if routing_path.exists() else {}
    controllers = {}
    for key, run_id in routing.items():
        w, s = (int(x) for x in key.split("-"))
        ckpt = ROOT / "runs" / run_id / "checkpoint.pt"
        if ckpt.exists():
            net, meta = load_policy(ckpt, device="cpu")
            controllers[(w, s)] = Controller(net, device="cpu",
                                             chunk_frames=meta["chunk_frames"])
            print(f"  net for {key}: {run_id}")
        else:
            print(f"  WARN missing checkpoint for {key}: {run_id}")
    return controllers


def stitch_video(frames, out_path):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            img = Image.fromarray(f.astype(np.uint8))
            img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
            img.save(td / f"f{i:06d}.png")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "f%06d.png"),
                        "-pix_fmt", "yuv420p", "-an", str(out_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "any%"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    route = json.loads((DATA / "route.json").read_text())[mode]
    waypoints = json.loads((DATA / "waypoints.json").read_text()) \
        if (DATA / "waypoints.json").exists() else {}

    print(f"[game_runner] mode={mode} route={route}")
    controllers = load_controllers()
    runner = GameRunner(controllers, waypoints=waypoints, capture=True)
    result = runner.run(seed=seed)

    run_id = new_run_id(f"game_{mode.replace('%','pct')}")
    d = run_dir(run_id)
    out_mp4 = d / "full_run.mp4"
    print(f"  stitching {len(runner.frames)} frames -> {out_mp4}")
    stitch_video(runner.frames, out_mp4)

    pl = result["per_level"]
    g = {
        "run_id": run_id, "schema_version": 1, "kind": "game", "mode": mode,
        "route": route, "git_rev": git_rev(), "generated_at": utc_now_iso(),
        "env_fingerprint": env_fingerprint(),
        "beat_game": bool(result["beat_game"]),
        "per_level": pl,
        "totals": {
            "levels_cleared": sum(1 for p in pl if p["cleared"]),
            "total_framerule": sum(p["framerule"] or 0 for p in pl),
            "total_deaths": sum(p["deaths"] for p in pl),
        },
        "search_assist_levels": [f"{p['world']}-{p['stage']}" for p in pl
                                 if p["search_assist_chunks"] > 0],
        "artifacts": {"video": "full_run.mp4"},
    }
    validate_game_result(g)
    write_json_atomic(d / "game_result.json", g)
    print(f"\nbeat_game={g['beat_game']} levels_cleared={g['totals']['levels_cleared']}/"
          f"{len(route)} total_framerule={g['totals']['total_framerule']} "
          f"deaths={g['totals']['total_deaths']}")
    print(f"search-assisted levels: {g['search_assist_levels']}")
    print(f"video: {out_mp4}")
    print(f"game_result: runs/{run_id}/game_result.json")


if __name__ == "__main__":
    main()
