"""Render a solved trajectory to an MP4 you can watch.

Replays the V0 search path frame-by-frame, captures every NES frame, and assembles an
upscaled 60fps video via the system ffmpeg.

    ./venv/bin/python scripts/replay_video.py [run_dir]
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

from mario.env import MarioSim  # noqa: E402

SCALE = 3
FPS = 60


def latest_v0_run() -> Path:
    runs = sorted((ROOT / "runs").glob("*-v0_1_1"))
    if not runs:
        sys.exit("no V0 run found — run scripts/v0_search_1_1.py first")
    return runs[-1]


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_v0_run()
    spec = json.loads((run / "search_path.json").read_text())
    path, cf = spec["path"], spec["chunk_frames"]

    sim = MarioSim(1, 1)
    sim.reset(seed=0)
    frames = [np.asarray(sim.last_obs).copy()]
    for a in path:
        for _ in range(cf):
            obs, _info, done = sim.step(a)
            frames.append(np.asarray(obs).copy())
            if done:
                break
        if done:
            break
    sim.close()
    print(f"captured {len(frames)} frames")

    out = run / "play.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            img = Image.fromarray(f.astype(np.uint8))
            img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
            img.save(td / f"f{i:05d}.png")
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "f%05d.png"),
               "-pix_fmt", "yuv420p", "-an", str(out)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out)


if __name__ == "__main__":
    main()
