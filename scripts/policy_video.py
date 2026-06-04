"""Render a video of a trained POLICY playing a level (not the search).

    ./venv/bin/python scripts/policy_video.py <checkpoint_run_id> [world] [stage] [seed]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.env import MarioSim
from mario.policy import Controller, load_policy

SCALE, FPS = 3, 60


def main() -> None:
    run = sys.argv[1]
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    stage = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    ckpt = ROOT / "runs" / run / "checkpoint.pt"
    net, meta = load_policy(ckpt, device="cpu")
    ctrl = Controller(net, device="cpu", chunk_frames=meta["chunk_frames"])
    sim = MarioSim(world, stage)
    sim.reset(seed=seed)
    ctrl.reset()

    frames = [np.asarray(sim.last_obs).copy()]
    done = False
    for _ in range(400):
        a = ctrl.act(sim.ram, sim.last_info)
        for _ in range(meta["chunk_frames"]):
            obs, info, done = sim.step(a)
            frames.append(np.asarray(obs).copy())
            if done:
                break
        if done:
            break
    beat = bool(sim.last_info.get("flag_get"))
    sim.close()
    print(f"frames={len(frames)} beat={beat}")

    out = ROOT / "runs" / run / f"play_{world}-{stage}.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            img = Image.fromarray(f.astype(np.uint8))
            img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
            img.save(td / f"f{i:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "f%05d.png"),
                        "-pix_fmt", "yuv420p", "-an", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out)


if __name__ == "__main__":
    main()
