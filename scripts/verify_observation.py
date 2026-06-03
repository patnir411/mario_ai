"""Visually verify the tile extractor: overlay extracted tiles on the rendered frame.

Red box = SOLID tile, blue = ENEMY slot, green = Mario's tile. If the red boxes line up
with ground/pipes/blocks and blue with enemies, extraction is correct.

    ./venv/bin/python scripts/verify_observation.py
writes runs/_debug/obs_overlay_*.png  (Read them to confirm by eye)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario import ram as R  # noqa: E402
from mario.env import MarioSim  # noqa: E402
from mario.observation import HUD_Y_OFFSET, N_ROWS, get_tile  # noqa: E402

OUT = ROOT / "runs" / "_debug"


def overlay(sim: MarioSim, name: str) -> str:
    ram = sim.ram
    frame = np.asarray(sim.last_obs).astype(np.uint8)
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    mario_x = R.mario_level_x(ram)
    mario_x_screen = int(ram[0x03AD])   # true on-screen pixel x (not 0x86)
    scroll = mario_x - mario_x_screen   # level-x of screen's left edge

    # solid tiles across the visible screen (16 cols x 13 rows)
    for col in range(16):
        for row in range(N_ROWS):
            lx = scroll + col * 16
            sy = row * 16 + HUD_Y_OFFSET
            if get_tile(ram, lx, sy):
                px, py = col * 16, sy
                d.rectangle([px, py, px + 15, py + 15], outline=(255, 0, 255, 255),
                            fill=(255, 0, 255, 90))  # magenta: contrasts with red bricks

    # enemies (blue)
    for i in range(5):
        if int(ram[R.ENEMY_ACTIVE.start + i]) == 0:
            continue
        ex = int(ram[R.ENEMY_X_LEVEL.start + i]) * 256 + int(ram[0x0087 + i])
        ey = int(ram[R.ENEMY_Y_ON_SCREEN.start + i])
        sx = ex - scroll
        d.rectangle([sx, ey, sx + 15, ey + 15], outline=(0, 80, 255, 255),
                    fill=(0, 80, 255, 110))

    # Mario's tile (green)
    my = int(ram[R.MARIO_Y_ON_SCREEN])
    d.rectangle([mario_x_screen, my, mario_x_screen + 15, my + 15],
                outline=(0, 255, 0, 255), width=2)

    out = Image.alpha_composite(img, ov).convert("RGB")
    out = out.resize((512, 480), Image.NEAREST)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"obs_overlay_{name}.png"
    out.save(p)
    return str(p)


def main() -> None:
    # Prefer the V0 solved path if available (lets us reach pipes/goombas alive).
    path_file = next(iter(sorted((ROOT / "runs").glob("*-v0_1_1/search_path.json"))), None)
    sim = MarioSim(1, 1)
    sim.reset(seed=0)
    print(overlay(sim, "00_start"))

    if path_file:
        import json
        path = json.loads(path_file.read_text())["path"]
        cf = json.loads(path_file.read_text())["chunk_frames"]
        shots = {20: "01", 45: "02_pipe", 70: "03", 95: "04_stairs"}
        for i, a in enumerate(path, 1):
            info, done = sim.run_chunk(a, cf)
            if i in shots:
                print(overlay(sim, shots[i]))
            if done:
                break
    else:
        # fallback: a few alive frames before the first goomba
        for n, tag in [(20, "01"), (40, "02")]:
            for _ in range(20):
                _o, _i, done = sim.step(3)
                if done:
                    break
            print(overlay(sim, tag))
    sim.close()


if __name__ == "__main__":
    main()
