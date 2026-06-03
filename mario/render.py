"""Contact-sheet renderer — the VISUAL verification channel.

Replays a search trajectory, samples key frames by x-progress, and lays them out in an
annotated grid PNG that Claude can read with the Read tool to confirm by eye what the
JSON claims (e.g. the flagpole appears in the last cell of a beaten level). Per the plan:
render ONCE by replaying the best trajectory, never during search.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mario.env import MarioSim
from mario.reward import is_death, is_success

# colors
GREEN = (40, 200, 60)
RED = (220, 40, 40)
YELLOW = (235, 200, 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def replay(world: int, stage: int, seed: int, path: list[int], chunk_frames: int):
    """Replay a chunk path; return list of per-chunk records (frame, info, idx, flags)."""
    sim = MarioSim(world, stage)
    info = sim.reset(seed=seed)
    frames = [{"frame": np.asarray(sim.last_obs).copy(), "info": dict(info),
               "idx": 0, "died": False, "flag": False}]
    for i, a in enumerate(path, start=1):
        info, done = sim.run_chunk(a, chunk_frames)
        died = is_death(info, done)
        frames.append({"frame": np.asarray(sim.last_obs).copy(), "info": dict(info),
                       "idx": i, "died": died, "flag": is_success(info)})
        if done:
            break
    sim.close()
    return frames


def _pick(records, n):
    """Pick n records sampled evenly by x-progress, always including first, x_max, last."""
    if len(records) <= n:
        return records
    x_max_idx = max(range(len(records)), key=lambda i: records[i]["info"].get("x_pos", 0))
    must = {0, len(records) - 1, x_max_idx}
    # evenly sample the rest by x order
    order = sorted(range(len(records)), key=lambda i: records[i]["info"].get("x_pos", 0))
    step = max(1, len(order) // (n - len(must)))
    chosen = set(must)
    for j in range(0, len(order), step):
        chosen.add(order[j])
        if len(chosen) >= n:
            break
    return [records[i] for i in sorted(chosen)]


def make_contact_sheet(world: int, stage: int, seed: int, path: list[int],
                       chunk_frames: int, out_path: str | Path,
                       cols: int = 5, rows: int = 5, scale: int = 1) -> dict:
    records = replay(world, stage, seed, path, chunk_frames)
    x_max_rec = max(records, key=lambda r: r["info"].get("x_pos", 0))
    x_max = x_max_rec["info"].get("x_pos", 0)

    picks = _pick(records, cols * rows)
    fh, fw = picks[0]["frame"].shape[:2]
    cw, ch = fw * scale, fh * scale
    pad, label_h = 4, 16
    cell_w, cell_h = cw + 2 * pad, ch + 2 * pad + label_h

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for k, rec in enumerate(picks):
        r, c = divmod(k, cols)
        x0, y0 = c * cell_w, r * cell_h
        img = Image.fromarray(rec["frame"].astype(np.uint8))
        if scale != 1:
            img = img.resize((cw, ch), Image.NEAREST)
        sheet.paste(img, (x0 + pad, y0 + pad + label_h))

        # border: red death, yellow x_max, green alive
        is_xmax = rec is x_max_rec
        color = RED if rec["died"] else (YELLOW if is_xmax else GREEN)
        if rec["flag"]:
            color = GREEN
        draw.rectangle([x0 + pad - 2, y0 + pad + label_h - 2,
                        x0 + pad + cw + 1, y0 + pad + label_h + ch + 1], outline=color, width=2)

        info = rec["info"]
        tag = "FLAG" if rec["flag"] else ("DEAD" if rec["died"] else ("XMAX" if is_xmax else ""))
        label = f"#{rec['idx']} x{info.get('x_pos',0)} t{info.get('time','?')} {tag}"
        draw.text((x0 + pad, y0 + 2), label, fill=WHITE)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return {
        "n_records": len(records),
        "x_max": int(x_max),
        "beat": bool(records[-1]["flag"]),
        "died": bool(records[-1]["died"]),
        "out": str(out_path),
    }
