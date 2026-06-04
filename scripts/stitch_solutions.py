"""Stitch cached per-level solutions (data/solutions/*.json) into one playthrough video.

Pure replay of already-solved paths in route order — no search, no net, deterministic.
Each level is replayed from its own reset to the flag; the clips are concatenated into a
single full_run.mp4 that reads as a continuous playthrough. "Beat the game" for a mode =
every level in that route has a solved cache.

    ./venv/bin/python scripts/stitch_solutions.py [any%|warpless]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.render import replay

DATA = ROOT / "data"
SOL = DATA / "solutions"
SCALE, FPS = 3, 60


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "any%"
    route = json.loads((DATA / "route.json").read_text())[mode]

    frames, summary, missing = [], [], []
    for lvl in route:
        cache = SOL / f"{lvl}.json"
        if not (cache.exists() and json.loads(cache.read_text()).get("solved")):
            missing.append(lvl)
            print(f"  {lvl}: MISSING solved cache — stopping here", flush=True)
            break
        c = json.loads(cache.read_text())
        w, s = (int(x) for x in lvl.split("-"))
        recs = replay(w, s, 0, c["path"], c.get("chunk_frames", 8))
        beat = recs[-1]["flag"]
        # title card: a few frames labeled with the level
        title = np.asarray(recs[0]["frame"]).copy()
        img = Image.fromarray(title.astype(np.uint8))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
        d.text((4, 3), f"WORLD {lvl}", fill=(255, 255, 255))
        for _ in range(20):
            frames.append(np.asarray(img))
        frames.extend(np.asarray(r["frame"]) for r in recs)
        summary.append({"level": lvl, "chunks": len(c["path"]), "beat": bool(beat),
                        "by": c.get("by")})
        print(f"  {lvl}: chunks={len(c['path'])} beat={beat} by={c.get('by')}", flush=True)

    out_dir = ROOT / "runs" / f"playthrough_{mode.replace('%','pct')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "full_run.mp4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            im = Image.fromarray(np.asarray(f).astype(np.uint8))
            im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
            im.save(td / f"f{i:06d}.png")
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "f%06d.png"),
                        "-pix_fmt", "yuv420p", "-an", str(out)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    n_beat = sum(1 for s in summary if s["beat"])
    result = {"mode": mode, "route_len": len(route), "levels_beaten": n_beat,
              "beat_game": n_beat == len(route) and not missing,
              "missing": missing, "levels": summary, "video": str(out)}
    (out_dir / "playthrough_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nmode={mode} beaten={n_beat}/{len(route)} beat_game={result['beat_game']}")
    if missing:
        print(f"missing: {missing}")
    print(f"video: {out}")


if __name__ == "__main__":
    main()
