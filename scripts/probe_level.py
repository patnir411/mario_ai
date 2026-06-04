"""Search a level, save the best path, and render a contact sheet to SEE where it stalls.
Used to author waypoints for maze/warp levels.

    ./venv/bin/python scripts/probe_level.py <world> <stage> [beam] [max_depth] [max_seconds]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.io import new_run_id, run_dir, write_json_atomic
from mario.render import make_contact_sheet
from mario.search import beam_search


def main() -> None:
    w = int(sys.argv[1]); s = int(sys.argv[2])
    beam = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    r = beam_search(w, s, beam_width=beam, max_depth=max_depth, progress_every=30)
    rid = new_run_id(f"probe_{w}_{s}")
    d = run_dir(rid)
    write_json_atomic(d / "search_path.json", {"path": r.path, "chunk_frames": r.chunk_frames})
    sheet = make_contact_sheet(w, s, 0, r.path, r.chunk_frames, d / "contact_sheet.png",
                               cols=6, rows=5)
    print(f"{w}-{s}: solved={r.solved} x_max={r.x_max} depth={r.depth_reached} "
          f"nodes={r.nodes_expanded} wall={r.wall_clock_s:.0f}s")
    print(f"contact sheet: {d/'contact_sheet.png'}  ({sheet})")


if __name__ == "__main__":
    main()
