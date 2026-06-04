"""Run Go-Explore on a level; save the solution path + a contact sheet.

    ./venv/bin/python scripts/goexplore_level.py <world> <stage> [time_budget_s] [chunk_frames]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.goexplore import go_explore
from mario.io import new_run_id, run_dir, write_json_atomic
from mario.render import make_contact_sheet

SOL = ROOT / "data" / "solutions"


def main() -> None:
    w = int(sys.argv[1]); s = int(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    cf = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    print(f"[go-explore] {w}-{s} budget={budget}s chunk_frames={cf}", flush=True)
    solved, path, stats = go_explore(w, s, chunk_frames=cf, time_budget_s=budget)
    print(f"{w}-{s}: solved={solved} path_len={len(path)} stats={stats}")

    rid = new_run_id(f"goex_{w}_{s}")
    d = run_dir(rid)
    write_json_atomic(d / "search_path.json", {"path": path, "chunk_frames": cf})
    try:
        sheet = make_contact_sheet(w, s, 0, path, cf, d / "contact_sheet.png", cols=6, rows=5)
        print("contact sheet:", sheet)
    except Exception as e:
        print("WARN sheet:", e)
    if solved:
        SOL.mkdir(parents=True, exist_ok=True)
        write_json_atomic(SOL / f"{w}-{s}.json",
                          {"path": path, "solved": True, "by": "go-explore",
                           "chunk_frames": cf})
        print(f"saved solution data/solutions/{w}-{s}.json")


if __name__ == "__main__":
    main()
