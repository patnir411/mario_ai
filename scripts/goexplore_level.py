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
from mario.solution_verification import replay_verify_stock_path

SOL = ROOT / "data" / "solutions"


def main() -> None:
    w = int(sys.argv[1]); s = int(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    cf = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    print(f"[go-explore] {w}-{s} budget={budget}s chunk_frames={cf}", flush=True)
    solved, path, stats = go_explore(w, s, chunk_frames=cf, time_budget_s=budget)
    replay_verified, replay_reason = (
        replay_verify_stock_path(w, s, list(path), cf, seed=0)
        if solved else (False, "search_did_not_reach_flag")
    )
    print(f"{w}-{s}: search_solved={solved} replay_verified={replay_verified} "
          f"path_len={len(path)} stats={stats}")

    rid = new_run_id(f"goex_{w}_{s}")
    d = run_dir(rid)
    write_json_atomic(d / "search_path.json", {
        "path": list(path),
        "chunk_frames": cf,
        "search_solved": bool(solved),
        "replay_verified": replay_verified,
        "replay_reason": replay_reason,
    })
    try:
        sheet = make_contact_sheet(w, s, 0, path, cf, d / "contact_sheet.png", cols=6, rows=5)
        print("contact sheet:", sheet)
    except Exception as e:
        print("WARN sheet:", e)
    if solved:
        SOL.mkdir(parents=True, exist_ok=True)
        write_json_atomic(SOL / f"{w}-{s}.json",
                          {"path": list(path),
                           "solved": replay_verified,
                           "replay_verified": replay_verified,
                           "search_claimed_solved": True,
                           "invalid_reason": (
                               None if replay_verified
                               else f"replay_{replay_reason}"),
                           "replay_seed": 0,
                           "by": "go-explore",
                           "chunk_frames": cf})
        print(f"saved replay-gated result data/solutions/{w}-{s}.json")


if __name__ == "__main__":
    main()
