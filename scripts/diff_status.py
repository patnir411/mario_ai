"""Regression gate: compare runs/status.json against runs/status.prev.json.

Run after update_status.py. Exits non-zero if any per-level best regressed beyond
tolerance, so the iteration loop can catch silent backsliding (see plan §5).

    ./venv/bin/python scripts/diff_status.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

COMPLETION_TOL = 0.02     # a drop larger than this is a regression
FRAMERULE_TOL = 1         # mid-level framerule increase larger than this is a regression


def main() -> int:
    cur_p, prev_p = RUNS / "status.json", RUNS / "status.prev.json"
    if not cur_p.exists():
        print("no status.json yet — run update_status.py first")
        return 0
    if not prev_p.exists():
        print("no previous status to diff against (first reconcile)")
        return 0

    cur = json.loads(cur_p.read_text()).get("current_best", {})
    prev = json.loads(prev_p.read_text()).get("current_best", {})
    regressions, improvements = [], []

    for lvl, pv in prev.items():
        cv = cur.get(lvl)
        if cv is None:
            continue
        if pv.get("beat") and not cv.get("beat"):
            regressions.append(f"{lvl}: lost a beating solution (was beat, now not)")
            continue
        dc = cv.get("completion_frac", 0) - pv.get("completion_frac", 0)
        if dc < -COMPLETION_TOL:
            regressions.append(f"{lvl}: completion {pv['completion_frac']:.2f} -> "
                               f"{cv['completion_frac']:.2f} ({dc:+.2f})")
        elif dc > COMPLETION_TOL:
            improvements.append(f"{lvl}: completion {dc:+.2f}")
        if pv.get("beat") and cv.get("beat") and lvl != "8-4":
            df = (cv.get("framerule_time") or 0) - (pv.get("framerule_time") or 0)
            if df > FRAMERULE_TOL:
                regressions.append(f"{lvl}: framerule {pv['framerule_time']} -> "
                                   f"{cv['framerule_time']} (+{df}) mid-level")
            elif df < 0:
                improvements.append(f"{lvl}: framerule {df}")

    for i in improvements:
        print("IMPROVED:", i)
    for r in regressions:
        print("REGRESSION:", r, file=sys.stderr)
    if not regressions and not improvements:
        print("no changes in per-level bests")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
