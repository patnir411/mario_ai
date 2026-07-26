"""Replay and inventory all 32 committed SMB1 stock-level manifests.

This is the canonical command behind the ``30/32 replay-verified`` headline:

    ./venv/bin/python scripts/verify_stock_solutions.py --expect-verified 30

Negative manifests are reported but are not errors.  A positive ``solved`` claim
that fails replay, a missing/invalid manifest, or an unexpected verified count
returns nonzero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.solution_verification import verify_stock_solution_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solutions-dir", type=Path,
        default=ROOT / "data" / "solutions")
    parser.add_argument("--expect-verified", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    checks = verify_stock_solution_set(args.solutions_dir, seed=0)
    verified = [check.level for check in checks if check.replay_verified]
    unsolved = [check.level for check in checks if check.status == "unsolved"]
    bad = [check for check in checks
           if check.status in {"missing", "invalid", "replay_failed"}]
    report = {
        "seed": 0,
        "verified_count": len(verified),
        "total": len(checks),
        "verified": verified,
        "unsolved": unsolved,
        "failures": [check.to_json() for check in bad],
        "checks": [check.to_json() for check in checks],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"replay-verified: {len(verified)}/{len(checks)}")
    print(f"verified levels: {', '.join(verified) or '(none)'}")
    print(f"unsolved levels: {', '.join(unsolved) or '(none)'}")
    for check in bad:
        print(
            f"FAIL {check.level}: {check.status} ({check.reason})",
            file=sys.stderr)

    if bad:
        return 1
    if args.expect_verified is not None and len(verified) != args.expect_verified:
        print(
            f"FAIL expected {args.expect_verified} verified, got {len(verified)}",
            file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
