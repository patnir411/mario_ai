"""One-command per-iteration verification loop.

Runs the gates in order and prints a single PASS/FAIL summary so Claude reads one block:
  0. determinism + snapshot gate   (RED => STOP, nothing downstream is trustworthy)
  1. every other repository test  (the gate files are excluded to avoid double counting)
  2. record the full-suite result -> runs/tests.json
  3. reconcile status (update_status.py) + regression diff (diff_status.py)

Does NOT run milestone search/training (those are long; run them explicitly). This is
the repository test gate to run after any code change.

    ./venv/bin/python scripts/verify_iteration.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / "venv" / "bin" / "python")
ENV = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}

from mario.io import utc_now_iso, write_json_atomic  # noqa: E402


def _run(cmd: list[str], env_extra=None) -> tuple[int, str]:
    import os
    env = {**os.environ, **(env_extra or {})}
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _pytest(args: list[str]) -> tuple[int, dict[str, int], str]:
    # NB: do not add another -q here — pyproject addopts already has -q; a second
    # one becomes -qq and suppresses the "N passed" summary line we parse.
    code, out = _run([PY, "-m", "pytest", *args], ENV)
    import re

    counts = {}
    for name in ("passed", "failed", "skipped", "xfailed", "xpassed"):
        match = re.search(rf"(\d+) {name}\b", out)
        counts[name] = int(match.group(1)) if match else 0
    match = re.search(r"(\d+) errors?\b", out)
    counts["errors"] = int(match.group(1)) if match else 0
    return code, counts, out


def _last_line(output: str) -> str:
    lines = output.strip().splitlines()
    return lines[-1] if lines else ""


def _record_tests(
    gate: dict[str, int],
    remainder: dict[str, int] | None,
    *,
    determinism: str,
) -> dict[str, object]:
    full_suite_complete = remainder is not None
    remainder = remainder or {name: 0 for name in gate}
    totals = {name: gate[name] + remainder[name] for name in gate}
    passed_gate = (
        determinism == "green"
        and full_suite_complete
        and totals.get("failed", 0) == 0
        and totals.get("errors", 0) == 0
    )
    report: dict[str, object] = {
        "last_run": utc_now_iso(),
        "scope": (
            "full repository suite (determinism/snapshot gate first)"
            if full_suite_complete
            else "determinism/snapshot gate only (remainder not run)"
        ),
        **totals,
        "determinism": determinism,
        "suite_complete": full_suite_complete,
        "passed_gate": passed_gate,
        # Backward-compatible field: complete means the complete suite passed.
        "complete": passed_gate,
    }
    write_json_atomic(ROOT / "runs" / "tests.json", report)
    return report


def main() -> int:
    print("== STEP 0: determinism + snapshot gate ==")
    code0, gate, out0 = _pytest(["tests/test_determinism.py", "tests/test_snapshot.py"])
    print(_last_line(out0))
    if code0 != 0:
        _record_tests(gate, None, determinism="red")
        print("\nFAIL: determinism/snapshot gate RED — STOP. Search results are invalid.")
        return 1

    print("== STEP 1: remaining repository suite ==")
    code1, remainder, out1 = _pytest([
        "--ignore=tests/test_determinism.py",
        "--ignore=tests/test_snapshot.py",
        "tests",
    ])
    print(_last_line(out1))
    report = _record_tests(gate, remainder, determinism="green")

    print("== STEP 2: reconcile status ==")
    code2, out2 = _run([PY, "scripts/update_status.py"], ENV)
    print(out2.strip())
    code3, out3 = _run([PY, "scripts/diff_status.py"], ENV)
    print(out3.strip())

    ok = code0 == 0 and code1 == 0 and code2 == 0 and code3 == 0
    print(f"\n{'PASS' if ok else 'FAIL'}: tests {report['passed']} passed / "
          f"{report['failed']} failed / {report['errors']} errors / "
          f"{report['skipped']} skipped; "
          f"regressions={'none' if code3 == 0 else 'SEE ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
