"""One-command per-iteration verification loop (plan §4).

Runs the gates in order and prints a single PASS/FAIL summary so Claude reads one block:
  0. determinism + snapshot gate   (RED => STOP, nothing downstream is trustworthy)
  1. layer invariants              (reward / artifacts schema)
  2. record tests -> runs/tests.json
  3. reconcile status (update_status.py) + regression diff (diff_status.py)

Does NOT run milestone search/training (those are long; run them explicitly). This is
the fast gate to run after any code change.

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


def _pytest(paths: list[str]) -> tuple[int, int, int, str]:
    # NB: do not add another -q here — pyproject addopts already has -q; a second
    # one becomes -qq and suppresses the "N passed" summary line we parse.
    code, out = _run([PY, "-m", "pytest", *paths], ENV)
    import re
    passed = failed = 0
    m = re.search(r"(\d+) passed", out)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", out)
    if m:
        failed = int(m.group(1))
    return code, passed, failed, out


def main() -> int:
    print("== STEP 0: determinism + snapshot gate ==")
    code0, p0, f0, out0 = _pytest(["tests/test_determinism.py", "tests/test_snapshot.py"])
    print(out0.strip().splitlines()[-1] if out0.strip() else "")
    if code0 != 0:
        print("\nFAIL: determinism/snapshot gate RED — STOP. Search results are invalid.")
        return 1

    print("== STEP 1: layer invariants ==")
    code1, p1, f1, out1 = _pytest(["tests/test_reward.py", "tests/test_artifacts_schema.py",
                                   "tests/test_observation.py", "tests/test_label.py",
                                   "tests/test_buffer.py", "tests/test_eval_schema.py"])
    print(out1.strip().splitlines()[-1] if out1.strip() else "")

    total_pass, total_fail = p0 + p1, f0 + f1
    write_json_atomic(ROOT / "runs" / "tests.json",
                      {"last_run": utc_now_iso(), "passed": total_pass,
                       "failed": total_fail,
                       "determinism": "green" if code0 == 0 else "red"})

    print("== STEP 2: reconcile status ==")
    _run([PY, "scripts/update_status.py"], ENV)
    code3, out3 = _run([PY, "scripts/diff_status.py"], ENV)
    print(out3.strip())

    ok = code0 == 0 and code1 == 0 and code3 == 0
    print(f"\n{'PASS' if ok else 'FAIL'}: tests {total_pass} passed / {total_fail} failed; "
          f"regressions={'none' if code3 == 0 else 'SEE ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
