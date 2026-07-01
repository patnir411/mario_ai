#!/usr/bin/env python3
"""Run all experiment configs for a phase.

    ./venv/bin/python learn_papers/sutton_1999/scripts/run_phase.py --phase 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent

PHASE_CONFIGS = {
    0: ["exp00_sanity.yaml"],
    1: [
        "exp01_reinforce.yaml",
        "exp02_const_baseline.yaml",
        "exp03_learned_baseline.yaml",
        "exp04_advantage_vs_return.yaml",
    ],
    2: ["exp05_compatibility.yaml", "exp06_policy_iteration.yaml"],
    3: ["exp07_oracle_q_gradient.yaml", "exp08_oracle_advantage_ac.yaml"],
    4: [
        "exp09_pg_vs_imitation.yaml",
        "exp10_value_greedy_chatter.yaml",
        "exp11_multi_level.yaml",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=range(6))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configs = PHASE_CONFIGS.get(args.phase, [])
    if not configs:
        raise SystemExit(f"No configs for phase {args.phase}")
    runner = LAB / "scripts" / "run_experiment.py"
    for name in configs:
        cfg = LAB / "configs" / name
        cmd = [sys.executable, str(runner), "--config", str(cfg)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
