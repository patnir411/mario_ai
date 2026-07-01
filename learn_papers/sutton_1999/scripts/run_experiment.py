#!/usr/bin/env python3
"""Generic experiment runner: config → train → eval → summary.json.

    ./venv/bin/python learn_papers/sutton_1999/scripts/run_experiment.py \
        --config learn_papers/sutton_1999/configs/exp01_reinforce.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from mario_pg.config import load_config  # noqa: E402

TRAINERS = {
    "sanity": "scripts/00_sanity_rollout.py",
    "reinforce": "scripts/train_reinforce.py",
    "actor_critic": "scripts/train_actor_critic.py",
    "oracle_ac": "scripts/train_actor_critic.py",  # oracle mode via cfg.baseline
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    args = ap.parse_args()
    cfg = load_config(args.config)
    target = TRAINERS.get(cfg.trainer)
    if target is None:
        raise SystemExit(f"Unknown trainer: {cfg.trainer}")
    script = LAB / target
    if not script.exists():
        print(f"Trainer {cfg.trainer} → {script} not implemented yet (Phase 1+)")
        print(f"Config loaded OK: exp_id={cfg.exp_id}")
        raise SystemExit(2)
    print(f"Dispatch to {script} — implement in Phase 1")


if __name__ == "__main__":
    main()
