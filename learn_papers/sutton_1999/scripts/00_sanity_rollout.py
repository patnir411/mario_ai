#!/usr/bin/env python3
"""Phase 0 — random-policy rollout smoke test.

Validates reward signs, episode termination, and artifact writing before any PG training.

    ./venv/bin/python learn_papers/sutton_1999/scripts/00_sanity_rollout.py
    ./venv/bin/python learn_papers/sutton_1999/scripts/run_experiment.py --config configs/exp00_sanity.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from mario_pg.config import load_config  # noqa: E402

# Phase 0 implementation: mario_pg/rollout.py + mario_pg/logging.py
# Until implemented, this script documents the entry point and validates config load.


def main() -> None:
    cfg_path = LAB / "configs" / "exp00_sanity.yaml"
    cfg = load_config(cfg_path)
    print(f"exp_id={cfg.exp_id} trainer={cfg.trainer} episodes={cfg.episodes}")
    print("Phase 0 not yet implemented — see PHASES.md for rollout.py + logging.py")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
