#!/usr/bin/env python3
"""REINFORCE trainer (Exp 01–04). Phase 1 deliverable.

See ARCHITECTURE.md § Training loops and EXPERIMENTS.md Phase 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

from mario_pg.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"train_reinforce: {cfg.exp_id} baseline={cfg.baseline}")
    print("Not implemented — build mario_pg/reinforce.py + rollout.py first")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
