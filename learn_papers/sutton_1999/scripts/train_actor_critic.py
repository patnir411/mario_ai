#!/usr/bin/env python3
"""Actor–critic trainer (Exp 05–06, 10). Phase 2 deliverable."""
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
    print(f"train_actor_critic: {cfg.exp_id} policy={cfg.policy}")
    print("Not implemented — build mario_pg/actor_critic.py first")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
