#!/usr/bin/env python3
"""Measure cos(∇ρ_true, ∇ρ_est) for compatibility ablation (Exp 05, 07)."""
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
    print(f"measure_grad_alignment: n_states={cfg.n_states_align}")
    print("Not implemented — build mario_pg/grad_align.py + oracle.py first")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
