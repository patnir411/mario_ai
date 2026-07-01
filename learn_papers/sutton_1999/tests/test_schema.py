"""Tests for Sutton 1999 lab infrastructure."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

LAB = Path(__file__).resolve().parent.parent


def test_all_configs_parse():
    for cfg_path in (LAB / "configs").glob("*.yaml"):
        raw = yaml.safe_load(cfg_path.read_text())
        assert "exp_id" in raw
        assert raw["exp_id"] in cfg_path.stem


def test_config_loader():
    sys_path = str(LAB)
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from mario_pg.config import load_config

    cfg = load_config(LAB / "configs" / "exp01_reinforce.yaml")
    assert cfg.exp_id == "exp01_reinforce"
    assert cfg.trainer == "reinforce"
    assert cfg.world == 1 and cfg.stage == 1


def test_schema_episodes_columns():
    """Document required episodes.csv columns (full validation once logging exists)."""
    required = {
        "episode", "return", "return_disc", "length", "beat", "died",
        "x_max", "completion_frac", "entropy_mean", "grad_norm", "grad_var",
        "loss", "wall_sec",
    }
    schema_doc = (LAB / "artifacts" / "SCHEMA.md").read_text()
    for col in required:
        assert col in schema_doc
