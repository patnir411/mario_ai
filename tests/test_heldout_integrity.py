"""Held-out integrity: the generalist trainer must REFUSE to load a held-out level.

The decisive-experiment guarantee — a level listed in `holdout_levels` may never enter the
training set. `train_generalist.load_levels` asserts this; this test locks that assertion so
a future split edit can't silently leak the test levels into training.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "data" / "entity_shards"


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_generalist", ROOT / "scripts" / "train_generalist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_holdout_overlap_raises():
    mod = _load_train_module()
    # a train list that overlaps the holdout must trip the integrity assertion
    with pytest.raises(AssertionError):
        mod.load_levels(["1-1", "3-1"], holdout={"3-1"})


def test_missing_train_shards_raise_actionable_error(tmp_path, monkeypatch):
    mod = _load_train_module()
    monkeypatch.setattr(mod, "SHARDS", tmp_path)
    with pytest.raises(FileNotFoundError, match="gen_entity_dataset"):
        mod.load_levels(["1-1"], holdout=set())


def test_disjoint_split_ok_if_shards_exist():
    """If any shard exists, a disjoint split loads without raising (smoke)."""
    mod = _load_train_module()
    have = sorted(p.stem.replace("level-", "") for p in SHARDS.glob("level-*.npz"))
    if not have:
        pytest.skip("no entity shards built yet")
    train = have[:1]
    holdout = {have[-1]} if len(have) > 1 else set()
    data = mod.load_levels(train, holdout)  # must not raise
    assert len(data["obs"]) >= 0
