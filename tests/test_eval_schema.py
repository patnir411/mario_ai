"""The eval artifact contract is enforced."""
from __future__ import annotations

import pytest
from jsonschema import ValidationError

from mario.artifacts import validate_eval


def _eval(**over):
    ev = {
        "run_id": "x", "schema_version": 1, "kind": "eval", "milestone": "V2",
        "git_rev": "abc", "env_fingerprint": {},
        "policy": {"checkpoint_run_id": "r", "K": 4, "chunk_frames": 8},
        "levels": {"1-1": {"completion_rate": 0.95, "n_beat": 19, "n_rollouts": 20,
                           "median_framerule_time": 58, "median_completion_frac": 1.0,
                           "deaths_by_cause": {"pit": 1}}},
        "aggregate": {"mean_completion_rate": 0.95, "total_beat": 19, "total_rollouts": 20},
        "pass": True, "pass_reason": "ok",
    }
    ev.update(over)
    return ev


def test_valid_eval_passes():
    validate_eval(_eval())


def test_bad_kind_fails():
    with pytest.raises(ValidationError):
        validate_eval(_eval(kind="rollout"))


def test_completion_rate_out_of_range_fails():
    ev = _eval()
    ev["levels"]["1-1"]["completion_rate"] = 1.5
    with pytest.raises(ValidationError):
        validate_eval(ev)


def test_missing_aggregate_fails():
    ev = _eval()
    del ev["aggregate"]
    with pytest.raises(ValidationError):
        validate_eval(ev)
