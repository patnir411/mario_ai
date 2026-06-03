"""The artifact contract is enforced: valid results validate, invalid ones fail."""
from __future__ import annotations

import pytest
from jsonschema import ValidationError

from mario.artifacts import build_result, framerule_time, validate_result


def _outcome(**over):
    base = dict(beat_level=True, death=False, death_cause=None, x_pos_reached=3161,
                x_pos_max=3161, level_length=3161, completion_frac=1.0, frames=1184,
                framerule_time=57, deaths=0)
    base.update(over)
    return base


def test_build_result_is_valid():
    r = build_result(run_id="20260603-000000-test", kind="search", milestone="V0",
                     world=1, stage=1, seed=0, outcome=_outcome(),
                     search={"beam_width": 256}, artifacts={"contact_sheet": "contact_sheet.png"},
                     passed=True, pass_reason="beat_level && deaths==0")
    validate_result(r)  # should not raise


def test_missing_required_outcome_field_fails():
    r = build_result(run_id="x", kind="search", milestone="V0", world=1, stage=1, seed=0,
                     outcome=_outcome(), passed=True, pass_reason="ok")
    del r["outcome"]["beat_level"]
    with pytest.raises(ValidationError):
        validate_result(r)


def test_completion_frac_out_of_range_fails():
    with pytest.raises(ValidationError):
        validate_result(build_result(
            run_id="x", kind="search", milestone="V0", world=1, stage=1, seed=0,
            outcome=_outcome(completion_frac=1.5), passed=False, pass_reason="bad"))


def test_bad_kind_fails():
    with pytest.raises(ValidationError):
        validate_result(build_result(
            run_id="x", kind="not_a_kind", milestone="V0", world=1, stage=1, seed=0,
            outcome=_outcome(), passed=False, pass_reason="bad"))


def test_framerule_time_rounds_up():
    assert framerule_time(1) == 1
    assert framerule_time(21) == 1
    assert framerule_time(22) == 2
    assert framerule_time(0) == 0
