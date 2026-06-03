"""Reward invariants — encodes the Tom7 failure-mode lessons as executable checks.

If these go red, the search teacher will produce bad labels and everything downstream
(distillation, DAgger) inherits the rot. See DESIGN.md §6.
"""
from __future__ import annotations

from mario import reward
from mario.reward import DEFAULT, RewardWeights, state_score

X0 = 40  # nominal start x


def info(x=X0, flag=False, time=300, y=120, score=0, coins=0):
    return {"x_pos": x, "flag_get": flag, "time": time, "y_pos": y,
            "score": score, "coins": coins}


def test_death_is_dominant_negative():
    """A death must score far below any live forward-progress state."""
    alive_far = state_score(info(x=X0 + 500), X0)
    dead = state_score(info(x=X0 + 500), X0, died=True)
    assert dead < -DEFAULT.death / 2, "death penalty not dominant"
    assert dead < alive_far, "dying scored no worse than living"


def test_standing_still_non_positive():
    """No progress, alive, no flag -> score <= 0 (no fake reward for idling)."""
    assert state_score(info(x=X0), X0) <= 0.0


def test_moving_right_is_positive():
    assert state_score(info(x=X0 + 200), X0) > 0.0


def test_further_right_scores_higher():
    assert state_score(info(x=X0 + 400), X0) > state_score(info(x=X0 + 100), X0)


def test_flag_is_large_positive():
    s = state_score(info(x=X0 + 1000, flag=True), X0)
    assert s > DEFAULT.flag / 2, "reaching the flag must dominate"


def test_time_weight_zero_midlevel():
    """Default weights: a 1-frame difference must NOT change score (framerule lesson)."""
    a = state_score(info(x=X0 + 300), X0, frames=100)
    b = state_score(info(x=X0 + 300), X0, frames=101)
    assert a == b, "time is being penalized mid-level despite framerules"


def test_speed_phase_time_penalty_applies():
    """When the speed phase raises the time weight, more frames must score lower."""
    w = RewardWeights(time=1.0)
    fast = state_score(info(x=X0 + 300), X0, frames=100, w=w)
    slow = state_score(info(x=X0 + 300), X0, frames=200, w=w)
    assert fast > slow


def test_fake_progress_counters_ignored():
    """Score/coins must NOT affect reward — they are fake-progress traps."""
    base = state_score(info(x=X0 + 100), X0)
    juiced = state_score(info(x=X0 + 100, score=99999, coins=99), X0)
    assert base == juiced, "reward leaked from score/coins counters"


def test_is_death_logic():
    assert reward.is_death(info(flag=False), done=True) is True
    assert reward.is_death(info(flag=True), done=True) is False   # flag = success
    assert reward.is_death(info(flag=False), done=False) is False  # not done yet


def test_death_cause_timeout():
    assert reward.death_cause(info(time=0)) == "timeout"
