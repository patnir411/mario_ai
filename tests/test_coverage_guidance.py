from __future__ import annotations

import numpy as np
import pytest

import mario.search as search
from mario.env import N_ACTIONS
from mario.ram import AREA_NUMBER, AREA_POINTER, MARIO_LEVEL_PAGE, MARIO_X_ON_SCREEN, MARIO_Y_ON_SCREEN


class FakeSim:
    def __init__(self, world=1, stage=1, actions=None):
        self.world = world
        self.stage = stage
        self.ram = np.zeros(2048, dtype=np.uint8)
        self._last_info = {}

    def reset(self, seed=0):
        self._set_x(0)
        self._last_info = {"world": self.world, "stage": self.stage, "x_pos": 0,
                           "y_pos": 80, "flag_get": False}
        return self._last_info

    @property
    def last_info(self):
        return self._last_info

    def snapshot(self):
        return int(self._last_info["x_pos"])

    def restore(self, snap):
        self._set_x(int(snap))
        self._last_info = {"world": self.world, "stage": self.stage, "x_pos": int(snap),
                           "y_pos": 80, "flag_get": False}

    def run_chunk(self, action_idx: int, frames: int):
        x = (action_idx + 1) * 16
        self._set_x(x)
        self._last_info = {"world": self.world, "stage": self.stage, "x_pos": x,
                           "y_pos": 80, "flag_get": False}
        return self._last_info, False

    def close(self):
        pass

    def _set_x(self, x: int):
        self.ram[MARIO_LEVEL_PAGE] = x // 256
        self.ram[MARIO_X_ON_SCREEN] = x % 256
        self.ram[MARIO_Y_ON_SCREEN] = 80
        self.ram[AREA_NUMBER] = 0
        self.ram[AREA_POINTER] = 0


class DummyPrior:
    def __init__(self, ranking):
        self.ranking = ranking

    def logp(self, ram, info):
        scores = np.full(N_ACTIONS, -20.0, dtype=np.float32)
        for rank, action in enumerate(self.ranking):
            scores[action] = -float(rank)
        return scores


class DummyValue:
    def __init__(self, preferred_x):
        self.preferred_x = preferred_x
        self.calls = []

    def p(self, obs):
        x = int(obs[0])
        self.calls.append(x)
        return 1.0 if x == self.preferred_x else 0.0


def test_coverage_policy_prior_soft_ranks_without_pruning(monkeypatch):
    monkeypatch.setattr(search, "MarioSim", FakeSim)
    monkeypatch.setattr(search, "state_score", lambda *a, **kw: 0.0)

    prior = DummyPrior([5])
    result = search.coverage_search(
        1, 1, beam_width=1, max_depth=1, progress_every=0,
        cov_bonus=0.0, policy_prior=prior, policy_weight=100.0,
    )

    assert result.nodes_expanded == N_ACTIONS
    assert result.path == [5]


def test_coverage_policy_topk_is_explicit_pruning(monkeypatch):
    monkeypatch.setattr(search, "MarioSim", FakeSim)
    monkeypatch.setattr(search, "state_score", lambda *a, **kw: 0.0)

    prior = DummyPrior([5, 3])
    result = search.coverage_search(
        1, 1, beam_width=1, max_depth=1, progress_every=0,
        cov_bonus=0.0, policy_prior=prior, policy_weight=100.0, policy_topk=2,
    )

    assert result.nodes_expanded == 2
    assert result.path == [5]


def test_coverage_value_guide_participates_in_ranking(monkeypatch):
    import mario.observation as observation

    monkeypatch.setattr(search, "MarioSim", FakeSim)
    monkeypatch.setattr(search, "state_score", lambda *a, **kw: 0.0)
    monkeypatch.setattr(observation, "observe",
                        lambda ram, info: np.array([info["x_pos"]], dtype=np.float32))

    value = DummyValue(preferred_x=(4 + 1) * 16)
    result = search.coverage_search(
        1, 1, beam_width=1, max_depth=1, progress_every=0,
        cov_bonus=0.0, value_guide=value, value_weight=100.0,
    )

    assert value.calls
    assert result.path == [4]


@pytest.mark.parametrize("search_fn", [search.beam_search, search.coverage_search])
def test_policy_topk_must_be_positive(search_fn):
    with pytest.raises(ValueError, match="policy_topk"):
        search_fn(policy_topk=0)
