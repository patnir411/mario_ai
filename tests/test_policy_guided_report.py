"""Artifact contracts for the matched policy-guided search benchmark."""
from __future__ import annotations

from types import SimpleNamespace

from scripts import policy_guided_search


def _result(*, solved=True):
    return SimpleNamespace(
        solved=solved,
        x_max=321,
        nodes_expanded=17,
        depth_reached=4,
        wall_clock_s=0.25,
        path=[1, 2, 3],
    )


def test_run_requires_independent_replay_for_solved_claim(monkeypatch):
    calls = []

    def fake_search(world, stage, **kwargs):
        calls.append((world, stage, kwargs))
        return _result()

    monkeypatch.setattr(policy_guided_search, "beam_search", fake_search)
    monkeypatch.setattr(
        policy_guided_search,
        "replay",
        lambda *_args: [{"flag": False}],
    )

    result = policy_guided_search.run(
        "beam",
        2,
        3,
        seed=7,
        chunk_frames=4,
        beam_width=6,
    )

    assert calls == [(
        2,
        3,
        {"seed": 7, "chunk_frames": 4, "beam_width": 6},
    )]
    assert result["search_solved"] is True
    assert result["solved"] is False
    assert result["replay_verified"] is False
    assert result["path"] == [1, 2, 3]


def test_run_records_replay_verified_trajectory(monkeypatch):
    monkeypatch.setattr(
        policy_guided_search,
        "coverage_search",
        lambda *_args, **_kwargs: _result(),
    )
    monkeypatch.setattr(
        policy_guided_search,
        "replay",
        lambda *_args: [{"flag": False}, {"flag": True}],
    )

    result = policy_guided_search.run(
        "coverage",
        1,
        1,
        seed=0,
        chunk_frames=8,
    )

    assert result["solved"] is True
    assert result["replay_verified"] is True
    assert result["path_length"] == 3
    assert result["chunk_frames"] == 8
