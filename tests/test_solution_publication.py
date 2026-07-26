from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import goexplore_level, solve_beam, solve_coverage


def _search_result():
    return SimpleNamespace(
        solved=True,
        path=[1, 2],
        chunk_frames=8,
        x_max=123,
        nodes_expanded=9,
        wall_clock_s=0.1,
    )


def test_beam_cannot_publish_search_only_success(tmp_path, monkeypatch):
    monkeypatch.setattr(solve_beam, "SOL", tmp_path)
    monkeypatch.setattr(solve_beam, "beam_search", lambda *_a, **_kw: _search_result())
    monkeypatch.setattr(
        solve_beam, "replay_verify_stock_path",
        lambda *_a, **_kw: (False, "seed0_no_flag"))
    monkeypatch.setattr(solve_beam.sys, "argv", ["solve_beam.py", "1", "1"])

    solve_beam.main()

    manifest = json.loads((tmp_path / "1-1.json").read_text())
    assert manifest["search_claimed_solved"] is True
    assert manifest["solved"] is False
    assert manifest["replay_verified"] is False


def test_coverage_cannot_publish_search_only_success(tmp_path, monkeypatch):
    solutions = tmp_path / "solutions"
    run = tmp_path / "run"
    monkeypatch.setattr(solve_coverage, "SOL", solutions)
    monkeypatch.setattr(
        solve_coverage, "coverage_search", lambda *_a, **_kw: _search_result())
    monkeypatch.setattr(
        solve_coverage, "replay_verify_stock_path",
        lambda *_a, **_kw: (False, "seed0_died_without_flag"))
    monkeypatch.setattr(solve_coverage, "new_run_id", lambda *_a: "attempt")
    monkeypatch.setattr(solve_coverage, "run_dir", lambda *_a: run)
    monkeypatch.setattr(solve_coverage, "make_contact_sheet", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        solve_coverage.sys, "argv", ["solve_coverage.py", "1", "1"])

    solve_coverage.main()

    manifest = json.loads((solutions / "1-1.json").read_text())
    assert manifest["search_claimed_solved"] is True
    assert manifest["solved"] is False
    assert manifest["replay_verified"] is False


def test_goexplore_cannot_publish_search_only_success(tmp_path, monkeypatch):
    solutions = tmp_path / "solutions"
    run = tmp_path / "run"
    monkeypatch.setattr(goexplore_level, "SOL", solutions)
    monkeypatch.setattr(
        goexplore_level, "go_explore",
        lambda *_a, **_kw: (True, [1, 2], {"cells": 3}))
    monkeypatch.setattr(
        goexplore_level, "replay_verify_stock_path",
        lambda *_a, **_kw: (False, "seed0_no_flag"))
    monkeypatch.setattr(goexplore_level, "new_run_id", lambda *_a: "attempt")
    monkeypatch.setattr(goexplore_level, "run_dir", lambda *_a: run)
    monkeypatch.setattr(goexplore_level, "make_contact_sheet", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        goexplore_level.sys, "argv", ["goexplore_level.py", "1", "1"])

    goexplore_level.main()

    manifest = json.loads((solutions / "1-1.json").read_text())
    assert manifest["search_claimed_solved"] is True
    assert manifest["solved"] is False
    assert manifest["replay_verified"] is False
