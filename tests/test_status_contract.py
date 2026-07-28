from __future__ import annotations

from scripts import diff_status, update_status


def test_source_controlled_baseline_survives_without_ignored_runs(
        tmp_path, monkeypatch):
    monkeypatch.setattr(update_status, "RUNS", tmp_path)
    status = update_status.build_status([], [])

    assert status["milestones"]["V0"]["status"] == "done"
    assert status["milestones"]["V2"]["best_run"] == "20260603-203756-v2_eval"
    assert status["current_best"]["1-1"]["beat"] is True
    assert status["tests"]["passed_gate"] is True
    assert status["source_rev_at_generation"] == status["git_rev"]
    assert "source at generation" in update_status.render_block(status)


def test_v2_cannot_be_completed_by_an_eval_that_omits_1_1():
    empty = {
        "milestones": {
            milestone: {"status": "todo"}
            for milestone in update_status.MILESTONES
        }
    }
    irrelevant_eval = {
        "milestone": "V2",
        "run_id": "wrong-level",
        "pass": True,
        "levels": {"1-2": {"completion_rate": 1.0}},
    }

    status = update_status.build_status([], [irrelevant_eval], baseline=empty)

    assert status["milestones"]["V2"]["status"] == "todo"


def test_disappearing_current_best_is_a_regression():
    regressions, improvements = diff_status.compare_current_best(
        {}, {"1-1": {"beat": True, "completion_frac": 1.0}})

    assert improvements == []
    assert regressions == ["1-1: evidence disappeared from current_best"]


def test_failed_full_suite_is_complete_but_not_passing(tmp_path, monkeypatch):
    monkeypatch.setattr(update_status, "RUNS", tmp_path)
    monkeypatch.setattr(update_status, "BASELINE", tmp_path / "missing.json")
    # The gate-report semantics are tested through its small private producer;
    # writing into tmp_path avoids touching the repository's ignored run ledger.
    from scripts import verify_iteration
    monkeypatch.setattr(verify_iteration, "ROOT", tmp_path)

    report = verify_iteration._record_tests(
        {"passed": 2, "failed": 0, "skipped": 0, "xfailed": 0,
         "xpassed": 0, "errors": 0},
        {"passed": 3, "failed": 1, "skipped": 0, "xfailed": 0,
         "xpassed": 0, "errors": 0},
        determinism="green",
    )

    assert report["suite_complete"] is True
    assert report["passed_gate"] is False
    assert report["complete"] is False
