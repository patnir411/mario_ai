from __future__ import annotations

import json

from mario.solution_verification import replay_verify_stock_path, verify_stock_solution


def _write(tmp_path, data, name="1-1.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_positive_solution_requires_successful_replay(tmp_path):
    path = _write(tmp_path, {
        "solved": True,
        "path": [1, 2, 3],
        "chunk_frames": 8,
    })
    calls = []

    def no_flag(world, stage, seed, actions, chunk_frames):
        calls.append((world, stage, seed, actions, chunk_frames))
        return [{"flag": False, "died": False}]

    check = verify_stock_solution(path, replay_fn=no_flag)
    assert calls == [(1, 1, 0, [1, 2, 3], 8)]
    assert check.claimed_solved
    assert not check.replay_verified
    assert check.status == "replay_failed"
    assert check.reason == "seed0_no_flag"


def test_successful_replay_accepts_positive_solution(tmp_path):
    path = _write(tmp_path, {
        "solved": True,
        "path": [4],
        "chunk_frames": 1,
    }, "8-4.json")

    check = verify_stock_solution(
        path,
        replay_fn=lambda *_args: [{"flag": True, "died": False}])
    assert check.level == "8-4"
    assert check.replay_verified
    assert check.status == "verified"


def test_negative_manifest_is_retained_without_replay(tmp_path):
    path = _write(tmp_path, {
        "solved": False,
        "path": [2],
        "chunk_frames": 8,
        "invalid_reason": "quarantined",
    }, "6-3.json")

    def should_not_run(*_args):
        raise AssertionError("negative manifests must not be replayed as positives")

    check = verify_stock_solution(path, replay_fn=should_not_run)
    assert check.status == "unsolved"
    assert check.reason == "quarantined"
    assert not check.replay_verified


def test_malformed_positive_manifest_is_invalid(tmp_path):
    path = _write(tmp_path, {
        "solved": True,
        "path": ["RIGHT"],
        "chunk_frames": 8,
    })
    check = verify_stock_solution(path, replay_fn=lambda *_args: [])
    assert check.status == "invalid"
    assert check.reason == "path_must_be_integer_list"


def test_solved_flag_must_be_literal_boolean(tmp_path):
    path = _write(tmp_path, {
        "solved": "false",
        "path": [1],
        "chunk_frames": 8,
    })
    check = verify_stock_solution(path, replay_fn=lambda *_args: [])
    assert check.status == "invalid"
    assert check.reason == "solved_must_be_boolean"


def test_chunk_frames_must_be_literal_positive_integer(tmp_path):
    for invalid in (True, 1.5, "8", 0, -1):
        path = _write(tmp_path, {
            "solved": True,
            "path": [1],
            "chunk_frames": invalid,
        })
        check = verify_stock_solution(path, replay_fn=lambda *_args: [])
        assert check.status == "invalid"
        assert check.reason == "chunk_frames_must_be_positive"


def test_out_of_range_action_is_invalid_before_replay(tmp_path):
    path = _write(tmp_path, {
        "solved": True,
        "path": [-1, 9],
        "chunk_frames": 8,
    })

    check = verify_stock_solution(
        path,
        replay_fn=lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid paths must not reach the emulator")))

    assert check.status == "invalid"
    assert check.reason == "action_index_out_of_range_0_8"


def test_search_candidate_gate_replays_before_publication():
    calls = []

    def reaches_flag(world, stage, seed, actions, chunk_frames):
        calls.append((world, stage, seed, actions, chunk_frames))
        return [{"flag": True, "died": False}]

    verified, reason = replay_verify_stock_path(
        3, 2, [1, 4], 8, seed=0, replay_fn=reaches_flag)

    assert verified
    assert reason == "seed0_flag"
    assert calls == [(3, 2, 0, [1, 4], 8)]


def test_search_candidate_gate_rejects_empty_path_without_emulator():
    verified, reason = replay_verify_stock_path(
        3, 2, [], 8,
        replay_fn=lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid candidates must not reach the emulator")))

    assert not verified
    assert reason == "empty_path"
