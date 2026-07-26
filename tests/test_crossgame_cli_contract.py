"""Artifact-promotion and terminal-replay contracts for cross-game CLIs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import replay_video_adapter, solve_sma4, solve_sma4_snapshot, solve_sml


PUBLISHERS = [
    solve_sma4._publish_attempt,
    solve_sma4_snapshot._publish_attempt,
    solve_sml._publish_attempt,
]


@pytest.mark.parametrize("publish", PUBLISHERS)
def test_failed_replay_stays_in_run_and_preserves_verified_solution(tmp_path, publish):
    run_dir = tmp_path / "runs" / "attempt"
    canonical = tmp_path / "data" / "solutions" / "1-1.json"
    canonical.parent.mkdir(parents=True)
    previous = {"solved": True, "replay_verified": True, "path": [7]}
    canonical.write_text(json.dumps(previous))

    artifact = publish(
        {
            "search_solved": True,
            "solved": True,
            "replay_verified": True,
            "replay_root": {"kind": "fresh_boot"},
            "path": [1],
        },
        run_dir,
        canonical,
        replay_verified=False,
    )

    assert artifact["solved"] is False
    assert artifact["replay_verified"] is False
    assert json.loads((run_dir / "attempt_summary.json").read_text()) == artifact
    assert json.loads(canonical.read_text()) == previous


@pytest.mark.parametrize("publish", PUBLISHERS)
def test_verified_replay_is_promoted_with_truthful_flags(tmp_path, publish):
    run_dir = tmp_path / "runs" / "attempt"
    canonical = tmp_path / "data" / "solutions" / "1-1.json"

    artifact = publish(
        {
            "search_solved": False,
            "solved": False,
            "replay_verified": False,
            "replay_root": {"kind": "fresh_boot"},
            "path": [1],
        },
        run_dir,
        canonical,
        replay_verified=True,
    )

    assert artifact["solved"] is True
    assert artifact["replay_verified"] is True
    assert json.loads(canonical.read_text()) == artifact


@pytest.mark.parametrize("publish", PUBLISHERS)
def test_replay_without_declared_root_cannot_be_promoted(tmp_path, publish):
    run_dir = tmp_path / "runs" / "attempt"
    canonical = tmp_path / "data" / "solutions" / "1-1.json"

    artifact = publish(
        {"search_solved": True, "path": [1]},
        run_dir,
        canonical,
        replay_verified=True,
    )

    assert artifact["solved"] is False
    assert artifact["replay_verified"] is False
    assert not canonical.exists()


class _SnapshotReplayAdapter:
    def __init__(self):
        self.restored = None
        self.actions: list[int] = []
        self._info = {"success": False, "death": False}

    def restore(self, root):
        self.restored = root
        self.actions.clear()
        self._info = {"success": False, "death": False}

    @property
    def last_info(self):
        return dict(self._info)

    def run_chunk(self, action: int, frames: int):
        del frames
        self.actions.append(action)
        self._info = {
            "success": action == 9,
            "death": action == 8,
        }
        return dict(self._info), False

    @staticmethod
    def is_success(info):
        return bool(info["success"])

    @staticmethod
    def is_death(info, done):
        return bool(done or info["death"])


def test_snapshot_verification_restores_declared_root_and_stops_at_success():
    adapter = _SnapshotReplayAdapter()
    root = ("entry-state", {"x_pos": 0})

    verified, final_info = solve_sma4_snapshot.replay_from_snapshot(
        adapter, root, [1, 9, 2], 6)

    assert verified is True
    assert final_info["success"] is True
    assert adapter.restored is root
    assert adapter.actions == [1, 9]


class _VideoReplayAdapter:
    def __init__(self, terminal_kind: str):
        self.terminal_kind = terminal_kind
        self.calls = 0
        self.seed = None
        self.last_obs = np.zeros((2, 2, 3), dtype=np.uint8)

    def reset(self, seed=0):
        self.seed = seed
        self.calls = 0
        return {"success": False, "death": False}

    def restore(self, root):
        self.seed = root
        self.calls = 0

    @property
    def last_info(self):
        return {"success": False, "death": False}

    def step(self, action):
        del action
        self.calls += 1
        self.last_obs = np.full((2, 2, 3), self.calls, dtype=np.uint8)
        info = {
            "success": self.terminal_kind == "success" and self.calls == 2,
            "death": self.terminal_kind == "death" and self.calls == 2,
        }
        return self.last_obs, info, False

    @staticmethod
    def is_success(info):
        return bool(info["success"])

    @staticmethod
    def is_death(info, done):
        return bool(done or info["death"])


@pytest.mark.parametrize("terminal_kind", ["success", "death"])
def test_video_replay_exits_all_remaining_actions_on_terminal(terminal_kind):
    adapter = _VideoReplayAdapter(terminal_kind)

    frames = replay_video_adapter.capture_replay_frames(
        adapter, [0, 1, 2], 4, seed=17)

    assert adapter.seed == 17
    assert adapter.calls == 2
    assert len(frames) == 3  # root plus the two executed primitive frames


def test_video_replay_can_start_from_declared_snapshot():
    adapter = _VideoReplayAdapter("success")

    frames = replay_video_adapter.capture_replay_frames(
        adapter, [0], 4, root_snapshot="entry-state")

    assert adapter.seed == "entry-state"
    assert adapter.calls == 2
    assert len(frames) == 3


def test_video_snapshot_rebases_life_loss_terminal_detector():
    class LifeAwareAdapter(_VideoReplayAdapter):
        def __init__(self):
            super().__init__("success")
            self._start_lives = 5

        @property
        def last_info(self):
            return {"success": False, "death": False, "lives": 3}

        def is_death(self, info, done):
            return bool(done or int(info["lives"]) < self._start_lives)

        def step(self, action):
            obs, info, done = super().step(action)
            info["lives"] = 3
            return obs, info, done

    adapter = LifeAwareAdapter()
    frames = replay_video_adapter.capture_replay_frames(
        adapter, [0], 4, root_snapshot="after-life-loss")

    assert adapter._start_lives == 3
    assert adapter.calls == 2
    assert len(frames) == 3
