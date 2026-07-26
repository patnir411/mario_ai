"""Reproducibility and integrity contracts for the experimental RC-RL driver."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from mario.solution_verification import SolutionVerification
from scripts import rc_rl


def _verification(
        path,
        *,
        level="1-1",
        replay_verified=True,
        status="verified",
        reason="seed0_flag",
        chunk_frames=6,
        path_length=3,
):
    return SolutionVerification(
        file=str(path),
        level=level,
        claimed_solved=True,
        replay_verified=replay_verified,
        status=status,
        reason=reason,
        chunk_frames=chunk_frames,
        path_length=path_length,
    )


def test_verified_demo_uses_manifest_chunk_frames_and_seed(tmp_path):
    manifest = tmp_path / "1-1.json"
    manifest.write_text(json.dumps({
        "solved": True,
        "path": [1, 2, 3],
        "chunk_frames": 6,
    }))
    calls = []

    def verifier(path, *, seed):
        calls.append((path, seed))
        return _verification(path)

    path, chunk_frames, check, raw = rc_rl.load_verified_demo(
        manifest, "1-1", 17, verifier=verifier)

    assert calls == [(manifest, 17)]
    assert path == [1, 2, 3]
    assert chunk_frames == 6
    assert check.replay_verified
    assert raw["chunk_frames"] == 6


@pytest.mark.parametrize(
    ("check", "message"),
    [
        (_verification("x", level="2-1"), "does not match"),
        (
            _verification(
                "x",
                replay_verified=False,
                status="replay_failed",
                reason="seed0_no_flag",
            ),
            "refusing unverified demonstration",
        ),
    ],
)
def test_demo_rejects_wrong_level_or_failed_replay(tmp_path, check, message):
    manifest = tmp_path / "1-1.json"
    manifest.write_text(json.dumps({
        "solved": True,
        "path": [1, 2, 3],
        "chunk_frames": 6,
    }))

    with pytest.raises(ValueError, match=message):
        rc_rl.load_verified_demo(
            manifest,
            "1-1",
            0,
            verifier=lambda *_args, **_kwargs: check,
        )


class _FakeSnapshotSim:
    def __init__(self):
        self.t = -1
        self.last_info = {}
        self.last_obs = None
        self.restore_call = None

    def reset(self, seed):
        self.t = 0
        self.last_info = {"t": 0, "seed": seed}
        self.last_obs = np.asarray([0, seed])

    def snapshot(self):
        return f"snapshot-{self.t}"

    def run_chunk(self, action, chunk_frames):
        self.t += 1
        self.last_info = {
            "t": self.t,
            "action": action,
            "chunk_frames": chunk_frames,
        }
        self.last_obs = np.asarray([self.t, action])
        return dict(self.last_info), False

    def restore(self, snapshot, *, cached_info, cached_obs):
        self.restore_call = (snapshot, cached_info, cached_obs)


def test_frontier_snapshots_preserve_wrapper_caches():
    sim = _FakeSnapshotSim()

    states = rc_rl.capture_frontier_states(
        sim,
        path=[4, 5, 6],
        chunk_frames=7,
        frontiers=[2, 0],
        seed=13,
    )

    assert states[0].emulator_state == "snapshot-0"
    assert states[0].info == {"t": 0, "seed": 13}
    assert states[0].obs.tolist() == [0, 13]
    assert states[2].emulator_state == "snapshot-2"
    assert states[2].info["action"] == 5

    rc_rl.restore_frontier(sim, states[0])
    snapshot, info, obs = sim.restore_call
    assert snapshot == "snapshot-0"
    assert info == {"t": 0, "seed": 13}
    assert obs.tolist() == [0, 13]


def test_candidate_path_uses_shared_verifier_and_declared_chunk_frames():
    observed = {}

    def verifier(path, *, seed):
        manifest = json.loads(path.read_text())
        observed.update(manifest)
        observed["seed"] = seed
        observed["name"] = path.name
        return _verification(
            path,
            level="3-2",
            chunk_frames=4,
            path_length=2,
        )

    check = rc_rl.verify_candidate_path(
        3, 2, [7, 8], 4, 23, verifier=verifier)

    assert check.replay_verified
    assert observed == {
        "solved": True,
        "path": [7, 8],
        "chunk_frames": 4,
        "seed": 23,
        "name": "3-2.json",
    }


def test_seed_helper_controls_python_numpy_and_torch():
    rc_rl.set_all_seeds(91)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )
    rc_rl.set_all_seeds(91)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )

    assert second == first


def test_explicit_artifact_paths_are_required():
    parser = rc_rl.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--level", "1-1"])


def test_clipped_double_q_requires_at_least_two_critics(tmp_path):
    parser = rc_rl.build_parser()
    args = parser.parse_args([
        "--level", "1-1",
        "--checkpoint-out", str(tmp_path / "checkpoint.pt"),
        "--report", str(tmp_path / "report.json"),
        "--critics", "1",
    ])

    with pytest.raises(ValueError, match="at least 2"):
        rc_rl.validate_args(args)
