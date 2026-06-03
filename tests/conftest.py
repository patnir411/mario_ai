"""Shared fixtures + helpers for the test suite.

Determinism/snapshot tests are intentionally torch-free (pure emulator + RAM) so that
MPS nondeterminism can never make them flaky. See DESIGN.md §2 and the plan.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mario.env import MarioSim

GOLDEN = Path(__file__).parent / "golden" / "ram_hashes.json"


def canonical_actions(n: int = 200) -> list[int]:
    """A fixed, RNG-free scripted input sequence used by determinism tests.

    Pattern over each 20-step window: run right (right+B), a few run-jumps
    (right+A+B), then a couple NOOPs to land. Exercises movement, jumps, and
    landing without any randomness so the resulting RAM is fully reproducible.
    """
    seq = []
    for i in range(n):
        phase = i % 20
        if phase < 14:
            seq.append(3)      # right+B
        elif phase < 17:
            seq.append(4)      # right+A+B
        else:
            seq.append(0)      # NOOP
    return seq


def ram_hash(sim: MarioSim) -> str:
    return hashlib.sha256(sim.ram.tobytes()).hexdigest()


def run_sequence(sim: MarioSim, actions: list[int], checkpoints=None):
    """Step `actions` one frame each, stopping early if the episode ends.

    Returns (final_hash, {step: hash at checkpoints}, done_at, done). Stopping on
    `done` is deterministic (Mario dies at the same frame every run), so all four
    values are reproducible and suitable for a regression golden.
    """
    checkpoints = set(checkpoints or [])
    marks = {}
    if 0 in checkpoints:
        marks[0] = ram_hash(sim)
    done = False
    done_at = 0
    for i, a in enumerate(actions, start=1):
        _obs, _info, done = sim.step(a)
        done_at = i
        if i in checkpoints:
            marks[i] = ram_hash(sim)
        if done:
            break
    return ram_hash(sim), marks, done_at, done


@pytest.fixture
def sim():
    s = MarioSim(world=1, stage=1)
    s.reset(seed=0)
    yield s
    s.close()
