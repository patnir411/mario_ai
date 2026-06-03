"""Determinism gate: same seed + same inputs -> identical RAM, always.

If this is RED, search is meaningless (a snapshot you restore wouldn't replay
the same future). This is the highest-priority invariant in the project.
"""
from __future__ import annotations

import json

from conftest import GOLDEN, canonical_actions, run_sequence
from mario.env import MarioSim

CHECKPOINTS = [0, 50, 100, 150, 200]


def _fresh_run():
    sim = MarioSim(world=1, stage=1)
    sim.reset(seed=0)
    try:
        return run_sequence(sim, canonical_actions(200), CHECKPOINTS)
    finally:
        sim.close()


def test_two_fresh_runs_identical():
    """Two independent envs, same seed + inputs, must reach bit-identical RAM."""
    final_a, marks_a, done_at_a, done_a = _fresh_run()
    final_b, marks_b, done_at_b, done_b = _fresh_run()
    assert (done_at_a, done_a) == (done_at_b, done_b), "episode ended at different frames"
    assert final_a == final_b, "final RAM differs across identical runs"
    assert marks_a == marks_b, f"per-checkpoint RAM diverged: {marks_a} vs {marks_b}"


def test_golden_regression():
    """Per-checkpoint RAM hashes match the committed golden (catches ROM/version drift)."""
    assert GOLDEN.exists(), (
        "golden missing — generate with scripts/gen_golden.py and commit it"
    )
    golden = json.loads(GOLDEN.read_text())
    final, marks, done_at, _done = _fresh_run()
    marks = {str(k): v for k, v in marks.items()}
    assert marks == golden["checkpoints"], (
        "RAM hashes diverged from golden — emulator/ROM/version changed, or determinism broke"
    )
    assert final == golden["final"], "final RAM hash diverged from golden"
    assert done_at == golden["done_at"], "episode end frame diverged from golden"
