"""label_state invariants — the teacher signal that V1/V2/V3 inherit.

Validated empirically (see CLAUDE.md notes): soft targets are sharp at HAZARDS (fatal
actions annihilated) and permissive on open ground. These tests lock the sharp-at-hazard
property and the structural contract; they're self-contained (no committed winning path).
"""
from __future__ import annotations

import numpy as np

from mario.env import MarioSim, N_ACTIONS
from mario.label import label_state, soft_entropy


def _chunks_until_death(action=3, chunk_frames=8, cap=60):
    """Deterministic: how many `action` chunks from reset until the episode ends."""
    sim = MarioSim(1, 1)
    sim.reset(seed=0)
    n = 0
    for i in range(1, cap + 1):
        _info, done = sim.run_chunk(action, chunk_frames)
        n = i
        if done:
            break
    sim.close()
    return n


def test_soft_targets_well_formed():
    r = label_state(1, 1, [], depth=4, beam_width=4)
    assert r.soft_targets.shape == (N_ACTIONS,)
    assert abs(float(r.soft_targets.sum()) - 1.0) < 1e-5
    assert np.all(r.soft_targets >= 0)
    assert np.isfinite(r.value)
    assert 0 <= r.best_action < N_ACTIONS
    assert r.reachable.shape == (N_ACTIONS,) and r.reachable.dtype == bool


def test_fatal_action_is_suppressed_at_hazard():
    """Approaching the first Goomba by running, the teacher must give the doomed
    right+B (action 3) near-zero soft mass while a survivable escape keeps real mass.

    We label d-2 (the genuine decision point). d-1 is the all-actions-fatal frame
    (Mario already on top of the Goomba) where softmax is degenerately uniform — those
    unavoidable-death states are excluded from the dataset, not cloned.
    """
    d = _chunks_until_death(action=3)
    assert d >= 4, "expected to survive several chunks before the first hazard"
    r = label_state(1, 1, [3] * (d - 2), depth=6, beam_width=4)
    assert r.soft_targets[3] < 0.05, f"doomed right+B not suppressed: {r.soft_targets}"
    assert r.per_action_value[3] < -1000, "doomed action not flagged in q"
    assert r.soft_targets.max() > 0.2, "no survivable escape carries mass"


def test_entropy_below_uniform_at_hazard():
    d = _chunks_until_death(action=3)
    r = label_state(1, 1, [3] * (d - 2), depth=6, beam_width=4)
    assert soft_entropy(r.soft_targets) < np.log(7) - 0.1  # strictly less than uniform
