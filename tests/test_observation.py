"""Observation invariants (DESIGN.md §4). Tile extraction correctness itself is verified
visually via scripts/verify_observation.py; these lock the structural contract."""
from __future__ import annotations

import numpy as np

from mario import observation as O
from mario.env import MarioSim


def test_grid_shape_and_vocab(sim):
    g = O.tile_grid(sim.ram)
    assert g.shape == (O.GRID_H, O.GRID_W)
    assert set(np.unique(g)).issubset({O.EMPTY, O.SOLID, O.ENEMY, O.MARIO})


def test_mario_is_center_and_unique(sim):
    g = O.tile_grid(sim.ram)
    assert g[O.UP, O.LEFT] == O.MARIO
    assert int((g == O.MARIO).sum()) == 1


def test_ground_below_mario_when_standing():
    """At level start Mario stands on ground -> a solid tile exists below his cell."""
    s = MarioSim(1, 1)
    s.reset(seed=0)
    try:
        g = O.tile_grid(s.ram)
        below = [g[O.UP + k, O.LEFT] for k in range(1, 4)]
        assert O.SOLID in below, f"no ground detected below standing Mario: {below}"
    finally:
        s.close()


def test_observe_vector_dim_and_finite(sim):
    obs = O.observe(sim.ram, sim.last_info)
    assert obs.shape == (O.OBS_DIM,)
    assert np.isfinite(obs).all()


def test_scalars_normalized_range(sim):
    sc = O.scalar_features(sim.ram, sim.last_info)
    assert sc.shape == (5,)
    assert np.all(np.abs(sc) <= 5.0)  # generous bound; velocities are /40
