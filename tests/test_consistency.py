"""Input-consistency loss + entity-policy structural invariants.

The consistency penalty must be non-negative, vanish at sigma=0, and respond to
larger perturbations.  These checks do not assert closed-loop contraction.  Also
lock the entity-policy forward contract and label_at_state's full restore fidelity.
"""
from __future__ import annotations

import numpy as np
import torch

from mario.entity import OBS_DIM_ENTITY, entity_obs
from mario.entity_policy import (
    EntityPolicyPrior,
    EntityTransformer,
    TemporalEntityTransformer,
    load_policy_checkpoint,
    save_policy,
)
from mario.env import MarioSim, N_ACTIONS
from mario.label import fast_label, label_at_state


def _net():
    torch.manual_seed(0)
    return EntityTransformer(d_model=64, nhead=4, layers=2)


def test_input_consistency_nonneg_and_zero_at_sigma0():
    from mario.consistency import input_consistency_penalty
    net = _net()
    x = torch.randn(16, OBS_DIM_ENTITY)
    logits = net.logits(x)
    assert float(input_consistency_penalty(logits, net, x, sigma=0.0).detach()) == 0.0
    p = float(input_consistency_penalty(logits, net, x, sigma=0.05).detach())
    assert p >= 0.0


def test_input_consistency_grows_with_sigma():
    from mario.consistency import input_consistency_penalty
    net = _net().eval()
    torch.manual_seed(1)
    x = torch.randn(64, OBS_DIM_ENTITY)
    logits = net.logits(x)
    torch.manual_seed(2)
    small = float(input_consistency_penalty(logits, net, x, sigma=0.02).detach())
    torch.manual_seed(2)
    big = float(input_consistency_penalty(logits, net, x, sigma=0.2).detach())
    assert big >= small


def test_input_consistency_disables_dropout_and_restores_training_mode():
    from mario.consistency import input_consistency_penalty
    net = _net().train()
    x = torch.randn(32, OBS_DIM_ENTITY)
    logits = net.logits(x)
    torch.manual_seed(7)
    first = input_consistency_penalty(logits, net, x, sigma=0.05)
    assert net.training
    torch.manual_seed(7)
    second = input_consistency_penalty(logits, net, x, sigma=0.05)
    assert net.training
    assert torch.allclose(first, second)


def test_entity_policy_forward_contract():
    net = _net().eval()   # eval() so dropout is off and the two forward passes match
    x = torch.randn(8, OBS_DIM_ENTITY)
    logits, value = net(x)
    assert logits.shape == (8, N_ACTIONS)
    assert value.shape == (8,)
    assert torch.allclose(net.logits(x), logits)


def test_entity_policy_save_load_preserves_architecture_and_metadata(tmp_path):
    torch.manual_seed(4)
    net = TemporalEntityTransformer(
        ctx_k=3, d_model=64, nhead=4, layers=2, dropout=0.0).eval()
    path = tmp_path / "policy.pt"
    save_policy(net, path, extra={
        "train_levels": ["1-1"],
        "holdout_levels": ["2-3"],
    })
    loaded, metadata = load_policy_checkpoint(path)
    x = torch.randn(2, 3, OBS_DIM_ENTITY)
    assert isinstance(loaded, TemporalEntityTransformer)
    assert torch.allclose(net.logits(x), loaded.logits(x))
    assert metadata["holdout_levels"] == ["2-3"]


def test_policy_shapes_and_temperature_are_validated():
    net = _net().eval()
    with np.testing.assert_raises(ValueError):
        net.logits(torch.randn(2, OBS_DIM_ENTITY - 1))
    with np.testing.assert_raises(ValueError):
        EntityPolicyPrior(net, temperature=0)


def test_label_at_state_matches_fast_label_and_restores():
    """The label sweep must restore emulator RAM, cached info, and cached frame."""
    from mario.ram import mario_level_x
    sim = MarioSim(1, 1); sim.reset(0)
    for a in [1, 1, 2, 1]:
        sim.run_chunk(a, 8)
    x0 = int(sim.last_info.get("x_pos", 0))
    info_before = dict(sim.last_info)
    obs_before = np.asarray(sim.last_obs).copy()
    ram_before = np.asarray(sim.ram).copy()
    ram_x_before = mario_level_x(sim.ram)
    soft, value, best, doomed, q = label_at_state(sim, x0, chunk_frames=8)
    # The 9-action sweep is fully undone in both emulator and wrapper views.
    assert mario_level_x(sim.ram) == ram_x_before
    assert np.array_equal(sim.ram, ram_before)
    assert sim.last_info == info_before
    assert np.array_equal(sim.last_obs, obs_before)
    assert int(sim.last_info["x_pos"]) == mario_level_x(sim.ram)
    assert soft.shape == (N_ACTIONS,) and abs(float(soft.sum()) - 1.0) < 1e-4
    sim.close()
    # fast_label (which now delegates to label_at_state) agrees on the same prefix
    _obs, soft2, value2, doomed2 = fast_label(1, 1, [1, 1, 2, 1], chunk_frames=8)
    assert np.allclose(soft, soft2, atol=1e-5)
    assert abs(value - value2) < 1e-3
    assert doomed == doomed2
