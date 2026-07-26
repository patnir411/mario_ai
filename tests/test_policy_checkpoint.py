"""Safe persistence for the original tile-observation policy."""
from __future__ import annotations

import torch

from mario.observation import OBS_DIM
from mario.policy import MarioPolicy, load_policy, save_checkpoint


def test_policy_checkpoint_roundtrip_and_metadata(tmp_path):
    torch.manual_seed(0)
    net = MarioPolicy(K=2, hidden=(16,), dropout=0.0).eval()
    path = tmp_path / "nested" / "policy.pt"
    save_checkpoint(
        path,
        net,
        chunk_frames=8,
        train_cfg={"seed": 0},
        val_metrics={"completion_rate": 0.5},
    )

    loaded, metadata = load_policy(path)
    x = torch.randn(3, 2 * OBS_DIM)

    assert torch.allclose(net(x)[0], loaded(x)[0])
    assert loaded.dropout == 0.0
    assert "state_dict" not in metadata
    assert metadata["chunk_frames"] == 8
    assert metadata["train_cfg"]["seed"] == 0
