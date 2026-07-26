"""Safe, JSON-ready value-checkpoint persistence."""
from __future__ import annotations

import json

import numpy as np
import torch

from mario.observation import OBS_DIM
from mario.value import ValueNet, load_value, save_value


def test_value_checkpoint_metadata_is_safe_and_json_ready(tmp_path):
    path = tmp_path / "nested" / "value.pt"
    net = ValueNet(hidden=(16,))
    save_value(path, net, {"val_auc": np.float64(0.75)})

    loaded, metadata = load_value(path)

    assert isinstance(loaded, ValueNet)
    assert metadata == {
        "hidden": [16],
        "obs_dim": OBS_DIM,
        "val_metrics": {"val_auc": 0.75},
    }
    json.dumps(metadata)


def test_legacy_numpy_scalar_metadata_uses_weights_only_loader(tmp_path):
    path = tmp_path / "legacy.pt"
    net = ValueNet(hidden=(8,))
    torch.save({
        "state_dict": net.state_dict(),
        "hidden": [8],
        "obs_dim": OBS_DIM,
        "val_metrics": {"val_auc": np.float64(0.5)},
    }, path)

    _loaded, metadata = load_value(path)

    assert metadata["val_metrics"]["val_auc"] == 0.5
    json.dumps(metadata)
