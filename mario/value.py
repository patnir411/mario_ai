"""Independent value net V(s) ≈ P(a flag-reaching continuation exists from s).

Deliberately SEPARATE from MarioPolicy (the audit's lesson: a value head on the policy's
shared trunk perturbs the policy and regressed 1-1). This is a small standalone MLP over a
SINGLE observe() vector (no K-stack — a position heuristic doesn't need the policy's temporal
context), trained by BCE on (on-winning-path = 1) vs (doomed/pre-death = 0). The search uses
sigmoid(V) as a guidance term so a NARROW beam follows the solvable lineage deep and prunes
doomed-but-high-progress branches — the "shallow search behaves deep" goal (DESIGN.md §10).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mario.observation import OBS_DIM


class ValueNet(nn.Module):
    def __init__(self, hidden=(256, 128), dropout: float = 0.1):
        super().__init__()
        self.hidden = list(hidden)
        d = OBS_DIM
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                      # x: [B, OBS_DIM] -> [B] logit
        return self.head(self.trunk(x)).squeeze(-1)


def save_value(path, net: ValueNet, val_metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    val_metrics = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in val_metrics.items()
    }
    torch.save({"state_dict": net.state_dict(), "hidden": net.hidden,
                "obs_dim": OBS_DIM, "val_metrics": val_metrics}, path)


def load_value(path, device="cpu"):
    # Older local checkpoints stored a NumPy scalar in val_metrics.  Allow only
    # the minimal NumPy constructors needed for that metadata while retaining
    # PyTorch's weights-only unpickler (never fall back to arbitrary pickle).
    safe_numpy = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_numpy):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    assert ckpt["obs_dim"] == OBS_DIM, f"value net obs_dim {ckpt['obs_dim']} != {OBS_DIM}"
    net = ValueNet(hidden=tuple(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    metadata = {key: value for key, value in ckpt.items() if key != "state_dict"}
    if isinstance(metadata.get("val_metrics"), dict):
        metadata["val_metrics"] = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in metadata["val_metrics"].items()
        }
    return net, metadata


class ValueGuide:
    """Wraps a trained ValueNet for use inside search: P(survivable/winning) for a RAM state."""

    def __init__(self, net: ValueNet, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def p(self, obs_vec: np.ndarray) -> float:
        x = torch.from_numpy(obs_vec.astype(np.float32)).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.net(x))[0])
