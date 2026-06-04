"""Tiny student policy: K-frame-stack MLP with policy + value heads.

Input = K stacked observe() vectors (K restores the Markov property that enemy-slot reuse
breaks). Outputs 7 chunk logits + 1 value. The Controller wraps a trained net as a
real-time controller: one act() per chunk, argmax action fed to MarioSim.run_chunk.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from mario.observation import OBS_DIM, observe

N_ACTIONS = 7


class MarioPolicy(nn.Module):
    def __init__(self, K: int = 4, hidden=(512, 256), dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.hidden = list(hidden)
        d = K * OBS_DIM
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, N_ACTIONS)
        self.value_head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.policy_head(z), self.value_head(z).squeeze(-1)


def save_checkpoint(path, net: MarioPolicy, *, chunk_frames: int, train_cfg: dict,
                    val_metrics: dict) -> None:
    torch.save({
        "state_dict": net.state_dict(),
        "arch": {"type": "mlp", "K": net.K, "obs_dim": OBS_DIM, "hidden": net.hidden},
        "chunk_frames": chunk_frames,
        "train_cfg": train_cfg,
        "val_metrics": val_metrics,
    }, path)


def load_policy(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["arch"]
    net = MarioPolicy(K=a["K"], hidden=tuple(a["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt


# Inference bias: under uncertainty (near-uniform logits) plain argmax ties to index 0
# (NOOP) and the policy stalls/times out. A mild bias against NOOP/left makes the
# controller prefer forward motion when it's unsure, without overriding confident NOOPs.
ACTION_BIAS = np.array([-0.6, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)  # NOOP, .., left


class Controller:
    """Wraps a trained net as a controller. act() returns one action per chunk."""

    def __init__(self, net: MarioPolicy, device="cpu", chunk_frames: int = 8,
                 action_bias=ACTION_BIAS):
        self.net = net.to(device).eval()
        self.device = device
        self.K = net.K
        self.chunk_frames = chunk_frames
        self.action_bias = (torch.from_numpy(np.asarray(action_bias, np.float32))
                            if action_bias is not None else None)
        self.reset()

    def reset(self) -> None:
        self.stack = deque([np.zeros(OBS_DIM, np.float32)] * self.K, maxlen=self.K)

    def _logits(self, ram, info):
        self.stack.append(observe(ram, info).astype(np.float32))
        x = torch.from_numpy(np.concatenate(self.stack)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        return logits[0], float(value[0])

    def act(self, ram, info) -> int:
        logits, _v = self._logits(ram, info)
        if self.action_bias is not None:
            logits = logits + self.action_bias
        return int(torch.argmax(logits).item())

    def act_with_entropy(self, ram, info):
        """Return (action, entropy, value) — used by DAgger to flag uncertain states."""
        logits, value = self._logits(ram, info)
        p = torch.softmax(logits, dim=-1)
        entropy = float(-(p * torch.log(p + 1e-9)).sum().item())
        return int(torch.argmax(logits).item()), entropy, value
