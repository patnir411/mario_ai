"""Dataset storage + loader for distillation.

Sharded .npz under data/shards/<level>/. Rows store the FLAT observe() vector; the
policy's K-frame stack is assembled at load time, trajectory-aware (never stacking
across trajectory boundaries — zero-pad the first K-1 of each trajectory). Pure
numpy/torch, no emulator dependency, so it's fast to unit-test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mario.observation import OBS_DIM

# row fields and dtypes (the dataset contract)
FIELDS = {
    "obs": np.float32,          # [N, OBS_DIM]
    "hard_action": np.int8,     # [N]   decisive action from a search trajectory (primary)
    "soft_targets": np.float32, # [N, 7] auxiliary fatal-masked distribution
    "value": np.float32,        # [N]   teacher value (max_a q)
    "level_id": np.int16,       # [N]   world*10 + stage
    "source": np.int8,          # [N]   0 onpath, 1 recover, 2 predeath
    "seed": np.int16,           # [N]
    "prefix_len": np.int16,     # [N]   chunks from reset (ordering within a trajectory)
    "trajectory_id": np.int32,  # [N]   K-stacks never cross this boundary
}
N_ACTIONS = 7


def encode_level_id(world: int, stage: int) -> int:
    return world * 10 + stage


def write_shard(path: str | Path, rows: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(rows[k], dtype=dt) for k, dt in FIELDS.items()}
    n = len(out["hard_action"])
    assert out["obs"].shape == (n, OBS_DIM), out["obs"].shape
    assert out["soft_targets"].shape == (n, N_ACTIONS), out["soft_targets"].shape
    np.savez_compressed(path, **out)
    return path


def read_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in FIELDS}


class DatasetIndex:
    """Loads all shards listed in a manifest, precomputes trajectory-aware K-stacks."""

    def __init__(self, manifest_path: str | Path, K: int = 4):
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.root = Path(manifest_path).parent
        self.K = K
        shards = []
        for lvl in self.manifest["levels"].values():
            shards.extend(lvl["shards"])
        parts = [read_shard(self.root / s) for s in shards]
        self.data = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
        self._build_stacks()
        self._build_weights()

    def _build_stacks(self) -> None:
        K, obs = self.K, self.data["obs"]
        N = len(obs)
        X = np.zeros((N, K, OBS_DIM), dtype=np.float32)
        # group rows by trajectory, order by prefix_len, stack within each group
        tid = self.data["trajectory_id"]
        plen = self.data["prefix_len"]
        for t in np.unique(tid):
            idx = np.where(tid == t)[0]
            idx = idx[np.argsort(plen[idx])]
            for j, i in enumerate(idx):
                for k in range(K):
                    src = j - (K - 1 - k)          # k=K-1 is current; earlier k look back
                    if src >= 0:
                        X[i, k] = obs[idx[src]]
                    # else leave zero-padding
        self.X = X.reshape(N, K * OBS_DIM)

    def _build_weights(self) -> None:
        ha = self.data["hard_action"]
        counts = np.bincount(ha, minlength=N_ACTIONS).astype(np.float64)
        with np.errstate(divide="ignore"):
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
        w = inv[ha]
        self.weight = (w / w.mean()).astype(np.float32)  # mean-1 normalized

    def __len__(self) -> int:
        return len(self.data["hard_action"])

    def split(self, val_trajectory_ids: set[int]):
        """Boolean train/val masks by trajectory id (val = held-out whole trajectories)."""
        tid = self.data["trajectory_id"]
        val = np.isin(tid, list(val_trajectory_ids))
        return ~val, val

    def iter_batches(self, batch_size: int, *, mask=None, shuffle=True, seed=0):
        idx = np.arange(len(self)) if mask is None else np.where(mask)[0]
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            yield {
                "X": self.X[b],
                "hard": self.data["hard_action"][b].astype(np.int64),
                "soft": self.data["soft_targets"][b],
                "value": self.data["value"][b],
                "weight": self.weight[b],
                "source": self.data["source"][b],
            }
