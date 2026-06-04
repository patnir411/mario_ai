"""Dataset buffer invariants — shard roundtrip + trajectory-aware K-stacking.

The K-stack must never cross a trajectory boundary (zero-pad the first K-1 of each
trajectory); getting this wrong silently corrupts the policy's temporal context.
"""
from __future__ import annotations

import numpy as np

from mario.buffer import FIELDS, DatasetIndex, read_shard, write_shard
from mario.env import N_ACTIONS
from mario.observation import OBS_DIM


def _synthetic_shard(path):
    # two trajectories: tid0 has 3 states (prefix 0,1,2), tid1 has 2 (prefix 0,1)
    tids = [0, 0, 0, 1, 1]
    plens = [0, 1, 2, 0, 1]
    n = len(tids)
    obs = np.zeros((n, OBS_DIM), np.float32)
    for i in range(n):
        obs[i, 0] = i + 1          # identifiable per row
    rows = {
        "obs": obs,
        "hard_action": np.array([1, 1, 3, 1, 0], np.int8),
        "soft_targets": np.tile(np.eye(N_ACTIONS)[1], (n, 1)).astype(np.float32),
        "value": np.zeros(n, np.float32),
        "level_id": np.full(n, 11, np.int16),
        "source": np.array([0, 0, 0, 1, 1], np.int8),
        "seed": np.zeros(n, np.int16),
        "prefix_len": np.array(plens, np.int16),
        "trajectory_id": np.array(tids, np.int32),
    }
    write_shard(path, rows)


def test_shard_roundtrip(tmp_path):
    p = tmp_path / "shards" / "level-1-1" / "shard-0000.npz"
    _synthetic_shard(p)
    back = read_shard(p)
    assert set(back) == set(FIELDS)
    assert back["obs"].shape == (5, OBS_DIM)
    assert back["hard_action"].tolist() == [1, 1, 3, 1, 0]


def test_kstack_is_trajectory_aware(tmp_path):
    p = tmp_path / "shards" / "level-1-1" / "shard-0000.npz"
    _synthetic_shard(p)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"levels": {"1-1": {"shards": ["shards/level-1-1/shard-0000.npz"]}}}')

    K = 3
    ds = DatasetIndex(manifest, K=K)
    assert len(ds) == 5
    assert ds.X.shape == (5, K * OBS_DIM)
    X = ds.X.reshape(5, K, OBS_DIM)
    ids = ds.data["obs"][:, 0]  # row identity == i+1

    # row 0 (tid0, prefix0): first K-1 slots zero-padded, last slot == itself
    assert X[0, 0, 0] == 0 and X[0, 1, 0] == 0 and X[0, 2, 0] == ids[0]
    # row 2 (tid0, prefix2): full stack of obs0,obs1,obs2
    assert X[2, 0, 0] == ids[0] and X[2, 1, 0] == ids[1] and X[2, 2, 0] == ids[2]
    # row 3 (tid1, prefix0): MUST be zero-padded — not bleed from tid0's last row
    assert X[3, 0, 0] == 0 and X[3, 1, 0] == 0 and X[3, 2, 0] == ids[3]
    # row 4 (tid1, prefix1): only obs3,obs4 (+ pad), never tid0
    assert X[4, 0, 0] == 0 and X[4, 1, 0] == ids[3] and X[4, 2, 0] == ids[4]


def test_class_balance_weights_mean_one(tmp_path):
    p = tmp_path / "shards" / "level-1-1" / "shard-0000.npz"
    _synthetic_shard(p)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"levels": {"1-1": {"shards": ["shards/level-1-1/shard-0000.npz"]}}}')
    ds = DatasetIndex(manifest, K=2)
    assert abs(float(ds.weight.mean()) - 1.0) < 1e-5
    # rarer action (3 appears once) must get more weight than common action (1 appears 3x)
    ha = ds.data["hard_action"]
    assert ds.weight[ha == 3].mean() > ds.weight[ha == 1].mean()
