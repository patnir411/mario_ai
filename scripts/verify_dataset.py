"""V1 gate: check data/manifest.json + shards against the dataset pass conditions.

    ./venv/bin/python scripts/verify_dataset.py [world-stage ...]   (default: all levels)
Exits non-zero if any checked level fails a gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from mario.buffer import read_shard
from mario.env import N_ACTIONS
from mario.observation import OBS_DIM

DATA = ROOT / "data"
MIN_SAMPLES = 400
# Soft targets are AUXILIARY (entropy-weighted to ~0 in the loss; hard labels carry the
# signal). With the fast 1-ply labeler they're near-uniform, so entropy is informational
# only — the real quality checks are a beaten trajectory + diverse hard labels + coverage.
MIN_DISTINCT_ACTIONS = 4


def check_level(key: str, lvl: dict) -> list[str]:
    fails = []
    if not lvl.get("has_beaten_trajectory"):
        fails.append("no beaten trajectory")
    if lvl.get("n_recover", 0) <= 0:
        fails.append("no off-path (recover) coverage")
    hist = lvl.get("hard_action_hist", [])
    if sum(1 for c in hist if c > 0) < MIN_DISTINCT_ACTIONS:
        fails.append(f"hard labels cover <{MIN_DISTINCT_ACTIONS} actions: {hist}")
    # shards load with correct shapes
    n = 0
    for s in lvl.get("shards", []):
        d = read_shard(DATA / s)
        if d["obs"].shape[1] != OBS_DIM or d["soft_targets"].shape[1] != N_ACTIONS:
            fails.append(f"bad shard shapes in {s}")
        n += len(d["hard_action"])
        if not np.isfinite(d["obs"]).all():
            fails.append(f"non-finite obs in {s}")
    if n < MIN_SAMPLES:  # use ACTUAL shard rows (manifest n_samples is just metadata)
        fails.append(f"actual samples {n} < {MIN_SAMPLES}")
    return fails


def main() -> int:
    mp = DATA / "manifest.json"
    if not mp.exists():
        print("no data/manifest.json — run gen_dataset.py first")
        return 1
    manifest = json.loads(mp.read_text())
    keys = sys.argv[1:] or list(manifest["levels"].keys())
    ok = True
    for k in keys:
        lvl = manifest["levels"].get(k)
        if lvl is None:
            print(f"{k}: MISSING from manifest"); ok = False; continue
        fails = check_level(k, lvl)
        if fails:
            ok = False
            print(f"{k}: FAIL — " + "; ".join(fails))
        else:
            print(f"{k}: PASS — n={lvl['n_samples']} (onpath={lvl['n_onpath']} "
                  f"recover={lvl['n_recover']}) entropy={lvl['soft_entropy_mean']} "
                  f"hist={lvl['hard_action_hist']}")
    print("\nV1 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
