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
from mario.observation import OBS_DIM

DATA = ROOT / "data"
MIN_SAMPLES = 400
# Soft targets are uniform on open ground and sharp only at hazards (validated), so the
# meaningful check is that DECISIVE states exist — the low tail (p05) is well below uniform.
ENT_P05_MAX = 1.70           # ln(7)=1.946; require the sharpest ~5% to be clearly sub-uniform
MIN_DISTINCT_ACTIONS = 4


def check_level(key: str, lvl: dict) -> list[str]:
    fails = []
    if not lvl.get("has_beaten_trajectory"):
        fails.append("no beaten trajectory")
    if lvl.get("n_samples", 0) < MIN_SAMPLES:
        fails.append(f"n_samples {lvl.get('n_samples')} < {MIN_SAMPLES}")
    if lvl.get("n_recover", 0) <= 0:
        fails.append("no off-path (recover) coverage")
    p05 = lvl.get("soft_entropy_p05", 99)
    if p05 > ENT_P05_MAX:
        fails.append(f"no decisive states: soft_entropy_p05 {p05} > {ENT_P05_MAX}")
    hist = lvl.get("hard_action_hist", [])
    if sum(1 for c in hist if c > 0) < MIN_DISTINCT_ACTIONS:
        fails.append(f"hard labels cover <{MIN_DISTINCT_ACTIONS} actions: {hist}")
    # shards load with correct shapes
    n = 0
    for s in lvl.get("shards", []):
        d = read_shard(DATA / s)
        if d["obs"].shape[1] != OBS_DIM or d["soft_targets"].shape[1] != 7:
            fails.append(f"bad shard shapes in {s}")
        n += len(d["hard_action"])
        if not np.isfinite(d["obs"]).all():
            fails.append(f"non-finite obs in {s}")
    if n != lvl.get("n_samples"):
        fails.append(f"shard rows {n} != manifest n_samples {lvl.get('n_samples')}")
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
