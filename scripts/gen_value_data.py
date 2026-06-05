"""Generate (obs, reach-flag) training data for the independent value net.

From every solved trajectory in data/solutions/*.json:
  POSITIVES (label 1): each state along the winning path (a flag-reaching continuation
    provably exists from it).
  NEGATIVES (label 0): from sampled on-path states, fork with perturbation actions and roll
    out a forward-biased random policy; the last few states before a death (done & no flag)
    are taken as doomed. This teaches V to separate "on a solvable line" from "heading to death".

Saves data/value_data.npz {obs:[N,OBS_DIM] float32, label:[N] float32, level_id:[N] int16}.

    ./venv/bin/python scripts/gen_value_data.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.env import MarioSim, N_ACTIONS
from mario.observation import observe, OBS_DIM
from mario.reward import is_death, is_success

SOL = ROOT / "data" / "solutions"
# forward-biased random rollout weights (favor right/jump/run, allow left/down)
WEIGHTS = [0.4, 1.0, 1.4, 1.4, 1.2, 0.7, 0.4, 0.5, 0.4]


def gen_level(world, stage, path, cf, *, neg_every=3, rollout=14, rng=None):
    rng = rng or random.Random(world * 10 + stage)
    sim = MarioSim(world, stage); sim.reset(0)
    pos, neg = [], []
    snaps = []
    for i, a in enumerate(path):
        info, done = sim.run_chunk(a, cf)
        if done:
            break
        pos.append(observe(sim.ram, info))               # on-path = solvable
        if i % neg_every == 0:
            snaps.append(sim.snapshot())
    # negatives: from sampled on-path snapshots, force a divergence then random rollout to death
    for snap in snaps:
        sim.restore(snap)
        traj_obs = []
        # first divergence: a deliberately off-policy action (left / down / random)
        for step in range(rollout):
            a = rng.choices(range(N_ACTIONS), weights=WEIGHTS, k=1)[0] if step else \
                rng.choice([6, 7, 0, 5])                  # start with a risky/idle move
            info, done = sim.run_chunk(a, cf)
            traj_obs.append(observe(sim.ram, info))
            if is_success(info):                          # accidentally won — discard as neg
                traj_obs = []
                break
            if is_death(info, done):                      # doomed: keep last few states
                neg.extend(traj_obs[-3:])
                break
    sim.close()
    return pos, neg


def main() -> None:
    obs_all, lab_all, lvl_all = [], [], []
    for f in sorted(SOL.glob("*.json")):
        if f.name.startswith("."):
            continue
        c = json.loads(f.read_text())
        if not c.get("solved"):
            continue
        w, s = (int(x) for x in f.stem.split("-"))
        pos, neg = gen_level(w, s, c["path"], c.get("chunk_frames", 8))
        for o in pos:
            obs_all.append(o); lab_all.append(1.0); lvl_all.append(w * 10 + s)
        for o in neg:
            obs_all.append(o); lab_all.append(0.0); lvl_all.append(w * 10 + s)
        print(f"  {f.stem}: +{len(pos)} pos / -{len(neg)} neg", flush=True)

    obs = np.asarray(obs_all, dtype=np.float32)
    lab = np.asarray(lab_all, dtype=np.float32)
    lvl = np.asarray(lvl_all, dtype=np.int16)
    assert obs.shape[1] == OBS_DIM, obs.shape
    out = ROOT / "data" / "value_data.npz"
    np.savez_compressed(out, obs=obs, label=lab, level_id=lvl)
    print(f"\nsaved {out}: N={len(lab)} pos={int(lab.sum())} neg={int((1-lab).sum())}")


if __name__ == "__main__":
    main()
