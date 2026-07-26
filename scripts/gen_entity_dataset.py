"""Build the generalist ENTITY dataset with procedure-cloning targets, one shard per level.

Per state we store the entity tokens PLUS the search teacher's intermediate computation
(soft action distribution + value) — the Procedure-Cloning signal (arXiv 2205.10816) that
V4's hard-CE-only entity experiment lacked. Coverage = on-path (the cached cf8 solution) +
perturb-and-recover branches (so the policy sees recovery from off-path states).

Per-level shards (`data/entity_shards/level-{w}-{s}.npz`) let the trainer include/exclude
levels by allowlist — the held-out integrity rule. Only cf8 solutions are usable as cf8
supervision; cf1 solutions (TAS-precise water/castle: 2-2,7-2,7-4,8-4) are logged + skipped.

    ./venv/bin/python scripts/gen_entity_dataset.py [only=W-S,...] [--force]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from mario.entity import entity_obs
from mario.env import MarioSim, N_ACTIONS
from mario.label import label_at_state
from mario.solution_verification import verify_stock_solution

SOL = ROOT / "data" / "solutions"
OUT = ROOT / "data" / "entity_shards"
CF = 8
# Recovery density traded for throughput: perturb-and-recover gives off-path coverage for the
# BC baseline, but the closed-loop DAgger phase is what actually hardens recovery, so a lighter
# anchor grid here keeps the full-26-level build to ~1 hr instead of ~4. (1-1..1-4 were built
# with the denser 3/2 grid before this retune — richer is fine; uniformity isn't required.)
ANCHOR_EVERY, N_PERTURB = 6, 1
RECOVER_KEEP, RECOVER_BEAM, RECOVER_DEPTH = 8, 16, 40
PERTURB_PREF = [3, 4, 1, 2, 5]


def cf8_levels() -> list[tuple[int, int]]:
    out = []
    for p in sorted(SOL.glob("*.json")):
        d = json.loads(p.read_text())
        if not d.get("solved"):
            continue
        w, s = (int(x) for x in p.stem.split("-"))
        cf = int(d.get("chunk_frames", 8))
        if cf != CF:
            print(f"  skip {w}-{s}: chunk_frames={cf} (not cf8-replayable as cf8 supervision)")
            continue
        check = verify_stock_solution(p, seed=0)
        if not check.replay_verified:
            print(f"  SKIP {w}-{s}: cached solved=True but does NOT replay to flag at seed 0 "
                  f"({check.reason}; excluded from dataset)")
            continue
        out.append((w, s))
    return out


def _recover_worker(spec):
    from mario.search import beam_search
    w, s, prefix, tid = spec
    res = beam_search(w, s, beam_width=RECOVER_BEAM, max_depth=RECOVER_DEPTH, seed=0,
                      start_prefix=prefix)
    return tid, res.path


def build_level(w: int, s: int) -> dict:
    path = json.loads((SOL / f"{w}-{s}.json").read_text())["path"]
    lid = w * 10 + s
    rows = {"obs": [], "hard_action": [], "soft_targets": [], "value": [],
            "level_id": [], "source": [], "traj": []}

    def add(obs, hard, soft, value, source, traj):
        rows["obs"].append(obs); rows["hard_action"].append(int(hard))
        rows["soft_targets"].append(soft); rows["value"].append(float(value))
        rows["level_id"].append(lid); rows["source"].append(int(source)); rows["traj"].append(int(traj))

    # 1) on-path: entity tokens + 1-ply soft/value at every state of the cf8 solution
    sim = MarioSim(w, s); sim.reset(0)
    for a in path:
        x_start = int(sim.last_info.get("x_pos", 0))
        obs = entity_obs(sim.ram, sim.last_info)
        soft, value, _best, all_doomed, _q = label_at_state(sim, x_start, chunk_frames=CF)
        if not all_doomed:
            add(obs, a, soft, value, 0, 0)
        if sim.run_chunk(a, CF)[1]:
            break
    sim.close()

    # 2) perturb-and-recover: beam-search recoveries from off-path anchors (parallel)
    specs, tid = [], 1
    for t in range(0, len(path), ANCHOR_EVERY):
        for pert in [a for a in PERTURB_PREF if a != path[t]][:N_PERTURB]:
            specs.append((w, s, path[:t] + [pert], tid)); tid += 1
    nproc = max(1, min(8, (os.cpu_count() or 4) - 1))
    with mp.Pool(nproc) as pool:
        recs = dict(pool.map(_recover_worker, specs))

    for (w_, s_, prefix, tid_) in specs:
        cont = recs.get(tid_) or []
        if not cont:
            continue
        sim = MarioSim(w, s); sim.reset(0); dead = False
        for a in prefix:
            if sim.run_chunk(a, CF)[1]:
                dead = True; break
        if not dead:
            for j in range(min(RECOVER_KEEP, len(cont))):
                x_start = int(sim.last_info.get("x_pos", 0))
                obs = entity_obs(sim.ram, sim.last_info)
                soft, value, _b, all_doomed, _q = label_at_state(sim, x_start, chunk_frames=CF)
                if not all_doomed:
                    add(obs, cont[j], soft, value, 1, tid_)
                if sim.run_chunk(cont[j], CF)[1]:
                    break
        sim.close()

    arr = {
        "obs": np.asarray(rows["obs"], np.float32),
        "hard_action": np.asarray(rows["hard_action"], np.int64),
        "soft_targets": np.asarray(rows["soft_targets"], np.float32),
        "value": np.asarray(rows["value"], np.float32),
        "level_id": np.asarray(rows["level_id"], np.int64),
        "source": np.asarray(rows["source"], np.int8),
        "traj": np.asarray(rows["traj"], np.int64),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"level-{w}-{s}.npz", **arr)
    n = len(arr["obs"])
    hist = np.bincount(arr["hard_action"], minlength=N_ACTIONS).tolist() if n else []
    return {"level": f"{w}-{s}", "n": n, "onpath": int((arr["source"] == 0).sum()),
            "recover": int((arr["source"] == 1).sum()), "action_hist": hist}


def main() -> None:
    only, force = None, False
    for a in sys.argv[1:]:
        if a == "--force":
            force = True
        elif a.startswith("only="):
            only = {tuple(int(x) for x in p.split("-")) for p in a[5:].split(",")}

    levels = cf8_levels()
    if only is not None:
        levels = [ws for ws in levels if ws in only]
    print(f"building entity shards for {len(levels)} cf8 levels: "
          f"{', '.join(f'{w}-{s}' for w, s in levels)}\n", flush=True)

    summary = []
    for (w, s) in levels:
        shard = OUT / f"level-{w}-{s}.npz"
        if shard.exists() and not force:
            d = np.load(shard)
            summary.append({"level": f"{w}-{s}", "n": len(d["obs"]), "cached": True})
            print(f"  {w}-{s}: cached n={len(d['obs'])}", flush=True)
            continue
        r = build_level(w, s)
        summary.append(r)
        print(f"  {w}-{s}: n={r['n']} onpath={r['onpath']} recover={r['recover']} "
              f"hist={r['action_hist']}", flush=True)

    total = sum(x["n"] for x in summary)
    print(f"\n=== DONE === {len(summary)} levels, {total} states total")
    (OUT / "manifest.json").write_text(json.dumps(
        {"levels": summary, "n_actions": N_ACTIONS, "chunk_frames": CF,
         "obs_kind": "entity"}, indent=2))


if __name__ == "__main__":
    main()
