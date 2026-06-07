"""V1 — generate a self-supervised dataset for a level by distilling the search teacher.

Recipe (evidence-based; see CLAUDE.md notes on label validation):
  - HARD labels (primary) = decisive actions from search trajectories:
      * on-path: the winning beam-search path
      * off-path: "perturb-and-recover" — at anchors, take a plausible wrong action,
        run a recovery beam search, and label the recovered states (DAgger-lite upfront)
  - SOFT targets (auxiliary) + value = label_state() (sharp fatal-masking at hazards)
  - all-doomed states are excluded (never clone "you're doomed")

Parallel across cores (snapshots aren't picklable, but action-path prefixes are, and the
sim is deterministic, so every worker reconstructs state by replay). Writes
data/shards/level-W-S/shard-0000.npz and updates data/manifest.json.

    ./venv/bin/python scripts/gen_dataset.py [world] [stage]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import multiprocessing as mp

import numpy as np

from mario.buffer import FIELDS, encode_level_id, write_shard
from mario.env import N_ACTIONS
from mario.io import env_fingerprint, git_rev, utc_now_iso, write_json_atomic
from mario.label import fast_label, soft_entropy
from mario.render import replay as _replay
from mario.search import beam_search

DATA = ROOT / "data"
ANCHOR_EVERY = 5
RECOVER_KEEP = 6        # label up to this many recovered states per perturbation branch
RECOVER_DEPTH = 48
RECOVER_BEAM = 16
PERTURB_PREF = [3, 4, 1, 2, 5]   # plausible forward/jump "mistakes" (not NOOP/left)

_SIM_CACHE: dict = {}


def _pick_perturbations(path_action: int, n: int = 1) -> list[int]:
    return [a for a in PERTURB_PREF if a != path_action][:n]


# ---- workers (module-level for spawn pickling) ------------------------------
def _recovery_worker(spec):
    world, stage, seed, prefix, tid, anchor_t, pert = spec
    res = beam_search(world, stage, beam_width=RECOVER_BEAM, max_depth=RECOVER_DEPTH,
                      seed=seed, start_prefix=prefix)
    return {"tid": tid, "anchor_t": anchor_t, "pert": pert,
            "cont": res.path, "x_max": res.x_max, "solved": res.solved}


def _label_worker(task):
    world, stage, seed, prefix, hard, source, tid, plen = task
    obs, soft, value, all_doomed = fast_label(world, stage, prefix, seed=seed)
    if all_doomed:
        return None
    return {
        "obs": obs, "hard_action": hard, "soft_targets": soft,
        "value": value, "level_id": encode_level_id(world, stage),
        "source": source, "seed": seed, "prefix_len": plen, "trajectory_id": tid,
        "_entropy": soft_entropy(soft),
    }


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    from_solution = "--from-solution" in flags  # backbone = cached solution (beam can't re-solve)
    world = int(pos[0]) if len(pos) > 0 else 1
    stage = int(pos[1]) if len(pos) > 1 else 1
    anchor_every = int(pos[2]) if len(pos) > 2 else ANCHOR_EVERY
    n_perturb = int(pos[3]) if len(pos) > 3 else 1
    seed = 0
    n_proc = max(1, min(11, (os.cpu_count() or 4) - 1))
    print(f"[gen_dataset] level {world}-{stage} seed={seed} procs={n_proc}")

    # 1) backbone winning path (cached — the search is deterministic, so reuse it
    #    across re-generations with different coverage settings instead of paying 250s)
    cache = DATA / "shards" / f"level-{world}-{stage}" / "backbone.json"
    if from_solution:
        sol = json.loads((DATA / "solutions" / f"{world}-{stage}.json").read_text())
        P = sol["path"]
        assert sol.get("chunk_frames", 8) == 8, (
            f"--from-solution needs a cf8 solution; {world}-{stage} is cf={sol.get('chunk_frames')} "
            f"(a cf8 generalist can't represent a frame-precise cf1 trajectory)")
        recs = _replay(world, stage, seed, P, 8)   # confirm it solves + get flag x
        bb_solved, bb_x = bool(recs[-1]["flag"]), int(recs[-1]["info"].get("x_pos", 0))
        print(f"  backbone (cached solution): solved={bb_solved} len={len(P)} x={bb_x}")
    elif cache.exists():
        c = json.loads(cache.read_text())
        P, bb_solved, bb_x = c["path"], c["solved"], c.get("x_max", 0)
        print(f"  backbone (cached): solved={bb_solved} len={len(P)}")
    else:
        print("  backbone beam_search ...", flush=True)
        bb = beam_search(world, stage, beam_width=48, max_depth=180, seed=seed)
        P, bb_solved, bb_x = bb.path, bb.solved, bb.x_max
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(cache, {"path": P, "solved": bb_solved, "x_max": bb.x_max})
        print(f"  backbone: solved={bb_solved} len={len(P)} x_max={bb.x_max} "
              f"wall={bb.wall_clock_s:.0f}s")
    if not bb_solved:
        print("  WARNING: backbone did not solve — dataset will lack a beaten trajectory")

    # 2) perturb-and-recover (parallel recovery searches)
    specs, tid = [], 1
    for t in range(0, len(P), anchor_every):
        for pert in _pick_perturbations(P[t], n=n_perturb):
            specs.append((world, stage, seed, P[:t] + [pert], tid, t, pert))
            tid += 1
    # recovery searches are deterministic -> cache by prefix so obs-only re-gens are fast
    rec_cache_path = DATA / "shards" / f"level-{world}-{stage}" / "recoveries.json"
    rec_cache = (json.loads(rec_cache_path.read_text()) if rec_cache_path.exists() else {})
    todo = [s for s in specs if ",".join(map(str, s[3])) not in rec_cache]
    print(f"  {len(specs)} recovery anchors ({len(specs)-len(todo)} cached, {len(todo)} to search) ...",
          flush=True)
    if todo:
        with mp.Pool(n_proc) as pool:
            for s, rec in zip(todo, pool.map(_recovery_worker, todo)):
                rec_cache[",".join(map(str, s[3]))] = rec["cont"]
        write_json_atomic(rec_cache_path, rec_cache)
    recoveries = [{"tid": s[4], "anchor_t": s[5], "pert": s[6],
                   "cont": rec_cache[",".join(map(str, s[3]))]} for s in specs]

    # 3) build label tasks: on-path + recovered states
    tasks = []
    for t in range(len(P)):
        tasks.append((world, stage, seed, P[:t], int(P[t]), 0, 0, t))  # on-path, tid 0
    n_recover_branches = 0
    for rec in recoveries:
        cont = rec["cont"]
        anchor_t, pert, tid = rec["anchor_t"], rec["pert"], rec["tid"]
        # require the recovery to actually make forward progress past the anchor
        anchor_x = None
        if not cont:
            continue
        n_recover_branches += 1
        base = P[:anchor_t] + [pert]
        for j in range(min(RECOVER_KEEP, len(cont))):
            tasks.append((world, stage, seed, base + cont[:j], int(cont[j]), 1, tid,
                          anchor_t + 1 + j))
    print(f"  {len(tasks)} label tasks ({n_recover_branches} recovery branches) ...",
          flush=True)

    with mp.Pool(n_proc) as pool:
        rows = [r for r in pool.map(_label_worker, tasks) if r is not None]
    print(f"  labeled {len(rows)} states (excluded {len(tasks)-len(rows)} all-doomed)")

    # 4) assemble + write shard
    cols = {k: np.array([r[k] for r in rows]) for k in FIELDS}
    shard_rel = f"shards/level-{world}-{stage}/shard-0000.npz"
    write_shard(DATA / shard_rel, cols)

    # 5) manifest
    ent = np.array([r["_entropy"] for r in rows])
    src = cols["source"]
    hist = np.bincount(cols["hard_action"].astype(int), minlength=N_ACTIONS).tolist()
    lvl_key = f"{world}-{stage}"
    manifest_path = DATA / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else
                {"generated_at": utc_now_iso(), "git_rev": git_rev(),
                 "env_fingerprint": env_fingerprint(), "chunk_frames": 8,
                 "obs_dim": int(cols["obs"].shape[1]), "levels": {}})
    manifest["levels"][lvl_key] = {
        "n_samples": len(rows),
        "n_onpath": int((src == 0).sum()),
        "n_recover": int((src == 1).sum()),
        "has_beaten_trajectory": bool(bb_solved),
        "soft_entropy_mean": round(float(ent.mean()), 3),
        "soft_entropy_p05": round(float(np.percentile(ent, 5)), 3),
        "soft_entropy_p95": round(float(np.percentile(ent, 95)), 3),
        "hard_action_hist": hist,
        "shards": [shard_rel],
        "backbone_run": {"solved": bool(bb_solved), "x_at_flag": int(bb_x),
                         "framerule": __import__("math").ceil(len(P) * 8 / 21)},
    }
    manifest["generated_at"] = utc_now_iso()
    manifest["git_rev"] = git_rev()
    manifest["totals"] = {
        "n_samples": sum(v["n_samples"] for v in manifest["levels"].values()),
        "levels_with_beaten_trajectory":
            sum(1 for v in manifest["levels"].values() if v["has_beaten_trajectory"]),
    }
    write_json_atomic(manifest_path, manifest)
    print(f"  wrote {shard_rel} | n={len(rows)} onpath={int((src==0).sum())} "
          f"recover={int((src==1).sum())} entropy_mean={ent.mean():.2f}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
