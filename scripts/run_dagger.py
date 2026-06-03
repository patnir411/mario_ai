"""V3 — DAgger loop: iterate until the learned policy clears 1-1.

Each round: roll out the current policy → collect its failure/uncertain states → expert-
label decisive recoveries (parallel) → append to the dataset → retrain (subprocess, MPS)
→ eval. Stop when 1-1 completion_rate >= 0.90 or max rounds.

    ./venv/bin/python scripts/run_dagger.py <bc_checkpoint_run_id> [rounds] [collect_seeds]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from mario.buffer import FIELDS, write_shard
from mario.dagger import collect_failures, dagger_label_worker
from mario.io import write_json_atomic

DATA = ROOT / "data"
PY = str(ROOT / "venv" / "bin" / "python")
WORLD, STAGE = 1, 1
KEEP = 12
TARGET = 0.90


def _append_shard_to_manifest(shard_rel: str, n_new: int, n_dagger_total: int) -> None:
    mp_path = DATA / "manifest.json"
    m = json.loads(mp_path.read_text())
    lvl = m["levels"][f"{WORLD}-{STAGE}"]
    if shard_rel not in lvl["shards"]:
        lvl["shards"].append(shard_rel)
    lvl["n_samples"] = lvl.get("n_samples", 0) + n_new
    lvl["n_dagger"] = lvl.get("n_dagger", 0) + n_new
    m["totals"]["n_samples"] = sum(v["n_samples"] for v in m["levels"].values())
    write_json_atomic(mp_path, m)


def _train() -> str:
    out = subprocess.run([PY, "-m", "mario.train"], cwd=ROOT,
                         env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                         capture_output=True, text=True)
    print(out.stdout[-400:])
    for line in out.stdout.splitlines():
        if line.startswith("RUN_ID="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("train did not emit RUN_ID\n" + out.stderr[-800:])


def _eval(run_id: str, n_seeds: int = 20) -> dict:
    from mario.eval import eval_suite
    ck = ROOT / "runs" / run_id / "checkpoint.pt"
    return eval_suite(ck, [(WORLD, STAGE)], list(range(n_seeds)))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_dagger.py <bc_checkpoint_run_id> [rounds] [seeds]")
    ckpt_run = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    collect_seeds = list(range(int(sys.argv[3]) if len(sys.argv) > 3 else 8))
    n_proc = max(1, min(11, (os.cpu_count() or 4) - 1))

    checkpoint = ROOT / "runs" / ckpt_run / "checkpoint.pt"
    history = []
    for rnd in range(1, rounds + 1):
        print(f"\n===== DAgger round {rnd} (policy={checkpoint.parent.name}) =====")
        anchors, outcomes, cf = collect_failures(checkpoint, WORLD, STAGE, collect_seeds)
        beat = sum(o["beat"] for o in outcomes)
        print(f"  rollouts: {beat}/{len(outcomes)} beat | {len(anchors)} recovery anchors")
        if not anchors:
            print("  no failure anchors — policy is clean")
            break

        specs = [(WORLD, STAGE, 0, a, 100000 + rnd * 1000 + i, KEEP)
                 for i, a in enumerate(anchors)]
        with mp.Pool(n_proc) as pool:
            rows = [r for batch in pool.map(dagger_label_worker, specs) for r in batch]
        print(f"  labeled {len(rows)} decisive correction states")
        if not rows:
            print("  no labelable corrections; stopping")
            break

        cols = {k: np.array([r[k] for r in rows]) for k in FIELDS}
        shard_rel = f"shards/level-{WORLD}-{STAGE}/dagger-r{rnd}.npz"
        write_shard(DATA / shard_rel, cols)
        _append_shard_to_manifest(shard_rel, len(rows), len(rows))

        new_run = _train()
        ev = _eval(new_run)
        cr = ev["levels"][f"{WORLD}-{STAGE}"]["completion_rate"]
        causes = ev["levels"][f"{WORLD}-{STAGE}"]["deaths_by_cause"]
        history.append({"round": rnd, "completion_rate": cr, "deaths": causes,
                        "checkpoint": new_run})
        print(f"  round {rnd}: completion_rate={cr} deaths={causes} ({new_run})")
        checkpoint = ROOT / "runs" / new_run / "checkpoint.pt"
        if cr >= TARGET:
            print(f"\nV3 GATE: PASS — 1-1 completion_rate {cr} >= {TARGET}")
            break
    else:
        print("\nmax rounds reached")

    write_json_atomic(ROOT / "runs" / "dagger_history.json", {"history": history})
    if history:
        best = max(history, key=lambda h: h["completion_rate"])
        print(f"\nbest: round {best['round']} completion_rate={best['completion_rate']} "
              f"checkpoint={best['checkpoint']}")


if __name__ == "__main__":
    main()
