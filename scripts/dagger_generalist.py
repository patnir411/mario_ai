"""V4 DAgger for the GENERALIST policy: roll out the single generalist on ALL trained levels,
collect its failure/uncertain states, expert-label decisive recoveries (source=2), append to the
dataset, then retrain the GENERALIST (warm-start, LEVEL_FILTER unset → still one net). One round.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/dagger_generalist.py <ckpt_run_id> [round] [collect_seeds]
"""
from __future__ import annotations
import os, sys, json, subprocess
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from mario.buffer import FIELDS, write_shard
from mario.dagger import collect_failures, dagger_label_worker
from mario.io import write_json_atomic

DATA = ROOT / "data"; PY = str(ROOT / "venv" / "bin" / "python")
KEEP = 12
LEVELS = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 3)]


def main() -> None:
    ckpt_arg = sys.argv[1]
    rnd = sys.argv[2] if len(sys.argv) > 2 else "1"
    collect_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ckpt_path = Path(ckpt_arg) if ckpt_arg.endswith(".pt") else ROOT / "runs" / ckpt_arg / "checkpoint.pt"
    nproc = max(1, min(11, (os.cpu_count() or 4) - 1))
    print(f"[dagger_generalist] ckpt={ckpt_path.parent.name} round={rnd} seeds={collect_seeds} "
          f"levels={len(LEVELS)}", flush=True)

    manifest = json.loads((DATA / "manifest.json").read_text())
    total_added = 0
    for (w, s) in LEVELS:
        anchors, outcomes, cf = collect_failures(str(ckpt_path), w, s, list(range(collect_seeds)))
        if not anchors:
            print(f"  {w}-{s}: no failure anchors", flush=True); continue
        specs = [(w, s, 0, a, tid, KEEP) for tid, a in enumerate(anchors, 1)]
        with mp.Pool(nproc) as pool:
            rows = [r for rs in pool.map(dagger_label_worker, specs) for r in rs]
        if not rows:
            print(f"  {w}-{s}: {len(anchors)} anchors -> 0 rows (no recovery)", flush=True); continue
        cols = {k: np.array([r[k] for r in rows]) for k in FIELDS}
        shard_rel = f"shards/level-{w}-{s}/dagger-g{rnd}.npz"
        write_shard(DATA / shard_rel, cols)
        lvl = manifest["levels"][f"{w}-{s}"]
        if shard_rel not in lvl["shards"]:
            lvl["shards"].append(shard_rel)
        lvl["n_samples"] = int(lvl.get("n_samples", 0)) + len(rows)
        lvl["n_dagger"] = int(lvl.get("n_dagger", 0)) + len(rows)
        total_added += len(rows)
        print(f"  {w}-{s}: {len(anchors)} anchors -> {len(rows)} corrections", flush=True)

    manifest["totals"]["n_samples"] = sum(int(v["n_samples"]) for v in manifest["levels"].values())
    write_json_atomic(DATA / "manifest.json", manifest)
    print(f"  added {total_added} correction rows; retraining generalist (warm-start) ...", flush=True)

    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "INIT_CHECKPOINT": str(ckpt_path)}
    env.pop("LEVEL_FILTER", None)   # GENERALIST retrain (all levels)
    out = subprocess.run([PY, "-m", "mario.train"], cwd=ROOT, env=env, capture_output=True, text=True)
    print("\n".join(out.stdout.splitlines()[-8:]), flush=True)
    if out.returncode != 0:
        print("RETRAIN FAILED:\n", out.stderr[-800:], flush=True)


if __name__ == "__main__":
    main()
