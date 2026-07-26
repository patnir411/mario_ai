"""Zero-shot held-out evaluation of the entity-transformer generalist — NO search rescue.

The decisive measurement: does a policy trained on the TRAIN levels reach the flag on
levels it never saw (and the oracle never relabeled)? Pure closed-loop policy rollout, per
held-out level × seeds. Emits `heldout_eval.json` with every action path,
independent positive replay checks, root-RAM fingerprints, aggregate completion
statistics, and a contact sheet per level.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/eval_heldout.py CKPT \
        [--config configs/heldout_split.json] [--levels 3-1,5-2] [--seeds 4] [--cf 8]

With --config it evaluates the checkpoint's held-out levels; --levels overrides explicitly
(use the train levels too, to confirm in-distribution competence).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from mario.entity_policy import EntityController, load_policy_checkpoint
from mario.env import MarioSim
from mario.io import env_fingerprint, git_rev, new_run_id, run_dir, utc_now_iso, write_json_atomic
from mario.ram import mario_level_x
from mario.render import make_contact_sheet
from mario.reward import death_cause, is_death, is_success
from mario.solution_verification import replay_verify_stock_path

SOL = ROOT / "data" / "solutions"


def level_end_x(w: int, s: int) -> int:
    p = SOL / f"{w}-{s}.json"
    if p.exists():
        d = json.loads(p.read_text())
        return int(d.get("x_at_flag") or d.get("x_max") or 3200)
    return 3200


def checkpoint_ref(path: str | Path) -> dict:
    p = Path(path)
    try:
        display = str(p.resolve().relative_to(ROOT))
    except ValueError:
        display = str(p)
    return {
        "path": display,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
    }


def rollout(ctrl: EntityController, w: int, s: int, seed: int, cf: int, max_chunks: int):
    sim = MarioSim(w, s)
    sim.reset(seed)
    root_ram_sha256 = hashlib.sha256(
        np.asarray(sim.ram, dtype=np.uint8).tobytes()).hexdigest()
    path = []
    x_max = 0
    stall = 0
    lastx = 0
    live_beat = False
    cause = "timeout"
    try:
        for _ in range(max_chunks):
            a = ctrl.act(sim.ram, sim.last_info)
            path.append(a)
            info, done = sim.run_chunk(a, cf)
            x_max = max(x_max, mario_level_x(sim.ram))
            if is_success(info):
                live_beat = True
                cause = "flag"
                break
            if is_death(info, done) or done:
                cause = death_cause(info)
                break
            x = mario_level_x(sim.ram)
            stall = stall + 1 if x <= lastx else 0
            lastx = max(lastx, x)
            if stall > 50:
                cause = "stall"
                break
    finally:
        sim.close()

    replay_verified, replay_reason = (
        replay_verify_stock_path(w, s, path, cf, seed=seed)
        if live_beat else (False, "live_rollout_did_not_reach_flag")
    )
    if live_beat and not replay_verified:
        cause = "replay_verification_failed"
    return {
        "seed": seed,
        "root_ram_sha256": root_ram_sha256,
        "chunk_frames": cf,
        "path": path,
        "path_length": len(path),
        "live_beat": live_beat,
        "beat": replay_verified,
        "replay_verified": replay_verified,
        "replay_reason": replay_reason,
        "x_max": int(x_max),
        "cause": cause,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--config", default="")
    ap.add_argument("--levels", default="")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--cf", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=500)
    ap.add_argument("--tag", default="generalist_heldout")
    args = ap.parse_args()
    if args.seeds <= 0 or args.cf <= 0 or args.max_chunks <= 0:
        ap.error("--seeds, --cf, and --max-chunks must be positive")

    cfg = json.loads(Path(args.config).read_text()) if args.config else None
    if args.levels:
        levels = [tuple(int(x) for x in p.split("-")) for p in args.levels.split(",")]
    elif cfg:
        levels = [tuple(int(x) for x in p.split("-")) for p in cfg["holdout_levels"]]
    else:
        raise SystemExit("provide --config or --levels")

    net, ckpt_meta = load_policy_checkpoint(args.ckpt, device="cpu")
    if cfg:
        expected_train = sorted(cfg["train_levels"])
        expected_holdout = sorted(cfg["holdout_levels"])
        actual_train = sorted(ckpt_meta.get("train_levels", []))
        actual_holdout = sorted(ckpt_meta.get("holdout_levels", []))
        if actual_train != expected_train or actual_holdout != expected_holdout:
            raise SystemExit(
                "checkpoint split metadata does not match --config: "
                f"train={actual_train} holdout={actual_holdout}")
    ctrl = EntityController(net, device="cpu")
    rid = new_run_id(args.tag); d = run_dir(rid)

    out = {
        "schema_version": 2,
        "generated_at": utc_now_iso(),
        "git_rev": git_rev(),
        "environment": env_fingerprint(),
        "checkpoint": checkpoint_ref(args.ckpt),
        "checkpoint_metadata": ckpt_meta,
        "cf": args.cf,
        "max_chunks": args.max_chunks,
        "seeds": list(range(args.seeds)),
        "levels": {},
    }
    total_beat = total = 0
    for (w, s) in levels:
        end_x = level_end_x(w, s)
        runs = [rollout(ctrl, w, s, seed, args.cf, args.max_chunks) for seed in range(args.seeds)]
        n_beat = sum(r["beat"] for r in runs)
        n_live_beat = sum(r["live_beat"] for r in runs)
        fracs = [min(1.0, r["x_max"] / max(1, end_x)) for r in runs]
        causes = {}
        for r in runs:
            if not r["beat"]:
                causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        best = max(runs, key=lambda r: (r["beat"], r["x_max"]))
        sheet = d / f"contact_{w}-{s}.png"
        try:
            make_contact_sheet(w, s, best["seed"], best["path"], args.cf, sheet, cols=6, rows=4)
        except Exception as e:
            print(f"  WARN sheet {w}-{s}: {e}")
        out["levels"][f"{w}-{s}"] = {
            "completion_rate": n_beat / args.seeds,
            "n_beat": int(n_beat),
            "n_live_beat": int(n_live_beat),
            "n_rollouts": args.seeds,
            "median_completion_frac": float(np.median(fracs)),
            "deaths_by_cause": causes,
            "distinct_root_ram_count": len({
                r["root_ram_sha256"] for r in runs}),
            "root_fingerprint_scope": (
                "NES CPU RAM only; duplicate hashes expose duplicate starts, "
                "but distinct hashes are not proof of statistical independence"),
            "rollouts": runs,
        }
        total_beat += n_beat; total += args.seeds
        print(f"  {w}-{s}: beat {n_beat}/{args.seeds}  median_frac="
              f"{np.median(fracs):.2f}  causes={causes}", flush=True)

    out["overall_completion_rate"] = total_beat / max(1, total)
    out["n_levels"] = len(levels)
    # NOTE: write to heldout_eval.json (NOT eval.json) so this generalization artifact does
    # not collide with the V2/V3 policy-eval contract that scripts/update_status.py scans.
    write_json_atomic(d / "heldout_eval.json", out)
    print(f"\nOVERALL: {total_beat}/{total} rollouts beat  "
          f"({out['overall_completion_rate']:.2%})\nwrote {d/'heldout_eval.json'}", flush=True)


if __name__ == "__main__":
    main()
