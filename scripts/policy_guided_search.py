"""Matched test of whether a learned policy prior makes emulator search cheaper.

A policy can contribute a soft ranking bonus or an explicitly incomplete top-k
pruning rule.  The latter may reduce branching, but can also remove the only
successful action; this script measures that trade-off rather than assuming it.
The preserved local result is one 1-1 comparison, not a general theorem or an
iterative AlphaZero-style training loop.

This driver runs the SAME level under matched configs and reports solved / nodes / seconds:
  - plain beam/coverage at a given width
  - guided beam/coverage: same width, with policy and/or value score bonuses
A win = policy-guided solves with FEWER nodes, or a narrow policy-guided beam solves where the
same-width plain beam fails (the deceptive/hard-level payoff).

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/policy_guided_search.py \
        --ckpt data/specialists/1-1.pt --level 1-1 [--width 6] [--topk 3] [--depth 400]

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/policy_guided_search.py \
        --search coverage --level 6-2 --ckpt data/generalist.pt \
        --value-ckpt runs/value/value.pt --width 160 --budget 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.entity_policy import (
    EntityPolicyPrior,
    TemporalEntityPolicyPrior,
    load_policy_checkpoint,
)
from mario.io import env_fingerprint, git_rev, utc_now_iso, write_json_atomic
from mario.render import replay
from mario.search import beam_search, coverage_search
from mario.value import ValueGuide, load_value


def run(search_kind, world, stage, *, seed: int, chunk_frames: int, **kw):
    fn = coverage_search if search_kind == "coverage" else beam_search
    r = fn(world, stage, seed=seed, chunk_frames=chunk_frames, **kw)
    replay_verified = False
    if r.solved:
        records = replay(world, stage, seed, list(r.path), chunk_frames)
        replay_verified = any(bool(record.get("flag")) for record in records)
    return {
        "search_solved": bool(r.solved),
        "solved": replay_verified,
        "replay_verified": replay_verified,
        "x": int(r.x_max),
        "nodes": int(r.nodes_expanded),
        "depth": int(r.depth_reached),
        "secs": round(r.wall_clock_s, 1),
        "path": list(r.path),
        "path_length": len(r.path),
        "chunk_frames": chunk_frames,
    }


def load_prior(path: str, temp: float):
    net, metadata = load_policy_checkpoint(path, device="cpu")
    if getattr(net, "cfg", {}).get("ctx_k"):
        prior = TemporalEntityPolicyPrior(net, device="cpu", temperature=temp)
        print(f"  (temporal prior, ctx_k={net.cfg['ctx_k']})", flush=True)
    else:
        prior = EntityPolicyPrior(net, device="cpu", temperature=temp)
    return prior, metadata


def file_ref(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        name = str(p.resolve().relative_to(ROOT))
    except ValueError:
        name = str(p)
    return {"path": name, "sha256": digest, "bytes": p.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", choices=["beam", "coverage"], default="beam")
    ap.add_argument("--ckpt")
    ap.add_argument("--level", required=True)
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--topk", type=int)
    ap.add_argument("--depth", type=int, default=400)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--cf", type=int, default=8)
    ap.add_argument("--policy-weight", type=float, default=80.0)
    ap.add_argument("--policy-mix-eps", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--value-ckpt")
    ap.add_argument(
        "--training-data", action="append", default=[],
        help="optional ignored training-data artifact to identify by path/hash; repeatable")
    ap.add_argument("--value-weight", type=float, default=600.0)
    ap.add_argument("--cov-bonus", type=float, default=30.0)
    ap.add_argument("--stuck-cap", type=int, default=80)
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--json-out", default="",
        help="optional durable JSON report path (checkpoints remain external)")
    args = ap.parse_args()
    world, stage = (int(x) for x in args.level.split("-"))
    if args.width <= 0 or args.depth <= 0 or args.cf <= 0:
        ap.error("--width, --depth, and --cf must be positive")
    if args.topk is not None and args.topk <= 0:
        ap.error("--topk must be positive")

    if not args.ckpt and not args.value_ckpt:
        ap.error("provide --ckpt and/or --value-ckpt for the guided run")

    prior, prior_metadata = (
        load_prior(args.ckpt, args.temp) if args.ckpt else (None, None))
    value = None
    value_metadata = None
    if args.value_ckpt:
        net, value_metadata = load_value(args.value_ckpt, device="cpu")
        value = ValueGuide(net, device="cpu")
        print(
            f"  (value guide, metrics={value_metadata.get('val_metrics', {})})",
            flush=True)

    if args.search == "coverage":
        common = dict(beam_width=args.width, max_depth=args.depth, time_budget_s=args.budget,
                      cov_bonus=args.cov_bonus,
                      stuck_cap=args.stuck_cap, progress_every=args.progress_every)
        topk = args.topk
    else:
        common = dict(beam_width=args.width, max_depth=args.depth,
                      stuck_cap=args.stuck_cap, progress_every=args.progress_every)
        topk = (args.topk if args.topk is not None else 3) if prior is not None else None
    print(f"== guided {args.search} {args.level} width={args.width} "
          f"topk={topk if topk is not None else 'none'} depth={args.depth} ==", flush=True)

    plain = run(
        args.search, world, stage, seed=args.seed, chunk_frames=args.cf, **common)
    print(f"  plain         : {plain}", flush=True)

    guided_kw = dict(common)
    if prior is not None:
        guided_kw.update(policy_prior=prior, policy_weight=args.policy_weight,
                         policy_topk=topk)
        if args.search == "coverage":
            guided_kw["policy_mix_eps"] = args.policy_mix_eps
    if value is not None:
        guided_kw.update(value_guide=value, value_weight=args.value_weight)
    guided = run(
        args.search, world, stage, seed=args.seed, chunk_frames=args.cf, **guided_kw)
    print(f"  policy-guided : {guided}", flush=True)

    ratio = None
    verdict = "neither_solved"
    if plain["solved"] and guided["solved"]:
        ratio = plain["nodes"] / max(1, guided["nodes"])
        print(f"  => both solved; node ratio plain/guided = {ratio:.2f}x", flush=True)
        verdict = "both_solved"
    elif guided["solved"] and not plain["solved"]:
        print("  => policy-guided SOLVED where plain FAILED (the hard-level payoff)", flush=True)
        verdict = "guided_only"
    elif plain["solved"] and not guided["solved"]:
        print("  => plain solved, guided did not (prior too restrictive — raise topk)", flush=True)
        verdict = "plain_only"
    else:
        print("  => neither solved at this width/depth", flush=True)
    report = {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "git_rev": git_rev(),
        "source": {
            "driver": file_ref(str(Path(__file__))),
            "actions": file_ref(str(ROOT / "mario" / "actions.py")),
            "environment": file_ref(str(ROOT / "mario" / "env.py")),
            "entity_observation": file_ref(str(ROOT / "mario" / "entity.py")),
            "policy": file_ref(str(ROOT / "mario" / "entity_policy.py")),
            "search": file_ref(str(ROOT / "mario" / "search.py")),
            "reward": file_ref(str(ROOT / "mario" / "reward.py")),
            "replay": file_ref(str(ROOT / "mario" / "render.py")),
        },
        "environment": env_fingerprint(),
        "seed": args.seed,
        "search": args.search,
        "level": args.level,
        "config": {
            "width": args.width,
            "topk": topk,
            "depth": args.depth,
            "budget_s": args.budget if args.search == "coverage" else None,
            "chunk_frames": args.cf,
            "policy_weight": args.policy_weight,
            "policy_mix_eps": (
                args.policy_mix_eps if args.search == "coverage" else None),
            "temperature": args.temp,
            "value_weight": args.value_weight,
            "coverage_bonus": (
                args.cov_bonus if args.search == "coverage" else None),
            "stuck_cap": args.stuck_cap,
        },
        "policy_checkpoint": file_ref(args.ckpt),
        "policy_checkpoint_metadata": prior_metadata,
        "training_data": [file_ref(path) for path in args.training_data],
        "value_checkpoint": file_ref(args.value_ckpt),
        "value_checkpoint_metadata": value_metadata,
        "plain": plain,
        "guided": guided,
        "node_ratio_plain_over_guided": ratio,
        "verdict": verdict,
        "reproducibility_caveat": (
            "The checkpoint is identified by hash but is intentionally not "
            "committed; training provenance must be recreated before this is "
            "a clean-clone reproducible result."
        ),
    }
    print(json.dumps(report), flush=True)
    if args.json_out:
        write_json_atomic(Path(args.json_out), report)
        print(f"  wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
