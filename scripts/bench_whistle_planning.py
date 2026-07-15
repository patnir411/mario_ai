"""Run the SMB3 (SMA4) option-MDP whistle-planning benchmark.

Compares planner classes (greedy / BFS / uniform-cost) on the same
effect-opaque whistle option library and reports, per planner: whether it
reached the goal, its plan, hops, symbolic frame cost, branches, option calls,
injected knowledge facts, and any opaque effects it discovered.

The headline result is the `contrast`: a myopic planner finishes the game the
long warpless way but never discovers the warp-whistle skip, while resettable
search discovers the opaque whistle payoff (World 1 -> World 8) and returns a
much cheaper plan.

This is the planning-layer benchmark with *symbolic* option costs; the option
endpoints and the whistle spend mechanic are ROM-verified separately
(`scripts/probe_sma4_whistle.py`, the replay-verified ClearLevel options).

Run:
    ./venv/bin/python scripts/bench_whistle_planning.py
    ./venv/bin/python scripts/bench_whistle_planning.py --warpless-blocked
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.options import KnowledgeTier  # noqa: E402
from mario.meta_planner import (  # noqa: E402
    WhistleBenchmarkConfig,
    run_whistle_benchmark,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warpless-blocked", action="store_true",
                    help="remove the warpless route so greedy is stuck and only "
                         "exploratory search (via the whistle) reaches the goal")
    ap.add_argument("--real-acquire", action="store_true",
                    help="model whistle acquisition as a real (non-hand-granted) "
                         "Tier-2 option with a frame cost")
    ap.add_argument("--max-tier", type=int, default=int(KnowledgeTier.TIER2_BLACK_BOX_OPTION),
                    help="max knowledge tier of options the planners may use")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default runs/<ts>-whistle-planning-bench)")
    args = ap.parse_args()

    cfg = WhistleBenchmarkConfig(
        warpless_blocked=args.warpless_blocked,
        hand_granted=not args.real_acquire,
        whistle_tier=(KnowledgeTier.TIER2_BLACK_BOX_OPTION if args.real_acquire
                      else KnowledgeTier.TIER1_ITEM_GIVEN),
    )
    report = run_whistle_benchmark(cfg, max_tier=KnowledgeTier(args.max_tier))

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{ts}-whistle-planning-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # Human-readable table.
    print("=" * 78)
    print("SMB3 (SMA4) option-MDP whistle-planning benchmark")
    print("=" * 78)
    hdr = f"{'planner':<14}{'found':<7}{'hops':<6}{'frames':<10}{'branches':<10}{'skip?':<7}{'plan'}"
    print(hdr)
    print("-" * 78)
    for name in ("greedy", "bfs", "uniform_cost"):
        r = report["planners"][name]
        print(f"{name:<14}{str(r['found']):<7}{r['hops']:<6}{r['frames']:<10}"
              f"{r['branches']:<10}{str(r['discovered_whistle_skip']):<7}"
              f"{'->'.join(s.replace('advance_world_', 'w') for s in r['plan'])}")
    print("-" * 78)
    c = report["contrast"]
    print(f"greedy frames           : {c['greedy_frames']}")
    print(f"cheapest whistle frames : {c['cheapest_whistle_frames']}")
    print(f"skip cost reduction     : {c['skip_cost_reduction_x']}x")
    print(f"greedy discovered skip  : {c['greedy_discovered_skip']}")
    print(f"search discovered skip  : {c['search_discovered_skip']}")
    print()
    print(c["thesis"])
    print()
    print("report:", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
