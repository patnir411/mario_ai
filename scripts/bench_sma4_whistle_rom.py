"""Run the SMA4 ROM-backed whistle option benchmark.

This is the ROM substitution for the symbolic `use_whistle` benchmark:
the whistle spend and warp-zone navigation options are effect-opaque and backed
by real Stable-Retro/mGBA execution.  World-advance and Bowser remain symbolic
planning-layer stand-ins; this pass tests whether resettable meta-search
discovers the W1->W8 payoff from actual RAM post-states.

Run:
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/bench_sma4_whistle_rom.py
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/bench_sma4_whistle_rom.py --warpless-blocked
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter  # noqa: E402
from mario.meta_planner import (  # noqa: E402
    SMA4WhistleROMConfig,
    build_sma4_whistle_rom_library,
    greedy_plan,
    search_options_uniform_cost,
)
from mario.options import (  # noqa: E402
    KnowledgeTier,
    MetaSearchResult,
    MetaState,
    OptionContext,
    SMA4WhistleExecutor,
    search_options,
)
from mario.provenance import StateAliasError  # noqa: E402


def _goal(state: MetaState) -> bool:
    return "beat_game" in state.flags


def _fresh_context() -> tuple[SMA4Adapter, MetaState, OptionContext]:
    core = SMA4Adapter(boot_target="overworld")
    for _ in range(60):
        core._step_buttons(())
    cursor = tuple(core.last_info.get("cursor", (32, 64)))
    start = MetaState(world=1, node=cursor)
    executor = SMA4WhistleExecutor(core)
    context = OptionContext(executor=executor, snapshots={start: executor.snapshot()})
    return core, start, context


def _discovered_rom_skip(result: MetaSearchResult) -> bool:
    return any(
        d.get("option") == "select_world8_pipe"
        and d.get("raw_world") == SMA4WhistleExecutor.WORLD_8_RAW
        for d in result.discovered_effects
    )


def _report(name: str, result: MetaSearchResult) -> dict:
    return {
        "planner": name,
        "found": result.found,
        "plan": result.path,
        "hops": len(result.path),
        "frames": result.total_cost.frames,
        "branches": result.branches,
        "option_calls": result.option_calls,
        "injected_facts": list(result.injected_facts),
        "attempted_injected_facts": list(result.attempted_injected_facts),
        "discovered_effects": result.discovered_effects,
        "option_boundaries": result.boundaries,
        "path_evidence": result.path_evidence,
        "alias_collisions": result.alias_collisions,
        "discovered_whistle_skip": _discovered_rom_skip(result),
        "used_whistle": any(p.startswith("use_whistle") for p in result.path),
        "states": [s.to_json() for s in result.states],
        "attempt_log": result.log,
    }


def _run_one(planner: str, cfg: SMA4WhistleROMConfig,
             max_tier: KnowledgeTier) -> dict:
    core, start, context = _fresh_context()
    try:
        lib = build_sma4_whistle_rom_library(cfg)
        try:
            if planner == "greedy":
                result = greedy_plan(lib, start, _goal, max_tier=max_tier,
                                     context=context)
            elif planner == "bfs":
                result = search_options(lib, start, _goal, max_tier=max_tier,
                                        context=context, max_depth=16)
            elif planner == "uniform_cost":
                result = search_options_uniform_cost(
                    lib, start, _goal, max_tier=max_tier, context=context,
                    max_depth=16)
            else:
                raise ValueError(planner)
        except StateAliasError as exc:
            return {
                "planner": planner,
                "found": False,
                "plan": [],
                "hops": 0,
                "frames": 0,
                "branches": 0,
                "option_calls": 0,
                "injected_facts": [],
                "attempted_injected_facts": [],
                "discovered_effects": [],
                "option_boundaries": [],
                "path_evidence": [],
                "alias_collisions": list(context.alias_collisions),
                "discovered_whistle_skip": False,
                "used_whistle": False,
                "states": [start.to_json()],
                "attempt_log": [],
                "hard_gate_failure": {
                    "kind": "StateAliasError",
                    "message": str(exc),
                    "collision": exc.collision,
                },
            }
        return _report(planner, result)
    finally:
        core.close()


def run_benchmark(cfg: SMA4WhistleROMConfig, *,
                  max_tier: KnowledgeTier) -> dict:
    reports = {
        name: _run_one(name, cfg, max_tier)
        for name in ("greedy", "bfs", "uniform_cost")
    }
    greedy_frames = reports["greedy"]["frames"] if reports["greedy"]["found"] else None
    cheapest = min(
        (r["frames"] for r in reports.values()
         if r["found"] and r["used_whistle"]),
        default=None,
    )
    speedup = (greedy_frames / cheapest) if (greedy_frames and cheapest) else None
    return {
        "config": {
            "world_advance_frames": cfg.world_advance_frames,
            "bowser_frames": cfg.bowser_frames,
            "warpless_blocked": cfg.warpless_blocked,
            "hand_granted": cfg.hand_granted,
            "acquire_whistle_1_3": not cfg.hand_granted,
            "acquire_whistle_fortress": not cfg.hand_granted,
            "max_tier": int(max_tier),
            "note": (
                "whistle spend and warp-zone navigation are ROM-backed; "
                "warpless advance and Bowser are symbolic stand-ins; "
                "hand_granted=True supplies two inventory whistles (no warp-zone "
                "re-grant); hand_granted=False uses Tier-2 acquire_whistle_1_3 "
                "+ acquire_whistle_fortress (independent P-Wing/leaf roots; "
                "inventory merge stacks the second 0x0C); boundary reports "
                "separate declared/effective tiers and all interventions"),
        },
        "rom": {
            "game": "Super Mario Advance 4 / SMB3",
            "emulator": "Stable-Retro/mGBA",
            "world_raw_world8": SMA4WhistleExecutor.WORLD_8_RAW,
            "world_raw_warp_zone": SMA4WhistleExecutor.WARP_ZONE_WORLD_RAW,
            "inventory_start": hex(SMA4WhistleExecutor.INVENTORY_START),
            "warp_whistle_item_id": hex(SMA4WhistleExecutor.WARP_WHISTLE),
        },
        "planners": reports,
        "contrast": {
            "greedy_found_goal": reports["greedy"]["found"],
            "greedy_discovered_skip": reports["greedy"]["discovered_whistle_skip"],
            "search_discovered_skip": reports["uniform_cost"]["discovered_whistle_skip"],
            "greedy_frames": greedy_frames,
            "cheapest_whistle_frames": cheapest,
            "skip_cost_reduction_x": round(speedup, 2) if speedup else None,
            "thesis": (
                (
                    "myopic planning cannot value the opaque whistle route; "
                    "resettable meta-search executes the ROM-backed options, "
                    "observes raw world 7 (World 8), and exploits the payoff"
                )
                if reports["uniform_cost"]["discovered_whistle_skip"]
                else (
                    "no whistle route was accepted because a physical-state "
                    "alias failed closed; inspect hard_gate_failure and its "
                    "retained/rejected physical states before making a "
                    "planner-comparison claim"
                )
                if reports["uniform_cost"].get("hard_gate_failure")
                else (
                    "no whistle route was accepted under the requested "
                    "knowledge tier; any found route is the symbolic warpless "
                    "fallback, so inspect attempted transitions before making "
                    "a planner-comparison claim"
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warpless-blocked", action="store_true",
                        help="remove symbolic warpless advance options")
    parser.add_argument("--no-hand-grant", action="store_true",
                        help="use Tier-2 acquire_whistle_1_3 + fortress instead of RAM poke")
    parser.add_argument("--max-tier", type=int, default=None,
                        help="max knowledge tier (default 1, or 2 with --no-hand-grant)")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir (default runs/<ts>-sma4-whistle-rom-bench)")
    args = parser.parse_args()

    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")

    hand_granted = not args.no_hand_grant
    if args.max_tier is None:
        max_tier = (KnowledgeTier.TIER2_BLACK_BOX_OPTION if args.no_hand_grant
                    else KnowledgeTier.TIER1_ITEM_GIVEN)
    else:
        max_tier = KnowledgeTier(args.max_tier)
    cfg = SMA4WhistleROMConfig(
        warpless_blocked=args.warpless_blocked,
        hand_granted=hand_granted,
    )
    report = run_benchmark(cfg, max_tier=max_tier)

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{ts}-sma4-whistle-rom-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("=" * 86)
    print("SMA4 ROM-backed whistle option benchmark")
    print("=" * 86)
    print(f"{'planner':<14}{'found':<7}{'hops':<6}{'frames':<10}"
          f"{'branches':<10}{'calls':<8}{'skip?':<7}{'plan'}")
    print("-" * 86)
    for name in ("greedy", "bfs", "uniform_cost"):
        row = report["planners"][name]
        print(f"{name:<14}{str(row['found']):<7}{row['hops']:<6}"
              f"{row['frames']:<10}{row['branches']:<10}"
              f"{row['option_calls']:<8}{str(row['discovered_whistle_skip']):<7}"
              f"{'->'.join(row['plan'])}")
    print("-" * 86)
    contrast = report["contrast"]
    print(f"greedy frames           : {contrast['greedy_frames']}")
    print(f"cheapest whistle frames : {contrast['cheapest_whistle_frames']}")
    reduction = contrast["skip_cost_reduction_x"]
    print(f"skip cost reduction     : {reduction}x" if reduction is not None
          else "skip cost reduction     : n/a")
    print(f"greedy discovered skip  : {contrast['greedy_discovered_skip']}")
    print(f"search discovered skip  : {contrast['search_discovered_skip']}")
    print()
    print(contrast["thesis"])
    print()
    print("report:", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
