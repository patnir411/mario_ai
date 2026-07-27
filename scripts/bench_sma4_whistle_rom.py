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
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter  # noqa: E402
from mario.meta_planner import (  # noqa: E402
    SMA4WhistleROMConfig,
    build_sma4_whistle_rom_library,
    greedy_plan,
)
from mario.options import (  # noqa: E402
    KnowledgeTier,
    MetaSearchResult,
    MetaState,
    OptionContext,
    SMA4WhistleExecutor,
)
from mario.physical_planner import (  # noqa: E402
    search_physical_options,
    search_physical_options_uniform_cost,
)
from mario.provenance import StateAliasError  # noqa: E402


def _goal(state: MetaState) -> bool:
    return "beat_game" in state.flags


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_artifact(package: str, filename_prefix: str) -> dict:
    matches = [
        Path(entry.locate())
        for entry in (importlib.metadata.files(package) or ())
        if Path(str(entry)).name.startswith(filename_prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one installed {package} artifact beginning "
            f"{filename_prefix!r}, "
            f"found {len(matches)}"
        )
    return {
        "basename": matches[0].name,
        "sha256": _file_hash(matches[0], "sha256"),
    }


def _source_control() -> dict:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    untracked_raw = subprocess.check_output([
        "git", "ls-files", "--others", "--exclude-standard", "-z",
    ])
    untracked = sorted(
        path.decode("utf-8")
        for path in untracked_raw.split(b"\0")
        if path
    )
    source_hash = hashlib.sha256()
    source_hash.update(patch)
    for relative in untracked:
        path = Path(relative)
        source_hash.update(relative.encode("utf-8"))
        source_hash.update(path.read_bytes())
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(status.strip()),
        "git_status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_source_files": untracked,
        "working_source_sha256": source_hash.hexdigest(),
    }


def _fresh_context(*, alias_policy: str) -> tuple[
        SMA4Adapter, MetaState, OptionContext]:
    core = SMA4Adapter(boot_target="overworld")
    for _ in range(60):
        core._step_buttons(())
    cursor = tuple(core.last_info.get("cursor", (32, 64)))
    start = MetaState(world=1, node=cursor)
    executor = SMA4WhistleExecutor(core)
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy=alias_policy,
    )
    return core, start, context


def _discovered_rom_skip(result: MetaSearchResult) -> bool:
    return any(
        d.get("option") == "select_world8_pipe"
        and d.get("raw_world") == SMA4WhistleExecutor.WORLD_8_RAW
        and d.get("world8_acceptance") is True
        for d in result.discovered_effects
    )


def _record_histories(result: MetaSearchResult) -> dict[int, list[tuple[str, ...]]]:
    registry = {
        int(row["record_id"]): row
        for row in result.representative_registry
    }
    memo: dict[int, list[tuple[str, ...]]] = {}

    def histories(record_id: int, seen: frozenset[int] = frozenset()):
        if record_id in memo:
            return memo[record_id]
        if record_id in seen:
            return []
        rows = []
        for arrival in registry.get(record_id, {}).get("arrivals", []):
            parent = arrival.get("parent_record_id")
            option = arrival.get("option")
            if parent is None or option is None:
                rows.append(())
                continue
            for prefix in histories(int(parent), seen | {record_id}):
                rows.append((*prefix, str(option)))
        memo[record_id] = sorted(set(rows))
        return memo[record_id]

    for record_id in registry:
        histories(record_id)
    return memo


def _acquisition_order_matrix(result: MetaSearchResult) -> dict:
    histories = _record_histories(result)
    blocks = result.partition_refinement.get("record_to_block", {})
    labels = {
        (
            "acquire_whistle_1_3",
            "acquire_whistle_fortress",
        ): "1-3_then_fortress",
        (
            "acquire_whistle_fortress",
            "acquire_whistle_1_3",
        ): "fortress_then_1-3",
    }
    matrix = {
        label: {
            "representatives_expanded": 0,
            "select_world8_attempts": [],
        }
        for label in labels.values()
    }
    for observation in result.transition_observations:
        if (
            observation.get("option") != "select_world8_pipe"
            or observation.get("applicability") != "enabled"
        ):
            continue
        record_id = observation.get("record_id")
        if record_id is None:
            continue
        for history in histories.get(int(record_id), []):
            acquisition_history = tuple(
                option for option in history
                if option in (
                    "acquire_whistle_1_3",
                    "acquire_whistle_fortress",
                )
            )
            label = labels.get(acquisition_history)
            if label is None:
                continue
            outcome = observation.get("outcome_signature") or {}
            row = {
                "record_id": int(record_id),
                "refined_block": blocks.get(str(record_id)),
                "option_history": list(history),
                "accepted": bool(outcome.get("accepted")),
                "termination_reason": outcome.get("termination_reason"),
                "repeat_conformant": observation.get(
                    "repeat_conformant"),
                "repeat_count": observation.get("repeat_count"),
                "terminal_invariants": outcome.get(
                    "terminal_invariants", {}),
                "reported_cost": outcome.get("reported_cost"),
                "retained_frames": outcome.get("retained_frames"),
            }
            if row not in matrix[label]["select_world8_attempts"]:
                matrix[label]["select_world8_attempts"].append(row)
    for value in matrix.values():
        value["representatives_expanded"] = len({
            row["record_id"] for row in value["select_world8_attempts"]
        })
        value["accepted"] = any(
            row["accepted"] for row in value["select_world8_attempts"]
        )
    matrix["both_orders_expanded"] = all(
        matrix[label]["representatives_expanded"] > 0
        for label in labels.values()
    )
    matrix["both_orders_accepted"] = all(
        matrix[label]["accepted"] for label in labels.values()
    )
    return matrix


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
        "visited": result.visited,
        "representative_path": result.physical_record_ids,
        "physical_visited": result.physical_visited,
        "search_nodes_visited": result.search_nodes_visited,
        "representative_registry": result.representative_registry,
        "transition_observations": result.transition_observations,
        "partition_refinement": result.partition_refinement,
        "repeat_conformance_failures": (
            result.repeat_conformance_failures
        ),
        "evaluation_work": result.evaluation_work,
        "acquisition_order_matrix": _acquisition_order_matrix(result),
    }


def _run_one(planner: str, cfg: SMA4WhistleROMConfig,
             max_tier: KnowledgeTier, *,
             reverse_option_insertion: bool = False) -> dict:
    multi = planner in ("bfs", "uniform_cost")
    core, start, context = _fresh_context(
        alias_policy="multi" if multi else "error")
    try:
        lib = build_sma4_whistle_rom_library(cfg)
        if reverse_option_insertion:
            lib.options = dict(reversed(list(lib.options.items())))
        try:
            if planner == "greedy":
                result = greedy_plan(lib, start, _goal, max_tier=max_tier,
                                     context=context)
            elif planner == "bfs":
                result = search_physical_options(
                    lib,
                    start,
                    _goal,
                    max_tier=max_tier,
                    context=context,
                    max_depth=16,
                    determinism_repeats=2,
                    exhaustive=True,
                )
            elif planner == "uniform_cost":
                result = search_physical_options_uniform_cost(
                    lib,
                    start,
                    _goal,
                    max_tier=max_tier,
                    context=context,
                    max_depth=16,
                    determinism_repeats=2,
                    exhaustive=True,
                )
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
                "visited": 1,
                "representative_path": [],
                "physical_visited": 1,
                "search_nodes_visited": 1,
                "representative_registry": [],
                "transition_observations": [],
                "partition_refinement": {},
                "repeat_conformance_failures": [],
                "evaluation_work": {},
                "acquisition_order_matrix": {},
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
                  max_tier: KnowledgeTier,
                  reverse_option_insertion: bool = False) -> dict:
    reports = {
        name: _run_one(
            name,
            cfg,
            max_tier,
            reverse_option_insertion=reverse_option_insertion,
        )
        for name in ("greedy", "bfs", "uniform_cost")
    }
    greedy_frames = reports["greedy"]["frames"] if reports["greedy"]["found"] else None
    cheapest = min(
        (r["frames"] for r in reports.values()
         if r["found"] and r["discovered_whistle_skip"]),
        default=None,
    )
    speedup = (greedy_frames / cheapest) if (greedy_frames and cheapest) else None
    return {
        "schema": "sma4-physical-option-benchmark-v2",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "invocation": {
            "python_executable": Path(sys.executable).name,
            "argv": list(sys.argv),
        },
        "source_control": _source_control(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "stable_retro": importlib.metadata.version("stable-retro"),
            "stable_retro_extension": _installed_artifact(
                "stable-retro", "_retro."),
            "mgba_core": _installed_artifact(
                "stable-retro", "mgba_libretro."),
        },
        "config": {
            "world_advance_frames": cfg.world_advance_frames,
            "bowser_frames": cfg.bowser_frames,
            "warpless_blocked": cfg.warpless_blocked,
            "hand_granted": cfg.hand_granted,
            "acquire_whistle_1_3": not cfg.hand_granted,
            "acquire_whistle_fortress": not cfg.hand_granted,
            "max_tier": int(max_tier),
            "reverse_option_insertion": bool(reverse_option_insertion),
            "note": (
                "whistle spend and warp-zone navigation are ROM-backed; "
                "warpless advance and Bowser are symbolic stand-ins; "
                "hand_granted=True supplies two inventory whistles (no warp-zone "
                "re-grant); hand_granted=False uses Tier-2 acquire_whistle_1_3 "
                "+ acquire_whistle_fortress (independent P-Wing/leaf roots; "
                "inventory merge stacks the second 0x0C); boundary reports "
                "separate declared/effective tiers and all interventions; "
                "BFS/UCS preserve physical representatives, repeat each "
                "admissible option twice, and refine only finite-library "
                "behavioral signatures"),
        },
        "rom": {
            "game": "Super Mario Advance 4 / SMB3",
            "emulator": "Stable-Retro/mGBA",
            "file_basename": Path(
                os.environ["MARIO_AI_SMA4_ROM"]).name,
            "sha1": _file_hash(
                Path(os.environ["MARIO_AI_SMA4_ROM"]), "sha1"),
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
                    "physical-representative meta-search executes the ROM-backed "
                    "options and accepts only an input-responsive raw-world-7 "
                    "World 8 successor"
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
    parser.add_argument(
        "--reverse-option-insertion",
        action="store_true",
        help="reverse library insertion order (physical planners canonicalize IDs)",
    )
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
    report = run_benchmark(
        cfg,
        max_tier=max_tier,
        reverse_option_insertion=args.reverse_option_insertion,
    )

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
