"""Planner-class contrast for the SMB3 (SMA4) option-MDP whistle benchmark.

The option-MDP scaffold (`mario.options`) gives us a resettable search over
options, including *effect-opaque* options whose payoff must be discovered by
execution.  The research question is "how little meta-intelligence does
any%-via-whistle completion require once low-level control is exact?".  To
answer it honestly we compare *planner classes* on the SAME option library:

* a **greedy / myopic** planner that commits to the locally-best progress
  option and (crucially) will not gamble on an option whose effect it cannot
  predict — i.e. it never explores opaque-effect options.  This is the
  "no meta-intelligence" baseline.
* the existing **resettable BFS** (`search_options`) and a cost-optimal
  **uniform-cost** search added here, both of which *execute* opaque options
  and learn their effect from the post-state.

The thesis prediction, which the benchmark tests: the greedy planner completes
the game the long warpless way but never discovers the warp-whistle skip, while
resettable search discovers the opaque whistle payoff (World 1 -> World 8) and
returns a far cheaper plan.  The gap between them is the meta-intelligence the
skip requires.

Option *costs here are symbolic* (planning-layer stand-ins); the option
endpoints and the whistle spend mechanic are ROM-verified separately (the
`scripts/probe_sma4_whistle.py` recon and the replay-verified ClearLevel
options).  Wiring real per-option ROM costs in is the follow-up.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable

from mario.options import (
    KnowledgeTier,
    MetaState,
    MetaSearchResult,
    Option,
    OptionContext,
    OptionCost,
    OptionResult,
    OptionLibrary,
    search_options,
)

Heuristic = Callable[[MetaState], float]


def default_world_heuristic(state: MetaState) -> float:
    """Myopic 'distance to a beaten game' — lower is closer.

    A greedy planner using this climbs the world counter and finishes by
    clearing Bowser on World 8.  It has no way to value hoarding a one-time
    consumable, which is exactly the blind spot the benchmark exercises.
    """
    if "beat_game" in state.flags:
        return 0.0
    # World 8 reached but Bowser not yet cleared is one step from the goal.
    if state.world >= 8:
        return 1.0
    return float(8 - state.world) + 1.0


def greedy_plan(library: OptionLibrary, start: MetaState,
                goal: Callable[[MetaState], bool], *,
                heuristic: Heuristic = default_world_heuristic,
                max_steps: int = 64,
                max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT,
                exploit_opaque: bool = False,
                context: OptionContext | None = None) -> MetaSearchResult:
    """Hill-climb on `heuristic`; never backtracks.

    At each state it does a one-step lookahead over the *predictable* options
    (those with a declared, non-opaque effect unless ``exploit_opaque``) and
    commits to the one whose resulting state minimizes ``heuristic`` (ties
    broken by lower cost, then option id).  It halts at the goal, when no
    option improves the heuristic, or at ``max_steps``.

    Opaque-effect options are invisible to it by default: a planner with no
    exploration cannot value a payoff it cannot predict.  That is the modelled
    "no meta-intelligence" baseline, not a bug.
    """
    context = context or OptionContext()
    state = start
    path: list[str] = []
    states: list[MetaState] = [start]
    total = OptionCost()
    branches = 0
    option_calls = 0
    injected: set[str] = set()
    log: list[dict] = []
    seen: set[MetaState] = {start}

    for _ in range(max_steps):
        if goal(state):
            return _greedy_result(True, path, states, total, branches,
                                  option_calls, injected, log)
        candidates = [
            opt for opt in library.applicable(state, max_tier=max_tier)
            if exploit_opaque or not opt.opaque_effect
        ]
        best: tuple[float, float, str] | None = None
        best_result = None
        best_option = None
        cur_h = heuristic(state)
        for opt in candidates:
            branches += 1
            option_calls += 1
            injected.update(opt.injected_facts)
            # Symbolic options are pure; restore keeps ROM-backed trials clean.
            context.restore(state)
            result = opt.execute(state, context)
            row = {
                "from": state.to_json(),
                "option": opt.id,
                "success": bool(result.success),
                "knowledge_tier": int(opt.knowledge_tier),
                "opaque_effect": bool(opt.opaque_effect),
                "trial": True,
            }
            if not result.success:
                log.append(row)
                continue
            h = heuristic(result.state)
            row["heuristic"] = h
            log.append(row)
            key = (h, float(result.cost.frames), opt.id)
            # Only move if it strictly improves the heuristic (no lateral hops,
            # so the greedy planner cannot wander into the opaque whistle by
            # accident through cost-free side moves).
            if h < cur_h and (best is None or key < best):
                best = key
                best_result = result
                best_option = opt
        if best_option is None or best_result is None:
            break  # stuck: no predictable option improves progress
        state = best_result.state
        path.append(best_option.id)
        states.append(state)
        total = total + best_result.cost
        log.append({"commit": best_option.id, "to": state.to_json(),
                    "cost": best_result.cost.to_json()})
        if state in seen:
            break  # cycle guard
        seen.add(state)

    return _greedy_result(goal(state), path, states, total, branches,
                          option_calls, injected, log)


def _greedy_result(found, path, states, total, branches, option_calls,
                   injected, log) -> MetaSearchResult:
    return MetaSearchResult(
        found=bool(found),
        path=list(path),
        states=list(states),
        total_cost=total,
        branches=branches,
        option_calls=option_calls,
        injected_facts=tuple(sorted(injected)),
        discovered_effects=[],  # greedy never explores opaque options
        visited=len(states),
        log=log,
    )


def search_options_uniform_cost(
        library: OptionLibrary, start: MetaState,
        goal: Callable[[MetaState], bool], *,
        cost_of: Callable[[OptionCost], float] = lambda c: float(c.frames),
        max_depth: int = 32,
        max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT,
        context: OptionContext | None = None) -> MetaSearchResult:
    """Dijkstra over option costs; cost-optimal counterpart of `search_options`.

    Like `search_options` it executes opaque-effect options and records their
    discovered effect, but it returns the *cheapest* goal-reaching plan rather
    than the fewest-hops one — so "the whistle skip is cheaper" is a rigorous
    claim, not an artifact of hop counting.
    """
    context = context or OptionContext()
    counter = itertools.count()
    if goal(start):
        return MetaSearchResult(found=True, states=[start], visited=1)

    frontier: list[tuple[float, int, MetaState, list[str], list[MetaState], OptionCost]] = [
        (0.0, next(counter), start, [], [start], OptionCost())
    ]
    best_cost: dict[MetaState, float] = {start: 0.0}
    branches = 0
    option_calls = 0
    injected: set[str] = set()
    discovered: list[dict] = []
    log: list[dict] = []

    while frontier:
        gcost, _, state, path, states, cost = heapq.heappop(frontier)
        if gcost > best_cost.get(state, float("inf")):
            continue
        if goal(state):
            return MetaSearchResult(
                found=True, path=path, states=states, total_cost=cost,
                branches=branches, option_calls=option_calls,
                injected_facts=tuple(sorted(injected)),
                discovered_effects=discovered, visited=len(best_cost), log=log)
        if len(path) >= max_depth:
            continue
        for option in library.applicable(state, max_tier=max_tier):
            branches += 1
            option_calls += 1
            injected.update(option.injected_facts)
            context.restore(state)
            result = option.execute(state, context)
            row = {"from": state.to_json(), "option": option.id,
                   "success": bool(result.success),
                   "knowledge_tier": int(option.knowledge_tier),
                   "opaque_effect": bool(option.opaque_effect)}
            if not result.success:
                log.append(row)
                continue
            nxt = result.state
            row["to"] = nxt.to_json()
            row["cost"] = result.cost.to_json()
            if option.opaque_effect:
                observed = result.observed_effect or {
                    "from": state.to_json(), "to": nxt.to_json()}
                observed = {"option": option.id, **observed}
                discovered.append(observed)
                row["observed_effect"] = observed
            log.append(row)
            new_cost = cost + result.cost
            g = gcost + cost_of(result.cost)
            if g < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = g
                heapq.heappush(frontier, (g, next(counter), nxt,
                                          path + [option.id],
                                          states + [nxt], new_cost))

    return MetaSearchResult(
        found=False, branches=branches, option_calls=option_calls,
        injected_facts=tuple(sorted(injected)), discovered_effects=discovered,
        visited=len(best_cost), log=log)


# --------------------------------------------------------------------------
# Symbolic whistle benchmark library
# --------------------------------------------------------------------------

@dataclass
class WhistleBenchmarkConfig:
    """Symbolic per-option costs (planning-layer stand-ins, in frames)."""

    world_advance_frames: int = 9000   # ~clear one world warpless
    bowser_frames: int = 6000          # clear the World-8 castle
    whistle_use_frames: int = 300      # blow the whistle + warp
    whistle_acquire_frames: int = 1500
    whistle_tier: KnowledgeTier = KnowledgeTier.TIER1_ITEM_GIVEN
    hand_granted: bool = True          # Tier-1 inventory injection
    warpless_blocked: bool = False     # if True, no warpless path exists


@dataclass
class SMA4WhistleROMConfig:
    """ROM-backed whistle-spend benchmark knobs.

    The whistle spend / warp effects are real emulator transitions.  World
    advance and Bowser remain symbolic terminal stand-ins so this pass tests the
    W1->W8 meta payoff without re-solving World 8.

    When ``hand_granted`` is False, the first whistle comes from the verified
    Tier-2 ``acquire_whistle_1_3`` option (P-Wing 1-3 entry snapshot + white-block
    route).  The second whistle in the warp zone is still Tier-1 re-granted until
    a second AcquireWhistle option exists.
    """

    world_advance_frames: int = 9000
    bowser_frames: int = 6000
    warpless_blocked: bool = False
    hand_granted: bool = True


def _opt(id_, kind, pre, transform, *, tier=KnowledgeTier.TIER0_GENERIC,
         frames=0, opaque=False, observed=None, injected=()):
    def runner(state: MetaState, _ctx: OptionContext):
        from mario.options import OptionResult
        nxt = transform(state)
        if nxt is None:
            return OptionResult(False, state, cost=OptionCost(frames=frames))
        return OptionResult(True, nxt, cost=OptionCost(frames=frames),
                            observed_effect=dict(observed) if observed else None)

    return Option(
        id=id_, kind=kind, precondition=pre, runner=runner,
        knowledge_tier=tier, cost=OptionCost(frames=frames),
        opaque_effect=opaque, injected_facts=tuple(injected),
    )


def build_whistle_benchmark_library(
        config: WhistleBenchmarkConfig | None = None) -> OptionLibrary:
    """A symbolic SMB3 any%-via-whistle option library.

    Warpless route: advance world-by-world (1->...->8) then clear Bowser — fully
    predictable, so a greedy planner can follow it.  Whistle route:
    acquire a whistle (Tier-1 hand-granted by default) then USE it, where the
    use option is *effect-opaque* (its World-8 warp must be discovered by
    execution).  The whistle route is far cheaper but invisible to greedy.
    """
    cfg = config or WhistleBenchmarkConfig()
    lib = OptionLibrary()

    if not cfg.warpless_blocked:
        for w in range(1, 8):
            lib.add(_opt(
                f"advance_world_{w}_to_{w + 1}", "AdvanceWorld",
                pre=(lambda s, w=w: s.world == w),
                transform=(lambda s, w=w: s.with_world_node(w + 1, (0, 0))),
                frames=cfg.world_advance_frames,
            ))

    lib.add(_opt(
        "clear_bowser", "ClearBoss",
        pre=lambda s: s.world >= 8 and "beat_game" not in s.flags,
        transform=lambda s: s.with_flag("beat_game"),
        frames=cfg.bowser_frames,
    ))

    acquire_injected = ("whistle_hand_granted",) if cfg.hand_granted else ()
    lib.add(_opt(
        "acquire_whistle", "AcquireItem",
        pre=lambda s: s.world == 1 and not s.has_item("whistle"),
        transform=lambda s: s.with_item("whistle"),
        tier=cfg.whistle_tier,
        frames=0 if cfg.hand_granted else cfg.whistle_acquire_frames,
        injected=acquire_injected,
    ))

    # The flagship effect-opaque option: planner must DISCOVER it warps to W8.
    lib.add(_opt(
        "use_whistle", "UseItem",
        pre=lambda s: s.has_item("whistle"),
        transform=lambda s: s.without_item("whistle").with_world_node(8, (64, 80)),
        tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        frames=cfg.whistle_use_frames,
        opaque=True,
        observed={"to_world": 8, "spent": "whistle"},
        injected=("whistle_in_inventory",),
    ))
    return lib


def _state_with(state: MetaState, *, world: int | None = None,
                node: tuple[int, int] | None = None,
                inventory: tuple[str, ...] | None = None,
                flags: tuple[str, ...] = ()) -> MetaState:
    return MetaState(
        world=state.world if world is None else world,
        node=state.node if node is None else node,
        cleared=state.cleared,
        inventory=state.inventory if inventory is None else inventory,
        flags=state.flags + tuple(flags),
    )


def _last_sample(summary: dict) -> dict:
    samples = list(summary.get("samples") or [])
    return dict(samples[-1]) if samples else {}


def _observed_from_summary(summary: dict, *, label: str) -> dict:
    final = _last_sample(summary)
    return {
        "label": label,
        "raw_world": final.get("world_raw_0_indexed"),
        "world": final.get("world_normalized"),
        "cursor": final.get("cursor"),
        "mode": final.get("mode"),
        "samples": summary.get("samples", []),
    }


def build_sma4_whistle_rom_library(
        config: SMA4WhistleROMConfig | None = None) -> OptionLibrary:
    """Option library whose whistle effects are real SMA4 emulator transitions.

    The Tier-1 inventory grant is explicit and logged as injected knowledge.
    `use_whistle`, `use_whistle_again`, and `select_world8_pipe` are
    effect-opaque: their value comes only from executing against the emulator
    and reading the resulting RAM state.
    """
    cfg = config or SMA4WhistleROMConfig()
    lib = OptionLibrary()

    if not cfg.warpless_blocked:
        for w in range(1, 8):
            lib.add(_opt(
                f"advance_world_{w}_to_{w + 1}", "AdvanceWorld",
                pre=(lambda s, w=w: s.world == w),
                transform=(lambda s, w=w: s.with_world_node(w + 1, (0, 0))),
                frames=cfg.world_advance_frames,
            ))

    lib.add(_opt(
        "clear_bowser", "ClearBoss",
        pre=lambda s: s.world == 8 and "beat_game" not in s.flags,
        transform=lambda s: s.with_flag("beat_game"),
        frames=cfg.bowser_frames,
    ))

    def _need_executor(context: OptionContext):
        executor = context.executor
        if executor is None:
            return None
        required = ("use_first_whistle", "use_second_whistle",
                    "select_world8_pipe", "snapshot")
        if not all(hasattr(executor, name) for name in required):
            return None
        if cfg.hand_granted and not hasattr(executor, "grant_whistles"):
            return None
        if not cfg.hand_granted and not hasattr(executor, "acquire_whistle_1_3"):
            return None
        return executor

    def grant_runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor = _need_executor(context)
        if executor is None:
            return OptionResult(False, state)
        summary = executor.grant_whistles(2)
        next_state = _state_with(
            state,
            inventory=("whistle",),
            flags=("two_whistles_hand_granted",),
        )
        return OptionResult(
            success=bool(summary.get("success")),
            state=next_state,
            cost=OptionCost(frames=int(summary.get("cost_frames", 0))),
            exit_snapshot=executor.snapshot(),
            info=summary,
            observed_effect=_observed_from_summary(summary, label="grant_two_whistles"),
        )

    lib.add(Option(
        id="grant_two_whistles",
        kind="GrantInventory",
        precondition=lambda s: cfg.hand_granted and s.world == 1
        and not s.has_item("whistle"),
        runner=grant_runner,
        knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        cost=OptionCost(frames=0),
        opaque_effect=False,
        known_effect={"inventory_add": ["whistle"], "count": 2},
        injected_facts=("whistle_hand_granted", "two_whistles_hand_granted"),
    ))

    def acquire_runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor = _need_executor(context)
        if executor is None:
            return OptionResult(False, state)
        summary = executor.acquire_whistle_1_3()
        next_state = _state_with(
            state,
            inventory=("whistle",),
            flags=("whistle_acquired_1_3",),
        )
        return OptionResult(
            success=bool(summary.get("success")),
            state=next_state,
            cost=OptionCost(frames=int(summary.get("cost_frames", 0))),
            exit_snapshot=executor.snapshot(),
            info=summary,
            observed_effect=_observed_from_summary(
                summary, label="acquire_whistle_1_3"),
        )

    lib.add(Option(
        id="acquire_whistle_1_3",
        kind="AcquireItem",
        precondition=lambda s: (not cfg.hand_granted) and s.world == 1
        and not s.has_item("whistle"),
        runner=acquire_runner,
        knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        cost=OptionCost(frames=2500),
        opaque_effect=False,
        known_effect={"inventory_add": ["whistle"], "count": 1, "source": "1-3"},
        injected_facts=("pwing_1_3_entry_snapshot",),
        source="data/solutions/sma4/acquire_whistle_1_3.json",
        verification={
            "verified": True,
            "replay_verified": True,
            "inventory_item": 0x0C,
        },
    ))

    def use_first_runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor = _need_executor(context)
        if executor is None:
            return OptionResult(False, state)
        summary = executor.use_first_whistle()
        final = _last_sample(summary)
        next_state = _state_with(
            state,
            world=9,  # raw 8 special warp-zone map, not World 9
            node=tuple(final.get("cursor") or (64, 80)),
            inventory=("whistle",),
            flags=("warp_zone", "first_whistle_spent"),
        )
        return OptionResult(
            success=bool(summary.get("success")),
            state=next_state,
            cost=OptionCost(frames=int(summary.get("cost_frames", 0))),
            exit_snapshot=executor.snapshot(),
            info=summary,
            observed_effect=_observed_from_summary(
                summary, label="first_whistle_to_warp_zone"),
        )

    lib.add(Option(
        id="use_whistle",
        kind="UseItem",
        precondition=lambda s: s.world == 1 and s.has_item("whistle"),
        runner=use_first_runner,
        knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        opaque_effect=True,
        injected_facts=("whistle_in_inventory",),
    ))

    def use_second_runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor = _need_executor(context)
        if executor is None:
            return OptionResult(False, state)
        summary = executor.use_second_whistle()
        final = _last_sample(summary)
        next_state = _state_with(
            state,
            world=9,
            node=tuple(final.get("cursor") or (128, 144)),
            inventory=(),
            flags=("warp_zone_5_8", "second_whistle_spent"),
        )
        return OptionResult(
            success=bool(summary.get("success")),
            state=next_state,
            cost=OptionCost(frames=int(summary.get("cost_frames", 0))),
            exit_snapshot=executor.snapshot(),
            info=summary,
            observed_effect=_observed_from_summary(
                summary, label="second_whistle_to_5_8_screen"),
        )

    lib.add(Option(
        id="use_whistle_again",
        kind="UseItem",
        precondition=lambda s: s.has_item("whistle") and "warp_zone" in s.flags,
        runner=use_second_runner,
        knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        opaque_effect=True,
        injected_facts=("whistle_regranted_in_warp_zone",),
    ))

    def select_runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor = _need_executor(context)
        if executor is None:
            return OptionResult(False, state)
        summary = executor.select_world8_pipe()
        final = _last_sample(summary)
        next_state = _state_with(
            state,
            world=8,
            node=tuple(final.get("cursor") or (32, 80)),
            inventory=(),
            flags=("world8_map",),
        )
        return OptionResult(
            success=bool(summary.get("success")),
            state=next_state,
            cost=OptionCost(frames=int(summary.get("cost_frames", 0))),
            exit_snapshot=executor.snapshot(),
            info=summary,
            observed_effect=_observed_from_summary(
                summary, label="select_world8_pipe_to_world8"),
        )

    lib.add(Option(
        id="select_world8_pipe",
        kind="NavigateWarpZone",
        precondition=lambda s: "warp_zone_5_8" in s.flags,
        runner=select_runner,
        knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        opaque_effect=True,
        injected_facts=("selected_world8_pipe",),
    ))
    return lib


def run_whistle_benchmark(config: WhistleBenchmarkConfig | None = None, *,
                          max_tier: KnowledgeTier = KnowledgeTier.TIER2_BLACK_BOX_OPTION,
                          start: MetaState | None = None) -> dict:
    """Run greedy / BFS / uniform-cost planners on the whistle library.

    Returns a comparison report keyed by planner, plus a `contrast` summary
    that states whether each planner discovered the whistle skip and the cost
    gap between greedy and the cheapest searcher.
    """
    cfg = config or WhistleBenchmarkConfig()
    start = start or MetaState(world=1, node=(0, 0))
    goal = lambda s: "beat_game" in s.flags

    def _report(name, res: MetaSearchResult) -> dict:
        discovered_skip = any(d.get("option") == "use_whistle"
                              for d in res.discovered_effects)
        used_whistle = "use_whistle" in res.path
        return {
            "planner": name,
            "found": res.found,
            "plan": res.path,
            "hops": len(res.path),
            "frames": res.total_cost.frames,
            "branches": res.branches,
            "option_calls": res.option_calls,
            "injected_facts": list(res.injected_facts),
            "discovered_effects": res.discovered_effects,
            "discovered_whistle_skip": discovered_skip,
            "used_whistle": used_whistle,
        }

    reports = {}
    # Each planner gets a fresh library (options are stateless, but be safe).
    reports["greedy"] = _report(
        "greedy", greedy_plan(build_whistle_benchmark_library(cfg), start,
                              goal, max_tier=max_tier))
    reports["bfs"] = _report(
        "bfs", search_options(build_whistle_benchmark_library(cfg), start,
                              goal, max_tier=max_tier))
    reports["uniform_cost"] = _report(
        "uniform_cost",
        search_options_uniform_cost(build_whistle_benchmark_library(cfg),
                                    start, goal, max_tier=max_tier))

    greedy_frames = reports["greedy"]["frames"] if reports["greedy"]["found"] else None
    cheapest = min(
        (r["frames"] for r in reports.values()
         if r["found"] and r["used_whistle"]), default=None)
    speedup = (greedy_frames / cheapest) if (greedy_frames and cheapest) else None

    contrast = {
        "greedy_found_goal": reports["greedy"]["found"],
        "greedy_discovered_skip": reports["greedy"]["discovered_whistle_skip"],
        "search_discovered_skip": reports["uniform_cost"]["discovered_whistle_skip"],
        "greedy_frames": greedy_frames,
        "cheapest_whistle_frames": cheapest,
        "skip_cost_reduction_x": round(speedup, 2) if speedup else None,
        "thesis": (
            "myopic planning completes the game but never discovers the "
            "warp-whistle skip; resettable search discovers the opaque "
            "whistle payoff (W1->W8) and returns a much cheaper plan"),
    }
    return {
        "config": {
            "world_advance_frames": cfg.world_advance_frames,
            "bowser_frames": cfg.bowser_frames,
            "whistle_use_frames": cfg.whistle_use_frames,
            "whistle_acquire_frames": cfg.whistle_acquire_frames,
            "whistle_tier": int(cfg.whistle_tier),
            "hand_granted": cfg.hand_granted,
            "warpless_blocked": cfg.warpless_blocked,
            "max_tier": int(max_tier),
            "note": "symbolic planning-layer costs; option endpoints + whistle "
                    "spend are ROM-verified separately",
        },
        "planners": reports,
        "contrast": contrast,
    }
