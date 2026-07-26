from __future__ import annotations

from pathlib import Path

from mario.options import (
    KnowledgeTier,
    MetaState,
    Option,
    OptionContext,
    OptionCost,
    OptionLibrary,
    OptionResult,
    load_default_sma4_clear_options,
    search_options,
)


def _known(id_: str, transform, *, tier=KnowledgeTier.TIER0_GENERIC,
           pre=lambda _s: True, cost=1) -> Option:
    def runner(state: MetaState, _context: OptionContext) -> OptionResult:
        return OptionResult(True, transform(state), cost=OptionCost(frames=cost))

    return Option(
        id=id_,
        kind="Toy",
        precondition=pre,
        runner=runner,
        knowledge_tier=tier,
        cost=OptionCost(frames=cost),
        known_effect={"toy": id_},
    )


def test_meta_search_executes_opaque_effect_and_exploits_discovered_shortcut():
    lib = OptionLibrary()
    lib.add(_known(
        "acquire_whistle",
        lambda s: s.with_item("whistle"),
        tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        cost=20,
    ))

    def use_whistle(state: MetaState, _context: OptionContext) -> OptionResult:
        if not state.has_item("whistle"):
            return OptionResult(False, state)
        next_state = state.without_item("whistle").with_world_node(8, (0, 0))
        return OptionResult(
            True,
            next_state,
            cost=OptionCost(frames=5),
            observed_effect={"to_world": 8, "spent": "whistle"},
        )

    lib.add(Option(
        id="use_whistle",
        kind="UseItem",
        precondition=lambda s: s.has_item("whistle"),
        runner=use_whistle,
        knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        cost=OptionCost(frames=5),
        opaque_effect=True,
        injected_facts=("whistle_in_inventory",),
    ))
    lib.add(_known(
        "clear_bowser",
        lambda s: s.with_flag("beat_game"),
        pre=lambda s: s.world == 8,
        cost=100,
    ))
    # A longer non-skip route exists so this is not a one-branch toy.
    lib.add(_known("walk_to_world_2", lambda s: s.with_world_node(2, (0, 0)), cost=50))
    lib.add(_known(
        "walk_to_world_8",
        lambda s: s.with_world_node(8, (0, 0)),
        pre=lambda s: s.world == 2,
        cost=500,
    ))

    result = search_options(
        lib,
        MetaState(world=1, node=(0, 0)),
        lambda s: "beat_game" in s.flags,
        max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
    )

    assert result.found
    assert result.path == ["acquire_whistle", "use_whistle", "clear_bowser"]
    assert result.option_calls >= 3
    assert result.branches >= result.option_calls
    assert "whistle_in_inventory" in result.injected_facts
    assert result.discovered_effects == [{
        "option": "use_whistle",
        "to_world": 8,
        "spent": "whistle",
    }]
    assert result.total_cost.frames == 125


def test_meta_search_tier_filter_blocks_injected_option_until_allowed():
    lib = OptionLibrary()
    lib.add(_known(
        "route_data_shortcut",
        lambda s: s.with_world_node(8, (0, 0)).with_flag("beat_game"),
        tier=KnowledgeTier.TIER4_LLM_OR_DISASM,
        cost=1,
    ))
    start = MetaState(world=1, node=(0, 0))

    blocked = search_options(
        lib, start, lambda s: "beat_game" in s.flags,
        max_tier=KnowledgeTier.TIER3_SUBGOAL_HINT,
    )
    allowed = search_options(
        lib, start, lambda s: "beat_game" in s.flags,
        max_tier=KnowledgeTier.TIER4_LLM_OR_DISASM,
    )

    assert not blocked.found
    assert blocked.option_calls == 0
    assert allowed.found
    assert allowed.path == ["route_data_shortcut"]


def test_sma4_solution_files_wrap_as_verified_clear_options():
    lib = load_default_sma4_clear_options()
    assert sorted(lib.options) == ["clear_sma4_1-1", "clear_sma4_1-2"]
    opt = lib.options["clear_sma4_1-1"]
    assert opt.verify()
    assert opt.knowledge_tier == KnowledgeTier.TIER0_GENERIC
    assert opt.entry_snapshot == "runs/sma4_cache/1-1_entry.pkl"
    assert Path(opt.source).exists()
    result = opt.execute(MetaState(world=1, node=(64, 32)))
    assert result.success
    assert result.info["execution_mode"] == "symbolic_manifest_only"
    assert result.info["physical_boundary_evidence"] is False
    assert "clear:1-1" in result.state.flags
    assert result.state.cleared & 1
    assert opt.cost.frames >= 1130
    assert opt.cost.nodes > 0


def test_sma4_symbolic_manifest_mode_is_visible_in_search_result():
    lib = load_default_sma4_clear_options()
    start = MetaState(world=1, node=(64, 32))

    result = search_options(
        lib,
        start,
        lambda state: "clear:1-1" in state.flags,
        max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
    )

    assert result.found
    assert result.path_evidence == [{
        "option": "clear_sma4_1-1",
        "execution_mode": "symbolic_manifest_only",
        "physical_boundary_evidence": False,
        "boundary_status": None,
    }]


class _UnsettledClearExecutor:
    def __init__(self):
        self.current = (b"root", {"mode": "level"})
        self.settle = True

    def restore(self, snap):
        self.current = snap

    def snapshot(self):
        return self.current

    def execute_clear_solution(self, _path, _snapshot):
        self.current = (b"goal-but-not-map", {"mode": "level"})
        return ({
            "solved": True,
            "settled_to_map": False,
            "settle_required": True,
            "final_info": {"mode": "level", "world": 1},
        }, self.current)


def test_physical_clear_fails_when_replay_does_not_settle_to_map():
    option = load_default_sma4_clear_options().options["clear_sma4_1-1"]
    start = MetaState(world=1, node=(64, 32))
    executor = _UnsettledClearExecutor()
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
    )

    result = option.execute(start, context)

    assert not result.success
    assert result.boundary["status"] == "failed"
    assert result.info["executor_summary"]["settled_to_map"] is False
