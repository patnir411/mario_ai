from __future__ import annotations

from mario.adapters import SMA4Adapter
from mario.options import KnowledgeTier, MetaState, OptionContext, search_options
from mario.meta_planner import (
    SMA4WhistleROMConfig,
    WhistleBenchmarkConfig,
    build_sma4_whistle_rom_library,
    build_whistle_benchmark_library,
    greedy_plan,
    run_whistle_benchmark,
    search_options_uniform_cost,
)


def _goal(s: MetaState) -> bool:
    return "beat_game" in s.flags


def test_greedy_completes_warpless_but_misses_the_whistle_skip():
    lib = build_whistle_benchmark_library()
    start = MetaState(world=1, node=(0, 0))
    res = greedy_plan(lib, start, _goal,
                      max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION)
    # Greedy still beats the game...
    assert res.found
    # ...but via the long warpless climb, never the whistle.
    assert "use_whistle" not in res.path
    assert res.path == [
        "advance_world_1_to_2", "advance_world_2_to_3", "advance_world_3_to_4",
        "advance_world_4_to_5", "advance_world_5_to_6", "advance_world_6_to_7",
        "advance_world_7_to_8", "clear_bowser",
    ]
    # It never explored an opaque option, so it discovered nothing.
    assert res.discovered_effects == []


def test_search_discovers_opaque_whistle_skip_and_is_cheaper():
    lib = build_whistle_benchmark_library()
    start = MetaState(world=1, node=(0, 0))
    res = search_options_uniform_cost(
        lib, start, _goal, max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION)
    assert res.found
    assert res.path == ["acquire_whistle", "use_whistle", "clear_bowser"]
    # The opaque whistle effect was discovered by execution.
    assert any(d["option"] == "use_whistle" and d.get("to_world") == 8
               for d in res.discovered_effects)
    assert "whistle_in_inventory" in res.injected_facts


def test_bfs_also_finds_the_skip_by_fewest_hops():
    lib = build_whistle_benchmark_library()
    start = MetaState(world=1, node=(0, 0))
    res = search_options(lib, start, _goal,
                         max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION)
    assert res.found
    assert res.path == ["acquire_whistle", "use_whistle", "clear_bowser"]


def test_benchmark_contrast_quantifies_the_gap():
    report = run_whistle_benchmark()
    c = report["contrast"]
    assert c["greedy_found_goal"] is True
    assert c["greedy_discovered_skip"] is False
    assert c["search_discovered_skip"] is True
    # whistle route is dramatically cheaper than the warpless climb
    assert c["skip_cost_reduction_x"] is not None
    assert c["skip_cost_reduction_x"] > 5
    assert report["planners"]["greedy"]["used_whistle"] is False
    assert report["planners"]["uniform_cost"]["used_whistle"] is True


def test_hand_granted_whistle_is_tier1_injected_fact():
    report = run_whistle_benchmark()
    # The cheap planners lean on a Tier-1 hand-granted whistle; that injection
    # must be recorded so the knowledge-ladder claim stays honest.
    facts = report["planners"]["uniform_cost"]["injected_facts"]
    assert "whistle_hand_granted" in facts
    assert "whistle_in_inventory" in facts


def test_warpless_blocked_makes_greedy_fail_but_search_still_skips():
    # If the warpless route is removed, a no-exploration planner is stuck while
    # resettable search still finds the whistle skip — the sharpest contrast.
    cfg = WhistleBenchmarkConfig(warpless_blocked=True)
    lib = build_whistle_benchmark_library(cfg)
    start = MetaState(world=1, node=(0, 0))
    greedy = greedy_plan(lib, start, _goal,
                         max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION)
    search = search_options_uniform_cost(
        lib, start, _goal, max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION)
    assert not greedy.found
    assert search.found
    assert "use_whistle" in search.path


class FakeWhistleExecutor:
    def __init__(self):
        self.state = ("start",)

    def restore(self, snap):
        self.state = snap

    def snapshot(self):
        return self.state

    def grant_whistles(self, count=2):
        self.state = ("world1", "two_whistles")
        return {
            "success": True,
            "cost_frames": 0,
            "samples": [{
                "label": "hand_granted_2_whistles",
                "world_raw_0_indexed": 0,
                "world_normalized": 1,
                "cursor": [32, 64],
                "mode": "overworld",
            }],
        }

    def acquire_whistle_1_3(self):
        self.state = ("world1", "whistle_acquired_1_3")
        return {
            "success": True,
            "cost_frames": 2500,
            "samples": [{
                "label": "after_acquire_whistle_1_3",
                "world_raw_0_indexed": 0,
                "world_normalized": 1,
                "cursor": [160, 32],
                "mode": "overworld",
                "inventory_first4": [0x0C, 0, 0, 0],
            }],
            "inventory_first4": [0x0C, 0, 0, 0],
            "injected_facts": ["pwing_1_3_entry_snapshot"],
        }

    def use_first_whistle(self):
        self.state = ("warp_zone", "1_4")
        return {
            "success": True,
            "cost_frames": 100,
            "samples": [{
                "label": "after_first_whistle_warp_zone",
                "world_raw_0_indexed": 8,
                "world_normalized": 9,
                "cursor": [64, 80],
                "mode": "overworld",
            }],
        }

    def use_second_whistle(self):
        self.state = ("warp_zone", "5_8")
        return {
            "success": True,
            "cost_frames": 200,
            "samples": [{
                "label": "after_second_whistle_warp_zone_5_8",
                "world_raw_0_indexed": 8,
                "world_normalized": 9,
                "cursor": [128, 144],
                "mode": "overworld",
            }],
        }

    def select_world8_pipe(self):
        self.state = ("world8",)
        return {
            "success": True,
            "cost_frames": 300,
            "samples": [{
                "label": "world8_map",
                "world_raw_0_indexed": 7,
                "world_normalized": 8,
                "cursor": [32, 80],
                "mode": "overworld",
            }],
        }


def test_rom_whistle_library_discovers_effects_with_resettable_executor():
    executor = FakeWhistleExecutor()
    start = MetaState(world=1, node=(32, 64))
    context = OptionContext(executor=executor, snapshots={start: executor.snapshot()})
    lib = build_sma4_whistle_rom_library(SMA4WhistleROMConfig(warpless_blocked=True))

    res = search_options_uniform_cost(
        lib, start, _goal, max_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
        context=context)

    assert res.found
    assert res.path == [
        "grant_two_whistles",
        "use_whistle",
        "use_whistle_again",
        "select_world8_pipe",
        "clear_bowser",
    ]
    assert any(d["option"] == "select_world8_pipe" and d.get("raw_world") == 7
               for d in res.discovered_effects)
    assert "whistle_regranted_in_warp_zone" not in res.injected_facts
    assert "whistle_hand_granted" in res.injected_facts
    assert res.total_cost.frames == 6600


def test_rom_whistle_library_uses_acquire_whistle_when_not_hand_granted():
    """One AcquireWhistle cannot complete the two-whistle W8 skip without regrant."""
    executor = FakeWhistleExecutor()
    start = MetaState(world=1, node=(32, 64))
    context = OptionContext(executor=executor, snapshots={start: executor.snapshot()})
    lib = build_sma4_whistle_rom_library(
        SMA4WhistleROMConfig(warpless_blocked=True, hand_granted=False))

    res = search_options_uniform_cost(
        lib, start, _goal, max_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        context=context)

    # Honest: acquire_1_3 yields one whistle; after use_whistle inventory is empty,
    # so use_whistle_again is unavailable and the W8 skip cannot close.
    assert not res.found
    assert "use_whistle_again" not in res.path
    assert "pwing_1_3_entry_snapshot" in res.injected_facts or any(
        "pwing_1_3_entry_snapshot" in getattr(o, "injected_facts", ())
        for o in lib.options.values()
    )
    assert "whistle_hand_granted" not in res.injected_facts
    assert "whistle_regranted_in_warp_zone" not in res.injected_facts


def test_sma4_mode_classifier_handles_warp_zone_and_world8_maps():
    assert SMA4Adapter._classify_mode(
        0, 32, 80, world_raw=7, item_menu_open=0) == "overworld"
    assert SMA4Adapter._classify_mode(
        0, 128, 144, world_raw=8, item_menu_open=0) == "overworld"
    assert SMA4Adapter._classify_mode(
        0, 32, 80, world_raw=6, item_menu_open=0) == "menu"
    assert SMA4Adapter._classify_mode(
        0, 64, 80, world_raw=8, item_menu_open=1) == "menu"
