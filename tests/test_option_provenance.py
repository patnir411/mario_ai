from __future__ import annotations

import pytest

from mario.options import (
    KnowledgeTier,
    MetaState,
    Option,
    OptionContext,
    OptionCost,
    OptionLibrary,
    OptionResult,
    SMA4WhistleExecutor,
    search_options,
)
from mario.meta_planner import search_options_uniform_cost
from mario.physical_planner import (
    refine_option_partitions,
    search_physical_options,
    search_physical_options_uniform_cost,
)
from mario.provenance import StateAliasError, snapshot_digest


class ByteExecutor:
    """Tiny complete byte-backed state machine for boundary contract tests."""

    def __init__(self, state: bytes = b"root"):
        self.state = bytes(state)
        self.info = {"held": [], "terminal": False}
        self.restore_calls = 0

    def snapshot(self):
        return self.state, dict(self.info)

    def restore(self, snap):
        self.restore_calls += 1
        state, info = snap
        self.state = bytes(state)
        self.info = dict(info)

    def snapshot_backend(self):
        return "byte-test"

    def rom_sha1(self):
        return "0" * 40


class PartialObservableByteExecutor(ByteExecutor):
    """Exposes a deliberately incomplete physical-state observation."""

    def snapshot_observable(self):
        return {"mode": "overworld"}


class NoncanonicalByteExecutor(ByteExecutor):
    """Savestate bytes alternate while exact emulated RAM stays unchanged."""

    def __init__(self):
        super().__init__(b"semantic-state")
        self._encoding = 0

    def snapshot(self):
        self._encoding ^= 1
        return (
            b"encoding-a" if self._encoding else b"encoding-b",
            dict(self.info),
        )

    def restore(self, snap):
        self.restore_calls += 1
        self.info = dict(snap[1])

    def snapshot_observable(self):
        return {
            "ram_blocks": {
                "0x00000000": {"bytes": 1, "sha256": "semantic-ram"}
            },
            "screen_sha256": (
                "host-display-a" if self._encoding else "host-display-b"
            ),
        }

    def snapshot_equivalence_signature(self, _snapshot):
        return {
            "kind": "fixed_noop_suffix",
            "frames": 4,
            "sha256": "same-future",
        }


class _FakeMemory:
    def __init__(self, values):
        self.values = values

    def assign(self, address, _dtype, value):
        self.values[address] = int(value)


class _FakeCore:
    def __init__(self):
        self.values = {0x03000010: 1}
        memory = _FakeMemory(self.values)
        self.env = type("Env", (), {
            "data": type("Data", (), {"memory": memory})()
        })()

    def _read_u8(self, address):
        return self.values.get(address, 0)


class _FailingProbeCore:
    def __init__(self):
        self.state = b"live"
        memory = type("Memory", (), {"blocks": {}})()
        self.env = type("Env", (), {
            "data": type("Data", (), {"memory": memory})()
        })()
        self.last_obs = b"display"
        self.last_info = {"mode": "overworld"}

    def snapshot(self):
        return self.state, dict(self.last_info)

    def restore(self, snap):
        self.state, info = snap
        self.last_info = dict(info)

    def _step_buttons(self, _buttons):
        self.state = b"speculative"
        raise RuntimeError("probe failure")


class _MapProbeCore:
    WORLD = 0x01
    MAP_CURSOR_X = 0x02
    MAP_CURSOR_Y = 0x03

    def __init__(self, *, responsive, autonomous=False):
        self.responsive = bool(responsive)
        self.autonomous = bool(autonomous)
        self.values = {
            self.WORLD: SMA4WhistleExecutor.WORLD_8_RAW,
            self.MAP_CURSOR_X: 32,
            self.MAP_CURSOR_Y: 80,
            SMA4WhistleExecutor.ITEM_MENU_OPEN: 0,
            SMA4WhistleExecutor.MAP_EVENT: 17,
            SMA4WhistleExecutor.MAP_DEST_OR_REGION: 0,
        }
        memory = type("Memory", (), {"blocks": {}})()
        self.env = type("Env", (), {
            "data": type("Data", (), {"memory": memory})()
        })()
        self.last_obs = b"display"
        self.last_info = {
            "world": 8,
            "is_warp_zone": False,
            "cursor": (32, 80),
            "mode": "overworld",
            "cleared": 0,
            "time": 0,
        }
        self._last_info = dict(self.last_info)
        self.action_names = []

    def _read_u8(self, address):
        return self.values.get(address, 0)

    def _step_buttons(self, buttons):
        if self.autonomous:
            self.values[self.MAP_CURSOR_X] += 1
        elif self.responsive:
            if "LEFT" in buttons:
                self.values[self.MAP_CURSOR_X] -= 1
            elif "RIGHT" in buttons:
                self.values[self.MAP_CURSOR_X] += 1
            elif "UP" in buttons:
                self.values[self.MAP_CURSOR_Y] -= 1
            elif "DOWN" in buttons:
                self.values[self.MAP_CURSOR_Y] += 1
        self.last_info = {
            **self.last_info,
            "cursor": (
                self.values[self.MAP_CURSOR_X],
                self.values[self.MAP_CURSOR_Y],
            ),
        }
        return self.last_obs, self.last_info, False

    def snapshot(self):
        payload = bytes([
            self.values[self.WORLD],
            self.values[self.MAP_CURSOR_X] & 0xFF,
            self.values[self.MAP_CURSOR_Y] & 0xFF,
        ])
        return payload, {
            "values": dict(self.values),
            "last_info": dict(self.last_info),
        }

    def restore(self, snapshot):
        _payload, metadata = snapshot
        self.values = dict(metadata["values"])
        self.last_info = dict(metadata["last_info"])
        self._last_info = dict(self.last_info)


def _physical_option(option_id, transform, runner):
    return Option(
        id=option_id,
        kind="PhysicalTest",
        precondition=lambda _state: True,
        runner=runner,
        knowledge_tier=KnowledgeTier.TIER0_GENERIC,
        requires_snapshot=True,
        known_effect={"test": option_id},
    )


def test_snapshot_digest_is_exact_and_mapping_order_independent():
    left = (b"\x00\x01\xff", {"b": [2, 3], "a": 1})
    right = (b"\x00\x01\xff", {"a": 1, "b": [2, 3]})

    a = snapshot_digest(
        left, adapter_context={"level": "1-2"}, backend="gba", rom_sha1="a" * 40)
    b = snapshot_digest(
        right, adapter_context={"level": "1-2"}, backend="gba", rom_sha1="a" * 40)
    other_context = snapshot_digest(
        right, adapter_context={"level": "1-F"}, backend="gba", rom_sha1="a" * 40)

    assert a.exact
    assert a.emulator_bytes == 3
    assert a.emulator_sha256 == b.emulator_sha256
    assert a.full_sha256 == b.full_sha256
    assert a.full_sha256 != other_context.full_sha256


def test_meta_state_inventory_preserves_item_multiplicity():
    one = MetaState(1, inventory=("whistle",))
    two = MetaState(1, inventory=("whistle", "whistle"))

    assert one != two
    assert two.without_item("whistle") == one


def test_required_physical_option_fails_closed_without_parent_snapshot():
    executor = ByteExecutor(b"stale-branch")
    context = OptionContext(executor=executor)
    called = False

    def runner(state, _context):
        nonlocal called
        called = True
        return OptionResult(True, state, exit_snapshot=executor.snapshot())

    option = _physical_option("requires_root", lambda state: state, runner)
    state = MetaState(1)
    result = option.execute(state, context)

    assert not result.success
    assert result.info["reason"] == "missing_parent_snapshot"
    assert not called
    assert executor.restore_calls == 0


def test_physical_option_restores_parent_exactly_once():
    executor = ByteExecutor()
    state = MetaState(1)
    context = OptionContext(
        executor=executor,
        snapshots={state: executor.snapshot()},
    )

    def runner(current, _context):
        executor.state += b"-advanced"
        return OptionResult(True, current.with_flag("done"),
                            exit_snapshot=executor.snapshot())

    result = _physical_option("advance", lambda s: s, runner).execute(
        state, context)

    assert result.success
    assert executor.restore_calls == 1
    assert result.entry_digest == context.snapshot_records[state].digest.to_json()


def test_failed_trial_never_poison_snapshot_context():
    executor = ByteExecutor()
    state = MetaState(1)
    predicted = state.with_flag("predicted")
    context = OptionContext(
        executor=executor,
        snapshots={state: executor.snapshot()},
    )

    def runner(_state, _context):
        executor.state = b"failed-physical-exit"
        return OptionResult(False, predicted, exit_snapshot=executor.snapshot())

    result = _physical_option("fail", lambda s: s, runner).execute(state, context)

    assert not result.success
    assert predicted not in context.snapshots
    assert predicted not in context.snapshot_records


def test_missing_runner_exit_snapshot_is_captured_from_live_executor():
    executor = ByteExecutor()
    state = MetaState(1)
    context = OptionContext(
        executor=executor,
        snapshots={state: executor.snapshot()},
    )

    def runner(current, _context):
        executor.state = b"live-exit"
        return OptionResult(True, current.with_flag("done"))

    result = _physical_option("implicit_exit", lambda s: s, runner).execute(
        state, context)

    assert result.exit_snapshot[0] == b"live-exit"
    assert result.exit_digest is not None
    context.remember(
        result.state,
        result.exit_snapshot,
        producer="implicit_exit",
        observable=result.exit_observable,
        adapter_context=result.exit_adapter_context,
    )
    executor.state = b"corrupted"
    assert context.restore(result.state)
    assert executor.state == b"live-exit"


def test_symbolic_alias_collision_is_detected_and_fails_closed():
    executor = ByteExecutor(b"representative-a")
    state = MetaState(1)
    context = OptionContext(
        executor=executor,
        snapshots={state: executor.snapshot()},
    )
    executor.state = b"representative-b"

    with pytest.raises(StateAliasError) as exc:
        context.remember(
            state,
            executor.snapshot(),
            producer="second_route",
        )

    collision = exc.value.collision
    assert collision == context.alias_collisions[-1]
    assert (collision["retained"]["digest"]["full_sha256"]
            != collision["rejected"]["digest"]["full_sha256"])
    assert context.snapshots[state][0] == b"representative-a"


def test_search_alias_records_distinct_parent_provenance():
    executor = ByteExecutor()
    start = MetaState(1)
    left = start.with_flag("left")
    right = start.with_flag("right")
    shared = start.with_flag("shared")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
    )

    def transition(option_id, source, target, physical):
        def runner(_state, _context):
            executor.state = physical
            return OptionResult(True, target, exit_snapshot=executor.snapshot())

        return Option(
            id=option_id,
            kind="PhysicalTest",
            precondition=lambda state: state == source,
            runner=runner,
            requires_snapshot=True,
        )

    def merge_runner(state, _context):
        executor.state = (
            b"shared-from-left" if state == left else b"shared-from-right"
        )
        return OptionResult(True, shared, exit_snapshot=executor.snapshot())

    library = OptionLibrary()
    library.extend([
        transition("reach_left", start, left, b"left"),
        transition("reach_right", start, right, b"right"),
        Option(
            id="merge",
            kind="PhysicalTest",
            precondition=lambda state: state in (left, right),
            runner=merge_runner,
            requires_snapshot=True,
        ),
    ])

    with pytest.raises(StateAliasError) as exc:
        search_options(
            library,
            start,
            lambda _state: False,
            context=context,
        )

    collision = exc.value.collision
    parents = {
        tuple(row["source"]["parent_state"]["flags"])
        for row in (collision["retained"], collision["rejected"])
    }
    assert parents == {("left",), ("right",)}


def test_noncanonical_restore_is_attested_but_cross_route_alias_still_fails():
    executor = NoncanonicalByteExecutor()
    state = MetaState(1)
    original = executor.snapshot()
    context = OptionContext(executor=executor, snapshots={state: original})
    root_id = context.record_ids_for(state)[0]

    assert context.restore(state)
    event = context.last_restore_event
    assert event["status"] == "restored_suffix_attested"
    assert event["raw_snapshot_match"] is False
    assert event["observable_match"] is True
    assert event["expected"]["emulator_sha256"] != event["actual"]["emulator_sha256"]
    assert context.record_ids_for(state) == (root_id,)
    assert len(context.physical_records) == 1

    executor.snapshot()  # advance past the retained encoding
    variant = executor.snapshot()
    with pytest.raises(StateAliasError):
        context.remember(
            state,
            variant,
            producer="independent_route_same_short_suffix",
            observable=executor.snapshot_observable(),
        )
    assert len(context.alias_collisions) == 1


def test_sma4_ram_write_ledger_records_address_values_reason_and_tier():
    executor = SMA4WhistleExecutor(_FakeCore())
    executor.begin_option_trace("test_write", None)
    executor._write_u8(
        0x03000010,
        12,
        reason="test_inventory_intervention",
        knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        frame=17,
    )
    trace = executor.end_option_trace()

    assert trace["ram_writes"] == [{
        "kind": "ram_write",
        "address": "0x03000010",
        "before": 1,
        "requested": 12,
        "after": 12,
        "changed": True,
        "reason": "test_inventory_intervention",
        "knowledge_tier": 2,
        "frame": 17,
    }]


def test_fixed_suffix_probe_rolls_back_even_when_probe_raises():
    core = _FailingProbeCore()
    executor = SMA4WhistleExecutor(core)

    with pytest.raises(RuntimeError, match="probe failure"):
        executor.snapshot_equivalence_signature(
            (b"candidate", {"mode": "overworld"}))

    assert core.state == b"live"
    assert core.last_info == {"mode": "overworld"}


@pytest.mark.parametrize(
    ("responsive", "accepted"), [(False, False), (True, True)])
def test_world8_acceptance_requires_a_responsive_map(responsive, accepted):
    core = _MapProbeCore(responsive=responsive)
    executor = SMA4WhistleExecutor(core)

    result = executor.select_world8_pipe()

    assert result["success"] is accepted
    assert result["world8_acceptance"] is accepted
    assert result["terminal_invariants"]["raw_world_is_world8"] is True
    assert result["terminal_invariants"]["cursor_responsive"] is responsive
    assert result["evaluation_frames"] > 0
    # The accepted-lineage state is exactly the state before the speculative
    # directional probe, irrespective of the probe outcome.
    assert list(core.last_info["cursor"]) == result["samples"][-1]["cursor"]
    assert executor._trace_events[-1] == {
        "kind": "behavioral_probe",
        "probe": "directional_map_cursor_responsiveness",
        "knowledge_tier": int(KnowledgeTier.TIER0_GENERIC),
        "result": responsive,
        "evaluation_frames": result["evaluation_frames"],
        "attempts": result["cursor_probe_attempts"],
        "rolled_back": True,
    }


def test_world8_acceptance_rejects_autonomous_cursor_motion():
    core = _MapProbeCore(responsive=False, autonomous=True)
    executor = SMA4WhistleExecutor(core)

    result = executor.select_world8_pipe()

    assert result["terminal_invariants"]["raw_world_is_world8"] is True
    assert result["cursor_responsive"] is False
    assert result["success"] is False
    assert all(
        not attempt["diverged_from_noop"]
        for attempt in result["cursor_probe_attempts"]
    )


def test_two_option_boundaries_chain_after_executor_corruption():
    executor = ByteExecutor()
    start = MetaState(1)
    middle = start.with_flag("middle")
    finish = middle.with_flag("finish")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
    )

    def first_runner(_state, _context):
        executor.state = b"first-exit"
        return OptionResult(True, middle, exit_snapshot=executor.snapshot())

    first = _physical_option("first", lambda s: middle, first_runner)
    first_result = first.execute(start, context)
    context.remember(middle, first_result.exit_snapshot, producer=first.id)

    executor.state = b"corrupted-between-options"
    executor.info["terminal"] = True

    def second_runner(_state, _context):
        assert executor.state == b"first-exit"
        assert executor.info["terminal"] is False
        executor.state = b"second-exit"
        return OptionResult(True, finish, exit_snapshot=executor.snapshot())

    second = _physical_option("second", lambda s: finish, second_runner)
    second_result = second.execute(middle, context)

    assert first_result.exit_digest == second_result.entry_digest
    assert second_result.boundary["restore"]["status"] == "restored_exact_bytes"
    assert second_result.boundary["raw_hash_continuous"] is True
    assert second_result.boundary["composition_attested"] is True


def test_uniform_cost_replaces_equal_snapshot_producer_on_cheaper_path():
    executor = ByteExecutor()
    start = MetaState(1)
    detour = start.with_flag("detour")
    middle = start.with_flag("middle")
    goal = middle.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
    )

    def transition(option_id, source, target, physical, cost):
        def runner(_state, _context):
            executor.state = physical
            return OptionResult(
                True,
                target,
                cost=OptionCost(frames=cost),
                exit_snapshot=executor.snapshot(),
            )

        return Option(
            id=option_id,
            kind="PhysicalTest",
            precondition=lambda state: state == source,
            runner=runner,
            cost=OptionCost(frames=cost),
            requires_snapshot=True,
        )

    options = [
        transition("direct_expensive", start, middle, b"middle", 10),
        transition("detour", start, detour, b"detour", 1),
        transition("cheap_to_middle", detour, middle, b"middle", 1),
        transition("finish", middle, goal, b"goal", 1),
    ]
    from mario.options import OptionLibrary
    library = OptionLibrary({option.id: option for option in options})

    result = search_options_uniform_cost(
        library,
        start,
        lambda state: state == goal,
        context=context,
    )

    assert result.path == ["detour", "cheap_to_middle", "finish"]
    assert result.total_cost.frames == 3
    assert context.snapshot_records[middle].producer == "cheap_to_middle"


def _physical_transition(executor, option_id, source, target, physical, cost=1):
    def runner(_state, _context):
        executor.state = physical
        return OptionResult(
            True,
            target,
            cost=OptionCost(frames=cost),
            exit_snapshot=executor.snapshot(),
        )

    return Option(
        id=option_id,
        kind="PhysicalTest",
        precondition=lambda state: state == source,
        runner=runner,
        cost=OptionCost(frames=cost),
        requires_snapshot=True,
    )


@pytest.mark.parametrize("reverse_insertion", [False, True])
def test_multi_search_preserves_representatives_regardless_of_option_order(
        reverse_insertion):
    executor = ByteExecutor()
    start = MetaState(1)
    middle = start.with_flag("middle")
    goal = middle.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )

    bad = _physical_transition(
        executor, "a_reach_bad", start, middle, b"bad", cost=1)
    good = _physical_transition(
        executor, "z_reach_good", start, middle, b"good", cost=1)

    def finish_runner(_state, _context):
        if executor.state != b"good":
            return OptionResult(False, middle, info={"reason": "locked"})
        executor.state = b"goal"
        return OptionResult(
            True,
            goal,
            cost=OptionCost(frames=1),
            exit_snapshot=executor.snapshot(),
        )

    finish = Option(
        id="finish",
        kind="PhysicalTest",
        precondition=lambda state: state == middle,
        runner=finish_runner,
        cost=OptionCost(frames=1),
        requires_snapshot=True,
    )
    options = [bad, good, finish]
    if reverse_insertion:
        options.reverse()
    library = OptionLibrary()
    library.extend(options)

    result = search_physical_options(
        library,
        start,
        lambda state: state == goal,
        context=context,
        determinism_repeats=2,
    )

    assert result.found
    assert result.path == ["z_reach_good", "finish"]
    middle_ids = context.record_ids_for(middle)
    assert len(middle_ids) == 2
    assert {
        context.physical_records[record_id].snapshot[0]
        for record_id in middle_ids
    } == {b"bad", b"good"}
    for record_id in middle_ids:
        assert context.restore_record(record_id, expected_state=middle)
        assert executor.state == context.physical_records[record_id].snapshot[0]
    assert any(
        row.get("resolution") == "retained_both"
        for row in result.alias_collisions
    )
    middle_blocks = {
        result.partition_refinement["record_to_block"][str(record_id)]
        for record_id in middle_ids
    }
    assert len(middle_blocks) == 2


def test_multi_uniform_cost_dominates_per_representative():
    executor = ByteExecutor()
    start = MetaState(1)
    middle = start.with_flag("middle")
    goal = middle.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )

    def finish_runner(_state, _context):
        if executor.state != b"good":
            return OptionResult(False, middle, info={"reason": "locked"})
        executor.state = b"goal"
        return OptionResult(
            True, goal, cost=OptionCost(frames=1),
            exit_snapshot=executor.snapshot())

    library = OptionLibrary()
    library.extend([
        _physical_transition(
            executor, "cheap_bad", start, middle, b"bad", cost=1),
        _physical_transition(
            executor, "expensive_good", start, middle, b"good", cost=10),
        Option(
            id="finish",
            kind="PhysicalTest",
            precondition=lambda state: state == middle,
            runner=finish_runner,
            cost=OptionCost(frames=1),
            requires_snapshot=True,
        ),
    ])

    result = search_physical_options_uniform_cost(
        library, start, lambda state: state == goal, context=context)

    assert result.found
    assert result.path == ["expensive_good", "finish"]
    assert result.total_cost.frames == 11
    assert len(context.record_ids_for(middle)) == 2


def test_multi_uniform_cost_exact_duplicate_uses_cheapest_arrival():
    executor = ByteExecutor()
    start = MetaState(1)
    middle = start.with_flag("middle")
    goal = middle.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )
    library = OptionLibrary()
    library.extend([
        _physical_transition(
            executor, "a_expensive", start, middle, b"same", cost=10),
        _physical_transition(
            executor, "z_cheap", start, middle, b"same", cost=1),
        _physical_transition(
            executor, "finish", middle, goal, b"goal", cost=1),
    ])

    result = search_physical_options_uniform_cost(
        library, start, lambda state: state == goal, context=context)

    assert result.path == ["z_cheap", "finish"]
    assert result.total_cost.frames == 2
    middle_ids = context.record_ids_for(middle)
    assert len(middle_ids) == 1
    assert len(context.record_arrivals[middle_ids[0]]) == 2
    assert not context.alias_collisions


def test_symbolic_successor_cannot_borrow_unrelated_physical_record():
    executor = ByteExecutor()
    start = MetaState(1)
    middle = start.with_flag("middle")
    goal = middle.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )
    context.add_root(
        middle,
        (b"unrelated-middle", {"held": [], "terminal": False}),
        source={"kind": "unrelated_declared_root"},
    )
    physical_runner_called = False

    def symbolic_runner(_state, _context):
        return OptionResult(True, middle, cost=OptionCost(frames=1))

    def physical_runner(_state, _context):
        nonlocal physical_runner_called
        physical_runner_called = True
        return OptionResult(True, goal, exit_snapshot=executor.snapshot())

    library = OptionLibrary()
    library.extend([
        Option(
            id="symbolic_to_middle",
            kind="SymbolicTest",
            precondition=lambda state: state == start,
            runner=symbolic_runner,
        ),
        Option(
            id="physical_finish",
            kind="PhysicalTest",
            precondition=lambda state: state == middle,
            runner=physical_runner,
            requires_snapshot=True,
        ),
    ])

    result = search_physical_options(
        library, start, lambda state: state == goal, context=context)

    assert not result.found
    assert not physical_runner_called
    missing = [
        row for row in result.log
        if row["option"] == "physical_finish"
    ]
    assert missing[0]["from_record_id"] is None
    assert missing[0]["reason"] == "missing_parent_snapshot"


def test_partition_refinement_propagates_delayed_split_and_is_order_invariant():
    parent = MetaState(1, flags=("parent",))
    child = MetaState(1, flags=("child",))
    merge = MetaState(1, flags=("merge",))
    records = {
        1: parent,
        2: parent,
        3: child,
        4: child,
        5: merge,
        6: merge,
    }

    def disabled(record_id, state, option):
        return {
            "record_id": record_id,
            "state": state.to_json(),
            "option": option,
            "physical_option": True,
            "applicability": "not_enabled",
            "repeat_conformant": None,
            "outcome_signature": None,
            "successor_state": None,
            "successor_record_id": None,
        }

    common_advance = {
        "accepted": True,
        "runner_success": True,
        "termination_reason": None,
        "successor_state": child.to_json(),
        "successor_goal": False,
        "reported_cost": {"frames": 1, "nodes": 0},
        "retained_frames": 1,
        "effective_knowledge_tier": 0,
        "terminal_invariants": {},
        "interventions": {},
    }
    probe_success = {
        **common_advance,
        "successor_state": MetaState(1, flags=("done",)).to_json(),
        "successor_goal": True,
    }
    probe_failure = {
        **common_advance,
        "accepted": False,
        "runner_success": False,
        "termination_reason": "locked",
        "successor_state": None,
        "successor_goal": False,
    }
    observations = [
        {
            "record_id": 1, "state": parent.to_json(), "option": "advance",
            "physical_option": True, "applicability": "enabled",
            "repeat_conformant": True, "outcome_signature": common_advance,
            "successor_state": child.to_json(), "successor_record_id": 3,
        },
        {
            "record_id": 2, "state": parent.to_json(), "option": "advance",
            "physical_option": True, "applicability": "enabled",
            "repeat_conformant": True, "outcome_signature": common_advance,
            "successor_state": child.to_json(), "successor_record_id": 4,
        },
        disabled(1, parent, "probe"),
        disabled(2, parent, "probe"),
        disabled(3, child, "advance"),
        disabled(4, child, "advance"),
        {
            "record_id": 3, "state": child.to_json(), "option": "probe",
            "physical_option": False, "applicability": "enabled",
            "repeat_conformant": True, "outcome_signature": probe_success,
            "successor_state": probe_success["successor_state"],
            "successor_record_id": None,
        },
        {
            "record_id": 4, "state": child.to_json(), "option": "probe",
            "physical_option": False, "applicability": "enabled",
            "repeat_conformant": True, "outcome_signature": probe_failure,
            "successor_state": None, "successor_record_id": None,
        },
        disabled(5, merge, "advance"),
        disabled(6, merge, "advance"),
        disabled(5, merge, "probe"),
        disabled(6, merge, "probe"),
    ]

    forward = refine_option_partitions(
        records, observations, option_ids=["advance", "probe"])
    reverse = refine_option_partitions(
        records, reversed(observations), option_ids=["probe", "advance"])

    def partition_sets(report):
        return {
            frozenset(row["member_record_ids"])
            for row in report["classes"]
        }

    assert partition_sets(forward) == partition_sets(reverse)
    mapping = forward["record_to_block"]
    assert mapping["3"] != mapping["4"]
    assert mapping["1"] != mapping["2"]
    assert mapping["5"] == mapping["6"]
    parent_witness = next(
        row for row in forward["distinguishing_option_suffixes"]
        if {row["left_record_id"], row["right_record_id"]} == {1, 2}
    )
    assert parent_witness["option_suffix"] == ["advance", "probe"]


def test_signature_repeats_fail_closed_on_nondeterminism():
    executor = ByteExecutor()
    start = MetaState(1)
    goal = start.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )
    calls = 0

    def toggling_runner(_state, _context):
        nonlocal calls
        calls += 1
        executor.state = b"goal"
        return OptionResult(
            calls % 2 == 1,
            goal,
            cost=OptionCost(frames=1),
            exit_snapshot=executor.snapshot(),
            info={} if calls % 2 == 1 else {"reason": "toggle"},
        )

    library = OptionLibrary()
    library.add(Option(
        id="toggle",
        kind="PhysicalTest",
        precondition=lambda state: state == start,
        runner=toggling_runner,
        requires_snapshot=True,
    ))

    result = search_physical_options(
        library,
        start,
        lambda state: state == goal,
        context=context,
        determinism_repeats=2,
    )

    assert not result.found
    assert calls == 2
    assert result.repeat_conformance_failures[0]["option"] == "toggle"
    assert len(context.physical_records) == 1
    assert not result.partition_refinement[
        "empirical_conformance_gate"
    ]


@pytest.mark.parametrize(
    "executor_type", [ByteExecutor, PartialObservableByteExecutor]
)
def test_signature_repeats_attest_the_physical_successor(executor_type):
    executor = executor_type()
    start = MetaState(1)
    goal = start.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )
    calls = 0

    def alternating_exit(_state, _context):
        nonlocal calls
        calls += 1
        executor.state = b"physical-a" if calls % 2 else b"physical-b"
        return OptionResult(
            True,
            goal,
            cost=OptionCost(frames=1),
            exit_snapshot=executor.snapshot(),
        )

    library = OptionLibrary()
    library.add(Option(
        id="alternating_physical_exit",
        kind="PhysicalTest",
        precondition=lambda state: state == start,
        runner=alternating_exit,
        requires_snapshot=True,
    ))

    result = search_physical_options(
        library,
        start,
        lambda state: state == goal,
        context=context,
        determinism_repeats=2,
    )

    assert not result.found
    assert calls == 2
    failure = result.repeat_conformance_failures[0]
    assert failure["repeat_signatures"][0] == failure["repeat_signatures"][1]
    assert (
        failure["repeat_physical_exit_attestations"][0]
        != failure["repeat_physical_exit_attestations"][1]
    )
    assert len(context.physical_records) == 1


def test_bounded_uniform_cost_keeps_nondominated_cost_depth_labels():
    executor = ByteExecutor()
    start = MetaState(1)
    long_1 = start.with_flag("long-1")
    long_2 = start.with_flag("long-2")
    middle = start.with_flag("middle")
    goal = middle.with_flag("goal")
    context = OptionContext(
        executor=executor,
        snapshots={start: executor.snapshot()},
        alias_policy="multi",
    )
    library = OptionLibrary()
    library.extend([
        _physical_transition(
            executor, "a_long_start", start, long_1, b"long-1", cost=0),
        _physical_transition(
            executor, "b_long_continue", long_1, long_2, b"long-2", cost=0),
        _physical_transition(
            executor, "c_long_merge", long_2, middle, b"middle", cost=0),
        _physical_transition(
            executor, "z_short_merge", start, middle, b"middle", cost=10),
        _physical_transition(
            executor, "finish", middle, goal, b"goal", cost=1),
    ])

    result = search_physical_options_uniform_cost(
        library,
        start,
        lambda state: state == goal,
        context=context,
        max_depth=3,
    )

    assert result.found
    assert result.path == ["z_short_merge", "finish"]
    assert result.total_cost.frames == 11


def test_partition_refinement_rejects_conflicting_duplicate_rows_order_independently():
    state = MetaState(1)
    records = {1: state, 2: state}
    success = {
        "accepted": True,
        "runner_success": True,
        "termination_reason": None,
        "successor_state": MetaState(1, flags=("goal",)).to_json(),
        "successor_goal": True,
        "reported_cost": {"frames": 1, "nodes": 0},
        "retained_frames": 1,
        "effective_knowledge_tier": 0,
        "terminal_invariants": {},
        "interventions": {},
    }
    failure = {
        **success,
        "accepted": False,
        "runner_success": False,
        "termination_reason": "locked",
        "successor_state": None,
        "successor_goal": False,
    }

    def row(record_id, outcome):
        return {
            "record_id": record_id,
            "state": state.to_json(),
            "option": "probe",
            "physical_option": False,
            "applicability": "enabled",
            "repeat_count": 2,
            "repeat_conformant": True,
            "outcome_signature": outcome,
            "successor_state": outcome["successor_state"],
            "successor_record_id": None,
        }

    observations = [row(1, success), row(1, failure), row(2, success)]
    forward = refine_option_partitions(
        records, observations, option_ids=["probe"])
    reverse = refine_option_partitions(
        records, reversed(observations), option_ids=["probe"])

    assert forward["record_to_block"] == reverse["record_to_block"]
    assert forward["conflicting_observations"] == reverse[
        "conflicting_observations"
    ]
    assert len(forward["conflicting_observations"]) == 1
    assert forward["record_to_block"]["1"] != forward[
        "record_to_block"]["2"]
    assert not forward["empirical_conformance_gate"]
