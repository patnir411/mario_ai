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

    assert context.restore(state)
    event = context.last_restore_event
    assert event["status"] == "restored_suffix_attested"
    assert event["raw_snapshot_match"] is False
    assert event["observable_match"] is True
    assert event["expected"]["emulator_sha256"] != event["actual"]["emulator_sha256"]

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
