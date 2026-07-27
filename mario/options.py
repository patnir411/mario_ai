"""Option-level meta-MDP scaffolding for hierarchical Mario planning.

The low-level solver produces verified *options* (clear a level, acquire an
item, use an item).  This module searches over those options in a small symbolic
state while still allowing opaque-effect options to be executed against a
resettable emulator and then classified from the observed post-state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

from mario.provenance import (
    SnapshotDigest,
    StateAliasError,
    file_sha256,
    snapshot_digest,
    stable_encode,
)


class KnowledgeTier(IntEnum):
    """How much route/domain knowledge an option injects into the benchmark."""

    TIER0_GENERIC = 0
    TIER1_ITEM_GIVEN = 1
    TIER2_BLACK_BOX_OPTION = 2
    TIER3_SUBGOAL_HINT = 3
    TIER4_LLM_OR_DISASM = 4
    TIER5_FULL_SCRIPT = 5


@dataclass(frozen=True)
class MetaState:
    """Symbolic state for the SMB3 overworld/item planning layer."""

    world: int
    node: tuple[int, int] = (0, 0)
    cleared: int = 0
    inventory: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "world", int(self.world))
        object.__setattr__(self, "node", (int(self.node[0]), int(self.node[1])))
        object.__setattr__(self, "cleared", int(self.cleared))
        # Inventory is a multiset.  Collapsing duplicates aliases one whistle
        # with two, which changes reachability of the two-whistle warp route.
        object.__setattr__(self, "inventory", tuple(sorted(self.inventory)))
        object.__setattr__(self, "flags", tuple(sorted(set(self.flags))))

    def has_item(self, item: str) -> bool:
        return item in self.inventory

    def with_item(self, item: str) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared,
                         self.inventory + (item,), self.flags)

    def without_item(self, item: str) -> "MetaState":
        inventory = list(self.inventory)
        if item in inventory:
            inventory.remove(item)
        return MetaState(self.world, self.node, self.cleared,
                         tuple(inventory), self.flags)

    def with_world_node(self, world: int, node: tuple[int, int]) -> "MetaState":
        return MetaState(world, node, self.cleared, self.inventory, self.flags)

    def with_cleared(self, mask: int) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared | int(mask),
                         self.inventory, self.flags)

    def with_flag(self, flag: str) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared,
                         self.inventory, self.flags + (flag,))

    def to_json(self) -> dict:
        return {
            "world": self.world,
            "node": list(self.node),
            "cleared": self.cleared,
            "inventory": list(self.inventory),
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class OptionCost:
    frames: int = 0
    nodes: int = 0
    wall_clock_s: float = 0.0

    def __add__(self, other: "OptionCost") -> "OptionCost":
        return OptionCost(
            frames=self.frames + other.frames,
            nodes=self.nodes + other.nodes,
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
        )

    def to_json(self) -> dict:
        return {
            "frames": int(self.frames),
            "nodes": int(self.nodes),
            "wall_clock_s": float(self.wall_clock_s),
        }


@dataclass
class OptionResult:
    success: bool
    state: MetaState
    cost: OptionCost = field(default_factory=OptionCost)
    entry_snapshot: Any = None
    exit_snapshot: Any = None
    info: dict = field(default_factory=dict)
    observed_effect: dict | None = None
    entry_digest: dict | None = None
    exit_digest: dict | None = None
    exit_observable: dict | None = None
    exit_adapter_context: dict | None = None
    boundary: dict | None = None
    entry_record_id: int | None = None
    exit_record_id: int | None = None


@dataclass
class SnapshotRecord:
    """A physical snapshot and the context/attestations needed to restore it."""

    snapshot: Any
    digest: SnapshotDigest
    adapter_context: dict = field(default_factory=dict)
    observable: dict = field(default_factory=dict)
    producer: str | None = None
    source: dict = field(default_factory=dict)
    state: MetaState | None = None
    record_id: int | None = None

    def to_json(self) -> dict:
        return {
            "digest": self.digest.to_json(),
            "adapter_context": dict(self.adapter_context),
            "observable": dict(self.observable),
            "producer": self.producer,
            "source": dict(self.source),
            "state": self.state.to_json() if self.state is not None else None,
            "record_id": self.record_id,
        }


@dataclass(frozen=True)
class SnapshotCommit:
    """Result of adding one successful physical transition to a context."""

    status: str
    record_id: int | None
    record: SnapshotRecord | None = None

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "record_id": self.record_id,
            "record": self.record.to_json() if self.record is not None else None,
        }


@dataclass
class OptionContext:
    """Execution context for resettable opaque-effect option exploration."""

    executor: Any = None
    snapshots: dict[MetaState, Any] = field(default_factory=dict)
    alias_policy: str = "error"
    snapshot_records: dict[MetaState, SnapshotRecord] = field(
        default_factory=dict, init=False)
    physical_records: dict[int, SnapshotRecord] = field(
        default_factory=dict, init=False)
    record_ids_by_state: dict[MetaState, list[int]] = field(
        default_factory=dict, init=False)
    exact_record_index: dict[tuple[MetaState, str], int] = field(
        default_factory=dict, init=False)
    record_arrivals: dict[int, list[dict]] = field(
        default_factory=dict, init=False)
    alias_collisions: list[dict] = field(default_factory=list, init=False)
    restore_events: list[dict] = field(default_factory=list, init=False)
    last_restore_event: dict | None = field(default=None, init=False)
    _next_record_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.alias_policy not in ("error", "multi"):
            raise ValueError(
                "alias_policy must be 'error' (legacy strict mode) or 'multi'"
            )
        declared = list(self.snapshots.items())
        self.snapshots = {}
        for state, snap in declared:
            self.add_root(
                state,
                snap,
                producer="initial_context",
                source={"kind": "declared_root"},
            )

    def _adapter_context(self) -> dict:
        if self.executor is not None and hasattr(self.executor, "snapshot_context"):
            return dict(self.executor.snapshot_context())
        return {}

    def _backend(self) -> str | None:
        if self.executor is None:
            return None
        if hasattr(self.executor, "snapshot_backend"):
            return str(self.executor.snapshot_backend())
        return type(self.executor).__name__

    def _rom_sha1(self) -> str | None:
        if self.executor is None:
            return None
        if hasattr(self.executor, "rom_sha1"):
            value = self.executor.rom_sha1()
            return str(value) if value else None
        core = getattr(self.executor, "core", None)
        value = getattr(core, "rom_sha1", None)
        return str(value) if value else None

    def observable(self) -> dict:
        if self.executor is not None and hasattr(
                self.executor, "snapshot_observable"):
            return dict(self.executor.snapshot_observable())
        return {}

    @staticmethod
    def _observables_match(expected: dict, actual: dict) -> bool:
        if not expected:
            return False
        # Stable-Retro's display buffer is a host-side cache and is not restored
        # by mGBA savestates.  Exact emulated memory is the restore invariant;
        # screen hashes remain diagnostic evidence in each record.
        if "ram_blocks" in expected and "ram_blocks" in actual:
            return expected["ram_blocks"] == actual["ram_blocks"]
        return expected == actual

    def digest(self, snapshot: Any, *,
               adapter_context: dict | None = None) -> SnapshotDigest:
        return snapshot_digest(
            snapshot,
            adapter_context=(self._adapter_context()
                             if adapter_context is None else adapter_context),
            backend=self._backend(),
            rom_sha1=self._rom_sha1(),
        )

    def _record(self, snapshot: Any, *, producer: str | None = None,
                source: dict | None = None,
                adapter_context: dict | None = None,
                observable: dict | None = None,
                state: MetaState | None = None,
                record_id: int | None = None) -> SnapshotRecord:
        context = (self._adapter_context()
                   if adapter_context is None else dict(adapter_context))
        return SnapshotRecord(
            snapshot=snapshot,
            digest=self.digest(snapshot, adapter_context=context),
            adapter_context=context,
            observable=(
                self.observable() if observable is None else dict(observable)
            ),
            producer=producer,
            source=dict(source or {}),
            state=state,
            record_id=record_id,
        )

    def _install_record(
        self,
        state: MetaState,
        candidate: SnapshotRecord,
        *,
        arrival: dict,
    ) -> SnapshotRecord:
        record_id = self._next_record_id
        self._next_record_id += 1
        record = SnapshotRecord(
            snapshot=candidate.snapshot,
            digest=candidate.digest,
            adapter_context=dict(candidate.adapter_context),
            observable=dict(candidate.observable),
            producer=candidate.producer,
            source=dict(candidate.source),
            state=state,
            record_id=record_id,
        )
        self.physical_records[record_id] = record
        self.record_ids_by_state.setdefault(state, []).append(record_id)
        self.exact_record_index[(state, record.digest.full_sha256)] = record_id
        self.record_arrivals[record_id] = [dict(arrival)]
        # The old single-representative dictionaries remain read-compatible.
        # They deliberately expose the first representative only.
        if state not in self.snapshot_records:
            self.snapshot_records[state] = record
            self.snapshots[state] = record.snapshot
        return record

    def add_root(
        self,
        state: MetaState,
        snapshot: Any,
        *,
        producer: str | None = "initial_context",
        source: dict | None = None,
        observable: dict | None = None,
        adapter_context: dict | None = None,
    ) -> int:
        """Register an explicit physical root and return its in-run identity."""
        candidate = self._record(
            snapshot,
            producer=producer,
            source=source or {"kind": "declared_root"},
            observable=observable,
            adapter_context=adapter_context,
            state=state,
        )
        duplicate = self.exact_record_index.get(
            (state, candidate.digest.full_sha256)
        )
        arrival = {
            "kind": "root",
            "producer": producer,
            "source": dict(source or {"kind": "declared_root"}),
        }
        if duplicate is not None:
            self.record_arrivals[duplicate].append(arrival)
            return duplicate
        existing = self.record_ids_by_state.get(state, [])
        if existing and self.alias_policy != "multi":
            retained = self.physical_records[existing[0]]
            collision = {
                "state": state.to_json(),
                "retained": retained.to_json(),
                "rejected": candidate.to_json(),
            }
            self.alias_collisions.append(collision)
            raise StateAliasError(collision)
        record = self._install_record(state, candidate, arrival=arrival)
        if existing:
            self.alias_collisions.append({
                "state": state.to_json(),
                "retained": self.physical_records[existing[0]].to_json(),
                "rejected": record.to_json(),
                "resolution": "retained_both",
                "representative_record_ids": [*existing, int(record.record_id)],
            })
        return int(record.record_id)

    def record_ids_for(self, state: MetaState) -> tuple[int, ...]:
        return tuple(self.record_ids_by_state.get(state, ()))

    def _missing_restore(
        self,
        state: MetaState,
        *,
        record_id: int | None = None,
        status: str = "missing",
    ) -> bool:
        event = {
            "kind": "parent_restore",
            "status": status,
            "state": state.to_json(),
            "physical_record_id": record_id,
        }
        self.restore_events.append(event)
        self.last_restore_event = event
        return False

    def _equivalence_signature(
            self, snapshot: Any,
            adapter_context: dict | None = None) -> dict | None:
        if self.executor is not None and hasattr(
                self.executor, "snapshot_equivalence_signature"):
            current_context = self._adapter_context()
            try:
                if (
                    adapter_context is not None
                    and hasattr(self.executor, "apply_snapshot_context")
                ):
                    self.executor.apply_snapshot_context(adapter_context)
                return dict(
                    self.executor.snapshot_equivalence_signature(snapshot)
                )
            finally:
                if hasattr(self.executor, "apply_snapshot_context"):
                    self.executor.apply_snapshot_context(current_context)
        return None

    def restore_record(
        self,
        record_id: int | None,
        *,
        expected_state: MetaState,
    ) -> bool:
        """Restore one exact in-run representative.

        Explicit ``None`` denotes a symbolic lineage.  It must never fall back
        to an unrelated physical representative that happens to share the same
        ``MetaState``.
        """
        if record_id is None:
            return self._missing_restore(
                expected_state, record_id=None, status="symbolic_parent"
            )
        record = self.physical_records.get(int(record_id))
        if self.executor is None or record is None:
            return self._missing_restore(
                expected_state, record_id=int(record_id), status="missing"
            )
        if record.state != expected_state:
            return self._missing_restore(
                expected_state,
                record_id=int(record_id),
                status="record_state_mismatch",
            )
        snap = record.snapshot
        if hasattr(self.executor, "apply_snapshot_context"):
            self.executor.apply_snapshot_context(record.adapter_context)
        self.executor.restore(snap)
        actual = self._record(
            self.executor.snapshot()
            if hasattr(self.executor, "snapshot") else snap,
            producer=record.producer,
            source=record.source,
        )
        raw_match = (
            actual.digest.full_sha256 == record.digest.full_sha256
        )
        wrapper_match = (
            actual.digest.metadata_sha256 == record.digest.metadata_sha256
            and actual.digest.adapter_context_sha256
            == record.digest.adapter_context_sha256
        )
        observable_match = (
            self._observables_match(record.observable, actual.observable)
        )
        expected_suffix = None
        actual_suffix = None
        suffix_match = False
        if not raw_match and wrapper_match and observable_match:
            expected_suffix = self._equivalence_signature(
                record.snapshot, record.adapter_context)
            actual_suffix = self._equivalence_signature(
                actual.snapshot, record.adapter_context)
            suffix_match = (
                expected_suffix is not None
                and expected_suffix == actual_suffix
            )
        if raw_match and record.digest.exact and actual.digest.exact:
            status = "restored_exact_bytes"
        elif raw_match:
            status = "restored_deterministic_digest"
        elif wrapper_match and observable_match and suffix_match:
            status = "restored_suffix_attested"
        else:
            status = "restore_mismatch"
        event = {
            "kind": "parent_restore",
            "status": status,
            "state": expected_state.to_json(),
            "physical_record_id": int(record_id),
            "expected": record.digest.to_json(),
            "actual": actual.digest.to_json(),
            "producer": record.producer,
            "source": dict(record.source),
            "raw_snapshot_match": raw_match,
            "wrapper_metadata_match": wrapper_match,
            "observable_match": observable_match,
            "fixed_suffix_match": suffix_match,
            "expected_fixed_suffix": expected_suffix,
            "actual_fixed_suffix": actual_suffix,
            "speculative_noop_frames": (
                0 if expected_suffix is None
                else expected_suffix["frames"] + actual_suffix["frames"]
            ),
            "expected_observable": dict(record.observable),
            "actual_observable": dict(actual.observable),
        }
        self.restore_events.append(event)
        self.last_restore_event = event
        if status == "restore_mismatch":
            raise RuntimeError(
                "executor restore did not reproduce the recorded physical state"
            )
        return True

    def restore(self, state: MetaState) -> bool:
        """Legacy state-only restore.

        Strict contexts retain the historical behavior.  Multi contexts require
        an explicit record ID once a symbolic state has multiple physical
        representatives; silently selecting the first would recreate the alias.
        """
        record_ids = self.record_ids_for(state)
        if not record_ids:
            return self._missing_restore(state)
        if len(record_ids) > 1:
            self._missing_restore(state, status="ambiguous_physical_parent")
            raise RuntimeError(
                "ambiguous physical parent: restore by physical record ID"
            )
        return self.restore_record(record_ids[0], expected_state=state)

    def commit_transition(
        self,
        state: MetaState,
        snapshot: Any,
        *,
        parent_record_id: int | None,
        option_id: str,
        observable: dict | None = None,
        adapter_context: dict | None = None,
        source: dict | None = None,
    ) -> SnapshotCommit:
        """Commit one accepted physical successor without collapsing aliases."""
        if snapshot is None:
            return SnapshotCommit("no_snapshot", None, None)
        source_payload = dict(source or {
            "kind": "option_exit",
            "option": option_id,
            "parent_record_id": parent_record_id,
        })
        candidate = self._record(
            snapshot,
            producer=option_id,
            source=source_payload,
            observable=observable,
            adapter_context=adapter_context,
            state=state,
        )
        arrival = {
            "kind": "option_exit",
            "option": option_id,
            "parent_record_id": parent_record_id,
            "source": source_payload,
        }
        duplicate = self.exact_record_index.get(
            (state, candidate.digest.full_sha256)
        )
        if duplicate is not None:
            self.record_arrivals[duplicate].append(arrival)
            return SnapshotCommit(
                "exact_duplicate",
                duplicate,
                self.physical_records[duplicate],
            )

        existing = list(self.record_ids_by_state.get(state, ()))
        if existing and self.alias_policy != "multi":
            collision = {
                "state": state.to_json(),
                "retained": self.physical_records[existing[0]].to_json(),
                "rejected": candidate.to_json(),
            }
            self.alias_collisions.append(collision)
            raise StateAliasError(collision)

        record = self._install_record(state, candidate, arrival=arrival)
        status = "stored"
        if existing:
            status = "retained_distinct_representative"
            self.alias_collisions.append({
                "state": state.to_json(),
                "retained": self.physical_records[existing[0]].to_json(),
                "rejected": record.to_json(),
                "resolution": "retained_both",
                "representative_record_ids": [
                    *existing,
                    int(record.record_id),
                ],
                "parent_record_id": parent_record_id,
                "option": option_id,
            })
        return SnapshotCommit(status, int(record.record_id), record)

    def remember(self, state: MetaState, snapshot: Any, *,
                 producer: str | None = None,
                 source: dict | None = None,
                 observable: dict | None = None,
                 adapter_context: dict | None = None,
                 replace_equivalent: bool = False) -> str:
        """Commit a successful transition snapshot or fail on an abstraction alias."""
        if snapshot is None:
            return "no_snapshot"
        candidate = self._record(
            snapshot,
            producer=producer,
            source=source or {"kind": "option_exit"},
            observable=observable,
            adapter_context=adapter_context,
            state=state,
        )
        retained = self.snapshot_records.get(state)
        if retained is None:
            self._install_record(
                state,
                candidate,
                arrival={
                    "kind": "legacy_remember",
                    "producer": producer,
                    "source": dict(source or {"kind": "option_exit"}),
                },
            )
            return "stored"
        if retained.digest.full_sha256 == candidate.digest.full_sha256:
            retained_id = int(retained.record_id)
            self.record_arrivals[retained_id].append({
                "kind": "legacy_remember",
                "producer": producer,
                "source": dict(source or {"kind": "option_exit"}),
            })
            if replace_equivalent:
                replacement = SnapshotRecord(
                    snapshot=candidate.snapshot,
                    digest=candidate.digest,
                    adapter_context=dict(candidate.adapter_context),
                    observable=dict(candidate.observable),
                    producer=candidate.producer,
                    source=dict(candidate.source),
                    state=state,
                    record_id=retained_id,
                )
                self.snapshots[state] = snapshot
                self.snapshot_records[state] = replacement
                self.physical_records[retained_id] = replacement
                return "replaced_same"
            return "same"
        collision = {
            "state": state.to_json(),
            "retained": retained.to_json(),
            "rejected": candidate.to_json(),
        }
        self.alias_collisions.append(collision)
        raise StateAliasError(collision)


Precondition = Callable[[MetaState], bool]
Runner = Callable[[MetaState, OptionContext], OptionResult]
Verifier = Callable[[Any], bool]
_UNSPECIFIED_RECORD_ID = object()


class _SMA4ProvenanceMixin:
    """Shared exact-state context and transition ledger for GBA executors."""

    core: Any

    def _init_provenance(self) -> None:
        self._active_option: str | None = None
        self._trace_events: list[dict] = []

    def snapshot_backend(self) -> str:
        return "stable-retro/mgba"

    def rom_sha1(self) -> str | None:
        value = getattr(self.core, "rom_sha1", None)
        return str(value) if value else None

    def snapshot_context(self) -> dict:
        return {
            "game_id": getattr(self.core, "game_id", None),
            "level_id": getattr(self.core, "level_id", None),
            "world": getattr(self.core, "world", None),
            "stage": getattr(self.core, "stage", None),
            "start_lives": getattr(self.core, "_start_lives", None),
            "action_names": list(getattr(self.core, "action_names", [])),
        }

    def snapshot_observable(self) -> dict:
        """Exact emulated-memory hashes plus a diagnostic host display hash."""
        blocks = {}
        memory = getattr(getattr(self.core, "env", None), "data", None)
        memory = getattr(memory, "memory", None)
        for base, block in sorted(getattr(memory, "blocks", {}).items()):
            raw = bytes(block)
            blocks[f"0x{int(base):08X}"] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        screen = self.core.last_obs
        screen_raw = (
            screen.tobytes(order="C")
            if hasattr(screen, "tobytes")
            else bytes(screen)
        )
        return {
            "ram_blocks": blocks,
            "screen_sha256": hashlib.sha256(screen_raw).hexdigest(),
            "screen_shape": list(getattr(screen, "shape", ())),
        }

    def snapshot_equivalence_signature(self, snapshot: Any, *,
                                       frames: int = 4) -> dict:
        """Replay a fixed NOOP suffix as a falsifier for hidden-state aliases.

        Savestate bytes from mGBA are not canonical after load.  Matching all
        emulated memory at the boundary is necessary but cannot see CPU/register
        state, so this additionally compares a deterministic forward suffix.
        It is an attestation, not a proof of equivalence for every action suffix.
        """
        current = self.snapshot()
        try:
            self.core.restore(snapshot)
            samples = []
            for frame in range(1, int(frames) + 1):
                _obs, info, done = self.core._step_buttons(())
                observable = self.snapshot_observable()
                samples.append({
                    "frame": frame,
                    "ram_blocks": observable["ram_blocks"],
                    "info": info,
                    "done": bool(done),
                })
            signature = hashlib.sha256(stable_encode(samples)).hexdigest()
        finally:
            self.core.restore(current)
        return {
            "kind": "fixed_noop_suffix",
            "frames": int(frames),
            "sha256": signature,
        }

    def apply_snapshot_context(self, context: dict) -> None:
        for attr, key in (
            ("level_id", "level_id"),
            ("world", "world"),
            ("stage", "stage"),
            ("_start_lives", "start_lives"),
        ):
            if key in context and context[key] is not None:
                setattr(self.core, attr, context[key])

    def _set_adapter_context(self, *, reason: str,
                             knowledge_tier: KnowledgeTier,
                             **changes: Any) -> None:
        before = self.snapshot_context()
        for attr, value in changes.items():
            setattr(self.core, attr, value)
        self._trace_events.append({
            "kind": "adapter_context_write",
            "reason": reason,
            "knowledge_tier": int(knowledge_tier),
            "before": before,
            "after": self.snapshot_context(),
            "changes": dict(changes),
        })

    def _snapshot_digest_json(self, snapshot: Any) -> dict:
        return snapshot_digest(
            snapshot,
            adapter_context=self.snapshot_context(),
            backend=self.snapshot_backend(),
            rom_sha1=self.rom_sha1(),
        ).to_json()

    def _record_solution_artifact(
        self,
        path: str | Path,
        solution: dict,
        *,
        action_field: str,
        knowledge_tier: KnowledgeTier,
    ) -> None:
        path = Path(path)
        action_payload = {
            "action_field": action_field,
            "actions": solution.get(action_field) or [],
            "action_names": solution.get("action_names") or [],
            "chunk_frames": int(solution.get("chunk_frames", 1)),
        }
        self._trace_events.append({
            "kind": "solution_artifact",
            "knowledge_tier": int(knowledge_tier),
            "path": str(path),
            "file_sha256": file_sha256(path),
            "payload_kind": "declared_manifest_actions",
            "declared_action_payload_sha256": hashlib.sha256(
                stable_encode(action_payload)
            ).hexdigest(),
            "n_actions": len(action_payload["actions"]),
            "chunk_frames": action_payload["chunk_frames"],
        })

    def begin_option_trace(self, option_id: str, entry_digest: dict | None) -> None:
        self._active_option = str(option_id)
        self._trace_events = []
        self._trace_entry_digest = entry_digest

    def end_option_trace(self) -> dict:
        out = {
            "option": self._active_option,
            "entry_digest": self._trace_entry_digest,
            "events": list(self._trace_events),
            "ram_writes": [
                event for event in self._trace_events
                if event.get("kind") == "ram_write"
            ],
            "snapshot_restores": [
                event for event in self._trace_events
                if event.get("kind") == "snapshot_restore"
            ],
        }
        self._active_option = None
        self._trace_events = []
        self._trace_entry_digest = None
        return out

    def _restore_external_snapshot(
        self,
        snapshot: Any,
        *,
        source_path: str | Path,
        reason: str,
        knowledge_tier: KnowledgeTier,
    ) -> None:
        before = self._snapshot_digest_json(self.snapshot())
        source_path = Path(source_path)
        expected = self._snapshot_digest_json(snapshot)
        self.core.restore(snapshot)
        actual_snapshot = self.snapshot()
        actual = self._snapshot_digest_json(actual_snapshot)
        actual_observable = self.snapshot_observable()
        # mGBA savestate serialization is not byte-canonical after load.  Load
        # the same declared root twice and require emulated memory plus a fixed
        # forward suffix to agree; retain all three raw hashes as provenance.
        self.core.restore(snapshot)
        repeated_snapshot = self.snapshot()
        repeated = self._snapshot_digest_json(repeated_snapshot)
        repeated_observable = self.snapshot_observable()
        source_suffix = self.snapshot_equivalence_signature(snapshot)
        actual_suffix = self.snapshot_equivalence_signature(actual_snapshot)
        repeated_suffix = self.snapshot_equivalence_signature(repeated_snapshot)
        event = {
            "kind": "snapshot_restore",
            "restore_kind": "external_root",
            "reason": reason,
            "knowledge_tier": int(knowledge_tier),
            "source_path": str(source_path),
            "source_file_sha256": file_sha256(source_path),
            "before": before,
            "expected": expected,
            "actual": actual,
            "repeated_load": repeated,
            "emulator_exact_match": (
                expected["emulator_sha256"] == actual["emulator_sha256"]
            ),
            "full_wrapper_match": (
                expected["full_sha256"] == actual["full_sha256"]
            ),
            "wrapper_metadata_redecoded": (
                expected["metadata_sha256"] != actual["metadata_sha256"]
            ),
            "load_observable": actual_observable,
            "repeated_load_observable": repeated_observable,
            "repeated_load_observable_match": (
                OptionContext._observables_match(
                    actual_observable, repeated_observable)
            ),
            "source_fixed_suffix": source_suffix,
            "actual_fixed_suffix": actual_suffix,
            "repeated_fixed_suffix": repeated_suffix,
            "fixed_suffix_match": (
                source_suffix == actual_suffix == repeated_suffix
            ),
            "restore_attempts": 2,
            "speculative_noop_frames": (
                source_suffix["frames"]
                + actual_suffix["frames"]
                + repeated_suffix["frames"]
            ),
        }
        self._trace_events.append(event)
        if not (
            event["repeated_load_observable_match"]
            and event["fixed_suffix_match"]
        ):
            raise RuntimeError(
                f"external snapshot load is not observable-deterministic for "
                f"{source_path}"
            )

    def _restore_runtime_snapshot(self, snapshot: Any, *, reason: str,
                                  frame: int | None = None,
                                  expected_observable: dict | None = None) -> None:
        before = self._snapshot_digest_json(self.snapshot())
        expected = self._snapshot_digest_json(snapshot)
        self.core.restore(snapshot)
        actual_snapshot = self.snapshot()
        actual = self._snapshot_digest_json(actual_snapshot)
        actual_observable = self.snapshot_observable()
        raw_match = expected["full_sha256"] == actual["full_sha256"]
        wrapper_match = (
            expected["metadata_sha256"] == actual["metadata_sha256"]
            and expected["adapter_context_sha256"]
            == actual["adapter_context_sha256"]
        )
        observable_match = (
            expected_observable is not None
            and OptionContext._observables_match(
                expected_observable, actual_observable)
        )
        expected_suffix = None
        actual_suffix = None
        suffix_match = False
        if not raw_match and wrapper_match and observable_match:
            expected_suffix = self.snapshot_equivalence_signature(snapshot)
            actual_suffix = self.snapshot_equivalence_signature(actual_snapshot)
            suffix_match = expected_suffix == actual_suffix
        event = {
            "kind": "snapshot_restore",
            "restore_kind": "runtime_rollback",
            "reason": reason,
            "frame": frame,
            "before": before,
            "expected": expected,
            "actual": actual,
            "raw_snapshot_match": raw_match,
            "wrapper_metadata_match": wrapper_match,
            "observable_match": observable_match,
            "expected_observable": expected_observable,
            "actual_observable": actual_observable,
            "expected_fixed_suffix": expected_suffix,
            "actual_fixed_suffix": actual_suffix,
            "fixed_suffix_match": suffix_match,
            "speculative_noop_frames": (
                0 if expected_suffix is None
                else expected_suffix["frames"] + actual_suffix["frames"]
            ),
            "restore_attested": (
                raw_match or (
                    wrapper_match and observable_match and suffix_match
                )
            ),
        }
        self._trace_events.append(event)
        if not event["restore_attested"]:
            raise RuntimeError(f"runtime snapshot restore mismatch: {reason}")


class SMA4SolutionExecutor(_SMA4ProvenanceMixin):
    """Adapter from option execution to the existing cached SMA4 replay path."""

    def __init__(self, core: Any, *, settle: bool = True):
        self.core = core
        self.settle = bool(settle)
        self._init_provenance()

    def restore(self, snapshot: Any) -> None:
        self.core.restore(snapshot)

    def snapshot(self) -> Any:
        return self.core.snapshot()

    def execute_clear_solution(self, solution_path: str | Path,
                               entry_snapshot: str | None = None) -> tuple[dict, Any]:
        from mario.overworld_search import _load_snapshot, replay_cached_solution

        solution_path = Path(solution_path)
        solution = json.loads(solution_path.read_text())
        level_id = str(solution.get("level_id") or solution_path.stem)
        try:
            world_text, stage_text = level_id.split("-", 1)
            world = int(world_text)
            stage = int(stage_text)
        except ValueError as exc:
            raise ValueError(
                f"clear solution has non-numeric level_id {level_id!r}"
            ) from exc
        self._set_adapter_context(
            reason="label_cached_clear_level_root",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
            level_id=level_id,
            world=world,
            stage=stage,
        )
        self._record_solution_artifact(
            solution_path,
            solution,
            action_field="path",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        )
        entry_path = Path(entry_snapshot or solution.get("snapshot"))
        entry_snap = _load_snapshot(entry_path)
        self._restore_external_snapshot(
            entry_snap,
            source_path=entry_path,
            reason="clear_level_cached_entry",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        )
        summary = replay_cached_solution(self.core, solution, entry_snap=None,
                                         settle=self.settle)
        summary["solution_file_sha256"] = file_sha256(solution_path)
        summary["entry_snapshot_file_sha256"] = file_sha256(entry_path)
        summary["settle_required"] = self.settle
        return summary, self.core.snapshot()


class SMA4WhistleExecutor(_SMA4ProvenanceMixin):
    """Executor for the SMA4 two-whistle warp route.

    Tier-1 path hand-grants whistles.  Tier-2 paths replay the verified
    `acquire_whistle_1_3` and `acquire_whistle_fortress` solutions from their
    declared cached roots.  Whistle spend options remain effect-opaque: the
    meta-search must execute them against the emulator and read the resulting
    RAM state.
    """

    INVENTORY_START = 0x03002C2E
    INVENTORY_SLOTS = 36
    WARP_WHISTLE = 0x0C
    ITEM_MENU_OPEN = 0x03003772
    MAP_EVENT = 0x03003774
    MAP_DEST_OR_REGION = 0x03003CB7
    WARP_ZONE_WORLD_RAW = 8
    WORLD_8_RAW = 7
    ACQUIRE_WHISTLE_1_3 = Path("data/solutions/sma4/acquire_whistle_1_3.json")
    ACQUIRE_WHISTLE_FORTRESS = Path(
        "data/solutions/sma4/acquire_whistle_fortress.json")

    def __init__(self, core: Any):
        self.core = core
        self._init_provenance()

    def restore(self, snapshot: Any) -> None:
        self.core.restore(snapshot)

    def snapshot(self) -> Any:
        return self.core.snapshot()

    def _read_u8(self, addr: int) -> int:
        return int(self.core._read_u8(addr))

    def _write_u8(self, addr: int, value: int, *, reason: str,
                  knowledge_tier: KnowledgeTier,
                  frame: int | None = None) -> None:
        before = self._read_u8(addr)
        self.core.env.data.memory.assign(addr, "|u1", int(value) & 0xFF)
        after = self._read_u8(addr)
        self._trace_events.append({
            "kind": "ram_write",
            "address": f"0x{int(addr):08X}",
            "before": before,
            "requested": int(value) & 0xFF,
            "after": after,
            "changed": before != after,
            "reason": reason,
            "knowledge_tier": int(knowledge_tier),
            "frame": frame,
        })

    def _step(self, buttons: tuple[str, ...], frames: int) -> int:
        for _ in range(frames):
            self.core._step_buttons(buttons)
        return int(frames)

    def _wait_until(self, predicate, *, max_frames: int = 1800,
                    step: int = 12) -> tuple[bool, int]:
        elapsed = 0
        while elapsed < max_frames:
            self._step((), step)
            elapsed += step
            if predicate():
                return True, elapsed
        return False, elapsed

    def _open_and_use_selected_item(self) -> int:
        frames = 0
        frames += self._step(("L",), 6)
        frames += self._step((), 30)
        frames += self._step(("A",), 6)
        return frames

    def inventory(self) -> list[int]:
        return [self._read_u8(self.INVENTORY_START + i)
                for i in range(self.INVENTORY_SLOTS)]

    def _cursor_info(self) -> dict:
        """Return one atomic adapter-resolved cursor observation."""
        if hasattr(self.core, "map_cursor_info"):
            return dict(self.core.map_cursor_info())
        cursor = self.core.last_info.get("cursor")
        if cursor is None:
            cursor = (
                self._read_u8(self.core.MAP_CURSOR_X),
                self._read_u8(self.core.MAP_CURSOR_Y),
            )
        return {
            "cursor": (int(cursor[0]), int(cursor[1])),
            "source": self.core.last_info.get(
                "cursor_source", "legacy_or_fake"),
            "raw_pointer": self.core.last_info.get("cursor_pointer"),
            "resolved_pointer": self.core.last_info.get(
                "cursor_resolved_pointer"),
            "pointer_in_iwram": None,
            "pointer_recognized": None,
            "resolved": bool(self.core.last_info.get(
                "cursor_resolved", True)),
            "resolver": "fake_or_legacy_cursor_v1",
            "legacy": tuple(self.core.last_info.get(
                "cursor_legacy", cursor)),
        }

    def _cursor_resolved(self, cursor_info: dict | None = None) -> bool:
        cursor_info = cursor_info or self._cursor_info()
        return bool(cursor_info.get("resolved", True))

    def _cursor(self) -> tuple[int, int]:
        """Return the adapter-resolved logical map cursor."""
        cursor = self._cursor_info()["cursor"]
        return int(cursor[0]), int(cursor[1])

    def _cursor_matches(self, expected: tuple[int, int]) -> bool:
        cursor_info = self._cursor_info()
        return (
            self._cursor_resolved(cursor_info)
            and tuple(cursor_info["cursor"]) == tuple(expected)
        )

    def sample(self, label: str, frame: int = 0) -> dict:
        cursor_info = self._cursor_info()
        cursor = cursor_info["cursor"]
        return {
            "label": label,
            "frame": int(frame),
            "world_raw_0_indexed": self._read_u8(self.core.WORLD),
            "world_normalized": int(self.core.last_info.get("world", 0)),
            "is_warp_zone": bool(self.core.last_info.get("is_warp_zone", False)),
            "cursor": list(cursor),
            "cursor_source": cursor_info["source"],
            "cursor_pointer": cursor_info["raw_pointer"],
            "cursor_resolved_pointer": cursor_info["resolved_pointer"],
            "cursor_pointer_in_iwram": cursor_info.get("pointer_in_iwram"),
            "cursor_pointer_recognized": cursor_info.get("pointer_recognized"),
            "cursor_resolved": self._cursor_resolved(cursor_info),
            "cursor_resolver": cursor_info.get("resolver"),
            "cursor_legacy": list(cursor_info["legacy"]),
            "mode": self.core.last_info.get("mode"),
            "panel_slots_raw": int(self.core.last_info.get("cleared", 0)),
            "time": int(self.core.last_info.get("time", 0)),
            "inventory_first4": self.inventory()[:4],
            "item_menu_open": self._read_u8(self.ITEM_MENU_OPEN),
            "map_event": self._read_u8(self.MAP_EVENT),
            "map_dest_or_region": self._read_u8(self.MAP_DEST_OR_REGION),
        }

    def decode_meta_state(self) -> dict:
        raw_world = self._read_u8(self.core.WORLD)
        inventory = self.inventory()
        cursor_info = self._cursor_info()
        cursor = cursor_info["cursor"]
        return {
            "world": 9 if raw_world == self.WARP_ZONE_WORLD_RAW
            else raw_world + 1,
            "node": list(cursor),
            "panel_slots_raw": int(self.core.last_info.get("cleared", 0)),
            "inventory": sorted(
                "whistle" for value in inventory if value == self.WARP_WHISTLE
            ),
            "mode": self.core.last_info.get("mode"),
            "raw_world": raw_world,
            "cursor_source": cursor_info["source"],
            "cursor_pointer": cursor_info["raw_pointer"],
            "cursor_resolved_pointer": cursor_info["resolved_pointer"],
            "cursor_legacy": list(cursor_info["legacy"]),
            "cursor_resolved": self._cursor_resolved(cursor_info),
        }

    @staticmethod
    def compare_meta_state(state: MetaState, decoded: dict) -> dict:
        checks = {
            "world": state.world == int(decoded["world"]),
            "node": list(state.node) == list(decoded["node"]),
            "inventory": list(state.inventory) == list(decoded["inventory"]),
            "mode": decoded.get("mode") == "overworld",
            "cursor_resolved": bool(decoded.get("cursor_resolved", True)),
        }
        return {
            "matches": all(checks.values()),
            "checks": checks,
            "declared": state.to_json(),
            "decoded": decoded,
            "note": (
                "flags and symbolic cleared bits are not decoded from RAM; "
                "0x03002C52 is exposed only as opaque panel_slots_raw"
            ),
        }

    def grant_whistles(self, count: int = 2) -> dict:
        for i in range(self.INVENTORY_SLOTS):
            self._write_u8(
                self.INVENTORY_START + i,
                0,
                reason="hand_grant_clear_inventory",
                knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
                frame=0,
            )
        for i in range(int(count)):
            self._write_u8(
                self.INVENTORY_START + i,
                self.WARP_WHISTLE,
                reason="hand_grant_warp_whistle",
                knowledge_tier=KnowledgeTier.TIER1_ITEM_GIVEN,
                frame=0,
            )
        self.core._last_info = self.core._normalize_info(self.core.last_info)
        return {
            "success": True,
            "cost_frames": 0,
            "samples": [self.sample(f"hand_granted_{count}_whistles")],
            "whistles": int(count),
        }

    def _load_snapshot_file(self, path: Path) -> Any:
        obj = pickle.loads(Path(path).read_bytes())
        if isinstance(obj, dict):
            for key in ("entry_snap", "snapshot", "snap"):
                if key in obj:
                    return obj[key]
        return obj

    def acquire_whistle_1_3(
            self,
            solution_path: str | Path | None = None,
            *,
            entry_snapshot: str | Path | None = None,
    ) -> dict:
        """Replay the verified 1-3 white-block AcquireWhistle option.

        Initiation restores the cached P-Wing 1-3 entry snapshot (documented as
        an injected precondition until map P-Wing use is itself an option).
        Success = inventory slot contains warp whistle ``0x0C`` after the
        Toad-house chest open + map exit.
        """
        sol_path = Path(solution_path or self.ACQUIRE_WHISTLE_1_3)
        sol = json.loads(sol_path.read_text())
        self._record_solution_artifact(
            sol_path,
            sol,
            action_field="path_buttons",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        )
        entry = Path(entry_snapshot or sol.get("entry_snapshot")
                     or "runs/sma4_cache/1-3_pwing_entry.pkl")
        buttons = [tuple(b) for b in sol["path_buttons"]]
        prior = self.inventory()
        prior_count = sum(1 for v in prior if v == self.WARP_WHISTLE)
        frames = 0
        samples = [self.sample("before_acquire_whistle_1_3", frames)]
        self._set_adapter_context(
            reason="label_cached_1_3_root",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
            level_id="1-3",
            world=1,
            stage=3,
        )
        self._restore_external_snapshot(
            self._load_snapshot_file(entry),
            source_path=entry,
            reason="acquire_whistle_1_3_pwing_entry",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        )
        # Preserve any already-held whistles across the P-Wing entry restore
        # (e.g. fortress acquire ran first; chest must be able to stack).
        for i, val in enumerate(prior[: self.INVENTORY_SLOTS]):
            if int(val):
                self._write_u8(
                    self.INVENTORY_START + i,
                    int(val),
                    reason="merge_predecessor_inventory_into_1_3_root",
                    knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
                    frame=frames,
                )
        samples.append(self.sample("restored_pwing_1_3_entry", frames))
        for bt in buttons:
            self.core._step_buttons(bt)
            frames += 1
        # The recorded option includes its own chest-house exit.  Do not infer
        # that exit from screen brightness or silently repair a truncated path:
        # the option succeeds only if replay itself reaches the overworld.
        frames += self._step((), 12)
        inv = self.inventory()
        after_count = sum(1 for v in inv if v == self.WARP_WHISTLE)
        final_mode = self.core.last_info.get("mode")
        cursor_info = self._cursor_info()
        final_cursor = tuple(cursor_info["cursor"])
        exit_verified = (
            final_mode == "overworld"
            and self._cursor_resolved(cursor_info)
            and final_cursor == (160, 32)
            and self._read_u8(self.ITEM_MENU_OPEN) == 0
        )
        success = (
            after_count >= max(1, prior_count + 1)
            and exit_verified
        )
        samples.append(self.sample("after_acquire_whistle_1_3", frames))
        injected = ["pwing_1_3_entry_snapshot"]
        if prior_count:
            injected.append("prior_whistle_inventory_merged")
        return {
            "success": bool(success),
            "cost_frames": int(frames),
            "samples": samples,
            "inventory_first4": inv[:4],
            "whistle_count": int(after_count),
            "final_mode": final_mode,
            "final_cursor": list(final_cursor),
            "cursor_resolved": self._cursor_resolved(cursor_info),
            "exit_verified": exit_verified,
            "solution": str(sol_path),
            "entry_snapshot": str(entry),
            "knowledge_tier": int(KnowledgeTier.TIER2_BLACK_BOX_OPTION),
            "injected_facts": injected,
        }

    def acquire_whistle_fortress(
            self,
            solution_path: str | Path | None = None,
            *,
            entry_snapshot: str | Path | None = None,
    ) -> dict:
        """Replay verified W1 Fortress AcquireWhistle (spawn → door → chest → map).

        Restores the leaf fortress-spawn entry (not a mid-level door snap).  If
        the live inventory already holds a whistle (e.g. after
        ``acquire_whistle_1_3``), those slots are copied onto the restored entry
        so the chest can stack a second ``0x0C``.  The root seeds P-speed; leaf
        power is re-held after damage.  After the chest, stop the scripted path,
        ``UP`` out of the treasure room, idle, then hold ``B`` so the map accepts
        L-menu.  Success = overworld with at least one more whistle than before
        and L-menu openable.
        """
        sol_path = Path(solution_path or self.ACQUIRE_WHISTLE_FORTRESS)
        sol = json.loads(sol_path.read_text())
        self._record_solution_artifact(
            sol_path,
            sol,
            action_field="path_buttons",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        )
        entry = Path(entry_snapshot or sol.get("entry_snapshot")
                     or "runs/sma4_cache/1-fortress_pwing_leaf_entry.pkl")
        buttons = [tuple(b) for b in sol["path_buttons"]]
        prior = self.inventory()
        prior_count = sum(1 for v in prior if v == self.WARP_WHISTLE)
        target_count = max(1, prior_count + 1)
        frames = 0
        samples = [self.sample("before_acquire_whistle_fortress", frames)]
        self._set_adapter_context(
            reason="label_cached_fortress_root",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
            level_id="1-fortress",
            world=1,
            stage=0,
        )
        self._restore_external_snapshot(
            self._load_snapshot_file(entry),
            source_path=entry,
            reason="acquire_whistle_fortress_pwing_leaf_entry",
            knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
        )
        # Preserve any already-held whistles across the spawn-entry restore.
        for i, val in enumerate(prior[: self.INVENTORY_SLOTS]):
            if int(val):
                self._write_u8(
                    self.INVENTORY_START + i,
                    int(val),
                    reason="merge_predecessor_inventory_into_fortress_root",
                    knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
                    frame=frames,
                )
        # Entry snap already holds leaf/pspeed — do not extra-settle (desyncs path).
        samples.append(self.sample("restored_fortress_pwing_entry", frames))
        for bt in buttons:
            self.core._step_buttons(bt)
            frames += 1
            if int(self.core.last_info.get("powerup") or 0) < 3:
                self._write_u8(
                    self.core.POWERUP,
                    3,
                    reason="fortress_leaf_rehold_after_damage",
                    knowledge_tier=KnowledgeTier.TIER2_BLACK_BOX_OPTION,
                    frame=frames,
                )
            # Truncate at chest — recorded exit settle is re-applied below.
            if sum(1 for v in self.inventory() if v == self.WARP_WHISTLE) >= target_count:
                break
        samples.append(self.sample("after_fortress_chest", frames))
        # Treasure-room exit: UP to map, idle, B-hold unlocks L-menu.
        if self.core.last_info.get("mode") != "overworld":
            for _ in range(120):
                self.core._step_buttons(("UP",))
                frames += 1
                if self.core.last_info.get("mode") == "overworld":
                    break
        if self.core.last_info.get("mode") != "overworld":
            for _ in range(400):
                self.core._step_buttons(("A",))
                frames += 1
                if self.core.last_info.get("mode") == "overworld":
                    break
                self.core._step_buttons(("LEFT",))
                frames += 1
                if self.core.last_info.get("mode") == "overworld":
                    break
        frames += self._step((), 200)
        frames += self._step(("B",), 40)
        samples.append(self.sample("after_fortress_map_settle", frames))
        inv = self.inventory()
        after_count = sum(1 for v in inv if v == self.WARP_WHISTLE)
        menu_ok = False
        if self.core.last_info.get("mode") == "overworld":
            snap = self.snapshot()
            snap_observable = self.snapshot_observable()
            for _ in range(20):
                self.core._step_buttons(("L",))
                frames += 1
                if self._read_u8(self.ITEM_MENU_OPEN):
                    menu_ok = True
                    break
            self._restore_runtime_snapshot(
                snap,
                reason="fortress_menu_probe_rollback",
                frame=frames,
                expected_observable=snap_observable,
            )
            frames += self._step((), 2)
        cursor_info = self._cursor_info()
        final_cursor = tuple(cursor_info["cursor"])
        success = (
            self.core.last_info.get("mode") == "overworld"
            and after_count >= target_count
            and menu_ok
            and self._cursor_resolved(cursor_info)
            and final_cursor == (96, 96)
            and self._read_u8(self.ITEM_MENU_OPEN) == 0
        )
        injected = [
            f for f in (sol.get("injected_facts") or [
                "pwing_fortress_entry_snapshot",
                "leaf_rehold_during_route",
                "pspeed_seeded_in_entry_snapshot",
            ])
            if f not in (
                "fortress_inventory_rehosted_to_pre_door_map",
                "fortress_door_entry_snapshot",
            )
        ]
        if prior_count:
            injected.append("prior_whistle_inventory_merged")
        samples.append(self.sample("after_acquire_whistle_fortress", frames))
        return {
            "success": bool(success),
            "cost_frames": int(frames),
            "samples": samples,
            "inventory_first4": inv[:4],
            "whistle_count": int(after_count),
            "final_cursor": list(final_cursor),
            "cursor_resolved": self._cursor_resolved(cursor_info),
            "menu_probe_succeeded": bool(menu_ok),
            "solution": str(sol_path),
            "entry_snapshot": str(entry),
            "knowledge_tier": int(KnowledgeTier.TIER2_BLACK_BOX_OPTION),
            "injected_facts": injected,
        }

    def use_first_whistle(self) -> dict:
        frames = 0
        samples = [self.sample("before_first_whistle", frames)]
        before_count = sum(
            1 for value in self.inventory() if value == self.WARP_WHISTLE
        )
        if before_count == 0:
            samples.append(self.sample("first_whistle_missing_inventory", frames))
            return {
                "success": False,
                "cost_frames": frames,
                "samples": samples,
                "reason": "no_whistle_in_inventory",
            }
        frames += self._open_and_use_selected_item()
        samples.append(self.sample("after_first_use_input", frames))
        # Wait for raw world 8, then require the canonical first warp-zone cell.
        # The adapter resolves the active cursor storage pair; no RAM repair is
        # needed when the 1-3 Toad-house exit leaves the primary pair stale.
        ok, elapsed = self._wait_until(
            lambda: self._read_u8(self.core.WORLD) == self.WARP_ZONE_WORLD_RAW,
            max_frames=2200,
        )
        frames += elapsed
        if ok:
            ok2, elapsed2 = self._wait_until(
                lambda: self._cursor_matches((64, 80)),
                max_frames=900,
            )
            frames += elapsed2
            ok = ok and ok2
        samples.append(self.sample("after_first_whistle_warp_zone", frames))
        if not ok:
            return {"success": False, "cost_frames": frames, "samples": samples}
        frames += self._step((), 180)
        samples.append(self.sample("after_first_whistle_settled", frames))
        after_count = sum(
            1 for value in self.inventory() if value == self.WARP_WHISTLE
        )
        final = samples[-1]
        terminal_invariants = {
            "raw_world_is_warp_zone": (
                final["world_raw_0_indexed"] == self.WARP_ZONE_WORLD_RAW
            ),
            "cursor_is_first_warp_cell": final["cursor"] == [64, 80],
            "cursor_resolved": bool(final["cursor_resolved"]),
            "mode_is_overworld": final.get("mode") == "overworld",
            "item_menu_closed": final.get("item_menu_open") == 0,
            "whistle_inventory_decremented": after_count == before_count - 1,
        }
        success = all(terminal_invariants.values())
        return {
            "success": bool(success),
            "cost_frames": frames,
            "samples": samples,
            "terminal_invariants": terminal_invariants,
            "reason": None if success else "first_whistle_postcondition_failed",
        }

    def use_second_whistle(self) -> dict:
        """Spend a remaining inventory whistle in the first warp-zone map.

        Requires a real ``0x0C`` already in inventory (from ``grant_whistles(2)``
        or a second AcquireWhistle).  Does **not** RAM-poke / re-grant — that
        injected fact was retired once the two-whistle inventory path was
        verified without it (``runs/20260715-fortress-unlock/``).
        """
        frames = 0
        samples = [self.sample("before_second_whistle", frames)]
        if self.WARP_WHISTLE not in self.inventory():
            samples.append(self.sample("second_whistle_missing_inventory", frames))
            return {
                "success": False,
                "cost_frames": frames,
                "samples": samples,
                "reason": "no_whistle_in_inventory",
            }
        if not self._cursor_matches((64, 80)):
            samples.append(self.sample(
                "second_whistle_invalid_source_cursor", frames))
            return {
                "success": False,
                "cost_frames": frames,
                "samples": samples,
                "reason": "not_at_first_warp_zone_cell",
            }
        before_count = sum(
            1 for value in self.inventory() if value == self.WARP_WHISTLE
        )
        frames += self._open_and_use_selected_item()
        samples.append(self.sample("after_second_use_input", frames))
        ok, elapsed = self._wait_until(
            lambda: self._read_u8(self.core.WORLD) == self.WARP_ZONE_WORLD_RAW
            and self._cursor_matches((128, 144)),
            max_frames=2400,
        )
        frames += elapsed
        samples.append(self.sample("after_second_whistle_warp_zone_5_8", frames))
        if not ok:
            return {"success": False, "cost_frames": frames, "samples": samples}
        frames += self._step((), 480)
        samples.append(self.sample("after_second_whistle_settled", frames))
        after_count = sum(
            1 for value in self.inventory() if value == self.WARP_WHISTLE
        )
        final = samples[-1]
        terminal_invariants = {
            "raw_world_is_warp_zone": (
                final["world_raw_0_indexed"] == self.WARP_ZONE_WORLD_RAW
            ),
            "cursor_is_second_warp_cell": final["cursor"] == [128, 144],
            "cursor_resolved": bool(final["cursor_resolved"]),
            "mode_is_overworld": final.get("mode") == "overworld",
            "item_menu_closed": final.get("item_menu_open") == 0,
            "whistle_inventory_decremented": after_count == before_count - 1,
        }
        success = all(terminal_invariants.values())
        return {
            "success": bool(success),
            "cost_frames": frames,
            "samples": samples,
            "terminal_invariants": terminal_invariants,
            "reason": None if success else "second_whistle_postcondition_failed",
        }

    def _probe_cursor_responsiveness(
            self, *, retained_frame: int | None = None) -> tuple[bool, int, list[dict]]:
        """Falsify a stuck map against a matched NOOP control.

        These frames are planner evaluation work, not option holding time.
        The final rollback is attested through the same runtime provenance path
        as other speculative checks.
        """
        snapshot = self.snapshot()
        observable = self.snapshot_observable()
        initial = self._cursor()
        attempts: list[dict] = []
        evaluation_frames = 0
        responsive = False

        def trace(buttons: tuple[str, ...]) -> list[list[int]]:
            nonlocal evaluation_frames
            samples = []
            for _ in range(12):
                self.core._step_buttons(buttons)
                evaluation_frames += 1
                samples.append(list(self._cursor()))
            for _ in range(6):
                self.core._step_buttons(())
                evaluation_frames += 1
                samples.append(list(self._cursor()))
            return samples

        try:
            noop_trace = trace(())
            attempts.append({
                "direction": "NOOP_CONTROL",
                "before": list(initial),
                "after": noop_trace[-1],
                "diverged_from_noop": False,
                "trace_sha256": hashlib.sha256(
                    stable_encode(noop_trace)).hexdigest(),
            })
            for direction in ("LEFT", "RIGHT", "UP", "DOWN"):
                self._restore_runtime_snapshot(
                    snapshot,
                    reason="world8_cursor_probe_direction_reset",
                    frame=retained_frame,
                    expected_observable=observable,
                )
                direction_trace = trace((direction,))
                diverged = direction_trace != noop_trace
                attempts.append({
                    "direction": direction,
                    "before": list(initial),
                    "after": direction_trace[-1],
                    "diverged_from_noop": diverged,
                    "trace_sha256": hashlib.sha256(
                        stable_encode(direction_trace)).hexdigest(),
                })
                if diverged:
                    responsive = True
                    break
        finally:
            self._restore_runtime_snapshot(
                snapshot,
                reason="world8_cursor_probe_rollback",
                frame=retained_frame,
                expected_observable=observable,
            )
            self._trace_events.append({
                "kind": "behavioral_probe",
                "probe": "directional_map_cursor_responsiveness",
                "knowledge_tier": int(KnowledgeTier.TIER0_GENERIC),
                "result": bool(responsive),
                "evaluation_frames": int(evaluation_frames),
                "attempts": attempts,
                "rolled_back": True,
            })
        return responsive, evaluation_frames, attempts

    def select_world8_pipe(self) -> dict:
        frames = 0
        samples = [self.sample("before_select_world8_pipe", frames)]
        frames += self._step(("RIGHT",), 16)
        frames += self._step((), 60)
        samples.append(self.sample("moved_to_world8_pipe", frames))
        frames += self._step(("A",), 16)
        frames += self._step((), 240)
        samples.append(self.sample("bowser_letter", frames))
        frames += self._step(("A",), 12)
        frames += self._step((), 240)
        samples.append(self.sample("world8_map", frames))
        cursor_responsive, evaluation_frames, probe_attempts = (
            self._probe_cursor_responsiveness(retained_frame=frames)
        )
        samples.append(self.sample(
            "world8_map_after_responsiveness_probe_rollback", frames))
        final = samples[-1]
        whistle_count = sum(
            1 for value in self.inventory() if value == self.WARP_WHISTLE
        )
        terminal_invariants = {
            "raw_world_is_world8": (
                final["world_raw_0_indexed"] == self.WORLD_8_RAW
            ),
            "mode_is_overworld": final.get("mode") == "overworld",
            "whistle_inventory_empty": whistle_count == 0,
            "cursor_responsive": bool(cursor_responsive),
            "cursor_is_world8_start": final.get("cursor") == [32, 80],
            "cursor_resolved": bool(final.get("cursor_resolved", True)),
            "item_menu_closed": final.get("item_menu_open") == 0,
            "raw_world": final["world_raw_0_indexed"],
            "mode": final.get("mode"),
            "whistle_count": int(whistle_count),
            "cursor": list(final.get("cursor") or ()),
            "map_event": final.get("map_event"),
            "item_menu_open": final.get("item_menu_open"),
        }
        accepted = all(
            terminal_invariants[key]
            for key in (
                "raw_world_is_world8",
                "mode_is_overworld",
                "whistle_inventory_empty",
                "cursor_responsive",
                "cursor_is_world8_start",
                "cursor_resolved",
                "item_menu_closed",
            )
        )
        return {
            "success": bool(accepted),
            "cost_frames": frames,
            "retained_cost_frames": frames,
            "evaluation_frames": int(evaluation_frames),
            "samples": samples,
            "world8_acceptance": bool(accepted),
            "cursor_responsive": bool(cursor_responsive),
            "terminal_invariants": terminal_invariants,
            "cursor_probe_attempts": probe_attempts,
        }


@dataclass
class Option:
    """One high-level transition in the benchmark option library."""

    id: str
    kind: str
    precondition: Precondition
    runner: Runner
    knowledge_tier: KnowledgeTier = KnowledgeTier.TIER0_GENERIC
    cost: OptionCost = field(default_factory=OptionCost)
    success_rate: float = 1.0
    opaque_effect: bool = False
    known_effect: dict | None = None
    injected_facts: tuple[str, ...] = ()
    entry_snapshot: str | None = None
    exit_snapshot: str | None = None
    source: str | None = None
    verification: dict = field(default_factory=dict)
    verify_fn: Verifier | None = None
    requires_snapshot: bool = False

    def applicable(self, state: MetaState, *, max_tier: KnowledgeTier) -> bool:
        return self.knowledge_tier <= max_tier and self.precondition(state)

    def execute(
        self,
        state: MetaState,
        context: OptionContext | None = None,
        *,
        parent_record_id: int | None | object = _UNSPECIFIED_RECORD_ID,
    ) -> OptionResult:
        context = context or OptionContext()
        physical = self.requires_snapshot and context.executor is not None
        symbolic_manifest_only = (
            self.requires_snapshot and context.executor is None
        )
        restore_event = None
        entry_digest = None
        entry_observable = None
        entry_record_id = None
        if physical:
            if parent_record_id is _UNSPECIFIED_RECORD_ID:
                restored = context.restore(state)
            else:
                restored = context.restore_record(
                    parent_record_id
                    if isinstance(parent_record_id, int) else None,
                    expected_state=state,
                )
            if not restored:
                restore_event = context.last_restore_event
                boundary = {
                    "option": self.id,
                    "from": state.to_json(),
                    "to": state.to_json(),
                    "status": "missing_parent_snapshot",
                    "restore": restore_event,
                    "entry_record_id": (
                        restore_event.get("physical_record_id")
                        if restore_event is not None else None
                    ),
                    "exit_record_id": None,
                    "raw_hash_continuous": False,
                    "composition_attested": False,
                }
                return OptionResult(
                    success=False,
                    state=state,
                    info={
                        "reason": "missing_parent_snapshot",
                        "boundary": boundary,
                    },
                    boundary=boundary,
                    entry_record_id=boundary["entry_record_id"],
                )
            restore_event = context.last_restore_event
            entry_record_id = (
                restore_event.get("physical_record_id")
                if restore_event is not None else None
            )
            runtime_entry = context.executor.snapshot()
            entry_digest = context.digest(runtime_entry).to_json()
            entry_observable = context.observable()
            if hasattr(context.executor, "begin_option_trace"):
                context.executor.begin_option_trace(self.id, entry_digest)

        try:
            result = self.runner(state, context)
        except Exception:
            if physical and hasattr(context.executor, "end_option_trace"):
                context.executor.end_option_trace()
            raise
        if result.cost == OptionCost():
            result.cost = self.cost
        if symbolic_manifest_only:
            result.info = dict(result.info)
            result.info["execution_mode"] = "symbolic_manifest_only"
            result.info["physical_boundary_evidence"] = False
        if physical:
            runner_supplied_exit = result.exit_snapshot
            runner_supplied_exit_digest = (
                context.digest(runner_supplied_exit).to_json()
                if runner_supplied_exit is not None else None
            )
            # The committed successor is always the executor's live state at
            # runner return.  A runner-supplied earlier snapshot is provenance,
            # never a substitute paired with the live observable.
            runtime_exit = context.executor.snapshot()
            result.exit_snapshot = runtime_exit
            exit_digest = context.digest(runtime_exit).to_json()
            exit_observable = context.observable()
            exit_adapter_context = context._adapter_context()
            trace = (
                context.executor.end_option_trace()
                if hasattr(context.executor, "end_option_trace")
                else {"option": self.id, "events": []}
            )
            decoded_meta = None
            meta_comparison = None
            if hasattr(context.executor, "decode_meta_state"):
                decoded_meta = context.executor.decode_meta_state()
                if hasattr(context.executor, "compare_meta_state"):
                    meta_comparison = context.executor.compare_meta_state(
                        result.state, decoded_meta)
                    if result.success and not meta_comparison["matches"]:
                        result.success = False
                        result.info = dict(result.info)
                        result.info["reason"] = (
                            "declared_meta_state_mismatches_physical_exit"
                        )
            external_roots = [
                event for event in trace.get("snapshot_restores", [])
                if event.get("restore_kind") == "external_root"
            ]
            event_tiers = [
                int(event["knowledge_tier"])
                for event in trace.get("events", [])
                if event.get("knowledge_tier") is not None
            ]
            effective_tier = max(
                [int(self.knowledge_tier), *event_tiers]
            )
            boundary = {
                "option": self.id,
                "from": state.to_json(),
                "to": result.state.to_json(),
                "status": "success" if result.success else "failed",
                "restore": restore_event,
                "entry_record_id": entry_record_id,
                "exit_record_id": None,
                "entry": entry_digest,
                "entry_observable": entry_observable,
                "exit": exit_digest,
                "exit_observable": exit_observable,
                "exit_adapter_context": exit_adapter_context,
                "runner_supplied_exit_digest": runner_supplied_exit_digest,
                "external_roots": external_roots,
                "ram_writes": trace.get("ram_writes", []),
                "events": trace.get("events", []),
                "decoded_exit": decoded_meta,
                "meta_state_comparison": meta_comparison,
                "raw_hash_continuous": (
                    bool(restore_event)
                    and restore_event.get("status") == "restored_exact_bytes"
                    and not external_roots
                ),
                "composition_attested": (
                    bool(restore_event)
                    and restore_event.get("status") in (
                        "restored_exact_bytes",
                        "restored_suffix_attested",
                    )
                    and not external_roots
                ),
                "declared_knowledge_tier": int(self.knowledge_tier),
                "effective_knowledge_tier": effective_tier,
            }
            result.info = dict(result.info)
            result.info["boundary"] = boundary
            result.entry_digest = entry_digest
            result.exit_digest = exit_digest
            result.exit_observable = exit_observable
            result.exit_adapter_context = exit_adapter_context
            result.boundary = boundary
            result.entry_record_id = entry_record_id
        return result

    def verify(self, executor: Any = None) -> bool:
        if self.verify_fn is not None:
            return bool(self.verify_fn(executor))
        return bool(self.verification.get("verified", False))

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "knowledge_tier": int(self.knowledge_tier),
            "cost": self.cost.to_json(),
            "success_rate": float(self.success_rate),
            "opaque_effect": bool(self.opaque_effect),
            "known_effect": self.known_effect,
            "injected_facts": list(self.injected_facts),
            "entry_snapshot": self.entry_snapshot,
            "exit_snapshot": self.exit_snapshot,
            "source": self.source,
            "verification": dict(self.verification),
            "requires_snapshot": bool(self.requires_snapshot),
        }


@dataclass
class OptionLibrary:
    options: dict[str, Option] = field(default_factory=dict)

    def add(self, option: Option) -> None:
        if option.id in self.options:
            raise ValueError(f"duplicate option id {option.id!r}")
        self.options[option.id] = option

    def extend(self, options: Iterable[Option]) -> None:
        for option in options:
            self.add(option)

    def applicable(self, state: MetaState, *,
                   max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT) -> list[Option]:
        return [opt for opt in self.options.values()
                if opt.applicable(state, max_tier=max_tier)]

    def summaries(self) -> list[dict]:
        return [self.options[k].summary() for k in sorted(self.options)]


@dataclass
class MetaSearchResult:
    found: bool
    path: list[str] = field(default_factory=list)
    states: list[MetaState] = field(default_factory=list)
    total_cost: OptionCost = field(default_factory=OptionCost)
    branches: int = 0
    option_calls: int = 0
    injected_facts: tuple[str, ...] = ()
    attempted_injected_facts: tuple[str, ...] = ()
    discovered_effects: list[dict] = field(default_factory=list)
    boundaries: list[dict] = field(default_factory=list)
    path_evidence: list[dict] = field(default_factory=list)
    alias_collisions: list[dict] = field(default_factory=list)
    visited: int = 0
    log: list[dict] = field(default_factory=list)
    physical_record_ids: list[int | None] = field(default_factory=list)
    physical_visited: int = 0
    search_nodes_visited: int = 0
    representative_registry: list[dict] = field(default_factory=list)
    transition_observations: list[dict] = field(default_factory=list)
    partition_refinement: dict = field(default_factory=dict)
    repeat_conformance_failures: list[dict] = field(default_factory=list)
    evaluation_work: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "found": self.found,
            "path": list(self.path),
            "states": [s.to_json() for s in self.states],
            "total_cost": self.total_cost.to_json(),
            "branches": self.branches,
            "option_calls": self.option_calls,
            "injected_facts": list(self.injected_facts),
            "attempted_injected_facts": list(self.attempted_injected_facts),
            "discovered_effects": self.discovered_effects,
            "boundaries": self.boundaries,
            "path_evidence": self.path_evidence,
            "alias_collisions": self.alias_collisions,
            "visited": self.visited,
            "log": self.log,
            "physical_record_ids": list(self.physical_record_ids),
            "physical_visited": self.physical_visited,
            "search_nodes_visited": self.search_nodes_visited,
            "representative_registry": self.representative_registry,
            "transition_observations": self.transition_observations,
            "partition_refinement": self.partition_refinement,
            "repeat_conformance_failures": self.repeat_conformance_failures,
            "evaluation_work": self.evaluation_work,
        }


def _transition_evidence(option: Option, result: OptionResult) -> dict:
    mode = result.info.get("execution_mode")
    if mode is None:
        mode = "physical_executor" if result.boundary is not None else "symbolic_model"
    evidence = {
        "option": option.id,
        "execution_mode": mode,
        "physical_boundary_evidence": result.boundary is not None,
        "boundary_status": (
            result.boundary.get("status")
            if result.boundary is not None else None
        ),
    }
    if result.entry_record_id is not None or result.exit_record_id is not None:
        evidence["entry_record_id"] = result.entry_record_id
        evidence["exit_record_id"] = result.exit_record_id
    return evidence


def search_options(library: OptionLibrary, start: MetaState,
                   goal: Callable[[MetaState], bool], *,
                   max_depth: int = 16,
                   max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT,
                   context: OptionContext | None = None) -> MetaSearchResult:
    """Breadth-first resettable search over option effects.

    Opaque options are executed exactly like known-effect options, but their
    transition is only learned from the returned post-state and recorded in
    `discovered_effects`.
    """
    context = context or OptionContext()
    if goal(start):
        return MetaSearchResult(found=True, states=[start], visited=1)

    queue = deque([
        (start, [], [start], OptionCost(), frozenset(), [], [])
    ])
    visited = {start}
    branches = 0
    option_calls = 0
    attempted_injected: set[str] = set()
    discovered: list[dict] = []
    log: list[dict] = []

    while queue:
        (state, path, states, cost, path_injected,
         path_boundaries, path_evidence) = queue.popleft()
        if len(path) >= max_depth:
            continue
        for option in library.applicable(state, max_tier=max_tier):
            branches += 1
            option_calls += 1
            result = option.execute(state, context)
            transition_injected = set(option.injected_facts)
            transition_injected.update(result.info.get("injected_facts") or ())
            attempted_injected.update(transition_injected)
            row = {
                "from": state.to_json(),
                "option": option.id,
                "success": bool(result.success),
                "knowledge_tier": int(option.knowledge_tier),
                "opaque_effect": bool(option.opaque_effect),
                "execution_mode": _transition_evidence(option, result),
            }
            if result.boundary is not None:
                row["boundary"] = result.boundary
                if int(result.boundary.get(
                        "effective_knowledge_tier",
                        int(option.knowledge_tier))) > int(max_tier):
                    row["success"] = False
                    row["reason"] = "runtime_knowledge_tier_exceeds_max"
                    log.append(row)
                    continue
            if not result.success:
                if result.info.get("reason"):
                    row["reason"] = result.info["reason"]
                log.append(row)
                continue
            next_state = result.state
            row["to"] = next_state.to_json()
            row["cost"] = result.cost.to_json()
            if option.opaque_effect:
                observed = result.observed_effect or {
                    "from": state.to_json(),
                    "to": next_state.to_json(),
                }
                observed = {"option": option.id, **observed}
                if observed not in discovered:
                    discovered.append(observed)
                row["observed_effect"] = observed
            log.append(row)
            if result.exit_snapshot is not None:
                context.remember(
                    next_state,
                    result.exit_snapshot,
                    producer=option.id,
                    source={
                        "kind": "option_exit",
                        "option": option.id,
                        "parent_state": state.to_json(),
                    },
                    observable=result.exit_observable,
                    adapter_context=result.exit_adapter_context,
                )
            if next_state in visited:
                continue
            next_path = path + [option.id]
            next_states = states + [next_state]
            next_cost = cost + result.cost
            next_injected = frozenset(
                set(path_injected) | transition_injected
            )
            next_boundaries = path_boundaries + (
                [result.boundary] if result.boundary is not None else []
            )
            next_evidence = path_evidence + [
                _transition_evidence(option, result)
            ]
            if goal(next_state):
                return MetaSearchResult(
                    found=True,
                    path=next_path,
                    states=next_states,
                    total_cost=next_cost,
                    branches=branches,
                    option_calls=option_calls,
                    injected_facts=tuple(sorted(next_injected)),
                    attempted_injected_facts=tuple(sorted(attempted_injected)),
                    discovered_effects=discovered,
                    boundaries=next_boundaries,
                    path_evidence=next_evidence,
                    alias_collisions=list(context.alias_collisions),
                    visited=len(visited) + 1,
                    log=log,
                )
            visited.add(next_state)
            queue.append((
                next_state,
                next_path,
                next_states,
                next_cost,
                next_injected,
                next_boundaries,
                next_evidence,
            ))

    return MetaSearchResult(
        found=False,
        branches=branches,
        option_calls=option_calls,
        injected_facts=(),
        attempted_injected_facts=tuple(sorted(attempted_injected)),
        discovered_effects=discovered,
        alias_collisions=list(context.alias_collisions),
        visited=len(visited),
        log=log,
    )


_SMA4_LEVEL_NODES = {
    "1-1": (64, 32),
    "1-2": (128, 32),
    "1-3": (160, 32),
}


def _level_clear_mask(level_id: str) -> int:
    try:
        _world, stage = level_id.split("-", 1)
        return 1 << (int(stage) - 1)
    except Exception:
        return 0


def load_sma4_clear_level_option(path: str | Path, *,
                                 node: tuple[int, int] | None = None,
                                 clear_mask: int | None = None) -> Option:
    """Wrap a replay-verified SMA4 solution JSON as a Tier-0 ClearLevel option."""
    path = Path(path)
    solution = json.loads(path.read_text())
    level_id = str(solution.get("level_id") or path.stem)
    node = node or _SMA4_LEVEL_NODES.get(level_id, (0, 0))
    clear_mask = _level_clear_mask(level_id) if clear_mask is None else int(clear_mask)
    final = dict(solution.get("final_info") or {})
    final_cursor = tuple(final.get("cursor") or node)
    final_world = int(final.get("world") or level_id.split("-", 1)[0])
    chunk_frames = int(solution.get("chunk_frames", 1))
    path_len = len(solution.get("path") or [])
    post_frames = int(solution.get("post_clear_advance_frames", 0))
    cost = OptionCost(
        frames=path_len * chunk_frames + post_frames,
        nodes=int(solution.get("nodes_expanded", 0) or 0),
        wall_clock_s=float(solution.get("wall_clock_s", 0.0) or 0.0),
    )
    snapshot = solution.get("snapshot")
    verified = bool(solution.get("solved") and solution.get("replay_verified") and snapshot)

    def precondition(state: MetaState) -> bool:
        return state.world == int(level_id.split("-", 1)[0]) and state.node == node

    def runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor_summary = None
        exit_snapshot = None
        if context.executor is not None and hasattr(context.executor, "execute_clear_solution"):
            executor_summary, exit_snapshot = context.executor.execute_clear_solution(
                path, snapshot)
            if (
                not executor_summary.get("solved")
                or (
                    executor_summary.get("settle_required", True)
                    and not executor_summary.get("settled_to_map")
                )
            ):
                return OptionResult(
                    success=False,
                    state=state,
                    cost=cost,
                    info={
                        "level_id": level_id,
                        "source": str(path),
                        "executor_summary": executor_summary,
                    },
                )
        physical_final = (
            dict(executor_summary.get("final_info") or {})
            if executor_summary is not None else {}
        )
        next_state = MetaState(
            world=int(physical_final.get("world", final_world)),
            node=tuple(physical_final.get("cursor") or final_cursor),
            cleared=state.cleared | clear_mask,
            inventory=state.inventory,
            flags=state.flags + (f"clear:{level_id}",),
        )
        return OptionResult(
            success=verified,
            state=next_state,
            cost=cost,
            entry_snapshot=snapshot,
            exit_snapshot=exit_snapshot,
            info={"level_id": level_id, "source": str(path), "final_info": final,
                  "executor_summary": executor_summary},
        )

    return Option(
        id=f"clear_sma4_{level_id}",
        kind="ClearLevel",
        precondition=precondition,
        runner=runner,
        knowledge_tier=KnowledgeTier.TIER0_GENERIC,
        cost=cost,
        success_rate=1.0 if verified else 0.0,
        opaque_effect=False,
        known_effect={
            "cleared_or_mask": clear_mask,
            "to_world": final_world,
            "to_node": list(final_cursor),
            "flag": f"clear:{level_id}",
        },
        injected_facts=("cached_level_entry_snapshot",),
        entry_snapshot=snapshot,
        source=str(path),
        verification={
            "verified": verified,
            "solved": bool(solution.get("solved")),
            "replay_verified": bool(solution.get("replay_verified")),
            "rom_sha1": solution.get("rom_sha1"),
        },
        requires_snapshot=True,
    )


def load_default_sma4_clear_options(
        solution_dir: str | Path = "data/solutions/sma4") -> OptionLibrary:
    lib = OptionLibrary()
    for level_id in ("1-1", "1-2"):
        path = Path(solution_dir) / f"{level_id}.json"
        if path.exists():
            lib.add(load_sma4_clear_level_option(path))
    return lib
