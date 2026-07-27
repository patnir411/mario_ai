"""Physical-representative option search and finite-library refinement.

The legacy planners intentionally key their frontier by :class:`MetaState`.
That remains a useful strict compatibility gate: encountering two physical
states under one symbolic state raises ``StateAliasError``.  This module is the
opt-in successor for experiments where aliases are expected and must be
preserved rather than collapsed.

Record IDs are immutable identities inside one ``OptionContext``.  Snapshot
hashes remain provenance evidence and exact-duplicate indexes; they are not
behavioral-equivalence signatures.  The refinement report is consequently a
finite-library, finite-horizon observational result unless its explicit
completeness and closure gates are satisfied.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import heapq
import itertools
from typing import Any, Callable, Iterable, Mapping

from mario.options import (
    KnowledgeTier,
    MetaSearchResult,
    MetaState,
    Option,
    OptionContext,
    OptionCost,
    OptionLibrary,
    OptionResult,
    _transition_evidence,
)
from mario.provenance import stable_encode


Goal = Callable[[MetaState], bool]
CostFunction = Callable[[OptionCost], float]
SIGNATURE_SCHEMA = "mario-ai.option-observation.v2"


@dataclass(frozen=True)
class SearchNodeKey:
    """A symbolic state paired with one exact in-run physical lineage."""

    state: MetaState
    physical_record_id: int | None


@dataclass
class _PathNode:
    key: SearchNodeKey
    path: list[str]
    states: list[MetaState]
    record_ids: list[int | None]
    cost: OptionCost
    injected: frozenset[str]
    boundaries: list[dict]
    evidence: list[dict]


@dataclass
class _Evaluation:
    accepted: bool
    result: OptionResult
    successor_key: SearchNodeKey | None
    observation: dict
    log_row: dict
    transition_injected: set[str]
    discovered_effect: dict | None
    repeat_conformance_failure: dict | None
    reported_frames: int
    diagnostic_frames: int
    repeat_count: int


@dataclass
class _Stats:
    branches: int = 0
    option_calls: int = 0
    attempted_injected: set[str] = field(default_factory=set)
    discovered: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    reported_evaluation_frames: int = 0
    diagnostic_frames: int = 0


def _sha(payload: Any) -> str:
    return hashlib.sha256(stable_encode(payload)).hexdigest()


def _semantic_interventions(boundary: dict | None) -> dict:
    """Normalize intervention semantics while excluding artifact identities.

    The complete boundary retains source and action hashes for auditability.
    Those hashes identify the experiment input, not the observed transition,
    so they must not split an otherwise equal behavioral partition.
    """
    if boundary is None:
        return {
            "external_roots": [],
            "ram_writes": [],
            "adapter_context_writes": [],
            "solution_artifacts": [],
            "behavioral_probes": [],
        }
    external_roots = []
    ram_writes = []
    adapter_writes = []
    solution_artifacts = []
    behavioral_probes = []
    for event in boundary.get("events", []):
        kind = event.get("kind")
        if kind == "snapshot_restore" and event.get("restore_kind") == "external_root":
            external_roots.append({
                "reason": event.get("reason"),
                "knowledge_tier": event.get("knowledge_tier"),
                "restore_deterministic": bool(
                    event.get("repeated_load_observable_match")
                    and event.get("fixed_suffix_match")
                ),
            })
        elif kind == "ram_write":
            ram_writes.append({
                "address": event.get("address"),
                "before": event.get("before"),
                "requested": event.get("requested"),
                "after": event.get("after"),
                "changed": bool(event.get("changed")),
                "reason": event.get("reason"),
                "knowledge_tier": event.get("knowledge_tier"),
            })
        elif kind == "adapter_context_write":
            adapter_writes.append({
                "reason": event.get("reason"),
                "knowledge_tier": event.get("knowledge_tier"),
                "changes": dict(event.get("changes") or {}),
            })
        elif kind == "solution_artifact":
            solution_artifacts.append({
                "knowledge_tier": event.get("knowledge_tier"),
                "n_actions": event.get("n_actions"),
                "chunk_frames": event.get("chunk_frames"),
            })
        elif kind == "behavioral_probe":
            behavioral_probes.append({
                "probe": event.get("probe"),
                "result": event.get("result"),
            })
    return {
        "external_roots": external_roots,
        "ram_writes": ram_writes,
        "adapter_context_writes": adapter_writes,
        "solution_artifacts": solution_artifacts,
        "behavioral_probes": behavioral_probes,
    }


def _diagnostic_frames(result: OptionResult) -> int:
    frames = int(result.info.get("evaluation_frames", 0) or 0)
    boundary = result.boundary or {}
    restore = boundary.get("restore") or {}
    frames += int(restore.get("speculative_noop_frames", 0) or 0)
    for event in boundary.get("events", []):
        frames += int(event.get("speculative_noop_frames", 0) or 0)
    return frames


def _physical_output_attestation(
    context: OptionContext,
    result: OptionResult,
) -> tuple[dict | None, int]:
    """Return finite physical-exit evidence without requiring canonical bytes."""
    if result.boundary is None or result.exit_snapshot is None:
        return None, 0
    observable = dict(result.exit_observable or {})
    if "ram_blocks" in observable:
        observable_evidence: dict | None = {
            "ram_blocks": observable["ram_blocks"],
        }
    elif observable:
        observable_evidence = {
            key: value
            for key, value in observable.items()
            if key not in ("screen_sha256", "screen_shape")
        }
    else:
        observable_evidence = None
    suffix = context._equivalence_signature(
        result.exit_snapshot,
        result.exit_adapter_context,
    )
    # Exact payload identity is the conservative fallback whenever an executor
    # supplies no forward suffix.  A coarse observable alone cannot attest the
    # complete physical exit.  mGBA supplies a suffix, so its known
    # noncanonical serialization bytes are not compared here.
    byte_fallback = None
    if suffix is None:
        byte_fallback = context.digest(
            result.exit_snapshot,
            adapter_context=result.exit_adapter_context,
        ).full_sha256
    return ({
        "adapter_context": dict(result.exit_adapter_context or {}),
        "observable": observable_evidence,
        "fixed_suffix": suffix,
        "byte_artifact_fallback_sha256": byte_fallback,
    }, int((suffix or {}).get("frames", 0) or 0))


def _outcome_signature(
    option: Option,
    result: OptionResult,
    *,
    max_tier: KnowledgeTier,
    goal: Goal,
) -> dict:
    boundary = result.boundary or {}
    effective_tier = int(
        boundary.get("effective_knowledge_tier", int(option.knowledge_tier))
    )
    tier_allowed = effective_tier <= int(max_tier)
    runner_success = bool(result.success)
    accepted = runner_success and tier_allowed
    if not runner_success:
        reason = result.info.get("reason") or "runner_reported_failure"
    elif not tier_allowed:
        reason = "runtime_knowledge_tier_exceeds_max"
    else:
        reason = None
    terminal = dict(result.info.get("terminal_invariants") or {})
    if "cursor_responsive" in result.info:
        terminal.setdefault(
            "cursor_responsive", bool(result.info["cursor_responsive"])
        )
    return {
        "accepted": accepted,
        "runner_success": runner_success,
        "termination_reason": reason,
        "successor_state": result.state.to_json() if accepted else None,
        "successor_goal": bool(goal(result.state)) if accepted else False,
        # Keep the reported legacy cost and the intended retained-duration field
        # distinct.  Some older options still mix a rolled-back probe into cost.
        "reported_cost": {
            "frames": int(result.cost.frames),
            "nodes": int(result.cost.nodes),
        },
        "retained_frames": int(
            result.info.get("retained_cost_frames", result.cost.frames) or 0
        ),
        "effective_knowledge_tier": effective_tier,
        "terminal_invariants": terminal,
        "interventions": _semantic_interventions(result.boundary),
    }


def _observation_stub(
    key: SearchNodeKey,
    option: Option,
    *,
    applicability: str,
    source_goal: bool,
) -> dict:
    return {
        "record_id": key.physical_record_id,
        "state": key.state.to_json(),
        "source_goal": bool(source_goal),
        "option": option.id,
        "physical_option": bool(option.requires_snapshot),
        "applicability": applicability,
        "repeat_count": 0,
        "repeat_conformant": None,
        "outcome_signature": None,
        "successor_state": None,
        "successor_record_id": None,
    }


def _evaluate(
    option: Option,
    node: _PathNode,
    context: OptionContext,
    *,
    max_tier: KnowledgeTier,
    goal: Goal,
    determinism_repeats: int,
) -> _Evaluation:
    repeats = max(1, int(determinism_repeats))
    results: list[OptionResult] = []
    signatures: list[dict] = []
    physical_attestations: list[dict | None] = []
    attestation_frames = 0
    for _ in range(repeats):
        result = option.execute(
            node.key.state,
            context,
            parent_record_id=node.key.physical_record_id,
        )
        results.append(result)
        signatures.append(
            _outcome_signature(
                option, result, max_tier=max_tier, goal=goal
            )
        )
        physical_attestation, frames = _physical_output_attestation(
            context, result)
        physical_attestations.append(physical_attestation)
        attestation_frames += frames
    encoded = [
        stable_encode({
            "outcome": signature,
            "physical_exit_attestation": attestation,
        })
        for signature, attestation in zip(
            signatures, physical_attestations, strict=True)
    ]
    repeat_conformant = all(
        payload == encoded[0] for payload in encoded[1:])
    first = results[0]
    first_signature = signatures[0]
    accepted = bool(first_signature["accepted"]) and repeat_conformant
    failure = None
    if not repeat_conformant:
        failure = {
            "state": node.key.state.to_json(),
            "record_id": node.key.physical_record_id,
            "option": option.id,
            "reason": "repeat_physical_or_outcome_mismatch",
            "repeat_signatures": signatures,
            "repeat_physical_exit_attestations": physical_attestations,
        }

    successor_key = None
    commit_json = None
    if accepted:
        successor_record_id = None
        if first.boundary is not None and first.exit_snapshot is not None:
            commit = context.commit_transition(
                first.state,
                first.exit_snapshot,
                parent_record_id=node.key.physical_record_id,
                option_id=option.id,
                observable=first.exit_observable,
                adapter_context=first.exit_adapter_context,
                source={
                    "kind": "option_exit",
                    "option": option.id,
                    "parent_state": node.key.state.to_json(),
                    "parent_record_id": node.key.physical_record_id,
                },
            )
            successor_record_id = commit.record_id
            first.exit_record_id = successor_record_id
            first.boundary["exit_record_id"] = successor_record_id
            first.boundary["record_commit"] = {
                "status": commit.status,
                "record_id": commit.record_id,
            }
            commit_json = first.boundary["record_commit"]
        # A symbolic edge intentionally drops physical identity.  This is what
        # prevents a later physical option from borrowing an unrelated record.
        successor_key = SearchNodeKey(first.state, successor_record_id)

    observation = {
        "record_id": node.key.physical_record_id,
        "state": node.key.state.to_json(),
        "source_goal": bool(goal(node.key.state)),
        "option": option.id,
        "physical_option": bool(option.requires_snapshot),
        "applicability": "enabled",
        "repeat_count": repeats,
        "repeat_conformant": repeat_conformant,
        "outcome_signature": first_signature,
        "repeat_signature_sha256": [_sha(signature) for signature in signatures],
        "repeat_physical_exit_attestations": physical_attestations,
        "repeat_combined_signature_sha256": [
            hashlib.sha256(payload).hexdigest() for payload in encoded
        ],
        "successor_state": (
            first.state.to_json() if accepted else None
        ),
        "successor_record_id": (
            successor_key.physical_record_id
            if successor_key is not None else None
        ),
        "record_commit": commit_json,
    }
    reason = first_signature["termination_reason"]
    if not repeat_conformant:
        reason = "repeat_physical_or_outcome_mismatch"
    row = {
        "from": node.key.state.to_json(),
        "from_record_id": node.key.physical_record_id,
        "option": option.id,
        "success": accepted,
        "runner_success": bool(first.success),
        "knowledge_tier": int(option.knowledge_tier),
        "opaque_effect": bool(option.opaque_effect),
        "execution_mode": _transition_evidence(option, first),
        "repeat_count": repeats,
        "repeat_conformant": repeat_conformant,
    }
    if first.boundary is not None:
        row["boundary"] = first.boundary
    if reason:
        row["reason"] = reason
    if accepted and successor_key is not None:
        row["to"] = successor_key.state.to_json()
        row["to_record_id"] = successor_key.physical_record_id
        row["cost"] = first.cost.to_json()

    transition_injected = set(option.injected_facts)
    for result in results:
        transition_injected.update(result.info.get("injected_facts") or ())
    discovered_effect = None
    if accepted and option.opaque_effect:
        observed = first.observed_effect or {
            "from": node.key.state.to_json(),
            "to": first.state.to_json(),
        }
        discovered_effect = {"option": option.id, **observed}
        row["observed_effect"] = discovered_effect
    return _Evaluation(
        accepted=accepted,
        result=first,
        successor_key=successor_key,
        observation=observation,
        log_row=row,
        transition_injected=transition_injected,
        discovered_effect=discovered_effect,
        repeat_conformance_failure=failure,
        reported_frames=sum(int(result.cost.frames) for result in results),
        diagnostic_frames=(
            sum(_diagnostic_frames(result) for result in results)
            + attestation_frames
        ),
        repeat_count=repeats,
    )


def _library_scope(
    library: OptionLibrary,
    *,
    max_depth: int,
    max_tier: KnowledgeTier,
    determinism_repeats: int,
) -> dict:
    options = []
    for option_id in sorted(library.options):
        option = library.options[option_id]
        options.append({
            "id": option.id,
            "kind": option.kind,
            "knowledge_tier": int(option.knowledge_tier),
            "cost": {
                "frames": int(option.cost.frames),
                "nodes": int(option.cost.nodes),
            },
            "opaque_effect": bool(option.opaque_effect),
            "requires_snapshot": bool(option.requires_snapshot),
            "known_effect": option.known_effect,
            "injected_facts": list(option.injected_facts),
            "source": option.source,
            "verification": dict(option.verification),
        })
    payload = {
        "signature_schema": SIGNATURE_SCHEMA,
        "max_depth": int(max_depth),
        "max_tier": int(max_tier),
        "option_repeat_count": int(determinism_repeats),
        "options": options,
    }
    return {**payload, "option_library_sha256": _sha(payload)}


def _relation(colors: Mapping[int, int]) -> frozenset[frozenset[int]]:
    groups: dict[int, set[int]] = defaultdict(set)
    for record_id, color in colors.items():
        groups[color].add(record_id)
    return frozenset(frozenset(group) for group in groups.values())


def refine_option_partitions(
    records: Mapping[int, MetaState],
    observations: Iterable[dict],
    *,
    option_ids: Iterable[str],
    scope: Mapping[str, Any] | None = None,
    goal_labels: Mapping[int, bool] | None = None,
) -> dict:
    """Refine physical representatives by repeat-conformant option behavior.

    Missing rows are represented as ``unobserved``.  They never authorize
    planner dominance or a complete-equivalence claim.
    """
    record_states = {int(k): v for k, v in records.items()}
    ids = sorted(record_states)
    options = tuple(sorted(set(str(option) for option in option_ids)))
    labels = {int(k): bool(v) for k, v in (goal_labels or {}).items()}
    grouped_rows: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in observations:
        record_id = row.get("record_id")
        option_id = row.get("option")
        if record_id is None or int(record_id) not in record_states:
            continue
        grouped_rows[(int(record_id), str(option_id))].append(dict(row))
    by_pair: dict[tuple[int, str], dict] = {}
    conflicts: list[dict] = []
    for pair, rows in grouped_rows.items():
        semantic_rows = [{
            "physical_option": row.get("physical_option"),
            "applicability": row.get("applicability"),
            "repeat_count": row.get("repeat_count"),
            "repeat_conformant": row.get("repeat_conformant"),
            "outcome_signature": row.get("outcome_signature"),
            "successor_state": row.get("successor_state"),
            "successor_record_id": row.get("successor_record_id"),
        } for row in rows]
        unique = {
            stable_encode(row): row for row in semantic_rows
        }
        if len(unique) == 1:
            by_pair[pair] = rows[0]
            continue
        signature_hashes = sorted(
            hashlib.sha256(encoded).hexdigest() for encoded in unique
        )
        conflict = {
            "record_id": pair[0],
            "option": pair[1],
            "conflicting_signature_sha256": signature_hashes,
            "observations": len(rows),
        }
        conflicts.append(conflict)
        by_pair[pair] = {
            "record_id": pair[0],
            "option": pair[1],
            "physical_option": any(
                bool(row.get("physical_option")) for row in rows),
            "applicability": "conflicting_observations",
            "repeat_count": 0,
            "repeat_conformant": False,
            "outcome_signature": {
                "conflicting_signature_sha256": signature_hashes,
            },
            "successor_state": None,
            "successor_record_id": None,
        }

    initial_keys = {
        record_id: {
            "meta_state": record_states[record_id].to_json(),
            "goal": labels.get(record_id, False),
        }
        for record_id in ids
    }
    encoded_initial = {
        record_id: stable_encode(key)
        for record_id, key in initial_keys.items()
    }
    unique_initial = {
        payload: color
        for color, payload in enumerate(sorted(set(encoded_initial.values())))
    }
    colors = {
        record_id: unique_initial[encoded_initial[record_id]]
        for record_id in ids
    }
    final_keys: dict[int, dict] = dict(initial_keys)
    iterations = 0

    def row_key(record_id: int, option_id: str,
                current_colors: Mapping[int, int]) -> dict:
        row = by_pair.get((record_id, option_id))
        if row is None:
            return {
                "option": option_id,
                "applicability": "unobserved",
                "repeat_conformant": None,
                "outcome": None,
                "successor_block": None,
                "successor_symbolic_state": None,
            }
        successor = row.get("successor_record_id")
        successor_state = row.get("successor_state")
        return {
            "option": option_id,
            "applicability": row.get("applicability", "unobserved"),
            "repeat_conformant": row.get("repeat_conformant"),
            "outcome": row.get("outcome_signature"),
            "successor_block": (
                current_colors.get(int(successor))
                if successor is not None else None
            ),
            "successor_symbolic_state": (
                successor_state if successor is None else None
            ),
        }

    for _ in range(max(1, len(ids) + 1)):
        iterations += 1
        keys = {
            record_id: {
                **initial_keys[record_id],
                "options": [
                    row_key(record_id, option_id, colors)
                    for option_id in options
                ],
            }
            for record_id in ids
        }
        encoded = {
            record_id: stable_encode(key)
            for record_id, key in keys.items()
        }
        unique = {
            payload: color
            for color, payload in enumerate(sorted(set(encoded.values())))
        }
        new_colors = {
            record_id: unique[encoded[record_id]]
            for record_id in ids
        }
        final_keys = keys
        if _relation(new_colors) == _relation(colors):
            colors = new_colors
            break
        colors = new_colors

    groups: dict[int, list[int]] = defaultdict(list)
    for record_id, color in colors.items():
        groups[color].append(record_id)
    block_ids: dict[int, str] = {}
    for color, members in groups.items():
        witness_key = final_keys[min(members)]
        block_ids[color] = _sha({
            "signature_schema": SIGNATURE_SCHEMA,
            "scope": dict(scope or {}),
            "behavior": witness_key,
        })

    complete_rows = True
    conformant_rows = True
    repeated_rows = True
    minimum_repeat_count: int | None = None
    closed = True
    for record_id in ids:
        for option_id in options:
            row = by_pair.get((record_id, option_id))
            if row is None or row.get("applicability") == "unobserved":
                complete_rows = False
                continue
            if (
                row.get("applicability") == "enabled"
                and row.get("repeat_conformant") is not True
            ):
                conformant_rows = False
            if row.get("applicability") == "enabled":
                repeat_count = int(row.get("repeat_count", 0) or 0)
                minimum_repeat_count = (
                    repeat_count
                    if minimum_repeat_count is None
                    else min(minimum_repeat_count, repeat_count)
                )
                if repeat_count < 2:
                    repeated_rows = False
            if (
                row.get("applicability") == "enabled"
                and row.get("outcome_signature", {}).get("accepted")
            ):
                successor_missing = (
                    row.get("successor_record_id") not in record_states
                )
                if row.get("physical_option") and successor_missing:
                    closed = False
                if (
                    not row.get("physical_option")
                    and successor_missing
                    and not row.get(
                        "outcome_signature", {}
                    ).get("successor_goal")
                ):
                    closed = False

    def local_row(record_id: int, option_id: str) -> dict:
        row = row_key(record_id, option_id, colors)
        row.pop("successor_block", None)
        return row

    def witness(left: int, right: int,
                seen: frozenset[tuple[int, int]] = frozenset()) -> list[str]:
        pair = (min(left, right), max(left, right))
        if pair in seen:
            return []
        next_seen = seen | {pair}
        for option_id in options:
            if local_row(left, option_id) != local_row(right, option_id):
                return [option_id]
            left_row = by_pair.get((left, option_id), {})
            right_row = by_pair.get((right, option_id), {})
            left_successor = left_row.get("successor_record_id")
            right_successor = right_row.get("successor_record_id")
            if (
                left_successor is not None
                and right_successor is not None
                and colors.get(int(left_successor))
                != colors.get(int(right_successor))
            ):
                child = witness(
                    int(left_successor), int(right_successor), next_seen
                )
                return [option_id, *child]
            if row_key(left, option_id, colors) != row_key(
                right, option_id, colors
            ):
                return [option_id]
        return []

    split_witnesses = []
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            if initial_keys[left] != initial_keys[right]:
                continue
            if colors[left] == colors[right]:
                continue
            split_witnesses.append({
                "left_record_id": left,
                "right_record_id": right,
                "option_suffix": witness(left, right),
            })

    classes = []
    for color, members in sorted(
        groups.items(), key=lambda item: block_ids[item[0]]
    ):
        member_complete = all(
            by_pair.get((record_id, option_id), {}).get("applicability")
            not in (None, "unobserved")
            for record_id in members
            for option_id in options
        )
        classes.append({
            "block_id": block_ids[color],
            "meta_state": record_states[members[0]].to_json(),
            "member_record_ids": sorted(members),
            "complete": member_complete,
        })
    equivalence_verified = bool(
        ids
        and options
        and complete_rows
        and conformant_rows
        and repeated_rows
        and closed
        and not conflicts
    )
    return {
        "signature_schema": SIGNATURE_SCHEMA,
        "scope": dict(scope or {}),
        "representatives": len(ids),
        "iterations": iterations,
        "converged": True,
        "complete_option_table": complete_rows,
        "all_repeat_attestations_match": conformant_rows,
        "each_enabled_option_repeated": repeated_rows,
        "minimum_enabled_repeat_count": minimum_repeat_count,
        "closed_under_modeled_successors": closed,
        "conflicting_observations": sorted(
            conflicts,
            key=lambda row: (row["record_id"], row["option"]),
        ),
        "empirical_conformance_gate": equivalence_verified,
        "claim": (
            "closed fixed point of the encountered finite observation table; "
            "repeated normalized outcomes and finite physical-exit attestations "
            "matched, but this is not proof of global bisimulation"
            if equivalence_verified
            else (
                "finite-library observational partition only; missing, "
                "conflicting, insufficiently repeated, nonconformant, or "
                "non-closed rows prohibit a bisimulation claim"
            )
        ),
        "record_to_block": {
            str(record_id): block_ids[colors[record_id]]
            for record_id in ids
        },
        "classes": classes,
        "distinguishing_option_suffixes": split_witnesses,
    }


def _representative_registry(context: OptionContext) -> list[dict]:
    registry = []
    for record_id in sorted(context.physical_records):
        record = context.physical_records[record_id]
        registry.append({
            **record.to_json(),
            "arrivals": list(context.record_arrivals.get(record_id, ())),
        })
    return registry


def _append_observation(
    stats: _Stats,
    key: SearchNodeKey,
    option: Option,
    *,
    applicability: str,
    goal: Goal,
) -> None:
    if key.physical_record_id is not None:
        stats.observations.append(
            _observation_stub(
                key,
                option,
                applicability=applicability,
                source_goal=goal(key.state),
            )
        )


def _record_evaluation(stats: _Stats, evaluation: _Evaluation) -> None:
    stats.option_calls += evaluation.repeat_count
    stats.attempted_injected.update(evaluation.transition_injected)
    stats.reported_evaluation_frames += evaluation.reported_frames
    stats.diagnostic_frames += evaluation.diagnostic_frames
    stats.log.append(evaluation.log_row)
    if evaluation.observation["record_id"] is not None:
        stats.observations.append(evaluation.observation)
    if (
        evaluation.discovered_effect is not None
        and evaluation.discovered_effect not in stats.discovered
    ):
        stats.discovered.append(evaluation.discovered_effect)
    if evaluation.repeat_conformance_failure is not None:
        stats.failures.append(evaluation.repeat_conformance_failure)


def _result(
    node: _PathNode | None,
    *,
    context: OptionContext,
    stats: _Stats,
    visited_keys: Iterable[SearchNodeKey],
    library: OptionLibrary,
    goal: Goal,
    max_depth: int,
    max_tier: KnowledgeTier,
    determinism_repeats: int,
) -> MetaSearchResult:
    keys = set(visited_keys)
    scope = _library_scope(
        library,
        max_depth=max_depth,
        max_tier=max_tier,
        determinism_repeats=determinism_repeats,
    )
    record_states = {
        record_id: record.state
        for record_id, record in context.physical_records.items()
        if record.state is not None
    }
    refinement = refine_option_partitions(
        record_states,
        stats.observations,
        option_ids=library.options,
        scope=scope,
        goal_labels={
            record_id: goal(state)
            for record_id, state in record_states.items()
        },
    )
    return MetaSearchResult(
        found=node is not None,
        path=list(node.path) if node is not None else [],
        states=list(node.states) if node is not None else [],
        total_cost=node.cost if node is not None else OptionCost(),
        branches=stats.branches,
        option_calls=stats.option_calls,
        injected_facts=(
            tuple(sorted(node.injected)) if node is not None else ()
        ),
        attempted_injected_facts=tuple(sorted(stats.attempted_injected)),
        discovered_effects=stats.discovered,
        boundaries=list(node.boundaries) if node is not None else [],
        path_evidence=list(node.evidence) if node is not None else [],
        alias_collisions=list(context.alias_collisions),
        visited=len({key.state for key in keys}),
        log=stats.log,
        physical_record_ids=(
            list(node.record_ids) if node is not None else []
        ),
        physical_visited=sum(
            key.physical_record_id is not None for key in keys
        ),
        search_nodes_visited=len(keys),
        representative_registry=_representative_registry(context),
        transition_observations=stats.observations,
        partition_refinement=refinement,
        repeat_conformance_failures=stats.failures,
        evaluation_work={
            "option_execute_calls": stats.option_calls,
            "reported_option_frames_all_repeats": (
                stats.reported_evaluation_frames
            ),
            "rolled_back_or_attestation_frames_observed": (
                stats.diagnostic_frames
            ),
            "note": (
                "reported option frames are distinct from rolled-back probes "
                "and restore attestations; older option costs may still include "
                "small rolled-back probes"
            ),
        },
    )


def _initial_nodes(
    start: MetaState,
    context: OptionContext,
) -> list[_PathNode]:
    record_ids = context.record_ids_for(start)
    if not record_ids:
        record_ids = (None,)
    return [
        _PathNode(
            key=SearchNodeKey(start, record_id),
            path=[],
            states=[start],
            record_ids=[record_id],
            cost=OptionCost(),
            injected=frozenset(),
            boundaries=[],
            evidence=[],
        )
        for record_id in record_ids
    ]


def _successor_node(
    parent: _PathNode,
    option: Option,
    evaluation: _Evaluation,
) -> _PathNode:
    assert evaluation.successor_key is not None
    result = evaluation.result
    return _PathNode(
        key=evaluation.successor_key,
        path=parent.path + [option.id],
        states=parent.states + [evaluation.successor_key.state],
        record_ids=parent.record_ids + [
            evaluation.successor_key.physical_record_id
        ],
        cost=parent.cost + result.cost,
        injected=frozenset(
            set(parent.injected) | evaluation.transition_injected
        ),
        boundaries=parent.boundaries + (
            [result.boundary] if result.boundary is not None else []
        ),
        evidence=parent.evidence + [_transition_evidence(option, result)],
    )


def search_physical_options(
    library: OptionLibrary,
    start: MetaState,
    goal: Goal,
    *,
    max_depth: int = 16,
    max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT,
    context: OptionContext | None = None,
    determinism_repeats: int = 1,
    exhaustive: bool = False,
) -> MetaSearchResult:
    """Breadth-first search keyed by ``(MetaState, physical_record_id)``."""
    context = context or OptionContext(alias_policy="multi")
    queue = deque(_initial_nodes(start, context))
    visited = {node.key for node in queue}
    stats = _Stats()
    best_goal: _PathNode | None = None

    while queue:
        node = queue.popleft()
        if goal(node.key.state):
            if best_goal is None:
                best_goal = node
            if not exhaustive:
                break
            continue
        if len(node.path) >= max_depth:
            continue
        for option_id in sorted(library.options):
            option = library.options[option_id]
            if not option.precondition(node.key.state):
                _append_observation(
                    stats, node.key, option,
                    applicability="not_enabled", goal=goal,
                )
                continue
            if option.knowledge_tier > max_tier:
                _append_observation(
                    stats, node.key, option,
                    applicability="tier_blocked", goal=goal,
                )
                continue
            stats.branches += 1
            evaluation = _evaluate(
                option,
                node,
                context,
                max_tier=max_tier,
                goal=goal,
                determinism_repeats=determinism_repeats,
            )
            _record_evaluation(stats, evaluation)
            if not evaluation.accepted or evaluation.successor_key is None:
                continue
            successor = _successor_node(node, option, evaluation)
            if successor.key in visited:
                continue
            visited.add(successor.key)
            if goal(successor.key.state) and not exhaustive:
                best_goal = successor
                queue.clear()
                break
            queue.append(successor)

    return _result(
        best_goal,
        context=context,
        stats=stats,
        visited_keys=visited,
        library=library,
        goal=goal,
        max_depth=max_depth,
        max_tier=max_tier,
        determinism_repeats=determinism_repeats,
    )


def search_physical_options_uniform_cost(
    library: OptionLibrary,
    start: MetaState,
    goal: Goal,
    *,
    cost_of: CostFunction = lambda cost: float(cost.frames),
    max_depth: int = 32,
    max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT,
    context: OptionContext | None = None,
    determinism_repeats: int = 1,
    exhaustive: bool = False,
) -> MetaSearchResult:
    """Bounded Dijkstra search with per-representative cost/depth dominance."""
    context = context or OptionContext(alias_policy="multi")
    counter = itertools.count()
    roots = _initial_nodes(start, context)
    frontier: list[tuple[float, int, tuple[str, ...], int, _PathNode]] = []
    labels: dict[SearchNodeKey, list[tuple[float, int]]] = {}

    def admit(key: SearchNodeKey, cost: float, depth: int) -> bool:
        current = labels.setdefault(key, [])
        if any(
            old_cost <= cost and old_depth <= depth
            for old_cost, old_depth in current
        ):
            return False
        current[:] = [
            (old_cost, old_depth)
            for old_cost, old_depth in current
            if not (cost <= old_cost and depth <= old_depth)
        ]
        current.append((cost, depth))
        return True

    for root in roots:
        admit(root.key, 0.0, 0)
        heapq.heappush(
            frontier,
            (0.0, 0, (), next(counter), root),
        )
    stats = _Stats()
    best_goal: _PathNode | None = None
    best_goal_cost = float("inf")
    expansion_cache: dict[
        SearchNodeKey, list[tuple[Option, _Evaluation]]
    ] = {}

    while frontier:
        gcost, depth, _, _, node = heapq.heappop(frontier)
        if (gcost, depth) not in labels.get(node.key, ()):
            continue
        if goal(node.key.state):
            if (
                gcost < best_goal_cost
                or (
                    gcost == best_goal_cost
                    and (
                        best_goal is None
                        or tuple(node.path) < tuple(best_goal.path)
                    )
                )
            ):
                best_goal = node
                best_goal_cost = gcost
            if not exhaustive:
                break
            continue
        if len(node.path) >= max_depth:
            continue
        cached = expansion_cache.get(node.key)
        if cached is None:
            cached = []
            for option_id in sorted(library.options):
                option = library.options[option_id]
                if not option.precondition(node.key.state):
                    _append_observation(
                        stats, node.key, option,
                        applicability="not_enabled", goal=goal,
                    )
                    continue
                if option.knowledge_tier > max_tier:
                    _append_observation(
                        stats, node.key, option,
                        applicability="tier_blocked", goal=goal,
                    )
                    continue
                stats.branches += 1
                evaluation = _evaluate(
                    option,
                    node,
                    context,
                    max_tier=max_tier,
                    goal=goal,
                    determinism_repeats=determinism_repeats,
                )
                _record_evaluation(stats, evaluation)
                cached.append((option, evaluation))
            expansion_cache[node.key] = cached
        for option, evaluation in cached:
            if not evaluation.accepted or evaluation.successor_key is None:
                continue
            successor = _successor_node(node, option, evaluation)
            step_cost = float(cost_of(evaluation.result.cost))
            if step_cost < 0:
                raise ValueError(
                    "uniform-cost physical search requires nonnegative costs"
                )
            next_cost = gcost + step_cost
            next_depth = len(successor.path)
            if not admit(successor.key, next_cost, next_depth):
                continue
            heapq.heappush(
                frontier,
                (
                    next_cost,
                    next_depth,
                    tuple(successor.path),
                    next(counter),
                    successor,
                ),
            )

    return _result(
        best_goal,
        context=context,
        stats=stats,
        visited_keys=labels,
        library=library,
        goal=goal,
        max_depth=max_depth,
        max_tier=max_tier,
        determinism_repeats=determinism_repeats,
    )
