"""Replay-based integrity checks for committed SMB1 solution manifests.

A search-time ``solved`` flag is a claim, not proof.  The canonical acceptance
gate is deterministic replay from seed 0 reaching the flag with the manifest's
declared action path and chunk length.

This module keeps the gate reusable by dataset builders, batch solvers, tests,
and the command-line inventory report.  It deliberately does not rewrite files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Callable

from mario.actions import SMB1_N_ACTIONS

_STOCK_NAME = re.compile(r"^([1-8])-([1-4])\.json$")


@dataclass(frozen=True)
class SolutionVerification:
    file: str
    level: str
    claimed_solved: bool
    replay_verified: bool
    status: str
    reason: str
    chunk_frames: int | None = None
    path_length: int = 0

    def to_json(self) -> dict:
        return asdict(self)


def stock_level_from_path(path: str | Path) -> tuple[int, int]:
    p = Path(path)
    match = _STOCK_NAME.fullmatch(p.name)
    if not match:
        raise ValueError(f"not a stock SMB1 solution filename: {p.name}")
    return int(match.group(1)), int(match.group(2))


def replay_verify_stock_path(
        world: int, stage: int, actions: list[int], chunk_frames: int, *,
        seed: int = 0, replay_fn: Callable | None = None) -> tuple[bool, str]:
    """Independently replay a search candidate before it becomes canonical."""
    if not actions:
        return False, "empty_path"
    if not all(isinstance(action, int) and not isinstance(action, bool)
               for action in actions):
        return False, "path_must_be_integer_list"
    if any(action < 0 or action >= SMB1_N_ACTIONS for action in actions):
        return False, f"action_index_out_of_range_0_{SMB1_N_ACTIONS - 1}"
    if not isinstance(chunk_frames, int) or isinstance(chunk_frames, bool) or chunk_frames <= 0:
        return False, "chunk_frames_must_be_positive"
    if replay_fn is None:
        from mario.render import replay
        replay_fn = replay
    try:
        records = replay_fn(world, stage, seed, actions, chunk_frames)
    except Exception as exc:
        return False, f"replay_exception:{type(exc).__name__}"
    reached_flag = any(bool(record.get("flag")) for record in records)
    died = bool(records and records[-1].get("died"))
    if reached_flag and not died:
        return True, f"seed{seed}_flag"
    return False, (f"seed{seed}_died_without_flag" if died else f"seed{seed}_no_flag")


def verify_stock_solution(
        path: str | Path, *, seed: int = 0,
        replay_fn: Callable | None = None) -> SolutionVerification:
    """Validate one stock manifest and replay positive claims.

    ``replay_fn`` is injectable for unit tests.  Its production signature is
    ``(world, stage, seed, path, chunk_frames) -> list[record]`` as provided by
    :func:`mario.render.replay`.
    """
    p = Path(path)
    try:
        world, stage = stock_level_from_path(p)
    except ValueError as exc:
        return SolutionVerification(
            str(p), p.stem, False, False, "invalid", str(exc))
    level = f"{world}-{stage}"

    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return SolutionVerification(
            str(p), level, False, False, "missing", "manifest_missing")
    except (json.JSONDecodeError, OSError) as exc:
        return SolutionVerification(
            str(p), level, False, False, "invalid",
            f"manifest_unreadable:{type(exc).__name__}")

    raw_solved = data.get("solved")
    if not isinstance(raw_solved, bool):
        return SolutionVerification(
            str(p), level, False, False, "invalid",
            "solved_must_be_boolean")
    claimed = raw_solved
    raw_path = data.get("path")
    raw_cf = data.get("chunk_frames", 8)
    if not isinstance(raw_path, list) or not all(
            isinstance(action, int) and not isinstance(action, bool)
            for action in raw_path):
        return SolutionVerification(
            str(p), level, claimed, False, "invalid",
            "path_must_be_integer_list")
    if any(action < 0 or action >= SMB1_N_ACTIONS for action in raw_path):
        return SolutionVerification(
            str(p), level, claimed, False, "invalid",
            f"action_index_out_of_range_0_{SMB1_N_ACTIONS - 1}",
            path_length=len(raw_path))
    if (not isinstance(raw_cf, int) or isinstance(raw_cf, bool)
            or raw_cf <= 0):
        return SolutionVerification(
            str(p), level, claimed, False, "invalid",
            "chunk_frames_must_be_positive", None, len(raw_path))
    chunk_frames = raw_cf
    if not claimed:
        return SolutionVerification(
            str(p), level, False, False, "unsolved",
            str(data.get("invalid_reason") or "manifest_claims_unsolved"),
            chunk_frames, len(raw_path))
    if not raw_path:
        return SolutionVerification(
            str(p), level, True, False, "invalid",
            "solved_manifest_has_empty_path", chunk_frames, 0)

    replay_verified, reason = replay_verify_stock_path(
        world, stage, raw_path, chunk_frames, seed=seed, replay_fn=replay_fn)
    if replay_verified:
        return SolutionVerification(
            str(p), level, True, True, "verified", reason,
            chunk_frames, len(raw_path))
    return SolutionVerification(
        str(p), level, True, False, "replay_failed", reason,
        chunk_frames, len(raw_path))


def stock_solution_paths(root: str | Path) -> list[Path]:
    base = Path(root)
    return [base / f"{world}-{stage}.json"
            for world in range(1, 9) for stage in range(1, 5)]


def verify_stock_solution_set(
        root: str | Path, *, seed: int = 0,
        replay_fn: Callable | None = None) -> list[SolutionVerification]:
    return [
        verify_stock_solution(path, seed=seed, replay_fn=replay_fn)
        for path in stock_solution_paths(root)
    ]
