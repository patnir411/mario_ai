"""Build, validate, and write run-result artifacts against the locked schema.

Centralizes the artifact contract so every milestone emits identical, schema-valid
result.json files and Claude can read them the same way each time (see DESIGN.md, plan).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from mario.io import (ROOT, env_fingerprint, git_rev, run_dir, utc_now_iso,
                      write_json_atomic)

SCHEMA_DIR = ROOT / "schemas"


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text())
    return Draft202012Validator(schema)


def validate_result(result: dict) -> None:
    """Raise jsonschema.ValidationError if `result` violates result.schema.json."""
    _validator("result.schema.json").validate(result)


def validate_eval(ev: dict) -> None:
    """Raise jsonschema.ValidationError if `ev` violates eval.schema.json."""
    _validator("eval.schema.json").validate(ev)


def validate_game_result(g: dict) -> None:
    """Raise jsonschema.ValidationError if `g` violates game_result.schema.json."""
    _validator("game_result.schema.json").validate(g)


def framerule_time(frames: int) -> int:
    """SMB level time in 21-frame framerules (the meaningful speed unit)."""
    return math.ceil(frames / 21)


def build_result(*, run_id: str, kind: str, milestone: str, world: int, stage: int,
                 seed: int, outcome: dict, search: dict | None = None,
                 artifacts: dict | None = None, passed: bool, pass_reason: str) -> dict:
    result = {
        "run_id": run_id,
        "schema_version": 1,
        "kind": kind,
        "milestone": milestone,
        "git_rev": git_rev(),
        "generated_at": utc_now_iso(),
        "env_fingerprint": env_fingerprint(),
        "level": {"world": world, "stage": stage},
        "seed": seed,
        "outcome": outcome,
        "artifacts": artifacts or {},
        "pass": bool(passed),
        "pass_reason": pass_reason,
    }
    if search is not None:
        result["search"] = search
    validate_result(result)
    return result


def write_result(result: dict) -> Path:
    validate_result(result)
    d = run_dir(result["run_id"])
    return write_json_atomic(d / "result.json", result)
