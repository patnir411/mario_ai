"""Shared artifact I/O: atomic JSON writes, env fingerprint, git rev, run ids.

Every script that emits ground-truth artifacts uses these so the artifact contract
(see schemas/ and DESIGN.md) is consistent and crash-safe.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

_FINGERPRINT_PKGS = ["gym-super-mario-bros", "nes-py", "gymnasium", "numpy", "torch"]


def write_json_atomic(path: str | Path, obj) -> Path:
    """Write JSON via temp file + atomic rename so a crash never leaves a half-file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)
    return path


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def env_fingerprint() -> dict:
    import platform
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        **{p: _pkg_version(p) for p in _FINGERPRINT_PKGS},
    }


def git_rev() -> str:
    """Short git rev, or a content fingerprint of tracked source if not a git repo yet."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=ROOT, stderr=subprocess.DEVNULL,
        )
        return rev + ("-dirty" if dirty else "")
    except Exception:
        src = sorted(ROOT.glob("mario/*.py")) + sorted(ROOT.glob("scripts/*.py"))
        h = hashlib.sha256()
        for f in src:
            h.update(f.read_bytes())
        return "nogit-" + h.hexdigest()[:12]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(tag: str) -> str:
    """Sortable, greppable run id: YYYYMMDD-HHMMSS-<tag>."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{tag}"


def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
