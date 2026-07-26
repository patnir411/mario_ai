"""Verify continuous post-1-2 -> World-1 Fortress entry from one exact root.

This is an entry-only falsification gate.  It restores the declared 1-2 level
root, replays the verified 1-2 solution, settles to a responsive overworld,
walks the explicit non-greedy DOWN,DOWN,LEFT route to (96,96), and enters the
fortress.  It never promotes or rewrites a cache and stops at the unpowered
fortress spawn; the current whistle route begins from a different P-Wing root.

Example:

    MARIO_AI_SMA4_ROM=roms/...gba ./venv/bin/python \
      scripts/diagnose_sma4_fortress_entry.py \
      --forbid-memory-writes --no-promote \
      --out runs/20260726-sma4-live-fortress-entry
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PSPEED_ACTIONS  # noqa: E402
from mario.overworld_search import (  # noqa: E402
    remap_solution_path,
    settle_to_map,
)
from mario.provenance import file_sha256, snapshot_digest  # noqa: E402


EXPECTED_ROM_SHA1 = "532f3307021637474b6dd37da059ca360f612337"
EXPECTED_TARGET = (96, 96)
EXPECTED_ROUTE = ("DOWN", "DOWN", "LEFT")
FORTRESS_ENTRY_EXPECTED = {
    "mode": "level",
    "cursor": [96, 96],
    "x_pos": 24,
    "y_pos": 112,
    "time": 196608,
    "powerup": 0,
    "pspeed": 0,
}


class _RuntimeAudit:
    """Measure retained/rewound work and forbid direct Python RAM writes."""

    def __init__(self, core: SMA4Adapter):
        self.core = core

    def __enter__(self):
        self._original_step = self.core._step_buttons
        self._original_snapshot = self.core.snapshot
        self._original_restore = self.core.restore
        self._memory_type = type(self.core.env.data.memory)
        self._original_assign = self._memory_type.assign
        self._active = False

        def step(buttons=()):
            if self._active:
                self.total_emulator_steps += 1
                self.retained_frames += 1
            return self._original_step(buttons)

        def snapshot():
            snap = self._original_snapshot()
            if self._active:
                self.snapshot_calls += 1
                self._snapshot_frames[id(snap)] = self.retained_frames
            return snap

        def restore(snap):
            before = self.retained_frames if self._active else 0
            target = (
                self._snapshot_frames.get(id(snap))
                if self._active else None
            )
            if self._active:
                self.restore_calls += 1
            out = self._original_restore(snap)
            if self._active:
                if target is None:
                    self.unknown_restore_targets += 1
                else:
                    if target < before:
                        self.probe_frames_rolled_back += before - target
                    self.retained_frames = target
            return out

        audit = self

        def forbidden_assign(memory, address, dtype, value):
            if audit._active:
                audit.memory_assign_calls.append({
                    "address": f"0x{int(address):08X}",
                    "dtype": str(dtype),
                    "value": int(value),
                })
                raise AssertionError(
                    f"forbidden direct memory.assign at 0x{int(address):08X}"
                )
            return audit._original_assign(memory, address, dtype, value)

        self.core._step_buttons = step
        self.core.snapshot = snapshot
        self.core.restore = restore
        self._memory_type.assign = forbidden_assign
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._active = False
        self.core._step_buttons = self._original_step
        self.core.snapshot = self._original_snapshot
        self.core.restore = self._original_restore
        self._memory_type.assign = self._original_assign

    def begin_run(self, root_snapshot: Any) -> None:
        self.total_emulator_steps = 0
        self.retained_frames = 0
        self.probe_frames_rolled_back = 0
        self.restore_calls = 0
        self.snapshot_calls = 0
        self.unknown_restore_targets = 0
        self.memory_assign_calls: list[dict] = []
        self._snapshot_frames = {id(root_snapshot): 0}
        self._active = True

    def metrics(self) -> dict:
        return {
            "retained_frames": int(self.retained_frames),
            "probe_frames_rolled_back": int(self.probe_frames_rolled_back),
            "total_emulator_steps": int(self.total_emulator_steps),
            "restore_calls": int(self.restore_calls),
            "snapshot_calls": int(self.snapshot_calls),
            "unknown_restore_targets": int(self.unknown_restore_targets),
            "direct_memory_assign_calls": len(self.memory_assign_calls),
            "memory_assign_events": list(self.memory_assign_calls),
            "accounting_identity_holds": (
                self.total_emulator_steps
                == self.retained_frames + self.probe_frames_rolled_back
            ),
        }


def _load_snapshot(path: Path) -> Any:
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def _context(core: SMA4Adapter) -> dict:
    return {
        "game_id": core.game_id,
        "level_id": core.level_id,
        "world": core.world,
        "stage": core.stage,
        "start_lives": core._start_lives,
        "action_names": list(core.action_names),
    }


def _digest(core: SMA4Adapter, snap=None) -> dict:
    snap = core.snapshot() if snap is None else snap
    return snapshot_digest(
        snap,
        adapter_context=_context(core),
        backend="stable-retro/mgba",
        rom_sha1=core.rom_sha1,
    ).to_json()


def _ram_block_bytes(core: SMA4Adapter) -> dict[int, bytes]:
    return {
        int(base): bytes(block)
        for base, block in core.env.data.memory.blocks.items()
    }


def _ram_blocks(core: SMA4Adapter) -> dict:
    blocks = {}
    for base, raw in sorted(_ram_block_bytes(core).items()):
        blocks[f"0x{int(base):08X}"] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return blocks


def _ram_diff(left: dict[int, bytes], right: dict[int, bytes]) -> dict:
    out = {}
    for base in sorted(set(left) | set(right)):
        a = left.get(base, b"")
        b = right.get(base, b"")
        n = max(len(a), len(b))
        changed = [
            index for index in range(n)
            if (a[index] if index < len(a) else None)
            != (b[index] if index < len(b) else None)
        ]
        out[f"0x{base:08X}"] = {
            "left_bytes": len(a),
            "right_bytes": len(b),
            "changed_bytes": len(changed),
            "first_changed_addresses": [
                f"0x{base + index:08X}" for index in changed[:32]
            ],
            "first_changed_values": [
                {
                    "address": f"0x{base + index:08X}",
                    "live": a[index] if index < len(a) else None,
                    "reference": b[index] if index < len(b) else None,
                }
                for index in changed[:32]
            ],
            "left_sha256": hashlib.sha256(a).hexdigest(),
            "right_sha256": hashlib.sha256(b).hexdigest(),
        }
    return out


def _boundary(core: SMA4Adapter, audit: _RuntimeAudit, label: str) -> dict:
    info = dict(core.last_info)
    panel_slots_raw = int(info.pop("cleared", 0))
    info["panel_slots_raw"] = panel_slots_raw
    return {
        "label": label,
        "runtime_accounting": audit.metrics(),
        "digest": _digest(core),
        "ram_blocks": _ram_blocks(core),
        "info": info,
        "cursor": list(info.get("cursor") or (0, 0)),
        # Adapter `info["cleared"]` currently mirrors 0x03002C52, which is
        # panel-slot data rather than a validated clear bitmap.
        "panel_slots_raw": panel_slots_raw,
        "powerup": int(info.get("powerup", 0)),
        "pspeed": int(info.get("pspeed", 0)),
    }


def _parse_pair(raw: str) -> tuple[int, int]:
    left, right = raw.split(",", 1)
    return int(left), int(right)


def _run_once(
    core: SMA4Adapter,
    *,
    root_snapshot: Any,
    solution: dict,
    route: tuple[str, ...],
    target: tuple[int, int],
    max_settle_frames: int,
    audit: _RuntimeAudit,
) -> dict:
    # Adapter metadata affects normalization and therefore belongs to the exact
    # wrapper state.  Apply it before restoring the 1-2 root.
    core.world = 1
    core.stage = 2
    core.level_id = "1-2"
    audit.begin_run(root_snapshot)
    core.restore(root_snapshot)
    boundaries = [_boundary(core, audit, "restored_1_2_root")]

    path, mapping = remap_solution_path(solution, list(core.action_names))
    if not mapping.get("source_action_names_valid"):
        raise AssertionError(f"unsafe action-index fallback: {mapping}")
    chunk_frames = int(solution.get("chunk_frames", 0))
    if chunk_frames != 1:
        raise AssertionError(f"expected chunk_frames=1, got {chunk_frames}")

    success_at = None
    for index, action in enumerate(path, start=1):
        info, done = core.run_chunk(action, chunk_frames)
        if core.is_success(info):
            success_at = index
            break
        if core.is_death(info, done):
            raise AssertionError(
                f"1-2 replay died before completion at action {index}"
            )
    if success_at != len(path):
        raise AssertionError(
            f"1-2 success boundary {success_at}, expected {len(path)}"
        )
    boundaries.append(_boundary(core, audit, "1_2_success"))

    settled = settle_to_map(core, max_frames=max_settle_frames)
    if not settled:
        raise AssertionError("1-2 replay did not settle to a responsive map")
    post_cursor = tuple(core.last_info.get("cursor") or ())
    if post_cursor != (128, 32):
        raise AssertionError(f"post-1-2 cursor {post_cursor}, expected (128, 32)")
    boundaries.append(_boundary(core, audit, "responsive_post_1_2_map"))

    route_edges = []
    expected_edges = {
        ("DOWN", (128, 32)): (128, 64),
        ("DOWN", (128, 64)): (128, 96),
        ("LEFT", (128, 96)): (96, 96),
    }
    for direction in route:
        before = tuple(core.last_info.get("cursor") or ())
        for _ in range(16):
            core._step_buttons((direction,))
        for _ in range(12):
            core._step_buttons(())
        after = tuple(core.last_info.get("cursor") or ())
        expected = expected_edges.get((direction, before))
        if expected is None or after != expected:
            raise AssertionError(
                f"map edge {direction} from {before} reached {after}; "
                f"expected {expected}"
            )
        route_edges.append({
            "direction": direction,
            "before": list(before),
            "after": list(after),
            "hold_frames": 16,
            "settle_frames": 12,
        })
        boundaries.append(_boundary(
            core, audit, f"map_{direction.lower()}_{after[0]}_{after[1]}"))
    if tuple(core.last_info.get("cursor") or ()) != target:
        raise AssertionError(
            f"route ended at {core.last_info.get('cursor')}, expected {target}"
        )

    # Relabel wrapper metadata before the entry transition.  No RAM is touched.
    core.stage = 0
    core.level_id = "1-fortress"
    entry_info = core.enter_level(max_presses=60, settle=360, probe=20)
    boundaries.append(_boundary(core, audit, "unpowered_fortress_spawn"))
    observed = {
        "mode": entry_info.get("mode"),
        "cursor": list(entry_info.get("cursor") or ()),
        "x_pos": int(entry_info.get("x_pos", -1)),
        "y_pos": int(entry_info.get("y_pos", -1)),
        "time": int(entry_info.get("time", -1)),
        "powerup": int(entry_info.get("powerup", -1)),
        "pspeed": int(entry_info.get("pspeed", -1)),
    }
    invariant_checks = {
        key: observed[key] == expected
        for key, expected in FORTRESS_ENTRY_EXPECTED.items()
    }
    if not all(invariant_checks.values()):
        raise AssertionError(
            f"fortress entry invariant failure: observed={observed}"
        )

    return {
        "success": True,
        "solution_actions": len(path),
        "success_at": success_at,
        "chunk_frames": chunk_frames,
        "action_mapping": mapping,
        "settled_to_map": settled,
        "route": route_edges,
        "target": list(target),
        "entry_observed": observed,
        "entry_invariant_checks": invariant_checks,
        "boundaries": boundaries,
        "runtime_accounting": audit.metrics(),
        "direct_memory_assign_calls": len(audit.memory_assign_calls),
        "external_snapshot_restores": ["declared_1_2_root"],
        "stopped_before_acquire_whistle_fortress": True,
    }


def _repeat_comparison(runs: list[dict]) -> dict:
    labels = [row["label"] for row in runs[0]["boundaries"]]
    comparisons = {}
    for label in labels:
        rows = [
            next(row for row in run["boundaries"] if row["label"] == label)
            for run in runs
        ]
        full = [row["digest"]["full_sha256"] for row in rows]
        emulator = [row["digest"]["emulator_sha256"] for row in rows]
        ram = [
            {base: value["sha256"]
             for base, value in row["ram_blocks"].items()}
            for row in rows
        ]
        comparisons[label] = {
            "full_snapshot_sha256": full,
            "emulator_sha256": emulator,
            "ram_sha256": ram,
            "full_match": len(set(full)) == 1,
            "emulator_match": len(set(emulator)) == 1,
            "ram_match": all(item == ram[0] for item in ram[1:]),
        }
    return {
        "boundaries": comparisons,
        "all_exact": all(
            row["full_match"] and row["emulator_match"] and row["ram_match"]
            for row in comparisons.values()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("runs/sma4_cache/1-2_entry.pkl"))
    parser.add_argument("--solution", type=Path,
                        default=Path("data/solutions/sma4/1-2.json"))
    parser.add_argument("--route", default="DOWN,DOWN,LEFT")
    parser.add_argument("--target", default="96,96")
    parser.add_argument("--reference", type=Path,
                        default=Path("runs/sma4_cache/1-fortress_real_entry.pkl"))
    parser.add_argument("--max-settle-frames", type=int, default=1400)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--forbid-memory-writes", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 for a determinism gate")
    route = tuple(part.strip().upper() for part in args.route.split(","))
    target = _parse_pair(args.target)
    if route != EXPECTED_ROUTE or target != EXPECTED_TARGET:
        raise SystemExit(
            f"this bounded gate requires route {EXPECTED_ROUTE} "
            f"and target {EXPECTED_TARGET}"
        )
    if not args.forbid_memory_writes:
        raise SystemExit("pass --forbid-memory-writes for the integrity gate")
    if not args.no_promote:
        raise SystemExit("pass --no-promote; this command never promotes caches")
    rom_path = os.environ.get("MARIO_AI_SMA4_ROM")
    if not rom_path:
        raise SystemExit("MARIO_AI_SMA4_ROM must point to the verified SMA4 ROM")

    solution = json.loads(args.solution.read_text())
    if solution.get("rom_sha1") != EXPECTED_ROM_SHA1:
        raise SystemExit("solution manifest ROM SHA-1 is missing or unexpected")
    if not solution.get("solved") or not solution.get("replay_verified"):
        raise SystemExit("1-2 solution is not replay-verified")
    root_snapshot = _load_snapshot(args.root)

    core = SMA4Adapter(
        rom_path=rom_path,
        actions=SMA4_PSPEED_ACTIONS,
        boot_target="overworld",
    )
    try:
        if core.rom_sha1 != EXPECTED_ROM_SHA1:
            raise AssertionError(f"unexpected ROM SHA-1 {core.rom_sha1}")
        with _RuntimeAudit(core) as audit:
            runs = [
                _run_once(
                    core,
                    root_snapshot=root_snapshot,
                    solution=solution,
                    route=route,
                    target=target,
                    max_settle_frames=args.max_settle_frames,
                    audit=audit,
                )
                for _ in range(args.repeats)
            ]
        repeat_comparison = _repeat_comparison(runs)
        if not repeat_comparison["all_exact"]:
            raise AssertionError("repeat boundary hashes diverged")

        live_final_snapshot = core.snapshot()
        live_final_ram = _ram_block_bytes(core)
        reference_snapshot = _load_snapshot(args.reference)
        core.world = 1
        core.stage = 0
        core.level_id = "1-fortress"
        reference_raw_digest = _digest(core, reference_snapshot)
        core.restore(reference_snapshot)
        reference_restored_digest = _digest(core)
        reference_info = dict(core.last_info)
        reference_info["panel_slots_raw"] = int(
            reference_info.pop("cleared", 0)
        )
        reference_ram = _ram_block_bytes(core)
        ram_diff = _ram_diff(live_final_ram, reference_ram)
        observed_final = runs[0]["boundaries"][-1]["digest"]
        report = {
            "schema": "sma4-live-fortress-entry-v1",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "success": True,
            "scope": (
                "continuous declared 1-2 root -> verified 1-2 replay -> "
                "responsive overworld -> DOWN,DOWN,LEFT -> unpowered fortress "
                "spawn; stops before the incompatible P-Wing whistle root"
            ),
            "rom": {
                "path_basename": Path(rom_path).name,
                "sha1": core.rom_sha1,
            },
            "artifacts": {
                "root": str(args.root),
                "root_file_sha256": file_sha256(args.root),
                "solution": str(args.solution),
                "solution_file_sha256": file_sha256(args.solution),
                "reference": str(args.reference),
                "reference_file_sha256": file_sha256(args.reference),
                "reference_raw_digest": reference_raw_digest,
                "reference_restored_digest": reference_restored_digest,
            },
            "source_files": {
                path: file_sha256(path)
                for path in (
                    "mario/adapters.py",
                    "mario/overworld_search.py",
                    "mario/provenance.py",
                    "scripts/diagnose_sma4_fortress_entry.py",
                )
            },
            "configuration": {
                "route": list(route),
                "target": list(target),
                "max_settle_frames": args.max_settle_frames,
                "repeats": args.repeats,
                "forbid_memory_writes": True,
                "promote": False,
            },
            "runs": runs,
            "repeat_comparison": repeat_comparison,
            "historical_reference": {
                "observed_final_emulator_sha256":
                    observed_final["emulator_sha256"],
                "reference_raw_emulator_sha256":
                    reference_raw_digest["emulator_sha256"],
                "reference_restored_emulator_sha256":
                    reference_restored_digest["emulator_sha256"],
                "emulator_hash_match": (
                    observed_final["emulator_sha256"]
                    == reference_restored_digest["emulator_sha256"]
                ),
                "reference_roundtrip_raw_hash_match": (
                    reference_raw_digest["emulator_sha256"]
                    == reference_restored_digest["emulator_sha256"]
                ),
                "note": (
                    "reference mismatch is reported, not repaired; semantic "
                    "entry invariants and repeat determinism are separate gates"
                ),
                "reference_info": reference_info,
                "ram_diff_live_vs_reference": ram_diff,
                "live_final_snapshot_retained_in_memory_only": (
                    _digest(core, live_final_snapshot)
                ),
            },
            "interventions": {
                "direct_memory_assign_calls": sum(
                    run["runtime_accounting"]["direct_memory_assign_calls"]
                    for run in runs
                ),
                "external_roots_after_declared_root": 0,
                "reference_oracle_restore_after_trajectory": 1,
                "cache_promotions": 0,
                "adapter_metadata_relabels": ["1-2", "1-fortress"],
                "documented_probe_rollbacks": (
                    "settle_to_map and enter_level use non-destructive snapshot "
                    "probes; each restores the immediately preceding live state"
                ),
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "stable_retro": importlib.metadata.version("stable-retro"),
                "emulator_object": type(core.env.em).__name__,
                "integration": core._GAME,
            },
            "source_control": {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True).strip()),
                "git_diff_sha256": hashlib.sha256(subprocess.check_output(
                    ["git", "diff", "--binary"])).hexdigest(),
            },
        }
    finally:
        core.close()

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "success": report["success"],
        "repeat_all_exact": report["repeat_comparison"]["all_exact"],
        "entry": report["runs"][0]["entry_observed"],
        "final_emulator_sha256": (
            report["runs"][0]["boundaries"][-1]["digest"]["emulator_sha256"]
        ),
        "reference_hash_match": (
            report["historical_reference"]["emulator_hash_match"]
        ),
        "retained_frames": report["runs"][0]["runtime_accounting"][
            "retained_frames"],
        "probe_frames_rolled_back": report["runs"][0][
            "runtime_accounting"]["probe_frames_rolled_back"],
        "total_emulator_steps": report["runs"][0]["runtime_accounting"][
            "total_emulator_steps"],
        "direct_memory_assign_calls": report["interventions"][
            "direct_memory_assign_calls"],
        "report": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
