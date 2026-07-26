"""Probe SMA4 movement/power-state RAM on the exact local ROM.

Known useful fields from Data Crystal are read directly through `SMA4Adapter`.
Unknown movement flags (airborne/flight/tail) are explored by diffing IWRAM
around controlled action sequences from either a cached entry snapshot or the
default 1-1 boot.

Examples:
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/probe_sma4_physics.py
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/probe_sma4_physics.py runs/sma4_cache/1-2_entry.pkl
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PHYSICS_ACTIONS  # noqa: E402

IWRAM = 0x03000000
WINDOWS = [
    (0x03003C00, 0x03004020),
    (0x03004680, 0x03004780),
]


def _load_snapshot(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def _read_windows(adapter):
    out = {}
    for start, end in WINDOWS:
        out[start] = adapter._read_block(start, end - start)
    return out


def _diff(before, after):
    rows = []
    for start, b0 in before.items():
        b1 = after[start]
        for i, (x0, x1) in enumerate(zip(b0, b1)):
            if x0 != x1:
                rows.append((start + i, x0, x1))
    return rows


def _info_subset(info):
    keys = (
        "x_pos", "x_fixed", "x_subpixel", "y_pos", "y_fixed", "y_subpixel",
        "speed", "speed_signed", "pspeed", "powerup", "powerup_set",
        "spin_jump", "time", "lives", "mode",
    )
    return {k: info.get(k) for k in keys if k in info}


def _run_sequence(adapter, name, buttons, frames, sample_every=5):
    before = _read_windows(adapter)
    samples = []
    for frame in range(1, frames + 1):
        adapter._step_buttons(buttons)
        if frame == 1 or frame % sample_every == 0:
            samples.append({"frame": frame, **_info_subset(adapter.last_info)})
    changed = _diff(before, _read_windows(adapter))
    return {
        "name": name,
        "buttons": list(buttons),
        "frames": frames,
        "samples": samples,
        "changed_count": len(changed),
        "changed": [{"addr": f"{addr:#010x}", "before": b0, "after": b1}
                    for addr, b0, b1 in changed[:160]],
    }


def _diff_candidates(adapter, action_idx, frames=90):
    base = _read_windows(adapter)
    counts = Counter()
    values = defaultdict(list)
    for frame in range(1, frames + 1):
        adapter.step(action_idx)
        for addr, b0, b1 in _diff(base, _read_windows(adapter)):
            counts[addr] += 1
            if len(values[addr]) < 12:
                values[addr].append((frame, b0, b1))
    rows = []
    for addr, n in counts.most_common(80):
        rows.append({
            "addr": f"{addr:#010x}",
            "n_changed_frames": n,
            "examples": [{"frame": f, "before": b0, "after": b1}
                         for f, b0, b1 in values[addr]],
        })
    return rows


def main(argv: list[str]) -> int:
    snap_path = Path(argv[1]) if len(argv) > 1 else None
    run_id = time.strftime("%Y%m%d-%H%M%S-sma4_physics_probe")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = SMA4Adapter(actions=SMA4_PHYSICS_ACTIONS, boot_target="overworld" if snap_path else "1-1")
    try:
        if snap_path is not None:
            adapter.restore(_load_snapshot(snap_path))
            adapter._initial_state = adapter.snapshot()
        start = adapter.snapshot()
        report = {
            "snapshot": str(snap_path) if snap_path else None,
            "start_info": _info_subset(adapter.last_info),
            "known_addresses": {
                "pspeed": f"{SMA4Adapter.PSPEED:#010x}",
                "speed": f"{SMA4Adapter.SPEED:#010x}",
                "powerup": f"{SMA4Adapter.POWERUP:#010x}",
                "powerup_set": f"{SMA4Adapter.POWERUP_SET:#010x}",
                "x_fixed": f"{SMA4Adapter.PLAYER_X_FIXED:#010x}",
                "y_fixed": f"{SMA4Adapter.PLAYER_Y_FIXED:#010x}",
                "spin_jump_candidate": f"{SMA4Adapter.SPIN_JUMP:#010x}",
            },
            "sequences": [],
            "candidate_diffs": {},
        }
        for name, buttons, frames in [
            ("run_right_b", ("RIGHT", "B"), 180),
            ("jump_right_ab", ("RIGHT", "A", "B"), 60),
            ("down_b", ("DOWN", "B"), 80),
            ("left_right_b", ("LEFT", "RIGHT", "B"), 80),
        ]:
            adapter.restore(start)
            report["sequences"].append(_run_sequence(adapter, name, buttons, frames))

        for action_name in ("RIGHT+A+B", "A+B", "RIGHT+B"):
            if action_name in adapter.action_names:
                adapter.restore(start)
                idx = adapter.action_names.index(action_name)
                report["candidate_diffs"][action_name] = _diff_candidates(adapter, idx)

        out = run_dir / "physics_probe.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "out": str(out),
            "start_info": report["start_info"],
            "known_addresses": report["known_addresses"],
        }, indent=2, sort_keys=True))
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
