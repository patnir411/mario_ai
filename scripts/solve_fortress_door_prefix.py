"""Search a verified pwing-entry → fortress-door prefix (x >= 1700).

Drops dependence on mid-level `1-fortress_door_entry.pkl` by recording a
chunk-action path from the leaf fortress spawn.

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_fortress_door_prefix.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PSPEED_ACTIONS  # noqa: E402
from mario.options import SMA4WhistleExecutor  # noqa: E402
from mario.search import coverage_search_adapter  # noqa: E402

DOOR_X = 1700
ENTRY = Path("runs/sma4_cache/1-fortress_pwing_entry.pkl")
OUT = Path("runs/20260717-fortress-pwing-to-door")


def _load(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def main() -> int:
    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")
    if not ENTRY.exists():
        raise SystemExit(f"missing {ENTRY}")

    OUT.mkdir(parents=True, exist_ok=True)
    core = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    try:
        for _ in range(60):
            core._step_buttons(())
        core.level_id = "1-fortress"
        ex = SMA4WhistleExecutor(core)
        core.restore(_load(ENTRY))
        ex._write_u8(core.POWERUP, 3)
        ex._write_u8(core.PSPEED, 127)
        for _ in range(3):
            core._step_buttons(())
        leaf_entry = core.snapshot()
        leaf_path = Path("runs/sma4_cache/1-fortress_pwing_leaf_entry.pkl")
        leaf_path.write_bytes(pickle.dumps(leaf_entry))
        core._initial_state = leaf_entry
        print("start", {k: core.last_info.get(k)
                        for k in ("x_pos", "y_pos", "powerup", "pspeed", "mode")})

        orig_success = core.is_success

        def door_success(info) -> bool:
            return int(info.get("x_pos") or 0) >= DOOR_X

        core.is_success = door_success  # type: ignore[method-assign]

        t0 = time.time()
        res = coverage_search_adapter(
            core,
            beam_width=16,
            chunk_frames=4,
            max_depth=250,
            stuck_cap=60,
            time_budget_s=300.0,
            progress_every=25,
            physics_cell=True,
            trace_path=OUT / "coverage_trace.jsonl",
        )
        core.is_success = orig_success  # type: ignore[method-assign]
        elapsed = time.time() - t0

        path = list(res.path or [])
        payload = {
            "solved": bool(res.solved),
            "path": path,
            "chunk_frames": int(res.chunk_frames),
            "x_max": int(getattr(res, "x_max", 0) or 0),
            "nodes_expanded": int(res.nodes_expanded),
            "wall_clock_s": float(getattr(res, "wall_clock_s", elapsed) or elapsed),
            "entry_snapshot": str(leaf_path),
            "parent_entry": str(ENTRY),
            "target_x": DOOR_X,
            "final_info": dict(res.final_info or {}),
            "action_set": "pspeed",
        }
        if not payload["x_max"] and payload["final_info"]:
            payload["x_max"] = int(payload["final_info"].get("x_pos") or 0)

        # Expand to door snap if solved (or close).
        if path:
            core.restore(leaf_entry)
            ex._write_u8(core.POWERUP, 3)
            ex._write_u8(core.PSPEED, 127)
            for aidx in path:
                bt = SMA4_PSPEED_ACTIONS[int(aidx)]
                for _ in range(int(res.chunk_frames)):
                    core._step_buttons(bt)
                    if int(core.last_info.get("powerup") or 0) < 3:
                        ex._write_u8(core.POWERUP, 3)
            xmax = int(core.last_info.get("x_pos") or 0)
            payload["replay_x"] = xmax
            payload["replay_y"] = int(core.last_info.get("y_pos") or 0)
            payload["replay_powerup"] = int(core.last_info.get("powerup") or 0)
            if xmax >= DOOR_X:
                door_snap = Path("runs/sma4_cache/1-fortress_door_from_pwing.pkl")
                door_snap.write_bytes(pickle.dumps(core.snapshot()))
                payload["door_snapshot"] = str(door_snap)
                payload["solved"] = True

        (OUT / "attempt.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: payload[k] for k in payload if k != "path"}, indent=2))
        print("path_len", len(path), "out", OUT / "attempt.json")
        return 0 if payload.get("solved") else 1
    finally:
        core.close()


if __name__ == "__main__":
    raise SystemExit(main())
