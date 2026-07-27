"""Coverage-search fortress door → whistle chest (inventory 0x0C).

Starts from ``1-fortress_door_from_pwing.pkl`` (live pwing→door prefix end).

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_fortress_chest_from_door.py
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

DOOR = Path("runs/sma4_cache/1-fortress_door_from_pwing.pkl")
OUT = Path("runs/20260717-fortress-door-to-chest")
WHISTLE = 0x0C
INV = 0x03002C2E


def _load(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def main() -> int:
    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM required")
    if not DOOR.exists():
        raise SystemExit(f"missing {DOOR}; run solve_fortress_door_prefix.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    core = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    try:
        for _ in range(60):
            core._step_buttons(())
        core.level_id = "1-fortress"
        ex = SMA4WhistleExecutor(core)
        core.restore(_load(DOOR))
        ex._write_u8(core.POWERUP, 3)
        ex._write_u8(core.PSPEED, 127)
        for _ in range(3):
            core._step_buttons(())
        start = core.snapshot()
        core._initial_state = start
        print("start", {k: core.last_info.get(k)
                        for k in ("x_pos", "y_pos", "powerup", "pspeed")})

        def whistle_success(info) -> bool:
            # Read live inventory; info may not carry slots.
            return int(core._read_u8(INV)) == WHISTLE

        orig = core.is_success
        core.is_success = whistle_success  # type: ignore[method-assign]
        res = coverage_search_adapter(
            core,
            beam_width=20,
            chunk_frames=4,
            max_depth=300,
            stuck_cap=80,
            time_budget_s=420.0,
            progress_every=25,
            physics_cell=True,
            trace_path=OUT / "coverage_trace.jsonl",
        )
        core.is_success = orig  # type: ignore[method-assign]

        path = list(res.path or [])
        payload = {
            "solved": bool(res.solved),
            "path": path,
            "chunk_frames": int(res.chunk_frames),
            "x_max": int(res.x_max),
            "nodes_expanded": int(res.nodes_expanded),
            "wall_clock_s": float(getattr(res, "wall_clock_s", 0) or 0),
            "entry_snapshot": str(DOOR),
            "final_info": dict(res.final_info or {}),
        }

        # Replay + expand frame buttons + exit settle if whistle held.
        if path:
            core.restore(start)
            ex._write_u8(core.POWERUP, 3)
            ex._write_u8(core.PSPEED, 127)
            frame_buttons: list[list[str]] = []
            for aidx in path:
                bt = SMA4_PSPEED_ACTIONS[int(aidx)]
                for _ in range(int(res.chunk_frames)):
                    core._step_buttons(bt)
                    frame_buttons.append(list(bt))
                    if int(core.last_info.get("powerup") or 0) < 3:
                        ex._write_u8(core.POWERUP, 3)
                    if int(core._read_u8(INV)) == WHISTLE:
                        break
                if int(core._read_u8(INV)) == WHISTLE:
                    break
            payload["replay_inv0"] = int(core._read_u8(INV))
            payload["replay_xy"] = [int(core.last_info.get("x_pos") or 0),
                                   int(core.last_info.get("y_pos") or 0)]
            payload["path_buttons_to_chest"] = frame_buttons
            if payload["replay_inv0"] == WHISTLE:
                payload["solved"] = True
                # UP + idle + B settle
                for _ in range(120):
                    core._step_buttons(("UP",))
                    frame_buttons.append(["UP"])
                    if core.last_info.get("mode") == "overworld":
                        break
                for _ in range(200):
                    core._step_buttons(())
                    frame_buttons.append([])
                for _ in range(40):
                    core._step_buttons(("B",))
                    frame_buttons.append(["B"])
                payload["path_buttons"] = frame_buttons
                payload["exit_mode"] = core.last_info.get("mode")
                payload["exit_cursor"] = list(ex._cursor())
                payload["exit_cursor_info"] = ex._cursor_info()
                # L-menu check
                snap = core.snapshot()
                menu = False
                for _ in range(20):
                    core._step_buttons(("L",))
                    if ex._read_u8(ex.ITEM_MENU_OPEN):
                        menu = True
                        break
                core.restore(snap)
                payload["menu_ok"] = menu

        (OUT / "attempt.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: payload[k] for k in payload
                          if k not in ("path", "path_buttons", "path_buttons_to_chest")},
                         indent=2))
        print("path_len", len(path), "out", OUT / "attempt.json")
        return 0 if payload.get("solved") and payload.get("menu_ok") else 1
    finally:
        core.close()


if __name__ == "__main__":
    raise SystemExit(main())
