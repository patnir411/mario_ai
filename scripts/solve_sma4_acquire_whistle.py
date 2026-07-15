"""Solve / replay-verify SMA4 1-3 AcquireWhistle (white-block → Toad chest).

Pipeline (Tier-2 option, P-Wing entry required):
  1. Restore ``runs/sma4_cache/1-3_pwing_entry.pkl`` (powerup=3, pspeed=127)
  2. Fly to x≈1680, land on the white block, hold DOWN until
     ``0x03003D06`` (behind-bg timer) is nonzero
  3. Sprint RIGHT+B past the black curtain (x≥2000 → x<200 room change)
  4. Walk to the chest and hold B (SMA4 opens Toad chests with B)
  5. Exit left back to the World-1 map with inventory ``0x0C``

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_sma4_acquire_whistle.py
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_sma4_acquire_whistle.py --replay-only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PSPEED_ACTIONS  # noqa: E402
from mario.options import SMA4WhistleExecutor  # noqa: E402

BEHIND_BG = 0x03003D06
INVENTORY_START = 0x03002C2E
WARP_WHISTLE = 0x0C
ENTRY = Path("runs/sma4_cache/1-3_pwing_entry.pkl")
SOLUTION = Path("data/solutions/sma4/acquire_whistle_1_3.json")


def _load(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def _inv(adapter: SMA4Adapter, n: int = 8) -> list[int]:
    return [adapter._read_u8(INVENTORY_START + i) for i in range(n)]


def _record_pipeline(adapter: SMA4Adapter, out: Path) -> dict:
    frames: list[tuple[str, ...]] = []

    def step(buttons: tuple[str, ...] = ()) -> None:
        frames.append(tuple(buttons))
        adapter._step_buttons(buttons)

    # Fly
    for f in range(3000):
        btns = ("RIGHT", "A", "B") if (f % 4) < 2 else ("RIGHT", "B")
        step(btns)
        if int(adapter.last_info["x_pos"]) >= 1680:
            break
    else:
        raise RuntimeError("flight failed")
    Image.fromarray(adapter.last_obs).save(out / "01_arrive.png")

    # Land on white block
    for _ in range(15):
        step(("LEFT",))
    for _ in range(160):
        step(())
    for _ in range(30):
        spd = int(adapter.last_info.get("speed_signed", 0))
        if abs(spd) <= 2:
            break
        step(("LEFT",) if spd > 0 else ("RIGHT",))
    for _ in range(8):
        step(())
    Image.fromarray(adapter.last_obs).save(out / "02_on_white.png")
    if not (int(adapter.last_info["powerup"]) >= 1
            and 40 <= int(adapter.last_info["y_pos"]) <= 90
            and 1620 <= int(adapter.last_info["x_pos"]) <= 1780):
        raise RuntimeError(f"land failed: {adapter.last_info}")

    # Duck
    act = None
    for i in range(320):
        step(("DOWN",))
        if adapter._read_u8(BEHIND_BG) > 0:
            act = i
            break
        if int(adapter.last_info["powerup"]) < 1 or int(adapter.last_info["y_pos"]) > 100:
            raise RuntimeError(f"duck failed at {i}")
    if act is None:
        raise RuntimeError("duck timeout")
    Image.fromarray(adapter.last_obs).save(out / "03_ACT.png")

    # Sprint to Toad house
    prev = int(adapter.last_info["x_pos"])
    trans = None
    for r in range(700):
        step(("RIGHT", "B"))
        x = int(adapter.last_info["x_pos"])
        if prev >= 2000 and x < 200:
            trans = r
            break
        prev = x
    if trans is None:
        raise RuntimeError("no toad transition")
    Image.fromarray(adapter.last_obs).save(out / "05_trans.png")

    for _ in range(300):
        step(())
        if (float(np.asarray(adapter.last_obs).mean()) > 40
                and int(adapter.last_info["x_pos"]) < 200):
            break
    Image.fromarray(adapter.last_obs).save(out / "06_house.png")

    # Open chest with B
    for _ in range(174):
        step(("RIGHT",))
    b_frames = None
    for f in range(120):
        step(("B",))
        if adapter._read_u8(INVENTORY_START) == WARP_WHISTLE:
            b_frames = f + 1
            break
    if b_frames is None:
        raise RuntimeError("chest open failed")
    for _ in range(40):
        step(())
    Image.fromarray(adapter.last_obs).save(out / "07_whistle.png")

    # Exit to map
    for i in range(300):
        step(("LEFT",))
        if float(np.asarray(adapter.last_obs).mean()) < 30:
            break
    for _ in range(120):
        step(())
    for _ in range(8):
        step(("B",))
        step(())
    for _ in range(30):
        step(())
    Image.fromarray(adapter.last_obs).save(out / "08_map.png")

    inv = _inv(adapter)
    if WARP_WHISTLE not in inv:
        raise RuntimeError(f"lost whistle on exit: {inv}")
    return {
        "frames": frames,
        "act_frame": act,
        "trans_frame": trans,
        "b_frames": b_frames,
        "inventory": inv,
        "final_info": dict(adapter.last_info),
    }


def _replay(adapter: SMA4Adapter, buttons: list[tuple[str, ...]], out: Path) -> list[int]:
    for bt in buttons:
        adapter._step_buttons(bt)
    Image.fromarray(adapter.last_obs).save(out / "replay_final.png")
    inv = _inv(adapter)
    if WARP_WHISTLE not in inv:
        raise RuntimeError(f"replay missing whistle: {inv}")
    return inv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-only", action="store_true",
                        help="only replay the cached solution JSON")
    args = parser.parse_args()

    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM required")
    if not ENTRY.exists():
        raise SystemExit(f"missing {ENTRY}")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y%m%d-%H%M%S-sma4-acquire-whistle")
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    if args.replay_only:
        if not SOLUTION.exists():
            raise SystemExit(f"missing {SOLUTION}")
        sol = json.loads(SOLUTION.read_text())
        buttons = [tuple(b) for b in sol["path_buttons"]]
        core = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
        try:
            core.level_id = "1-3"
            core.restore(_load(Path(sol["entry_snapshot"])))
            inv = _replay(core, buttons, out)
        finally:
            core.close()
        print(json.dumps({"replay_verified": True, "inventory": inv, "out": str(out)},
                         indent=2))
        return 0

    # Prefer the executor path so solution + option stay in sync.
    core = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    t0 = time.perf_counter()
    try:
        if SOLUTION.exists():
            ex = SMA4WhistleExecutor(core)
            summary = ex.acquire_whistle_1_3(SOLUTION)
            Image.fromarray(core.last_obs).save(out / "executor_final.png")
            report = {
                "via": "SMA4WhistleExecutor.acquire_whistle_1_3",
                "summary": {
                    k: summary[k] for k in summary
                    if k != "samples"
                },
                "wall_clock_s": time.perf_counter() - t0,
                "out_dir": str(out),
            }
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return 0 if summary.get("success") else 1

        core.level_id = "1-3"
        core.restore(_load(ENTRY))
        result = _record_pipeline(core, out)
        buttons = result["frames"]
    finally:
        core.close()

    # Replay-verify freshly recorded path.
    core = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    try:
        core.level_id = "1-3"
        core.restore(_load(ENTRY))
        inv = _replay(core, buttons, out)
    finally:
        core.close()

    sol = {
        "game_id": "sma4",
        "level_id": "1-3",
        "option": "AcquireWhistle",
        "option_id": "acquire_whistle_1_3",
        "knowledge_tier": 2,
        "solved": True,
        "replay_verified": True,
        "exit_verified": True,
        "inventory_item": WARP_WHISTLE,
        "inventory_after": inv,
        "entry_snapshot": str(ENTRY),
        "requires": {
            "powerup_min": 1,
            "entry_powerup": 3,
            "entry_pspeed": 127,
            "pwing_or_super": True,
        },
        "behind_bg_addr": hex(BEHIND_BG),
        "solver": "scripted_whiteblock_pwing_pipeline",
        "chunk_frames": 1,
        "n_frames": len(buttons),
        "path_buttons": [list(b) for b in buttons],
        "recipe": {
            "fly_target_x": 1680,
            "land": {"left": 15, "settle": 160, "brake": True},
            "duck_hold_down_until_bg": True,
            "sprint_buttons": ["RIGHT", "B"],
            "transition": "x_collapse_from_ge_2000_to_lt_200",
            "chest": {"walk_right": 174, "hold_b": result["b_frames"]},
            "exit": {"walk_left_until_black": True, "settle_frames": 120},
            "act_frame_observed": result["act_frame"],
            "trans_frame_observed": result["trans_frame"],
        },
        "artifacts": {"run_dir": str(out)},
        "notes": (
            "Tier-2: white-block duck requires Super/Raccoon (P-Wing entry). "
            "Chest opens with B. Inventory 0x03002C2E gets 0x0C."
        ),
    }
    SOLUTION.parent.mkdir(parents=True, exist_ok=True)
    SOLUTION.write_text(json.dumps(sol, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps({
        "solved": True, "replay_verified": True, "n_frames": len(buttons),
        "inventory": inv, "solution": str(SOLUTION),
    }, indent=2) + "\n")
    print(json.dumps({"solution": str(SOLUTION), "n_frames": len(buttons),
                      "inventory": inv, "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
