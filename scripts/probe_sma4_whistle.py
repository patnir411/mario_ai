"""Probe SMA4 warp-whistle inventory/use/warp mechanics.

This is a feasibility probe for the option-level whistle benchmark.  It uses
hand-granted inventory items to decouple the item-spend mechanic from the 1-3
route-data acquisition trick.

Run:
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/probe_sma4_whistle.py

Findings written to runs/<ts>-sma4-whistle-probe/report.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter  # noqa: E402


INVENTORY_START = 0x03002C2E
INVENTORY_SLOTS = 36
WARP_WHISTLE = 0x0C
ITEM_MENU_OPEN = 0x03003772
MAP_EVENT = 0x03003774
MAP_DEST_OR_REGION = 0x03003CB7


def _step(adapter: SMA4Adapter, buttons: tuple[str, ...], frames: int) -> None:
    for _ in range(frames):
        adapter._step_buttons(buttons)


def _u8(adapter: SMA4Adapter, addr: int) -> int:
    return adapter._read_u8(addr)


def _inventory(adapter: SMA4Adapter) -> list[int]:
    return [_u8(adapter, INVENTORY_START + i) for i in range(INVENTORY_SLOTS)]


def _write_u8(adapter: SMA4Adapter, addr: int, value: int) -> None:
    adapter.env.data.memory.assign(addr, "|u1", int(value) & 0xFF)


def _grant_whistles(adapter: SMA4Adapter, count: int) -> None:
    for i in range(INVENTORY_SLOTS):
        _write_u8(adapter, INVENTORY_START + i, 0)
    for i in range(count):
        _write_u8(adapter, INVENTORY_START + i, WARP_WHISTLE)


def _sample(adapter: SMA4Adapter, label: str, frame: int) -> dict:
    cursor = adapter.map_cursor_info()
    return {
        "label": label,
        "frame": frame,
        "world_raw_0_indexed": _u8(adapter, adapter.WORLD),
        "world_normalized": int(adapter.last_info.get("world", 0)),
        "cursor": list(cursor["cursor"]),
        "cursor_source": cursor["source"],
        "cursor_pointer": cursor["raw_pointer"],
        "cursor_resolved_pointer": cursor["resolved_pointer"],
        "cursor_resolved": cursor["resolved"],
        "cursor_legacy": list(cursor["legacy"]),
        "mode": adapter.last_info.get("mode"),
        "time": int(adapter.last_info.get("time", 0)),
        "inventory_first4": _inventory(adapter)[:4],
        "item_menu_open": _u8(adapter, ITEM_MENU_OPEN),
        "map_event": _u8(adapter, MAP_EVENT),
        "map_dest_or_region": _u8(adapter, MAP_DEST_OR_REGION),
    }


def _shot(adapter: SMA4Adapter, out_dir: Path, label: str) -> str:
    out = out_dir / f"{label}.png"
    Image.fromarray(adapter.last_obs).save(out)
    return str(out)


def _open_and_use_selected_item(adapter: SMA4Adapter) -> int:
    frames = 0
    _step(adapter, ("L",), 6)
    frames += 6
    _step(adapter, (), 30)
    frames += 30
    _step(adapter, ("A",), 6)
    frames += 6
    return frames


def _wait_until(adapter: SMA4Adapter, predicate, *, max_frames: int = 1800,
                step: int = 12) -> tuple[bool, int]:
    elapsed = 0
    while elapsed < max_frames:
        _step(adapter, (), step)
        elapsed += step
        if predicate(adapter):
            return True, elapsed
    return False, elapsed


def main() -> int:
    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs") / f"{ts}-sma4-whistle-probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    screenshots: dict[str, str] = {}
    frame = 0

    adapter = SMA4Adapter(boot_target="overworld")
    try:
        _step(adapter, (), 60)
        frame += 60
        samples.append(_sample(adapter, "boot_overworld", frame))
        screenshots["boot_overworld"] = _shot(adapter, out_dir, "boot_overworld")

        _grant_whistles(adapter, 2)
        samples.append(_sample(adapter, "hand_granted_two_whistles", frame))

        frame += _open_and_use_selected_item(adapter)
        samples.append(_sample(adapter, "after_first_use_input", frame))
        ok, elapsed = _wait_until(
            adapter,
            lambda a: _u8(a, a.WORLD) == 8
            and a.map_cursor_info()["resolved"]
            and a.map_cursor() == (64, 80),
            max_frames=2200,
        )
        frame += elapsed
        samples.append(_sample(adapter, "after_first_whistle_warp_zone", frame))
        screenshots["after_first_whistle"] = _shot(adapter, out_dir, "after_first_whistle")
        _step(adapter, (), 180)
        frame += 180
        samples.append(_sample(adapter, "after_first_whistle_settled", frame))

        # Re-grant in the warp-zone state for this feasibility probe.  The item
        # slots visibly retain one whistle after the first use, but the copied
        # inventory state does not reliably reopen the item menu in this
        # hand-poked setup.  Real acquisition should populate the same internal
        # inventory metadata; Phase 3 must verify that path separately.
        _grant_whistles(adapter, 2)
        samples.append(_sample(adapter, "hand_regranted_two_whistles_in_warp_zone", frame))

        frame += _open_and_use_selected_item(adapter)
        samples.append(_sample(adapter, "after_second_use_input", frame))
        ok2, elapsed = _wait_until(
            adapter,
            lambda a: _u8(a, a.WORLD) == 8
            and a.map_cursor_info()["resolved"]
            and a.map_cursor() == (128, 144),
            max_frames=2400,
        )
        frame += elapsed
        samples.append(_sample(adapter, "after_second_whistle_warp_zone_5_8", frame))
        screenshots["after_second_whistle"] = _shot(adapter, out_dir, "after_second_whistle")
        _step(adapter, (), 480)
        frame += 480
        samples.append(_sample(adapter, "after_second_whistle_settled", frame))

        _step(adapter, ("RIGHT",), 16)
        _step(adapter, (), 60)
        frame += 76
        samples.append(_sample(adapter, "moved_to_world8_pipe", frame))
        screenshots["moved_to_world8_pipe"] = _shot(adapter, out_dir, "moved_to_world8_pipe")

        _step(adapter, ("A",), 16)
        _step(adapter, (), 240)
        frame += 256
        samples.append(_sample(adapter, "bowser_letter", frame))
        screenshots["bowser_letter"] = _shot(adapter, out_dir, "bowser_letter")

        _step(adapter, ("A",), 12)
        _step(adapter, (), 240)
        frame += 252
        samples.append(_sample(adapter, "world8_map", frame))
        screenshots["world8_map"] = _shot(adapter, out_dir, "world8_map")
    finally:
        adapter.close()

    final = samples[-1]
    report = {
        "timestamp": ts,
        "verdict": {
            "two_whistle_world8_executable": final["world_raw_0_indexed"] == 7,
            "one_whistle_direct_to_world8": False,
            "blockers": [] if final["world_raw_0_indexed"] == 7 else [
                "did not reach raw world byte 7 (World 8) in scripted probe"
            ],
        },
        "knowledge_injection": {
            "tier": 1,
            "facts": [
                "hand-granted two warp whistles into Mario inventory slots",
                "re-granted two warp whistles after the first warp-zone transition",
                "selected first inventory slot with L then A",
                "selected the second warp-zone World-8 pipe with RIGHT then A",
            ],
        },
        "ram": {
            "inventory_start": hex(INVENTORY_START),
            "inventory_slots": INVENTORY_SLOTS,
            "warp_whistle_item_id": hex(WARP_WHISTLE),
            "item_menu_open_flag": hex(ITEM_MENU_OPEN),
            "world": hex(SMA4Adapter.WORLD),
            "map_cursor_pointer": hex(SMA4Adapter.MAP_CURSOR_PTR),
            "map_cursor_known_bases": [
                hex(base) for base in sorted(SMA4Adapter.MAP_CURSOR_BASES)
            ],
            "legacy_map_cursor_x": hex(SMA4Adapter.MAP_CURSOR_X),
            "legacy_map_cursor_y": hex(SMA4Adapter.MAP_CURSOR_Y),
            "map_event": hex(MAP_EVENT),
            "map_dest_or_region": hex(MAP_DEST_OR_REGION),
        },
        "controls": {
            "open_inventory": "L",
            "confirm_selected_item": "A",
            "select_world8_pipe_after_second_whistle": ["RIGHT", "A"],
        },
        "samples": samples,
        "screenshots": screenshots,
        "notes": [
            "Raw world byte 8 is the special warp-zone map, not normalized World 8.",
            "Raw world byte 7 after dismissing Bowser's letter is SMA4 World 8.",
            "The live cursor is read through MAP_CURSOR_PTR; fixed legacy bytes "
            "can be stale after the 1-3 Toad-house exit.",
            "The adapter mode classifier special-cases raw world bytes 7/8 so World 8 and "
            "warp-zone map states read as overworld despite off-grid cursor Y positions.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    print("report:", out_dir / "report.json")
    return 0 if report["verdict"]["two_whistle_world8_executable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
