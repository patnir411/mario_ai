#!/usr/bin/env python3
"""Finer-grid teleport enter scan for W1 Fortress after cached 1-1/1-2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mario.adapters import SMA4OverworldAdapter, SMA4Adapter, SMA4_PSPEED_ACTIONS
from mario.overworld_search import (
    _load_snapshot,
    discover_nodes,
    replay_cached_solution,
    settle_to_map,
)

ROM = "roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba"
OUT = Path("runs/20260715-fortress-recon4")


def shot(core, name: str) -> None:
    Image.fromarray(np.asarray(core.last_obs)).save(OUT / name)


def cursor(core) -> tuple[int, int]:
    return tuple(core.last_info.get("cursor", (0, 0)))


def set_cursor(core, x: int, y: int) -> None:
    cursor_info = core.map_cursor_info()
    base = cursor_info["resolved_pointer"]
    if base is None:
        raise RuntimeError(f"cannot inject unresolved map cursor: {cursor_info}")
    core.env.data.memory.assign(base + 4, "|u1", int(x) & 0xFF)
    core.env.data.memory.assign(base, "|u1", int(y) & 0xFF)
    for _ in range(20):
        core._step_buttons(())


def ff12(ow: SMA4OverworldAdapter) -> None:
    core = ow.core
    for label in ("1-1", "1-2"):
        sol = json.loads(Path(f"data/solutions/sma4/{label}.json").read_text())
        snap = _load_snapshot(Path(f"runs/sma4_cache/{label}_entry.pkl"))
        summary = replay_cached_solution(core, sol, entry_snap=snap, settle=True)
        print("ff", label, "solved", summary.get("solved"),
              "cursor", cursor(core), "mode", core.last_info.get("mode"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = SMA4Adapter(ROM, actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    ow = SMA4OverworldAdapter(ROM, core=core)
    settle_to_map(core, max_frames=600)
    ff12(ow)
    shot(core, "post12.png")
    base = core.snapshot()
    g = discover_nodes(ow, max_nodes=48)
    print("reachable", sorted(g.snaps))
    print("enterable", g.levels())

    # Data Crystal: 03002C85=02 makes all levels available (Tier-3 unlock).
    unlock_addr = 0x03002C85
    core.restore(base)
    old_flag = core._read_u8(unlock_addr)
    core.env.data.memory.assign(unlock_addr, "|u1", 2)
    for _ in range(40):
        core._step_buttons(())
    unlocked = core.snapshot()
    shot(core, "after_c85_unlock.png")
    print("c85 old", old_flag, "new", core._read_u8(unlock_addr))

    # Probe whether LEFT from (128,64) opens after unlock
    set_cursor(core, 128, 64)
    before = cursor(core)
    for _ in range(24):
        core._step_buttons(("LEFT",))
    for _ in range(40):
        core._step_buttons(())
    print("left from 128,64:", before, "->", cursor(core))
    core.restore(unlocked)

    # Re-discover after unlock
    ow2 = SMA4OverworldAdapter(ROM, core=core)
    g2 = discover_nodes(ow2, max_nodes=64)
    print("unlocked reachable", sorted(g2.snaps))
    print("unlocked enterable", g2.levels())
    for cell in sorted(g2.levels()):
        if cell not in g.levels():
            core.restore(g2.snaps[cell])
            info = core.enter_level()
            shot(core, f"new_enter_{cell[0]}_{cell[1]}.png")
            print("NEW ENTERABLE", cell, "x", info.get("x_pos"), "y", info.get("y_pos"),
                  "mean", float(np.asarray(core.last_obs).mean()))

    hits = []
    interesting = []
    for y in range(0, 201, 16):
        for x in range(0, 201, 16):
            core.restore(unlocked)
            set_cursor(core, x, y)
            cx, cy = cursor(core)
            info = core.enter_level()
            entered = info.get("mode") == "level"
            row = {
                "tp": [x, y],
                "cur": [cx, cy],
                "entered": entered,
                "x": int(info.get("x_pos", 0)),
                "y": int(info.get("y_pos", 0)),
                "time": int(info.get("time", 0)),
                "mean": float(np.asarray(core.last_obs).mean()),
            }
            if entered:
                hits.append(row)
                # Only screenshot newly enterable / fortress-looking cells
                known = set(g.levels())
                if (cx, cy) not in known and (x, y) not in known:
                    shot(core, f"enter_{x}_{y}.png")
                    print("ENTER", row)
            elif x % 32 == 0 and y % 32 == 0 and 32 <= x <= 160 and 32 <= y <= 128:
                interesting.append(row)

    Path(OUT / "hits.json").write_text(json.dumps({
        "hits": hits,
        "interesting_count": len(interesting),
        "reachable_before": [list(c) for c in sorted(g.snaps)],
        "enterable_before": [list(c) for c in g.levels()],
        "reachable_after_c85": [list(c) for c in sorted(g2.snaps)],
        "enterable_after_c85": [list(c) for c in g2.levels()],
        "c85_old": old_flag,
    }, indent=2))
    core.close()
    print("DONE", OUT, "enters", len(hits))


if __name__ == "__main__":
    main()
