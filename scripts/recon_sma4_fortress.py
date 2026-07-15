#!/usr/bin/env python3
"""Recon: find / unlock World-1 Fortress for the second AcquireWhistle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mario.adapters import SMA4OverworldAdapter
from mario.overworld_search import (
    discover_nodes,
    remap_solution_path,
    replay_cached_solution,
    settle_to_map,
)

ROM_DEFAULT = "roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba"
OUT = Path("runs/20260715-fortress-recon3")


def _shot(core, path: Path) -> None:
    Image.fromarray(np.asarray(core.last_obs)).save(path)


def _set_cursor(core, x: int, y: int) -> None:
    core.env.data.memory.assign(core.MAP_CURSOR_X, "|u1", int(x) & 0xFF)
    core.env.data.memory.assign(core.MAP_CURSOR_Y, "|u1", int(y) & 0xFF)
    for _ in range(20):
        core.step(0)
    core._last_info = core._normalize_info(core.last_info)


def _try_enter(ow: SMA4OverworldAdapter, frames: int = 240) -> dict:
    """Enter via core.enter_level (same path as discover_nodes)."""
    del frames
    core = ow.core
    before = dict(core.last_info)
    info = core.enter_level()
    entered = info.get("mode") == "level"
    return {
        "entered": bool(entered),
        "mode": info.get("mode"),
        "time": int(info.get("time", 0)),
        "x": int(info.get("x_pos", 0)),
        "y": int(info.get("y_pos", 0)),
        "powerup": int(info.get("powerup", 0) or 0),
        "obs_mean": float(np.asarray(core.last_obs).mean()),
        "cursor": list(info.get("cursor", before.get("cursor", (0, 0)))),
        "enterable_strict": entered and int(info.get("x_pos", 0)) > 5,
    }


def fast_forward_12(ow: SMA4OverworldAdapter) -> None:
    core = ow.core
    for label in ("1-1", "1-2"):
        sol = json.loads(Path(f"data/solutions/sma4/{label}.json").read_text())
        snap = Path(f"runs/sma4_cache/{label}_entry.pkl").read_bytes()
        path = remap_solution_path(sol)
        core.restore(snap)
        ok = replay_cached_solution(core, path, settle_frames=400)
        print(f"ff {label} ok={ok} mode={ow.mode()} cursor={ow.cursor_pos()}")
        settle_to_map(ow.core, max_frames=900)


def scan_grid(ow: SMA4OverworldAdapter, out: Path) -> list[dict]:
    """Teleport cursor across the 0x20 grid and try A-enter."""
    core = ow.core
    base = core.snapshot()
    hits = []
    for y in range(0, 241, 32):
        for x in range(0, 241, 32):
            core.restore(base)
            _set_cursor(core, x, y)
            cx, cy = ow.cursor_pos()
            # cursor may snap; record actual
            result = _try_enter(ow)
            row = {"teleport": [x, y], "cursor": [cx, cy], **result}
            if result.get("entered") or (x, y) in {
                (32, 64), (64, 64), (64, 96), (96, 64), (32, 96), (0, 64),
            }:
                tag = f"t_{x}_{y}"
                _shot(core, out / f"{tag}.png")
                print("HIT/interesting", row)
            if result.get("entered"):
                hits.append(row)
            core.restore(base)
    return hits


def walk_reach(ow: SMA4OverworldAdapter) -> dict:
    """BFS walk without teleport; record reachability + enter attempts."""
    from collections import deque

    core = ow.core
    start = tuple(ow.cursor_pos())
    q = deque([start])
    seen = {start}
    edges = {}
    enterable = {}
    while q:
        node = q.popleft()
        snap = core.snapshot()
        _set_cursor(core, node[0], node[1])
        # prefer walking if possible — but teleport for reliability after snap
        ent = _try_enter(ow, frames=180)
        enterable[str(node)] = ent
        core.restore(snap)
        _set_cursor(core, node[0], node[1])
        edges[str(node)] = {}
        for button, name in ((("UP",), "UP"), (("DOWN",), "DOWN"),
                             (("LEFT",), "LEFT"), (("RIGHT",), "RIGHT")):
            before = tuple(ow.cursor_pos())
            for _ in range(24):
                core._step_buttons(button)
            for _ in range(40):
                core.step(0)
            after = tuple(ow.cursor_pos())
            edges[str(node)][name] = list(after)
            if after not in seen and after != before:
                seen.add(after)
                q.append(after)
            core.restore(snap)
            _set_cursor(core, node[0], node[1])
    return {
        "start": list(start),
        "reachable": [list(p) for p in sorted(seen)],
        "edges": edges,
        "enterable": enterable,
    }


def poke_progress_sweep(ow: SMA4OverworldAdapter, out: Path) -> list[dict]:
    """Flip bytes near PROGRESS and see if (64,64) becomes enterable."""
    core = ow.core
    base = core.snapshot()
    progress_addr = core.PROGRESS
    results = []
    # Dump neighborhood
    dump = bytes(core.env.data.memory[progress_addr - 32: progress_addr + 64])
    (out / "progress_neighborhood.bin").write_bytes(dump)
    candidates = []
    for off in range(-16, 48):
        addr = progress_addr + off
        cur = int(core.env.data.memory[addr])
        candidates.append((addr, cur, 0xFF))
        if cur != 0:
            candidates.append((addr, cur, 0))
    for addr, old, new in candidates:
        core.restore(base)
        core.env.data.memory.assign(addr, "|u1", new)
        for _ in range(30):
            core.step(0)
        # walk/teleport to likely fortress cells
        for cell in ((64, 64), (32, 64), (64, 96), (96, 64)):
            _set_cursor(core, *cell)
            ent = _try_enter(ow, frames=120)
            if ent.get("entered"):
                row = {"addr": hex(addr), "old": old, "new": new, "cell": list(cell), **ent}
                results.append(row)
                _shot(core, out / f"unlock_{addr:08x}_{cell[0]}_{cell[1]}.png")
                print("UNLOCK", row)
                break
        else:
            continue
        break
    return results


def fresh_boot_scan(rom: str, out: Path) -> dict:
    grid = out / "fresh_grid"
    grid.mkdir(parents=True, exist_ok=True)
    ow = SMA4OverworldAdapter(rom)
    settle_to_map(ow.core, max_frames=600)
    _shot(ow.core, out / "fresh_map.png")
    g = discover_nodes(ow, max_nodes=48)
    print("fresh discover enterable", g.levels())
    # Stand on every discovered node and screenshot + enter
    node_shots = {}
    for cursor, snap in g.snaps.items():
        ow.core.restore(snap)
        tag = f"n_{cursor[0]}_{cursor[1]}"
        _shot(ow.core, out / f"fresh_{tag}.png")
        ent = _try_enter(ow)
        node_shots[str(cursor)] = ent
        if ent.get("entered"):
            _shot(ow.core, out / f"fresh_entered_{tag}.png")
            print("fresh ENTERED", cursor, ent)
    walk = walk_reach(ow)
    hits = scan_grid(ow, grid)
    ow.core.close()
    return {
        "discover_enterable": {str(k): v for k, v in g.enterable.items()},
        "discover_nodes": [list(c) for c in sorted(g.snaps)],
        "node_enter_detail": node_shots,
        "walk": walk,
        "teleport_hits": hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default=ROM_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--mode", choices=("post12", "fresh", "both"), default="both")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "post12_grid").mkdir(parents=True, exist_ok=True)
    (args.out / "fresh_grid").mkdir(parents=True, exist_ok=True)

    report: dict = {}
    if args.mode in ("fresh", "both"):
        print("=== FRESH BOOT ===")
        report["fresh"] = fresh_boot_scan(args.rom, args.out)

    if args.mode in ("post12", "both"):
        print("=== POST 1-1/1-2 ===")
        ow = SMA4OverworldAdapter(args.rom)
        settle_to_map(ow.core, max_frames=600)
        fast_forward_12(ow)
        _shot(ow.core, args.out / "post12_map.png")
        print("cursor", ow.cursor_pos(), "mode", ow.mode())
        walk = walk_reach(ow)
        report["post12_walk"] = walk
        hits = scan_grid(ow, args.out / "post12_grid")
        report["post12_teleport_hits"] = hits
        unlocks = poke_progress_sweep(ow, args.out)
        report["progress_unlocks"] = unlocks
        # Also try entering while standing on every reachable node via walk only
        enterable_true = {
            k: v for k, v in walk["enterable"].items() if v.get("entered")
        }
        report["post12_walk_enterable"] = enterable_true
        print("walk enterable", enterable_true)
        ow.core.close()

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("DONE", args.out)


if __name__ == "__main__":
    main()
