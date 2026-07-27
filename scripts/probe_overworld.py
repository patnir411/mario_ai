"""Confirm/refresh the SMA4 (GBA) overworld RAM map by controlled experiments.

The SMB3 overworld is a discrete navigation meta-game: a map cursor moves on a
0x20-pixel grid, a per-level progress structure records clears, an inventory holds
one-time items, and the game alternates between an overworld mode and an in-level
mode.  None of these have published GBA addresses (Data Crystal / NES `$7D00`/
`$7D80` semantics are NES-only), so we discovered them empirically by booting onto
the World-1 map (`SMA4Adapter(boot_target="overworld")`) and diffing RAM under
known inputs.

This script is the reproducible record of that reverse-engineering: it re-runs the
confirming experiments and prints/asserts the findings, so the addresses baked into
`mario.adapters` can be re-verified after any ROM/emulator change.

Confirmed (IWRAM):
    MAP_CURSOR_PTR = 0x03007824  live cursor-object base pointer
    cursor Y       = ptr + 0     units of 0x10/0x20 by map layout
    cursor X       = ptr + 4     +0x20 per horizontal node step
    known bases    = 0x03003DE0, 0x03004EF8
    WORLD        = 0x03002A69   0-indexed (0 == World 1)
    PROGRESS     = 0x03002C52   flips 0 -> 3 after clearing 1-1 (provisional bitmap)
    mode: in a level y_pos != 0 and the level timer runs; on the map the timer is 0
          and the cursor sits on the 0x20 grid.

Run:
    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/probe_overworld.py [--clear-1-1]
Report is written to runs/<ts>-owprobe/report.txt.  Pass --clear-1-1 to re-run the
(slower) level-clear experiment that locates the progress byte.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter  # noqa: E402

WORLD = 0x03002A69
PROGRESS = 0x03002C52


def _hold(adapter, buttons, n):
    for _ in range(n):
        adapter._step_buttons(buttons)


def _u8(adapter, addr):
    return adapter._read_u8(addr)


def _on_map(adapter):
    cursor = adapter.map_cursor_info()
    return (
        adapter.last_info["time"] == 0
        and bool(cursor["resolved"])
        and adapter.last_info["mode"] == "overworld"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    do_clear = "--clear-1-1" in argv
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("runs") / f"{ts}-owprobe"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"# SMA4 overworld RAM confirmation ({ts})")

    a = SMA4Adapter(boot_target="overworld")
    try:
        _hold(a, (), 90)  # settle at START
        cx0, cy0 = a.map_cursor()
        world = _u8(a, WORLD)
        cursor_info = a.map_cursor_info()
        pointer = cursor_info["resolved_pointer"]
        pointer_label = f"{pointer:#010x}" if pointer is not None else "unresolved"
        log(f"START: cursor=({cx0},{cy0}) pointer="
            f"{pointer_label} "
            f"world={world} on_map={bool(_on_map(a))}")
        assert _on_map(a), "boot_target='overworld' did not land on the map"

        # Cursor X tracks RIGHT; cursor Y tracks UP; each move steps the 0x20 grid.
        _hold(a, ("RIGHT",), 16); _hold(a, (), 20)
        cx1, cy1 = a.map_cursor()
        log(f"after RIGHT: cursor=({cx1},{cy1})  dx={cx1 - cx0} dy={cy1 - cy0}")
        assert cx1 > cx0 and cy1 == cy0, "cursor X did not track RIGHT"
        _hold(a, ("UP",), 16); _hold(a, (), 20)
        cx2, cy2 = a.map_cursor()
        log(f"after UP:    cursor=({cx2},{cy2})  dx={cx2 - cx1} dy={cy2 - cy1}")
        assert cy2 < cy1 and cx2 == cx1, "cursor Y did not track UP"

        # Enter the level under the cursor and confirm the mode transition.
        node = a.snapshot()
        entered_x = 0
        for _ in range(60):
            a._step_buttons(("A",)); a._step_buttons(())
            if a.last_info["x_pos"] > 5:
                entered_x = a.last_info["x_pos"]
                break
        log(f"enter-level: x_pos={entered_x} y_pos={a.last_info['y_pos']} "
            f"time={a.last_info['time']} on_map={bool(_on_map(a))}")
        assert entered_x > 5 and not _on_map(a), "pressing A did not enter a level"
        a.restore(node)
        assert _on_map(a), "restore did not return to the overworld"
        log("mode transition overworld<->level confirmed (and snapshot/restore reversible)")
    finally:
        a.close()

    if do_clear:
        log("")
        log("## level-clear progress experiment (beat 1-1, diff map progress region)")
        sol = json.loads(Path("data/solutions/sma4/1-1.json").read_text())
        b = SMA4Adapter()  # default boot to 1-1 gameplay (matches the cached solution)
        try:
            before = _u8(b, PROGRESS)
            cf = sol["chunk_frames"]
            for act in sol["path"]:
                b.run_chunk(act, cf)
                if b.is_success(b.last_info):
                    break
            on_map = False
            for _ in range(2000):
                b._step_buttons(())
                if _on_map(b):
                    on_map = True
                    break
            after = _u8(b, PROGRESS)
            cursor = b.map_cursor()
            log(f"progress {PROGRESS:#010x}: before={before} after={after} "
                f"(back_on_map={on_map}, cursor advanced to "
                f"{cursor})")
        finally:
            b.close()

    report = out_dir / "report.txt"
    report.write_text("\n".join(lines) + "\n")
    log("")
    log(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
