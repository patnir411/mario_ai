"""SMA4 (Stable-Retro/mGBA) throughput bench on this machine.

Mirrors ``scripts/bench.py`` for nes-py SMB1: step rate, snapshot dump/load,
and a realistic search-node cost (restore → chunk → snapshot).

Writes:
  bench/sma4_step_rate.json
  bench/sma4_snapshot_cost.json

    MARIO_AI_SMA4_ROM=roms/...gba ./venv/bin/python scripts/bench_sma4.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.adapters import SMA4Adapter  # noqa: E402
from mario.io import env_fingerprint, utc_now_iso, write_json_atomic  # noqa: E402

BENCH = ROOT / "bench"


def _require_rom() -> None:
    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM must point to a legally obtained SMA4 ROM")


def bench_step_rate(n_frames: int = 10000) -> dict:
    adapter = SMA4Adapter(boot_target="1-1")
    try:
        # Get into gameplay (timer running).
        for _ in range(180):
            adapter._step_buttons(("RIGHT", "B"))
        t0 = time.perf_counter()
        steps = 0
        while steps < n_frames:
            adapter._step_buttons(("RIGHT", "B"))
            steps += 1
            if adapter.is_death(adapter.last_info, False) or adapter.is_success(adapter.last_info):
                adapter.reset()
                for _ in range(60):
                    adapter._step_buttons(("RIGHT", "B"))
        dt = time.perf_counter() - t0
    finally:
        adapter.close()
    return {
        "n_frames": steps,
        "wall_clock_s": round(dt, 4),
        "fps": round(steps / dt, 1),
        "us_per_frame": round(dt / steps * 1e6, 2),
        "mode": "level_gameplay",
    }


def bench_snapshot(n: int = 2000) -> dict:
    adapter = SMA4Adapter(boot_target="1-1")
    try:
        for _ in range(180):
            adapter._step_buttons(("RIGHT", "B"))
        t0 = time.perf_counter()
        snaps = [adapter.snapshot() for _ in range(n)]
        dump_dt = time.perf_counter() - t0
        snap = snaps[-1]
        t0 = time.perf_counter()
        for _ in range(n):
            adapter.restore(snap)
        load_dt = time.perf_counter() - t0
    finally:
        adapter.close()
    return {
        "n": n,
        "dump_us": round(dump_dt / n * 1e6, 3),
        "load_us": round(load_dt / n * 1e6, 3),
        "roundtrip_us": round((dump_dt + load_dt) / n * 1e6, 3),
    }


def estimate_search_node(chunk_frames: int = 4, n: int = 2000) -> dict:
    adapter = SMA4Adapter(boot_target="1-1")
    try:
        for _ in range(180):
            adapter._step_buttons(("RIGHT", "B"))
        # Prefer RIGHT+B action index if present.
        try:
            action_idx = adapter.action_names.index("RIGHT+B")
        except ValueError:
            action_idx = min(1, adapter.n_actions - 1)
        parent = adapter.snapshot()
        t0 = time.perf_counter()
        for _ in range(n):
            adapter.restore(parent)
            adapter.run_chunk(action_idx, chunk_frames)
            _child = adapter.snapshot()
        dt = time.perf_counter() - t0
    finally:
        adapter.close()
    return {
        "chunk_frames": chunk_frames,
        "n_nodes": n,
        "us_per_node": round(dt / n * 1e6, 2),
        "nodes_per_s": round(n / dt, 1),
    }


def main() -> None:
    _require_rom()
    fp = env_fingerprint()
    fp["emulator"] = "stable-retro/mGBA"
    fp["game"] = "sma4"

    print("benching SMA4 step rate…")
    step = bench_step_rate()
    print("benching SMA4 snapshots…")
    snap = bench_snapshot()
    print("benching SMA4 search node…")
    node = estimate_search_node()

    write_json_atomic(
        BENCH / "sma4_step_rate.json",
        {"generated_at": utc_now_iso(), "env": fp, **step},
    )
    write_json_atomic(
        BENCH / "sma4_snapshot_cost.json",
        {"generated_at": utc_now_iso(), "env": fp, **snap, "search_node": node},
    )

    print("step rate   :", step["fps"], "fps  (", step["us_per_frame"], "us/frame )")
    print(
        "snapshot    : dump", snap["dump_us"], "us | load", snap["load_us"],
        "us | roundtrip", snap["roundtrip_us"], "us",
    )
    print(
        "search node :", node["nodes_per_s"], "nodes/s  (", node["us_per_node"],
        "us/node, chunk=", node["chunk_frames"], ")",
    )


if __name__ == "__main__":
    main()
