"""Microbenchmark the two costs that bound beam-search throughput on this machine:
  1. headless env.step rate (frames/sec)
  2. dump_state/load_state round-trip cost (the per-node clone cost)

Writes bench/step_rate.json and bench/snapshot_cost.json. Per the plan, per-node clone
cost (not raw fps) usually dominates, so we also report an estimated nodes/sec for a
realistic search node = (snapshot + run a 4-frame chunk + restore).

    ./venv/bin/python scripts/bench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.env import MarioSim  # noqa: E402
from mario.io import env_fingerprint, utc_now_iso, write_json_atomic  # noqa: E402

BENCH = ROOT / "bench"


def bench_step_rate(n_frames: int = 20000) -> dict:
    sim = MarioSim(1, 1)
    sim.reset(seed=0)
    t0 = time.perf_counter()
    steps = 0
    while steps < n_frames:
        _o, _info, done = sim.step(3)  # right+B
        steps += 1
        if done:
            sim.reset(seed=0)
    dt = time.perf_counter() - t0
    sim.close()
    return {
        "n_frames": steps,
        "wall_clock_s": round(dt, 4),
        "fps": round(steps / dt, 1),
        "us_per_frame": round(dt / steps * 1e6, 2),
    }


def bench_snapshot(n: int = 5000) -> dict:
    sim = MarioSim(1, 1)
    sim.reset(seed=0)
    for _ in range(60):  # get into a representative mid-level state
        sim.step(3)

    # dump cost
    t0 = time.perf_counter()
    snaps = [sim.snapshot() for _ in range(n)]
    dump_dt = time.perf_counter() - t0

    # load cost (restore the same snapshot repeatedly)
    snap = snaps[-1]
    t0 = time.perf_counter()
    for _ in range(n):
        sim.restore(snap)
    load_dt = time.perf_counter() - t0
    sim.close()

    return {
        "n": n,
        "dump_us": round(dump_dt / n * 1e6, 3),
        "load_us": round(load_dt / n * 1e6, 3),
        "roundtrip_us": round((dump_dt + load_dt) / n * 1e6, 3),
    }


def estimate_search_node(chunk_frames: int = 4, n: int = 5000) -> dict:
    """A realistic beam-search node: restore parent -> run a chunk -> snapshot child."""
    sim = MarioSim(1, 1)
    sim.reset(seed=0)
    for _ in range(60):
        sim.step(3)
    parent = sim.snapshot()
    t0 = time.perf_counter()
    for _ in range(n):
        sim.restore(parent)
        for _ in range(chunk_frames):
            sim.step(3)
        _child = sim.snapshot()
    dt = time.perf_counter() - t0
    sim.close()
    return {
        "chunk_frames": chunk_frames,
        "n_nodes": n,
        "us_per_node": round(dt / n * 1e6, 2),
        "nodes_per_s": round(n / dt, 1),
    }


def main() -> None:
    fp = env_fingerprint()
    step = bench_step_rate()
    snap = bench_snapshot()
    node = estimate_search_node()

    write_json_atomic(BENCH / "step_rate.json",
                      {"generated_at": utc_now_iso(), "env": fp, **step})
    write_json_atomic(BENCH / "snapshot_cost.json",
                      {"generated_at": utc_now_iso(), "env": fp, **snap,
                       "search_node": node})

    print("step rate   :", step["fps"], "fps  (", step["us_per_frame"], "us/frame )")
    print("snapshot    : dump", snap["dump_us"], "us | load", snap["load_us"],
          "us | roundtrip", snap["roundtrip_us"], "us")
    print("search node :", node["nodes_per_s"], "nodes/s  (", node["us_per_node"],
          "us/node, chunk=", node["chunk_frames"], ")")


if __name__ == "__main__":
    main()
