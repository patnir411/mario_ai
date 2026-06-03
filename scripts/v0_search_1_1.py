"""V0 milestone: beat World 1-1 with pure beam search (no neural net).

Emits the full artifact set so the result is objectively verifiable:
  runs/<id>/result.json        — outcome (beat_level, deaths, completion, framerule_time)
  runs/<id>/trajectory.jsonl   — per-chunk trace
  runs/<id>/contact_sheet.png  — annotated key frames (flagpole visible iff beaten)

PASS condition: outcome.beat_level == true AND outcome.deaths == 0.

    ./venv/bin/python scripts/v0_search_1_1.py [beam_width] [max_depth]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.artifacts import build_result, framerule_time, write_result  # noqa: E402
from mario.env import MarioSim  # noqa: E402
from mario.io import new_run_id, run_dir, write_json_atomic  # noqa: E402
from mario.ram import MARIO_X_SPEED, signed  # noqa: E402
from mario.render import make_contact_sheet  # noqa: E402
from mario.reward import DEFAULT, is_death, is_success  # noqa: E402
from mario.search import beam_search  # noqa: E402

WORLD, STAGE, SEED = 1, 1, 0
LEVEL_LENGTH = 3161  # approx flagpole x in 1-1 (for completion_frac when not solved)


def write_trajectory(path: list[int], chunk_frames: int, dest: Path) -> int:
    """Replay the path, logging one JSON row per chunk; return deaths counted."""
    sim = MarioSim(WORLD, STAGE)
    info = sim.reset(seed=SEED)
    deaths = 0
    with open(dest, "w") as f:
        for i, a in enumerate(path, start=1):
            info, done = sim.run_chunk(a, chunk_frames)
            row = {
                "t": i, "x_pos": info.get("x_pos"), "y_pos": info.get("y_pos"),
                "vx": signed(int(sim.ram[MARIO_X_SPEED])), "action": a,
                "time": info.get("time"), "status": info.get("status"),
                "alive": not is_death(info, done), "flag": is_success(info),
            }
            f.write(json.dumps(row) + "\n")
            if is_death(info, done):
                deaths += 1
            if done:
                break
    sim.close()
    return deaths


def main() -> None:
    beam_width = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 180

    run_id = new_run_id("v0_1_1")
    d = run_dir(run_id)
    print(f"[{run_id}] beam_search 1-1  beam_width={beam_width} max_depth={max_depth}")

    res = beam_search(WORLD, STAGE, beam_width=beam_width, chunk_frames=8,
                      max_depth=max_depth, weights=DEFAULT, seed=SEED, progress_every=10)

    # Persist the EXPENSIVE result first so a downstream (e.g. render) bug never wastes
    # a multi-minute search — lesson learned in the first V0 run.
    write_json_atomic(d / "search_path.json",
                      {"path": res.path, "chunk_frames": res.chunk_frames})
    deaths = write_trajectory(res.path, res.chunk_frames, d / "trajectory.jsonl")

    beat = res.solved
    completion = 1.0 if beat else min(1.0, res.x_max / LEVEL_LENGTH)
    outcome = {
        "beat_level": beat,
        "death": deaths > 0,
        "death_cause": None,
        "x_pos_reached": int(res.final_info.get("x_pos", 0)),
        "x_pos_max": int(res.x_max),
        "level_length": LEVEL_LENGTH,
        "completion_frac": round(completion, 4),
        "frames": int(res.frames),
        "framerule_time": framerule_time(int(res.frames)),
        "deaths": deaths,
    }
    passed = beat and deaths == 0
    result = build_result(
        run_id=run_id, kind="search", milestone="V0", world=WORLD, stage=STAGE, seed=SEED,
        outcome=outcome,
        search={"beam_width": beam_width, "max_depth": max_depth, "chunk_frames": 8,
                "nodes_expanded": res.nodes_expanded,
                "wall_clock_s": round(res.wall_clock_s, 2),
                "nodes_per_s": round(res.nodes_expanded / max(res.wall_clock_s, 1e-9), 1),
                "reward_weights": DEFAULT.__dict__},
        artifacts={"contact_sheet": "contact_sheet.png", "trajectory": "trajectory.jsonl"},
        passed=passed,
        pass_reason="beat_level && deaths==0" if passed else f"not solved (x_max={res.x_max})")
    write_result(result)

    # Render last (cosmetic) and guarded — never let it sink the run.
    try:
        sheet = make_contact_sheet(WORLD, STAGE, SEED, res.path, res.chunk_frames,
                                   d / "contact_sheet.png")
        print("contact sheet:", sheet)
    except Exception as e:  # pragma: no cover
        print("WARN: contact sheet render failed:", e)

    print(f"\nRESULT: beat={beat} x_max={res.x_max} deaths={deaths} "
          f"frames={res.frames} framerule_time={outcome['framerule_time']} "
          f"nodes={res.nodes_expanded} wall={res.wall_clock_s:.0f}s PASS={passed}")
    print(f"run dir: {d}")


if __name__ == "__main__":
    main()
