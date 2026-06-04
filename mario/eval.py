"""Evaluate a trained policy: roll it out across seeds and emit eval.json + a contact
sheet, reusing the V0 outcome fields so the verification channel is identical.

Eval runs on CPU for reproducibility (the mps_parity gate guarantees CPU and MPS pick the
same actions, so this loses nothing and avoids any MPS rollout nondeterminism).

    ./venv/bin/python -m mario.eval <checkpoint_run_id_or_path> [n_seeds]
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

from mario.artifacts import framerule_time, validate_eval
from mario.env import MarioSim
from mario.io import (RUNS, env_fingerprint, git_rev, new_run_id, run_dir,
                      utc_now_iso, write_json_atomic)
from mario.policy import Controller, load_policy
from mario.render import make_contact_sheet
from mario.reward import death_cause, is_death, is_success

LEVEL_LENGTHS = {(1, 1): 3161}   # fallback; primary source is the manifest backbone_run
PASS_THRESHOLD = 0.90


def level_length(world: int, stage: int, fallback: int) -> int:
    """Level length (flag x) from the dataset manifest's backbone solve; fallback otherwise."""
    import json
    mp = Path(__file__).resolve().parent.parent / "data" / "manifest.json"
    if mp.exists():
        lv = json.loads(mp.read_text()).get("levels", {}).get(f"{world}-{stage}", {})
        x = lv.get("backbone_run", {}).get("x_at_flag", 0)
        if x and x > 0:
            return int(x)
    return LEVEL_LENGTHS.get((world, stage), max(fallback, 1))


def rollout(controller: Controller, world: int, stage: int, seed: int,
            max_chunks: int = 400):
    sim = MarioSim(world, stage)
    info = sim.reset(seed=seed)
    controller.reset()
    path, x_max, frames, done = [], int(info.get("x_pos", 0)), 0, False
    for _ in range(max_chunks):
        a = controller.act(sim.ram, sim.last_info)
        path.append(a)
        info, done = sim.run_chunk(a, controller.chunk_frames)
        frames += controller.chunk_frames
        x_max = max(x_max, int(info.get("x_pos", 0)))
        if done:
            break
    sim.close()
    beat = is_success(info)
    died = is_death(info, done)
    length = level_length(world, stage, x_max)
    outcome = {
        "beat_level": beat, "death": died,
        "death_cause": death_cause(info) if died else None,
        "x_pos_reached": int(info.get("x_pos", 0)), "x_pos_max": int(x_max),
        "level_length": length,
        "completion_frac": 1.0 if beat else round(min(1.0, x_max / length), 4),
        "frames": frames, "framerule_time": framerule_time(frames),
        "deaths": 1 if died else 0,
    }
    return outcome, path


def eval_suite(checkpoint_path, levels, seeds, *, milestone="V2") -> dict:
    net, ckpt = load_policy(checkpoint_path, device="cpu")
    controller = Controller(net, device="cpu", chunk_frames=ckpt["chunk_frames"])
    ck_run = Path(checkpoint_path).parent.name

    out_levels, best_for_sheet = {}, None
    total_beat = total = 0
    for (world, stage) in levels:
        rs = [rollout(controller, world, stage, s) for s in seeds]
        outs = [o for o, _p in rs]
        n_beat = sum(o["beat_level"] for o in outs)
        beaten_fr = [o["framerule_time"] for o in outs if o["beat_level"]]
        causes: dict[str, int] = {}
        for o in outs:
            if o["death_cause"]:
                causes[o["death_cause"]] = causes.get(o["death_cause"], 0) + 1
        out_levels[f"{world}-{stage}"] = {
            "completion_rate": round(n_beat / len(seeds), 4),
            "n_beat": n_beat, "n_rollouts": len(seeds),
            "median_framerule_time": int(statistics.median(beaten_fr)) if beaten_fr else None,
            "median_completion_frac": round(statistics.median(o["completion_frac"] for o in outs), 4),
            "deaths_by_cause": causes,
            "seeds": list(seeds),
        }
        total_beat += n_beat
        total += len(seeds)
        # remember a beaten rollout (else furthest) for the contact sheet — keep the
        # rollout's actual seed so the deterministic replay reproduces that path
        beaten = [(seeds[i], rs[i][1]) for i in range(len(rs)) if rs[i][0]["beat_level"]]
        if beaten:
            best_for_sheet = (world, stage, beaten[0][0], beaten[0][1])
        elif best_for_sheet is None:
            i = max(range(len(rs)), key=lambda j: rs[j][0]["x_pos_max"])
            best_for_sheet = (world, stage, seeds[i], rs[i][1])

    mean_cr = round(sum(lv["completion_rate"] for lv in out_levels.values()) / len(out_levels), 4)
    worst = min(lv["completion_rate"] for lv in out_levels.values())
    worst_lvl = min(out_levels, key=lambda k: out_levels[k]["completion_rate"])
    passed = worst >= PASS_THRESHOLD   # ALL evaluated levels must clear the bar

    run_id = new_run_id("v2_eval")
    d = run_dir(run_id)
    sheet_rel = None
    if best_for_sheet:
        w, s, sd, p = best_for_sheet
        try:
            make_contact_sheet(w, s, sd, p, ckpt["chunk_frames"], d / "contact_sheet.png")
            sheet_rel = "contact_sheet.png"
        except Exception as e:
            print("WARN contact sheet:", e)

    parity = {}
    pp = Path(__file__).resolve().parent.parent / "bench" / "mps_parity.json"
    if pp.exists():
        import json
        parity = json.loads(pp.read_text())

    ev = {
        "run_id": run_id, "schema_version": 1, "kind": "eval", "milestone": milestone,
        "git_rev": git_rev(), "generated_at": utc_now_iso(),
        "env_fingerprint": env_fingerprint(),
        "policy": {"checkpoint_run_id": ck_run, "K": ckpt["arch"]["K"],
                   "chunk_frames": ckpt["chunk_frames"]},
        "levels": out_levels,
        "aggregate": {"mean_completion_rate": mean_cr, "total_beat": total_beat,
                      "total_rollouts": total},
        "mps_parity": parity,
        "artifacts": {"contact_sheet": sheet_rel} if sheet_rel else {},
        "pass": bool(passed),
        "pass_reason": f"worst level {worst_lvl} completion_rate {worst} "
                       f"{'>=' if passed else '<'} {PASS_THRESHOLD}",
    }
    validate_eval(ev)
    write_json_atomic(d / "eval.json", ev)
    return ev


def _resolve_checkpoint(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    cand = RUNS / arg / "checkpoint.pt"
    if cand.is_file():
        return cand
    raise SystemExit(f"checkpoint not found: {arg}")


def parse_levels(arg: str) -> list[tuple[int, int]]:
    """'1-1' or '1-1,1-2,1-3' -> [(1,1),(1,2),(1,3)]."""
    out = []
    for tok in arg.split(","):
        w, s = tok.strip().split("-")
        out.append((int(w), int(s)))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m mario.eval <checkpoint> [n_seeds] [levels e.g. 1-1,1-2]")
    ckpt = _resolve_checkpoint(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    levels = parse_levels(sys.argv[3]) if len(sys.argv) > 3 else [(1, 1)]
    ev = eval_suite(ckpt, levels, list(range(n_seeds)))
    for key, lv in ev["levels"].items():
        print(f"{key}: completion_rate={lv['completion_rate']} ({lv['n_beat']}/{lv['n_rollouts']}) "
              f"median_framerule={lv['median_framerule_time']} deaths={lv['deaths_by_cause']}")
    print(f"GATE: {'PASS' if ev['pass'] else 'FAIL'} — {ev['pass_reason']}")
    print(f"eval.json: runs/{ev['run_id']}/eval.json")


if __name__ == "__main__":
    main()
