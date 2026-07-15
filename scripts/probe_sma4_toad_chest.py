"""Reproduce 1-3 white-block → Toad house and open the whistle chest with B.

Uses the cached P-Wing entry snapshot. Artifacts under
``runs/<ts>-sma4-toad-chest/``.

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/probe_sma4_toad_chest.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PSPEED_ACTIONS  # noqa: E402

BEHIND_BG = 0x03003D06
INVENTORY_START = 0x03002C2E
WARP_WHISTLE = 0x0C
ENTRY = Path("runs/sma4_cache/1-3_pwing_entry.pkl")


def _load(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def _step(a: SMA4Adapter, buttons=()):
    a._step_buttons(buttons)


def _inv(a: SMA4Adapter, n: int = 36) -> list[int]:
    return [a._read_u8(INVENTORY_START + i) for i in range(n)]


def _bright(a: SMA4Adapter) -> float:
    return float(np.asarray(a.last_obs).mean())


def _save(a: SMA4Adapter, out: Path, name: str):
    Image.fromarray(a.last_obs).save(out / name)


def _scan_whistle(a: SMA4Adapter) -> list[str]:
    hits = []
    for base, size in ((0x03002C00, 0x400), (0x03003000, 0x400)):
        for i in range(size):
            if a._read_u8(base + i) == WARP_WHISTLE:
                hits.append(hex(base + i))
    return sorted(set(hits))


def _iwram_nonzero(a: SMA4Adapter, start: int = 0x03002C00, size: int = 0x200) -> dict:
    out = {}
    for i in range(size):
        v = a._read_u8(start + i)
        if v:
            out[hex(start + i)] = v
    return out


def fly_to_white(a: SMA4Adapter, out: Path, target_x: int = 1680) -> None:
    for f in range(3000):
        btns = ("RIGHT", "A", "B") if (f % 4) < 2 else ("RIGHT", "B")
        _step(a, btns)
        if int(a.last_info["x_pos"]) >= target_x:
            _save(a, out, "01_arrive.png")
            return
        if a.is_death(a.last_info, False):
            raise RuntimeError("died during flight")
    raise RuntimeError(f"flight stalled at x={a.last_info['x_pos']}")


def land_on_white(a: SMA4Adapter, out: Path) -> bool:
    """Brake/descend until grounded on the white block as powered Mario."""
    arrive = a.snapshot()
    # Soft descent: release A, tap LEFT to kill speed, keep y elevated.
    for recipe_i, (left_every, n_frames, wait) in enumerate([
        (2, 90, 40),
        (2, 70, 60),
        (3, 100, 50),
        (2, 110, 30),
        (1, 40, 80),
    ]):
        a.restore(arrive)
        for i in range(n_frames):
            if left_every and i % left_every == 0:
                _step(a, ("LEFT",))
            else:
                _step(a, ())
        for _ in range(wait):
            _step(a, ())
        # Extra settle without lateral input.
        for _ in range(40):
            spd = int(a.last_info.get("speed_signed", 0))
            if abs(spd) <= 2 and 45 <= int(a.last_info["y_pos"]) <= 80:
                break
            if spd > 2:
                _step(a, ("LEFT",))
            elif spd < -2:
                _step(a, ("RIGHT",))
            else:
                _step(a, ())
        x, y, pow_ = (int(a.last_info["x_pos"]), int(a.last_info["y_pos"]),
                      int(a.last_info["powerup"]))
        _save(a, out, f"02_land_r{recipe_i}_x{x}_y{y}_p{pow_}.png")
        print(f" land recipe {recipe_i}: x={x} y={y} pow={pow_} "
              f"spd={a.last_info.get('speed_signed')}", flush=True)
        if pow_ >= 1 and 45 <= y <= 80 and 1620 <= x <= 1780:
            _save(a, out, "02_on_white.png")
            return True
    return False


def activate_behind(a: SMA4Adapter, out: Path) -> int | None:
    base = a.snapshot()
    for i in range(400):
        _step(a, ("DOWN",))
        bg = a._read_u8(BEHIND_BG)
        pow_ = int(a.last_info["powerup"])
        y = int(a.last_info["y_pos"])
        if i % 40 == 0:
            print(f" duck {i}: bg={bg} y={y} pow={pow_} x={a.last_info['x_pos']}",
                  flush=True)
        if bg > 0:
            _save(a, out, f"03_ACT_f{i}_bg{bg}.png")
            print(f" ACTIVATE f={i} bg={bg} x={a.last_info['x_pos']} y={y} pow={pow_}",
                  flush=True)
            return i
        if pow_ < 1 or y > 100:
            # Koopa/damage or fell off — abort this attempt.
            break
    a.restore(base)
    return None


def sprint_to_toad(a: SMA4Adapter, out: Path) -> bool:
    prev_x = int(a.last_info["x_pos"])
    for r in range(700):
        _step(a, ("RIGHT", "B"))
        x = int(a.last_info["x_pos"])
        y = int(a.last_info["y_pos"])
        bg = a._read_u8(BEHIND_BG)
        mode = a.last_info.get("mode")
        b = _bright(a)
        if r % 40 == 0:
            print(f" run {r}: x={x} y={y} bg={bg} mode={mode} bright={b:.1f}",
                  flush=True)
            _save(a, out, f"04_run_{r:03d}_x{x}_bg{bg}.png")
        # Real room change: x collapses from the end-of-level region.
        if prev_x >= 2000 and x < 200:
            print(f" TRANSITION x-collapse r={r} {prev_x}->{x} bright={b:.1f}",
                  flush=True)
            _save(a, out, "05_trans.png")
            return True
        if mode == "menu" and x < 200 and r > 40:
            print(f" TRANSITION mode=menu r={r} x={x}", flush=True)
            _save(a, out, "05_trans.png")
            return True
        # Death / kickout guard.
        if a.is_death(a.last_info, False) or (mode == "overworld" and r > 20):
            print(f" FAIL death/overworld r={r} x={x} mode={mode}", flush=True)
            _save(a, out, "FAIL_kickout.png")
            return False
        prev_x = x
    print(" FAIL no transition", flush=True)
    return False


def wait_fade(a: SMA4Adapter, out: Path) -> None:
    for w in range(400):
        _step(a, ())
        b = _bright(a)
        if w % 20 == 0:
            print(f" fade {w}: bright={b:.1f} x={a.last_info['x_pos']} "
                  f"y={a.last_info['y_pos']} mode={a.last_info['mode']}", flush=True)
        if b > 40 and int(a.last_info.get("x_pos", 0)) < 200:
            _save(a, out, "06_house.png")
            print(f" FADED IN w={w} bright={b:.1f}", flush=True)
            return
    _save(a, out, "06_house_timeout.png")


def try_open_chest(a: SMA4Adapter, out: Path, house_snap) -> dict | None:
    pre_ram = _iwram_nonzero(a)
    (out / "pre_house_ram.json").write_text(json.dumps(pre_ram, indent=2) + "\n")

    # Clear dialogue with slow A pulses first (common blocker).
    for clear_a in (0, 40, 80, 120):
        for walk in list(range(0, 160, 5)):
            for open_btn in (("B",), ("A",), ("UP", "B"), ("DOWN", "B")):
                a.restore(house_snap)
                for i in range(clear_a):
                    _step(a, ("A",) if i % 3 == 0 else ())
                for _ in range(walk):
                    _step(a, ("RIGHT",))
                x0, y0 = int(a.last_info["x_pos"]), int(a.last_info["y_pos"])
                for p in range(12):
                    _step(a, open_btn)
                    for _ in range(6):
                        _step(a, ())
                    hits = _scan_whistle(a)
                    inv = _inv(a)
                    if WARP_WHISTLE in inv or hits:
                        info = {
                            "clear_a": clear_a,
                            "walk": walk,
                            "open_btn": list(open_btn),
                            "pulse": p,
                            "x": x0,
                            "y": y0,
                            "inv": inv,
                            "hits": hits,
                            "mode": a.last_info.get("mode"),
                            "post_ram_whistle": hits,
                        }
                        _save(a, out, "WHISTLE.png")
                        (out / "SUCCESS.json").write_text(json.dumps(info, indent=2) + "\n")
                        print(" SUCCESS", info, flush=True)
                        return info
            # Progress breadcrumb every few walks.
            if walk % 40 == 0:
                a.restore(house_snap)
                for i in range(clear_a):
                    _step(a, ("A",) if i % 3 == 0 else ())
                for _ in range(walk):
                    _step(a, ("RIGHT",))
                _save(a, out, f"pos_c{clear_a}_w{walk}_x{a.last_info['x_pos']}.png")
                print(f" probed clear={clear_a} walk={walk} "
                      f"x={a.last_info['x_pos']} y={a.last_info['y_pos']} "
                      f"inv={_inv(a)[:4]}", flush=True)

    # Jump-on-chest + B
    for walk in range(40, 130, 5):
        for hold in (3, 6, 10, 14):
            a.restore(house_snap)
            for i in range(80):
                _step(a, ("A",) if i % 3 == 0 else ())
            for _ in range(walk):
                _step(a, ("RIGHT",))
            for _ in range(hold):
                _step(a, ("A",))
            for _ in range(25):
                _step(a, ())
            for _ in range(15):
                _step(a, ("B",))
                for _ in range(4):
                    _step(a, ())
                hits = _scan_whistle(a)
                if WARP_WHISTLE in _inv(a) or hits:
                    info = {
                        "method": "jump_b",
                        "walk": walk,
                        "hold": hold,
                        "inv": _inv(a),
                        "hits": hits,
                        "x": a.last_info["x_pos"],
                        "y": a.last_info["y_pos"],
                    }
                    _save(a, out, "WHISTLE_jump.png")
                    (out / "SUCCESS.json").write_text(json.dumps(info, indent=2) + "\n")
                    print(" SUCCESS JUMP", info, flush=True)
                    return info

    # Final: mash everything at chest center-ish while dumping RAM diffs.
    a.restore(house_snap)
    for i in range(100):
        _step(a, ("A",) if i % 3 == 0 else ())
    for _ in range(70):
        _step(a, ("RIGHT",))
    _save(a, out, "07_at_chest.png")
    before = _iwram_nonzero(a)
    for name, seq in [
        ("Bhold", [("B",)] * 40),
        ("Ahold", [("A",)] * 40),
        ("Bpulse", sum(([(("B",)), (())] for _ in range(30)), [])),
        ("UPpulse", sum(([(("UP",)), (("B",)), (())] for _ in range(20)), [])),
        ("STARTB", sum(([(("START",)), (()), (("B",)), (())] for _ in range(10)), [])),
        ("Lb", sum(([(("L",)), (()), (("B",)), (())] for _ in range(10)), [])),
    ]:
        a.restore(house_snap)
        for i in range(100):
            _step(a, ("A",) if i % 3 == 0 else ())
        for _ in range(70):
            _step(a, ("RIGHT",))
        for bt in seq:
            _step(a, bt)
        after = _iwram_nonzero(a)
        diff = {k: after[k] for k in after if before.get(k) != after[k]}
        diff.update({k: 0 for k in before if k not in after})
        hits = _scan_whistle(a)
        _save(a, out, f"08_{name}.png")
        print(f" soup {name}: inv={_inv(a)[:8]} hits={hits} diff_n={len(diff)}",
              flush=True)
        (out / f"ramdiff_{name}.json").write_text(json.dumps({
            "inv": _inv(a), "hits": hits, "diff": diff,
        }, indent=2) + "\n")
        if WARP_WHISTLE in _inv(a) or hits:
            info = {"method": name, "inv": _inv(a), "hits": hits, "diff": diff}
            (out / "SUCCESS.json").write_text(json.dumps(info, indent=2) + "\n")
            return info
    return None


def main() -> int:
    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM required")
    if not ENTRY.exists():
        raise SystemExit(f"missing {ENTRY}")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S-sma4-toad-chest")
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    a = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    try:
        a.level_id = "1-3"
        a.restore(_load(ENTRY))
        print("ENTRY", {k: a.last_info.get(k) for k in
                        ("x_pos", "y_pos", "powerup", "pspeed", "lives")}, flush=True)
        _save(a, out, "00_entry.png")

        fly_to_white(a, out)
        print("ARRIVE", a.last_info["x_pos"], a.last_info["y_pos"],
              "pow", a.last_info["powerup"], flush=True)

        if not land_on_white(a, out):
            (out / "report.json").write_text(json.dumps({"ok": False, "stage": "land"}) + "\n")
            return 2
        print("ON WHITE", a.last_info["x_pos"], a.last_info["y_pos"],
              "pow", a.last_info["powerup"], flush=True)

        act = activate_behind(a, out)
        if act is None:
            (out / "report.json").write_text(json.dumps({"ok": False, "stage": "duck"}) + "\n")
            return 3

        if not sprint_to_toad(a, out):
            (out / "report.json").write_text(json.dumps({
                "ok": False, "stage": "sprint", "act_frame": act,
            }) + "\n")
            return 4

        wait_fade(a, out)
        house = a.snapshot()
        with open(out / "house_snap.pkl", "wb") as f:
            pickle.dump({"snap": house, "info": dict(a.last_info), "inv": _inv(a)}, f)
        print("HOUSE", {k: a.last_info.get(k) for k in
                        ("x_pos", "y_pos", "mode", "time", "powerup")},
              "inv", _inv(a)[:8], "hits", _scan_whistle(a), flush=True)

        result = try_open_chest(a, out, house)
        report = {
            "ok": bool(result),
            "act_frame": act,
            "house_info": dict(a.last_info),
            "result": result,
            "out_dir": str(out),
        }
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0 if result else 5
    finally:
        a.close()


if __name__ == "__main__":
    raise SystemExit(main())
