"""Probe SMA4 1-3 white/blue-block duck → behind-background → whistle path.

Uses the cached 1-3 entry snapshot plus the prior P-speed beam path that reached
x≈2546 (``runs/20260627-113530-sma4_snapshot_1_3_pspeed/``), then sweeps LEFT
from late-level checkpoints while holding DOWN and watching:

  0x03003D06 — behind-the-background timer (Data Crystal)

Artifacts under runs/<ts>-sma4-whiteblock-probe/

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/probe_sma4_whiteblock.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pickle
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.adapters import SMA4Adapter, SMA4_PSPEED_ACTIONS  # noqa: E402

BEHIND_BG = 0x03003D06
INVENTORY_START = 0x03002C2E
WARP_WHISTLE = 0x0C
WATCH = list(range(0x03003CF0, 0x03003D20))
IWRAM_LO = 0x03003C80
IWRAM_HI = 0x03004020
PATH_JSON = Path("runs/20260627-113530-sma4_snapshot_1_3_pspeed/attempt_summary_interrupted.json")
ENTRY = Path("runs/sma4_cache/1-3_entry.pkl")


def _load_snapshot(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def _u8(a: SMA4Adapter, addr: int) -> int:
    return a._read_u8(addr)


def _inventory(a: SMA4Adapter) -> list[int]:
    return [_u8(a, INVENTORY_START + i) for i in range(8)]


def _shot(a: SMA4Adapter, out_dir: Path, label: str) -> str:
    out = out_dir / f"{label}.png"
    Image.fromarray(a.last_obs).save(out)
    return str(out)


def _watch(a: SMA4Adapter) -> dict:
    return {f"{addr:#010x}": _u8(a, addr) for addr in WATCH}


def _diff_block(before: bytes, after: bytes, base: int) -> list[dict]:
    return [
        {"addr": f"{base + i:#010x}", "before": b0, "after": b1}
        for i, (b0, b1) in enumerate(zip(before, after))
        if b0 != b1
    ]


def _sample(a: SMA4Adapter, label: str, frame: int) -> dict:
    info = a.last_info
    return {
        "label": label,
        "frame": frame,
        "x_pos": int(info.get("x_pos", 0)),
        "y_pos": int(info.get("y_pos", 0)),
        "mode": info.get("mode"),
        "powerup": int(info.get("powerup", 0)),
        "pspeed": int(info.get("pspeed", 0)),
        "endwalk": int(info.get("endwalk", 0)),
        "behind_bg": _u8(a, BEHIND_BG),
        "inventory_first8": _inventory(a),
        "watch": _watch(a),
    }


def _replay_to_x(adapter: SMA4Adapter, path: list[int], chunk_frames: int,
                 target_x: int) -> tuple[int, object | None]:
    """Replay path chunks until x_pos >= target_x; return (frames, snap|None)."""
    frames = 0
    best_snap = None
    best_x = -1
    for action_idx in path:
        adapter.run_chunk(action_idx, chunk_frames)
        frames += chunk_frames
        x = int(adapter.last_info.get("x_pos", 0))
        if x > best_x and not adapter.is_death(adapter.last_info, False):
            best_x = x
            best_snap = adapter.snapshot()
        if x >= target_x and not adapter.is_death(adapter.last_info, False):
            return frames, adapter.snapshot()
        if adapter.is_death(adapter.last_info, False) or adapter.is_success(adapter.last_info):
            break
    return frames, best_snap


def _try_duck(adapter: SMA4Adapter, out_dir: Path, label: str) -> dict:
    before_watch = _watch(adapter)
    before_block = adapter._read_block(IWRAM_LO, IWRAM_HI - IWRAM_LO)
    # Ground settle.
    for _ in range(10):
        adapter._step_buttons(())
    peak_watch_vals: dict[str, int] = {f"{a:#010x}": 0 for a in WATCH}
    activated_at = None
    for dframe in range(1, 420):
        adapter._step_buttons(("DOWN",))
        w = _watch(adapter)
        for k, v in w.items():
            if v > peak_watch_vals[k]:
                peak_watch_vals[k] = v
        bg = w[f"{BEHIND_BG:#010x}"]
        if bg and activated_at is None:
            activated_at = dframe
            after_block = adapter._read_block(IWRAM_LO, IWRAM_HI - IWRAM_LO)
            shot = _shot(adapter, out_dir, f"behind_{label}_f{dframe}")
            # Sprint right for Toad house / secret exit.
            post = []
            whistle = False
            for rframe in range(1, 1200):
                adapter._step_buttons(("RIGHT", "B"))
                if rframe % 40 == 0:
                    post.append(_sample(adapter, f"behind_run_{label}", rframe))
                inv = _inventory(adapter)
                mode = adapter.last_info.get("mode")
                if WARP_WHISTLE in inv:
                    whistle = True
                    post.append(_sample(adapter, f"whistle_{label}", rframe))
                    _shot(adapter, out_dir, f"whistle_{label}")
                    break
                if mode in ("menu", "overworld") and rframe > 30:
                    post.append(_sample(adapter, f"mode_{mode}_{label}", rframe))
                    _shot(adapter, out_dir, f"mode_{mode}_{label}")
                    # Open chest with A if in toad house-like menu.
                    for _ in range(90):
                        adapter._step_buttons(("A",))
                        if WARP_WHISTLE in _inventory(adapter):
                            whistle = True
                            post.append(_sample(adapter, f"whistle_after_a_{label}", rframe))
                            _shot(adapter, out_dir, f"whistle_after_a_{label}")
                            break
                    break
                if adapter.is_death(adapter.last_info, False):
                    post.append(_sample(adapter, f"died_behind_{label}", rframe))
                    break
            return {
                "label": label,
                "activated": True,
                "dframe": activated_at,
                "x_pos": int(adapter.last_info.get("x_pos", 0)),
                "y_pos": int(adapter.last_info.get("y_pos", 0)),
                "behind_bg": bg,
                "watch_before": before_watch,
                "watch_at_activate": w,
                "iwram_diff": _diff_block(before_block, after_block, IWRAM_LO)[:100],
                "shot": shot,
                "whistle": whistle,
                "inventory": _inventory(adapter),
                "post_samples": post[-20:],
            }
    return {
        "label": label,
        "activated": False,
        "dframe": 420,
        "x_pos": int(adapter.last_info.get("x_pos", 0)),
        "y_pos": int(adapter.last_info.get("y_pos", 0)),
        "behind_bg": _u8(adapter, BEHIND_BG),
        "watch_before": before_watch,
        "watch_after": _watch(adapter),
        "peak_watch": peak_watch_vals,
    }


def main() -> int:
    if not os.environ.get("MARIO_AI_SMA4_ROM"):
        raise SystemExit("MARIO_AI_SMA4_ROM required")
    if not ENTRY.exists() or not PATH_JSON.exists():
        raise SystemExit(f"need {ENTRY} and {PATH_JSON}")

    path_doc = json.loads(PATH_JSON.read_text())
    path = list(path_doc["path"])
    chunk_frames = int(path_doc["chunk_frames"])

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S-sma4-whiteblock-probe")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = SMA4Adapter(actions=SMA4_PSPEED_ACTIONS, boot_target="overworld")
    samples: list[dict] = []
    duck_hits: list[dict] = []
    try:
        adapter.level_id = "1-3"
        entry = _load_snapshot(ENTRY)
        checkpoints = [1600, 1800, 2000, 2100, 2200, 2300, 2400, 2500]
        for target_x in checkpoints:
            adapter.restore(entry)
            frames, snap = _replay_to_x(adapter, path, chunk_frames, target_x)
            if snap is None:
                duck_hits.append({"label": f"cp_{target_x}", "error": "no_snap", "frames": frames})
                continue
            adapter.restore(snap)
            x0 = int(adapter.last_info.get("x_pos", 0))
            samples.append(_sample(adapter, f"checkpoint_{target_x}", frames))
            _shot(adapter, out_dir, f"cp_{target_x}_x{x0}")
            base_snap = adapter.snapshot()

            # At checkpoint: try duck in place, then walk left in steps and duck.
            offsets = [("here", 0)] + [(f"left_{n}", n) for n in range(8, 280, 8)]
            for off_label, left_n in offsets:
                adapter.restore(base_snap)
                for _ in range(left_n):
                    adapter._step_buttons(("LEFT",))
                # Brief settle / small right nudge to land on a block top.
                for _ in range(6):
                    adapter._step_buttons(())
                label = f"cp{target_x}_{off_label}_x{int(adapter.last_info.get('x_pos', 0))}"
                hit = _try_duck(adapter, out_dir, label)
                duck_hits.append(hit)
                if hit.get("activated"):
                    samples.append(_sample(adapter, f"activated_{label}", 0))
                    # Continue sweeping a bit more for robustness, but stop early
                    # once we also get a whistle.
                    if hit.get("whistle"):
                        break
            if any(h.get("whistle") for h in duck_hits):
                break

        report = {
            "rom": os.environ["MARIO_AI_SMA4_ROM"],
            "snapshot": str(ENTRY),
            "path_json": str(PATH_JSON),
            "behind_bg_addr": hex(BEHIND_BG),
            "activated": any(h.get("activated") for h in duck_hits),
            "whistle_acquired": any(h.get("whistle") for h in duck_hits),
            "n_duck_trials": len(duck_hits),
            "duck_hits_activated": [h for h in duck_hits if h.get("activated")],
            "duck_hits_sample_negatives": [h for h in duck_hits if not h.get("activated")][:12],
            "samples_tail": samples[-40:],
            "screenshots": sorted(p.name for p in out_dir.glob("*.png")),
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "out_dir": str(out_dir),
            "activated": report["activated"],
            "whistle": report["whistle_acquired"],
            "n_trials": report["n_duck_trials"],
            "n_activated": len(report["duck_hits_activated"]),
        }, indent=2))
        return 0 if report["activated"] else 2
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
