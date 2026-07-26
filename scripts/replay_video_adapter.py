"""Render a solved adapter-game trajectory (SMA4/SML) to an MP4.

Loads a solution JSON, rebuilds the matching adapter, replays the path frame by
frame capturing the rendered screen, and assembles an upscaled video via the
system ffmpeg.  The SMB1 equivalent is `scripts/replay_video.py`.

    MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/replay_video_adapter.py data/solutions/sma4/1-1.json
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.adapters import (  # noqa: E402
    SMA4Adapter,
    SMA4_PHYSICS_ACTIONS,
    SMA4_PSPEED_ACTIONS,
    SMLAdapter,
    SML_SURFACE_ACTIONS,
)

SCALE = 3
FPS = 60


def _build_adapter(spec: dict):
    game = spec.get("game_id")
    world, stage = (int(x) for x in spec.get("level_id", "1-1").split("-"))
    if game == "sma4":
        action_set = spec.get("action_set", "surface")
        if action_set == "surface":
            actions = None
        elif action_set == "physics":
            actions = SMA4_PHYSICS_ACTIONS
        elif action_set == "pspeed":
            actions = SMA4_PSPEED_ACTIONS
        else:
            raise SystemExit(f"unsupported SMA4 action_set {action_set!r}")
        root_kind = (spec.get("replay_root") or {}).get("kind")
        boot_target = "overworld" if root_kind == "snapshot" else f"{world}-{stage}"
        adapter = SMA4Adapter(
            world=world, stage=stage, actions=actions, boot_target=boot_target)
        adapter.level_id = f"{world}-{stage}"
    elif game == "sml":
        adapter = SMLAdapter(
            world=world, stage=stage, actions=SML_SURFACE_ACTIONS)
    else:
        raise SystemExit(f"unsupported game_id {game!r}")
    expected_actions = spec.get("action_names")
    if expected_actions and list(expected_actions) != list(adapter.action_names):
        adapter.close()
        raise SystemExit(
            "solution action_names do not match the selected adapter vocabulary")
    return adapter


def capture_replay_frames(
        adapter, path: list[int], chunk_frames: int, *, seed: int = 0,
        root_snapshot=None):
    """Replay until the first terminal success/death, across chunk boundaries."""
    if root_snapshot is None:
        info = adapter.reset(seed=seed)
    else:
        adapter.restore(root_snapshot)
        info = adapter.last_info
        # A standalone snapshot is the declared replay root.  Rebase the
        # adapter's life-loss detector to that root; otherwise a snapshot taken
        # after an earlier life loss can be declared dead before frame 1.
        if hasattr(adapter, "_start_lives") and "lives" in info:
            adapter._start_lives = int(info["lives"])
    frames = [np.asarray(adapter.last_obs).copy()]
    terminal = bool(adapter.is_success(info) or adapter.is_death(info, False))
    for action in path:
        if terminal:
            break
        for _ in range(chunk_frames):
            _obs, info, done = adapter.step(action)
            frames.append(np.asarray(adapter.last_obs).copy())
            terminal = bool(
                done
                or adapter.is_success(info)
                or adapter.is_death(info, done)
            )
            if terminal:
                break
    return frames


def _load_snapshot(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict):
        for key in ("entry_snap", "snapshot", "snap"):
            if key in obj:
                return obj[key]
    return obj


def _declared_root(spec: dict, adapter):
    root = spec.get("replay_root")
    if not isinstance(root, dict) or not root:
        raise SystemExit(
            "solution has no declared replay_root; regenerate or migrate the "
            "manifest before rendering it as verified evidence")
    expected_rom = root.get("rom_sha1") or spec.get("rom_sha1")
    actual_rom = getattr(adapter, "rom_sha1", None)
    if expected_rom and actual_rom != expected_rom:
        raise SystemExit(
            f"ROM hash mismatch: manifest={expected_rom}, adapter={actual_rom}")
    kind = root.get("kind")
    if kind == "fresh_boot":
        return None
    if kind != "snapshot":
        raise SystemExit(f"unsupported replay_root kind {kind!r}")
    snapshot_path = Path(root.get("path", ""))
    if not snapshot_path.is_absolute():
        snapshot_path = ROOT / snapshot_path
    if not snapshot_path.is_file():
        raise SystemExit(f"declared replay snapshot is unavailable: {snapshot_path}")
    expected_hash = root.get("sha256")
    actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise SystemExit(
            f"snapshot hash mismatch: manifest={expected_hash}, actual={actual_hash}")
    return _load_snapshot(snapshot_path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: replay_video_adapter.py <solution.json> [out.mp4]")
    sol_path = Path(sys.argv[1])
    spec = json.loads(sol_path.read_text())
    path, cf = spec["path"], spec.get("chunk_frames", 1)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else sol_path.with_suffix(".mp4")

    adapter = _build_adapter(spec)
    try:
        root_snapshot = _declared_root(spec, adapter)
        frames = capture_replay_frames(
            adapter, path, cf, seed=spec.get("seed", 0),
            root_snapshot=root_snapshot)
    finally:
        adapter.close()
    print(f"captured {len(frames)} frames")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            img = Image.fromarray(f.astype(np.uint8))
            img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
            img.save(td / f"f{i:05d}.png")
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(td / "f%05d.png"),
               "-pix_fmt", "yuv420p", "-an", str(out)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out)


if __name__ == "__main__":
    main()
