"""Game adapter layer for emulator-backed search.

The existing project started with a single NES SMB1 environment.  This module defines
the narrow contract the search core needs so other Mario games can plug in without
threading game-specific conditionals through the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
from typing import Any, Protocol

from mario.actions import SMB1_ACTIONS
from mario.reward import is_death as smb1_is_death
from mario.reward import is_success as smb1_is_success


def _load_mario_sim():
    """Load the optional NES backend only when an SMB1 adapter is constructed."""
    try:
        from mario.env import MarioSim
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "SMB1Adapter requires the NES emulator stack. Install this project "
            "with `uv pip install -e '.[nes]'` in a dedicated NES environment."
        ) from exc
    return MarioSim


class GameAdapter(Protocol):
    game_id: str
    level_id: str
    action_names: list[str]
    n_actions: int

    def reset(self, seed: int = 0) -> dict: ...
    def close(self) -> None: ...
    def snapshot(self): ...
    def restore(self, snap) -> None: ...
    def step(self, action_idx: int): ...
    def run_chunk(self, action_idx: int, frames: int): ...
    def is_success(self, info: dict) -> bool: ...
    def is_death(self, info: dict, done: bool) -> bool: ...
    def progress(self, info: dict) -> float: ...
    def cell(self, tile: int = 16) -> tuple: ...

    @property
    def ram(self): ...

    @property
    def last_info(self) -> dict: ...

    @property
    def last_obs(self): ...


def _buttons_name(buttons: list[str]) -> str:
    return "NOOP" if not buttons else "+".join(buttons)


def _s8(v: int) -> int:
    v = int(v) & 0xff
    return v - 0x100 if v & 0x80 else v


def _action_name(action) -> str:
    """Human-readable name for a held-button action or short frame macro."""
    if not action:
        return "NOOP"
    first = action[0]
    if isinstance(first, str):
        return _buttons_name(list(action))
    return "/".join(_buttons_name(list(step)) for step in action)


@dataclass(frozen=True)
class SMB1Snapshot:
    """Emulator state plus wrapper caches required for exact adapter restore."""

    emulator_state: Any
    info: dict
    obs: Any


class SMB1Adapter:
    """Adapter shim for the current SMB1 `MarioSim`.

    This deliberately keeps `MarioSim` intact so existing scripts/tests continue to
    import it directly while new generic code can consume the adapter contract.
    """

    game_id = "smb1"

    def __init__(self, world: int = 1, stage: int = 1, *, seed_version: str = "v0",
                 actions=SMB1_ACTIONS, multi_stage: bool = False):
        MarioSim = _load_mario_sim()
        self.world = int(world)
        self.stage = int(stage)
        self.version = seed_version
        self.level_id = f"{self.world}-{self.stage}"
        self._sim = MarioSim(self.world, self.stage, version=self.version,
                             actions=actions, multi_stage=multi_stage)
        self.action_names = [_buttons_name(list(a)) for a in actions]
        self.n_actions = len(self.action_names)

    def _normalize(self, info: dict) -> dict:
        out = dict(info)
        out.setdefault("game_id", self.game_id)
        out.setdefault("level_id", self.level_id)
        out.setdefault("world", self.world)
        out.setdefault("stage", self.stage)
        out.setdefault("x_pos", int(out.get("x_pos", 0)))
        out.setdefault("y_pos", int(out.get("y_pos", 0)))
        out.setdefault("status", out.get("status", "unknown"))
        out.setdefault("mode", "platformer")
        return out

    def reset(self, seed: int = 0) -> dict:
        return self._normalize(self._sim.reset(seed=seed))

    def close(self) -> None:
        self._sim.close()

    @property
    def ram(self):
        return self._sim.ram

    @property
    def last_info(self) -> dict:
        return self._normalize(self._sim.last_info)

    @property
    def last_obs(self):
        return self._sim.last_obs

    def snapshot(self):
        obs = self._sim.last_obs
        return SMB1Snapshot(
            emulator_state=self._sim.snapshot(),
            info=dict(self._sim.last_info),
            obs=obs.copy() if hasattr(obs, "copy") else obs,
        )

    def restore(self, snap) -> None:
        if not isinstance(snap, SMB1Snapshot):
            raise TypeError("SMB1Adapter.restore requires an SMB1Snapshot")
        self._sim.restore(
            snap.emulator_state,
            cached_info=snap.info,
            cached_obs=snap.obs,
        )

    def step(self, action_idx: int):
        obs, info, done = self._sim.step(action_idx)
        return obs, self._normalize(info), done

    def run_chunk(self, action_idx: int, frames: int):
        info, done = self._sim.run_chunk(action_idx, frames)
        return self._normalize(info), done

    def is_success(self, info: dict) -> bool:
        return smb1_is_success(info)

    def is_death(self, info: dict, done: bool) -> bool:
        return smb1_is_death(info, done)

    def progress(self, info: dict) -> float:
        return float(info.get("x_pos", 0))

    def cell(self, tile: int = 16) -> tuple:
        info = self.last_info
        return (self.game_id, self.level_id,
                int(info.get("x_pos", 0)) // tile,
                int(info.get("y_pos", 0)) // tile,
                info.get("status", "unknown"))


SML_PLATFORM_ACTIONS: list[tuple[str, ...]] = [
    (),
    ("right",),
    ("right", "b"),
    ("right", "a"),
    ("right", "a", "b"),
    ("a",),
    ("a", "b"),
    ("left",),
    ("left", "b"),
    ("left", "a"),
    ("left", "a", "b"),
    ("down",),
    ("down", "b"),
]
SML_PLATFORM_ACTION_NAMES = [_buttons_name(list(a)) for a in SML_PLATFORM_ACTIONS]

SML_SURFACE_ACTIONS: list[tuple[str, ...]] = [
    (),
    ("right",),
    ("right", "b"),
    ("right", "a"),
    ("right", "a", "b"),
    ("a",),
    ("a", "b"),
]


SMA4_SURFACE_ACTIONS: list[tuple[str, ...]] = [
    (),
    ("RIGHT",),
    ("RIGHT", "B"),
    ("RIGHT", "A"),
    ("RIGHT", "A", "B"),
    ("A",),
    ("A", "B"),
]


# Richer action vocabulary for SMB3 physics/power-state searches.  The lean
# default above remains the safe baseline; opt into this set for hard levels
# where backtracking, crouch/pipe entry, left+right acceleration, or A-pulsing
# can matter.
SMA4_PHYSICS_ACTIONS = [
    (),
    ("RIGHT",),
    ("RIGHT", "B"),
    ("RIGHT", "A"),
    ("RIGHT", "A", "B"),
    ("A",),
    ("A", "B"),
    ("LEFT",),
    ("LEFT", "B"),
    ("LEFT", "A"),
    ("LEFT", "A", "B"),
    ("DOWN",),
    ("DOWN", "B"),
    ("RIGHT", "DOWN"),
    ("RIGHT", "DOWN", "B"),
    ("RIGHT", "DOWN", "A", "B"),
    ("LEFT", "DOWN"),
    ("LEFT", "DOWN", "B"),
    ("LEFT", "RIGHT", "B"),
    ("LEFT", "RIGHT", "A", "B"),
    # Macro actions: one entry is advanced frame-by-frame inside run_chunk.
    (("RIGHT", "A", "B"), ("RIGHT", "B")),
    (("RIGHT", "A", "B"), ("RIGHT", "B"), ("RIGHT", "B"), ("RIGHT", "B")),
    (("A", "B"), ("B",)),
]


SMA4_PSPEED_ACTIONS = [
    (),
    ("RIGHT",),
    ("RIGHT", "B"),
    ("RIGHT", "A"),
    ("RIGHT", "A", "B"),
    ("DOWN",),
    ("DOWN", "B"),
    ("RIGHT", "DOWN", "B"),
    ("LEFT", "RIGHT", "B"),
    ("LEFT", "RIGHT", "A", "B"),
    (("RIGHT", "A", "B"), ("RIGHT", "B")),
    (("RIGHT", "A", "B"), ("RIGHT", "B"), ("RIGHT", "B"), ("RIGHT", "B")),
    ("LEFT",),
    ("LEFT", "B"),
    ("LEFT", "A"),
    ("LEFT", "A", "B"),
    ("A",),
    ("A", "B"),
]


SMA4_OVERWORLD_ACTIONS: list[tuple[str, ...]] = [
    (),
    ("LEFT",),
    ("RIGHT",),
    ("UP",),
    ("DOWN",),
    ("A",),
]


class SMA4Adapter:
    """Stable-Retro/mGBA-backed Super Mario Advance 4 adapter.

    V1 targets World 1-1.  The adapter boots through the GBA title/menu/world-map
    sequence and snapshots the first playable frame of 1-1, so search starts from
    gameplay rather than from the title screen.
    """

    game_id = "sma4"
    level_id = "1-1"
    _GAME = "SuperMarioAdvance4-Gba-v0"
    _ROM_SHA1 = "532f3307021637474b6dd37da059ca360f612337"
    _EWRAM = 0x02000000
    _IWRAM = 0x03000000

    # SMA4 RAM anchors from Data Crystal plus the fixed-point coordinate region.
    WORLD = 0x03002A69
    LIVES = 0x03002A6A
    KICKED_OUT = 0x03003AE7
    PSPEED = 0x03003C9F
    TIME = 0x03003D0B
    PLAYER_X_FIXED = 0x03003F24
    PLAYER_Y_FIXED = 0x03003F28
    PLAYER_X = 0x03003F25
    PLAYER_Y = 0x03003F29
    SPEED = 0x03003F40
    SPIN_JUMP = 0x03003F42
    POWERUP = 0x03003F5B
    END_LEVEL_WALK = 0x03003F67
    POWERUP_SET = 0x03003F74
    GOAL_CARD = 0x03004733

    # Overworld map state (reverse-engineered via scripts/probe_overworld.py).
    # Cursor coordinates live on a 0x20-pixel grid; world is 0-indexed.
    MAP_CURSOR_X = 0x03003DE4
    MAP_CURSOR_Y = 0x03003DE0
    PROGRESS = 0x03002C52  # per-world clear progress; flips on level completion
    ITEM_MENU_OPEN = 0x03003772
    MAP_EVENT = 0x03003774  # 0x11 on the live overworld, including stale-cursor exits

    # Overworld cursor nodes per (world, stage), discovered via the map probe.
    # Levels gated behind a prior clear list their prerequisites; the meta-planner
    # (M2) will grow this table from automatic node discovery.
    NODE_TABLE: dict[tuple[int, int], dict] = {
        (1, 1): {"cursor": (64, 32), "prereqs": ()},
    }

    def __init__(self, rom_path: str | os.PathLike[str] | None = None,
                 *, actions: list[tuple[str, ...]] | None = None,
                 boot_target: str = "1-1", world: int = 1, stage: int = 1):
        raw_rom_path = rom_path or os.environ.get("MARIO_AI_SMA4_ROM")
        if not raw_rom_path:
            raise ValueError("SMA4 ROM path is required or MARIO_AI_SMA4_ROM must be set")
        self.rom_path = Path(raw_rom_path)
        if not self.rom_path.exists():
            raise FileNotFoundError(f"SMA4 ROM not found: {self.rom_path}")
        self.rom_sha1 = hashlib.sha1(self.rom_path.read_bytes()).hexdigest()
        if self.rom_sha1 != self._ROM_SHA1:
            raise ValueError(f"Unsupported SMA4 ROM SHA1 {self.rom_sha1}")

        try:
            import stable_retro as retro
            from stable_retro.data import Integrations
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SMA4Adapter requires Stable-Retro. Install with "
                "`uv pip install -e '.[gba]'` in a dedicated GBA environment."
            ) from exc

        self.retro = retro
        self.Integrations = Integrations
        self.world = int(world)
        self.stage = int(stage)
        self.level_id = f"{self.world}-{self.stage}"
        self.actions = actions or SMA4_SURFACE_ACTIONS
        self.action_names = [_action_name(a) for a in self.actions]
        self.n_actions = len(self.action_names)
        self._last_info: dict[str, Any] = {}
        self._initial_state = None
        self._start_lives = 0

        self._register_integration()
        self.env = retro.make(
            self._GAME,
            state=retro.State.NONE,
            inttype=Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL,
            render_mode="rgb_array",
        )
        self.button_idx = {b: i for i, b in enumerate(self.env.buttons) if b is not None}
        self.env.reset()
        if boot_target == "overworld":
            self._boot_to_overworld()
        else:
            self.boot_to_level(self.world, self.stage)
        self._initial_state = self.snapshot()
        self._start_lives = int(self._last_info.get("lives", 0))

    def _register_integration(self) -> None:
        import json

        root = Path("runs") / "retro_custom"
        game = root / self._GAME
        game.mkdir(parents=True, exist_ok=True)
        link = game / "rom.gba"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(self.rom_path.resolve())
        (game / "rom.sha").write_text(self._ROM_SHA1 + "\n")
        (game / "metadata.json").write_text(json.dumps({"default_state": None}, indent=2))
        (game / "data.json").write_text(json.dumps({"info": {
            "world": {"address": self.WORLD, "type": "|u1"},
            "lives": {"address": self.LIVES, "type": "<u2"},
            "kicked": {"address": self.KICKED_OUT, "type": "|u1"},
            "pspeed": {"address": self.PSPEED, "type": "|u1"},
            "time": {"address": self.TIME, "type": ">u3"},
            "x": {"address": self.PLAYER_X, "type": "|u1"},
            "y": {"address": self.PLAYER_Y, "type": "|u1"},
            "speed": {"address": self.SPEED, "type": "|u1"},
            "spin": {"address": self.SPIN_JUMP, "type": "|u1"},
            "powerup": {"address": self.POWERUP, "type": "|u1"},
            "endwalk": {"address": self.END_LEVEL_WALK, "type": "|u1"},
            "powerup_set": {"address": self.POWERUP_SET, "type": "|u1"},
            "goalcard": {"address": self.GOAL_CARD, "type": "|u1"},
        }}, indent=2))
        (game / "scenario.json").write_text(json.dumps({
            "reward": {"variables": {}},
            "done": {"variables": {}},
        }, indent=2))
        self.retro.data.add_custom_integration(str(root.resolve()))

    def _mask(self, buttons: tuple[str, ...] | set[str]) -> Any:
        import numpy as np

        arr = np.zeros(self.env.num_buttons, dtype=np.uint8)
        for button in buttons:
            arr[self.button_idx[button]] = 1
        return arr

    def _read_block(self, addr: int, n: int = 1) -> bytes:
        for base, block in self.env.data.memory.blocks.items():
            if base <= addr < base + len(block):
                off = addr - base
                return bytes(block[off:off + n])
        raise KeyError(hex(addr))

    def _read_u8(self, addr: int) -> int:
        return self._read_block(addr, 1)[0]

    def _read_u16(self, addr: int) -> int:
        b = self._read_block(addr, 2)
        return b[0] | (b[1] << 8)

    def _read_u32(self, addr: int) -> int:
        b = self._read_block(addr, 4)
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    @staticmethod
    def _classify_mode(time: int, cx: int, cy: int, *,
                       world_raw: int = 0, item_menu_open: int = 0,
                       map_event: int = 0) -> str:
        """overworld / level / menu from the level timer and the map cursor.

        In a level the level timer runs (>0); on the map the timer reads 0 and the
        cursor normally sits on the 0x20 grid.  SMA4's warp-zone and World-8 map
        use off-grid Y positions, so those raw-world maps are special-cased.
        Anything else (title, file select, Toad house, pause, item menu) is
        reported as ``menu``.
        """
        if time > 0:
            return "level"
        if item_menu_open:
            return "menu"
        # Some verified Toad-house exits leave MAP_CURSOR_* off-grid even though
        # the live World-1 map is already visible. MAP_EVENT=0x11 distinguishes
        # that overworld state from title/file-select menus.
        if map_event == 0x11 and (cx or cy):
            return "overworld"
        if world_raw in (7, 8) and (cx or cy):
            return "overworld"
        if cx % 0x20 == 0 and cy % 0x20 == 0 and (cx or cy):
            return "overworld"
        return "menu"

    def _normalize_info(self, base_info: dict | None = None) -> dict:
        base_info = dict(base_info or {})
        x_fixed = self._read_u32(self.PLAYER_X_FIXED)
        y_fixed = self._read_u32(self.PLAYER_Y_FIXED)
        time = int(base_info.get("time", self._read_u32(self.TIME)))
        cx, cy = self._read_u8(self.MAP_CURSOR_X), self._read_u8(self.MAP_CURSOR_Y)
        world_raw = int(self._read_u8(self.WORLD))
        item_menu_open = int(self._read_u8(self.ITEM_MENU_OPEN))
        map_event = int(self._read_u8(self.MAP_EVENT))
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "world": world_raw + 1,
            "world_raw": world_raw,
            "is_warp_zone": world_raw == 8,
            "stage": getattr(self, "stage", 1),
            "x_pos": x_fixed >> 8,
            "x_fixed": x_fixed,
            "x_subpixel": x_fixed & 0xff,
            "y_pos": self._read_u8(self.PLAYER_Y),
            "y_fixed": y_fixed,
            "y_subpixel": y_fixed & 0xff,
            "time": time,
            "lives": int(base_info.get("lives", self._read_u16(self.LIVES))),
            "status": "unknown",
            "mode": self._classify_mode(
                time, cx, cy, world_raw=world_raw,
                item_menu_open=item_menu_open, map_event=map_event),
            "cursor": (cx, cy),
            "cleared": int(self._read_u8(self.PROGRESS)),
            "item_menu_open": item_menu_open,
            "map_event": map_event,
            "flag_get": False,
            "pspeed": int(base_info.get("pspeed", self._read_u8(self.PSPEED))),
            "speed": int(base_info.get("speed", self._read_u8(self.SPEED))),
            "speed_signed": _s8(base_info.get("speed", self._read_u8(self.SPEED))),
            "spin_jump": int(base_info.get("spin", self._read_u8(self.SPIN_JUMP))),
            "powerup": int(base_info.get("powerup", self._read_u8(self.POWERUP))),
            "powerup_set": int(base_info.get("powerup_set", self._read_u8(self.POWERUP_SET))),
            "endwalk": int(base_info.get("endwalk", self._read_u8(self.END_LEVEL_WALK))),
            "goalcard": int(base_info.get("goalcard", self._read_u8(self.GOAL_CARD))),
            "kicked": int(base_info.get("kicked", self._read_u8(self.KICKED_OUT))),
        }

    def _step_buttons(self, buttons: tuple[str, ...] = ()):
        obs, _reward, terminated, truncated, base_info = self.env.step(self._mask(buttons))
        self._last_info = self._normalize_info(base_info)
        return obs, dict(self._last_info), bool(terminated or truncated)

    @staticmethod
    def _menu_buttons(frame: int) -> tuple[str, ...]:
        """Title -> file-select button mashing for the boot routine."""
        phase = frame % 80
        if 120 < frame < 2050:
            if phase < 25:
                return ("A",)
            if 40 <= phase < 65:
                return ("START",)
        return ()

    def _boot_to_overworld(self) -> None:
        """Drive title -> file select -> world map, stopping ON the overworld map.

        This is the first ~2400 frames of the boot routine (menu mashing then a
        settle), before any map-cursor navigation or level entry.  Frame numbers
        are preserved so the level-entry windows in `_boot_to_1_1` stay aligned.
        """
        info: dict = {}
        done = False
        for frame in range(1, 2401):
            _obs, info, done = self._step_buttons(self._menu_buttons(frame))
            if done:
                break
        self._last_info = dict(info)

    def _boot_to_1_1(self) -> None:
        self._boot_to_overworld()
        info = self._last_info
        done = False
        for frame in range(2401, 3401):
            buttons: tuple[str, ...] = ()
            if 2500 <= frame < 2530:
                buttons = ("RIGHT",)
            elif 2680 <= frame < 2710:
                buttons = ("UP",)
            elif 2860 <= frame < 2890:
                buttons = ("A",)
            _obs, info, done = self._step_buttons(buttons)
            if done:
                break
        self._last_info = dict(info)
        if int(self._last_info.get("x_pos", 0)) < 10 or int(self._last_info.get("y_pos", 0)) == 0:
            raise RuntimeError(f"failed to boot SMA4 1-1, info={self._last_info}")

    def boot_to_level(self, world: int, stage: int) -> dict:
        """Boot the emulator to the start of a level.

        1-1 uses the verified scripted boot.  Other levels are reached by booting
        onto the overworld and navigating the map; gated levels (behind a fortress
        or an uncleared prerequisite) need the M2 meta-planner, which grows
        ``NODE_TABLE`` from automatic node discovery.
        """
        key = (int(world), int(stage))
        if key == (1, 1):
            self._boot_to_1_1()
            return dict(self._last_info)
        node = self.NODE_TABLE.get(key)
        if node is None:
            raise NotImplementedError(
                f"no overworld route to SMA4 {world}-{stage} yet; the M2 meta-planner "
                f"discovers map nodes. Known nodes: {sorted(self.NODE_TABLE)}")
        if node.get("prereqs"):
            raise NotImplementedError(
                "gated-level prerequisite clearing arrives with the M2 meta-planner")
        self._boot_to_overworld()
        if not self.goto_node(node["cursor"]):
            raise RuntimeError(
                f"could not navigate to SMA4 {world}-{stage} node {node['cursor']}")
        info = self.enter_level()
        if info.get("mode") != "level" or int(info.get("x_pos", 0)) <= 5:
            raise RuntimeError(f"failed to enter SMA4 {world}-{stage}, info={info}")
        return info

    # ---- overworld navigation (used by SMA4OverworldAdapter / the M2 driver) ----
    def is_overworld(self) -> bool:
        return self._last_info.get("mode") == "overworld"

    def _map_walk(self, button: tuple[str, ...], *, hold: int = 16, settle: int = 12) -> bool:
        """Hold a map direction then settle; return True if the cursor moved."""
        before = self._last_info.get("cursor")
        for _ in range(hold):
            self._step_buttons(button)
        for _ in range(settle):
            self._step_buttons(())
        return self._last_info.get("cursor") != before

    def goto_node(self, target: tuple[int, int], *, rounds: int = 8) -> bool:
        """Greedy cursor navigation to a (cursor_x, cursor_y) grid node.

        Map paths are directional, so this succeeds where a monotone greedy path
        exists (the full map topology arrives with the M2 meta-planner).  Returns
        whether the cursor reached the target.
        """
        tx, ty = target
        for _ in range(rounds):
            cx, cy = self._last_info.get("cursor", (0, 0))
            if (cx, cy) == (tx, ty):
                return True
            moved = False
            if cx != tx:
                moved |= self._map_walk(("RIGHT",) if tx > cx else ("LEFT",))
            cx, cy = self._last_info.get("cursor", (0, 0))
            if cy != ty:
                moved |= self._map_walk(("UP",) if ty < cy else ("DOWN",))
            if not moved:
                break
        return self._last_info.get("cursor") == (tx, ty)

    def _controllable(self) -> bool:
        """Non-destructively test whether the player responds to RIGHT+B yet."""
        try:
            rb = self.actions.index(("RIGHT", "B"))
        except ValueError:
            return True
        snap = self.snapshot()
        x0 = int(self._last_info.get("x_pos", 0))
        for _ in range(12):
            self._step_buttons(self.actions[rb])
        moved = int(self._last_info.get("x_pos", 0)) != x0
        self.restore(snap)
        return moved

    def enter_level(self, *, max_presses: int = 60, settle: int = 360,
                    probe: int = 20) -> dict:
        """Press A (edge-triggered) until the level loads, then settle past the
        intro lockout until the player is controllable.

        The mode flips to ``level`` during the load fade and the level then plays
        a short non-interactive intro (~140 frames on 1-1, longer for pipe-entry
        levels) where inputs are ignored.  Returning before that leaves a beam
        search stuck at the spawn (x_max == entry x, empty path), so we NOOP —
        probing controllability every ``probe`` frames — until the player moves.
        """
        for _ in range(max_presses):
            self._step_buttons(("A",))
            self._step_buttons(())
            if self._last_info.get("mode") == "level":
                break
        else:
            return dict(self._last_info)
        elapsed = 0
        while elapsed < settle:
            for _ in range(probe):
                self._step_buttons(())
            elapsed += probe
            if self._controllable():
                break
        return dict(self._last_info)

    def reset(self, seed: int = 0) -> dict:
        del seed
        if self._initial_state is not None:
            self.restore(self._initial_state)
        return dict(self._last_info)

    def close(self) -> None:
        self.env.close()

    @property
    def ram(self):
        return self.env.get_ram()

    @property
    def last_info(self) -> dict:
        return dict(self._last_info)

    @property
    def last_obs(self):
        return self.env.get_screen(apply_rotation=True)

    def snapshot(self):
        return self.env.em.get_state(), dict(self._last_info)

    def restore(self, snap) -> None:
        state, info = snap
        self.env.em.set_state(state)
        self.env.data.reset()
        self.env.data.update_ram()
        self._last_info = self._normalize_info(info)

    def step(self, action_idx: int):
        return self._step_buttons(self._action_buttons(self.actions[action_idx], 0))

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        info = self._last_info
        action = self.actions[action_idx]
        for frame in range(frames):
            _obs, info, done = self._step_buttons(self._action_buttons(action, frame))
            if done or self.is_success(info) or self.is_death(info, done):
                break
        return dict(info), done

    @staticmethod
    def _action_buttons(action, frame: int = 0) -> tuple[str, ...]:
        if not action:
            return ()
        first = action[0]
        if isinstance(first, str):
            return tuple(action)
        seq = tuple(tuple(step) for step in action)
        return seq[frame % len(seq)]

    def is_success(self, info: dict) -> bool:
        # `goalcard` can change during ordinary level play; it is not terminal by itself.
        # Hitting the 1-1 roulette card puts the end-level routine at 255 before
        # the post-card walkout settles, so any non-zero value is a terminal clear.
        return int(info.get("endwalk", 0)) != 0

    def is_death(self, info: dict, done: bool) -> bool:
        if done:
            return not self.is_success(info)
        if int(info.get("lives", self._start_lives)) < self._start_lives:
            return True
        if int(info.get("kicked", 0)) != 0:
            return True
        return int(info.get("time", 1)) <= 0

    def progress(self, info: dict) -> float:
        return float(info.get("x_pos", 0))

    def cell(self, tile: int = 16) -> tuple:
        info = self.last_info
        x = int(info.get("x_pos", 0))
        y = int(info.get("y_pos", 0))
        return (self.game_id, self.level_id, x // tile, (x % tile) // 4,
                y // tile, int(info.get("speed", 0)) // 16,
                int(info.get("pspeed", 0)) // 16, int(info.get("powerup", 0)))


class SMA4OverworldAdapter:
    """Overworld (world-map) view over a shared :class:`SMA4Adapter` core.

    The level view (`SMA4Adapter`) and this overworld view drive the *same*
    emulator, so snapshots are interchangeable: the meta-search restores a map
    node here, enters a level, and hands the resulting snapshot to the level view
    for `beam_search_adapter` — and the cleared level's snapshot flows straight
    back.  Pass ``core=`` to attach to an already-booted adapter (the M2 driver
    shares one core); otherwise a fresh core is booted onto the map.

    As a `GameAdapter`, "progress" is the popcount of the cleared bitmap and
    "success" means a level was entered, so the generic `beam_search_adapter`
    could search the map directly; in practice the meta-planner uses
    `goto_node`/`enter_level` against the known small graph.
    """

    game_id = "sma4"
    level_id = "overworld"

    def __init__(self, rom_path=None, *, core: SMA4Adapter | None = None,
                 actions: list[tuple[str, ...]] | None = None):
        if core is None:
            core = SMA4Adapter(rom_path, boot_target="overworld")
            self._owns_core = True
        else:
            self._owns_core = False
        self.core = core
        self.actions = actions or SMA4_OVERWORLD_ACTIONS
        self.action_names = [_buttons_name(list(a)) for a in self.actions]
        self.n_actions = len(self.actions)
        if not self.core.is_overworld():
            self.core._boot_to_overworld()
        for _ in range(60):  # settle at the map node before snapshotting
            self.core._step_buttons(())
        self._initial_state = self.core.snapshot()

    def _ow_info(self, info: dict | None = None) -> dict:
        out = dict(info if info is not None else self.core.last_info)
        out["level_id"] = self.level_id
        return out

    @property
    def rom_sha1(self):
        return self.core.rom_sha1

    @property
    def ram(self):
        return self.core.ram

    @property
    def env(self):
        return self.core.env

    @property
    def last_info(self) -> dict:
        return self._ow_info()

    @property
    def last_obs(self):
        return self.core.last_obs

    def reset(self, seed: int = 0) -> dict:
        del seed
        self.core.restore(self._initial_state)
        return self.last_info

    def close(self) -> None:
        if self._owns_core:
            self.core.close()

    def snapshot(self):
        return self.core.snapshot()

    def restore(self, snap) -> None:
        self.core.restore(snap)

    def step(self, action_idx: int):
        obs, info, done = self.core._step_buttons(self.actions[action_idx])
        return obs, self._ow_info(info), done

    def run_chunk(self, action_idx: int, frames: int):
        info = self.core.last_info
        done = False
        for _ in range(frames):
            _obs, info, done = self.step(action_idx)
            if done or self.is_success(info):
                break
        return self._ow_info(info), done

    def is_success(self, info: dict) -> bool:
        return info.get("mode") == "level"

    def is_death(self, info: dict, done: bool) -> bool:
        return False  # no death on the overworld map

    def progress(self, info: dict) -> float:
        return float(bin(int(info.get("cleared", 0))).count("1"))

    def cell(self, tile: int = 16) -> tuple:
        info = self.core.last_info
        cx, cy = info.get("cursor", (0, 0))
        return (self.game_id, "overworld", cx, cy, int(info.get("cleared", 0)))

    # ---- meta-planner helpers (delegate to the shared core) ----
    def goto_node(self, target: tuple[int, int], **kw) -> bool:
        return self.core.goto_node(target, **kw)

    def enter_level(self, **kw) -> dict:
        return self._ow_info(self.core.enter_level(**kw))

    def cleared_mask(self) -> int:
        return int(self.core.last_info.get("cleared", 0))


class SMLAdapter:
    """PyBoy-backed Super Mario Land adapter.

    Requires a legally obtained ROM path, normally supplied via `MARIO_AI_SML_ROM`.
    V1 targets platform stages; shooter stages can use the same adapter with a wider
    action vocabulary later.
    """

    game_id = "sml"
    _ALL_BUTTONS = ("left", "right", "up", "down", "a", "b", "start", "select")

    # Data Crystal SML RAM anchors.
    MARIO_Y = 0xC201
    MARIO_X = 0xC202
    JUMP_STATE = 0xC207
    Y_SPEED = 0xC208
    ON_GROUND = 0xC20A
    X_SPEED_ABS = 0xC20C
    FACING = 0xC20D
    TIMER_FRAMES = 0xDA00
    TIMER_SECONDS = 0xDA01
    TIMER_HUNDREDS = 0xDA02
    LIVES = 0xDA15
    TIME_UP = 0xDA1D
    MUSIC_REQUEST = 0xDFE8
    MUSIC_CURRENT = 0xDFE9
    POWER_STATUS = 0xFF99
    POWER_DEATH_TIMER = 0xFFA6
    GAME_OVER = 0xFFB3
    SUPERBALL = 0xFFB5

    def __init__(self, rom_path: str | os.PathLike[str] | None = None,
                 *, world: int = 1, stage: int = 1,
                 actions: list[tuple[str, ...]] | None = None,
                 window: str = "null"):
        raw_rom_path = rom_path or os.environ.get("MARIO_AI_SML_ROM")
        if not raw_rom_path:
            raise ValueError("SML ROM path is required or MARIO_AI_SML_ROM must be set")
        self.rom_path = Path(raw_rom_path)
        if not self.rom_path.exists():
            raise FileNotFoundError(f"SML ROM not found: {self.rom_path}")

        try:
            from pyboy import PyBoy
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep
            raise RuntimeError(
                "SMLAdapter requires PyBoy. Install with `uv pip install -e '.[sml]'`."
            ) from exc

        self.world = int(world)
        self.stage = int(stage)
        self.level_id = f"{self.world}-{self.stage}"
        self.actions = actions or SML_PLATFORM_ACTIONS
        self.action_names = [_buttons_name(list(a)) for a in self.actions]
        self.n_actions = len(self.action_names)
        self.rom_sha1 = hashlib.sha1(self.rom_path.read_bytes()).hexdigest()

        self.pyboy = PyBoy(str(self.rom_path), window=window)
        self.pyboy.set_emulation_speed(0)
        self._pressed: set[str] = set()
        self._last_info: dict[str, Any] = {}
        self._initial_state: tuple[bytes, tuple[str, ...]] | None = None
        self._start_world = (self.world, self.stage)
        self._start_lives = 0
        self._boot_to_level()
        self._initial_state = self.snapshot()
        self.reset(seed=0)

    def _read(self, addr: int) -> int:
        return int(self.pyboy.memory[addr])

    def _wrapper_value(self, name: str, default=None):
        wrapper = getattr(self.pyboy, "game_wrapper", None)
        if wrapper is None:
            return default
        try:
            return getattr(wrapper, name)
        except Exception:
            return default

    def _current_world(self) -> tuple[int, int]:
        world = self._wrapper_value("world", None)
        if isinstance(world, (tuple, list)) and len(world) >= 2:
            return int(world[0]), int(world[1])
        return self.world, self.stage

    def _boot_to_level(self) -> None:
        wrapper = getattr(self.pyboy, "game_wrapper", None)
        if wrapper is not None and hasattr(wrapper, "start_game"):
            wrapper.start_game()
            self.pyboy.tick(1, False)
        else:
            self.pyboy.button("start", 2)
            self.pyboy.tick(180, False)
        self._release_all()
        self._last_info = self._normalize_info()
        self._start_world = self._current_world()
        self._start_lives = int(self._last_info.get("lives", self._read(self.LIVES)))

    def _release_all(self) -> None:
        for button in self._ALL_BUTTONS:
            try:
                self.pyboy.button_release(button)
            except Exception:
                pass
        self._pressed = set()
        self.pyboy.tick(1, False)

    def _force_buttons(self, buttons: set[str]) -> None:
        for button in self._ALL_BUTTONS:
            try:
                self.pyboy.button_release(button)
            except Exception:
                pass
        for button in sorted(buttons):
            self.pyboy.button_press(button)
        self._pressed = set(buttons)

    def _set_buttons(self, buttons: set[str]) -> None:
        for button in sorted(self._pressed - buttons):
            self.pyboy.button_release(button)
        for button in sorted(buttons - self._pressed):
            self.pyboy.button_press(button)
        self._pressed = set(buttons)

    def _normalize_info(self) -> dict:
        w, s = self._current_world()
        progress = self._wrapper_value("level_progress", None)
        if progress is None:
            progress = self._read(self.MARIO_X)
        lives = self._wrapper_value("lives_left", self._read(self.LIVES))
        time_left = self._wrapper_value("time_left", None)
        if time_left is None:
            time_left = self._read(self.TIMER_HUNDREDS) * 100 + self._read(self.TIMER_SECONDS)
        power = self._read(self.POWER_STATUS)
        return {
            "game_id": self.game_id,
            "level_id": f"{w}-{s}",
            "world": w,
            "stage": s,
            "x_pos": int(progress),
            "y_pos": self._read(self.MARIO_Y),
            "time": int(time_left),
            "lives": int(lives),
            "status": "small" if power == 0 else ("big" if power == 2 else f"power{power}"),
            "mode": "platformer",
            "flag_get": False,
            "jump_state": self._read(self.JUMP_STATE),
            "y_speed": self._read(self.Y_SPEED),
            "on_ground": self._read(self.ON_GROUND),
            "x_speed_abs": self._read(self.X_SPEED_ABS),
            "facing": self._read(self.FACING),
            "superball": self._read(self.SUPERBALL),
        }

    def reset(self, seed: int = 0) -> dict:
        del seed  # SML is deterministic from the loaded state.
        if self._initial_state is not None:
            self.restore(self._initial_state)
        self._last_info = self._normalize_info()
        return dict(self._last_info)

    def close(self) -> None:
        self._release_all()
        self.pyboy.stop()

    @property
    def ram(self):
        return self.pyboy.memory

    @property
    def last_info(self) -> dict:
        return dict(self._last_info)

    @property
    def last_obs(self):
        return self.pyboy.screen.image

    def snapshot(self):
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue(), tuple(sorted(self._pressed))

    def restore(self, snap) -> None:
        state, pressed = snap
        self.pyboy.load_state(io.BytesIO(state))
        self._force_buttons(set(pressed))
        self._last_info = self._normalize_info()

    def step(self, action_idx: int, *, render: bool = False):
        buttons = set(self.actions[action_idx])
        self._set_buttons(buttons)
        alive = bool(self.pyboy.tick(1, render))
        self._last_info = self._normalize_info()
        return self.last_obs, dict(self._last_info), not alive

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        for _ in range(frames):
            _obs, info, done = self.step(action_idx)
            if done or self.is_success(info) or self.is_death(info, done):
                break
        return dict(self._last_info), done

    def is_success(self, info: dict) -> bool:
        cur_world = (int(info.get("world", self.world)), int(info.get("stage", self.stage)))
        if cur_world != self._start_world:
            return True
        return self._read(self.MUSIC_REQUEST) == 0x01 or self._read(self.MUSIC_CURRENT) == 0x01

    def is_death(self, info: dict, done: bool) -> bool:
        if done:
            return not self.is_success(info)
        if int(info.get("lives", self._start_lives)) < self._start_lives:
            return True
        if int(info.get("time", 1)) <= 0 or self._read(self.TIME_UP) == 0xFF:
            return True
        if self._read(self.GAME_OVER) == 0x39:
            return True
        if self._read(self.POWER_DEATH_TIMER) == 0x90:
            return True
        return self._read(self.MUSIC_REQUEST) == 0x02 or self._read(self.MUSIC_CURRENT) == 0x02

    def progress(self, info: dict) -> float:
        return float(info.get("x_pos", 0))

    def cell(self, tile: int = 16) -> tuple:
        info = self.last_info
        x = int(info.get("x_pos", 0))
        y = int(info.get("y_pos", 0))
        return (self.game_id, info.get("level_id", self.level_id),
                x // tile,
                (x % tile) // 4,
                y // tile,
                info.get("status", "unknown"),
                int(info.get("on_ground", 0)),
                int(info.get("jump_state", 0)),
                int(info.get("x_speed_abs", 0)) // 16,
                int(info.get("y_speed", 0)) // 16)
