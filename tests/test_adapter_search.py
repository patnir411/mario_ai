from __future__ import annotations

import itertools
import json

import numpy as np

from mario import adapters, search as search_module
from mario.adapters import (
    SMA4Adapter,
    SMA4_PHYSICS_ACTIONS,
    SMA4_PSPEED_ACTIONS,
    SMB1Adapter,
)
from mario.ram import MARIO_LEVEL_PAGE, MARIO_X_ON_SCREEN, MARIO_Y_ON_SCREEN
from mario.render import make_contact_sheet_adapter
from mario.search import (_adapter_physics_cell, beam_search_adapter,
                          coverage_search_adapter, goal_suffix_search)


class FakeAdapter:
    game_id = "fake"
    level_id = "1-1"
    action_names = ["wait", "right", "die"]
    n_actions = 3

    def __init__(self):
        self.x = 0
        self.y = 0
        self.dead = False
        self._last_info = self._info()

    def _info(self):
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "x_pos": self.x,
            "y_pos": self.y,
            "flag_get": self.x >= 3,
            "status": "ok",
        }

    def reset(self, seed: int = 0):
        del seed
        self.x = 0
        self.y = 0
        self.dead = False
        self._last_info = self._info()
        return dict(self._last_info)

    def close(self):
        pass

    @property
    def ram(self):
        return b""

    @property
    def last_info(self):
        return dict(self._last_info)

    @property
    def last_obs(self):
        return None

    def snapshot(self):
        return self.x, self.y, self.dead

    def restore(self, snap):
        self.x, self.y, self.dead = snap
        self._last_info = self._info()

    def step(self, action_idx: int):
        if action_idx == 1:
            self.x += 1
        elif action_idx == 2:
            self.dead = True
        self._last_info = self._info()
        return None, dict(self._last_info), self.dead

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        for _ in range(frames):
            _obs, info, done = self.step(action_idx)
            if done or self.is_success(info):
                break
        return dict(self._last_info), done

    def is_success(self, info: dict) -> bool:
        return bool(info.get("flag_get", False))

    def is_death(self, info: dict, done: bool) -> bool:
        return bool(done) and not self.is_success(info)

    def progress(self, info: dict) -> float:
        return float(info["x_pos"])

    def cell(self, tile: int = 16):
        return self.game_id, self.x // tile, self.y // tile, self.dead


class NoveltyRetentionAdapter:
    """A pruned candidate reaches a cell later needed by the retained lineage."""

    game_id = "novelty-retention"
    level_id = "synthetic"

    def __init__(self, action_names):
        self.action_names = list(action_names)
        self.n_actions = len(self.action_names)
        self.reset()

    def _info(self):
        coordinates = {
            "root": (0, 0),
            "high": (12, 0),
            "target": (11, 1),
            "goal": (11, 1),
        }
        x, y = coordinates[self.state]
        return {
            "game_id": self.game_id,
            "level_id": self.level_id,
            "x_pos": x,
            "y_pos": y,
            "flag_get": self.state == "goal",
            "status": "ok",
        }

    def reset(self, seed: int = 0):
        del seed
        self.state = "root"
        self._last_info = self._info()
        return dict(self._last_info)

    def close(self):
        pass

    @property
    def ram(self):
        return b""

    @property
    def last_info(self):
        return dict(self._last_info)

    @property
    def last_obs(self):
        return None

    def snapshot(self):
        return self.state

    def restore(self, snap):
        self.state = snap
        self._last_info = self._info()

    def run_chunk(self, action_idx: int, frames: int):
        del frames
        action = self.action_names[action_idx]
        if self.state == "root":
            if action == "high":
                self.state = "high"
            elif action == "target":
                self.state = "target"
        elif self.state == "high" and action == "target":
            self.state = "target"
        elif self.state == "target" and action == "finish":
            self.state = "goal"
        self._last_info = self._info()
        return dict(self._last_info), False

    def is_success(self, info: dict) -> bool:
        return bool(info["flag_get"])

    def is_death(self, info: dict, done: bool) -> bool:
        del info, done
        return False

    def progress(self, info: dict) -> float:
        return float(info["x_pos"])


class FakeMarioSim:
    def __init__(self, *_args, **_kwargs):
        self.x = 0
        self._last_info = {"x_pos": 0, "y_pos": 0, "status": "small"}
        self._last_obs = np.asarray([0])

    def reset(self, seed=0):
        del seed
        self.x = 0
        self._last_info = {"x_pos": 0, "y_pos": 0, "status": "small"}
        self._last_obs = np.asarray([0])
        return dict(self._last_info)

    def close(self):
        pass

    @property
    def ram(self):
        return b""

    @property
    def last_info(self):
        return self._last_info

    @property
    def last_obs(self):
        return self._last_obs

    def snapshot(self):
        return self.x

    def restore(self, state, *, cached_info=None, cached_obs=None):
        self.x = state
        if cached_info is not None:
            self._last_info = dict(cached_info)
        if cached_obs is not None:
            self._last_obs = cached_obs.copy()

    def step(self, action_idx):
        del action_idx
        self.x += 20
        self._last_info = {
            "x_pos": self.x, "y_pos": 0, "status": "small"}
        self._last_obs = np.asarray([self.x])
        return self._last_obs, dict(self._last_info), False

    def run_chunk(self, action_idx, frames):
        info = self._last_info
        for _ in range(frames):
            _obs, info, _done = self.step(action_idx)
        return info, False


class FakePipeMacroSim:
    """Requires two consecutive DOWN chunks to enter a new area."""

    def __init__(self, *, goal_on_entry=True):
        self.goal_on_entry = goal_on_entry
        self._ram = bytearray(0x800)
        self.total_run_calls = 0
        self.closed = False
        self.reset()

    def _info(self):
        return {
            "x_pos": 0,
            "y_pos": 0,
            "status": "small",
            "flag_get": self.finished or (
                self.goal_on_entry and self.down_chunks >= 2
            ),
        }

    def reset(self, seed=0):
        del seed
        self._ram[:] = b"\0" * len(self._ram)
        self.down_chunks = 0
        self.finished = False
        self.total_run_calls = 0
        self.closed = False
        self._last_info = self._info()
        return dict(self._last_info)

    def close(self):
        self.closed = True

    @property
    def ram(self):
        return self._ram

    @property
    def last_info(self):
        return dict(self._last_info)

    def snapshot(self):
        return (
            self.down_chunks,
            self.finished,
            bytes(self._ram),
            dict(self._last_info),
        )

    def restore(self, snap):
        self.down_chunks, self.finished, raw_ram, info = snap
        self._ram[:] = raw_ram
        self._last_info = dict(info)

    def run_chunk(self, action_idx, frames):
        del frames
        self.total_run_calls += 1
        if action_idx == 7:
            self.down_chunks += 1
            if self.down_chunks >= 2:
                self._ram[0x0760] = 1
        elif self._ram[0x0760] == 1 and action_idx == 0:
            self.finished = True
        self._last_info = self._info()
        return dict(self._last_info), False


class PrunedCellLoopSim:
    """Width-1 retains x=500 but also generates and prunes x=130."""

    X = {"root": 100, "high": 500, "old": 130}

    def __init__(self):
        self._ram = bytearray(0x800)
        self.expanded_from = []
        self.reset()

    @property
    def ram(self):
        return self._ram

    @property
    def last_info(self):
        return dict(self._last_info)

    def _set(self, state):
        self.state = state
        x = self.X[state]
        self._ram[MARIO_LEVEL_PAGE] = x // 256
        self._ram[MARIO_X_ON_SCREEN] = x % 256
        self._ram[MARIO_Y_ON_SCREEN] = 80
        self._last_info = {
            "x_pos": x,
            "y_pos": 80,
            "status": "small",
            "flag_get": False,
        }

    def reset(self, seed=0):
        del seed
        self._ram[:] = b"\0" * len(self._ram)
        self.expanded_from.clear()
        self._set("root")
        return self.last_info

    def close(self):
        pass

    def snapshot(self):
        return self.state, bytes(self._ram), self.last_info

    def restore(self, snap):
        self.state, raw_ram, info = snap
        self._ram[:] = raw_ram
        self._last_info = dict(info)

    def run_chunk(self, action_idx, frames):
        del frames
        self.expanded_from.append(self.state)
        if self.state == "root":
            self._set("high" if action_idx == 0 else "old")
        elif self.state == "high":
            self._set("high" if action_idx == 0 else "old")
        else:
            self._set("old")
        return self.last_info, False


class NativeNoveltyRetentionSim:
    """A native-search graph where a pruned cell is needed one layer later."""

    POSITION = {
        "root": (0, 0),
        "high": (12, 0),
        "target": (11, 1),
        "goal": (11, 1),
    }

    def __init__(self):
        self._ram = bytearray(0x800)
        self.reset()

    @property
    def ram(self):
        return self._ram

    @property
    def last_info(self):
        return dict(self._last_info)

    def _set(self, state):
        self.state = state
        x, y = self.POSITION[state]
        self._ram[MARIO_LEVEL_PAGE] = x // 256
        self._ram[MARIO_X_ON_SCREEN] = x % 256
        self._ram[MARIO_Y_ON_SCREEN] = y
        self._last_info = {
            "x_pos": x,
            "y_pos": y,
            "status": "small",
            "flag_get": state == "goal",
        }

    def reset(self, seed=0):
        del seed
        self._ram[:] = b"\0" * len(self._ram)
        self._set("root")
        return self.last_info

    def close(self):
        pass

    def snapshot(self):
        return self.state, bytes(self._ram), self.last_info

    def restore(self, snap):
        self.state, raw_ram, info = snap
        self._ram[:] = raw_ram
        self._last_info = dict(info)

    def run_chunk(self, action_idx, frames):
        del frames
        if self.state == "root":
            self._set(
                "high" if action_idx == 0
                else "target" if action_idx == 1
                else "root"
            )
        elif self.state == "high":
            self._set("target" if action_idx == 1 else "high")
        elif self.state == "target":
            self._set("goal" if action_idx == 2 else "target")
        return self.last_info, False


def test_smb1_adapter_snapshot_restores_info_obs_and_cell(monkeypatch):
    monkeypatch.setattr(adapters, "_load_mario_sim", lambda: FakeMarioSim)
    adapter = SMB1Adapter()
    adapter.reset()
    adapter.step(1)
    snap = adapter.snapshot()
    adapter.step(1)
    assert adapter.last_info["x_pos"] == 40

    adapter.restore(snap)

    assert adapter.last_info["x_pos"] == 20
    assert adapter.last_obs.tolist() == [20]
    assert adapter.cell(tile=16)[2] == 1


def test_adapter_beam_solves_with_adapter_contract():
    result = beam_search_adapter(FakeAdapter(), beam_width=2, chunk_frames=1, max_depth=5)
    assert result.solved
    assert result.path == [1, 1, 1]
    assert result.final_info["flag_get"]


def test_adapter_beam_writes_search_trace(tmp_path):
    trace = tmp_path / "trace.jsonl"
    result = beam_search_adapter(FakeAdapter(), beam_width=2, chunk_frames=1, max_depth=5,
                                 trace_path=trace)
    assert result.solved
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows
    assert rows[0]["game_id"] == "fake"
    assert {"node_id", "parent_id", "action_idx", "score", "progress", "cell"} <= set(rows[0])
    assert any(row["success"] for row in rows)


def test_sma4_success_accepts_card_hit_walkout_state():
    adapter = object()
    assert not SMA4Adapter.is_success(adapter, {"endwalk": 0})
    assert SMA4Adapter.is_success(adapter, {"endwalk": 1})
    assert SMA4Adapter.is_success(adapter, {"endwalk": 255})


def test_sma4_physics_actions_include_backtrack_pipe_and_macro_inputs():
    names = []
    for action in SMA4_PHYSICS_ACTIONS:
        if not action:
            names.append("NOOP")
        elif isinstance(action[0], str):
            names.append("+".join(action))
        else:
            names.append("/".join("+".join(step) for step in action))
    assert "LEFT" in names
    assert "DOWN+B" in names
    assert "LEFT+RIGHT+B" in names
    pspeed_names = ["+".join(a) if a and isinstance(a[0], str) else "NOOP"
                    for a in SMA4_PSPEED_ACTIONS if not a or isinstance(a[0], str)]
    assert "LEFT+RIGHT+B" in pspeed_names
    assert "DOWN+B" in pspeed_names
    assert "LEFT+A" in pspeed_names
    assert "LEFT+A+B" in pspeed_names
    macro = (("RIGHT", "A", "B"), ("RIGHT", "B"))
    assert SMA4Adapter._action_buttons(macro, 0) == ("RIGHT", "A", "B")
    assert SMA4Adapter._action_buttons(macro, 1) == ("RIGHT", "B")
    assert SMA4Adapter._action_buttons(macro, 2) == ("RIGHT", "A", "B")


def test_adapter_physics_cell_splits_movement_affordances():
    base = {"x_pos": 64, "y_pos": 32, "x_fixed": 64 << 8}
    powered = dict(base, powerup=3)
    pspeed = dict(base, pspeed=0x7f)
    fast = dict(base, speed_signed=56)
    subpixel = dict(base, x_fixed=(64 << 8) | 0xe0)
    cells = {
        _adapter_physics_cell(base),
        _adapter_physics_cell(powered),
        _adapter_physics_cell(pspeed),
        _adapter_physics_cell(fast),
        _adapter_physics_cell(subpixel),
    }
    assert len(cells) == 5


class CappedGoalAdapter:
    """A level whose progress (x) caps at the goal, like SMA4 levels.

    Running advances x up to CAP and plateaus; the goal is only triggered by a
    jump action at the cap.  Stands in for the goal-card finisher contract so the
    finisher logic is exercised without a ROM.
    """

    CAP = 10
    game_id = "capped"
    level_id = "1-1"
    action_names = ["NOOP", "RIGHT+B", "RIGHT+A"]
    n_actions = 3

    def __init__(self):
        self.reset()

    def _info(self):
        return {"game_id": self.game_id, "x_pos": self.x, "touched": self.touched}

    def reset(self, seed: int = 0):
        del seed
        self.x = 0
        self.touched = False
        self._last_info = self._info()
        return dict(self._last_info)

    def close(self):
        pass

    @property
    def last_info(self):
        return dict(self._last_info)

    @property
    def last_obs(self):
        return np.full((8, 8, 3), min(self.x * 20, 255), dtype=np.uint8)

    def snapshot(self):
        return self.x, self.touched

    def restore(self, snap):
        self.x, self.touched = snap
        self._last_info = self._info()

    def step(self, action_idx: int):
        if action_idx == 1:  # RIGHT+B advances, then plateaus at CAP
            self.x = min(self.x + 1, self.CAP)
        elif action_idx == 2 and self.x >= self.CAP:  # RIGHT+A at the goal
            self.touched = True
        self._last_info = self._info()
        return None, dict(self._last_info), False

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        for _ in range(frames):
            _o, info, done = self.step(action_idx)
            if self.is_success(info):
                break
        return dict(self._last_info), done

    def is_success(self, info: dict) -> bool:
        return bool(info.get("touched", False))

    def is_death(self, info: dict, done: bool) -> bool:
        return False

    def progress(self, info: dict) -> float:
        return float(info["x_pos"])

    def cell(self, tile: int = 16):
        return self.game_id, self.x // tile, self.touched


class BrakeGoalAdapter:
    """A card finish that requires slowing down before the jump hit."""

    game_id = "brake"
    level_id = "1-2"
    action_names = ["NOOP", "RIGHT+B", "RIGHT+A", "LEFT+B"]
    n_actions = 4

    def __init__(self):
        self.reset()

    def _info(self):
        return {
            "game_id": self.game_id,
            "x_pos": self.x,
            "y_pos": 80,
            "speed": self.speed,
            "speed_signed": self.speed,
            "goalcard": int(self.x >= 18),
            "endwalk": int(self.touched),
        }

    def reset(self, seed: int = 0):
        del seed
        self.x = 0
        self.speed = 0
        self.touched = False
        self._last_info = self._info()
        return dict(self._last_info)

    def close(self):
        pass

    @property
    def last_info(self):
        return dict(self._last_info)

    @property
    def last_obs(self):
        return np.full((8, 8, 3), min(self.x * 10, 255), dtype=np.uint8)

    def snapshot(self):
        return self.x, self.speed, self.touched

    def restore(self, snap):
        self.x, self.speed, self.touched = snap
        self._last_info = self._info()

    def step(self, action_idx: int):
        if action_idx == 1:  # fast run, overshoots the hit if used straight through
            self.speed = 4
            self.x = min(self.x + self.speed, 26)
        elif action_idx == 2:  # jump hit only works when slow in the card window
            if 18 <= self.x <= 22 and self.speed <= 1:
                self.touched = True
            self.x = min(self.x + max(self.speed, 1), 26)
        elif action_idx == 3:  # brake
            self.speed = max(0, self.speed - 1)
        else:  # coast
            self.speed = max(0, self.speed - 1)
        self._last_info = self._info()
        return None, dict(self._last_info), False

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        for _ in range(frames):
            _o, info, done = self.step(action_idx)
            if self.is_success(info):
                break
        return dict(self._last_info), done

    def is_success(self, info: dict) -> bool:
        return bool(info.get("endwalk", 0))

    def is_death(self, info: dict, done: bool) -> bool:
        return False

    def progress(self, info: dict) -> float:
        return float(info["x_pos"])

    def cell(self, tile: int = 16):
        return self.game_id, self.x // tile, self.speed, self.touched


class ReturnCardAdapter:
    """A capped card finish that must settle, walk back left, then jump."""

    game_id = "return-card"
    level_id = "1-2"
    action_names = ["NOOP", "RIGHT+B", "RIGHT+A+B", "LEFT+B", "LEFT"]
    n_actions = 5

    def __init__(self):
        self.reset()

    def _info(self):
        return {
            "game_id": self.game_id,
            "x_pos": self.x,
            "y_pos": self.y,
            "speed_signed": self.speed,
            "goalcard": int(self.x >= 8),
            "endwalk": int(self.touched),
        }

    def reset(self, seed: int = 0):
        del seed
        self.x = 0
        self.y = 80
        self.speed = 0
        self.touched = False
        self._last_info = self._info()
        return dict(self._last_info)

    def close(self):
        pass

    @property
    def last_info(self):
        return dict(self._last_info)

    @property
    def last_obs(self):
        return np.full((8, 8, 3), min(self.x * 20, 255), dtype=np.uint8)

    def snapshot(self):
        return self.x, self.y, self.speed, self.touched

    def restore(self, snap):
        self.x, self.y, self.speed, self.touched = snap
        self._last_info = self._info()

    def step(self, action_idx: int):
        if action_idx == 1:
            self.speed = 2
            self.x = min(self.x + self.speed, 10)
            self.y = 80
        elif action_idx == 2:
            if self.x == 5 and self.y == 128:
                self.touched = True
            self.y = 96
        elif action_idx == 3:
            self.speed = -1
            self.x = max(0, self.x - 1)
            self.y = 128
        elif action_idx == 4:
            self.speed = -1
            self.x = max(0, self.x - 1)
            self.y = 128
        else:
            self.speed = 0
            self.y = min(128, self.y + 24)
        self._last_info = self._info()
        return None, dict(self._last_info), False

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        for _ in range(frames):
            _o, info, done = self.step(action_idx)
            if self.is_success(info):
                break
        return dict(self._last_info), done

    def is_success(self, info: dict) -> bool:
        return bool(info.get("endwalk", 0))

    def is_death(self, info: dict, done: bool) -> bool:
        return False

    def progress(self, info: dict) -> float:
        return float(info["x_pos"])

    def cell(self, tile: int = 16):
        return self.game_id, self.x // tile, self.touched


def test_goal_suffix_search_finishes_capped_level():
    adapter = CappedGoalAdapter()
    run = [1] * adapter.CAP  # reach the plateau without ever jumping
    suffix = goal_suffix_search(adapter, run, chunk_frames=1)
    assert suffix is not None
    assert suffix["chunk_frames"] == 1
    assert suffix["final_info"]["touched"]
    # the returned path actually reaches the goal on replay
    adapter.reset()
    info = adapter.last_info
    for a in suffix["path"]:
        _o, info, _d = adapter.step(a)
        if adapter.is_success(info):
            break
    assert adapter.is_success(info)


def test_goal_suffix_search_can_brake_before_card_hit():
    adapter = BrakeGoalAdapter()
    run = [1] * 5  # reaches the goal window too fast for a direct jump
    suffix = goal_suffix_search(
        adapter, run, chunk_frames=1, band=12, max_snapshots=8,
        run_action="RIGHT+B", jump_actions=("RIGHT+A",),
        delays=range(0, 1), holds=range(1, 2), tail=2,
        brake_actions=("NOOP", "LEFT+B"), brake_delays=range(0, 5),
        tail_actions=("NOOP",), card_return_action="MISSING", post_settle_frames=0)
    assert suffix is not None
    assert suffix["final_info"]["endwalk"] == 1
    assert suffix["metadata"]["template"] == "brake_jump_tail"


def test_goal_suffix_search_can_return_left_to_card_after_cap():
    adapter = ReturnCardAdapter()
    run = [1] * 5  # overshoots to the capped end area without touching the card
    suffix = goal_suffix_search(
        adapter, run, chunk_frames=1, band=4, max_snapshots=4,
        run_action="RIGHT+B", jump_actions=("RIGHT+A+B",),
        post_settle_frames=4, card_return_action="LEFT+B",
        card_return_frames=(5,), card_return_holds=(1,),
        card_return_tail_actions=("LEFT",), tail=4,
        delays=range(0, 1), holds=range(1, 2),
        brake_actions=("NOOP",), brake_delays=range(0, 1),
        tail_actions=("NOOP",))
    assert suffix is not None
    assert suffix["final_info"]["endwalk"] == 1
    assert suffix["metadata"]["template"] == "card_return_jump"
    assert len(suffix["path"]) > len(run) + 5  # includes post-settle frames before returning left

    adapter.reset()
    info = adapter.last_info
    for action in suffix["path"]:
        _o, info, _d = adapter.step(action)
        if adapter.is_success(info):
            break
    assert adapter.is_success(info)


def test_goal_suffix_search_returns_none_without_run_action():
    # FakeAdapter has no RIGHT+B/RIGHT+A action names, so the finisher cannot apply.
    assert goal_suffix_search(FakeAdapter(), [1, 1, 1], chunk_frames=1) is None


def test_coverage_search_adapter_solves_fake():
    result = coverage_search_adapter(FakeAdapter(), beam_width=2, chunk_frames=1, max_depth=6)
    assert result.solved
    assert result.final_info["flag_get"]


def test_coverage_search_adapter_solves_capped_goal():
    # Reaches the plateau then triggers the goal jump at the cap (no finisher needed).
    result = coverage_search_adapter(CappedGoalAdapter(), beam_width=3, chunk_frames=1,
                                     max_depth=20)
    assert result.solved
    assert result.final_info["touched"]


def test_coverage_novelty_is_committed_only_for_retained_frontier():
    for order in itertools.permutations(("high", "target", "finish")):
        result = coverage_search_adapter(
            NoveltyRetentionAdapter(order),
            beam_width=1,
            chunk_frames=1,
            max_depth=4,
            stuck_cap=4,
            tile=1,
            physics_cell=False,
        )

        assert result.solved
        assert result.final_info["flag_get"]
        assert tuple(order[idx] for idx in result.path) == (
            "high",
            "target",
            "finish",
        )


def test_coverage_search_prunes_loop_to_generated_cell(monkeypatch):
    sim = PrunedCellLoopSim()
    monkeypatch.setattr(
        search_module, "_new_nes_sim", lambda *_args, **_kwargs: sim
    )

    result = search_module.coverage_search(
        actions=[(), ()],
        beam_width=1,
        chunk_frames=1,
        max_depth=3,
        cov_bonus=1000,
        loop_back_px=200,
        loop_needs_visited=True,
        progress_every=0,
    )

    assert not result.solved
    assert sim.expanded_from == [
        "root",
        "root",
        "high",
        "high",
        "high",
        "high",
    ]


def test_native_coverage_novelty_is_committed_only_for_retained_frontier(
        monkeypatch):
    sim = NativeNoveltyRetentionSim()
    monkeypatch.setattr(
        search_module, "_new_nes_sim", lambda *_args, **_kwargs: sim
    )

    result = search_module.coverage_search(
        actions=[(), (), ()],
        beam_width=1,
        chunk_frames=1,
        max_depth=4,
        stuck_cap=4,
        tile=1,
        cov_bonus=50,
        progress_every=0,
    )

    assert result.solved
    assert result.path == [0, 1, 2]
    assert result.final_info["flag_get"]


def test_area_search_pruned_cell_remains_loop_evidence(monkeypatch):
    sim = PrunedCellLoopSim()
    monkeypatch.setattr(
        search_module, "_new_nes_sim", lambda *_args, **_kwargs: sim
    )

    changed, path, info = search_module.area_search(
        1,
        1,
        actions=[(), ()],
        beam_width=1,
        chunk_frames=1,
        max_depth=2,
        x_jump_px=200,
        cov_bonus=50,
        progress_every=0,
    )

    assert not changed
    assert path == [0]
    assert info == {}


def test_coverage_pipe_macro_counts_executed_chunks(monkeypatch):
    sim = FakePipeMacroSim()
    monkeypatch.setattr(search_module, "_new_nes_sim", lambda *_a, **_kw: sim)

    result = search_module.coverage_search(
        actions=[()] * 8,
        beam_width=1,
        chunk_frames=4,
        max_depth=1,
        stuck_cap=4,
        pipe_macro_chunks=3,
        progress_every=0,
    )

    assert result.solved
    assert result.path == [7, 7]
    assert result.frames == 8
    assert result.nodes_expanded == 10
    assert sim.total_run_calls == 10
    assert sim.closed


def test_coverage_pipe_macro_child_keeps_actual_chunk_accounting(monkeypatch):
    sim = FakePipeMacroSim(goal_on_entry=False)
    monkeypatch.setattr(search_module, "_new_nes_sim", lambda *_a, **_kw: sim)

    result = search_module.coverage_search(
        actions=[()] * 8,
        beam_width=1,
        chunk_frames=4,
        max_depth=2,
        stuck_cap=4,
        pipe_macro_chunks=3,
        progress_every=0,
    )

    assert result.solved
    assert result.path == [7, 7, 0]
    assert result.frames == 12
    assert result.nodes_expanded == 11
    assert sim.total_run_calls == 11
    assert sim.closed


def test_make_contact_sheet_adapter_renders(tmp_path):
    out = tmp_path / "sheet.png"
    meta = make_contact_sheet_adapter(CappedGoalAdapter(), [1] * 10, 1, out,
                                      cols=3, rows=3)
    assert out.exists()
    assert meta["n_records"] == 11
    assert meta["x_max"] == CappedGoalAdapter.CAP
