"""Non-ROM tests for SMB3 overworld map-node discovery and the post-clear settle.

Uses small fake map cores (no emulator) to guard the BFS/discovery/settle logic
that the SMA4 meta-planner relies on.  The full two-tier `overworld_solve` is
verified against the real ROM.
"""
from __future__ import annotations

from mario.overworld_search import (MapGraph, discover_nodes, remap_solution_path,
                                    replay_cached_solution, settle_to_map)


class FakeMapCore:
    """A tiny linear overworld: cursor x in 0..2; (1,0) and (2,0) are levels.

    `unlock` delays cursor responsiveness to model the post-clear results screen
    (mode reads 'overworld' but input is ignored until then).
    """

    LEVELS = {(1, 0), (2, 0)}

    def __init__(self, unlock: int = 0):
        self.cx = 0
        self.cy = 0
        self.mode = "overworld"
        self.x_pos = 0
        self.frame = 0
        self.unlock = unlock
        self._last = ()

    @property
    def last_info(self):
        return {"cursor": (self.cx, self.cy), "mode": self.mode, "x_pos": self.x_pos}

    def snapshot(self):
        return (self.cx, self.cy, self.mode, self.x_pos, self.frame, self._last)

    def restore(self, snap):
        self.cx, self.cy, self.mode, self.x_pos, self.frame, self._last = snap

    def _step_buttons(self, buttons=()):
        self.frame += 1
        # edge-triggered: a held direction moves at most one node (until released)
        is_press = bool(buttons) and buttons != self._last
        if is_press and self.frame >= self.unlock and self.mode == "overworld":
            if "RIGHT" in buttons:
                self.cx = min(self.cx + 1, 2)
            elif "LEFT" in buttons:
                self.cx = max(self.cx - 1, 0)
        self._last = buttons
        return None, self.last_info, False

    def enter_level(self, **kw):
        if (self.cx, self.cy) in self.LEVELS:
            self.mode = "level"
            self.x_pos = 20
        return self.last_info


class FakeOW:
    def __init__(self, core):
        self.core = core


class FakeReplayCore:
    action_names = ["NOOP", "RIGHT+B", "LEFT", "RIGHT+A"]

    def __init__(self):
        self.cursor = (1, 0)
        self.mode = "level"
        self.x = 0
        self.endwalk = 0
        self.frame = 0
        self._last = ()

    @property
    def last_info(self):
        return {
            "cursor": self.cursor,
            "mode": self.mode,
            "x_pos": self.x,
            "endwalk": self.endwalk,
            "pspeed": int(self.x >= 2),
            "speed_signed": 4,
        }

    def snapshot(self):
        return self.cursor, self.mode, self.x, self.endwalk, self.frame, self._last

    def restore(self, snap):
        self.cursor, self.mode, self.x, self.endwalk, self.frame, self._last = snap

    def run_chunk(self, action_idx: int, frames: int):
        done = False
        for _ in range(frames):
            if self.mode == "level":
                if self.action_names[action_idx] == "RIGHT+B":
                    self.x += 1
                elif self.action_names[action_idx] == "RIGHT+A" and self.x >= 2:
                    self.endwalk = 255
                    self.mode = "clearing"
            if self.is_success(self.last_info):
                break
        return dict(self.last_info), done

    def _step_buttons(self, buttons=()):
        self.frame += 1
        if self.mode == "clearing" and self.frame >= 3:
            self.mode = "overworld"
        is_press = bool(buttons) and buttons != self._last
        if self.mode == "overworld" and is_press:
            x, y = self.cursor
            if "RIGHT" in buttons:
                self.cursor = (x + 1, y)
            elif "LEFT" in buttons:
                self.cursor = (x - 1, y)
        self._last = buttons
        return None, dict(self.last_info), False

    def is_success(self, info):
        return int(info.get("endwalk", 0)) != 0

    def is_death(self, info, done):
        return False


def test_discover_finds_enterable_levels():
    g = discover_nodes(FakeOW(FakeMapCore()), max_nodes=16)
    assert isinstance(g, MapGraph)
    assert set(g.snaps) == {(0, 0), (1, 0), (2, 0)}
    assert g.levels() == [(1, 0), (2, 0)]
    assert g.edges[(0, 0)]["RIGHT"] == (1, 0)


def test_discover_enter_test_is_non_destructive():
    core = FakeMapCore()
    discover_nodes(FakeOW(core), max_nodes=16)
    # discovery leaves the core on the overworld (enter tests are rolled back)
    assert core.last_info["mode"] == "overworld"


def test_settle_to_map_waits_out_unresponsive_screen():
    core = FakeMapCore(unlock=130)
    assert settle_to_map(core, max_frames=600, probe_every=60)
    assert core.last_info["mode"] == "overworld"


def test_cached_solution_path_remaps_action_names():
    solution = {
        "action_names": ["NOOP", "RIGHT+A", "RIGHT+B"],
        "path": [2, 2, 1],
    }
    path, meta = remap_solution_path(solution, ["NOOP", "RIGHT+B", "LEFT", "RIGHT+A"])
    assert path == [1, 1, 3]
    assert meta["remapped"]


def test_replay_cached_solution_restores_entry_and_settles_to_map():
    core = FakeReplayCore()
    entry = core.snapshot()
    core.x = 99
    solution = {
        "level_id": "1-test",
        "action_names": ["NOOP", "RIGHT+A", "RIGHT+B"],
        "path": [2, 2, 1],
        "chunk_frames": 1,
    }
    summary = replay_cached_solution(core, solution, entry_snap=entry)
    assert summary["solved"]
    assert summary["settled_to_map"]
    assert summary["success_at"] == 3
    assert summary["x_max"] == 2
    assert core.last_info["mode"] == "overworld"
