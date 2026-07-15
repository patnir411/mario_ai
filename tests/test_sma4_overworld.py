"""SMA4 overworld-view adapter checks (ROM-gated, slow).

Confirms the reverse-engineered map RAM and the overworld<->level transition that
the two-tier meta-planner relies on.  Requires MARIO_AI_SMA4_ROM + stable_retro;
skipped otherwise.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from mario.adapters import SMA4OverworldAdapter

# SMA4_OVERWORLD_ACTIONS = [NOOP, LEFT, RIGHT, UP, DOWN, A]
NOOP, LEFT, RIGHT, UP, DOWN, A = range(6)
NODE_1_1 = (64, 32)  # World-1 1-1 cursor node (probe-confirmed)

pytestmark = pytest.mark.slow


def _rom_path():
    path = os.environ.get("MARIO_AI_SMA4_ROM")
    if not path:
        pytest.skip("MARIO_AI_SMA4_ROM is not set")
    pytest.importorskip("stable_retro")
    return path


@pytest.fixture
def ow():
    adapter = SMA4OverworldAdapter(_rom_path())
    try:
        yield adapter
    finally:
        adapter.close()


def _iwram_hash(adapter) -> str:
    return hashlib.sha256(bytes(adapter.env.data.memory.blocks[0x03000000])).hexdigest()


def _advance(adapter, actions):
    for a in actions:
        adapter.step(a)


def test_overworld_reset_reports_overworld_mode(ow):
    info = ow.reset()
    assert info["game_id"] == "sma4"
    assert info["level_id"] == "overworld"
    assert info["mode"] == "overworld"
    assert info["world"] >= 1
    assert ow.n_actions == len(ow.action_names)
    cx, cy = info["cursor"]
    assert cx % 0x20 == 0 and cy % 0x20 == 0


def test_overworld_snapshot_roundtrip_identical(ow):
    ow.reset()
    _advance(ow, [RIGHT] + [NOOP] * 10)
    snap = ow.snapshot()
    suffix = [UP] + [NOOP] * 10
    _advance(ow, suffix)
    h1 = _iwram_hash(ow)

    ow.restore(snap)
    _advance(ow, suffix)
    h2 = _iwram_hash(ow)
    assert h1 == h2


def test_overworld_cursor_moves_deterministically(ow):
    base = ow.reset()["cursor"]
    _advance(ow, [RIGHT] + [NOOP] * 12)
    moved = ow.last_info["cursor"]
    assert moved != base
    assert moved[0] > base[0]  # RIGHT increments the X grid
    assert moved[1] == base[1]


def test_overworld_enter_level_detects_mode_transition(ow):
    # The success criterion: navigate to a level node and enter it.
    ow.reset()
    assert ow.goto_node(NODE_1_1), f"could not reach {NODE_1_1}"
    info = ow.enter_level()
    assert info["mode"] == "level"
    assert ow.is_success(info)
    assert info["x_pos"] > 5  # player actually spawned in the level


def test_overworld_cleared_bitmap_readable(ow):
    info = ow.reset()
    assert isinstance(info["cleared"], int)
    assert ow.cleared_mask() == info["cleared"]
    # progress popcount drives meta-search progress and must be finite/non-negative
    assert ow.progress(info) >= 0
