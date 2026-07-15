from __future__ import annotations

import hashlib
import os

import pytest

from mario.adapters import SMLAdapter


def _rom_path():
    path = os.environ.get("MARIO_AI_SML_ROM")
    if not path:
        pytest.skip("MARIO_AI_SML_ROM is not set")
    pytest.importorskip("pyboy")
    return path


@pytest.fixture
def sml():
    adapter = SMLAdapter(_rom_path())
    try:
        yield adapter
    finally:
        adapter.close()


def _ram_hash(adapter: SMLAdapter) -> str:
    # WRAM/HRAM are enough for deterministic state checks and much cheaper than hashing ROM.
    data = bytes(int(adapter.ram[i]) for i in range(0xC000, 0x10000))
    return hashlib.sha256(data).hexdigest()


def _advance(adapter: SMLAdapter, actions: list[int], frames: int = 2):
    for action in actions:
        info, done = adapter.run_chunk(action, frames)
        if done or adapter.is_success(info) or adapter.is_death(info, done):
            break


def test_sml_reset_normalizes_info(sml):
    info = sml.reset(seed=0)
    assert info["game_id"] == "sml"
    assert info["world"] >= 1
    assert info["stage"] >= 1
    assert isinstance(info["x_pos"], int)
    assert isinstance(info["lives"], int)
    assert sml.n_actions == len(sml.action_names)


def test_sml_snapshot_roundtrip_identical(sml):
    _advance(sml, [1, 2, 3, 4, 0])
    snap = sml.snapshot()
    suffix = [1, 1, 3, 4, 0, 2]
    _advance(sml, suffix)
    h1 = _ram_hash(sml)

    sml.restore(snap)
    _advance(sml, suffix)
    h2 = _ram_hash(sml)
    assert h1 == h2


def test_sml_snapshot_has_no_side_effects(sml):
    _advance(sml, [1, 2, 3])
    before = _ram_hash(sml)
    _ = sml.snapshot()
    after = _ram_hash(sml)
    assert before == after


def test_sml_snapshot_reusable(sml):
    _advance(sml, [1, 2, 3])
    snap = sml.snapshot()

    sml.restore(snap)
    _advance(sml, [1, 4, 4])
    h1 = _ram_hash(sml)

    sml.restore(snap)
    _advance(sml, [1, 4, 4])
    h2 = _ram_hash(sml)
    assert h1 == h2


@pytest.mark.slow
def test_sml_1_1_solve_slow():
    if os.environ.get("MARIO_AI_RUN_SLOW_SML") != "1":
        pytest.skip("set MARIO_AI_RUN_SLOW_SML=1 to run the SML solve smoke test")
    adapter = SMLAdapter(_rom_path())
    try:
        from mario.search import beam_search_adapter

        result = beam_search_adapter(adapter, beam_width=80, chunk_frames=4,
                                     max_depth=1200, stuck_cap=36)
        assert result.solved
        adapter.reset(seed=0)
        info = adapter.last_info
        done = False
        for action in result.path:
            info, done = adapter.run_chunk(action, result.chunk_frames)
            if adapter.is_success(info) or adapter.is_death(info, done):
                break
        assert adapter.is_success(info)
    finally:
        adapter.close()
