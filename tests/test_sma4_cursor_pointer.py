from __future__ import annotations

from mario.adapters import SMA4Adapter


def _adapter(values: dict[int, int]) -> SMA4Adapter:
    adapter = object.__new__(SMA4Adapter)
    adapter._read_u8 = lambda address: int(values.get(address, 0))

    def read_u32(address: int) -> int:
        return sum(int(values.get(address + offset, 0)) << (8 * offset)
                   for offset in range(4))

    adapter._read_u32 = read_u32
    return adapter


def _write_u32(values: dict[int, int], address: int, value: int) -> None:
    for offset in range(4):
        values[address + offset] = (int(value) >> (8 * offset)) & 0xFF


def test_sma4_cursor_resolver_follows_relocated_live_object():
    values = {
        SMA4Adapter.MAP_CURSOR_X: 14,
        SMA4Adapter.MAP_CURSOR_Y: 135,
    }
    relocated = 0x03004EF8
    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, relocated)
    values[relocated] = 32
    values[relocated + 4] = 160
    adapter = _adapter(values)

    info = adapter._resolve_map_cursor(time=0, world_raw=0, map_event=0x11)

    assert info["cursor"] == (160, 32)
    assert info["source"] == "pointer"
    assert info["raw_pointer"] == relocated
    assert info["resolved_pointer"] == relocated
    assert info["legacy"] == (14, 135)


def test_sma4_cursor_resolver_keeps_animation_coordinates_between_grid_cells():
    values: dict[int, int] = {}
    base = SMA4Adapter.MAP_CURSOR_Y
    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, base)
    values[base] = 80
    values[base + 4] = 130
    adapter = _adapter(values)

    info = adapter._resolve_map_cursor(time=0, world_raw=8, map_event=0x11)

    assert info["cursor"] == (130, 80)
    assert info["source"] == "pointer"


def test_sma4_cursor_resolver_labels_transient_null_instead_of_splicing_cache():
    values = {
        SMA4Adapter.MAP_CURSOR_X: 14,
        SMA4Adapter.MAP_CURSOR_Y: 135,
    }
    relocated = 0x03004EF8
    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, relocated)
    values[relocated] = 144
    values[relocated + 4] = 128
    adapter = _adapter(values)
    assert adapter._resolve_map_cursor(
        time=0, world_raw=8, map_event=0x11)["cursor"] == (128, 144)

    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, 0)
    values[relocated] = 80
    values[relocated + 4] = 32
    transient = adapter._resolve_map_cursor(
        time=0, world_raw=7, map_event=0x0A)

    assert transient["cursor"] == (0, 0)
    assert transient["source"] == "transient_null_pointer"
    assert transient["raw_pointer"] == 0
    assert transient["resolved_pointer"] is None

    # Even a structurally valid IWRAM address must not be trusted until its
    # cursor-object layout has been observed and allowlisted.
    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, 0x03004000)
    invalid = adapter._resolve_map_cursor(
        time=0, world_raw=7, map_event=0x11)
    assert invalid["cursor"] == (0, 0)
    assert invalid["source"] == "unrecognized_pointer"
    assert invalid["resolved_pointer"] is None
    assert invalid["pointer_in_iwram"] is True
    assert invalid["pointer_recognized"] is False


def test_sma4_cursor_pointer_is_ignored_outside_map_context():
    values = {
        SMA4Adapter.MAP_CURSOR_X: 14,
        SMA4Adapter.MAP_CURSOR_Y: 135,
    }
    relocated = 0x03004EF8
    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, relocated)
    values[relocated] = 32
    values[relocated + 4] = 160
    adapter = _adapter(values)

    info = adapter._resolve_map_cursor(time=300, world_raw=0, map_event=0x1A)

    assert info["cursor"] == (14, 135)
    assert info["source"] == "legacy_fallback"


def test_sma4_cursor_info_uses_normalized_three_byte_timer():
    values = {
        SMA4Adapter.TIME: 0,
        SMA4Adapter.TIME + 1: 0,
        SMA4Adapter.TIME + 2: 0,
        # The adjacent byte is not part of Stable-Retro's >u3 timer field.
        SMA4Adapter.TIME + 3: 1,
        SMA4Adapter.WORLD: 0,
        SMA4Adapter.MAP_EVENT: 0x11,
    }
    base = SMA4Adapter.MAP_CURSOR_Y
    _write_u32(values, SMA4Adapter.MAP_CURSOR_PTR, base)
    values[base] = 32
    values[base + 4] = 64
    adapter = _adapter(values)
    adapter._last_info = {"time": 0}

    info = adapter.map_cursor_info()

    assert info["resolved"] is True
    assert info["cursor"] == (64, 32)


def test_sma4_cursor_pointer_validation_stays_inside_iwram():
    assert SMA4Adapter._pointer_in_iwram(0x03000000)
    assert SMA4Adapter._pointer_in_iwram(0x03007FF8)
    assert not SMA4Adapter._pointer_in_iwram(0x03007FFC)
    assert not SMA4Adapter._pointer_in_iwram(0x03004EFA)
    assert not SMA4Adapter._pointer_in_iwram(0x02000000)
