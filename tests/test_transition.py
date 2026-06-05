"""Gate for `real_transition` — the corrected pipe/area-transition detector.

The gym wrapper fast-forwards pipe transitions inside env.step(), so `pipe_entering` read after
the step is blind to entries whose destination shares area bytes (2-2). `real_transition` must
fire on the 2-2 side-pipe entry (via the post-step x-jump / info-vs-RAM mismatch), must NOT fire
on a normal forward step (1-1), and must suppress a maze-loop teleport (visited-cell discriminator).
"""
from __future__ import annotations

import json
from pathlib import Path

from mario.env import MarioSim
from mario.ram import (real_transition, real_transition_strict, mario_level_x,
                       AREA_NUMBER, AREA_POINTER, MARIO_Y_ON_SCREEN)

ROOT = Path(__file__).resolve().parent.parent
SOL = ROOT / "data" / "solutions"


def _cell(ram, tile=16):
    return (int(ram[AREA_NUMBER]), mario_level_x(ram) // tile, int(ram[MARIO_Y_ON_SCREEN]) // tile)


def _replay_with_detector(world, stage):
    """Replay a cached solution step-by-step; return list of (fired, reason) and final flag."""
    d = json.loads((SOL / f"{world}-{stage}.json").read_text())
    path, cf = d["path"], d.get("chunk_frames", 8)
    sim = MarioSim(world, stage); sim.reset(seed=0)
    info_before = dict(sim.last_info); ram_before = sim.ram.copy()
    visited = {_cell(sim.ram)}
    fires, final_flag = [], False
    for a in path:
        info_after, done = sim.run_chunk(a, cf)
        ram_after = sim.ram
        fired, reason = real_transition(info_before, ram_before, info_after, ram_after,
                                        visited_cells=visited)
        if fired:
            fires.append(reason)
        visited.add(_cell(sim.ram))
        info_before, ram_before = dict(info_after), ram_after.copy()
        if info_after.get("flag_get"):
            final_flag = True
        if done:
            break
    sim.close()
    return fires, final_flag


def test_22_side_pipe_fires_and_flags():
    """2-2: the side-pipe entry must be detected (x_jump / info_ram_mismatch) and the run beats."""
    fires, flag = _replay_with_detector(2, 2)
    assert flag, "2-2 cached solution should reach the flag"
    assert any(r in ("x_jump", "info_ram_mismatch", "area_key", "stage", "stage_ram")
               for r in fires), f"expected a transition fire on 2-2 entry, got {fires}"


def test_strict_fires_on_22_without_area_key():
    """The STRICT detector (no bare area_key) must still fire on 2-2's real side-pipe entry."""
    d = json.loads((SOL / "2-2.json").read_text())
    path, cf = d["path"], d.get("chunk_frames", 8)
    sim = MarioSim(2, 2); sim.reset(seed=0)
    info_before = dict(sim.last_info); ram_before = sim.ram.copy()
    visited = {_cell(sim.ram)}; fires = []
    for a in path:
        info_after, done = sim.run_chunk(a, cf)
        fired, reason = real_transition_strict(info_before, ram_before, info_after, sim.ram,
                                               visited_cells=visited)
        if fired:
            fires.append(reason)
        visited.add(_cell(sim.ram))
        info_before, ram_before = dict(info_after), sim.ram.copy()
        if done:
            break
    sim.close()
    assert any(r in ("x_jump", "info_ram_mismatch", "stage", "stage_ram", "flag")
               for r in fires), f"strict detector should fire on 2-2 entry, got {fires}"


def test_strict_rejects_bare_marker_flip():
    """A bare $0750/$0760 (area_key) flip with NO teleport/stage/flag is a PENDING marker, not an
    entry: the OLD detector false-fires ('area_key'); the STRICT one must NOT fire."""
    from mario.ram import CHANGE_AREA_TIMER, GAME_ENGINE_SUBROUTINE
    sim = MarioSim(8, 4); sim.reset(seed=0)
    before = sim.ram.copy(); after = sim.ram.copy()
    # isolate the bare-marker case: normal engine (not mid-entry), position unchanged
    for r in (before, after):
        r[CHANGE_AREA_TIMER] = 0; r[GAME_ENGINE_SUBROUTINE] = 8
    after[AREA_POINTER] = (int(before[AREA_POINTER]) + 1) & 0xFF   # ONLY the marker flips
    # info consistent with RAM x (no teleport) so info_ram_mismatch can't false-fire
    info = {"world": 8, "stage": 4, "x_pos": mario_level_x(after), "flag_get": False}
    old_fired, old_reason = real_transition(info, before, info, after, visited_cells=set())
    strict_fired, _ = real_transition_strict(info, before, info, after, visited_cells=set())
    assert old_fired and old_reason == "area_key", "sanity: old detector false-fires on bare marker"
    assert not strict_fired, "STRICT detector must reject a bare area-key/$0750 marker flip"


def test_11_forward_steps_do_not_fire():
    """1-1: normal forward play must not trigger a (false) transition in the first ~30 chunks."""
    d = json.loads((SOL / "1-1.json").read_text())
    path, cf = d["path"], d.get("chunk_frames", 8)
    sim = MarioSim(1, 1); sim.reset(seed=0)
    info_before = dict(sim.last_info); ram_before = sim.ram.copy()
    visited = {_cell(sim.ram)}
    for a in path[:30]:
        info_after, done = sim.run_chunk(a, cf)
        fired, reason = real_transition(info_before, ram_before, info_after, sim.ram,
                                        visited_cells=visited)
        assert not fired, f"false transition on 1-1 forward step: {reason}"
        visited.add(_cell(sim.ram))
        info_before, ram_before = dict(info_after), sim.ram.copy()
        if done:
            break
    sim.close()


def test_loop_vs_pipe_discriminator():
    """A large backward x-jump fires ONLY when the landing cell is unvisited (pipe), not when it
    returns to a visited cell (maze loop)."""
    sim = MarioSim(1, 1); sim.reset(seed=0)
    info = dict(sim.last_info)
    before = sim.ram.copy()
    after = sim.ram.copy()
    after[AREA_NUMBER] = before[AREA_NUMBER]          # same area
    # synthetic ~200px backward jump: before at page 2, after at page 1 (level-x -= 256 + nudge)
    before[0x006D] = 2; before[0x0086] = 100          # level-x = 612
    after[0x006D] = 1;  after[0x0086] = 156           # level-x = 412  (backward 200)
    # consistent info dicts so signal (7) info_ram_mismatch doesn't fire spuriously
    info_b = {"world": 1, "stage": 1, "x_pos": 612, "flag_get": False}
    info_a = {"world": 1, "stage": 1, "x_pos": 412, "flag_get": False}
    landing = _cell(after)
    fired_new, reason_new = real_transition(info_b, before, info_a, after, visited_cells=set())
    fired_loop, _ = real_transition(info_b, before, info_a, after, visited_cells={landing})
    assert fired_new and reason_new == "x_jump", "unvisited backward jump should fire as x_jump"
    assert not fired_loop, "backward jump into a VISITED cell is a loop, must be suppressed"
