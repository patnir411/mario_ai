"""Super Mario Bros (NES) RAM map — verified addresses (Data Crystal).

Used for compact observations and, where richer than the env `info` dict, for reward
and death-cause diagnosis. Read against the live 2048-byte `sim.ram` uint8 array.

IMPORTANT (Tom7 lesson, DESIGN.md §6): do NOT feed counter-style addresses (score,
timer, coins) as "progress" signals — they create fake monotone progress that traps
search in local optima. Those are intentionally excluded from the reward path.
"""
from __future__ import annotations

# --- player -------------------------------------------------------------
MARIO_LEVEL_PAGE = 0x006D   # horizontal page (level)
MARIO_X_ON_SCREEN = 0x0086  # x within current screen
MARIO_Y_ON_SCREEN = 0x00CE  # y on screen
MARIO_X_SPEED = 0x0057      # signed horizontal speed
MARIO_Y_VELOCITY = 0x009F   # signed vertical velocity
PLAYER_FLOAT_STATE = 0x001D  # 0 ground/coasting, 1 jumping/mid-swim-stroke, 2 ledge, 3 flagpole
PLAYER_STATE = 0x000E       # climbing/pipe/dying/transforming
POWERUP_STATE = 0x0756      # 0 small, 1 big, >=2 fiery
PLAYER_VIABLE = 0x000E      # alias for player state (death animation == 0x06/0x0B)
FACING_DIR = 0x0033         # PlayerFacingDir: 1 = right, 2 = left (side-pipe entry needs ==1)
SWIMMING_FLAG = 0x0704      # nonzero while underwater (forces a swim default for $001D each frame)

# --- enemies (5 slots) --------------------------------------------------
ENEMY_X_LEVEL = range(0x006E, 0x0073)   # horizontal position in level
ENEMY_Y_ON_SCREEN = range(0x00CF, 0x00D4)
ENEMY_TYPE = range(0x0016, 0x001B)
ENEMY_ACTIVE = range(0x000F, 0x0014)

# --- level layout -------------------------------------------------------
TILE_DATA = range(0x0500, 0x06A0)  # current on-screen tile grid in RAM
SCREEN_IN_LEVEL = 0x071A

# --- area / transition signals ------------------------------------------
# CORRECTED transition semantics (verified by Codex against the SMB disassembly, Jun-4):
#   * $0750 / $0751 are PENDING-DESTINATION markers (set by row-$0e objects scrolling in) —
#     "a pipe destination is queued", NOT "Mario entered a pipe". Keying success on a bare
#     $0750 flip gives false positives.
#   * $0760 (AREA_NUMBER) is OFTEN STABLE across sub-area/pipe transitions — not the signal.
#   * DECISIVE signals: a real pipe/area entry is underway when CHANGE_AREA_TIMER ($06DE) != 0
#     OR GameEngineSubroutine ($000E) is in the pipe/transition states {2,3}; a level/stage
#     actually ADVANCED when the STAGE byte ($075c) increments — the right completion signal
#     for flagpole-less WATER levels (2-2/7-2 exit via a pipe -> next stage, no flag slide).
AREA_NUMBER = 0x0760
AREA_POINTER = 0x0750        # PENDING area-object-set pointer (destination marker, not entry)
CHANGE_AREA_TIMER = 0x06DE   # nonzero while an area/pipe transition is ACTUALLY in progress
GAME_ENGINE_SUBROUTINE = 0x000E  # pipe/transition states {2,3} == real entry underway
STAGE_NUMBER = 0x075C        # current stage-1 (increments on level complete; +1 to display)
WORLD_NUMBER = 0x075F        # current world-1
Y_HIGH_POS = 0x00B5          # vertical "page" of Mario's Y (stays 1 in 8-4 area-1)
PIPE_TRANSITION_STATES = (2, 3)  # GameEngineSubroutine values during a pipe/area entry


def mario_level_x(ram) -> int:
    """Absolute x position in the level = page*256 + x_on_screen."""
    return int(ram[MARIO_LEVEL_PAGE]) * 256 + int(ram[MARIO_X_ON_SCREEN])


def area(ram) -> int:
    """Raw area byte ($0760). NOTE: in gym-super-mario-bros this is often STABLE across a
    level's sub-area (pipe/room) transitions — use `area_pointer` for those."""
    return int(ram[AREA_NUMBER])


def area_pointer(ram) -> int:
    """Area-object-set pointer ($0750). EMPIRICALLY this is the byte that changes on a
    pipe/sub-area transition in this env (verified on 1-2, 4-2), whereas $0760 stays put.
    This is the correct 'entered a new room' signal for maze castles (8-4)."""
    return int(ram[AREA_POINTER])


def area_key(ram) -> tuple:
    """Combined area identity = (raw area, area pointer) — distinct per sub-area/room.

    NOTE: $0750 here is a PENDING-destination marker, so this key can change BEFORE Mario
    actually enters a pipe. For a decisive "really transitioned" test use `stage_key` +
    `pipe_entering` instead (see corrected semantics above)."""
    return (int(ram[AREA_NUMBER]), int(ram[AREA_POINTER]))


def stage_key(ram) -> tuple:
    """(world, stage) as stored (0-based). Increments on a REAL level completion — the
    correct success signal for flagpole-less water levels (2-2/7-2 exit via pipe)."""
    return (int(ram[WORLD_NUMBER]), int(ram[STAGE_NUMBER]))


def pipe_entering(ram) -> bool:
    """True iff a real pipe/area entry is in progress RIGHT NOW (decisive, per disassembly):
    ChangeAreaTimer ($06DE) nonzero OR GameEngineSubroutine ($000E) in a transition state.
    Distinct from the $0750 pending-destination marker, which only means 'queued'."""
    return int(ram[CHANGE_AREA_TIMER]) != 0 or int(ram[GAME_ENGINE_SUBROUTINE]) in PIPE_TRANSITION_STATES


def real_transition(info_before: dict, ram_before, info_after: dict, ram_after,
                    *, visited_cells=None, x_jump_px: int = 100, tile: int = 16):
    """Decide whether a REAL area/level transition happened across ONE wrapped env.step.

    THE detection-bug fix: gym_super_mario_bros runs _skip_change_area / _skip_occupied_states
    INSIDE env.step(), fast-forwarding a pipe/area transition before live RAM is read — so the
    transient $06DE/$000E are already cleaned up and `pipe_entering(ram_after)` is usually blind
    (verified on 2-2, whose area-2 also keeps $0760/$0750/$075c). This combines only
    POST-step-observable signals to catch the entry anyway, and distinguishes a real entry from a
    maze-LOOP teleport (8-4 page16->page12) via the visited-cell check.

    Returns (fired: bool, reason: str). `visited_cells`: set of cells `(area, x//tile, y//tile)`
    already seen by the caller — if the backward-jump lands in a visited cell it's a LOOP (suppress);
    if None, any large backward jump counts (the 2-2 / linear case).
    """
    if info_after.get("flag_get"):
        return True, "flag"
    wb, sb = info_before.get("world"), info_before.get("stage")
    wa, sa = info_after.get("world"), info_after.get("stage")
    if (wa, sa) != (wb, sb) and wa is not None:
        return True, "stage"
    if stage_key(ram_after) != stage_key(ram_before):
        return True, "stage_ram"
    if area_key(ram_after) != area_key(ram_before):
        return True, "area_key"
    if pipe_entering(ram_after):
        return True, "pipe_live"
    xb, xa = mario_level_x(ram_before), mario_level_x(ram_after)
    if (xb - xa) > x_jump_px:                      # large BACKWARD jump = teleport
        cell = (int(ram_after[AREA_NUMBER]), xa // tile, int(ram_after[MARIO_Y_ON_SCREEN]) // tile)
        if visited_cells is None or cell not in visited_cells:
            return True, "x_jump"                 # landed somewhere new -> pipe/area entry
        # else: returned to a visited cell -> maze loop, NOT a transition
    info_x = info_after.get("x_pos")               # gym `info` x_pos is the PRE-skip frame value
    if info_x is not None and abs(int(info_x) - xa) > x_jump_px:
        return True, "info_ram_mismatch"
    return False, ""


def real_transition_strict(info_before: dict, ram_before, info_after: dict, ram_after,
                           *, visited_cells=None, x_jump_px: int = 100, tile: int = 16):
    """STRICT variant of `real_transition` — drops the bare `area_key` ($0750/$0760) signal,
    which is a FALSE POSITIVE (Codex/disasm + ram.py docs: $0750 is a PENDING-destination marker
    set by row-$0e objects scrolling in, NOT a pipe entry). A bare marker flip while Mario keeps
    walking is NOT a transition. A *real* entry instead shows as one of the position/engine
    signals below (the gym in-step skip teleports Mario, so a down-pipe entry surfaces as a large
    info.x_pos-vs-live-RAM-x mismatch even though the area bytes also change).

    Real entry iff: flag | (world,stage) change | stage byte advance | pipe_entering
    ($06DE!=0 / $000E in {2,3}) | a backward x-jump into an UNVISITED cell (2-2-class side-pipe;
    a VISITED cell = maze loop, suppressed) | |info.x_pos - live-RAM-x| > x_jump_px (the down-pipe
    FORWARD teleport during the skip). Returns (fired, reason)."""
    if info_after.get("flag_get"):
        return True, "flag"
    wb, sb = info_before.get("world"), info_before.get("stage")
    wa, sa = info_after.get("world"), info_after.get("stage")
    if (wa, sa) != (wb, sb) and wa is not None:
        return True, "stage"
    if stage_key(ram_after) != stage_key(ram_before):
        return True, "stage_ram"
    if pipe_entering(ram_after):
        return True, "pipe_live"
    xb, xa = mario_level_x(ram_before), mario_level_x(ram_after)
    if (xb - xa) > x_jump_px:                       # large BACKWARD jump
        cell = (int(ram_after[AREA_NUMBER]), xa // tile, int(ram_after[MARIO_Y_ON_SCREEN]) // tile)
        if visited_cells is None or cell not in visited_cells:
            return True, "x_jump"
    info_x = info_after.get("x_pos")
    if info_x is not None and abs(int(info_x) - xa) > x_jump_px:
        return True, "info_ram_mismatch"           # catches the down-pipe FORWARD teleport
    return False, ""


def facing(ram) -> int:
    """PlayerFacingDir ($0033): 1 = right, 2 = left. The side-pipe entry needs facing right."""
    return int(ram[FACING_DIR])


def swimming(ram) -> bool:
    """True while underwater ($0704 nonzero). In water $001D is the swim default (1) only
    mid-stroke; it is 0 while coasting/sinking — the common state the side-pipe entry wants."""
    return int(ram[SWIMMING_FLAG]) != 0


def firing_state(ram) -> dict:
    """Diagnostic snapshot of every byte the side-pipe entry predicate depends on — used to
    capture/verify a real trigger frame (Codex/disasm: side metatile 0x6c/0x1f + $001D==0 +
    facing right + $000E in {7,8} -> sets $06DE and $000E=2)."""
    return {
        "x": mario_level_x(ram), "x_sub": int(ram[MARIO_X_ON_SCREEN]),
        "y": int(ram[MARIO_Y_ON_SCREEN]), "float_1d": int(ram[PLAYER_FLOAT_STATE]),
        "facing_33": int(ram[FACING_DIR]), "vx": signed(int(ram[MARIO_X_SPEED])),
        "vy": signed(int(ram[MARIO_Y_VELOCITY])), "engine_0e": int(ram[GAME_ENGINE_SUBROUTINE]),
        "chg_06de": int(ram[CHANGE_AREA_TIMER]), "swim_0704": int(ram[SWIMMING_FLAG]),
    }


def signed(byte: int) -> int:
    """Interpret a uint8 as a signed int8."""
    return byte - 256 if byte >= 128 else byte
