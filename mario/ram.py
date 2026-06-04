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
PLAYER_FLOAT_STATE = 0x001D  # 0 ground, 1 jumping, 2 ledge, 3 flagpole
PLAYER_STATE = 0x000E       # climbing/pipe/dying/transforming
POWERUP_STATE = 0x0756      # 0 small, 1 big, >=2 fiery
PLAYER_VIABLE = 0x000E      # alias for player state (death animation == 0x06/0x0B)

# --- enemies (5 slots) --------------------------------------------------
ENEMY_X_LEVEL = range(0x006E, 0x0073)   # horizontal position in level
ENEMY_Y_ON_SCREEN = range(0x00CF, 0x00D4)
ENEMY_TYPE = range(0x0016, 0x001B)
ENEMY_ACTIVE = range(0x000F, 0x0014)

# --- level layout -------------------------------------------------------
TILE_DATA = range(0x0500, 0x06A0)  # current on-screen tile grid in RAM
SCREEN_IN_LEVEL = 0x071A

# --- area / transition signals ------------------------------------------
# AREA_NUMBER is the RAW gym-side area byte; entering a pipe/area transition flips it.
# It is the primary "did I escape this room" success signal for maze castles (8-4).
# (Note: gym's info `_area` reports ram[AREA_NUMBER] + 1, so raw 3 == logical area 4.)
AREA_NUMBER = 0x0760
AREA_POINTER = 0x0750        # which area-object set is loaded
CHANGE_AREA_TIMER = 0x06DE   # nonzero while an area/pipe transition is in progress
Y_HIGH_POS = 0x00B5          # vertical "page" of Mario's Y (stays 1 in 8-4 area-1)


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
    """Combined area identity = (raw area, area pointer) — distinct per sub-area/room."""
    return (int(ram[AREA_NUMBER]), int(ram[AREA_POINTER]))


def signed(byte: int) -> int:
    """Interpret a uint8 as a signed int8."""
    return byte - 256 if byte >= 128 else byte
