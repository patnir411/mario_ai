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


def mario_level_x(ram) -> int:
    """Absolute x position in the level = page*256 + x_on_screen."""
    return int(ram[MARIO_LEVEL_PAGE]) * 256 + int(ram[MARIO_X_ON_SCREEN])


def signed(byte: int) -> int:
    """Interpret a uint8 as a signed int8."""
    return byte - 256 if byte >= 128 else byte
