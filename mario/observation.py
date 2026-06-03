"""Compact, ego-centric observation for the policy (V2) and dataset (V1).

RAM/tile features, not pixels (DESIGN.md §4): an ego-centric grid of tile codes around
Mario plus a few scalar features. Position-relative so it generalizes across levels —
absolute level-x is NEVER a feature.

Tile extraction uses the well-known SMB scheme: the 0x0500 region is a 2-screen ring
buffer of 13 rows x 16 cols of tiles; page = (level_x // 256) % 2. Correctness is
verified visually with scripts/verify_observation.py (overlay vs the rendered frame).
"""
from __future__ import annotations

import numpy as np

from mario import ram as R

# tile codes in the ego grid
EMPTY = 0
SOLID = 1
ENEMY = 2
MARIO = 3

TILE_BASE = 0x0500
PAGE_BYTES = 208          # 13 rows * 16 cols
N_ROWS = 13
HUD_Y_OFFSET = 32         # top 2 tile rows are HUD/sky above the 13-row playfield
# Mario's stored y (0x00CE) sits ~1 tile above his feet; this offset puts his ego-grid
# cell directly ABOVE the ground tile when standing (calibrated vs verify_observation).
MARIO_ROW_ADJUST = 1

# ego window (tiles): columns left/right of Mario, rows up/down of Mario
LEFT, RIGHT = 4, 11       # see more ahead (right) than behind — pits/enemies incoming
UP, DOWN = 6, 6
GRID_W = LEFT + RIGHT + 1
GRID_H = UP + DOWN + 1


def mario_level_x(ram) -> int:
    return R.mario_level_x(ram)


def mario_screen_y(ram) -> int:
    """Mario y on screen in pixels (for tile-row lookup)."""
    return int(ram[R.MARIO_Y_ON_SCREEN])


def get_tile(ram, level_x: int, screen_y: int) -> int:
    """Solidity of the tile at level pixel x, screen pixel y. 1 solid, 0 empty/sky."""
    row = (screen_y - HUD_Y_OFFSET) // 16
    if row < 0 or row >= N_ROWS:
        return EMPTY
    page = (level_x // 256) % 2
    col = (level_x % 256) // 16
    val = int(ram[TILE_BASE + page * PAGE_BYTES + row * 16 + col])
    return SOLID if val != 0 else EMPTY


def _enemy_cells(ram, mario_x: int, mario_col: int, mario_row: int):
    """Yield (gy, gx) ego-grid cells occupied by active enemies."""
    for i in range(5):
        if int(ram[R.ENEMY_ACTIVE.start + i]) == 0:
            continue
        ex = int(ram[R.ENEMY_X_LEVEL.start + i]) * 256 + int(ram[0x0087 + i])
        ey = int(ram[R.ENEMY_Y_ON_SCREEN.start + i])
        ecol = ex // 16
        erow = (ey - HUD_Y_OFFSET) // 16 + MARIO_ROW_ADJUST
        gx = (ecol - mario_col) + LEFT
        gy = (erow - mario_row) + UP
        if 0 <= gy < GRID_H and 0 <= gx < GRID_W:
            yield gy, gx


def tile_grid(ram) -> np.ndarray:
    """Ego-centric GRID_H x GRID_W int8 grid of {EMPTY,SOLID,ENEMY,MARIO}."""
    mario_x = mario_level_x(ram)
    mario_y = mario_screen_y(ram)
    mario_col = mario_x // 16
    mario_row = (mario_y - HUD_Y_OFFSET) // 16 + MARIO_ROW_ADJUST

    grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            col = mario_col + (gx - LEFT)
            row = mario_row + (gy - UP)
            lx = col * 16
            sy = row * 16 + HUD_Y_OFFSET
            grid[gy, gx] = get_tile(ram, lx, sy)

    for gy, gx in _enemy_cells(ram, mario_x, mario_col, mario_row):
        grid[gy, gx] = ENEMY
    grid[UP, LEFT] = MARIO  # ego center is always Mario
    return grid


def scalar_features(ram, info: dict) -> np.ndarray:
    """Small vector of velocity / state scalars (relative / normalized, no absolute x)."""
    vx = R.signed(int(ram[R.MARIO_X_SPEED]))
    vy = R.signed(int(ram[R.MARIO_Y_VELOCITY]))
    powerup = int(ram[R.POWERUP_STATE])
    float_state = int(ram[R.PLAYER_FLOAT_STATE])
    on_ground = 1.0 if float_state == 0 else 0.0
    return np.array([
        vx / 40.0, vy / 40.0,
        min(powerup, 2) / 2.0,
        on_ground,
        1.0 if float_state == 1 else 0.0,  # jumping
    ], dtype=np.float32)


def observe(ram, info: dict) -> np.ndarray:
    """Flat observation vector = one-hot tile grid + scalars. Used by the policy."""
    grid = tile_grid(ram)
    onehot = np.zeros((4, GRID_H, GRID_W), dtype=np.float32)
    for code in (EMPTY, SOLID, ENEMY, MARIO):
        onehot[code] = (grid == code)
    return np.concatenate([onehot.reshape(-1), scalar_features(ram, info)])


OBS_DIM = 4 * GRID_H * GRID_W + 5
