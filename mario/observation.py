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

# tile codes in the ego grid (MARIO marks the center cell of the grid)
EMPTY = 0
SOLID = 1
ENEMY = 2
MARIO = 3
HAZARD = 4   # projectiles/firebars/hammers/Bowser-fire — its own one-hot channel
N_CHANNELS = 5

from mario.levels import UNDERWATER  # noqa: E402  (level-type knowledge lives in one place)

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


def gap_ahead(grid: np.ndarray) -> float:
    """Distance (normalized) to the nearest pit ahead at Mario's foot level.

    The tile grid is 16px-quantized so the policy can't time a jump from the grid alone;
    this scalar (with x_subtile below) makes 'a pit is N tiles ahead' explicit — the key
    signal for clearing the first 1-1 gap. Returns 1.0 if no gap within the window.
    """
    foot = UP + 1  # ground sits directly below Mario's cell (MARIO_ROW_ADJUST)
    if foot >= GRID_H:
        return 1.0
    for gx in range(LEFT, GRID_W):
        if grid[foot, gx] != SOLID:        # missing ground support ahead = pit/edge
            return (gx - LEFT) / max(RIGHT, 1)
    return 1.0


def is_water_level(info: dict) -> bool:
    return (int(info.get("world", 0)), int(info.get("stage", 0))) in UNDERWATER


def scalar_features(ram, info: dict, grid: np.ndarray | None = None) -> np.ndarray:
    """Velocity / state scalars + sub-tile position + pit sensor + water flag.

    sub-tile x and gap_ahead are what let the policy TIME a jump — within a 16px tile the
    ego grid is identical, so without these 'jump now' vs 'wait' are indistinguishable.
    is_water lets one net separate swimming from running; underwater the pit sensor is
    meaningless (no floor to fall off) so it's gated to 1.0.
    """
    if grid is None:
        grid = tile_grid(ram)
    vx = R.signed(int(ram[R.MARIO_X_SPEED]))
    vy = R.signed(int(ram[R.MARIO_Y_VELOCITY]))
    powerup = int(ram[R.POWERUP_STATE])
    float_state = int(ram[R.PLAYER_FLOAT_STATE])
    on_ground = 1.0 if float_state == 0 else 0.0
    x_subtile = (mario_level_x(ram) % 16) / 16.0
    water = is_water_level(info)
    return np.array([
        vx / 40.0, vy / 40.0,
        min(powerup, 2) / 2.0,
        on_ground,
        1.0 if float_state == 1 else 0.0,  # jumping
        x_subtile,                          # sub-tile horizontal phase (jump timing)
        1.0 if water else gap_ahead(grid),  # pit distance (gated off underwater)
        1.0 if water else 0.0,              # is_water
    ], dtype=np.float32)


N_SCALARS = 8


def _hazard_cells(ram, mario_col: int, mario_row: int):
    """Ego-grid cells occupied by projectiles/firebars/hammers/Bowser-fire.

    Placeholder: 1-1..1-3 have no such hazards (Goombas/Koopas are ENEMY, not HAZARD), so
    this returns nothing for them. Populated + visually verified on 1-4 (first firebars)
    before castle levels rely on it. The 5th channel exists now so OBS_DIM is locked and
    World-1 datasets don't need re-generation when hazard reading is calibrated.
    """
    return ()  # TODO(1-4): read firebar/hammer/fireball RAM, classify into the grid


def observe(ram, info: dict) -> np.ndarray:
    """Flat observation vector = 5-channel one-hot tile grid + scalars. Used by the policy."""
    grid = tile_grid(ram)
    onehot = np.zeros((N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    for code in (EMPTY, SOLID, ENEMY, MARIO):
        onehot[code] = (grid == code)
    mario_col = mario_level_x(ram) // 16
    mario_row = (mario_screen_y(ram) - HUD_Y_OFFSET) // 16 + MARIO_ROW_ADJUST
    for gy, gx in _hazard_cells(ram, mario_col, mario_row):
        if 0 <= gy < GRID_H and 0 <= gx < GRID_W:
            onehot[HAZARD, gy, gx] = 1.0
    return np.concatenate([onehot.reshape(-1), scalar_features(ram, info, grid)])


OBS_DIM = N_CHANNELS * GRID_H * GRID_W + N_SCALARS
