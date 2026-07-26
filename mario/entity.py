"""Object-centric ENTITY observation — the shared schema for the generalist policy.

A structured token set {player, enemies, terrain columns} instead of a flat tile grid.
Promoted from the V4 experiment (`scripts/exp_entity.py`) so the dataset builder, the
policy, the trainer, and the eval/DAgger loops all import ONE definition.

Why this sidesteps V4's "empty HAZARD channel" culprit: that bug was in the flat tile obs
(`observation.py::_hazard_cells` returns ()). Here, hazardous objects (firebars, cheep-
cheeps, etc.) are real NES enemies that already populate the enemy slots ($000F/$0016), so
ETYPE encodes them — no separate hazard channel to forget to fill.

Related object/entity-centric work includes OCCAM (arXiv:2504.03024) and
Entity-Centric RL (ICLR 2024).  Those papers motivate the inductive bias in
different domains; they do not establish cross-level Mario generalization.
"""
from __future__ import annotations

import numpy as np

from mario import ram as R
from mario.ram import mario_level_x, signed
from mario.observation import GRID_H, GRID_W, LEFT, UP, gap_ahead, tile_grid

# ---- entity schema ----------------------------------------------------------
N_ENT, N_TERR = 5, 8
N_TOK = 1 + N_ENT + N_TERR          # player + 5 enemies + 8 terrain columns = 14
D_TOK = 17
OBS_DIM_ENTITY = N_TOK * D_TOK      # 238
# token feature layout
P_IS, E_IS, T_IS = 0, 1, 2          # token-type one-hot
DX, DY, VX, VY = 3, 4, 5, 6
ETYPE, GROUND = 7, 8
ON_GND, JUMP, ACTIVE, PIT = 9, 10, 11, 12
POW, FACE, GAP, XSUB = 13, 14, 15, 16
ENT_X_PAGE = 0x006E   # ENEMY_X_LEVEL.start
ENT_X_SCR = 0x0087    # enemy x within screen


def entity_obs(ram, info=None) -> np.ndarray:
    """Build the (N_TOK*D_TOK,) entity-token vector from live NES RAM."""
    toks = np.zeros((N_TOK, D_TOK), np.float32)
    mx = mario_level_x(ram); my = int(ram[R.MARIO_Y_ON_SCREEN])
    grid = tile_grid(ram)
    p = toks[0]
    p[P_IS] = 1.0
    p[VX] = np.clip(signed(int(ram[R.MARIO_X_SPEED])) / 4.0, -4, 4)
    p[VY] = np.clip(signed(int(ram[R.MARIO_Y_VELOCITY])) / 4.0, -4, 4)
    p[ON_GND] = 1.0 if int(ram[R.PLAYER_FLOAT_STATE]) == 0 else 0.0
    p[JUMP] = 1.0 if int(ram[R.PLAYER_FLOAT_STATE]) == 1 else 0.0
    p[POW] = min(int(ram[R.POWERUP_STATE]), 2) / 2.0
    p[FACE] = 1.0 if int(ram[R.FACING_DIR]) == 1 else 0.0
    p[GAP] = float(gap_ahead(grid))
    p[XSUB] = (int(ram[R.MARIO_X_ON_SCREEN]) % 16) / 16.0
    for i in range(N_ENT):
        if int(ram[0x000F + i]) == 0:      # ENEMY_ACTIVE
            continue
        ex = int(ram[ENT_X_PAGE + i]) * 256 + int(ram[ENT_X_SCR + i])
        ey = int(ram[0x00CF + i])          # ENEMY_Y_ON_SCREEN
        t = toks[1 + i]
        t[E_IS] = 1.0; t[ACTIVE] = 1.0
        t[DX] = np.clip((ex - mx) / 48.0, -4, 4)
        t[DY] = np.clip((ey - my) / 48.0, -4, 4)
        t[ETYPE] = int(ram[0x0016 + i]) / 48.0   # ENEMY_TYPE (firebar/cheep/etc. encoded here)
    for j in range(N_TERR):
        col = LEFT + 1 + j
        t = toks[1 + N_ENT + j]
        t[T_IS] = 1.0
        t[DX] = (j + 1) / N_TERR
        if col < GRID_W:
            below = np.where(grid[UP:, col] == 1)[0]   # solid rows at/below Mario
            if len(below):
                t[GROUND] = below[0] / (GRID_H - UP)
            else:
                t[PIT] = 1.0; t[GROUND] = 1.0
    return toks.reshape(-1)
