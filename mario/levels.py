"""Per-level metadata shared across observation, reward, eval, and the game runner.

Single home for level-type knowledge (overworld/underground/underwater/castle) so the
pit sensor, death-cause classifier, and is_water scalar stay consistent.
"""
from __future__ import annotations

# Underwater levels: swimming physics, no gravity-fall → no pits, never "pit" deaths.
UNDERWATER = {(2, 2), (7, 2)}
# Castle levels (firebars / Bowser): x-4 in every world.
CASTLE = {(w, 4) for w in range(1, 9)}
# Underground (pipes, ceilings): the classic ones.
UNDERGROUND = {(1, 2), (4, 2)}


def level_type(world: int, stage: int) -> str:
    if (world, stage) in UNDERWATER:
        return "underwater"
    if (world, stage) in CASTLE:
        return "castle"
    if (world, stage) in UNDERGROUND:
        return "underground"
    return "overworld"


def is_underwater(world: int, stage: int) -> bool:
    return (world, stage) in UNDERWATER


# any% warp route (the ~8 levels that beat the game fastest). 1-2 and 4-2 take warps.
ANY_PERCENT = [(1, 1), (1, 2), (4, 1), (4, 2), (8, 1), (8, 2), (8, 3), (8, 4)]
# warpless: all 32 in order.
WARPLESS = [(w, s) for w in range(1, 9) for s in range(1, 5)]
