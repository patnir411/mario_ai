"""Go-Explore over the resettable emulator — the fix for deceptive maze-loop levels.

Why this beats progress-search: levels like 8-4/4-4/7-4 have a RAM loop-checkpoint that
sends you back 4 pages unless you cross specific pages at a specific HEIGHT (Y). An
x-greedy beam can't value "be at Y=240 when crossing page $10", so it arrives at the
wrong Y and loops forever (the beam empties). Go-Explore instead keeps an ARCHIVE of
cells keyed on (world, area, x-tile, y-TILE), RETURNS to a promising cell by restoring its
snapshot (free for us), then EXPLORES with random actions, scoring COVERAGE not x. The
y-tile makes "page boundary at the correct height" a distinct cell that gets explored, so
the correct (non-looping) route is discovered without any hand-authored pipe coordinates.

This is the canonical Go-Explore "phase 1" (Ecoffet et al., Nature 2021), and a resettable
deterministic emulator is exactly the substrate it was designed for.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from mario.env import MarioSim, N_ACTIONS
from mario.ram import mario_level_x
from mario.reward import is_death, is_success

AREA_ADDR = 0x0760   # area within stage (gym convention)
Y_ADDR = 0x00CE      # player Y on screen (the dimension x-greedy search ignored)

# action sampling weights (index = action id): favor forward/jump, allow down (pipes/
# descent), up (vines), A (swim strokes / jump-in-place), rare left.
#            NOOP right r+A  r+B r+A+B  A  left down  up
WEIGHTS = [0.4, 0.8, 1.5, 1.5, 1.5, 0.9, 0.3, 0.8, 0.5]


@dataclass
class Cell:
    snap: object
    path: list
    x: int           # best (highest) level-x reached for this cell
    visits: int = 0


def cell_key(sim, info, tile: int) -> tuple:
    ram = sim.ram
    return (int(info.get("world", 0)), int(ram[AREA_ADDR]),
            mario_level_x(ram) // tile, int(ram[Y_ADDR]) // tile)


def _select(archive: dict, rng: random.Random) -> tuple:
    """Pick a frontier cell: usually low-visit (explore), sometimes the furthest (exploit)."""
    keys = list(archive.keys())
    if rng.random() < 0.25:                       # exploit: push the frontier forward
        return max(keys, key=lambda k: archive[k].x)
    weights = [1.0 / (1.0 + archive[k].visits) ** 0.5 for k in keys]   # explore: novelty
    return rng.choices(keys, weights=weights, k=1)[0]


def go_explore(world: int, stage: int, *, chunk_frames: int = 4, explore_chunks: int = 24,
               max_iters: int = 60000, time_budget_s: float = 900.0, tile: int = 16,
               seed: int = 0, rng_seed: int = 0, progress_every: int = 3000):
    """Returns (solved, path, stats). `path` is the action-chunk sequence (chunk_frames each)
    that reaches the flag, or the furthest-progress path found within budget."""
    sim = MarioSim(world, stage)
    info = sim.reset(seed=seed)
    rng = random.Random(rng_seed)
    actions = list(range(N_ACTIONS))

    start = cell_key(sim, info, tile)
    archive = {start: Cell(sim.snapshot(), [], mario_level_x(sim.ram))}
    best_path, best_x = [], mario_level_x(sim.ram)
    t0 = time.perf_counter()
    iters = 0

    for iters in range(1, max_iters + 1):
        if time.perf_counter() - t0 > time_budget_s:
            break
        key = _select(archive, rng)
        cell = archive[key]
        cell.visits += 1
        sim.restore(cell.snap)
        path = list(cell.path)

        a, sticky = 0, 0
        for _ in range(explore_chunks):
            if sticky <= 0:                       # sticky actions = coherent moves
                a = rng.choices(actions, weights=WEIGHTS, k=1)[0]
                sticky = rng.randint(1, 4)
            sticky -= 1
            info, done = sim.run_chunk(a, chunk_frames)
            path.append(a)
            if is_success(info):
                sim.close()
                return True, path, {"iters": iters, "cells": len(archive),
                                    "wall_s": time.perf_counter() - t0}
            if is_death(info, done):
                break
            lx = mario_level_x(sim.ram)
            k = cell_key(sim, info, tile)
            ex = archive.get(k)
            if ex is None or lx > ex.x:           # new cell, or a better (further) way to it
                archive[k] = Cell(sim.snapshot(), list(path), lx)
            if lx > best_x:
                best_x, best_path = lx, list(path)

        if progress_every and iters % progress_every == 0:
            print(f"  iter {iters:6d} | cells {len(archive):5d} | best_x {best_x:5d} "
                  f"| {iters/(time.perf_counter()-t0):5.0f} it/s", flush=True)

    sim.close()
    return False, best_path, {"iters": iters, "cells": len(archive),
                              "wall_s": time.perf_counter() - t0}
