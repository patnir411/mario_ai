"""Sweep x-capped area_search across 8-4 area-1 to test EVERY pipe for enterability.

The gap in all prior 8-4 probes: `down` was only tried on the brick floor between pipes,
never while standing ON a pipe-top (where a down-pipe is actually entered). With area_search's
new `max_x` cap, the search explores LOCALLY (mounts pipe-tops, tries down/left/right) inside
a narrow x-window instead of running right to the page-16 loop. Sweep the window across the
corridor; if any window flips the area byte ($0760), we've found the escape pipe.

    ./venv/bin/python scripts/pipe_sweep_8_4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.env import MarioSim
from mario.ram import mario_level_x
from mario.search import area_search

APPROACH = ROOT / "runs" / "20260604-053533-cov_8_4" / "search_path.json"
# wider action set so the local search can also try left+A (mount a pipe from the left)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
WIDE = SIMPLE_MOVEMENT + [["down"], ["up"], ["left", "A"], ["left", "A", "B"]]


def prefix_to(path, target_x):
    sim = MarioSim(8, 4); sim.reset(0); pre = []
    for a in path:
        _i, done = sim.run_chunk(a, 8); pre.append(a)
        if mario_level_x(sim.ram) >= target_x or done:
            break
    x = mario_level_x(sim.ram); sim.close()
    return pre, x


def main() -> None:
    path = json.loads(APPROACH.read_text())["path"]
    windows = list(range(800, 3900, 280))   # sweep the whole corridor in ~280px windows
    # Forbid the known WRONG-route rooms so a "hit" means a genuinely new area (the real
    # escape pipe to water/Bowser), not the 229 dead-end or the 2 loop region.
    FORBID = {(3, 101), (3, 229), (3, 2)}
    print(f"sweeping {len(windows)} windows across 8-4 area-1 (wide actions, x-capped, $0750 signal, "
          f"forbidding {FORBID})")
    for w in windows:
        pre, px = prefix_to(path, w)
        changed, sub, info = area_search(
            8, 4, start_prefix=pre, actions=WIDE, beam_width=160, chunk_frames=8,
            max_x=px + 300, forbid_keys=FORBID, cov_bonus=60.0, stuck_cap=140,
            time_budget_s=130, progress_every=0)
        tag = "*** AREA CHANGED ***" if changed else "no change"
        print(f"  window x~{px:4d} (cap {px+300}): {tag}"
              + (f" new_area={info.get('area')} x={info.get('x_pos')}" if changed else ""), flush=True)
        if changed:
            full = pre + sub
            json.dump({"prefix": pre, "escape": sub, "full": full, "window_x": px,
                       "new_area": info.get("area")},
                      open(ROOT / "data" / "solutions" / ".area1_escape_8_4.json", "w"))
            print(f"  SAVED escape: prefix={len(pre)} + escape={len(sub)} chunks at x~{px}")
            return
    print("\nNO window flipped the area byte — no enterable pipe reachable in area-1 by local search.")


if __name__ == "__main__":
    main()
