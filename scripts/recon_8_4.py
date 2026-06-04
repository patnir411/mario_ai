"""Read-only recon of 8-4 area-1: is there a LIFT, and where is an enterable DOWN-PIPE?

Resolves the one open factual question that decides the 8-4 solver shape:
  1. Replay a known approach path to the pipe region (snapshot each chunk).
  2. Decode every enemy/object slot seen across the area — distinguish LIFTS ($24-$2C)
     from Cheep-Cheeps / Piranha plants (the oscillating-y-at-fixed-x slots).
  3. For every grounded snapshot, run a SUB-TILE-CENTERED down-probe: nudge Mario's x to
     each sub-tile offset, hold DOWN, and check whether the AREA byte ($0760) flips
     (= an enterable pipe). This is the decisive test flat search never performs.

Pure read/restore — no edits, no solutions written.

    ./venv/bin/python scripts/recon_8_4.py [approach_path.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario import ram as R
from mario.env import MarioSim
from mario.ram import mario_level_x, signed
from mario.reward import is_death, is_success

# SMB enemy/object type IDs (Data Crystal). Lifts/platforms live in $24-$2C.
ENEMY_NAMES = {
    0x00: "GreenKoopa", 0x01: "RedKoopa", 0x02: "BuzzyBeetle", 0x05: "HammerBro",
    0x06: "Goomba", 0x07: "Blooper", 0x08: "BulletBill", 0x0A: "CheepGreen",
    0x0B: "CheepRed", 0x0C: "Podoboo", 0x0D: "PiranhaPlant", 0x0E: "CheepFlying",
    0x12: "Lakitu", 0x13: "Spiny",
    0x24: "LIFT_balance", 0x25: "LIFT_updown", 0x26: "LIFT_up", 0x27: "LIFT_down",
    0x28: "LIFT_leftright", 0x2A: "LIFT_falling", 0x2B: "LIFT_rope", 0x2C: "LIFT_platform",
}
LIFT_IDS = set(range(0x24, 0x2D))
APPROACH = ROOT / "runs" / "20260604-053533-cov_8_4" / "search_path.json"


def enemy_slots(ram):
    """Active enemy/object slots: (slot, type, x_level, y_screen, vx)."""
    out = []
    for i in range(5):
        if int(ram[R.ENEMY_ACTIVE.start + i]) == 0:
            continue
        t = int(ram[R.ENEMY_TYPE.start + i])
        ex = int(ram[R.ENEMY_X_LEVEL.start + i]) * 256 + int(ram[0x0087 + i])
        ey = int(ram[R.ENEMY_Y_ON_SCREEN.start + i])
        out.append((i, t, ex, ey))
    return out


def probe_pipe(sim, snap, *, down_frames=32, nudge=12):
    """From snap, try entering a pipe here: for each sub-tile offset, center + hold DOWN.
    Returns (entered, offset, frames, new_area) for the first offset that flips $0760."""
    base_area = R.area(sim.ram)
    # try a spread of horizontal nudges to center on a pipe lip, then sustained down
    for direction, act in ((0, None), (1, 1), (-1, 6)):   # in-place, right, left
        sim.restore(snap)
        dead = False
        if act is not None:
            for _ in range(nudge):
                info, done = sim.run_chunk(act, 1)
                if R.area(sim.ram) != base_area:
                    return True, direction * nudge, 0, R.area(sim.ram)
                if done:        # walked into a pit/lava — abandon this direction
                    dead = True
                    break
        if dead:
            continue
        # now hold DOWN
        for f in range(down_frames):
            info, done = sim.run_chunk(7, 1)
            if R.area(sim.ram) != base_area:
                return True, direction, f, R.area(sim.ram)
            if done:
                break
    return False, 0, 0, base_area


def main() -> None:
    approach_file = Path(sys.argv[1]) if len(sys.argv) > 1 else APPROACH
    path = json.loads(approach_file.read_text())["path"] if approach_file.exists() else None
    if path is None:
        print(f"no approach path at {approach_file}; running a quick area_search to get one")
        from mario.search import area_search
        _c, path, _i = area_search(8, 4, beam_width=64, time_budget_s=240, progress_every=40)

    sim = MarioSim(8, 4)
    sim.reset(0)
    start_area = R.area(sim.ram)
    print(f"start area={start_area}")

    lift_sightings = []          # (mario_x, slot, type, ex, ey)
    type_history = {}            # slot -> list of (ex, ey, type) to detect oscillation
    enter_hits = []              # (mario_x, offset, frames, new_area)
    snaps = []                   # (mario_x, grounded, snapshot)

    for i, a in enumerate(path):
        info, done = sim.run_chunk(a, 8)
        if done:
            break
        mx = mario_level_x(sim.ram)
        grounded = int(sim.ram[R.PLAYER_FLOAT_STATE]) == 0
        for slot, t, ex, ey in enemy_slots(sim.ram):
            type_history.setdefault(slot, []).append((ex, ey, t))
            if t in LIFT_IDS:
                lift_sightings.append((mx, slot, t, ex, ey))
        if grounded:
            snaps.append((mx, sim.snapshot()))

    print(f"\nreplayed {len(path)} chunks; reached x={mario_level_x(sim.ram)} "
          f"area={R.area(sim.ram)} grounded_snaps={len(snaps)}")

    # 1) lifts?
    all_types = sorted({t for h in type_history.values() for (_, _, t) in h})
    print("\n=== enemy/object types seen (id: name) ===")
    for t in all_types:
        print(f"  0x{t:02x}: {ENEMY_NAMES.get(t, '?')}")
    if lift_sightings:
        print(f"\n*** LIFT(S) DETECTED: {len(lift_sightings)} sightings ***")
        for mx, slot, t, ex, ey in lift_sightings[:12]:
            print(f"  mario_x={mx} slot{slot} type=0x{t:02x}({ENEMY_NAMES.get(t,'?')}) lift_x={ex} y={ey}")
    else:
        print("\nNO lift-type ($24-$2C) objects seen on this approach route.")

    # 2) enterable pipe? sub-tile-centered down-probe at each grounded snapshot
    print("\n=== centered down-probe (looking for $0760 flip = enterable pipe) ===")
    seen_x = set()
    for mx, snap in snaps:
        bucket = mx // 8
        if bucket in seen_x:
            continue
        seen_x.add(bucket)
        entered, off, frames, na = probe_pipe(sim, snap)
        if entered:
            enter_hits.append((mx, off, frames, na))
            print(f"  *** ENTERABLE @ mario_x={mx} offset={off} frames={frames} area {start_area}->{na} ***")

    print("\n=== RECON SUMMARY ===")
    print(f"lift_present: {bool(lift_sightings)}")
    print(f"enterable_pipe_x: {[h[0] for h in enter_hits] or 'NONE FOUND on reachable route'}")
    if enter_hits:
        best = enter_hits[0]
        print(f"-> Phase 1 = macro-only pipe-enter at x≈{best[0]} (offset {best[1]}, {best[2]} down-frames)")
    elif lift_sightings:
        print("-> Phase 1 = lift-gated: author waypoint to ride lift, then probe far-side pipe")
    else:
        print("-> NO lift + NO reachable enterable pipe on this route -> escalate (reverse-curriculum / off-route exploration)")
    sim.close()


if __name__ == "__main__":
    main()
