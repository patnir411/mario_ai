"""Verify Codex's 8-4 route model BEFORE building a solver. Read-only.

Codex's claims to test:
  (1) $0750 (AreaPointer) flips along the surface walk are DESTINATION MARKERS, not real
      transitions — so ChangeAreaTimer $06DE stays 0 and Player_State stays normal at those
      flips. Confirm the ($0750,$0751) marker sequence matches: 0x65 -> 0xe5 -> 0x65 -> 0x02.
  (2) The real escape is a PIPE entry, detectable as $06DE going NONZERO (or Player_State
      entering pipe mode) followed by a reload. Test: at every grounded spot on the path,
      hold DOWN and watch $06DE / $000E / $0760 / ($0750,$0751) — does a real entry fire?

    ./venv/bin/python scripts/verify_route_8_4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mario.ram as R
from mario.env import MarioSim
from mario.ram import mario_level_x

APPROACH = ROOT / "runs" / "20260604-053533-cov_8_4" / "search_path.json"
AREA, APTR, EPAGE, CAT, PSTATE, FLOAT = 0x0760, 0x0750, 0x0751, 0x06DE, 0x000E, 0x001D


def snap_state(ram):
    return dict(x=mario_level_x(ram), area=int(ram[AREA]), ptr=int(ram[APTR]),
                epage=int(ram[EPAGE]), cat=int(ram[CAT]), pstate=int(ram[PSTATE]),
                fl=int(ram[FLOAT]))


def main() -> None:
    path = json.loads(APPROACH.read_text())["path"]
    sim = MarioSim(8, 4); sim.reset(0)
    s0 = snap_state(sim.ram)
    print(f"start: {s0}")

    # (1) marker timeline + did $06DE ever fire while walking?
    print("\n=== (1) ($0750,$0751) marker sequence along the surface walk ===")
    last = (s0["ptr"], s0["epage"])
    cat_fired = False
    grounded_snaps = []
    for i, a in enumerate(path):
        info, done = sim.run_chunk(a, 8)
        if done:
            break
        st = snap_state(sim.ram)
        if (st["ptr"], st["epage"]) != last:
            print(f"  chunk{i} x={st['x']}: ($0750,$0751) {last} -> ({st['ptr']},{st['epage']})"
                  f"  [$06DE={st['cat']} $000E={st['pstate']}]")
            last = (st["ptr"], st["epage"])
        if st["cat"] != 0:
            cat_fired = True
        if st["fl"] == 0:
            grounded_snaps.append((st["x"], sim.snapshot()))
    print(f"  ChangeAreaTimer($06DE) ever nonzero while walking? {cat_fired}  "
          f"(Codex predicts: False — these are markers)")

    # (2) does holding DOWN at any grounded spot trigger a REAL entry ($06DE / state / reload)?
    print(f"\n=== (2) sustained-down probe at {len(grounded_snaps)} grounded spots "
          f"(watching $06DE / $000E / $0760 / ($0750,$0751)) ===")
    hits = []
    seen = set()
    for x, snap in grounded_snaps:
        b = x // 8
        if b in seen:
            continue
        seen.add(b)
        sim.restore(snap)
        base = snap_state(sim.ram)
        for f in range(24):
            info, done = sim.run_chunk(7, 1)
            st = snap_state(sim.ram)
            if st["cat"] != 0 or st["pstate"] != base["pstate"] or st["area"] != base["area"]:
                hits.append((x, f, base, st))
                print(f"  *** ENTRY SIGNAL @ x={x} after {f}f: $06DE {base['cat']}->{st['cat']} "
                      f"$000E {base['pstate']}->{st['pstate']} $0760 {base['area']}->{st['area']} "
                      f"($0750,$0751) ({base['ptr']},{base['epage']})->({st['ptr']},{st['epage']}) ***")
                break
            if done:
                break

    print("\n=== VERDICT ===")
    print(f"marker flips seen on walk: yes; $06DE during walk: {cat_fired}")
    print(f"down-triggered real-entry signals: {len(hits)}"
          + (f" at x={[h[0] for h in hits]}" if hits else " — NONE on the surface path"))
    if not hits:
        print("  -> on the reachable surface, DOWN never triggers a real pipe entry; the "
              "enterable pipe is off this path (must be MOUNTED) or needs a different trigger.")
    sim.close()


if __name__ == "__main__":
    main()
