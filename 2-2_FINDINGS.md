# 2-2 (underwater) — exit fully decoded; side-pipe entry is frame-precise (8-4-class)

**Status:** swim section solved to the exit; the final side-pipe entry is a frame-precise
maneuver that blind action-chunk search does not compose (same class as 8-4's lava pipe-mount).
Decode is complete and RAM/disassembly-verified (with the Codex consultant).

## What 2-2 is
- One underwater area (area key `($0760,$0750)=(2,37)`), **no flagpole in the water area**.
- It ends with a **horizontal side-pipe** whose mouth faces left; entering it transitions to
  a second above-ground area (brown staircase + flagpole). The official TMK map is 3840px
  wide; map-pixel-x is NOT in-game level-x for the late section (disassembly object stream is
  authoritative).

## Exit geometry (from the live RAM metatile buffer `$0500`, verified)
- The water object stream places a **WaterPipe** at page 11 / col 13 → **in-game level-x 3024**.
- Mouth metatiles at **col 3024**: row 5 = `0x6b` (upper lip), **row 6 = `0x6c` (the entry tile)**.
- A 3-row tunnel (rows 4–6, all `0x00`) at col 3008 leads to it; col 3024 rows 0–4 and 7+ are
  solid `0x69`. A block staircase climbs up-right to it: col2960 row10 → col2976 row9 →
  col2992 row8 → tunnel.

## Completion / transition signals (corrected — the load-bearing fix)
The old `area_search` keyed success on a `$0750` flip. **Codex correction (disassembly):**
- `$0750/$0751` are **pending-destination markers** (set by row-`$0e` objects), NOT entry proof.
- A **real pipe/area entry** is underway iff `CHANGE_AREA_TIMER ($06DE) != 0` OR
  `GameEngineSubroutine ($000E) ∈ {2,3}`.
- A level **actually advanced** iff the **stage byte `$075c`** increments — the right completion
  signal for flagpole-less water levels.

Implemented in `mario/ram.py` (`pipe_entering`, `stage_key`, corrected `AREA_POINTER` docs) and
`mario/search.py` (`area_search` now succeeds on `transitioned() = flag | pipe_entering |
stage-change | new-room`). Tests green.

## Why the entry resists search (the precise wall)
The engine's right-collision probe (Codex, from the disassembly) samples at
**`(Player_X + 13, Player_Y + 24)`**; the side-pipe entry fires only if that probe samples
`0x6c` AND Player_State `$001D == 0` (grounded; A released through the contact frame).

The geometry makes this a few-frame needle:
- To put the probe on `0x6c` (col 3024, row 6) Mario needs **Player_X ≥ 3011**.
- At **row 6 height while grounded**, his feet rest on the **row-7 floor**, which continues
  **solid into col 3024** — so walking right his feet hit `col3024 row7 (0x69)` and he is
  **blocked at x = 3010** (probe `X+13=3023` → empty col 3008, no trigger).
- One row **higher** he *can* reach X ≥ 3011, but then the probe samples **row 5 `0x6b`**
  (solid lip, no entry).
- So `0x6c` at row 6 is sandwiched: too high → `0x6b`; correct height → blocked by the floor.
  Entry needs a precise landing/swim frame where the body is purely in row 6, `$001D` reads 0,
  and X crosses 3011 in the same frame — a frame-precise maneuver.

### Search evidence
- `beam_search` (greedy-x) wedges at x3061 hugging the ceiling.
- `area_search` coverage and a purpose-built `scripts/solve_22_entry.py` (score = drive the
  exact probe `(X+13,Y+24)` onto `0x6c`) thoroughly explore (~360 cells/depth × 180 depths,
  CF=2) and reach the probe on **`0x6b`** (row 5, X≥3011) — never `0x6c`. Snapshot/restore
  fidelity was independently verified perfect, so this is a real geometric wall, not a bug.

## Reusable wins (independent of 2-2)
- The corrected transition semantics (`$06DE`/`$000E`/`$075c`, `$0750`=pending) are general and
  apply to 7-2 (the other water level) and any pipe/area transition.
- `scripts/solve_22_entry.py` demonstrates probe-targeted scoring against an exact engine
  predicate — a template for frame-precise option search.

## Reverse-curriculum attempt (DONE — rigorous negative)
The from-scratch reverse-curriculum solver (`scripts/solve_22_land.py`) was built and run, and the
exact entry predicate was pinned with Codex from the disassembly:

> Side-pipe entry fires when, in one `PlayerBGCollision` pass: `$000E==8`, `$001D==0`,
> `$0033==1`, `$0086 & 0x0f != 0`, and the right side-probe tile `(Player_X+13, row(Player_Y+24))`
> is `0x6c`. In water the SwimmingFlag `$0704=1` forces `$001D=1` at the START of every pass, so
> `$001D==0` only ever holds on a genuine **foot-landing** frame (`LandPlyr`, later in the pass).
> `0x6c` is **solid** to a descending Mario; `col3024` is a solid wall on rows 4–7.

Three independent results prove the trigger cell is **unreachable**, so the entry is sub-pixel
TAS / route-data only — exactly 8-4-class:
1. **Construct-via-poke fails:** poking `$001D=0` + `$0033=1` + a `0x6c` probe and stepping fires
   nothing — the SwimmingFlag overwrites `$001D` back to 1 before the side-pipe check (so the goal
   state cannot even be hand-constructed; it must arise from a real landing frame).
2. **Landing sweep fails:** 575 distinct reachable `x≥3008` near-mouth states × 243 sink/right/down
   patterns = **139,725 cf=1 rollouts → 0 entries**.
3. **Reachability proof:** a dedicated beam maximizing the trigger cell reached **0** states with
   `x≥3011 AND Y∈[104,119]`. Geometric reason: for the probe to sample `col3024 row6 (0x6c)` Mario
   needs `X∈[3011,3023]` AND `Y∈[104,119]`; but at that Y his body spans rows 6–7 and his right
   edge hits the solid `col3024` wall (rows 4–7) → ejected back to `x≤3010`. At `x≥3011` he can
   only be at row 5/above (probe → `0x6b`, no trigger). The only way into the trigger cell is the
   transition itself → chicken-and-egg → unreachable from outside.

**Conclusion:** 2-2's in-water side-pipe entry needs sub-pixel TAS or imported route data, the same
as 8-4. 7-2 is the same level type. The decode, the corrected transition detector, and
`solve_22_land.py` (reverse-curriculum harness) are the reusable artifacts.
