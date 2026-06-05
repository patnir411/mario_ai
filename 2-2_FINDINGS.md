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

## To actually beat 2-2 (options, not yet done)
1. **Reverse-curriculum** from a constructed post-pipe state in area 2 (drive to the flagpole),
   then learn/search the entry backward — same approach earmarked for 8-4.
2. A **sub-pixel-Y swim controller** that targets the landing frame (body in row 6, `$001D→0`,
   push right) — a dedicated frame-precise option, not blind chunk search.
3. Import a verified route for the entry frames and distill (the route-data benchmark path).
