# 2-2 (underwater) — SOLVED from scratch (was a detection bug, not a hard level)

**Status: SOLVED.** `data/solutions/2-2.json` solved=True; `render.replay` beat=True, flag at
x_pos=3161; contact sheet shows underwater swim → side-pipe entry → surface area-2 → flagpole.
38/38 tests green. The earlier "frame-precise / TAS-only" verdicts (below) were ALL WRONG — the
root cause was a **detection bug**, not level difficulty.

## The resolution (Codex consult, gpt-5.5, decisive)
The side-pipe entry is reachable by **simply holding RIGHT** from the `data/solutions/2-2.json`
prefix `path[:220]` (x≈2916, Y≈48): Mario sinks while drifting right and at **x=3011, Y=112 on a
foot-landing frame** (`$001D=0`) the right side-probe `(X+13)=3024` first reaches the mouth `$6c`
→ `$000E=2`, `$06DE=43` (SideExitPipeEntry). At x=3010 the probe is 3023 (col3008) — 1px short,
which is the "wall." Full solve = `path[:220]` (cf8) + **81×right** (cf1) + a 17-action area-2
beam to the flagpole, normalized to cf1.

### Why every prior attempt "failed" — the gym-wrapper detection bug (GENERAL, applies to 7-2/8-4)
`gym_super_mario_bros` runs `_skip_change_area()` / `_skip_occupied_states()` INSIDE `env.step()`
AFTER the frame (smb_env.py ~:397). So a pipe/area transition is **fast-forwarded before you read
live RAM**: by the time `run_chunk` returns, `$06DE`/`$000E` are back to 0 and the player has
already jumped to the next area's coordinates. Our `pipe_entering` detector (checking RAM after the
wrapped step) therefore **never saw the transient** — the searches DID enter the pipe many times
(the "ejections to x≈2872" we dismissed as deaths were the post-transition area-2 spawn), we just
threw the successes away. **Detect pipe transitions by the post-step signature** (large backward
x-jump / `info` discontinuity) or by stepping the raw native frame, NOT by reading `$06DE`/`$000E`
after the wrapped step. The flagpole at the end of area-2 sets `flag_get` normally, so the *final*
success was always detectable — the bug only hid the intermediate pipe entry.

### Exact mechanics (disassembly-verified, for the record)
- Trigger (`ChkPBtm`): `$001D==0` (grounded) AND facing right (`$0033==1`) AND side-collided
  metatile ∈ {`$6c`,`$1f`}. Right side-probe offset = `(Player_X+13, Player_Y+24)` for small Mario.
- In water `$001D` is forced to 1 each frame by SwimmingFlag `$0704`; it reads 0 only on a real
  foot-landing frame (LandPlyr), which runs before the side check — so a descending-right approach
  that lands at the mouth row satisfies grounded + probe-on-`$6c` on the same frame.
- Only one enterable tile in 2-2 (`$6c` at level-x 3024 row6); no `$1f` variant.

---
## (Superseded) prior incorrect analysis kept for the record
The sections below concluded "frame-precise / TAS-only." They were wrong (detection bug above).

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
