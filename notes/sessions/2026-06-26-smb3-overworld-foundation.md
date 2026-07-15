# 2026-06-26 — SMB3 (SMA4) two-tier foundation: overworld instrumentation + generalized solver

## Objective

Begin the SMB3 end-to-end direction (see `mario-smb3-direction` memory and the approved
plan): build the foundation for a **two-tier emulator-backed search agent** — a high-level
planner over the SMB3 overworld that invokes the existing per-level beam search as an option.
This session delivers M0 (overworld instrumentation) and the M1 solver-generalization work.
Platform decision: stay on the SMA4 / Stable-Retro / mGBA runtime; reverse-engineer GBA RAM
empirically, cross-referenced against Data Crystal SMA4 + the NES SMB3 disassembly.

## Reverse-engineered SMA4 overworld RAM (confirmed)

Discovered by booting onto the World-1 map and diffing RAM under known inputs
(`scripts/probe_overworld.py`, reproducible — re-run with `--clear-1-1` for the progress
experiment). All in IWRAM:

- `MAP_CURSOR_X = 0x03003DE4` — map cursor X, on a 0x20-pixel grid (+0x20 per rightward node step).
- `MAP_CURSOR_Y = 0x03003DE0` — map cursor Y, on a 0x20 grid (−0x20 per upward step).
- `WORLD = 0x03002A69` — current world, **0-indexed** (0 = World 1).
- `PROGRESS = 0x03002C52` — per-world clear progress; flipped 0 → 3 after clearing 1-1
  (provisional bitmap; full structure to validate across more levels in M2).
- Mode detection is **behavioral** (no clean single mode byte separated): in a level the level
  timer (`TIME 0x03003D0B`) runs > 0; on the map the timer is 0 and the cursor sits on the
  0x20 grid. The post-level **results screen** ("LEVEL CLEARED / YOU GOT A PANEL") also reads
  timer 0 with a stale cursor, so it is a known false-positive for "overworld" — wait it out
  (~660 NOOP frames) until the cursor becomes input-responsive before treating the map as live.
- Cart SRAM (`0x0E000000`) is **not** written in real time (only at save points), so progress
  lives in IWRAM, not SRAM.
- Stable-Retro is **single emulator instance per process** — hence the shared-core design
  (one emulator, two adapter views) is mandatory for M2, not just convenient.

## Implemented

- `mario/adapters.py`:
  - `SMA4Adapter` gains overworld constants, a behavioral `_classify_mode`, `is_overworld`,
    `_map_walk`/`goto_node`/`enter_level` (closed-loop cursor nav), `boot_target="overworld"`,
    a `_boot_to_overworld` split out of `_boot_to_1_1`, `world`/`stage` constructor args, a
    `boot_to_level(world, stage)` (1-1 via the verified scripted boot; other nodes via map
    navigation through `NODE_TABLE`; gated levels deferred to M2 with a clear `NotImplementedError`),
    and `mode`/`cursor`/`cleared` fields in `_normalize_info`.
  - New `SMA4OverworldAdapter`: the overworld `GameAdapter` view over a shared `SMA4Adapter`
    core (`core=` to attach to an already-booted core; snapshots interchangeable between views).
    `is_success` = "entered a level"; `progress` = popcount of the cleared bitmap.
- `mario/search.py::goal_suffix_search(...)` — level-agnostic end-of-level finisher
  (snapshots states within `band` px of the run's `x_max`, frame-searches run→jump→run
  suffixes, accepts on the adapter's `is_success`). Replaces the hardcoded 1-1 card-window hack.
- `mario/render.py::replay_adapter` + `make_contact_sheet_adapter` (adapter-generic, reuse the
  SMB1 grid renderer via the new `_render_sheet`).
- `scripts/probe_overworld.py` (RE instrument), `scripts/replay_video_adapter.py` (adapter MP4).
- `scripts/solve_sma4.py` — now `world stage [...]` parameterized, uses `goal_suffix_search`,
  writes `data/solutions/sma4/{world}-{stage}.json`.
- Tests: `tests/test_sma4_overworld.py` (5, ROM-gated/slow — all green); `tests/test_adapter_search.py`
  gains non-ROM `goal_suffix_search` + `make_contact_sheet_adapter` coverage via a CappedGoalAdapter.

## Verification

- `MARIO_AI_SMA4_ROM=… pytest tests/test_sma4_overworld.py -m slow -q` → 5 passed (cursor moves,
  snapshot determinism, **overworld→level transition**, cleared bitmap readable).
- `pytest -q` (no ROM) → all green (ROM tests skip), incl. the new finisher/viz tests.
- `scripts/probe_overworld.py --clear-1-1` asserts all confirmed addresses + the 1-1 clear flip.

## Scope adjustment (honest)

M1's planned verification was "solve SMA4 1-2 from reset". On investigation, **1-2 is gated**
(must clear 1-1 first), and reaching it needs systematic overworld node discovery — which is
exactly the M2 meta-planner's job (BFS over `(node, cleared, inventory)`). A prototype BFS over
cursor moves was built but needs the post-results-screen "controllable map" handling and a more
robust enter test; the cleanest boundary is to land node discovery + gated-level solving in M2,
not hardcode a brittle 1-2 route in M1. So M1 is verified by **re-solving 1-1 with the
generalized pipeline** (`goal_suffix_search`, no level-specific constants) + adapter contact
sheet, which proves boot mechanism + finisher generalization + visualization without regression.

## M2 — two-tier meta-search: the loop is PROVEN (same session)

Built `mario/overworld_search.py`: `discover_nodes` (BFS over cursor moves + non-destructive
enterability test via the proven `enter_level`), `settle_to_map` (waits out the post-clear
results screen), and `overworld_solve` (the two-tier loop). `scripts/solve_sma4_world.py` drives
it; `tests/test_overworld_search.py` covers discovery/settle on fake cores (non-ROM, green).
`update_status.py` made game_id-keyed (additive, default `smb1`).

**Two bugs fixed during M2:**
1. Cached frame-precise solutions **desync** when a level is entered via the overworld (entry
   timing differs) → the planner solves each level FRESH from its entry snapshot and caches
   `(entry_snapshot, path)` (a restored snapshot replays exactly; a re-entered path does not).
2. `enter_level` returned during the ~140-frame **level-intro lockout** (inputs ignored),
   leaving the beam stuck at spawn (`x_max=24, path_len=0`). Fixed to settle until the player is
   controllable (cheap periodic RIGHT+B probe); beam then progresses normally.

**Run result (`runs/20260626-180705-sma4_world/`, ~920s):** the two-tier loop works end-to-end —
discovered 1-1 `(64,32)` → **solved it fresh from the overworld entry** (x_max=2803, path 1194,
beam+goal_suffix) → **the map advanced** (discovery then found `(96,32)`,`(128,32)` with 1-2
`(128,32)` newly enterable — gating confirmed) → attempted 1-2 → **stalled at x=629** (a geometry
obstacle the plain beam can't pass, the SMB3 analogue of the SMB1 levels that needed
`coverage_search`). So: meta-loop PROVEN; hard-level solving is the remaining gap.

## M2 follow-on — coverage port + the 1-2 wall (same session)

Built `mario/search.py::coverage_search_adapter` — a lean adapter-generic Go-Explore (cell
archive + one-time novelty bonus on new *coarse* `(x//16, y//16)` cells; degrades to plain beam
on open stretches). Wired into `overworld_solve` as an escalation ladder: beam → coverage →
`goal_suffix_search`. Non-ROM tests green (`tests/test_adapter_search.py`).

**1-2 is a genuine hard level (the first SMB3 per-level wall).** Diagnosed via screenshots from the
cached 1-2 entry snapshot (`runs/sma4_cache/1-2_entry.pkl`): 1-2 is a **pipe-terrain** level.
- Ground route: plain beam (cf6) clears the first pipe (x≈339) but stalls at a **tall T-capped
  pipe at x=629** that a ground jump can't clear. Coverage (cf6) stalls at the same wall (beam
  empties ~depth 110) — so it's not a search-breadth problem.
- Finer chunks (cf4) did **worse** — the search climbed *onto* the first pipe (x=342, y=7, standing
  on top) and got stuck, revealing an over-the-pipes route it couldn't traverse. So cf granularity
  is not the fix either.
- Conclusion: 1-2 needs dedicated route work — likely **flight / a powerup** (the ? blocks early in
  1-2; SMB3 is designed around the P-meter/raccoon, and the research flagged flight as the general
  SMB3 unlock) or a precise P-speed pipe-top traversal. This is the SMB3 analogue of the SMB1
  castle grind: per-level, not a quick search-param fix.

## Status + Next (strategic fork)

DONE this session: **M0 + M1 + M2 (two-tier loop architecture, incl. coverage escalation)** — all
verified. The meta-loop provably solves a level, advances the map, and discovers the next gated
level (1-1 → 1-2). World-1 *completion* is now gated on per-level hard-level solving, of which 1-2
is the first instance.

Next-step options (a genuine fork — get user steer):
1. **M5 flight-aware search first** (likely the general SMB3 unlock): add P-meter-build + flight
   macros; flight plausibly trivializes 1-2's tall pipe and many W1 levels at once. Highest-leverage.
2. **Grind W1 ground levels per-level** (1-2 pipe-top route / powerup-from-?-block / P-speed jump),
   like the SMB1 castle saga. Slow, level-by-level.
3. Scale the meta-search across worlds with whatever levels currently solve, then revisit hard ones.
- Then: full World-1 video from cached `(entry_snap, path)`; M3 (8 worlds); M4 (warp-whistle skip).
