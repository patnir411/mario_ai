# 2026-06-26 — SMA4 physics-state search and the 1-2 wall

## Objective

Turn the 1-2 wall from a note-backed claim into artifact-backed telemetry, then test whether
physics/power-state-aware search with LEFT/DOWN/LEFT+RIGHT/P-speed affordances can break the
`x≈629` tall-pipe blocker before building a Raccoon-flight subsystem.

## Implemented

- `mario/adapters.py`
  - Added SMA4 physics fields to normalized info:
    `x_subpixel`, `y_subpixel`, `speed_signed`, `spin_jump`, `powerup`, and `powerup_set`.
  - Added richer action vocabularies:
    - default `SMA4_SURFACE_ACTIONS` remains the lean right-biased baseline,
    - `SMA4_PHYSICS_ACTIONS` adds LEFT/DOWN/pipe/backtrack/LEFT+RIGHT and A-pulse macros,
    - `SMA4_PSPEED_ACTIONS` is the narrower P-speed/pipe-entry experiment set.
  - Added frame-macro execution inside `SMA4Adapter.run_chunk` without changing the generic
    adapter search API.
- `mario/search.py`
  - `coverage_search_adapter` now uses a physics-aware cell by default:
    `(x_tile, y_tile, powerup, pspeed_bucket, speed_bucket, airborne_candidate,
    flight_bucket, x_subpixel_bucket)`.
  - `coverage_search_adapter` can emit JSONL traces.
- `mario/overworld_search.py`, `scripts/solve_sma4.py`, `scripts/solve_sma4_world.py`,
  `scripts/solve_sma4_snapshot.py`
  - Failed/partial attempts now emit JSON summaries, JSONL traces, and adapter contact sheets.
  - `WorldResult.beat_world` is no longer permanently false; it is set when the configured
    clear cap is reached or discovery exhausts targets without failures.
- `scripts/probe_sma4_physics.py`
  - Probe for known SMA4 movement/power RAM plus candidate movement-state bytes by IWRAM diffing.
- `DESIGN.md`
  - Updated stale toolchain text: Stable-Retro/mGBA is now the accepted SMA4 path.

## RAM probe

Artifact: `runs/20260626-195128-sma4_physics_probe/physics_probe.json`

Confirmed on the cached 1-2 entry snapshot:
- `0x03003C9F` P-speed / P-meter moves under running/jumping.
- `0x03003F40` speed moves under running and LEFT+RIGHT acceleration.
- `0x03003F5B` current power-up reads `0` (small) at 1-2 entry.
- `0x03003F74` queued/set power-up reads `0` at 1-2 entry.
- `0x03003F24`/`0x03003F28` fixed-point x/y remain the best progress/subpixel anchors.

Observed sequence samples:
- `RIGHT+B` for 180 frames: `x=329`, `speed=22`, `pspeed=0`.
- `LEFT+RIGHT+B` for 80 frames: `x=276`, `speed=68`, `pspeed=3`.

Airborne/flight/tail state is not yet confirmed. Diff candidates include movement-region bytes
around `0x03003F78`, `0x03003F82`, and `0x03003F87`, but these should stay candidates until a
Raccoon/flight-specific probe validates semantics.

## 1-2 ladder from `runs/sma4_cache/1-2_entry.pkl`

All runs used `scripts/solve_sma4_snapshot.py` and wrote trace/contact artifacts.

| Run | Action set | Solver result | x_max | Nodes | Wall time | Artifact |
|---|---|---:|---:|---:|---:|---|
| Surface baseline | 7-action right-biased | failed | 629 | 14,952 | 64.6s | `runs/20260626-195202-sma4_snapshot_1_2_surface/` |
| Full physics | LEFT/DOWN/LEFT+RIGHT/macros | failed | 630 | 48,231 | 182.5s | `runs/20260626-195917-sma4_snapshot_1_2_physics/` |
| P-speed focused | narrow LEFT+RIGHT/P-speed/pipe set | failed terminal, wall broken | 2803 | 105,756 | 468.4s | `runs/20260626-201118-sma4_snapshot_1_2_pspeed/` |

Conclusion: **the `x≈629` wall is not a Raccoon-flight requirement.** It falls to the focused
P-speed/LEFT+RIGHT action set. The run traverses the level to the end-region plateau (`x=2803`),
but does not produce a terminal clear. A replay with 1,800 NOOP post-frames stayed in-level at
`x=2800`, `y=128`, `endwalk=0`, so the remaining problem is the 1-2 end-card/finish route, not the
tall pipe.

## Important correction

`0x03002C52` is still **not** validated as a stable cleared-level bitmap. Earlier 1-1 clear
experiments saw `0 -> 3`, while the cached 1-2 entry reports `cleared=2`. Treat it as a progress
marker until multiple W1 clears are compared from the same map state.

## Verification

- `./venv/bin/python -m pytest -q tests/test_adapter_search.py tests/test_overworld_search.py`
  -> 13 passed.
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python -m pytest -q tests/test_sma4_overworld.py`
  -> 5 passed.
- `./venv/bin/python -m pytest -q`
  -> all non-ROM tests green, ROM-gated tests skipped when env vars are unset.

## Next

Do **not** build Raccoon flight for 1-2 yet. First add a more general SMA4 end-card finisher or
route-specific final-panel objective and convert the P-speed path into a replay-verified 1-2 clear.
After 1-2 is terminal-clean, use 1-3/fortress to probe Raccoon flight and warp-whistle options.
