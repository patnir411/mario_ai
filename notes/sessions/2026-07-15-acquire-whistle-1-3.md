# 2026-07-15 — Tier-2 `AcquireWhistle_1_3` (white-block → Toad chest)

## Goal
Replace the Tier-1 hand-grant for the **first** warp whistle with a replay-verified
Tier-2 option from SMA4 1-3 (white-block duck → behind scenery → secret Toad house).

## Result
**DONE.** `data/solutions/sma4/acquire_whistle_1_3.json` (`solved=true`,
`replay_verified=true`, `n_frames=2533`, inventory `0x0C` at `0x03002C2E`).

ROM rebench without hand-grant:
- Open warpless: greedy 69000 frames / skip=False; search 12099 / skip=True → **5.7×**
  (`runs/20260715-sma4-whistle-rom-bench-acquire-open/report.json`)
- Warpless blocked: greedy stuck; search still skips
  (`runs/20260715-sma4-whistle-rom-bench-acquire/report.json`)

## Pipeline (verified)
1. Entry: `runs/sma4_cache/1-3_pwing_entry.pkl` (powerup=3, pspeed=127). Small Mario
   cannot crouch-activate the white block; mid-level powerup pokes are cleared next frame.
2. Fly to `x≥1680`, land with `LEFT×15` + settle 160 + brake.
3. Hold `DOWN` until behind-bg timer `0x03003D06` becomes nonzero (~239f).
4. Sprint `RIGHT+B` until `x` collapses from ≥2000 → <200 (Toad house).
5. Fade-in, walk `RIGHT×174`, hold `B` (~52f). **B opens chests in SMA4** (IGN); A only
   advances dialogue.
6. Exit `LEFT` through black fade → World-1 map with whistle still in inventory.

Visual proofs: `runs/20260715-acquire-whistle-e2e/{house,whistle,map_after_exit}.png`.

## Wiring
- `SMA4WhistleExecutor.acquire_whistle_1_3()` replays the solution.
- `build_sma4_whistle_rom_library(hand_granted=False)` exposes `acquire_whistle_1_3`
  (Tier-2) instead of `grant_two_whistles`.
- `scripts/bench_sma4_whistle_rom.py --no-hand-grant` (default max-tier 2).
- Non-ROM: `tests/test_acquire_whistle_solution.py` + meta-planner fake test.

## Honesty / remaining injections
| Fact | Status |
|---|---|
| First whistle from 1-3 chest (no RAM poke) | **Real** (Tier-2 option) |
| P-Wing 1-3 entry snapshot | Injected precondition (`pwing_1_3_entry_snapshot`) until map P-Wing use is an option |
| Second whistle in warp zone | Still Tier-1 re-grant (`whistle_regranted_in_warp_zone`) |
| MAP_CURSOR sync after AcquireWhistle exit | Technical assist — exit leaves cursor RAM off-grid so L-inventory breaks without sync to `(64,80)` / `(128,144)` |
| Warpless / Bowser costs | Still symbolic |

## Commands
```bash
MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/solve_sma4_acquire_whistle.py --replay-only
MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/bench_sma4_whistle_rom.py --no-hand-grant
MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/bench_sma4_whistle_rom.py --no-hand-grant --warpless-blocked
pytest tests/test_acquire_whistle_solution.py tests/test_meta_planner.py -q
```

## Next
1. Second W1 whistle (fortress flight) as `AcquireWhistle` to drop re-grant.
2. Fold map P-Wing use into the option initiation (remove entry-snapshot injection).
3. Optional: normal 1-3 card clear (B1).
