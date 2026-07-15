# 2026-07-15 — Fortress second whistle: path blocker + regrant removed

## Goal
Build a second W1 `AcquireWhistle` (fortress flight / hidden door), replay-verify
it, and remove `whistle_regranted_in_warp_zone` from the ROM whistle library.

## Verdict
- **Regrant removed** (honesty win). Two inventory whistles → both spends are
  real emulator transitions; no warp-zone RAM poke.
- **Fortress AcquireWhistle blocked** this pass. After clearing 1-1 and 1-2
  (cached *and* natural entry→replay), the walkable overworld graph never
  includes the fortress node. Wiki says fortress opens after 1-2; SMA4 map
  edges disagree under our clear path.

## Fortress recon (artifact-backed)

| Finding | Evidence |
|---|---|
| `(160,64)` is **not** fortress | Prior: sky/athletic; confirmed again |
| Fortress icon ~`(96,80)` (Y off 0x20 grid) | `runs/20260715-fortress-recon5/stand_96_80_at_96_80.png` |
| Cursor poke is cosmetic | `0x03003788` tile id does **not** update on poke; enter → `menu` |
| Natural walk 1-2 column | `(128,32)→(128,64)→(128,96)` spade only |
| `LEFT` from `(128,64)` never moves | hold 8–90f; hammer item poke 0x01–0x1F no break |
| `DOWN` from `(96,32)` never moves | post-1-2 |
| `03002C85=02` (all levels) | Does **not** open fortress walk edge |
| Natural 1-1→enter 1-2→replay clear | Same reachable set as double-cache FF; fortress still missing |
| Misread `PROGRESS` | Data Crystal: `0x03002C52` = panel slots, not clear bitmap |

Scripts: `scripts/recon_sma4_fortress.py`, `scripts/recon_sma4_fortress_cells.py`.
Runs: `runs/20260715-fortress-{recon3,recon4,recon5,natural12,pathbits,iwram,unlock}/`.

## Regrant removal (done)

1. `SMA4WhistleExecutor.use_second_whistle` — no `grant_whistles`; fails with
   `reason=no_whistle_in_inventory` if slot empty.
2. `use_whistle` MetaState: keeps inventory whistle only if
   `two_whistles_hand_granted` (else one-whistle acquire is spent).
3. `use_whistle_again.injected_facts` → `()`.
4. ROM check: grant 2 → use_first → use_second **without regrant** → cursor
   `[128,144]`, world raw `8` (warp 5-8 screen).

### Planner-class consequences

| Config | Expected |
|---|---|
| `hand_granted=True` | W8 skip still works; injected fact is `whistle_hand_granted` only |
| `hand_granted=False` (acquire_1_3 only) | **Cannot** close two-whistle skip (honest); warpless-blocked → `found=False` |

Prior `--no-hand-grant` ×5.7 “skip” used the regrant cheat for the second spend.

## Next options
1. Diff map path/tile RAM against a known-good human save that can walk to
   fortress (or find the path-bit that 1-2 clear is supposed to set).
2. Stage/boot Tier-3 fortress entry (labeled) only if path unlock stays opaque.
3. Until then: keep one-whistle acquire thesis; W8 skip demo stays on
   `grant_two_whistles` (Tier-1) without warp-zone regrant.

## Verification
```bash
pytest tests/test_meta_planner.py tests/test_options.py tests/test_acquire_whistle_solution.py -q
# ROM: grant 2 → use_first → use_second (no regrant) succeeds
./venv/bin/python scripts/bench_sma4_whistle_rom.py --out runs/20260715-sma4-whistle-rom-bench-noregrant-open
./venv/bin/python scripts/bench_sma4_whistle_rom.py --warpless-blocked --out runs/20260715-sma4-whistle-rom-bench-noregrant-blocked
./venv/bin/python scripts/bench_sma4_whistle_rom.py --no-hand-grant --warpless-blocked \
  --out runs/20260715-sma4-whistle-rom-bench-acquire-noregrant-blocked
```
