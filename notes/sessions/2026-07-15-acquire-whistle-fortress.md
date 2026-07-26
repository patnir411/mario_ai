# 2026-07-15 — Tier-2 `AcquireWhistle_fortress` (W1 Fortress whistle)

## Result

**DONE.** `data/solutions/sma4/acquire_whistle_fortress.json`
(`solved=true`, `replay_verified=true`, 1018 frames from door entry).

Map entry `(96,96)` after 1-1/1-2 clear bits is the real W1 Fortress (not the
mislabeled sky node at `(160,64)`). Leaf poke must write `POWERUP=3` only —
writing `POWERUP_SET` collapses the form.

## Route (SMA4-specific)

1. Leaf fortress spawn `runs/sma4_cache/1-fortress_pwing_leaf_entry.pkl` (not mid-level door).
2. Coverage prefix → door area `x≈1705` (`runs/20260717-fortress-pwing-to-door/`).
3. `LEFT+B×50` align, then roof/chest script (same fly→UP→A/B).
4. `UP` → map; idle + hold `B` unlocks L-menu.

Prior inventory whistle merges across the door restore so 1-3 + fortress stack
to two `0x0C` (executor copies slots before replaying).

## Injected facts (honesty)

| Fact | Why |
|---|---|
| `pwing_fortress_entry_snapshot` | Leaf fortress spawn entry (replaces mid-level door snap) |
| `leaf_rehold_during_route` | POWERUP re-written if damage drops form |
| `pspeed_seeded_in_entry_snapshot` | Entry root already has P-speed=127; the July 26 write-ledger audit confirmed no runtime P-speed poke |
| `prior_whistle_inventory_merged` | Only when chaining after another acquire |
| ~~`fortress_door_entry_snapshot`~~ | **REMOVED** — coverage prefix from pwing spawn to door |
| ~~`fortress_inventory_rehosted_to_pre_door_map`~~ | **REMOVED** — `B`-settle unlocks L-menu |

## Wiring

- `SMA4WhistleExecutor.acquire_whistle_fortress` (inventory merge; chest-truncate + B-settle exit)
- `build_sma4_whistle_rom_library(hand_granted=False)` exposes both acquires;
  `two_whistles_acquired` keeps a spare for `use_whistle_again`.
- Option snapshots are first-wins (uniform-cost must not clobber a usable post-state).
- Tests: `tests/test_acquire_whistle_fortress_solution.py` + updated
  `tests/test_meta_planner.py`.

## ROM rebench (`--no-hand-grant`)

| Config | greedy | uniform_cost | speedup | artifact |
|---|---|---|---|---|
| open (with rehost, superseded) | 69000 | 13192 skip | 5.23× | `…-two-acquire-open/` |
| open **no rehost** | 69000 | 12817 skip | 5.38× | `…-norehost-open/` |
| open **no door snap** | 69000 | 13320 skip (`1_3→fortress→…`) | **5.18×** | `…-nodoor-snap-open/` |
| `--warpless-blocked` no door snap | stuck | 13320 skip | n/a | `…-nodoor-snap-blocked/` |

No `whistle_hand_granted` / regrant / rehost / `fortress_door_entry_snapshot`.

## Artifacts

- Solution: `data/solutions/sma4/acquire_whistle_fortress.json`
- Entry: `runs/sma4_cache/1-fortress_pwing_leaf_entry.pkl`
- Door prefix: `runs/20260717-fortress-pwing-to-door/attempt.json`
- Parent spawn: `runs/sma4_cache/1-fortress_pwing_entry.pkl`
- Visuals: `runs/20260715-acquire-whistle-fortress/`

## Verification

```bash
pytest tests/test_acquire_whistle_fortress_solution.py tests/test_meta_planner.py -q
MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/bench_sma4_whistle_rom.py --no-hand-grant \
  --out runs/20260715-sma4-whistle-rom-bench-two-acquire-open
MARIO_AI_SMA4_ROM=... ./venv/bin/python scripts/bench_sma4_whistle_rom.py --no-hand-grant --warpless-blocked \
  --out runs/20260715-sma4-whistle-rom-bench-two-acquire-blocked
```

## Rehost burn (2026-07-15 evening)

Chest-room `UP` exit left L-menu locked; holding **B (~10–40f)** after idle unlocks
it natively. Executor truncates the scripted path at chest open (old tail was
mid-room wander), then `UP` → idle 200 → `B` 40. Rebench: **5.38×** open /
blocked still skips (`runs/20260715-sma4-whistle-rom-bench-norehost-{open,blocked}/`).

## Door-snap burn (2026-07-17)

Replaced mid-level door entry with pwing-spawn + coverage door prefix + align.
Dropped `fortress_door_entry_snapshot`. Still injected: P-Wing fortress entry,
leaf rehold, and P-speed seeded in that entry root. The former
`pspeed_poke_during_fly` label was corrected by the July 26 runtime ledger.

## Next honesty burn

Live overworld→`(96,96)`→enter (drop `pwing_fortress_entry_snapshot`), then
supply power legitimately and drop the leaf rehold/root-seeded P-speed.
