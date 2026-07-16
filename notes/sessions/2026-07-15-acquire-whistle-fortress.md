# 2026-07-15 — Tier-2 `AcquireWhistle_fortress` (W1 Fortress whistle)

## Result

**DONE.** `data/solutions/sma4/acquire_whistle_fortress.json`
(`solved=true`, `replay_verified=true`, 1018 frames from door entry).

Map entry `(96,96)` after 1-1/1-2 clear bits is the real W1 Fortress (not the
mislabeled sky node at `(160,64)`). Leaf poke must write `POWERUP=3` only —
writing `POWERUP_SET` collapses the form.

## Route (SMA4-specific)

1. Door snap at `x≈1765` with leaf + pspeed (`runs/sma4_cache/1-fortress_door_entry.pkl`).
2. Left-run + RNG fly to `y≤2` near the `?` block.
3. Alternate `RIGHT+A+B` / `RIGHT` — y wraps `0→255…` then lands on the roof
   (`x≈1826`).
4. `UP` → chest room; **A/B** opens the chest (`0x0C`).
5. Exit to overworld at cursor `(96,96)`.

Prior inventory whistle merges across the door restore so 1-3 + fortress stack
to two `0x0C` (executor copies slots before replaying).

## Injected facts (honesty)

| Fact | Why |
|---|---|
| `fortress_door_entry_snapshot` | Door-area entry, not full overworld→door search |
| `leaf_rehold_during_route` | POWERUP re-written if damage drops form |
| `pspeed_poke_during_fly` | P-meter poke during the roof takeoff |
| `prior_whistle_inventory_merged` | Only when chaining after another acquire |
| ~~`fortress_inventory_rehosted_to_pre_door_map`~~ | **REMOVED** — treasure-room exit now truncates at chest, `UP` to map, idle, hold `B` (native L-menu unlock) |

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
| open **no rehost** | 69000 | 12817 skip (`1_3→fortress→use→use→pipe→bowser`) | **5.38×** | `…-norehost-open/` |
| `--warpless-blocked` no rehost | stuck | 12817 skip | n/a | `…-norehost-blocked/` |

No `whistle_hand_granted` / `whistle_regranted_in_warp_zone` /
`fortress_inventory_rehosted_to_pre_door_map`.

## Artifacts

- Solution: `data/solutions/sma4/acquire_whistle_fortress.json`
- Entry: `runs/sma4_cache/1-fortress_door_entry.pkl`
- Leaf start: `runs/sma4_cache/1-fortress_pwing_entry.pkl`
- Door path search: `runs/20260715-141410-sma4_snapshot_1_fortress_pspeed/`
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

## Next honesty burn

Replace `fortress_door_entry_snapshot` with live overworld→fortress→door (and/or
drop leaf/pspeed rehold).
