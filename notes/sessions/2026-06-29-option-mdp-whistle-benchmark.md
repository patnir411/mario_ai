# 2026-06-29 — Option-MDP + Whistle Planning Benchmark Scaffold

## Goal

Start the actual SMB3/SMA4 planning benchmark rather than grinding more platforming levels:

> How little meta-intelligence does any%-via-whistle planning require once low-level control is
> exact?

This pass built Phase 1 (Option-MDP scaffold) and Phase 2 (whistle spend/warp recon). It did not
run the full multi-world benchmark.

## Phase 1 — Option-MDP Scaffold

Added `mario/options.py`:

- `MetaState(world, node, cleared, inventory, flags)`
- `KnowledgeTier`
  - Tier 0: generic verified options
  - Tier 1: item already in inventory
  - Tier 2: black-box verified option
  - Tier 3: subgoal hint
  - Tier 4: LLM/disassembly proposer
  - Tier 5: full human script
- `OptionCost`, `OptionResult`, `OptionContext`
- `Option` and `OptionLibrary`
- `search_options(...)`: resettable BFS over options; opaque-effect options are executed and their
  observed post-state is logged as a discovered effect.
- `SMA4SolutionExecutor`: bridge from option execution to the existing cached replay path.
- `load_sma4_clear_level_option(...)` / `load_default_sma4_clear_options(...)`: wraps
  `data/solutions/sma4/{1-1,1-2}.json` as verified Tier-0 `ClearLevel` options.

Added `tests/test_options.py`:

- Proves tier gating blocks a high-tier shortcut until the search is allowed to use that tier.
- Proves an opaque-effect whistle analogue works: the search executes `use_whistle`, observes that
  it lands in World 8, then applies `clear_bowser`.
- Proves SMA4 `1-1`/`1-2` solution JSONs wrap as verified Tier-0 clear options with measured costs.

This is still a symbolic scaffold, not the final benchmark runner. The important design point is
that opaque effects are discovered by execution and logged with branch/option-call/injected-fact
counters.

## Phase 2 — SMA4 Whistle Plumbing Recon

Added `scripts/probe_sma4_whistle.py`.

Passing artifact:

- `runs/20260629-130158-sma4-whistle-probe/report.json`

Confirmed/probed RAM and controls:

- Inventory slots: `0x03002C2E..0x03002C51`
- Warp whistle item id: `0x0C`
- Item menu flag: `0x03003772` flips `0 -> 1` when the menu is open
- World byte: `0x03002A69`
- Map cursor X/Y: `0x03003DE4` / `0x03003DE0`
- Map event byte tracked in the probe: `0x03003774`
- `L` opens the inventory/item menu on the map
- `A` uses the selected item
- Stable-Retro/mGBA RAM writes work via `env.data.memory.assign(addr, "|u1", value)`

Spend-path verdict:

- One hand-granted whistle from World 1 reaches the special warp-zone map, not World 8.
- Raw world byte `8` is the warp-zone map; the adapter's current `world = raw + 1` normalization
  reports that as World 9, so Phase 3 must special-case this.
- A second whistle used from the warp-zone map opens the 5-8 warp-zone screen.
- `RIGHT` then `A` selects the World-8 pipe; dismissing Bowser's letter with `A` lands on World 8.
- Raw world byte `7` confirms World 8.
- The existing behavioral mode classifier reports the World-8 map as `menu` because its cursor Y
  is not on the early World-1 0x20-grid assumption. Do not use the current `mode == "overworld"`
  check as the only World-8 map detector.

Important caveat:

- The probe hand-grants/re-grants whistles to validate the spend and warp mechanics. This is Tier-1
  injected knowledge. The real benchmark still needs verified `AcquireWhistle` options that populate
  inventory through game mechanics, not RAM pokes.
- Hand-granting two whistles at boot leaves one visible after the first spend, but that copied
  state did not reliably reopen the item menu in the settled warp-zone screen. Re-granting in the
  warp-zone state made the spend path deterministic. Phase 3 should treat this as a hand-poke
  limitation until real acquisition proves the inventory metadata path.

## Verification

- `./venv/bin/python -m pytest -q tests/test_options.py`
  - `3 passed`
- `./venv/bin/python -m pytest -q tests/test_options.py tests/test_overworld_search.py tests/test_adapter_search.py`
  - `20 passed`
- `./venv/bin/python -m pytest --disable-warnings`
  - `72 passed / 10 skipped`
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python -m pytest -q tests/test_sma4_overworld.py`
  - `5 passed`
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python scripts/probe_sma4_whistle.py`
  - Passing artifact: `runs/20260629-130158-sma4-whistle-probe/report.json`

## Recommended Phase 3

Build the benchmark runner in this order:

1. Add effect-opaque item-use options:
   - `UseWhistle1`: precondition inventory contains whistle; effect opaque; executor observes raw
     world `8` / warp-zone state.
   - `UseWhistle2`: precondition inventory contains whistle and current raw world is warp-zone;
     effect opaque; executor observes the 5-8 warp-zone screen.
   - `SelectWorld8Pipe`: precondition raw world is warp-zone and cursor on the 5-8 screen; effect
     opaque; executor observes raw world `7`.
2. Start Tier-1 benchmark with two hand-granted whistles. This tests whether the meta-search
   discovers the payoff of spending consumables without pre-labeling "`use_whistle -> World 8`".
3. Then replace hand-granted inventory with Tier-2 black-box `AcquireWhistle` options:
   - First whistle: 1-3 white-block route-data option.
   - Second whistle: fortress/other W1 whistle option.
4. Only after that reduce knowledge:
   - Tier 3: hint the 1-3 white block but not the duck duration.
   - Tier 4: LLM/disassembly proposer emits candidate option definitions, emulator verifies.
   - Tier 0 raw search baseline is expected to fail on the hidden white-block trick; keep it as an
     honest negative rather than spending compute there.

## Phase 3 (started, Claude pass) — planner-class contrast on the opaque whistle library

Built the planning-layer benchmark core: the sharp test of the thesis is not "can a planner reach
World 8" but "which *class* of planner discovers the warp-whistle skip". Added a **greedy / myopic**
planner as the no-meta-intelligence baseline and contrasted it with the resettable searches on the
*same* effect-opaque option library.

Added `mario/meta_planner.py`:

- `greedy_plan(...)`: hill-climbs a myopic heuristic (`default_world_heuristic` = distance to a
  beaten game), one-step lookahead, no backtracking. By default it will NOT consider opaque-effect
  options (`exploit_opaque=False`) — a planner with no exploration cannot value a payoff it cannot
  predict. This is the modelled baseline, not a bug.
- `search_options_uniform_cost(...)`: Dijkstra over `OptionCost.frames`; the cost-optimal
  counterpart of `search_options` (BFS), so "the whistle skip is cheaper" is a rigorous claim, not a
  hop-counting artifact. Like BFS it executes opaque options and logs discovered effects.
- `build_whistle_benchmark_library(WhistleBenchmarkConfig)`: a symbolic SMB3 any%-via-whistle option
  set — warpless world-by-world advance (1->...->8) + `clear_bowser` (fully predictable), plus
  `acquire_whistle` (Tier-1 hand-granted by default) and the flagship **effect-opaque** `use_whistle`
  (its World-8 warp must be discovered by execution). `warpless_blocked=True` removes the warpless
  route for the sharpest contrast.
- `run_whistle_benchmark(...)`: runs greedy / BFS / uniform-cost and returns a per-planner report
  plus a `contrast` summary (who found the goal, who discovered the skip, the greedy-vs-cheapest
  frame gap).

Added `scripts/bench_whistle_planning.py` (CLI + JSON artifact + table) and
`tests/test_meta_planner.py` (6 tests).

Result (`runs/20260629-175809-whistle-planning-bench/report.json`):

| planner | found | hops | frames | discovered skip |
|---|---|---|---|---|
| greedy | True | 8 | 69000 | **False** |
| bfs | True | 3 | 6300 | True |
| uniform_cost | True | 3 | 6300 | True |

- **greedy completes the game the long warpless way but never discovers the 6-world whistle skip**;
  resettable search discovers the opaque `use_whistle -> World 8` payoff and returns a **10.95×
  cheaper** plan (69000 -> 6300 symbolic frames).
- `--warpless-blocked`: greedy is **stuck (found=False)** while search still escapes via the whistle
  — the sharpest version of the contrast.
- Honesty: option costs are *symbolic* planning-layer stand-ins; the option endpoints + whistle
  spend are ROM-verified separately. The Tier-1 hand-granted whistle is recorded as an injected fact
  (`whistle_hand_granted`, `whistle_in_inventory`) so the knowledge-ladder claim stays honest.

Verification:

- `./venv/bin/python -m pytest -q tests/test_meta_planner.py` -> `6 passed`
- `./venv/bin/python -m pytest --disable-warnings` -> `78 passed / 10 skipped`

## Phase 4 (Codex pass) — ROM-Backed Opaque Whistle Effects

Replaced the symbolic whistle transition with a ROM-backed option sequence while keeping the
symbolic planner-class contrast intact.

Code:

- `mario/adapters.py`
  - fixed `_classify_mode(...)` for raw world `7` (World 8) and raw world `8` (warp-zone), whose
    cursor Y positions are off the early World-1 0x20 grid.
  - added `world_raw`, `is_warp_zone`, and `item_menu_open` to normalized SMA4 info.
- `mario/options.py`
  - added `SMA4WhistleExecutor`, which hand-grants Tier-1 whistle inventory and drives the
    map-inventory UI against the real Stable-Retro/mGBA core.
- `mario/meta_planner.py`
  - added `SMA4WhistleROMConfig` and `build_sma4_whistle_rom_library(...)`.
  - The real library decomposes the skip into resettable opaque options:
    `grant_two_whistles` (Tier-1 known injection), `use_whistle` (observes raw `8` warp-zone),
    `use_whistle_again` (observes raw `8` 5-8 screen), `select_world8_pipe` (observes raw `7`
    World 8), then symbolic `clear_bowser`.
- `scripts/bench_sma4_whistle_rom.py`
  - CLI + JSON artifact for the ROM-backed planner-class benchmark.
- `tests/test_meta_planner.py`
  - fake-executor regression for the ROM option sequence.
  - pure classifier regression for raw world `7` / raw world `8` map states.

Default ROM-backed result:

- Artifact: `runs/20260629-202831-sma4-whistle-rom-bench/report.json`

| planner | found | hops | frames | option calls | discovered skip |
|---|---|---:|---:|---:|---|
| greedy | True | 8 | 69000 | 9 | False |
| bfs | True | 5 | 8900 | 14 | True |
| uniform_cost | True | 5 | 8900 | 8 | True |

Uniform-cost discovered effects:

- `use_whistle`: raw world `8`, cursor `[64,80]`, mode `overworld`
- `use_whistle_again`: raw world `8`, cursor `[128,144]`, mode `overworld`
- `select_world8_pipe`: raw world `7`, cursor `[32,80]`, mode `overworld`

The ROM-backed skip route is `69000 / 8900 = 7.75x` cheaper than the greedy warpless route under
the current mixed cost model. The costs remain partly symbolic because Bowser and warpless world
advance are not being executed in this pass; the whistle spend and W1->W8 observation are real
emulator transitions.

Blocked-route result:

- Artifact: `runs/20260629-202846-sma4-whistle-rom-bench/report.json`
- `--warpless-blocked`: greedy is stuck (`found=false`, 1 option call); BFS and uniform-cost still
  reach World 8 through the ROM-backed whistle route.

Knowledge-ladder honesty:

- This remains **Tier 1** because `grant_two_whistles` hand-writes inventory slots
  `0x03002C2E..0x03002C51` with item id `0x0C`.
- The regrant inside the warp-zone state is still recorded as an injected fact
  (`whistle_regranted_in_warp_zone`). A real acquisition path should remove that injected fact by
  letting the game populate any hidden inventory metadata naturally.

Verification:

- `./venv/bin/python -m pytest --disable-warnings` -> `80 passed / 10 skipped`
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python -m pytest -q tests/test_sma4_overworld.py` -> `5 passed`
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python scripts/bench_sma4_whistle_rom.py`
  - artifact: `runs/20260629-202831-sma4-whistle-rom-bench/report.json`
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python scripts/bench_sma4_whistle_rom.py --warpless-blocked`
  - artifact: `runs/20260629-202846-sma4-whistle-rom-bench/report.json`

Next:

- Replace Tier-1 hand-granted whistles with verified `AcquireWhistle` options. The first target is
  the 1-3 white-block route-data trick; the second target is the other early whistle source.
