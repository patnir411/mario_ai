# 2026-06-27 — SMA4 Cached World-1 Sweep

## Goal

Use cached replay to advance past solved World-1 levels without re-solving, then attempt the next
reachable nodes with telemetry. No flight subsystem and no real meta-planner in this pass.

## Cached Fast-Forward

Added a cached replay path in `mario/overworld_search.py`:

- `remap_solution_path(...)`
- `replay_cached_solution(...)`
- `fast_forward_cached_solutions(...)`

`scripts/solve_sma4_world.py` now accepts:

- `--fast-forward-solved`
- `--target-cursor=X,Y`
- `--target-label=...`

Important details:

- Cached paths are action indices, so the replay helper remaps through source `action_names` when
  possible.
- `data/solutions/sma4/1-2.json` had a valid path but stale `action_names`; the generated JSON and
  run summary were corrected to list the current focused P-speed action names.
- 1-1 also needed a cached boot-entry snapshot. Re-entering 1-1 from the map desyncs and dies at
  `x_max=168`, so `runs/sma4_cache/1-1_entry.pkl` was generated from a fresh-process
  `boot_to_level(1,1)` entry.

Working fast-forward:

- `runs/20260627-113025-sma4_world/fast_forward/1-1_cached_replay.json`
- `runs/20260627-113025-sma4_world/fast_forward/1-2_cached_replay.json`

Both cached replays reported `solved=true` and `settled_to_map=true`.

## Entry Snapshots

New/confirmed entry snapshots:

- `runs/sma4_cache/1-1_entry.pkl`
- `runs/sma4_cache/1-2_entry.pkl`
- `runs/sma4_cache/1-3_entry.pkl`
  - cursor `(160,32)`
  - entry `x_pos=24`, `y_pos=128`, `cleared=2`
- `runs/sma4_cache/1-fortress_entry.pkl`
  - cursor `(160,64)`
  - entry `x_pos=24`, `y_pos=64`, `cleared=2`

The `(160,64)` label is provisional. The contact sheet does not look like a fortress/castle; it
looks like a sky/athletic side level.

## 1-3 Attempt

Artifact directory:

- `runs/20260627-113530-sma4_snapshot_1_3_pspeed/`

Key files:

- `beam_trace.jsonl`
- `coverage_trace.jsonl`
- `attempt_summary_interrupted.json`
- `beam_best_contact.png`

Result:

- `solved=false`
- best beam `x_max=2546`
- final best state: `x_pos=2546`, `y_pos=80`, `goalcard=2`, `endwalk=0`, `powerup=0`, `pspeed=1`
- coverage underperformed the beam in this bounded run (`x≈1453` in the tail of the trace)

Visual read:

- The normal route reaches the roulette-card area.
- The wall is another terminal card-timing miss: Mario reaches/runs past the visible card but does
  not trigger `endwalk`.
- This is not a flight wall.
- This is also separate from the whistle route.

Whistle assessment:

- The known 1-3 whistle requires standing on a specific white block and ducking for several
  seconds to fall behind the scenery.
- The current objective strongly rewards rightward progress and has no subgoal for waiting/ducking
  on that block, so the route-data trick is not expected to emerge from ordinary forward search.
- Treat whistle acquisition as a later route-data/subgoal option, not part of the normal-goal solve.

## `(160,64)` Side Node Attempt

Artifact directory:

- `runs/20260627-115920-sma4_snapshot_1_fortress_pspeed/`

Key files:

- `beam_trace.jsonl`
- `coverage_trace.jsonl`
- `attempt_summary_interrupted.json`
- `beam_best_contact.png`

Result:

- `solved=false`
- best beam `x_max=329`
- best non-terminal state: `x_pos=329`, `y_pos=229`, `goalcard=0`, `endwalk=0`, `powerup=0`
- beam terminal deaths around `x≈295`
- coverage underperformed (`x_max≈261` in tail trace)

Visual read:

- The screen is a sky/athletic platform/gap layout, not a fortress/castle room.
- The current P-speed-focused action set reaches the right edge/air and fails to land on the next
  platform. This looks like a precise platforming/jump-arc wall, possibly needing a more targeted
  vertical/platform cell objective or a level-specific macro, not flight yet.

## Code and Tests

Changed:

- `mario/overworld_search.py`
  - cached replay helpers,
  - target cursor/label support in `overworld_solve`,
  - entry snapshot persistence,
  - suffix telemetry fix in attempt summaries.
- `scripts/solve_sma4_world.py`
  - `--fast-forward-solved`,
  - `--target-cursor`,
  - `--target-label`.
- `mario/search.py`
  - `goal_suffix_search(..., time_budget_s=30.0)` to prevent diagnostic runs from hanging in broad
    suffix scans.
- `tests/test_overworld_search.py`
  - non-ROM cached path remap/replay regression.

Verification:

- `./venv/bin/python -m pytest --disable-warnings`
  - `69 passed, 10 skipped`
- `MARIO_AI_SMA4_ROM='roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba' ./venv/bin/python -m pytest -q tests/test_sma4_overworld.py`
  - `5 passed`
- `git diff --check`
  - clean

## Next

Do not build flight yet. The immediate blockers are:

1. Make the card finisher faster/more targeted for 1-3's normal route.
2. Correctly label the post-1-2 map nodes; `(160,64)` is not obviously the fortress.
3. Add a platforming-focused objective/macro for the `(160,64)` side-node gap if it remains on the
   required route.
4. Treat the 1-3 whistle as a route-data/subgoal option for the later meta-planner pass.
