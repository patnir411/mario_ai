# 2026-06-26 — Cross-game Adapter Push: SML and SMA4/SMB3 1-1

## Objective

Extend the SMB1 search-teacher project toward nearby Mario games, starting with Super Mario Land and Super Mario Advance 4 / Super Mario Bros. 3, without treating the NES-specific environment as the permanent core abstraction.

The working hypothesis stayed intact: use the emulator as the forward model and keep search as the solver. The adapter owns game-specific boot, RAM, actions, snapshot/restore, terminal checks, and progress.

## Implemented State

Core additions now present in the working tree:

- `mario/adapters.py`
  - `GameAdapter` protocol.
  - `SMB1Adapter`.
  - `SMLAdapter` using PyBoy.
  - `SMA4Adapter` using Stable-Retro/mGBA.
- `mario/search.py`
  - `beam_search_adapter(...)`, a generic adapter-backed beam search with optional JSONL tracing.
- `scripts/solve_sml.py`
  - SML solver entrypoint.
- `scripts/solve_sma4.py`
  - SMA4 World 1-1 solver entrypoint.
  - `finish_card_suffix(...)`, a local frame-level finisher for the SMA4 roulette-card end.
- `tests/test_adapter_search.py`
  - Generic adapter contract tests.
  - SMA4 card-hit success regression.
- `tests/test_sml_adapter.py`
  - SML adapter/snapshot checks, gated by `MARIO_AI_SML_ROM`.

Local ROM convention:

- `roms/` is ignored in `.gitignore`.
- Do not commit ROMs or generated videos/checkpoints.

## ROMs Observed Locally

- `roms/Super Mario Land (World) (Rev 1).gb`
  - SHA1 `418203621b887caa090215d97e3f509b79affd3e`
- `roms/Super Mario Advance 4 - Super Mario Bros. 3 (USA, Australia) (Rev 1).gba`
  - SHA1 `532f3307021637474b6dd37da059ca360f612337`
- `roms/Super Mario Advance 2 - Super Mario World (USA, Australia).gba`
  - SHA1 `5101ddf223d1d918928fe1f306b63a42ada14a5e`

## SMA4 / SMB3 1-1 Result

The current local SMA4 1-1 artifact is:

- `data/solutions/sma4/1-1.json`

Key fields:

- `solved: true`
- `replay_verified: true`
- `solver: beam_search_adapter+card_suffix`
- `chunk_frames: 1`
- path length: 1130
- card suffix:
  - cut chunk: 176
  - delay: 26 frames
  - jump action: `RIGHT+A`
  - hold: 32 frames
  - success frame: 74
- post-clear transition: 489 frames

Visual artifacts:

- `runs/sma4_1_1_playthrough/play.mp4`
- `runs/sma4_1_1_playthrough/contact_sheet.png`

Verification:

- Full test suite: `./venv/bin/python -m pytest` -> 55 passed, 5 skipped.
- Direct saved-solution replay verified clear and post-level transition.

## Important Technical Findings

SMA4 1-1 does not behave like SMB1 flagpole progress near the end:

- The world `x_pos` caps near the end boundary around `2800`.
- Generic progress-only beam search reaches the roulette-card area alive, then runs under or past the card.
- Many suffix states deduplicate into the same capped-progress bucket, so x-progress alone is a bad late-game objective.
- `goalcard` alone is not a reliable terminal signal; it can change during ordinary card cycling.
- A true roulette-card hit was observed as `endwalk=255`, with the card shell emptying and a star dropping.
- The adapter success predicate is now `endwalk != 0`.

The current finisher is pragmatic and level-specific:

- Replay the near-end beam prefix.
- Snapshot approach states with `2500 <= x_pos <= 2635`.
- Try short frame-level `RIGHT+B` delays followed by `RIGHT+A` / `RIGHT+A+B`.
- Accept only the adapter's success predicate, then replay from reset.

This is enough to solve SMA4 1-1, but it should not be treated as a general GBA end-level solver.

## Recommended Next Experiments

1. Generalize late-goal scoring for GBA Mario.
   - Add screen-space or object-state scoring near terminal objects.
   - Avoid relying only on world `x_pos` where it caps.
   - Candidate signals: rendered-frame card/Mario geometry, screen-X RAM discovery, end-level routine bytes.

2. Make SMA4 1-1 fully reproducible from a fresh solve.
   - Keep `finish_card_suffix(...)`, but add a cheaper suffix-search trace/contact sheet output.
   - Add a replay-video script for adapter games instead of ad hoc rendering snippets.

3. Expand to the next GBA target only after the terminal-object scoring is cleaner.
   - SMA4 1-2 or SMA4 1-1 variants are closest.
   - SMA2 / Super Mario World will need a separate RAM map for position, death, level mode, overworld/level boot, and success.

## Current Direction

For these Mario-family ports, keep search as the solver and use per-game adapters. A generalist policy should not replace search at this stage. The useful learned model, if any, is still a search prior or value heuristic after the adapter and terminal semantics are reliable.
