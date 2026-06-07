# mario_ai

A from-scratch AI that **beats and speedruns Super Mario Bros (NES)**, running entirely locally
on an Apple-silicon Mac. It uses the emulator itself as a forward model — deterministic
snapshot/restore over `gym-super-mario-bros` / `nes-py` — searched with beam + Go-Explore and a
global progress potential, with the hardest levels routed via 6502-disassembly analysis, then
distilled into compact neural policies. Every result is backed by an artifact: a replayable action
sequence, a passing test, and a human-readable contact-sheet PNG.

## Highlights

- **any% — the full game is beaten, 8/8 (`beat_game=True`):** 1-1 → 1-2 → 4-1 → 4-2 → 8-1 → 8-2 → 8-3 → 8-4.
- **Every hard level solved from scratch** — the underwater side-pipe levels (2-2, 7-2) and all
  three castles, including **7-4** (the notorious 6-gate multi-loop) and **8-4** (a down-pipe chain
  through water to Bowser and the axe).
- **16 distinct levels solved**, stitched into a single back-to-back showcase video.
- A **distilled neural policy** clears 1-1, and the full **search → distill → DAgger** learning
  pipeline is implemented end-to-end.
- **Fully verified & reproducible:** deterministic seeds, `render.replay` `beat=True` per solution,
  contact sheets, and a green pytest suite.

## Results

| Level(s) | Type | Method |
|---|---|---|
| 1-1 | overworld | learned net (BC + DAgger) / beam |
| 1-2, 1-3, 2-1 | overworld | beam search |
| 1-4 | castle | beam search |
| 2-2, 7-2 | underwater | hold-right side-pipe entry + area beam |
| 4-1 | overworld | beam search |
| 4-2 | warp-zone maze | Go-Explore (`coverage_search`) |
| 4-4 | castle (height maze) | gate-aware Go-Explore (lower-path descent) |
| 7-4 | castle (6-gate multi-loop) | per-triplet segmented search |
| 8-1, 8-2, 8-3 | overworld | beam search |
| 8-4 | castle → water → Bowser | down-pipe chain + side-pipe + axe beam |

Cached solutions live in `data/solutions/*.json` (action sequences, replayable from reset); visual
proof in `runs/*_solved_contact.png`; full playthroughs via `scripts/stitch_solutions.py`.

## How it works

**Forward-model search.** The NES emulator is wrapped (`mario/env.py`) to expose exact
snapshot/restore and chunked stepping, making it a perfect deterministic forward model. On top of
it:

- `beam_search` — width-bounded best-first search scored by a **global progress potential Φ**
  (`area_seq*K + (x − area_entry_x)`), so pipe/area transitions read as forward progress.
- `coverage_search` — beam + **Go-Explore** (cell archive over `(area, x-tile, y-tile)` + novelty),
  which cracks vertical/maze levels where a plain beam stalls.
- `search_from_state` — beam from any live state, used for in-run rescue.
- A value network (`mario/value.py`) for value-guided best-first search (ExIt-style).

**Disassembly-grounded routing.** The castle/water levels are gated by exact engine mechanics, so
they are routed against the SMB 6502 disassembly: `HandlePipeEntry`'s pipe-top metatile predicate,
`ProcLoopCommand`'s height-gates, and 7-4's multi-loop counters — turned into mechanic-aware search
shaping.

**Reliable transition detection.** Because the wrapper fast-forwards pipe/area transitions inside a
step, transitions are detected from a per-frame live-RAM x-jump (or raw `nes_py` frames) rather than
post-step RAM — the key signal that unlocked the underwater and castle routes.

**Verification spine.** `scripts/update_status.py` regenerates a status block from `runs/status.json`;
every solution is re-validated by `mario/render.replay` and a contact sheet; `pytest` guards
determinism, snapshot exactness, reward invariants, and the transition detector.

## Learning pipeline

The search acts as an expert teacher that is distilled into small policies:

- **Observation** (`mario/observation.py`) — a compact, ego-centric, position-free state
  (tile grid + scalars) that generalizes across levels.
- **Distillation** (`mario/label.py`, `mario/buffer.py`, `mario/train.py`) — search trajectories +
  perturb-and-recover coverage are labeled and behavior-cloned into a `MarioPolicy`; **DAgger**
  (`scripts/run_dagger.py`) adds closed-loop corrections.
- **Structured policy research** (`scripts/exp_entity.py`) — an object-centric schema
  (`{player, enemies, terrain}` tokens) fed to a tiny ~150K-param **entity-transformer** with a
  single-token action head, trainable in minutes on the M2 and fast enough to control in real time —
  the vehicle for cross-level / cross-game generalization.

See `V4_FINDINGS.md` for the consolidated write-up, including the cross-game ("any Mario game")
feasibility and cost/time analysis.

## Repository layout

- `mario/` — env wrapper, RAM map, search family, reward, observation, policy, value net, renderer,
  multi-stage runner.
- `scripts/` — per-level solvers (`solve_84_*.py`, `solve_castle.py`, `solve_44_*.py`,
  `solve_74_seg2.py`, …), dataset/training (`gen_dataset.py`, `run_dagger.py`, `exp_entity.py`),
  `stitch_solutions.py`, `update_status.py`.
- `tests/` — determinism, snapshot exactness, reward invariants, transition detector (pytest).
- `CLAUDE.md` — curated working narrative + live status; `DESIGN.md` — architecture;
  `BEAM_SEARCH.md`, `*_FINDINGS.md` — deep dives.

## Setup

```bash
uv venv --python 3.13 venv          # native arm64 (x86/Rosetta python disables torch MPS)
./venv/bin/pip install -e .         # gym-super-mario-bros 8.0 / nes-py 9.0 / torch
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m pytest -q
```

## Usage

```bash
# Stitch the full any% playthrough (replays cached solutions; beat_game=True)
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/stitch_solutions.py "any%"

# Solve a level from scratch (search) — e.g. a castle
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_castle.py 4 4

# Build a dataset + train + evaluate a learned policy
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/gen_dataset.py 1 1
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m mario.train
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m mario.eval <run_id> 5 1-1
```

> The SMB ROM ships with the `gym-super-mario-bros` package and is **not** in this repo.
> Large artifacts (`venv/`, videos, datasets, model weights) are gitignored; the small,
> replayable solution JSONs and contact sheets are committed.
