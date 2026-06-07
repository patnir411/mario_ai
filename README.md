# mario_ai

A from-scratch AI that **beats and speedruns Super Mario Bros (NES)**, running locally on
an Apple-silicon Mac. It uses the emulator itself as a forward model: deterministic
snapshot/restore over `gym-super-mario-bros` / `nes-py`, searched with beam + Go-Explore,
then distilled into a small policy. Everything is verified by artifacts — JSON results,
pytest, and human-readable contact-sheet PNGs.

## Status

- **any% — full game beaten, 8/8 (`beat_game=True`).** 1-1 → 1-2 → 4-1 → 4-2 → 8-1 → 8-2 → 8-3 → 8-4.
- **Every hard castle / water level solved:** 1-4, 2-2, 4-4, 7-2, **7-4** (the notorious 6-gate
  multi-loop), **8-4** (down-pipe chain → water → Bowser → axe).
- 16 distinct levels solved; 1-1 also cleared by a distilled neural policy (BC + DAgger).

Cached solutions live in `data/solutions/*.json` (action sequences, replayable from reset);
visual proof in `runs/*_solved_contact.png`.

## How the hard levels fell

The recurring wall was an **instrumentation bug, not difficulty**: `gym-super-mario-bros`
fast-forwards pipe/area transitions *inside* `env.step`, so post-step RAM is blind to entries.
The fix — detect transitions via a per-frame live-RAM x-jump (or raw `nes_py` frames) — plus
disassembly-grounded route decoding (SMB's `ProcLoopCommand` loop gates, `HandlePipeEntry`
predicate) cracked 2-2, 7-2, 8-4, 4-4, and 7-4.

## Layout

- `mario/` — env wrapper, RAM map, search (`beam_search`, `coverage_search`, `search_from_state`),
  reward, observation, policy, value net, renderer.
- `scripts/` — per-level solvers (`solve_84_*.py`, `solve_castle.py`, `solve_44_*.py`,
  `solve_74_seg2.py`, …), `stitch_solutions.py`, `update_status.py`.
- `tests/` — determinism, snapshot exactness, reward invariants, transition detector (pytest).
- `CLAUDE.md` — curated working narrative + status (ground truth in `runs/status.json`).
- `DESIGN.md` — architecture; `BEAM_SEARCH.md`, `*_FINDINGS.md` — deep dives.

## Setup

```bash
uv venv --python 3.13 venv          # native arm64 (x86/Rosetta python disables torch MPS)
./venv/bin/pip install -e .         # gym-super-mario-bros 8.0 / nes-py 9.0 / torch
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m pytest -q
```

Replay a cached solve / stitch a playthrough:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/stitch_solutions.py "any%"
```

> Note: the SMB ROM ships with the `gym-super-mario-bros` package and is **not** in this repo.
> Large artifacts (`venv/`, videos, datasets, model weights) are gitignored.
