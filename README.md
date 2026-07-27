# mario_ai

> 🎮 **[Interactive explainer & walkthrough →](https://storage.googleapis.com/learn-mario-ai-422ee6/index.html?v=2)** — a visual, end-to-end tour of this project: the search→distill→DAgger journey, the hard-level debugging stories, and how each idea maps to an ML research interview. Made for fun, out of curiosity.

A from-scratch research system for solving Mario games locally on an Apple-silicon Mac. Its
strongest result is exact forward-model search over emulator snapshots: beam/coverage search,
mechanic-aware routing, and replay-gated action sequences. Learned policies are being tested as
search guides rather than assumed to replace the solver. Cross-game work uses emulator adapters,
and the current Super Mario Advance 4 (SMA4) route work is a segmented option-planning prototype.

The evidence standard is deliberately strict: a game result is positive only after independent
replay reaches the terminal condition. In the **July 25, 2026 audit** (updated July 27), SMB1 stock
coverage is **30/32 replay-verified**. Level 6-2 remains unsolved and 6-3 is quarantined after its
seed-0 replay failed.

## Highlights

- **any% — the full game is beaten, 8/8 (`beat_game=True`):** 1-1 → 1-2 → 4-1 → 4-2 → 8-1 → 8-2 → 8-3 → 8-4.
- **30/32 stock SMB1 levels replay-verified**, including the underwater side-pipe routes and the
  difficult 4-4, 7-4, and 8-4 castle routes.
- The full **search → distill → DAgger** research path is implemented, including entity-centric
  policies, held-out split checks, input-consistency regularization, and policy-guided search.
- On one local SMB1 1-1 benchmark, a learned prior preserved the solve while reducing beam
  expansions from **7005 to 2770 nodes (60.5% fewer; 2.53 plain/guided ratio)**. This is a single-level result; the local
  checkpoint and training rows do not yet have clean-clone training provenance.
- An adapter layer supports NES SMB1, GB Super Mario Land, and GBA SMA4 experiments without
  putting game-specific RAM assumptions in the generic search API.

## Results

| Result | Status | Method / evidence |
|---|---|---|
| SMB1 any% | 8/8, `beat_game=True` | cached segments replayed as one route |
| SMB1 stock set | 30/32 replay-verified | `data/solutions/[1-8]-[1-4].json` |
| SMB1 6-2 | unsolved | retained as an explicit negative |
| SMB1 6-3 | quarantined | prior candidate died during seed-0 replay |
| 1-1 learned search prior | 7005 → 2770 nodes | `notes/artifacts/2026-07-25-policy-guided-1-1.json`; training provenance incomplete |
| SMA4 post-1-2 → fortress entry | verified from one declared local root | exact across two repeats; zero direct RAM writes |
| SMA4 whistle route | segmented mixed route; maximum Tier 2 | both cached acquisition orders reach responsive World 8 with zero cursor writes; independent roots and symbolic Bowser remain |

Small canonical solution manifests live under `data/solutions/`. Generated datasets, checkpoints,
videos, contact sheets, and run reports remain local under ignored paths unless deliberately
promoted as small evidence artifacts.

## How it works

**Forward-model search.** The emulator supplies snapshot/restore and chunked stepping. Search
remains the solver:

- `beam_search` — a width-bounded frontier ranked by a progress/death heuristic. The project uses
  potential-inspired progress features; it does not claim the formal guarantee of
  potential-based reward shaping.
- `coverage_search` — a **Go-Explore-style** beam/archive hybrid (cell coverage
  over `(area, x-tile, y-tile)` plus novelty), useful on vertical and deceptive
  routes where a narrow beam stalls. It is not the full Go-Explore algorithm.
- `search_from_state` — beam from any live state, used for in-run rescue.
- Optional learned policy/value signals order or softly bias expansions. Hard top-k pruning is
  experimental because an inaccurate prior can remove the only successful action.

**Disassembly-grounded routing.** The castle/water levels are gated by exact engine mechanics, so
they are routed against the SMB 6502 disassembly: `HandlePipeEntry`'s pipe-top metatile predicate,
`ProcLoopCommand`'s height-gates, and 7-4's multi-loop counters — turned into mechanic-aware search
shaping.

**Adapters and options.** `mario.adapters.GameAdapter` isolates emulator-specific state, progress,
and terminal semantics. The SMA4 lane combines level and overworld adapters with an option library
and a symbolic meta-planner. Exact boundary hashes and intervention ledgers verify the live
post-1-2 `DOWN,DOWN,LEFT -> (96,96)` fortress entry. Physical-representative search preserves both
acquisition histories, and the adapter follows SMA4's live cursor-object pointer rather than a
fixed RAM pair; this removed the apparent order asymmetry and all cursor repair writes. The route
is still not one continuous Option-SMDP execution: it composes independent power-state roots, an
inventory merge, a fortress leaf rehold, and a symbolic Bowser edge. See
`notes/sessions/2026-07-27-sma4-live-cursor-pointer-tier2.md`.

**Verification spine.** `mario.solution_verification` and
`scripts/verify_stock_solutions.py` replay the canonical stock manifests. Cross-game CLI promotion
also requires an independent replay before a run may replace a canonical solution. Tests cover
snapshot determinism, restore integrity, reward/search contracts, adapter behavior, and publishing
gates.

## Learning pipeline

Search supplies labels to compact policies:

- **Observation** (`mario/observation.py`, `mario/entity.py`) — ego-centric tile/scalar features or
  structured player/enemy/terrain tokens.
- **Distillation** (`mario/label.py`, `mario/buffer.py`, `mario/train.py`) — search trajectories +
  perturb-and-recover coverage are behavior-cloned; DAgger drivers add learner-distribution
  corrections.
- **Input consistency** (`mario/consistency.py`) — penalizes policy-output changes under small input
  perturbations. This is a sensitivity surrogate, not Stable-BC's closed-loop dynamics criterion.
- **Search guidance** (`mario/entity_policy.py`, `mario/search.py`) — policy priors can bias
  expansion while the exact emulator still decides outcomes.

`V5_FINDINGS.md` and `V6_FINDINGS.md` preserve the generalist-policy experiments and the pivot to
learned search guidance, including their current provenance limits.

## Repository layout

- `mario/` — emulator wrappers/adapters, search, RAM/entity observations, policies, verification,
  options, and meta-planning.
- `scripts/` — SMB1 solvers and replay tools, learning experiments, SML/SMA4 drivers, benchmarks,
  and status/verification commands.
- `data/solutions/` — small canonical replay manifests; ROMs and large generated data are ignored.
- `tests/` — deterministic core, learning/data integrity, adapter/option, and CLI publishing tests.
- `CLAUDE.md` — curated working narrative + live status; `DESIGN.md` — architecture;
  `BEAM_SEARCH.md`, `*_FINDINGS.md` — deep dives.

## Setup

The NES and GBA stacks require incompatible `pyglet` versions, so use separate environments.

```bash
# NES / SMB1
uv venv --python 3.13 venv
uv pip install --python venv/bin/python -e '.[nes,dev]'
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m pytest -q

# GBA SMA4 + GB Super Mario Land (separate from NES)
uv venv --python 3.13 venv-gba
uv pip install --python venv-gba/bin/python -e '.[gba,sml,dev]'
```

## Usage

```bash
# Replay-gate all 32 stock manifests; currently expects 30 positives
./venv/bin/python scripts/verify_stock_solutions.py --expect-verified 30

# Stitch the verified any% playthrough (beat_game=True)
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/stitch_solutions.py "any%"

# Solve a level from scratch (search) — e.g. a castle
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/solve_castle.py 4 4

# Build a dataset + train + evaluate a learned policy
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/gen_dataset.py 1 1
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m mario.train
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m mario.eval <run_id> 5 1-1
```

> The SMB ROM ships with the `gym-super-mario-bros` package and is **not** in this repo.
> GBA/GB ROMs stay local under ignored `roms/`. Environments, videos, datasets, model weights,
> savestates, and run outputs are also ignored.
