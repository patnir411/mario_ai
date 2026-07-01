# Sutton (1999) — Policy Gradient Methods for RL with Function Approximation

**Paper:** Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour,
*Policy Gradient Methods for Reinforcement Learning with Function Approximation* (NIPS 1999).

**PDF in repo context:** uploaded as `REINFORCE_95a1.pdf` (same paper).

**Goal:** Understand the paper end-to-end — Policy Gradient Theorem, REINFORCE, baselines,
advantages, compatibility, policy iteration — by building, running, and analyzing experiments
on Super Mario Bros. 1-1 (primary) with extensions to multi-level scale.

---

## What this lab adds to mario_ai

The main repo implements **search → BC → DAgger** (imitation). This lab implements the paper's
alternative: **direct policy optimization** via \(\nabla_\theta \rho(\pi_\theta)\).

| Repo today | This lab |
|---|---|
| Supervised CE on teacher actions | \(\Delta\theta \propto \nabla\log\pi_\theta(a|s)\,\hat{A}(s,a)\) |
| Shared-trunk value head (BC auxiliary) | Independent / compatible critic (paper §2–3) |
| Offline shards | On-policy rollouts + optional oracle labels from `label_state` |
| `mario/eval.py` completion metrics | PG-specific: return variance, grad alignment, visitation \(d^\pi\) |

---

## Reading order (paper ↔ code)

| Step | Paper | Repo files | Lab doc |
|---|---|---|---|
| 1 | Abstract + §1 Policy Gradient Theorem | `mario/reward.py`, `DESIGN.md` §6 | [PAPER_MAP.md](PAPER_MAP.md) §1 |
| 2 | §2 Compatibility + Theorem 2 | `mario/policy.py`, `mario/value.py` | [PAPER_MAP.md](PAPER_MAP.md) §2 |
| 3 | §3 Advantages + baselines | `mario/label.py` (`per_action_value`) | [PAPER_MAP.md](PAPER_MAP.md) §3 |
| 4 | §4 Policy iteration convergence | `mario/dagger.py` (contrast: imitation ≠ PG) | [PAPER_MAP.md](PAPER_MAP.md) §4 |
| 5 | Run experiments | `scripts/` below | [EXPERIMENTS.md](EXPERIMENTS.md) |
| 6 | Analyze artifacts | `artifacts/SCHEMA.md` | [analysis/README.md](analysis/README.md) |

**Suggested first pass:** read PAPER_MAP §1–3, skim ARCHITECTURE, run Phase 0 sanity check, then
Phase 1 Experiment 01.

---

## End-to-end roadmap (5 phases)

```
Phase 0 ──► Infrastructure + sanity (rollout logger, reward contract)
    │
Phase 1 ──► REINFORCE family (Exp 01–03): raw → baseline → learned V
    │
Phase 2 ──► Actor–critic + compatibility (Exp 04–06): shared vs compatible critic
    │
Phase 3 ──► Oracle ablations (Exp 07–08): search-labeled advantages, gradient alignment
    │
Phase 4 ──► Comparisons at scale (Exp 09–11): vs BC/DAgger, chattering demo, multi-level
    │
Phase 5 ──► Synthesis notebook + written findings → FINDINGS.md
```

Details per phase: [PHASES.md](PHASES.md).

---

## Folder layout

```
learn_papers/sutton_1999/
├── README.md                 ← you are here
├── PAPER_MAP.md              ← theorem ↔ code mapping
├── EXPERIMENTS.md            ← full experiment matrix
├── PHASES.md                 ← implementation phases + acceptance criteria
├── ARCHITECTURE.md           ← modules to build, interfaces
├── FINDINGS.md               ← fill in as experiments complete
├── configs/                  ← one YAML per experiment
├── scripts/                  ← experiment runners (thin wrappers)
├── mario_pg/                 ← PG-specific Python package (new code lives here)
├── analysis/                 ← post-hoc plotting / tables
├── artifacts/                ← gitignored run outputs (schema documented)
│   └── SCHEMA.md
└── notes/                    ← scratch / derivations
```

---

## Prerequisites

```bash
# From repo root — native arm64 venv (see CLAUDE.md)
uv venv --python 3.13 venv
source venv/bin/activate
pip install -e ".[dev]"   # or project’s usual install path
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify base stack
./venv/bin/python -m pytest tests/ -q
./venv/bin/python -m mario.eval runs/<existing_checkpoint> 3   # if checkpoint exists
```

**Level:** World 1-1 only until Exp 09+. Chunk frames default **8** (match production); Exp 05
ablations use cf=4 near hazards.

**Compute:** Phase 1–3 are CPU-friendly (single level, &lt;2k episodes). Oracle experiments
(`label_state` per step) are expensive — budget ~minutes per 200 states on one core.

---

## Quick commands (once implemented)

```bash
# Phase 0 — rollout + logging smoke test
./venv/bin/python learn_papers/sutton_1999/scripts/00_sanity_rollout.py

# Phase 1 — REINFORCE on 1-1
./venv/bin/python learn_papers/sutton_1999/scripts/run_experiment.py --config configs/exp01_reinforce.yaml

# Analyze one run
./venv/bin/python learn_papers/sutton_1999/analysis/plot_learning_curves.py \
    --run artifacts/runs/exp01_reinforce_*/

# Run full Phase 1 matrix
./venv/bin/python learn_papers/sutton_1999/scripts/run_phase.py --phase 1
```

---

## Success criteria (lab complete)

| Criterion | Evidence |
|---|---|
| Policy Gradient Theorem understood operationally | Exp 07: estimated \(\nabla\rho\) correlates with oracle direction |
| Variance reduction from baselines | Exp 02–03: \(\mathrm{Var}(\hat{g})\) drops with learned \(V\) |
| Compatibility condition understood | Exp 05: compatible critic has higher \(\cos(\nabla_{true}, \nabla_{est})\) |
| PG vs imitation tradeoffs articulated | Exp 09: BC/DAgger vs actor–critic at fixed wall-clock |
| Scale implications documented | Exp 11 + [FINDINGS.md](FINDINGS.md) §Scale |

---

## Related repo docs

- `DESIGN.md` §8–10 — distillation / DAgger / value-guided search (contrast with PG)
- `V4_FINDINGS.md` — why generalist imitation plateaus (motivation for PG study)
- `mario/label.py` — oracle Q for teacher policy at arbitrary prefixes
