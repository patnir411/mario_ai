# Experiment matrix (Sutton 1999 lab)

Each experiment: **one hypothesis**, fixed **1-1** (unless noted), outputs under
`learn_papers/sutton_1999/artifacts/runs/<exp_id>_<timestamp>/`.

Run via:
```bash
./venv/bin/python learn_papers/sutton_1999/scripts/run_experiment.py --config configs/<config>.yaml
```

---

## Summary table

| ID | Name | Phase | Hypothesis | Primary metric |
|---|---|---|---|---|
| 00 | sanity_rollout | 0 | Logging pipeline works | artifacts validate against SCHEMA |
| 01 | reinforce_raw | 1 | REINFORCE can increase return on 1-1 | `mean_return` vs episode |
| 02 | reinforce_const_baseline | 1 | Constant baseline reduces grad variance | `grad_var` ↓, same or better return |
| 03 | reinforce_learned_baseline | 1 | Learned \(V(s)\) beats constant baseline | `grad_var` ↓↓, sample efficiency ↑ |
| 04 | advantage_vs_return | 1 | Advantage target beats raw \(G_t\) | `grad_var`, episodes to 50% completion |
| 05 | compatibility_ablation | 2 | Compatible critic ↑ gradient alignment | `cos_grad` |
| 06 | policy_iteration_loop | 2 | Alternate critic/actor fits Theorem 3 | monotonic `mean_return` windows |
| 07 | oracle_q_gradient | 3 | Oracle \(Q\) recovers true gradient direction | `cos_grad` → 1.0 |
| 08 | oracle_advantage_ac | 3 | Search-labeled \(\hat{A}\) is sample-efficient AC | episodes to 90% completion |
| 09 | pg_vs_imitation | 4 | PG vs BC/DAgger at fixed wall-clock | completion @ T seconds |
| 10 | value_greedy_chatter | 4 | Greedy-from-\(\hat{Q}\) flips actions discontinuously | `action_flip_rate` |
| 11 | multi_level_interference | 4 | Single-net PG hits interference like V4 DAgger | per-level completion matrix |

---

## Phase 0

### Exp 00 — `sanity_rollout`

**Purpose:** Validate rollout, reward, logging before any learning.

**Procedure:**
1. Random policy, 10 episodes, 1-1, seed 0–9
2. Log every chunk: `obs_hash`, `action`, `reward`, `x_pos`, `done`, `log_prob` (if stochastic)

**Pass:** `episodes.csv` + `config.json` written; rewards finite; death/flag detected correctly.

**Config:** `configs/exp00_sanity.yaml`

---

## Phase 1 — REINFORCE family

### Exp 01 — `reinforce_raw`

**Hypothesis:** Even high-variance REINFORCE shows positive return trend on 1-1.

**Algorithm:**
```
for episode in 1..N:
  rollout stochastic π_θ, store (s_t, a_t, log π, r_t)
  G_t = discounted sum of r from t
  loss = -Σ_t log π(a_t|s_t) * G_t
  θ ← θ - α ∇loss
```

**Hyperparams (defaults):** `episodes=2000`, `lr=3e-4`, `gamma=0.99`, `chunk_frames=8`, `entropy_coef=0.01`

**Metrics:** `mean_return`, `completion_frac@30seeds`, `policy_entropy`, `grad_norm`

**Baselines:** none (first PG run)

---

### Exp 02 — `reinforce_const_baseline`

**Hypothesis:** \( \hat{A}_t = G_t - \bar{G}_{episode} \) lowers gradient variance without bias.

**Change vs 01:** subtract per-episode mean return from each \(G_t\).

**Metrics:** `grad_var` (primary), `mean_return` (secondary — should match or beat 01)

---

### Exp 03 — `reinforce_learned_baseline`

**Hypothesis:** Learned \(V_\phi(s)\) baseline beats constant baseline (paper §3).

**Change vs 02:** train \(V_\phi\) with MSE to \(G_t\) (or TD(λ)); use \(\hat{A}_t = G_t - V_\phi(s_t)\).

**Critic:** new `mario_pg/critic.py` — single obs vector, not K-stack (faster).

**Metrics:** `grad_var`, episodes to reach `completion_frac >= 0.5`

---

### Exp 04 — `advantage_vs_return`

**Hypothesis:** Critic predicting advantage (centered per state) beats critic predicting raw return.

**Arms:**
- A: \(\hat{A} = G_t - V(s_t)\) with \(V\) trained on returns
- B: \(\hat{A} = G_t\) (no centering)

**Metrics:** side-by-side `grad_var` distribution (boxplot input)

---

## Phase 2 — Actor–critic + compatibility

### Exp 05 — `compatibility_ablation`

**Hypothesis:** Compatible parameterization yields higher \(\cos(\nabla\rho^{oracle}, \nabla\rho^{est})\).

**Arms:**

| Arm | Actor | Critic |
|---|---|---|
| `compatible_linear` | softmax(\(W\pi \cdot \phi_{sa}\)) | linear advantage on \(\phi_{sa} - \mathbb{E}_\pi[\phi]\) |
| `mlp_independent` | `MarioPolicy` trunk | separate MLP → per-action advantage |
| `shared_trunk` | `MarioPolicy` | shared trunk value head (repo anti-pattern) |

**Procedure:** Freeze random \(\theta\); at 500 on-policy states, compute:
- Oracle direction from `label_state` Q (Exp 07 subroutine)
- Estimated direction from each critic parameterization

**Metrics:** `cos_grad` per arm (mean ± std over states)

---

### Exp 06 — `policy_iteration_loop`

**Hypothesis:** Alternating critic convergence + actor step improves return monotonically (local).

**Loop (K rounds):**
```
1. Roll out π_θ, collect trajectories
2. Fit f_w to minimize E[(Q^π - f_w)^2] or TD error  (until val loss plateaus)
3. Actor: θ ← θ + α E[∇log π · f_w(s,a)]  (one or few epochs)
```

**Metrics:** `mean_return` per round; critic val MSE

**Pass:** no catastrophic collapse; round-over-round median return non-decreasing for ≥3 rounds

---

## Phase 3 — Oracle (search-assisted) ablations

### Exp 07 — `oracle_q_gradient`

**Hypothesis:** With exact \(Q\) from search, estimated policy gradient aligns with theorem.

**Procedure:** At sampled prefixes along 1-1, call `fast_label` / `label_state`:
```python
q = label_result.per_action_value  # shape [N_ACTIONS]
g_true = sum_a (dlogpi[a] * q[a])   # for current π_θ
```
Compare to REINFORCE sample \(\nabla\log\pi(a|s) \cdot G_t\) over same states.

**Metrics:** `cos_grad`, `relative_norm_error`

**Cost control:** `n_states=200`, label depth/shallow per `mario/label.py` defaults

---

### Exp 08 — `oracle_advantage_ac`

**Hypothesis:** Actor–critic with \(\hat{A}\) from `label_state` is the sample-efficiency ceiling.

**Procedure:** Every `label_every=4` chunks during rollout, refresh \(\hat{A}(s,\cdot)\) from teacher.
Actor update uses oracle advantage; critic optional or identity.

**Metrics:** episodes to `completion_frac >= 0.9` vs Exp 03

**Interpretation:** separates "PG is slow" from "credit assignment is hard"

---

## Phase 4 — Comparisons and scale

### Exp 09 — `pg_vs_imitation`

**Hypothesis:** At fixed wall-clock, imitation (BC/DAgger) beats raw PG; oracle AC is competitive.

**Arms (same 30 min budget on 1-1):**
- BC retrain from existing shards (`mario/train.py`)
- DAgger one round (`scripts/run_dagger.py`)
- Exp 03 actor–critic
- Exp 08 oracle AC

**Metrics:** `completion_frac@30seeds`, `best_x_pos`, wall-clock seconds

---

### Exp 10 — `value_greedy_chatter`

**Hypothesis:** \(\arg\max_a \hat{Q}(s,a)\) from a moving critic flips actions more than PG.

**Procedure:**
1. Train per-action Q-head (or use `per_action_value` from rolling critic)
2. Each training step ε: measure fraction of states where \(\arg\max\) changes
3. Same for PG: measure \(\Delta\) KL between \(\pi_{\theta}\) and \(\pi_{\theta+\epsilon}\)

**Metrics:** `action_flip_rate` vs `kl_policy_change`

**Validates:** paper's motivation vs value-based control

---

### Exp 11 — `multi_level_interference`

**Hypothesis:** One net trained with PG on 11 levels shows per-level oscillation (V4 parallel).

**Levels:** manifest levels from `data/manifest.json` (cf8 subset)

**Procedure:** Multi-task PG or independent per-level PG (control arm)

**Metrics:** per-level `completion_frac` heatmap over training time

**Link:** `V4_FINDINGS.md` §2d DAgger oscillation (1-1 35→77→35)

---

## Shared hyperparameters

| Param | Default | Notes |
|---|---|---|
| `world, stage` | 1, 1 | |
| `chunk_frames` | 8 | match production; Exp 05b uses 4 near pit |
| `gamma` | 0.99 | per-chunk discount |
| `seeds_eval` | 30 | eval harness |
| `max_chunks` | 400 | ~3200 frames |
| `reward` | `mario.reward.DEFAULT` | progress + flag − death; time=0 |
| `device` | cpu for PG (repro) | mps optional for critic |

---

## Analysis outputs (all experiments)

See [artifacts/SCHEMA.md](artifacts/SCHEMA.md). Minimum per run:

- `config.json` — frozen hyperparams
- `episodes.csv` — one row per episode
- `checkpoints/` — optional `final.pt`
- `summary.json` — aggregated metrics for compare script

Phase 2+ also: `grad_alignment.json`, `advantages.npz`
