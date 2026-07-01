# Findings — Sutton (1999) lab

_Status: template — fill in as experiments complete._

## Executive summary

<!-- 3–5 sentences: what we learned about policy gradients in SMB, oracle vs learned critic, vs imitation -->

---

## Results table

| Exp | Key metric | Result | Pass? | Run path |
|---|---|---|---|---|
| 01 reinforce_raw | `mean_return_last_100` | — | — | |
| 02 const_baseline | `grad_var` vs 01 | — | — | |
| 03 learned_baseline | episodes to 50% completion | — | — | |
| 04 advantage_vs_return | `grad_var` | — | — | |
| 05 compatibility | `cos_grad` compatible vs shared | — | — | |
| 06 policy_iteration | return monotonicity | — | — | |
| 07 oracle_q | `cos_grad` | — | — | |
| 08 oracle_ac | episodes to 90% | — | — | |
| 09 pg_vs_imitation | completion @ 30min | — | — | |
| 10 chattering | `action_flip_rate` | — | — | |
| 11 multi_level | interference pattern | — | — | |

---

## §1 Policy Gradient Theorem

**Operational understanding:**

<!-- Did cosine alignment with oracle Q validate Eq. 2? What broke with Monte Carlo G_t? -->

---

## §2 Compatibility

**Operational understanding:**

<!-- Numbers from Exp 05. Link to mario/value.py independence and train.py shared-head regression. -->

---

## §3 Advantages / baselines

**Operational understanding:**

<!-- grad_var curves 01→02→03→04 -->

---

## §4 Policy iteration

**Operational understanding:**

<!-- Exp 06 vs DAgger -->

---

## Implications at scale

### 1. Perfect simulator + search oracle

<!-- Exp 07–08: how much sample efficiency search buys PG -->

### 2. Chunked action space (cf=8)

<!-- Credit assignment; cf=4 ablation if run -->

### 3. Multi-level single-net θ

<!-- Exp 11 parallel to V4_FINDINGS DAgger oscillation -->

---

## PG vs imitation (this repo's stack)

| Method | 1-1 completion | Sample / wall-clock | Notes |
|---|---|---|---|
| BC | | | `mario/train.py` |
| DAgger | | | `mario/dagger.py` |
| REINFORCE + learned V | | | Exp 03 |
| Oracle actor–critic | | | Exp 08 |

---

## Follow-ups (out of v1 scope)

- PPO / clipped objective for stable fine-tune after search init (`DESIGN.md` §10)
- Per-level θ with shared φ features (mixture of experts)
- Frame-level PG near hazards (cf=1 segment)
- Stochastic policy at pipe entries (2-2 / 8-4 class)

---

## Reproduce main figures

```bash
# After runs exist:
./venv/bin/python learn_papers/sutton_1999/analysis/compare_experiments.py \
  --runs artifacts/runs/exp01_* artifacts/runs/exp02_* ...
```
