# Paper ↔ codebase map (Sutton 1999)

Section-by-section guide: what the paper claims, where it lives in `mario_ai`, and what the
lab measures.

---

## §1 — Policy Gradient Theorem (Eq. 2)

**Claim:**
\[
\nabla_\theta \rho(\pi) = \sum_s d^\pi(s) \sum_a \frac{\partial\pi(s,a)}{\partial\theta}\, Q^\pi(s,a)
\]
No \(\nabla_\theta d^\pi(s)\) term — state visitation can be treated as constant when estimating
the gradient from on-policy samples.

### In Mario

| Symbol | Implementation |
|---|---|
| \(\pi_\theta(a\|s)\) | `MarioPolicy` logits → `softmax` (`mario/policy.py`) |
| \(\rho(\pi)\) | Episodic return: progress + flag − death (`mario/reward.py`) |
| \(Q^\pi(s,a)\) | Oracle: `label_state(...).per_action_value[a]` (`mario/label.py`) |
| \(d^\pi(s)\) | Empirical visitation over discretized cells `(area, x//16, y//16)` — logged in rollouts |
| Sample estimator | \(\frac{\partial\log\pi(a_t\|s_t)}{\partial\theta}\, G_t\) (REINFORCE) |

### Lab measurements

- **Exp 01:** Monte Carlo return \(G_t\) as \(Q\) substitute — does \(\rho\) improve?
- **Exp 07:** Oracle \(Q\) from `label_state` — compute true direction vs estimated direction
- **Analysis:** `visitation.npz` heatmaps over level-x tiles

### Key gotcha (this repo)

Actions are **chunks** (8 frames). \(G_t\) is summed over chunk rewards, not per-frame NES rewards.
Credit assignment is coarser than the paper's per-timestep MDP — document chunk index as \(t\).

---

## §2 — Function approximation + compatibility (Theorem 2, Eq. 4–5)

**Claim:** If critic \(f_w\) is at a local optimum of MSE to \(Q^\pi\) *and* satisfies
\[
\frac{\partial f_w(s,a)}{\partial w} \propto \frac{1}{\pi(s,a)}\frac{\partial\pi(s,a)}{\partial\theta}
\]
then \(\nabla_\theta \rho = \sum_s d^\pi(s)\sum_a \frac{\partial\pi}{\partial\theta} f_w(s,a)\).

### In Mario today

| Component | Compatible? | Location |
|---|---|---|
| `MarioPolicy` policy head | softmax over actions | `mario/policy.py` |
| `MarioPolicy` value head | **No** — shared trunk, scalar \(V(s)\), not per-action | same file |
| `ValueNet` | **No** — independent but \(V(s)\) only, BCE on survival | `mario/value.py` |

The production repo **already learned** that a shared value head perturbs BC (`train.py` comment).
Theorem 2 explains *why*: incompatible critics bias the policy gradient.

### Lab builds (`mario_pg/`)

| Module | Role |
|---|---|
| `CompatibleActorCritic` | Linear softmax policy + linear advantage critic on shared \(\phi_{sa}\) |
| `MLPActor` | Reuse `MarioPolicy` trunk, separate advantage head |
| `SharedTrunkActorCritic` | Deliberate anti-pattern for Exp 05 |

### Lab measurements

- **Exp 05:** \(\cos(\nabla_\theta \rho^{oracle}, \nabla_\theta \rho^{est})\) per architecture
- **Exp 06:** Policy iteration loop (Theorem 3) — critic fit then actor step

---

## §3 — Advantages and baselines (Eq. after §3)

**Claim:** \(f_w\) should approximate the **advantage** \(A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)\),
not raw \(Q\). Adding any \(v(s)\) to the critic doesn't change the expected gradient (baseline
invariance) but changes variance.

### In Mario

| Quantity | Source |
|---|---|
| \(Q^{teacher}(s,a)\) | `LabelResult.per_action_value` |
| \(V^{teacher}(s)\) | `max_a Q` or `LabelResult.value` |
| \(A^{teacher}(s,a)\) | `Q[s,a] - sum_b pi(b|s) Q[s,b]` (compute in lab) |
| Monte Carlo \(G_t\) | Sum of chunk rewards from rollout |
| Learned baseline | `ValueNet` or new `mario_pg/critic.py` |

### Lab measurements

- **Exp 02:** constant baseline \( \bar{G} \)
- **Exp 03:** learned \(V_\phi(s)\) from on-policy returns
- **Exp 04:** advantage vs raw return — plot gradient estimator variance
- **Exp 08:** oracle advantage from `label_state` every N chunks (upper bound)

---

## §4 — Policy iteration with FA (Theorem 3)

**Claim:** Alternate (1) fit \(f_w\) to \(Q^{\pi_k}\), (2) policy gradient step on \(\pi_k\) →
locally optimal \(\pi\) (under step-size conditions).

### Contrast with DAgger (`mario/dagger.py`)

| | Policy iteration (paper) | DAgger (repo) |
|---|---|---|
| Objective | Maximize \(\rho(\pi)\) | Minimize imitation loss on aggregated data |
| Expert use | Critic target only | Hard labels for BC |
| On-policy | Yes — roll out \(\pi_\theta\) | Yes — but supervised update |
| Convergence | Local optimum of \(\rho\) | No guarantee on return |

DAgger is the right **imitation** baseline for Exp 09; don't conflate it with Theorem 3.

### Lab measurements

- **Exp 06:** explicit `{critic_fit → actor_step}` loop
- **Exp 09:** wall-clock vs BC / DAgger on 1-1 completion rate

---

## Williams REINFORCE (episodic form)

**Update:** \(\Delta\theta \propto \frac{\partial\log\pi(a_t|s_t)}{\partial\theta} R_t\)

With baseline: replace \(R_t\) with \(R_t - b(s_t)\).

### Implementation notes for SMB

1. **Stochastic policy required** — `Categorical(logits).sample()`, not `argmax`
2. **Episode boundary** — one level attempt = one episode; death or flag = terminal
3. **Reward** — use `state_score` deltas or sparse `{+flag, -death, +Δx}` (match search teacher)
4. **Entropy** — monitor collapse; optional entropy bonus for exploration (not in original paper)

---

## Symbols cheat sheet

| Paper | Code |
|---|---|
| \(\theta\) | `net.policy` parameters (actor) |
| \(w\) | critic parameters |
| \(\pi(s,a)\) | `softmax(logits)[a]` |
| \(\rho(\pi)\) | `episode_return` in `episodes.csv` |
| \(d^\pi(s)\) | `visitation.npz` |
| \(\hat{A}\) | `advantages.npz` column `a_hat` |
| \(Q^{oracle}\) | `label_state` → `per_action_value` |
