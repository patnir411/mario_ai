# Architecture — Sutton 1999 lab code

New code lives under `learn_papers/sutton_1999/mario_pg/` to avoid polluting production
`mario/` until patterns stabilize. Production imports are explicit and read-only.

```
learn_papers/sutton_1999/
  mario_pg/
    __init__.py
    config.py          # YAML → dataclass
    rollout.py         # on-policy trajectories
    reward.py          # chunk-level reward from state_score deltas
    reinforce.py       # REINFORCE + baselines
    baseline.py        # constant / learned V
    actor_critic.py    # compatible + MLP variants
    features.py        # φ(s,a) construction
    critic.py          # advantage critic heads
    oracle.py          # label_state wrappers, batch Q/A
    grad_align.py      # cos(∇_true, ∇_est)
    logging.py         # episodes.csv, checkpoints
    policy.py          # thin wrappers around MarioPolicy / linear policy
  scripts/
    run_experiment.py  # generic runner: load config → train → eval → summary
    run_phase.py       # run all configs in a phase
    00_sanity_rollout.py
    train_reinforce.py
    train_actor_critic.py
    measure_grad_alignment.py
    oracle_gradient_check.py
    benchmark_pg_vs_imitation.py
    value_greedy_chatter.py
  analysis/
    plot_learning_curves.py
    plot_grad_variance.py
    plot_visitation.py
    compare_experiments.py
  tests/
    test_rollout.py
    test_reinforce.py
    test_schema.py
```

---

## Dependency on production `mario/`

| Production module | Lab usage |
|---|---|
| `mario.env.MarioSim` | rollouts, snapshots |
| `mario.observation.observe` | φ(s) base features |
| `mario.policy.MarioPolicy` | optional actor backbone (Exp 05 arm) |
| `mario.reward.state_score, DEFAULT` | reward shaping (match teacher) |
| `mario.label.label_state, fast_label` | oracle Q, advantages (Exp 07–08) |
| `mario.eval.rollout, eval_suite` | final evaluation contract |
| `mario.buffer.DatasetIndex` | Exp 09 BC baseline data |

**Do not modify** `mario/train.py` loss for PG — keep imitation path untouched.

---

## Core types

```python
# mario_pg/config.py
@dataclass
class ExperimentConfig:
    exp_id: str
    world: int = 1
    stage: int = 1
    chunk_frames: int = 8
    episodes: int = 2000
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    baseline: Literal["none", "constant", "learned", "oracle"] = "none"
    policy: Literal["mlp", "linear_compatible"] = "mlp"
    K: int = 4
    seed: int = 0
    eval_seeds: list[int] = field(default_factory=lambda: list(range(30)))
```

---

## Reward: chunk-level MDP

The paper assumes per-step reward. We define a **chunk MDP**:

\[
r_t = \text{state\_score}(info_{t+1}) - \text{state\_score}(info_t)
\]

Terminal: `is_success` → episode ends with flag bonus already in score; `is_death` → large negative.

This aligns PG with the search teacher's objective (`mario/reward.py`).

```python
# mario_pg/reward.py
def chunk_reward(prev_info, next_info, *, died: bool, w=DEFAULT) -> float:
    prev_s = state_score(prev_info, x_start=..., ...)
    next_s = state_score(next_info, x_start=..., ...)
    return next_s - prev_s
```

---

## Policy parameterizations

### 1. `LinearCompatiblePolicy` (Exp 05, 06 — paper-faithful)

```python
class LinearCompatiblePolicy(nn.Module):
    """π(a|s) = softmax(W @ φ_sa).  Critic linear in φ_sa - E_π[φ]."""
    def forward(self, obs): -> logits, advantage_hat  # [B,A], [B,A]
```

### 2. `MLPActor` (Exp 01–04 — practical)

Reuse `MarioPolicy` but train with PG loss, not CE. Inference during training: **sample**;
eval: argmax (report both in `summary.json`).

### 3. `SharedTrunkActorCritic` (Exp 05 negative control)

`MarioPolicy` forward → policy logits + scalar value; critic advantage = `q_a - v` derived
incorrectly from shared trunk — reproduces repo regression pattern.

---

## Training loops

### REINFORCE (`mario_pg/reinforce.py`)

```python
for ep in range(episodes):
    traj = rollout_episode(net, sim, stochastic=True)
    returns = compute_returns(traj, gamma)
    adv = baseline.subtract(returns, traj) if baseline else returns
    loss = -(log_probs * adv).sum() - entropy_coef * entropy
    loss.backward(); optimizer.step()
    logger.log_episode(ep, traj, adv, loss)
    if ep % eval_every == 0:
        metrics = eval_suite(net, seeds=eval_seeds)
```

### Actor–critic (`mario_pg/actor_critic.py`)

```python
for ep in range(episodes):
    traj = rollout_episode(...)
  # Critic update (per step or per episode)
    critic_loss = mse(fw(s,a), target_A)  # or TD
    # Actor update — stopgrad on advantage
    actor_loss = -(log_prob * fw(s,a).detach()).sum()
```

### Policy iteration (`Exp 06`)

Outer loop `k in 1..K_rounds`:
1. Collect D_k with π_θ
2. Fit f_w on D_k until val loss plateaus (inner loop)
3. One actor epoch on D_k using f_w as advantage

---

## Oracle module (`mario_pg/oracle.py`)

```python
def oracle_q(sim, prefix: list[int], seed: int) -> np.ndarray:
    """Returns per_action_value[N_ACTIONS] from label_state."""
    res = fast_label(world, stage, prefix, seed=seed)
    return res.per_action_value

def oracle_advantage(q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    v = (pi * q).sum()
    return q - v

def policy_gradient_oracle(q, log_probs) -> np.ndarray:
    """Σ_a dlogπ(a) * Q(s,a) — reference direction for grad_align."""
```

**Caching:** key = `hash(tuple(prefix))` → save in `artifacts/cache/labels.npz` to reuse across arms.

---

## Gradient alignment (`mario_pg/grad_align.py`)

```python
@dataclass
class GradAlignResult:
    cos_sim: float
    norm_true: float
    norm_est: float

def align(net, states, oracle_q_fn) -> GradAlignResult:
    g_true = ...
    g_est = ...
    return GradAlignResult(cos_sim=dot(g_true,g_est)/(norm*norm), ...)
```

---

## Logging (`mario_pg/logging.py`)

```python
class RunLogger:
    def __init__(self, run_dir: Path): ...
    def log_episode(self, ep, metrics: dict): ...  # append episodes.csv
    def log_step(self, ep, step, metrics: dict): ...  # optional steps.csv
    def save_checkpoint(self, net, name: str): ...
    def write_summary(self, summary: dict): ...  # summary.json at end
```

Run directory: `artifacts/runs/{exp_id}_{utc}/`

---

## Config → runner

```python
# scripts/run_experiment.py
def main():
    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg.exp_id)
    set_seed(cfg.seed)
    net = build_policy(cfg)
    baseline = build_baseline(cfg)
    train_fn = DISPATCH[cfg.trainer]  # reinforce | actor_critic | oracle_ac
    train_fn(net, cfg, run_dir)
    eval_and_summarize(net, cfg, run_dir)
```

---

## Evaluation contract

Match production where possible:

```python
from mario.eval import rollout, PASS_THRESHOLD
# PG-specific additions in summary.json:
{
  "completion_frac": 0.0,
  "mean_return_last_100": 0.0,
  "grad_var_last_100": 0.0,
  "policy_entropy_last_100": 0.0,
  "label_state_calls": 0,      # oracle exps only
  "wall_clock_sec": 0.0,
}
```

Pass threshold for 1-1 specialist: **0.90** completion @ 30 seeds (same as `mario/eval.py`).

---

## What we explicitly do NOT build (v1)

- PPO / TRPO (out of scope; mention in FINDINGS as follow-up)
- Pixel observations
- Multi-stage any% PG
- GPU-distributed rollouts

These are listed in `FINDINGS.md` §Follow-ups.
