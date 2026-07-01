# Artifact schema

All experiment outputs go under:

```
learn_papers/sutton_1999/artifacts/runs/<exp_id>_<YYYYMMDD-HHMMSS>/
```

This directory is **gitignored** except bundled snapshots in `artifacts/bundled/` (optional,
for paper figures).

---

## Required files (every run)

### `config.json`

Frozen copy of the YAML config at run start.

```json
{
  "exp_id": "exp01_reinforce",
  "world": 1,
  "stage": 1,
  "chunk_frames": 8,
  "episodes": 2000,
  "gamma": 0.99,
  "lr_actor": 0.0003,
  "baseline": "none",
  "policy": "mlp",
  "K": 4,
  "seed": 0,
  "git_rev": "abc1234"
}
```

### `episodes.csv`

One row per training episode.

| Column | Type | Description |
|---|---|---|
| `episode` | int | 0-indexed |
| `return` | float | undiscounted sum of chunk rewards |
| `return_disc` | float | discounted return |
| `length` | int | chunks until done |
| `beat` | bool | reached flag |
| `died` | bool | death without flag |
| `x_max` | int | max x_pos |
| `completion_frac` | float | x_max / level_length |
| `entropy_mean` | float | mean policy entropy over episode |
| `grad_norm` | float | ‖∇θ‖ after update |
| `grad_var` | float | variance of per-step gradient contributions |
| `loss` | float | training loss |
| `wall_sec` | float | cumulative wall time |

### `summary.json`

Written at end of run.

```json
{
  "exp_id": "exp01_reinforce",
  "episodes_total": 2000,
  "completion_frac_eval": 0.0,
  "mean_return_last_100": 12.5,
  "grad_var_last_100": 0.042,
  "policy_entropy_last_100": 1.8,
  "best_x_max_eval": 450,
  "label_state_calls": 0,
  "wall_clock_sec": 3600,
  "pass": false
}
```

### `eval.json`

Same schema as production `mario/eval.py` output (per-level outcomes, seeds).

---

## Optional files

### `steps.csv`

Per-chunk detail (enable with `log_steps: true` in config). Large — use for debugging only.

| Column | Type |
|---|---|
| `episode`, `step` | int |
| `action` | int |
| `reward` | float |
| `log_prob` | float |
| `advantage` | float |
| `x_pos`, `y_pos` | int |
| `value_baseline` | float |

### `advantages.npz`

```python
{
  "prefix_hash": int[],      # identifier
  "q_oracle": float[N, A],
  "pi": float[N, A],
  "a_oracle": float[N, A],
  "a_hat": float[N, A],
  "g_true": float[N, theta_dim],  # optional, large
  "g_est": float[N, theta_dim],
}
```

### `grad_alignment.json` (Exp 05, 07)

```json
{
  "arm": "compatible_linear",
  "n_states": 200,
  "cos_sim_mean": 0.91,
  "cos_sim_std": 0.08,
  "norm_ratio_mean": 1.05
}
```

### `visitation.npz`

```python
{
  "cells": int[N, 3],   # (area, x_tile, y_tile)
  "counts": int[N],
}
```

### `checkpoints/`

- `latest.pt` — every `save_every` episodes
- `final.pt` — end of training

Checkpoint format:

```python
{
  "state_dict": ...,
  "config": {...},
  "episode": int,
}
```

### `cache/labels.npz`

Oracle label cache keyed by `prefix_hash` to avoid redundant `label_state` calls.

---

## Validation

```bash
./venv/bin/python -m pytest learn_papers/sutton_1999/tests/test_schema.py
```

`test_schema.py` checks a synthetic run directory against this contract.

---

## Compare script input

`analysis/compare_experiments.py` reads `summary.json` from multiple run dirs and emits
`artifacts/comparisons/<name>.csv`:

| exp_id | completion_frac_eval | grad_var_last_100 | wall_clock_sec | pass |
