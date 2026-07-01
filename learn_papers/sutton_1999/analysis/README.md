# Analysis scripts

Post-hoc tools — read only from `artifacts/runs/`, no emulator required.

| Script | Input | Output |
|---|---|---|
| `plot_learning_curves.py` | `episodes.csv` | PNG: return, x_max, entropy vs episode |
| `plot_grad_variance.py` | multiple `episodes.csv` | PNG: grad_var comparison (Exp 01–04) |
| `plot_visitation.py` | `visitation.npz` | PNG: heatmap over x_tile |
| `plot_grad_alignment.py` | `grad_alignment.json` | bar chart per architecture arm |
| `compare_experiments.py` | `summary.json` glob | CSV + markdown table |

## Usage

```bash
./venv/bin/python learn_papers/sutton_1999/analysis/plot_learning_curves.py \
  --run learn_papers/sutton_1999/artifacts/runs/exp01_reinforce_20260701-120000

./venv/bin/python learn_papers/sutton_1999/analysis/compare_experiments.py \
  --runs learn_papers/sutton_1999/artifacts/runs/exp0*/
```

## Dependencies

matplotlib (add to dev extras if missing). Scripts fail gracefully with install hint.

## Planned figures (for FINDINGS.md)

1. **Fig 1:** Learning curves — Exp 01 vs 02 vs 03 (return + grad_var)
2. **Fig 2:** Gradient alignment bars — Exp 05 arms
3. **Fig 3:** Wall-clock comparison — Exp 09
4. **Fig 4:** Chattering — Exp 10 flip rate over training
5. **Fig 5:** Multi-level heatmap — Exp 11
