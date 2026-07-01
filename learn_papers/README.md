# learn_papers/

Hands-on labs that connect classic RL papers to the Mario AI codebase. Each subfolder is a
self-contained study track: paper mapping, experiments, runnable scripts, and analyzable artifacts.

| Paper | Folder | Status |
|---|---|---|
| Sutton et al. (1999) — Policy Gradient Methods for RL with Function Approximation | [`sutton_1999/`](sutton_1999/) | **planned** — see [README](sutton_1999/README.md) |

## Design principles

1. **Ground truth from the simulator.** We own a perfect forward model and a search teacher — use
   them to measure oracle quantities (Q, advantages, gradient alignment) that textbook RL can't.
2. **Artifacts over vibes.** Every experiment writes CSV/JSON/NPZ under its run directory; analysis
   scripts are reproducible from those files alone.
3. **Minimal scope per experiment.** One hypothesis, one level (usually 1-1), one chart. Scale comes
   from running the matrix, not from one mega-script.
4. **Reuse production modules.** `mario/env.py`, `mario/label.py`, `mario/policy.py`, etc. — the
   lab adds PG-specific training and analysis, not a parallel stack.

## Quick start

```bash
cd /path/to/mario_ai
./venv/bin/python scripts/update_status.py   # optional: refresh STATUS block

# Sutton 1999 lab hub
cat learn_papers/sutton_1999/README.md
```
