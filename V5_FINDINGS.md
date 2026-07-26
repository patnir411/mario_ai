# V5 Findings — Generalist-policy experiment and reverse-curriculum hypothesis

> Historical consolidation of the 2026-06-13/14 V5 effort, reconciled to the
> 2026-07-25 repository audit. Builds on `V4_FINDINGS.md`. The retained result is a
> useful negative for the tested data, model, and training recipes—not a universal
> limitation of imitation learning.

## 1. What we set out to test

V4 identified standalone generalist control as the bottleneck. V5 tested whether
greater level diversity would let one entity-transformer transfer to held-out stock
levels, using a fixed 20-train/6-holdout split. The motivating data-scaling result
came from robotic manipulation; V5 was an experiment to see whether its qualitative
lesson transferred, not a justified Mario scaling law.

## 2. What was built

- `scripts/solve_all_stock.py` — batch search plus deterministic replay gating. The
  July 25 audit establishes **30/32**, not the original 31/32 headline: 6-2 is
  unsolved and 6-3 is quarantined after its seed-0 replay failed.
- `mario/entity.py` and `mario/entity_policy.py` — object-centric observations and
  entity/temporal policy models with optional value heads and search-prior adapters.
- `mario/consistency.py` — an input-perturbation/output-consistency penalty. The
  original `stable_bc.py` name was incorrect: this does not construct or constrain
  the closed-loop dynamics Jacobian required by Stable-BC.
- `mario.label.label_at_state` — a shared snapshot Q-sweep primitive with exact
  emulator and wrapper-cache restoration.
- `scripts/gen_entity_dataset.py`, `scripts/train_generalist.py`, and
  `scripts/eval_heldout.py` — replay-gated data generation, held-out-integrity
  checks, training, and no-rescue evaluation.
- `scripts/dagger_generalist_entity.py` — DAgger/RoaD-style data aggregation.
- `scripts/rc_rl.py` — a reverse-curriculum Double-DQN prototype using resets along
  an oracle route.
- `configs/heldout_split.json` and focused integrity/consistency tests.

The source paths above are part of the commit candidate. Large datasets and
checkpoints remain intentionally ignored.

## 3. Historical observations and evidence limits

The June experiment log recorded:

| Stage | Held-out (6 levels) | Train sample | Historical observation |
|---|---:|---:|---|
| entity-transformer BC | 0/24 | 0/24 | validation accuracy 0.639; early death/stall |
| DAgger variants | 0/24 | 4/24 | validation accuracy about 0.68–0.69 |
| reverse-curriculum RL on 1-1 | — | partial suffixes | frontier reportedly moved backward, then stalled |

These are valuable development observations, but they do **not** currently have a
complete clean-clone provenance chain. The generated shards, checkpoints, detailed
DAgger reports, and reverse-curriculum report were not preserved as commit-sized
artifacts. Accordingly:

1. The tested BC/DAgger configurations did not produce held-out completions.
2. The logged validation-accuracy stability is not itself control success.
3. The reverse-curriculum behavior is a hypothesis-generating observation, not a
   reproduced result or proof that the curriculum mechanism works.
4. No result here establishes an imitation-learning ceiling beyond these local
   recipes and observations.

Future learned claims must record configuration, seed, code revision, split,
dataset/checkpoint hashes, rollout outcomes, and an independently replayable report.

## 4. What the literature actually supports

**Scaling analogy, not Mario sample complexity.** Procgen demonstrates that broad
procedural diversity can matter and that hundreds of training levels may still
overfit in its benchmark. Its reported level counts do not imply that Mario needs
exactly \(10^3\)–\(10^4\) levels, nor that 26 levels is a mathematically meaningful
"400× deficit." The defensible inference is simply that a small, heterogeneous
stock-level set is weak evidence for zero-shot generalization.

**Long-horizon imitation remains conditional.** Classical BC/DAgger analyses explain
compounding error and learner-distribution correction under stated assumptions.
They do not imply that DAgger has no generalization mechanism in every setting, that
optimal demonstrations are always least generalizable, or that reverse-curriculum
RL must dominate imitation here.

**Exact simulator access changes the design priority.** The emulator makes online
search labels, counterfactual rollouts, and local resets cheap relative to many
robotics settings. That supports testing search guidance, selective expert queries,
and reverse curricula. It does not by itself choose one universally superior
learning algorithm.

## 5. Reverse-curriculum prototype

`scripts/rc_rl.py` resets to snapshots along a known solution, begins near the flag,
and attempts to move the start boundary backward. Its learner combines Double-DQN,
an ensemble, demo/online replay, target-network updates, and self-imitation.

The June session attributed frontier stalls to plasticity/primacy effects and tried
several mitigations. Because no structured evaluation report was preserved, the
next honest step is not to add another remedy; it is to make one seeded run fully
reproducible, emit per-frontier success curves and checkpoint/config hashes, and
compare against BC and search under a declared budget. Periodic resets, n-step
returns, PPO, or a maintained RL implementation become justified only after that
baseline identifies the failure.

## 6. Key references and how they are used

- *Data Scaling Laws in Imitation Learning* (ICLR 2025) — motivating manipulation
  result; not a Mario-level-count law.
- Cobbe et al., *Procgen* (2020) — caution about procedural train/test overfitting;
  benchmark analogy only.
- Ross, Gordon, and Bagnell, *DAgger* (AISTATS 2011) — learner-distribution data
  aggregation.
- Foster et al., *Is Behavior Cloning All You Need?* (NeurIPS 2024) — conditional
  horizon/error analysis.
- Go-Explore, RFCL, and reverse-curriculum work — candidate uses of saved-state
  access, not proof that the local prototype succeeds.
- RLPD and BBF — possible stable/offline-to-online RL components.
- Mehta et al., *Stable-BC* (2024) — the closed-loop stability method that the
  local input-consistency surrogate must not be confused with.

See `notes/research-bibliography.md` for date-checked links and annotations.

## 7. Recommendation

1. Keep exact search plus replay-gated solutions as the primary deliverable. It
   currently verifies **30/32**, not every stock level and not route optimality.
2. Treat learned policies first as search priors or selective proposal mechanisms.
   The positive V6 result is local and must be broadened without lowering solve rate.
3. If standalone control remains scientifically important, reproduce one
   reverse-curriculum baseline end to end before expanding the architecture.
4. Treat PCG/generalist control as a separate research program. Procgen and related
   work motivate diversity and careful evaluation, not a predetermined generator
   size or an expected zero-shot outcome.
