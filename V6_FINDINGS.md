# V6 Findings — Learned guidance for exact emulator search (2026-06-14)

> Reconciled on 2026-07-25. The durable conclusion is that learned proposals can
> help exact search, while the current positive evidence is **one local SMB1 1-1
> comparison**. This is neural-guided emulator search, not AlphaZero: there is no
> iterative self-play/tree-visit improvement loop in the measured experiment.

## 1. The reframe

V5 treated a standalone generalist controller as the main learned deliverable.
V6 instead asked a narrower question:

> Can a policy trained from search reduce exact emulator-search work without
> lowering solve rate?

That framing fits the repository's strongest asset: deterministic snapshot/restore.
The policy proposes or biases actions; emulator expansion and independent replay
remain the decision and verification mechanisms.

This does **not** imply that search universally beats RL, that imitation must fail,
or that learned world models are a dead end in every future configuration. It means
that learning dynamics is currently lower priority because the local emulator is
exact and callable. Model learning may become relevant for abstraction, transfer,
or throughput only after those become measured bottlenecks.

Relevant literature includes Mario A* controllers, Expert Iteration,
policy-guided heuristic search, DAgger, modern distillation analyses, and
reverse-curriculum work. AlphaZero is an architectural analogy only when an actual
iterative policy/value-guided tree-search loop is implemented and evaluated.

## 2. Phase 1 — specialist imitation experiment

`scripts/exit_specialist.py` explored dense on-route labels, deeper-lookahead
teacher scores, hazard-weighted focal loss, and DAgger-style off-route corrections.
The work found useful implementation problems:

1. A batch larger than the 164-row dataset produced only one update per epoch.
2. One-ply labels over-selected NOOP because coasting looked locally safe.
3. Warm-started, heavily weighted correction data destabilized the on-route policy;
   from-scratch/count-balanced retraining was then tried.

The June notes recorded round-0 completion fractions around 0.45–0.48, one lower
seed, and a +391-correction round at 0.22. Those DAgger numbers have **no preserved
structured evaluation report**, and the ignored checkpoint/training rows do not
have a clean-clone training-provenance chain. They are therefore retained as
historical debugging observations, not a reproduced scientific result.

The correct conclusion is limited:

- the tested thin-data, entity-policy recipes did not yield reliable standalone
  1-1 control;
- deeper labels and data balancing did not visibly solve that local failure in the
  recorded runs;
- nothing here proves a general 0.5 imitation ceiling or shows that covariate shift
  cannot be addressed by other observations, data, algorithms, or budgets.

## 3. Phase 2 — policy-guided search

`mario.search.beam_search` accepts an optional policy prior, and
`mario.entity_policy.EntityPolicyPrior` maps an emulator state to action scores.
The original experiment used top-k expansion plus a log-prior bonus.

The July audit reproduced and independently replay-verified the key local
comparison (`notes/artifacts/2026-07-25-policy-guided-1-1.json`):

| SMB1 1-1 configuration | Solved | Expanded nodes |
|---|---:|---:|
| plain beam, width 6 | yes | 7005 |
| policy-guided beam, width 6, top-3 | yes | **2770** |

That is **60.5% fewer expanded nodes (2.53 plain/guided ratio) with solve preservation in this one
configuration**. Historical wall times were machine/run dependent and are not
promoted as a general speedup claim. The checkpoint used for the rerun is local,
and its original training data/configuration cannot yet be regenerated from a
clean clone.

A tighter top-2 setting failed in the original experiment, illustrating the
central risk: hard pruning can discard the only successful action. The prior hook
is useful, but one level and one checkpoint do not establish cross-level
acceleration.

## 4. Temporal-context addendum

`TemporalEntityTransformer`, `TemporalEntityPolicyPrior`, and per-node observation
history were added as opt-in paths. The local 1-1 comparison recorded:

| Prior context | Expanded nodes | Reduction versus 7005 plain nodes |
|---|---:|---:|
| 1 frame | 2902 | 2.41× |
| 4 frames | 3274 | 2.14× |

These two temporal rows are historical console observations: no machine-readable
comparison report or clean training provenance was preserved for them. They are
not at the evidence level of the replay-backed 7005→2770 result above.

The four-frame model did not improve this benchmark. That is a negative for these
data/configurations, not evidence that temporal state is generally unhelpful. A
moving-platform level such as 6-2 is a more discriminating temporal-state test.

## 5. What to test next

The immediate research target is not a larger controller. It is a paired,
artifact-backed search benchmark:

1. Use at least six levels covering linear, athletic/moving-platform, castle, and
   deceptive-routing cases, with multiple starts or seeds where meaningful.
2. Compare plain search, a static action-frequency prior, a random prior, fixed
   top-k, a soft log-prior, uniform-mixture fallback, entropy-adaptive guidance,
   and a PHS/Levin-style completeness-safe alternative.
3. Fix node and wall-clock budgets; report solve rate, nodes, time, and uncertainty
   with paired bootstrap intervals.
4. Require solve-rate non-inferiority before celebrating node savings.
5. Persist config, source revision, seed, checkpoint/data hashes, and result JSON.

For SMB1 completion, keep 6-3 quarantined until a seed-0 replay succeeds. Treat 6-2
as a platform-phase/state-representation problem: expose relevant phase variables
and test variable-duration actions before assuming a stronger neural prior is the
answer.

## 6. LLM/VLM question

The June literature pass found no repository-specific evidence that text
pretraining improves low-level action choice over the structured entity model.
No local LLM/VLM control benchmark was run, so absolute claims about transfer,
latency, or model-size thresholds are not justified here.

A more plausible, falsifiable use is **offline subgoal proposal**: ask a model for
candidate landmarks once per unseen level, then let exact search accept or reject
them. This should be compared against hand-written, random, and search-tree-mined
subgoals under the same budget. Sprite/tile labeling is another offline tooling
use, separate from control.

## 7. Source and artifact inventory

- `scripts/exit_specialist.py` — specialist imitation/DAgger experiment driver.
- `scripts/policy_guided_search.py` — plain-versus-guided comparison with a
  machine-readable report path.
- `notes/artifacts/2026-07-25-policy-guided-1-1.json` — committed config,
  checkpoint hash, returned paths, and independent replay results for the
  7005→2770 comparison.
- `mario/entity_policy.py` — entity and temporal priors.
- `mario/search.py` — opt-in policy/value guidance for beam and coverage search.
- `data/specialists/1-1_prior.pt` and `1-1_rows.npz` — local ignored artifacts,
  useful for audit reruns but not clean-clone training provenance.

The next positive claim should cite a committed small report even when large
weights/data remain ignored. See `notes/research-bibliography.md` and
`notes/sessions/2026-07-25-project-audit-and-next-steps.md` for the ranked research
program and date-checked citations.
