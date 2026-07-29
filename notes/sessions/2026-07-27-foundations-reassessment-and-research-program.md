# 2026-07-27 — Foundations reassessment and falsifiable research program

> This note is the durable result of a whole-project, first-principles review.
> The requested literature cutoff is **2026-07-25**; repository inspection and
> verification continued on July 27 in America/Detroit. Observations and claim
> boundaries come before the plan. The review began from clean commit
> `43f55b6`; the clean normal/reverse SMA4 reports were already present, while a
> rolling local ledger reported 171 passing tests before this pass. After the
> observations were recorded, this pass also completed the bounded Gate-0
> novelty, pipe-macro accounting, and provenance-label fixes described in
> §3.6; it did not alter the SMA4 route. The final local working-overlay checks
> are source-bound in
> `notes/artifacts/2026-07-27-foundations-gate0-verification.json`.
>
> **Supersession note (2026-07-28):** the evidence ledger, mathematical
> cautions, and completed Gate-0A work remain historical facts. The serialized
> roadmap and “one-root SMA4 next” priority in §§7–8 are superseded by
> `notes/sessions/2026-07-28-strategic-reassessment-and-option-contract.md`,
> which adds explicit controller phase and splits portable science from the
> bounded SMA4 integrity case.

## 0. Review question and standard

The question is not merely what to implement next. It is:

> What mathematical object does this repository actually manipulate, which
> claims have been earned by replayable evidence, which hidden assumptions can
> still fabricate success, and what sequence of falsifiable experiments could
> turn the current systems work into a defensible research contribution?

The governing standard is:

1. distinguish exact emulator facts from decoded observations, abstractions,
   interventions, and prose;
2. distinguish a controller-realizable game trajectory from a counterfactual
   composition assembled with restores or writes;
3. state the access model, option alphabet, cost, horizon, goal, and permitted
   interventions with every planning or abstraction claim;
4. preserve negative and contradictory evidence;
5. let exact independent replay reject a result even when every summary metric
   looks favorable; and
6. prefer a smaller claim with a reproducible certificate to a larger claim
   built from untested state sufficiency.

## 1. Evidence ledger before interpretation

### 1.1 Reliable substrate

The strongest layer is deterministic, resettable forward execution:

\[
x_{t+1}=F(x_t,a_t),
\]

where \(x_t\) is the complete emulator plus wrapper state and \(a_t\) is a legal
controller input. For a fixed root, ROM, core, wrapper, and action vocabulary,
this is naturally a deterministic weighted labelled transition system and a
shortest-path problem.

Artifact-backed current status:

- SMB1 any% replays 8/8 from the preserved route.
- The stock SMB1 set is 30/32 at seed 0. `6-2` is unsolved and `6-3` remains
  quarantined after replay failure.
- At the reviewed commit, the rolling local ledger recorded 171 passes and ten
  expected skips. The bounded working overlay from this pass records 177 passes
  and the same ten skips in the source-bound Gate-0 attestation.
- The narrow learned-prior result on SMB1 1-1 reproduces 7,005 plain versus
  2,770 guided expansions and both paths replay. Its training provenance is not
  yet clean-clone complete.

These facts support exact-search and replay infrastructure. They do not by
themselves support a general learning, optimality, or cross-game claim.

### 1.2 SMA4 result at commit `43f55b6`

The two clean reports are:

```text
runs/20260727-sma4-live-cursor-tier2-final/report.json
runs/20260727-sma4-live-cursor-tier2-reverse-final/report.json
```

They attest clean source at commit `43f55b6`. In the encountered option table:

- BFS finds the mixed goal at 13,266 reported frames;
- uniform-cost search finds it at 13,194 reported frames;
- 6,000 of either total is the symbolic Bowser edge;
- the reported ROM-option prefixes are therefore 7,266 and 7,194 frames;
- there are 13 physical records, seven option identifiers, and 91 classified
  record-option rows;
- only 14 rows are enabled;
- each enabled row is executed twice, for 28 executions;
- no repeat-conformance failure is reported;
- finite partition refinement returns 11 blocks;
- the two post-second-whistle records merge under the recorded signature;
- the two World-8 records merge only within a table whose outgoing Bowser edge
  is symbolic;
- the two post-first-whistle records remain split because
  `use_whistle_again` takes 1,302 versus 1,290 reported frames; and
- both acquisition orders reach a pointer-resolved, controller-responsive
  World-8 map with no cursor writes and maximum effective Tier 2.

The live-cursor correction is a real result. Pointer `0x03007824` selects a
cursor object at either `0x03003DE0` or relocated `0x03004EF8`. A matched
`DOWN`/NOOP test shows that both low-level storage realizations expose the same
logical controller response. The earlier history asymmetry was an observer
error, not a control failure.

### 1.3 The decisive remaining honesty debt

The current acquisitions are executed from independent external powered roots.
The second acquisition copies prior whistle inventory into its root, and the
fortress segment reholds leaf power after damage. The final Bowser transition is
symbolic.

Consequently the current route is:

> a repeatable counterfactual composition of useful physical segments,

not:

> one controller-realizable physical trajectory through an Option-SMDP.

This distinction is not cosmetic. External restoration bypasses the option
composition condition that the realized termination state of one option be in
the initiation set of the next.

## 2. Mathematical foundations

### 2.1 Exact fixed-root planning

For root \(x_0\), goal set \(G\), legal controller inputs \(A(x)\), and
nonnegative primitive cost \(c(x,a)\), the immediate problem is:

\[
\min_{a_{0:T-1}} \sum_{t=0}^{T-1} c(x_t,a_t)
\quad\text{subject to}\quad
x_{t+1}=F(x_t,a_t),\;x_T\in G.
\]

When one frame costs one unit, the retained route cost is \(T\). A fixed
multi-frame hold is just an edge in an induced graph:

\[
F_d(x,a)=F^d(x,a),\qquad c(x,a,d)=d.
\]

Using only \(d=8\) changes the reachable graph; it is not merely a faster way to
search the primitive graph. A duration-one fallback is needed if primitive
representability is part of the claim.

### 2.2 Genuine options and the deterministic SMDP boundary

An option is \(o=(I_o,\pi_o,\beta_o)\): an initiation set, an intra-option
policy, and a termination condition. Starting at \(x\), it induces a joint
distribution over terminal state \(x'\), duration \(K\), and accumulated cost
\(C\):

\[
\Pr(C,K,X'=x'\mid H_t,o)
=
\Pr(C,K,X'=x'\mid X_t=x,o).
\]

In the present deterministic emulator, a fully specified option from an exact
state normally induces a point mass. Let
\(c_o(x)=\mathbb E[C\mid x,o]\). The shortest-path Bellman equation is
therefore more natural than stochastic language:

\[
J(x)=
\begin{cases}
0, & x\in G,\\
\min_{o\in O(x)}\left[c_o(x)+J(F_o(x))\right], & x\notin G.
\end{cases}
\]

If stochastic outcomes are later admitted, use the stochastic-shortest-path
form:

\[
J(x)=
\begin{cases}
0, & x\in G,\\
\min_{o\in O(x)}
\left[c_o(x)+\sum_{x',k}P_o(x',k\mid x)J(x')\right],
& x\notin G,
\end{cases}
\]

with declared terminal states, nonnegative retained costs, and proper-policy
conditions.

A cached action trace may be a valid singleton-root replay macro. It becomes a
broad option only after its policy succeeds across a declared initiation set.
A procedure that restores an unrelated root or writes missing resources is an
experimental intervention, not a controller policy in the same underlying
game trajectory.

### 2.3 Exact abstraction, controlled lumpability, and homomorphism

Let \(\phi:X\to Z\) be a proposed option-boundary abstraction. If
\(\phi(x)=\phi(y)\), exact deterministic option abstraction requires agreement,
for every modeled option, on:

\[
\begin{aligned}
&\mathbf 1[x\in I_o]=\mathbf 1[y\in I_o],\\
&\text{success/termination}_o(x)
 =\text{success/termination}_o(y),\\
&K_o(x)=K_o(y),\qquad C_o(x)=C_o(y),\\
&\phi(F_o(x))=\phi(F_o(y)),\\
&g(x)=g(y).
\end{aligned}
\]

For a stochastic option process, equality is required for the joint pushforward
law

\[
\Pr(C,K,\phi(X')=z'\mid x,o)
=
\Pr(C,K,\phi(X')=z'\mid y,o)
\]

for every abstract successor \(z'\), not merely for separate marginal means or
distributions. A state-action SMDP homomorphism may also include a
state-dependent option recoding; the current code uses the identity option-name
map.

The condition is recursive. Equal immediate labels do not suffice: successor
states must themselves remain equivalent. This is the bisimulation/partition
refinement idea behind the current finite table.

### 2.4 What the current 11 blocks do and do not mean

`refine_option_partitions` begins with exact `MetaState` plus a goal label and
only splits that starting partition. It does not merge records with different
`MetaState`s. Therefore:

- the result is a stable refinement of the encountered, deterministic,
  option-labelled finite table;
- it is useful as a conformance report and counterexample generator;
- it is not the coarsest quotient induced by the current option alphabet;
- it is not a proof that `MetaState` is Markov-sufficient;
- it is not primitive-controller bisimulation;
- it is not global ROM equivalence;
- it is not stochastic lumpability; and
- it is not yet used to merge the live planner frontier, which remains keyed by
  `(MetaState, physical_record_id)`.

The strongest exact statement is:

> Within a table of 13 encountered physical records and seven option
> identifiers, all 91 rows are classified, 14 enabled rows repeat twice without
> an observed mismatch, and refinement of the pre-existing `MetaState` coloring
> is stable at 11 blocks under the recorded outcome, cost, successor, and
> intervention signatures.

That is an empirical finite conformance claim, not an SMDP abstraction theorem.

### 2.5 There is no single correct abstraction

Abstraction is objective-relative. The project should maintain an explicit
refinement hierarchy. The following are three nested points in the broader
partition lattice ordered from coarser to finer:

1. **Reachability quotient:** preserve enabledness, terminal labels, and
   abstract successors. Records 7 and 8 may merge here.
2. **Exact retained-time quotient:** additionally preserve retained duration
   and route cost. Records 7 and 8 must split by at least 12 frames.
3. **Intervention/provenance quotient:** additionally preserve external roots,
   writes, symbolic edges, and lineage. States with identical controller
   dynamics may still split here.

Every abstraction claim must name the option alphabet, objective, cost
semantics, horizon, access model, goal labels, and allowed interventions.

### 2.6 Predictive state and finite distinguishing tests

Two histories are predictively equivalent only when no controlled future test
distinguishes them:

\[
h\sim h'
\iff
\Pr(\tau\mid h)=\Pr(\tau\mid h')
\quad\text{for every action/observation test }\tau.
\]

The repository's option suffixes are finite predictive tests. They are strong
falsifiers: one disagreement disproves a proposed merge. Agreement on a finite
suite cannot prove equality for every suffix without an explicit fault-domain
assumption.

A concrete next representation experiment is a finite predictive table whose
columns pair a controlled suffix \(\tau\) with a terminal observation event
\(E\):

\[
H_{h,(\tau,E)}
=
\Pr(E\mid h,do(\tau)).
\]

Treat exact duration/cost/output values as part of the finite observation
alphabet, or expand each categorical outcome to an event column; do not apply
linear-algebraic rank to an arbitrarily encoded tuple. In the deterministic
corpus these entries are zero or one. Exact pivot/rank selection can propose a
compact basis for this **declared finite matrix** before any neural
representation is considered. It does not establish a globally sufficient
predictive state. Held-out columns then falsify the proposed finite basis.

History should be added to state only when it changes a future-test prediction.
That is more precise than accumulating arbitrary flags.

### 2.7 Causal abstraction and the intervention ledger

For low-to-high map \(\tau\) and intervention map \(\omega\), an exact causal
abstraction asks for a commuting diagram:

\[
\tau_\#P_L^{do(i)}
=
P_H^{do(\omega(i))}
\quad\text{for every allowed intervention }i,
\]

or in the deterministic case:

\[
\tau(F_L(x,i))
=
F_H(\tau(x),\omega(i)).
\]

The cursor experiment supplies good local evidence for two low-level storage
realizations and a small input set. It supports treating the resolved cursor
cell as control state while retaining pointer/base as provenance. It does not
prove invariance for every later allocation, control, or emulator build.

The code and reports should distinguish:

\[
(x,z,h,e,i),
\]

where:

- \(x\): complete physical emulator/wrapper state;
- \(z\): proposed task abstraction;
- \(h\): physical lineage and parentage;
- \(e\): epistemic state—what option effects have been learned;
- \(i\): intervention ledger—roots, probes, rollbacks, writes, and symbolic
  transitions.

`MetaState.flags` currently mixes some of these concerns. That is conservative
for search but ambiguous for scientific interpretation.

### 2.8 Approximate abstraction needs a theorem for this objective

Bisimulation metrics and approximate state-abstraction bounds are useful
starting points. A standard discounted form lifts a state metric through
transition distributions:

\[
\mathcal F(d)(x,y)=
\max_o\left[
c_R|r(x,o)-r(y,o)|
+c_T W_d(P_x^o,P_y^o)
\right].
\]

But discounted-MDP value bounds do not transfer automatically to this
repository's undiscounted \(\gamma=1\), episodic frame-minimization problem.
Before approximate merges can authorize planning, either:

1. formulate a discounted option MDP and accept the changed objective; or
2. derive a finite-horizon or proper stochastic-shortest-path bound with
   declared maximum option depth and bounded costs.

Without such assumptions, small local model error can induce arbitrarily large
undiscounted route error.

### 2.9 Separate route cost, search cost, and legitimacy

The selected plan and the process used to discover or attest it answer
different questions:

\[
C_{\mathrm{route}}
=\sum \text{retained controller frames},
\]

\[
C_{\mathrm{search}}
=
(\text{evaluated primitive frames},
\text{option calls},\text{wall time},\text{memory}).
\]

Legitimacy belongs in constraints, not in a scalar that can trade it away:

\[
\begin{aligned}
&N_{\mathrm{external\ roots\ after\ start}}=0,\\
&N_{\mathrm{direct\ writes}}=0,\\
&N_{\mathrm{symbolic\ edges}}=0,\\
&\text{physical terminal}=1,\\
&\text{independent full replay}=1.
\end{aligned}
\]

Two repeated executions are conformance/determinism checks. They are not two
independent stochastic samples. Robustness requires distinct legal roots,
histories, configurations, or core builds.

### 2.10 Reset access is an experimental resource

Three uses of restore must be named separately:

1. **Search oracle:** restore a previously visited exact snapshot to branch
   counterfactually.
2. **Candidate plan:** after its single declared root, use only legal controller
   inputs.
3. **Testing oracle:** probe and roll back to test a boundary, with the work
   charged and reported separately.

The search oracle may be broad while the accepted route remains controller-only.
Hybrid reset access is a strength of the experimental system, but it must be
part of the problem definition rather than invisible infrastructure.

## 3. New implementation-level audit findings

These findings were derived from source inspection. They do not invalidate
already replayed paths, but they narrow algorithm and resource claims.

### 3.1 Generated-but-pruned candidates can consume novelty

`coverage_search_adapter` computes novelty against a global `visited` set, then
adds every per-cell generated winner to `visited` before beam truncation. A
candidate discarded by the width limit can permanently consume the novelty
credit of a later retained lineage.

The legacy `coverage_search` and `area_search` mark cells even earlier, while
generating candidates. Consequences:

- results can depend on action enumeration, parent iteration, and beam order;
- a pruned state can suppress another state's novelty;
- the implementation is not Iterated Width, whose novelty is defined by first
  occurrence of feature tuples under a declared search order; and
- it is not full Go-Explore, whose essential object is an archive of restorable
  representatives.

Accurate current label: **novelty-augmented beam search**. `mario/goexplore.py`
is closer to Phase-1 Go-Explore because it explicitly archives, selects,
restores, and expands cells, though it retains only one representative per
coarse cell.

This is the highest-priority search-semantics defect because it can create
action-order sensitivity in the algorithm meant to escape deceptive regions.

Resolution in this pass: adapter, native coverage, and area search now commit
only retained frontier cells to global novelty. A synthetic deceptive graph
reaches the same necessary cell after an earlier candidate to that cell is
beam-pruned, and solves under all six action-vocabulary permutations. Separate
native regressions verify that generated-but-pruned cells still count as
evidence against a later maze-loop or false hidden-pipe classification.

### 3.2 The learned beam prior is edge-guided, not a path policy

The current beam score adds only the newly selected edge's
\(\log\pi(a\mid x)\) to a freshly computed state score. It does not maintain:

\[
\log\pi(n)
=
\sum_{i=1}^{d(n)}\log\pi(a_i\mid x_i).
\]

Hard `policy_topk` also assigns excluded actions zero support. The positive 1-1
result remains valid as a replayed empirical comparison, but the method should
be called **hard-pruned, edge-guided beam search**, not Levin search, PHS, or a
completeness-preserving learned planner.

A principled full-support policy is:

\[
\pi_\epsilon(a\mid x)
=(1-\epsilon)\pi_\theta(a\mid x)+\epsilon/|A|.
\]

Levin tree search can then prioritize \(d(n)/\pi(n)\), and PHS can price actual
search loss. Start with \(h=0\); the current progress score is a ranking
function, not a proved admissible cost-to-go estimate.

### 3.3 The pipe macro undercounts work and path duration

In `coverage_search`, the sustained-DOWN pipe macro executes several emulator
chunks but:

- does not increment `nodes` for those chunks;
- returns success with `node.frames` unchanged; and
- stores a macro child with `node.frames` unchanged.

Its action path can still replay, but node/frame accounting is not comparable to
ordinary edges. Fix the metric and add regression tests before using these
fields in a matched search-effort claim.

Resolution in this pass: the macro records the number of `run_chunk` calls
made before success/entry, counts each call in the legacy evaluation counter,
and adds scheduled chunk-frame cost to terminal results and retained children.
The regression requires two DOWN calls, then checks the exact chunk-action
path, eight scheduled frames at four frames/chunk, and ten total chunk
evaluations (eight ordinary candidates plus two macro calls).

This is not yet exact primitive-frame accounting. `MarioSim.run_chunk` may stop
early when the episode terminates but does not return the actual step count.
The `frames` field therefore remains scheduled chunk-equivalent path cost, and
`nodes_expanded` is a legacy name for successor/chunk evaluations rather than a
clean tree-node count. Decision depth, successor evaluations, chunk calls, and
actual primitive frames still need separate fields for comparative research.

### 3.4 The greedy contrast is a constructed nonexploration control

`greedy_plan` excludes opaque options by default and accepts only strict
one-step heuristic improvements. It therefore cannot explore the whistle or
take lateral moves by construction. Its failure is useful as a zero-exploration
negative control, but not evidence of a general BFS/UCS advantage.

The benchmark should compare matched explorers and effect-cache planners under
the same option library, roots, and query budgets. If perfect-effect greedy
matches UCS, the result concerns effect discovery rather than broad planning
intelligence.

### 3.5 Reproducibility labels need cleanup

- The July 27 notes call a seven-test group “ROM-backed,” but two acquisition
  tests are explicitly non-ROM schema/flag checks. Five tests are ROM-gated.
- The generated STATUS label records the commit visible before the eventual
  documentation commit. A committed file cannot self-reference its own final
  commit hash. The label should be described as generation-source provenance,
  not “current commit.”
- `working_source_sha256` hashes the tracked diff plus untracked source
  contents. On a clean tree it is the SHA-256 of empty bytes, not a source-tree
  digest. Rename it to a working-diff/source-overlay digest or add the Git tree
  object ID as the source-tree identity.
- Clean reports identify the Git commit correctly, but local ignored roots
  remain necessary and no lockfile pins the complete dependency graph.

Resolution in this pass:

- status output now says `source at generation`, and preserves the old
  `git_rev` field only for compatibility;
- future whistle reports add `git_tree` and rename the overlay digest to
  `working_overlay_sha256`, with its byte-level definition in the report; and
- documentation now calls the seven-test group five ROM-gated adapter tests
  plus two non-ROM acquisition manifest/schema tests.

### 3.6 Bounded Gate-0 implementation verification

Changed implementation:

- `mario/search.py`;
- `mario/meta_planner.py` (claim-label cleanup);
- `scripts/update_status.py`;
- `scripts/bench_sma4_whistle_rom.py`;
- `scripts/bench_whistle_planning.py` (claim-label cleanup);
- `tests/test_adapter_search.py`; and
- `tests/test_status_contract.py`.

Verification:

```bash
./venv/bin/python -m pytest -q \
  tests/test_adapter_search.py tests/test_status_contract.py
# 23 passed

./venv/bin/python scripts/verify_iteration.py
# determinism/snapshot gate: 6 passed
# remaining suite: 171 passed, 10 skipped
# total: 177 passed, 0 failed, 0 errors, 10 skipped

./venv/bin/python scripts/verify_stock_solutions.py --expect-verified 30
# replay-verified: 30/32; unresolved: 6-2, 6-3

./venv/bin/python -m compileall -q mario scripts tests
git diff --check
# both pass; every repository JSON file also parses
```

The exact tested source identity, commands, environment, results, and limitations
are recorded in
`notes/artifacts/2026-07-27-foundations-gate0-verification.json`. This bounded
slice closes retained-frontier novelty commitment, preserves generated cells as
loop evidence, repairs scheduled pipe-macro accounting, makes coverage naming
precise, corrects the test labels, and repairs provenance-field semantics. Gate
0A below is therefore passed. The parallel Gate-0B measurement track remains
open: instrument exact primitive steps and distinct work counters, separate
planning execution from post-solution conformance repetition, and implement a
cumulative full-support policy-guided algorithm rather than relabeling the
current edge-guided beam.

## 4. Strongest scientific reframing

The current system should not be sold as “an Option-SMDP agent discovers the
whistle.” The stronger and more defensible thesis is:

> Given hybrid reset access to a deterministic emulator, can active
> counterexample-guided testing construct a compact, task-relative option
> abstraction that preserves controller-realizable reachability and cost on
> held-out legal histories, while reducing planning queries relative to exact
> physical-state search?

The current work already supplies unusual ingredients for that thesis:

- exact forward dynamics;
- cheap snapshot branching;
- replay-gated promotion;
- physical parent/child lineage;
- intervention and RAM-write ledgers;
- multiple physical representatives under one symbolic state;
- failure-preserving alias checks;
- option-suffix witnesses; and
- a real example where preserving disagreement exposed an observer bug.

One-root continuity is the entrance exam. The research contribution begins
when the system learns and falsifies abstractions across many legal histories.

## 5. Competing hypotheses to preserve

The next experiments must be able to support these less flattering
interpretations:

1. **Library hypothesis:** success comes mainly from a hand-curated option
   library; planner intelligence contributes little.
2. **Counterfactual-splice hypothesis:** the useful acquisition segments cannot
   coexist in one legal history from the chosen root.
3. **Simple-decoder hypothesis:** pointer correction is sufficient and the raw
   alias distinctions have no future behavioral consequence.
4. **Hidden-state hypothesis:** new legal histories reveal additional cursor,
   power, timer, platform, or inventory state that `MetaState` omits.
5. **Accounting-artifact hypothesis:** the 12-frame split is caused by probes or
   settling conventions rather than retained controller time.
6. **Methodology hypothesis:** the publishable result is the
   provenance/conformance/abstraction method, not the whistle route itself.
7. **Search-semantics hypothesis:** much of the existing coverage advantage is
   sensitive to action order, coarse cells, or macro accounting.
8. **Guidance hypothesis:** the 1-1 prior helps through generic action bias or
   hard pruning, not learned cross-level information.

These are not rhetorical caveats. Each needs a registered measurement capable
of making it the leading explanation.

## 6. Research foundations and what to borrow

### 6.1 Options, homomorphisms, and value preservation

- Sutton, Precup, and Singh,
  [“Between MDPs and Semi-MDPs”](https://doi.org/10.1016/S0004-3702(99)00052-1):
  use the exact initiation/policy/termination boundary.
- Ravindran and Barto,
  [“SMDP Homomorphisms”](https://www.ijcai.org/Proceedings/03/Papers/145.pdf):
  use controlled transition, duration, cost, and option-mapping conditions.
- Abel et al.,
  [“Value Preserving State-Action Abstractions”](https://proceedings.mlr.press/v108/abel20a.html):
  ask whether the abstraction plus available options can still represent a
  near-optimal policy. The current library cannot support a global claim while
  legitimate power acquisition and physical World 8 are absent.
- Ahmetoglu et al.,
  [“Skill-Driven Neurosymbolic State Abstractions”](https://papers.nips.cc/paper_files/paper/2025/hash/0fa694fb9f1e265117e8da75966820fe-Abstract-Conference.html):
  treat option initiation/effects as symbol-inducing evidence, with exact
  emulator records retained as the acceptance oracle.

### 6.2 Predictive state, refinement, and conformance

- Littman, Sutton, and Singh,
  [“Predictive Representations of State”](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html):
  use action-conditional future tests rather than arbitrary history flags.
- Clarke et al.,
  [“Counterexample-Guided Abstraction Refinement”](https://web.stanford.edu/class/cs357/cegar.pdf):
  treat a fabricated abstract route or held-out mismatch as a refinement
  counterexample.
- Vaandrager and Melse,
  [“New Fault Domains for Conformance Testing of Finite State Machines”](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CONCUR.2025.34):
  declare exactly which finite implementation class a suffix suite covers.
- Giraud et al.,
  [“L-SCALE”](https://publikationen.bibliothek.kit.edu/1000195438), whose KIT
  repository record was posted 2026-07-20: use snapshots for efficient active
  testing and internal coverage as the observation/equivalence signal, while
  treating threshold-sensitive similarity as a heuristic, never merge
  authority.
- Turkenburg et al.,
  [“Constructing Witnesses for Lower Bounds on Behavioural Distances”](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CSL.2026.25):
  seek finite, quantitative witnesses for a split. Their result is for labelled
  Markov chains, so adapting it to controlled, costed SMDPs is new work.

### 6.3 Quantitative, Markov, and causal abstraction

- Ferns, Panangaden, and Precup,
  [“Metrics for Finite Markov Decision Processes”](https://s.aaai.org/Library/AAAI/2004/aaai04-124.php):
  use behavioral distance only after declaring the value/horizon assumptions.
- Allen et al.,
  [“Learning Markov State Abstractions”](https://papers.nips.cc/paper/2021/hash/454cecc4829279e64d624cd8a8c9ddf1-Abstract.html):
  test whether an abstract state erases history dependence; inverse dynamics
  alone is insufficient.
- Rubenstein et al.,
  [“Causal Consistency of Structural Equation Models”](https://is.mpg.de/publications/rubensteinetal17),
  and Beckers, Eberhardt, and Halpern,
  [“Approximate Causal Abstractions”](https://proceedings.mlr.press/v115/beckers20a.html):
  make the intervention set and low/high commuting map explicit.
- Xia and Bareinboim,
  [“Causal Abstraction Inference under Lossy Representations”](https://proceedings.mlr.press/v267/xia25a.html):
  study multiple low-level realizations of one high-level variable without
  confusing intervention equivalence with controller reachability.

### 6.4 Search, novelty, and learned guidance

- Orseau et al.,
  [“Single-Agent Policy Tree Search With Guarantees”](https://papers.nips.cc/paper_files/paper/2018/hash/52c5189391854c93e8a0e1326e56c14f-Abstract.html):
  replace hard top-\(k\) with cumulative, full-support path-policy search.
- Orseau and Lelis,
  [“Policy-Guided Heuristic Search with Guarantees”](https://ojs.aaai.org/index.php/AAAI/article/view/17469):
  price emulator work as search loss; begin with \(h=0\).
- Lipovetzky,
  [width-based planning survey](https://www.ijcai.org/proceedings/2021/702),
  and Lipovetzky and Geffner,
  [BFWS](https://ojs.aaai.org/index.php/AAAI/article/view/11027):
  implement actual feature-tuple novelty before using width terminology.
- Ecoffet et al.,
  [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9):
  compare one versus multiple exact representatives per archive cell.
- Tuero, Buro, and Lelis,
  [subgoal-guided PHS from failed trajectories](https://proceedings.mlr.press/v267/tuero25a.html):
  use failed search trees as data rather than training only on successes.
- Chatterjee and Khardon,
  [temporally extended/variable actions](https://papers.nips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html):
  test duration as a planning variable while retaining duration one.
- Platnick et al.,
  [“Breadth-First Search vs. Restarting Random Walks for Escaping Uninformed
  Heuristic Regions”](https://ojs.aaai.org/index.php/AAAI/article/view/41044),
  AAAI 2026: saved snapshots make restarting walks from exact plateau states a
  cheap portfolio arm after the primary IW/BFWS/PHS baselines are correct.

### 6.5 Access models and uncertain option effects

- Krishnamurthy, Li, and Sekhari,
  [“The Role of Environment Access in Agnostic Reinforcement Learning”](https://proceedings.mlr.press/v291/krishnamurthy25a.html):
  report the generative/reset/hybrid-reset access model.
- Rohatgi and Foster,
  [“Necessary and Sufficient Oracles”](https://proceedings.mlr.press/v291/rohatgi25b.html):
  treat reset access as a computational resource, not an implementation detail.
- Percassi, Saetti, and Scala,
  [“Planning with Uncertain Action Models”](https://ojs.aaai.org/index.php/AAAI/article/view/40954):
  compare a PUMA-like effect cache, but key reusable effects by a validated
  physical class rather than globally by option name.

## 7. Gated execution program and dependencies

No later result can repair a prerequisite on which it depends. Gate 0 is split
accordingly: the evidence-semantics slice in 0A blocks every later claim and is
passed; the deeper measurement work in 0B can proceed beside physical route
construction but must pass before any search/planner performance claim in Gate
4. The physical legitimacy path remains Gate 1 \(\rightarrow\) Gate 2
\(\rightarrow\) Gate 3.

### Gate 0A — Repair blocking evidence semantics (passed)

Purpose: prevent current labels, novelty state, macro accounting, or provenance
from fabricating the evidence used by later gates.

Completed:

1. commit only retained cells to novelty in adapter and native searches, while
   preserving all generated cells as loop/transition evidence;
2. exercise all six action-vocabulary permutations on the deceptive synthetic
   graph and add native novelty plus loop counterexamples;
3. label current coverage methods novelty-augmented beam search
   unless/until they implement a declared IW or archive algorithm;
4. count actual pipe-macro chunk calls and label path frames as scheduled
   chunk-frame cost;
5. retain both Git-tree identity and an explicitly defined working
   overlay digest;
6. use the corrected five-ROM-gated plus two-non-ROM test wording; and
7. label generated STATUS revision as generation-source provenance.

Acceptance:

- the synthetic action-vocabulary permutations agree;
- generated-but-pruned cells cannot steal novelty but still prevent false loop
  and hidden-pipe classifications;
- terminal and retained-child macro paths report the actual chunk calls and
  scheduled cost exercised by the synthetic simulator;
- focused tests, full suite, stock replay gate, compile, JSON, and diff checks
  pass; and
- a tracked attestation binds those results to the tested working overlay.

### Gate 0B — Complete measurement semantics (parallel; blocks Gate 4)

Actions:

1. add parent-frontier permutations where parent order is configurable;
2. instrument actual primitive emulator steps;
3. distinguish decision depth, successor evaluations, macro chunk calls,
   conformance repetitions, wall time, and peak snapshots/memory;
4. separate search execution from post-solution repeat/conformance auditing;
5. define retained route cost independently of every evaluated/probed cost; and
6. implement cumulative full-support policy guidance before using Levin/PHS
   terminology or making completeness-sensitive learned-search claims.

Acceptance before any search/planner performance claim:

- every reported work field has one definition and a synthetic invariant test;
- macro paths independently replay and exact primitive work equals executed
  primitive work;
- planner discovery, retained execution, and conformance costs are separately
  reproducible; and
- action/parent order is either invariant or explicitly an experimental factor.

### Gate 1 — Establish one continuous SMA4 physical lineage

Purpose: replace the counterfactual splice with a candidate legal route.

First concrete subproblem:

> supply and preserve the required power from the earliest reproducible
> World-1 root.

The current live post-1-2 fortress entry is small Mario (`powerup=0`), while
the cached fortress acquisition assumes leaf/P-speed. Solve this dependency
before treating the two acquisitions as composable.

Actions:

1. select one earliest reproducible World-1 root;
2. map legitimate P-Wing/leaf acquisition and preservation paths reachable
   from it;
3. solve exact physical handoffs incrementally:
   power acquisition/preservation \(\rightarrow\) first whistle
   \(\rightarrow\) second whistle \(\rightarrow\) both spends
   \(\rightarrow\) World 8;
4. make external acquisition roots and direct writes unavailable in flagship
   mode so current macros fail intentionally;
5. record parent exit and child entry identities at every boundary; and
6. replay the concatenated primitive input transcript in a fresh process.

Acceptance:

```text
external_roots_total = 1
external_roots_after_declared_root = 0
direct_memory_assign_calls = 0
symbolic_edges = 0 for every completed physical prefix
parent_exit_record == child_entry_record at every option handoff
cursor_resolved_at_all_accepted_map_boundaries = true
full_executed_input_sha256 = <present>
independent_replay_verified = true
```

Failure is useful: it confirms exactly which cached segment lacks a legal
initiation state and narrows or kills the claimed composition.

### Gate 2 — Replace the physical endpoint

Actions:

1. traverse the selected World-8 pipe physically;
2. build replay-verified World-8 map/level options;
3. replace symbolic `clear_bowser` with an independently observed physical
   terminal;
4. report deaths, retries, retained frames, evaluated frames, restores, option
   calls, wall time, and peak memory separately; and
5. independently replay the full controller transcript.

Until this gate passes, the defensible endpoint is “responsive World-8 map,”
not “beat SMA4.”

### Gate 3 — Build the abstraction experiment

Purpose: turn the current finite diagnostic into held-out research.

Corpus factors:

- independently reached legal roots;
- frame offsets and settling histories;
- both acquisition orders;
- power/damage/inventory variants;
- menu timing;
- prior-level histories;
- cursor allocations;
- save/reload cycles; and
- emulator/core configuration where feasible.

Protocol:

1. pre-register discovery and held-out histories;
2. pre-register discovery and held-out suffix tests;
3. separate \(x,z,h,e,i\) in the schema;
4. execute every real option plus a reserved primitive-probe alphabet;
5. infer separate reachability, exact-cost, and intervention quotients;
6. run one discovery refinement without seeding the initial coloring by full
   `MetaState`, while retaining required goal/output labels;
7. attach a minimal suffix and quantitative lower-bound witness to every split;
8. use CEGAR/active conformance testing to add counterexamples; and
9. retain exact physical execution as the validity oracle, and call
   physical-record search exhaustive only inside a predeclared finite
   root/horizon/action domain.

Primary predictions:

- records 7/8 merge for reachability but split for exact cost;
- records with different `MetaState` but no distinguishing modeled behavior may
  merge under the option-restricted quotient;
- records 9/10 remain equal under held-out physical probes only if their merge
  reflects a real task abstraction;
- record 11/12 equivalence is likely to refine once symbolic Bowser is replaced
  by a physical suffix.

Report false merges, false splits, classes, option queries, suffix length,
behavioral-distance lower bounds, route-cost error, and wall time.

### Gate 4 — Establish principled search baselines

Run on a fixed-root SMB1 6-2 benchmark after Gate 0B:

- current beam;
- corrected novelty-augmented beam;
- Go-Explore with \(K=1,2,4\) representatives per cell;
- IW(1), IW(2), and BFWS;
- bounded restarting walks from exact plateau snapshots;
- uniform LevinTS;
- learned full-support LevinTS;
- PHS with \(h=0\); and
- explicitly aggressive PHS* with the current progress ranking.

Freeze the initial feature vocabulary before observing outcomes:

- area/page;
- x/y tile or region;
- power;
- grounded/airborne;
- signed velocity;
- subpixel buckets;
- platform contact/phase;
- selected enemy/hazard phase; and
- pipe/transition state.

Duration ablation:

\[
\{8\},\qquad \{1\},\qquad \{1,2,4,8,16\}.
\]

Match budgets on evaluated primitive emulator frames, wall time, expansions,
peak memory/snapshots, and exact roots. Report independent replay, censored time
to first solution, incumbent retained frames, exact/coarse states, and
action-order permutations. Primitive inputs remain available when a
completeness claim is desired.

### Gate 5 — Learning, failed trees, and transfer

Only after the physical and search semantics are stable:

1. learn subgoal proposals from failed trees;
2. build generated option graphs with controlled aliases and
   class-dependent effects for PUMA-like cache experiments;
3. generalize the 1-1 prior across level types using full-support path policies
   and a solve-rate noninferiority gate;
4. preserve training rows, configs, seeds, splits, checkpoint/data hashes, and
   machine-readable paired reports; and
5. test a second game only with the same replay and provenance contract.

Do not build a larger neural controller, learned world model, MCTS stack, or
neural option-discovery system yet. None addresses the present failure of
continuous composition or the measurement defects above.

## 8. Immediate action after this documentation checkpoint

The single project-level action remains:

> Starting from one live World-1 root, obtain and preserve the power required
> for both whistle acquisitions through controller input, remove the
> independent second acquisition root plus inventory/leaf writes, and replay
> the resulting continuous physical lineage with zero direct writes.

Gate 0A passed in this pass. Gate 0B proceeds in parallel and must close before
Gate 4 performance claims, but it does not displace the immediate Gate-1
physical-lineage objective.

## 9. Independent claim axes for future reports

Route integrity, abstraction evidence, and planner evidence are not one ladder.
Report an earned vector across these axes rather than implying that progress on
one authorizes the next.

### Route integrity

1. **R1 — Replayable segment:** succeeds from one exact stored root.
2. **R2 — Continuous controller prefix:** multiple segments compose from one
   root with no writes, symbolic edges, or unrelated restores.
3. **R3 — Continuous physical completion:** one controller transcript reaches
   the actual terminal.
4. **R4 — Independently reproduced completion:** R3 replays in a fresh process
   under the declared environment.

### Abstraction evidence

1. **A1 — Declared-domain option:** succeeds over a stated initiation corpus.
2. **A2 — Discovery-set finite conformance:** preserves declared
   enabledness/outcomes/costs on the records used to construct it.
3. **A3 — Held-out finite exact conformance:** preserves those fields on a
   predeclared held-out corpus. Reserve “exact abstraction” for a declared
   finite domain with a completeness proof.
4. **A4 — Bounded approximate abstraction:** satisfies a theorem and empirical
   regret gate for the actual horizon/objective.

### Planner evidence

1. **P1 — Declared-model fixed-instance solve:** returns a plan valid under a
   predeclared transition/intervention model, action set, and budget. Physical
   edges are replay-attested individually; any external root, write, or symbolic
   edge remains explicit on the independent route-integrity axis.
2. **P2 — Matched planner advantage:** beats exploratory/effect-cache baselines
   under equal discovery and execution budgets.
3. **P3 — Generalization:** preserves the registered metric on predeclared new
   roots, levels, games, or configurations.

At commit `43f55b6`, individual SMA4 components earn R1; the independently
verified post-1-2-to-fortress route is an R2 prefix. The full two-whistle route
remains a segmented composition and does not earn R2. The 13-record table is an
A2 diagnostic, not A3 or an abstraction theorem. The mixed planner earns P1
only for its declared hybrid segmented model; it has no P2 or P3 claim.
