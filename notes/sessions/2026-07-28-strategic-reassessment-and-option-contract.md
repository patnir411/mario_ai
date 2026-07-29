# 2026-07-28 — Strategic reassessment and phase-aware option contract

> Research availability cutoff: **2026-07-28**. This note records a
> whole-project review, a mathematical reframing, and a documentation decision.
> It introduces no new route, benchmark, test, planner behavior, or v3 artifact.
> Current code still emits `mario-ai.option-observation.v2`; the proposed
> `OptionMachine` / `OptionTrace` v3 contract is documented in
> `notes/theory/option-machine-trace-v3.md` and remains unimplemented.

## 0. Executive decision

The strongest defensible research identity is now:

> **Query-efficient discovery and finite-domain certification of phase-aware,
> weighted option quotients in resettable black-box environments.**

Exact search remains the solver and exact replay remains the validity oracle.
Mario is the adversarial case study, not the whole scientific claim.

The program splits into two branches sharing one contract:

- **Science branch:** synthetic ground truth, automatic counterexample-guided
  discovery, reset-access ablations, matched search baselines, then an unchanged
  open second domain.
- **Mario-integrity branch:** a bounded attempt to obtain one physical,
  controller-realizable, write-free SMA4 lineage. It is an important integrity
  test, but it no longer blocks the portable science.

The new roadmap is chosen because the current repository already proves that
exact execution and replay can solve difficult Mario content, but it does not
yet prove that its abstraction is automatically discovered, portable,
query-efficient, or useful to planning. Another bespoke route can deepen the
case study; it cannot by itself close those scientific gaps.

## 1. Plain-English recap

The project began by searching the real emulator from scratch. It found and
replayed full SMB1 routes. Neural policies were then trained from search, but
standalone generalization was weak. One learned policy did help search on 1-1,
so learning was demoted from “replacement driver” to “search guide.”

The SMA4 work asked whether long routes could be composed from named skills such
as clearing a level, acquiring a whistle, and spending it. That work exposed a
more fundamental problem: two states that look identical in a small summary can
have different physical histories, costs, hidden phases, and future behavior.
The repository learned to keep the physical records separate and to test them
with option suffixes.

The next research question is therefore not “can we add a larger planner?” It
is:

> Can the system discover exactly which distinctions matter, using as few
> exact emulator experiments as possible, and produce a small model whose plans
> replay correctly?

## 2. Evidence before interpretation

### 2.1 What is reliable

- SMB1 any% remains independently replay-verified 8/8.
- Stock SMB1 remains 30/32 replay-verified: 6-2 is unsolved and 6-3 is
  quarantined.
- Exact resettable emulator search is the reliable low-level solver.
- The local learned-prior result on 1-1 remains 7005 to 2770 expanded nodes
  (60.5% fewer) with solve preservation in that configuration; its clean-clone
  training provenance remains incomplete.
- The first Gate-0 search-accounting cleanup is source-bound in
  `notes/artifacts/2026-07-27-foundations-gate0-verification.json`.
- The repository suite remains 177 passed, zero failed/errors, ten expected
  skips at the start of this documentation pass.

### 2.2 What the learning experiments actually say

The flat MLP, entity Transformer, held-out behavior-cloning, DAgger, and
reverse-curriculum attempts are useful local negative results. They show that
validation accuracy and short-horizon imitation are poor substitutes for
closed-loop completion in the tested regime. They do not prove that imitation
learning, Transformers, or reinforcement learning cannot solve Mario.

The positive learning result is narrower: a prior can reduce work inside exact
search. Future guidance claims therefore require:

1. solve-rate non-inferiority;
2. exact primitive-frame work, not nodes alone;
3. wall time and memory;
4. multiple levels and roots; and
5. complete action support or an explicit fallback.

### 2.3 What the SMA4 experiments actually say

The corrected SMA4 table contains 13 physical records, seven options, 91
representative/option rows, and 14 enabled transitions repeated twice. It
refines to 11 stable finite blocks with no repeat-conformance conflict in that
encountered table.

That is useful finite evidence. It is not:

- a continuous physical completion;
- a globally minimal quotient;
- proof that `MetaState` is Markov;
- a global bisimulation;
- a stochastic Option-SMDP result; or
- a planner speedup.

The route still uses independent powered roots, an inventory merge, a fortress
leaf rehold, and a symbolic 6,000-frame Bowser edge. Physical lineage is
therefore a correctness constraint, not a cosmetic provenance field.

## 3. Mathematical diagnosis

### 3.1 The physical system

For fixed ROM/core/wrapper/action semantics,

\[
x_{t+1}=F(x_t,a_t)
\]

is a deterministic weighted labelled transition system. Search from a fixed
root is a shortest-path problem over exact resettable state.

The word “exact” applies to the transition oracle and independently replayed
controller actions. It does not make a decoded RAM summary exact, and it does
not make two noncanonical savestate byte strings equivalent.

### 3.2 The missing state variable

The current `Option` is a legacy orchestration wrapper around an arbitrary
runner. A cached executor can have memory: action index, branch, retry count,
timer, or feedback-controller mode. If that memory is hidden, the boundary
process need not be Markov.

The corrected semantic object is

\[
\Omega=(id,r,I,Q,q_0,Q_{\mathrm{term}},\pi,\delta,\beta),
\]

and execution evolves on \((x,q)\). A fixed action trace is the simplest
finite-state controller, with phase equal to primitive step index.

This does not claim that a platform timer or enemy cycle is controller phase.
Those belong in physical state \(x\) or in the observation contract. Controller
phase \(q\) is the machine's own memory.

### 3.3 The finite quotient claim

For ordinary invocation, the declared finite physical boundary domain is
\(D_X\subseteq X\). Typed failure symbols form a disjoint set
\(B_{\mathrm{failure}}\), so the closed quotient carrier is
\(\widetilde D=D_X\uplus B_{\mathrm{failure}}\), not a set that pretends a
failure label is an emulator state. Each machine \(o\in O\) executes internally
on its own \(X\times Q_o\), resets to \(q_0^o\), and maps a physical boundary
partially into \(\widetilde D\); there is no global phase space shared between
different machines. Failure symbols are formal zero-weight absorbing states
after their first typed failure edge. Two carrier states may share a block only
if every admissible machine agrees on:

1. enabledness/initiation;
2. task/goal labels;
3. outcome and termination, including terminal phase only when declared as an
   output of that machine;
4. exact retained weight under a hashed algebra
   \((W,\oplus,0_W)\); and
5. successor quotient block, including any typed failure sink.

Under deterministic execution and closure, this produces a well-defined
weighted quotient preserving option-word reachability and accumulated weight **inside
\(\widetilde D\)**. The proof is induction on the option-word length. It is useful, but by
itself it is close to classical automata/MDP minimization. The publishable
question begins upstream: how the distinctions, probes, and option machines are
discovered with controlled query cost.

The initial exact route weight is retained emulator frames. Primitive calls may
be a separately reported component. Discovery/conformance frames, restores,
nodes, wall time, and memory are experiment work, not route-edge weight.
Interruption/resumption is outside the first profile; a future resumable state
would require a machine-tagged disjoint union of phase spaces.

### 3.4 The lineage theorem

If every trace is a complete primitive execution, no state-changing
intervention occurs after entry, and each actual parent exit equals the child
entry, induction over trace steps and route segments establishes one legal
controller trajectory.

An unrelated state from the same abstract block cannot replace the actual
successor. Quotient equivalence can justify future predictions only after its
premises are earned; it cannot retroactively repair broken physical lineage.

### 3.5 Reset access is an experimental variable

Snapshots provide membership-style queries: restore a declared boundary,
execute a suffix, and observe the result. They do not provide an unrestricted
equivalence oracle.

Every result should state one of:

- **full snapshot:** arbitrary encountered boundaries can be restored;
- **root-only:** only declared roots can be restored; or
- **no arbitrary reset:** continuation must follow the current live history.

The abstraction method must be ablated across these modes. Otherwise cheap
reset access may hide the method's real sample complexity or composability
failure.

### 3.6 Discovery queries are not certification queries

Three counts must never be merged:

1. **learner membership queries** used to construct a hypothesis;
2. **hidden benchmark-oracle evaluations** used only to score recovery against
   synthetic ground truth; and
3. **certification/conformance queries** needed to discharge the declared
   finite fault-domain guarantee.

Active selection can reduce the first count while an exhaustive hidden oracle
shows zero false merges. That is exact external evaluation, not a smaller
self-contained certificate. Exact certification of unqueried rows additionally
requires a bounded hypothesis/fault domain plus a complete conformance suite,
symbolic ground-truth proof, or exhaustive validation. Report all three counts
and total exact interaction.

## 4. What is potentially novel—and what is not

### 4.1 Foundation, not contribution by itself

- hand-written options;
- a finite-state controller encoding of an action trace;
- deterministic option-level uniform-cost search;
- partition refinement on a small enumerated table;
- hashing snapshots and replaying actions; or
- a theorem that a transition-respecting finite quotient preserves reachability.

These are valuable infrastructure and correctness foundations. Presenting them
alone as the main research novelty would be weak.

### 4.2 Stronger contribution candidates

1. **Automatic phase/predicate discovery.** Begin with deliberately coarse
   candidate observations; select controlled suffixes that distinguish
   histories; split only on replay-backed counterexamples.
2. **Query-efficient discovery and scoped conformance.** Reduce learner
   membership queries relative to exhaustive option-table construction, then
   state separately what additional queries or proof discharge the finite
   certification claim.
3. **Limited-reset abstraction.** Quantify what is learnable and useful under
   full, root-only, and no-arbitrary-reset access.
4. **Planning utility.** Show that the learned quotient reduces end-to-end
   primitive work or wall time, not merely representation size.
5. **Cross-domain invariance.** Run the same contract and algorithm, without
   game-specific patches, on synthetic ground truth, Mario, and an open second
   environment.
6. **Model-proposer/exact-verifier separation.** Let learned or foundation
   models propose predicates, controller programs, or distinguishing probes;
   let exact execution reject or certify them.

## 5. July 28 research map

Venue status is explicit. Preprints are hypotheses and novelty warnings, not
settled results.

### 5.1 Active automata learning and conformance

- Fortz et al., “A research agenda for active automata learning,” STTT 2026,
  is peer-reviewed/open access:
  https://link.springer.com/article/10.1007/s10009-026-00839-z
  - Relevance: choose the teacher/access model, query alphabet, fault domain,
    noise assumptions, and guarantee before importing an \(L^*\)-style story.
  - Experiment: compare exhaustive suffix tables, random suffixes, and active
    distinguishing-suffix selection at equal learner-membership budgets;
    report hidden scoring and conformance/certification work separately.
- Nixon, “The Myhill-Nerode Theorem for Bounded Interaction,” is a 2026
  preprint:
  https://arxiv.org/abs/2603.21399
  - Relevance: a finite controller probe family inducing a canonical quotient
    is very close to this program and is therefore both a formal lead and a
    novelty-collision warning.
  - Boundary: its finite-POMDP and probe-family theorems do not automatically
    cover exact weighted emulator options.

### 5.2 Abstraction semantics

- Zhang, Luo, and Baltieri, “Compositional Behavioral Semantics for State
  Abstraction in Reinforcement Learning,” is listed for ICML 2026; the public
  manuscript is:
  https://arxiv.org/abs/2606.25357
  - Relevance: specify the exact behavior to preserve rather than asking for an
    abstraction that is vaguely “good.”
  - Experiment: instantiate separate reachability-, cost-, and
    intervention-preserving quotients and measure when they disagree.
- Abel et al.'s value-preserving state-action abstractions, SMDP
  homomorphisms, probabilistic bisimulation metrics, predictive-state
  representations, and CEGAR remain the mathematical backbone already listed
  in `notes/research-bibliography.md`.

### 5.3 Search and temporal decomposition

- Tuero et al., “Structure-Induced Information for Rerooting Levin Tree
  Search,” is listed for ICML 2026:
  https://arxiv.org/abs/2605.30664
  - Relevance: useful decomposition may be represented by allocating search
    effort to discovered structure, without reconstructing explicit symbolic
    subgoals.
  - Experiment: compare explicit option/subgoal planning with clustering,
    heuristic, and hybrid rerooters on the same exact tree-search budget.
- Chang et al., “The Surprising Difficulty of Search in Model-Based
  Reinforcement Learning,” ICML 2026:
  https://arxiv.org/abs/2601.21306
  - Relevance: even an accurate model does not make a learned-value search
    wrapper automatically beneficial; search-induced distribution shift and
    overestimation must be measured.
  - Boundary: its learned continuous-control setting differs from an exact
    discrete emulator.
- Nayyar and Srivastava, “Autonomous Option Invention for Continual
  Hierarchical Reinforcement Learning and Planning,” AAAI 2025:
  https://ojs.aaai.org/index.php/AAAI/article/view/34163
  - Relevance: compare manual Mario options with invented symbolic options on
    composability, reuse, and independence across generated tasks.

### 5.4 Program synthesis and model-generated certificates

- Macfarlane et al., “Gradient-Based Program Synthesis with Neurally
  Interpreted Languages,” ICLR 2026:
  https://iclr.cc/virtual/2026/poster/10009887
  - Relevance: a bounded alternative to a hand-written option DSL.
  - Experiment: compare learned discrete programs, explicit finite-state
    machines, memorized traces, and neural behavior cloning; replay remains the
    authority.
- Rajabpour et al., “Revisiting OOD Generalization in Programmatic RL,” is
  listed for ICML 2026; the earlier public version is:
  https://openreview.net/forum?id=e26MPyczN9
  - Relevance: apparent programmatic-policy advantages can arise from unmatched
    observations, reward design, or search capacity.
  - Experiment: match those factors plus compute/tuning before attributing
    transfer to the program representation.
- Taheri et al., “BarrierBench,” L4DC 2026, PMLR 331:
  https://proceedings.mlr.press/v331/taheri26a.html
  - Relevance: it demonstrates a promising division of labor—models propose
    certificate structure while a formal solver validates candidates.
  - Translation: models may propose predicates, phases, or distinguishing
    suffixes; exact emulator traces and finite-domain checks validate them.
    Barrier certificates themselves are not the current Mario theorem.

### 5.5 Future interactive models

- ARC Prize Foundation, “ARC-AGI-3,” 2026 preprint:
  https://arxiv.org/abs/2603.24621
  - It reported frontier systems below 1% in March 2026 on novel interactive
    environments.
- ARC Prize's public July 24 result reports Claude Opus 5 High at 30.2%:
  https://arcprize.org/results/anthropic-claude-opus-5
  - The rapid movement over four months is evidence against making “current
    models cannot play games” a durable thesis.
- Ouyang et al., “GameWorld,” 2026 preprint:
  https://arxiv.org/abs/2604.07429
  - Its 34-game/170-task benchmark reports leading multimodal agents still far
    from human capability and emphasizes state-verifiable outcomes.
- Rodionov, “Executable World Models for ARC-AGI-3,” public 2026 manuscript
  accepted at AGI-2026 according to its arXiv comments:
  https://arxiv.org/abs/2605.05138
  - Relevance: code-generating agents can propose executable dynamics and test
    them against observations.
  - Boundary: results are on public games and private validation remains
    untested, so this is an experimental lead rather than established
    generalization.

The durable design principle is:

> **Models propose; exact environments falsify and finite artifacts certify.**

## 6. Devil's-advocate review

### Advocate: continue the SMA4 route first

The project needs one uninterrupted physical route to prove its option story is
not built on counterfactual splicing. Finishing the route may reveal the real
state variables and controller constraints that synthetic systems miss.

### Critic: stop making Mario the gate

Route completion is vulnerable to unbounded reverse engineering, ROM-specific
patches, and unpublishable artifacts. Even success would not establish
automatic discovery, query efficiency, or transfer. A frontier model may soon
make raw completion uninteresting.

### Advocate: build the quotient implementation now

A phase-aware trace schema and a finite theorem create the clean substrate
needed for every later comparison.

### Critic: the theorem is textbook

If predicates, options, boundary roots, and tests are all hand-selected, the
system can manufacture a tiny self-confirming model. Certification without
automatic experiment selection may be overhead rather than research.

### Resolution

Do both, but change their roles:

- freeze the correctness contract and test it against synthetic ground truth;
- run a bounded SMA4 integrity track in parallel;
- make automatic discovery, query reduction, reset ablation, and cross-domain
  evidence the publication gate.

## 7. Systems reassessment on the M2 Pro

Apple's reference specification for the 2023 14-inch M2 Pro configuration lists
a 12-core CPU with eight performance and four efficiency cores, a 19-core GPU,
16 GB unified memory, and 200 GB/s memory bandwidth:
https://support.apple.com/en-us/111340

Tracked one-shot NES diagnostics report:

- 1,375.7 emulator frames/s;
- 73.076 microseconds per isolated snapshot dump/load roundtrip; and
- 2,968.65 microseconds per restore → four steps → child snapshot, or 336.9
  successors/s.

The isolated snapshot fraction is

\[
\frac{73.076}{2968.65}=0.0246.
\]

Even deleting that cost entirely gives an Amdahl ceiling of

\[
\frac{1}{1-0.0246}\approx1.025.
\]

Therefore the old “snapshot cloning usually dominates” narrative is false for
this measured NES workload. Emulator stepping and total query count dominate.
The separately measured SMA4 loops are not additive and cannot support a causal
breakdown.

Near-term systems work should be:

- worker-local persistent emulator processes;
- compact content-addressed job descriptions;
- opaque snapshots kept inside their owning worker;
- deterministic ordered aggregation;
- a serial independent replay gate; and
- repeated end-to-end throughput, memory, thermal, and coordinator-overhead
  measurements.

Parallel scaling remains a hypothesis until a repeated source-bound artifact is
preserved. Do not prioritize snapshot compression, custom Metal kernels,
shared-tree concurrency, or a systems-only paper.

MPS remains optional for batched learned inference. Apple's MPS page documents
the backend and still labels it beta:
https://developer.apple.com/metal/pytorch/
Backend selection must follow parity plus measured batch crossover and
end-to-end outcomes; there is no blanket MPS/CPU rule.

## 8. Two-branch experiment program

### Shared Gate S0 — contract and accounting

Deliver:

- additive `OptionMachine` and `OptionTrace` v3 schemas;
- exact primitive action/phase interception;
- retained-route versus discovery/conformance/attestation work separation;
- content-addressed environment, machine, and lineage identity;
- full/root-only/no-reset access labels; and
- serial/cached/reordered/worker-local invariance checks.

Kill condition: do not emit or promote v3 if any phase, action, intervention, or
cost invariant can be bypassed.

### Science Gate S1 — synthetic quotient oracle

Build generated deterministic systems with known minimal weighted quotients and
blinded structural seeds. Include:

- hidden delayed effects and timers;
- inventory/power bits;
- noncommutative actions/options;
- observation aliases;
- explicit controller phase;
- variable cost;
- spurious state labels; and
- histories that tempt representative substitution.

Measure:

- false merges and false splits;
- recovered versus true quotient size;
- learner membership queries, hidden benchmark-oracle evaluations, and
  certification/conformance queries separately;
- total exact primitive work;
- shortest distinguishing suffix;
- lifted-plan replay validity;
- cost regret; and
- seed-to-seed uncertainty.

Pass: zero false merges and zero invalid lifts against the hidden ground-truth
oracle over the declared finite domain. This is external recovery evaluation;
it becomes a self-contained certificate only when the separately declared
fault-domain/conformance or exhaustive-validation obligation also passes.

### Science Gate S2 — automatic experiment selection

Compare exhaustive rows, random suffixes, uncertainty sampling, and
counterexample-guided distinguishing suffixes. Predicates/phases may be proposed
from RAM/entities or by a model, but every split and merge is execution-tested.

Pass, with the S1 external-recovery gate intact: active selection uses at least
2x fewer **learner membership queries** than exhaustive construction **or**,
across at least 30 blinded paired seeds, at least 25% fewer than random
selection with the paired 95% bootstrap confidence interval for the
learner-query ratio entirely below 1.0. Hidden scoring and exact certification
counts are reported separately and are not credited as learner-query savings.

### Science Gate S3 — matched planning tournament

At equal primitive-frame, wall-time, and memory budgets compare:

- BFS/UCS;
- the current novelty-augmented beam;
- actual Go-Explore with one and multiple representatives per cell;
- IW(1), IW(2), and BFWS;
- restarting random walks;
- full-support Levin/PHS-style search;
- explicit option/subgoal planning;
- structure-induced rerooting; and
- one bounded PUCT/MCTS control.

Pass for abstraction utility: at least 2x state/query compression **or** 20%
end-to-end planning savings without solve-rate loss or replay-invalid plans.

### Science Gate S4 — reset ablations

Repeat the winning configuration under full snapshot, root-only, and no
arbitrary reset. Report failures rather than silently changing the interface.

Pass: characterize the degradation and retain an advantage in at least one
limited-reset regime. A full-reset-only result remains useful but must be
claimed narrowly.

### Science Gate S5 — open second domain

The predeclared candidate set is MiniHack and Crafter. Before any
outcome-bearing quotient experiment, publish a non-outcome selection spike:
legal redistribution, reproducible installation, exact-seed/reset behavior,
explicit terminal predicates, baseline runtime, and adapter-specific source
lines. Redistribution and replay are mandatory; among survivors choose the
candidate with fewer adapter-specific lines, then lower baseline runtime as the
tie-breaker. Freeze that domain, task manifest, and harness before inspecting
method outcomes. If neither survives, add no replacement without a dated
pre-outcome amendment.

Pass: no game-specific predicate patches, no schema change, and a predeclared
subset of S1–S4 metrics reproduced.

### Mario Gate M1 — bounded physical continuity

Attempt one earliest World-1 root, legitimate power, both whistle acquisitions,
zero writes, no unrelated restores, a physical World-8/Bowser endpoint, and
fresh-process primitive replay.

Budget: at most two predeclared legitimate-power approaches or 40 recorded
ROM-backed process-hours, whichever comes first. Parallel processes count
separately. Record commands, roots, artifacts, and failures.

Success makes SMA4 a strong adversarial case study. Failure freezes route
expansion and becomes a negative option-composition case; it does not stop the
science branch.

### Learning Gate L1 — guidance after measurement

Compare plain search, static/random priors, soft cumulative priors, full-support
PHS/Levin guidance, and any learned proposer. Require solve-rate
non-inferiority and at least 20% savings in both exact primitive work and wall
time. Node-only savings do not pass.

### Model Gate W1 — surrogate only if stepping dominates enough

Test a learned dynamics surrogate only if exact stepping remains the measured
bottleneck and the surrogate offers at least 5x end-to-end candidate
throughput. All accepted plans must replay in the exact emulator. Model
prediction accuracy alone is not a pass criterion.

## 9. 2036 → 2026 backcast

This is a strategic hypothesis, not a forecast.

By 2036, it is plausible that generalist multimodal/coding agents make these
tasks cheap:

- screen-to-controller completion of known games;
- automatic RAM/entity mapping;
- adapter and reward/terminal glue;
- route demonstration generation;
- ordinary search distillation;
- option and predicate proposals; and
- local model fitting.

The July 2026 jump in public ARC-AGI-3 performance is a concrete warning that a
thesis built on today's model weakness can expire quickly.

Outputs more likely to remain useful are:

- a precise oracle/action/reset ABI;
- legal open environments with known ground-truth quotients;
- complete physical lineage DAGs;
- controller phase and primitive execution traces;
- minimal distinguishing suffixes and alias counterexamples;
- query-complexity and limited-reset results;
- lifting/replay certificates;
- negative-result corpora that reproduce;
- model-agnostic proposer/verifier interfaces; and
- evaluation that counts real interaction and rejects teleportation.

Looking backward from that future, the right 2026 investment is not a larger
Mario-specific neural controller. It is the exact evidence and experiment
machinery that lets increasingly capable models make bold proposals without
being trusted blindly.

## 10. Predeclared pivots

- Freeze the SMA4 route branch if M1 exhausts its budget.
- Drop an “exact quotient” claim on the first false merge or invalid lifted
  replay inside its declared domain.
- Treat quotient certification as infrastructure if it produces neither 2x
  compression nor 20% end-to-end planning savings.
- Drop method-paper framing if success requires level IDs, absolute coordinates,
  or a manual predicate for each counterexample.
- Drop programmatic-option superiority if matched observations, reward,
  training compute, and search capacity erase the difference.
- Do not pursue a systems paper without open environments, repeated
  correctness-constrained multicore scaling, and a nontrivial systems result.
- Do not pursue a benchmark paper if artifacts cannot be redistributed or
  fewer than three serious algorithm families reproduce.
- Pivot toward the counterexample/verification dataset if the quotient theorem
  is textbook and automatic discovery adds no query or planning advantage.

## 11. Ordered next steps

1. Freeze v3 schema semantics and review them against current code without
   changing legacy artifacts.
2. Implement JSON Schemas, validators, synthetic known-quotient fixtures, and
   adversarial invariant tests.
3. Add exact primitive/phase/cost/intervention tracing behind an opt-in path.
4. Prove serial, cached, reordered, and worker-local conformance identical.
5. Implement exhaustive and active suffix selection on blinded synthetic
   seeds.
6. In parallel, run the bounded SMA4 M1 approaches and preserve either success
   or failure.
7. Run the matched search tournament and reset ablations.
8. Freeze the harness, select the open second domain, and repeat without
   game-specific changes.
9. Only then add learned/model proposers and evaluate whether they reduce exact
   query or planning cost.

The immediate action is narrower than this roadmap:

> **Freeze the phase-aware option/trace schema and make the synthetic
> known-quotient suite pass serial, cached, reordered, and worker-local
> invariance checks.**
