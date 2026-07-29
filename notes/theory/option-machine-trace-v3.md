# Phase-aware `OptionMachine` / `OptionTrace` v3 contract

Status: **normative design draft; not implemented**

Contract date: **2026-07-28**

Proposed schema names: `mario-ai.option-machine.v3` and
`mario-ai.option-trace.v3`

This note freezes the semantic target for the next implementation slice. It
does not change the current `Option`, `OptionResult`, planner, or
`mario-ai.option-observation.v2` artifacts. No existing route or trace should be
relabeled as v3.

The point of the contract is simple: an option with memory is not determined by
the emulator state alone. Its controller phase must be explicit, its retained
primitive execution must be replayable, and route cost must not be mixed with
the work used to discover or attest the route.

## 1. Vocabulary and fixed environment

Fix a ROM, emulator/core build, adapter, wrapper configuration, action
contract, and observation decoder. Let

\[
x_{t+1}=F(x_t,a_t), \qquad y_t=h(x_t),
\]

where:

- \(x_t\) is the complete physical state needed for exact continuation,
  including emulator, wrapper, scheduler, and adapter context;
- \(a_t\) is one canonical primitive adapter action under the declared action
  contract; and
- \(y_t\) is the declared controller observation.

One primitive transition row means one call to the declared primitive
transition oracle. If an adapter call advances more than one emulator frame,
that fact is part of the action contract and the exact frame count is recorded
separately.

`MetaState` is a task annotation. It is not complete physical identity and it
is not controller phase.

## 2. Semantic object

A deterministic phase-aware option machine is

\[
\Omega=(id,r,I,Q,q_0,Q_{\mathrm{term}},\pi,\delta,\beta),
\]

with:

- stable machine identifier \(id\) and positive semantic revision \(r\);
- initiation set \(I\) over complete physical entry states;
- finite controller-phase set \(Q\);
- initial phase \(q_0\in Q\), reset on ordinary invocation;
- terminal phase set \(Q_{\mathrm{term}}\subset Q\), with
  \(q_0\notin Q_{\mathrm{term}}\);
- action policy
  \(\pi:Y\times(Q\setminus Q_{\mathrm{term}})\rightarrow A\);
- deterministic phase update
  \(\delta:Y\times(Q\setminus Q_{\mathrm{term}})\times A\times Y
  \rightarrow Q\); and
- termination predicate \(\beta:Y\times Q\rightarrow\{0,1\}\), evaluated after
  a primitive transition.

Terminal phases emit no action and must satisfy
\(\beta(y,q)=1\) for every \(q\in Q_{\mathrm{term}}\) and declared observation
\(y\). The predicate may also stop after a physical postcondition is observed
in a nonterminal phase.

Execution from \(x_0\in I\) is

\[
\begin{aligned}
y_t &= h(x_t),\\
a_t &= \pi(y_t,q_t),\\
x_{t+1} &= F(x_t,a_t),\\
q_{t+1} &= \delta(y_t,q_t,a_t,h(x_{t+1})).
\end{aligned}
\]

Execution stops when \(\beta(h(x_{t+1}),q_{t+1})=1\), the environment
terminates, a declared limit is reached, or the executor interrupts with an
explicit reason. The Markov state during execution is \((x_t,q_t)\), not
`MetaState` and not \(x_t\) alone.

In v3's first serializable profile, a cached action trace is the degenerate
finite-state controller:

\[
T\ge1,\quad Q=\{0,\ldots,T\},\quad q_0=0,\quad
Q_{\mathrm{term}}=\{T\},
\]

with actions \(a_0,\ldots,a_{T-1}\),
\(\pi(y,q)=a_q\) for \(q<T\), and \(\delta(q)=q+1\). The controller ignores
\(y\), but its step-index phase is still explicit. The initial profile
disallows zero-step options. A feedback-controller profile is reserved until
the repository has a content-addressed predicate/guard language.

Here **phase** means controller memory \(q\). It does not mean platform phase,
enemy phase, a hidden game timer, subpixel position, route-option index,
`MetaState.flags`, or an unexpanded action-chunk number.

## 3. `OptionMachine` artifact

The machine artifact defines reusable controller semantics. It does not contain
an empirical success rate, a measured route cost, or a run-local record ID.

Required fields:

| Field | Contract |
|---|---|
| `schema` | Exactly `mario-ai.option-machine.v3`. |
| `machine_id` | Stable nonempty identifier, not a local path. |
| `revision` | Positive integer; every semantic controller change increments it. |
| `machine_type` | Initially `open_loop_trace`. Reserve `deterministic_fsc`, but reject it until its guard language exists. |
| `action_contract` | Adapter/action-space ID, ordered canonical actions or buttons, primitive-step semantics, and SHA-256. |
| `observation_contract` | Decoder ID and implementation/source SHA-256. Open-loop machines may declare that the observation is ignored. |
| `phase_contract` | Encoding, initial phase, finite phase count, and terminal phases. For `open_loop_trace`, use `primitive_step_index`, phase 0, and terminal phase \(T\). |
| `initiation` | Predicate ID, implementation hash, and declared state-domain semantics. A `MetaState` predicate alone is not physical initiation proof. |
| `controller` | Canonically expanded primitive actions or a content-addressed reference, expanded length, and action/phase SHA-256. |
| `termination` | Program exhaustion plus any physical postcondition ID/hash. Exhaustion means “controller ended,” not automatically “task succeeded.” |
| `audit` | Nonsemantic knowledge tier, boundary evidence policy, declared experimental intervention classes, and source artifact hashes. An allowed state-changing intervention is not a controller action. |
| `hash_encoding` | Versioned canonical byte encoding; initially `mario-ai.stable-encode.v1`. |
| `controller_sha256` | Hash of normalized machine type, action/observation contracts, phase machine, initiation semantics, controller, and termination semantics. |
| `artifact_sha256` | Hash of the normalized complete manifest, including ID/revision and audit metadata but excluding itself, local display paths, and timestamps. |

Run-length compression is permitted for storage. Semantic hashing and replay
operate on the canonical primitive expansion. Each expanded phase emits exactly
one declared primitive action except a terminal phase, which emits none.

Prior cost estimates may live in a nonsemantic `annotations` object. Measured
cost belongs to an execution trace.

Knowledge tier, evidence policy, local provenance, and empirical verification do
not split behaviorally equal controllers. `controller_sha256` is the cache and
behavior identity; `artifact_sha256` binds which reviewed manifest and
provenance were used in an experiment.

## 4. `OptionTrace` artifact

An `OptionTrace` is one concrete execution episode. It is not a machine
definition, an option-effect cache entry, or a quotient row.

Required top-level fields:

| Field | Contract |
|---|---|
| `schema` | Exactly `mario-ai.option-trace.v3`. |
| `trace_id` | SHA-256 of the normalized retained-episode identity defined below. |
| `machine_ref` | `machine_id`, `revision`, `controller_sha256`, and exact `artifact_sha256`. |
| `environment_fingerprint` | Game/ROM, emulator/core, adapter, wrapper/config, action contract, and observation contract identities. |
| `weight_contract_ref` | Content hash of the experiment/table-level route-weight algebra. |
| `reset_access` | Declared access mode: `full_snapshot`, `root_only`, or `no_arbitrary_reset`. |
| `setup` | Harness-only establishment of the entry state. Its method must be legal under `reset_access`; setup is not a controller action. |
| `entry` | Physical record ID, nullable lineage parent for declared roots, identity/evidence mode, digests, observable, adapter context, `MetaState` annotation, and entry phase. |
| `steps` | Ordered retained primitive transitions. |
| `evidence_cadence` | Required intermediate observation/physical-evidence schedule for this machine profile. |
| `termination` | `succeeded`, `failed`, or `interrupted`; reason; final phase; machine-termination and physical-postcondition results. |
| `exit` | Authoritative live exit record/evidence, observable, adapter context, and `MetaState` annotation. |
| `retained_cost` | Primitive transition calls, emulator frames advanced, and any declared additive task weight. |
| `evaluation_work` | Discovery/evaluation frames, rolled-back probes, restores, option calls, search nodes, attestation work, wall time, and peak memory. |
| `interventions` | Complete typed ledger after machine entry: external roots, RAM writes, adapter-context mutations, lookahead/rollback, symbolic edges, and undeclared side effects. |
| `attestations` | Replay identity, repeat count, action/phase match, terminal match, entry/exit evidence, and the derived `controller_realizable` verdict. |
| `result_sha256` | Hash of the normalized retained-result signature used for serial/cache/order/worker invariance. |
| `action_phase_sha256` | Hash of the canonical `(phase_before, action, phase_after)` sequence. |

Every `steps[t]` row contains:

- `index == t`;
- `phase_before`;
- canonical primitive action;
- exactly one primitive transition-oracle call;
- exact emulator frames advanced;
- `phase_after`;
- environment terminal flag; and
- any observation/physical evidence required at index \(t\) by
  `evidence_cadence`.

Full observations and snapshots may be content-addressed rather than embedded.
For the initial open-loop profile, entry/exit evidence plus action/phase/frame
rows and independent replay are mandatory; intermediate evidence cadence is a
declared storage/overhead choice. A future feedback controller must record the
policy-input observation digest at every decision point. The live executor
state after the final retained step is authoritative; a runner-supplied earlier
snapshot is provenance only.

### 4.1 Trace identity and result comparison

Every v3 content hash is SHA-256 over the exact bytes produced by the declared
`hash_encoding`. `mario-ai.stable-encode.v1` means the framed, type-tagged,
mapping-key-sorted encoding implemented by `mario.provenance.stable_encode` at
the frozen v3 source identity. An encoding change requires a new encoding
version. Before hashing, normalize the documented field set and remove the hash
field being computed; self-reference is never encoded.

`trace_id` hashes the normalized retained episode:

- `machine_ref` and semantic environment/action/observation fingerprints;
- reset-access mode and content-addressed setup root/predecessor evidence;
- entry physical evidence and phase, excluding run-local record numbers;
- ordered primitive action/phase rows, frame counts, declared-cadence
  observation/evidence digests, and terminal flags;
- termination/outcome, exit evidence, and retained cost; and
- the typed semantic post-entry intervention events, with local
  paths/timestamps excluded, plus `controller_realizable`.

It excludes timestamps, local paths, job/worker IDs, run-local record numbers,
repeat labels, wall time, peak memory, cache hits, discovery/search order,
`evaluation_work`, and attestation counts. Those fields remain required
scientific telemetry, but they are not the identity of the retained controller
episode. A whole-file artifact hash may bind the complete serialized envelope
separately.

The serial/cache/reorder/worker invariant compares the multiset of normalized
retained-result signatures for the same declared jobs. The result signature
uses `controller_sha256`, not `artifact_sha256`, and otherwise includes the
semantic environment/root, actions/phases/frames, outcome, exit evidence,
retained weight, and typed interventions. Provenance, telemetry, and scheduling
fields may differ. Exact evaluation counters are compared only in experiments
whose schedule/cache contract declares that they should be equal. Wall time and
memory are never required to be bit-identical.

`controller_sha256`, `artifact_sha256`, `trace_id`, the retained-result
signature, and `action_phase_sha256` each declare their normalization profile
and exclude their own field. Schemas must ship golden hash vectors before any
producer is accepted.

### 4.2 Setup legality

- `full_snapshot` may establish any content-addressed physical record inside
  the declared experimental domain.
- `root_only` may establish only a predeclared root from the job manifest.
- `no_arbitrary_reset` permits a canonical environment start or continuation
  from the actual live predecessor. It forbids an interior snapshot restore.

The validator rejects a setup method inconsistent with its access label.
Regardless of access mode, route composition still requires actual
parent-exit/child-entry continuity.

### 4.3 Physical identity and evidence modes

Every entry, exit, and lineage comparison names its evidence mode:

- `live_lineage`: the actual in-memory predecessor exit continues as the child
  entry without a restore;
- `canonical_snapshot`: equality under a backend serialization contract that is
  explicitly canonical and locally determinism-tested;
- `exact_artifact_restore`: restoration of the same content-addressed snapshot
  plus wrapper/context artifact; this identifies that experimental root but
  does not equate a different artifact; or
- `finite_restore_attestation`: declared RAM/wrapper observations and fixed
  suffixes agree after restore.

Only actual continuity or an explicitly assumed equality contract can discharge
the equality premise in the lifting theorem. A finite restore attestation is a
behavioral falsifier over its tested observations/suffixes; it never becomes
theorem-level physical equality. Stable-Retro/mGBA byte noncanonicity therefore
stays visible rather than being hidden behind the word “exact.”

## 5. Required invariants

1. **Phase continuity:** step \(t\)'s `phase_after` equals step \(t+1\)'s
   `phase_before`.
2. **Physical continuity:** each retained post-state is the next retained
   pre-state.
3. **Machine agreement:** the emitted action and next phase agree with the exact
   referenced machine.
4. **Invocation:** ordinary entry phase equals \(q_0\); final phase and reason
   satisfy the termination contract.
5. **Authoritative exit:** the executor's live final state defines the exit.
6. **Exact accounting:** `retained_cost.primitive_steps == len(steps)`.
   Emulator frames are separate and adapter-dependent.
7. **No probe laundering:** rolled-back, speculative, conformance, and
   attestation transitions never appear in `steps` or retained route cost.
8. **Intervention closure:** every state-changing operation after entry is
   typed and recorded.
9. **Controller realizability:** external roots, RAM writes, reset-based
   lookahead, symbolic transitions, or behavior-changing context mutations
   after entry force that episode's `controller_realizable=false`. An
   independent clean replay may produce a separate realizable trace; it does
   not erase the original intervention ledger.
10. **Cache knowledge is not control:** reading a cached artifact can change the
    knowledge tier but does not itself make execution unrealizable.
11. **No symbolic traces:** a symbolic manifest edge is not an `OptionTrace`.
12. **Lineage:** route composition requires the actual preceding exit to equal
    the next entry under an evidence/equality mode that discharges the claimed
    theorem premise. An unrelated representative from the same abstract block
    is never a substitute.
13. **Finite evidence:** repeat agreement and fixed suffixes are finite
    conformance evidence, not global determinism or equivalence proofs.

## 6. Cache and table identity

An execution cache key must bind at least:

\[
(\text{environment},\text{root record},\text{controller SHA-256},
\text{entry phase},\text{reset mode},\text{executor build}).
\]

Include seed/repeat identity whenever the declared environment contract makes it
semantically relevant. Repeated conformance executions remain separate
experiments; a cache hit must never replace a required repeat.

The current `mario-ai.option-observation.v2` table records option-boundary
outcomes. It is useful diagnostic evidence, but it lacks primitive
action/phase traces and is not a v3-certified quotient table.

For ordinary invocation, the declared finite physical option-boundary domain is
\(D_X\subseteq X\). Let \(B_{\mathrm{failure}}\) be a finite set of typed
failure symbols disjoint from \(X\), and define the table carrier

\[
\widetilde D=D_X\uplus B_{\mathrm{failure}}.
\]

Every machine \(o\) has its own phase set \(Q_o\), executes internally on
\(X\times Q_o\), starts at \(q_0^o\), and induces a partial physical-boundary
effect \(G_o:D_X\rightharpoonup\widetilde D\). A v3 table is exhaustive only
when every admissible `(representative, machine)` row is present or explicitly
disabled.

Interruption/resumption is outside the initial profile. If it is added later,
its boundary state is machine-tagged:

\[
\bigsqcup_{o\in O}X\times\{o\}\times Q_o,
\]

not \(X\times Q\) for one ill-defined global phase set.

Candidate members of one quotient block must agree on:

1. task/goal labels;
2. initiation/enabledness;
3. success/failure and termination label, including terminal phase only when it
   is part of that machine's declared output contract;
4. the declared exact retained weight;
5. successor quotient block.

Missing rows, repeat disagreement, conflicting evidence, unknown initiation, or
an unrecognized live boundary fail closed.

### 6.1 Weight contract

Every experiment/table declares and hashes one `weight_contract`:

- codomain \(W\), dimensions, units, and canonical numeric representation;
- identity \(0_W\) and associative additive operation \(\oplus\);
- exact componentwise equality, or an explicitly approximate comparator;
- scalar, lexicographic, or Pareto planning order; and
- nonnegativity or another well-founded condition whenever shortest-path
  optimality is claimed.

The initial recommended exact route weight is the nonnegative integer number of
retained emulator frames. Primitive transition calls may be a second reported
dimension. Discovery frames, probes, restores, search nodes, wall time, and
memory remain `evaluation_work`; they do not enter a route edge merely because
the legacy `OptionCost` can store them.

An enabled execution either reaches a physical successor in \(D_X\), reaches a
predeclared typed failure sink
\(\bot_{\mathrm{reason}}\in B_{\mathrm{failure}}\), or makes the table
incomplete. For totalized option-word semantics, failure symbols are formal
absorbing states: every remaining machine self-loops with weight \(0_W\), no
controller executes, and the original failure label is retained. This
bookkeeping does not assert that a failure symbol is a physical state in \(X\).
An interrupted harness run is not an admissible quotient edge. Disabled
initiation at a physical boundary is a missing/partial transition, not a
failure execution.

## 7. Theorem-sized claim

### 7.1 Controller-realizable concatenation

Assume fixed deterministic \(F\), exact machine semantics, complete primitive
traces, no unmodeled mutation, and actual parent-exit/child-entry equality under
the declared equality contract.
Induction over primitive rows, then over route segments, shows that concatenated
recorded actions form one legal primitive trajectory.

Artifacts attest the assumptions. A hash or finite suffix does not by itself
prove physical equality.

### 7.2 Finite weighted option quotient

Let \(D_X\subseteq X\) be a finite physical boundary domain,
\(B_{\mathrm{failure}}\cap X=\varnothing\) a finite typed failure set, and
\(\widetilde D=D_X\uplus B_{\mathrm{failure}}\) the finite carrier closed under
the physical effects and the formal zero-weight absorbing failure transitions.
Let \(O\) be a finite terminating machine library and
\((W,\oplus,0_W)\) the declared additive weight monoid.
Let an equivalence relation preserve task labels, machine enabledness,
termination/outcome, exact retained weight in \(W\), and successor equivalence
block for every admissible machine, including declared failure sinks. Then the
induced deterministic weighted quotient is well-defined and preserves
option-word reachability and accumulated weight inside \(\widetilde D\).

The proof is induction on option-word length. The base case preserves boundary
labels. The step uses enabledness and outcome agreement, equal edge weight, and
closure into the same successor block.

For an approximate scalar/vector weight, additionally declare a subadditive
norm \(\|\cdot\|\). If transitions, labels, and enabledness remain exact while
each retained edge differs by at most \(\epsilon\) in that norm, then a fixed
\(H\)-option lifted path differs by at most \(H\epsilon\). This is not a
policy-value bound outside the fixed path or finite domain.

This theorem does **not** establish:

- global ROM bisimulation or a globally coarsest quotient;
- `MetaState` Markov sufficiency;
- a stochastic SMDP result;
- causal abstraction;
- an equivalence oracle over an unbounded system;
- automatic state, predicate, phase, or option discovery; or
- correctness outside the enumerated domain and machine library.

Exact reset supplies membership-style experiments: execute a controller suffix
from a declared root and observe the result. It does not supply unrestricted
equivalence queries.

Query-efficient hypothesis discovery is separate from exact certification.
Report learner membership queries, hidden ground-truth scoring queries, and
certification/conformance queries independently. Zero error against a hidden
synthetic oracle evaluates recovery; unqueried behavior is certified only by a
declared complete finite conformance suite/fault-domain bound, symbolic proof,
or exhaustive validation.

## 8. Initial implementation profile

The first implementation should be additive and deliberately narrow:

1. add JSON Schemas and validation helpers for open-loop machines/traces;
2. add synthetic known-quotient fixtures with hidden delayed effects,
   noncommutative actions, variable costs, aliases, and explicit controller
   phase;
3. intercept and count primitive execution without changing legacy `Option`;
4. prove serial, cached, job-reordered, and worker-local normalized result
   signatures identical while reporting telemetry separately;
5. emit v3 traces for new experiments only; and
6. translate a Mario trace only after synthetic conformance passes.

The synthetic suite should fail on phase erasure, representative substitution,
probe-cost laundering, action-order changes, repeat suppression by cache,
unknown boundaries, and an intentionally spurious quotient merge.

## 9. Explicitly unimplemented on 2026-07-28

- No `OptionMachine` or `OptionTrace` Python class exists yet.
- No v3 JSON Schema or validator exists yet.
- Legacy `Option`, `OptionResult`, and planner behavior are unchanged.
- No primitive-action interception or phase trace is emitted.
- No feedback-FSC guard/predicate language exists.
- No option interruption/resumption or cross-option phase persistence exists.
- No stochastic controller, duration distribution, or success model exists.
- No phase-aware online frontier merge exists.
- No automatic predicate, phase, or option discovery exists.
- Existing SMA4 external-root, write-assisted, or symbolic runners are not
  promoted.
- `mario-ai.option-observation.v2` and
  `sma4-physical-option-benchmark-v2` retain their current names and scope.

This separation is intentional. A precise unimplemented contract is a better
research foundation than silently upgrading the meaning of historical
artifacts.
