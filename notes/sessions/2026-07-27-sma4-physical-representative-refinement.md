# 2026-07-27 — SMA4 physical representatives and option refinement

> Evidence-first continuation of the July 25 audit and the July 26 boundary
> integrity pass. Literature availability is cut off at **2026-07-25**.
> Implementation and verification continued on July 26–27 in America/Detroit.
> Observations come before interpretation and planning.
>
> **Corrected later on 2026-07-27:** the reported acquisition-order
> responsiveness split came from reading an abandoned fixed cursor object.
> Both histories pass at maximum Tier 2 when the live pointer is followed; see
> `2026-07-27-sma4-live-cursor-pointer-tier2.md`. The multi-representative and
> finite-refinement architecture remains valid, while the stuck-cursor
> diagnosis and P1/P2 repair plan are superseded.

## 1. Question

The preceding pass proved that the old one-snapshot-per-`MetaState` contract was
unsound. Opposite whistle-acquisition orders reached the same symbolic state but
different emulator states, so strict search failed closed with
`StateAliasError`.

This pass asks:

1. Can search retain both physical histories without silently borrowing or
   replacing snapshots?
2. Can every retained representative be tested against the same finite option
   library and refined by observed behavior?
3. Does the result remain invariant to planner and option insertion order?
4. Does the formerly hidden asymmetry reveal a useful next obstacle rather than
   merely moving the alias error?

The answer to the first two questions is yes. The answer to the fourth is also
yes: the two orders now survive through the second whistle and separate at a
real input-responsiveness test on the World-8 map. The final insertion-order and
Tier-2 acceptance runs are listed below and are part of this pass's final gate.

## 2. Evidence before interpretation

### 2.1 ROM-free verification

The focused architecture suite after adversarial review is:

```bash
./venv/bin/python -m pytest -q \
  tests/test_option_provenance.py tests/test_meta_planner.py tests/test_options.py
# 41 passed
```

The canonical verification reports:

```text
determinism/snapshot gate: 6 passed
remaining suite: 157 passed, 10 skipped
total: 163 passed, 0 failed, 0 errors, 10 skipped
```

The new regressions cover:

- preserving multiple raw-distinct representatives under one `MetaState`;
- exact-duplicate deduplication and arrival provenance;
- explicit-record restoration and rejection of ambiguous state-only restore;
- a symbolic edge being unable to borrow another lineage's physical snapshot;
- BFS and bounded uniform-cost search over physical record identities;
- Pareto cost/depth labels for depth-bounded uniform-cost search;
- outcome-repeat mismatch and physically different exit-repeat mismatch;
- the latter even when an executor exposes only a coarse observable;
- noncanonical byte artifacts that agree on RAM and a fixed suffix;
- delayed successor-block refinement and distinguishing option suffixes;
- duplicate-row conflict detection independent of input order; and
- World-8 rejection for a stuck cursor and for autonomous motion that would
  fool a direction-only probe.

Three independent read-only reviews checked code, report semantics, and
documentation scope. The final physical-planner review found no remaining
correctness blocker after the coarse-observable regression was fixed.

The ROM-backed adapter/acquisition gate passes 7/7, and the stock replay gate
remains 30/32 with 6-2 and 6-3 unsolved/quarantined.

### 2.2 Normal-order live Tier-3 candidate

Command:

```bash
MARIO_AI_SMA4_ROM=... ./venv/bin/python \
  scripts/bench_sma4_whistle_rom.py \
  --warpless-blocked --no-hand-grant --max-tier 3 \
  --out runs/20260726-sma4-multi-representative-tier3-final-candidate
```

Report:
`runs/20260726-sma4-multi-representative-tier3-final-candidate/report.json`.

| Measurement | BFS | Uniform cost |
|---|---:|---:|
| found mixed goal | true | true |
| reported mixed plan cost | 13,320 | 13,320 |
| physical representatives encountered | 12 | 12 |
| total search keys, including symbolic goal | 13 | 13 |
| enabled transition branches | 13 | 13 |
| option executions, two per enabled branch | 26 | 26 |
| repeat-conformance failures | 0 | 0 |
| complete finite option table | true | true |
| closed under modeled physical successors | true | true |
| refinement iterations | 2 | 2 |

Both planners select:

```text
acquire_whistle_1_3
-> acquire_whistle_fortress
-> use_whistle
-> use_whistle_again
-> select_world8_pipe
-> clear_bowser
```

The reported 13,320 is a **mixed planner cost**:

- 7,320 reported frames belong to the five emulator-backed options; and
- 6,000 frames belong to the symbolic `clear_bowser` stand-in.

It is not a continuous full-ROM completion time. The fortress option's legacy
reported cost also includes a small rolled-back menu probe. The report now
separates repeated reported work from rolled-back probe/attestation work, but a
future cost-contract cleanup is still required.

Greedy is stuck because the symbolic warpless route is disabled. Therefore
there is no matched positive greedy cost and no defensible speedup ratio:
`skip_cost_reduction_x=null`.

### 2.3 Both acquisition orders survive; only one passes the World-8 gate

The search expands one second-whistle representative for each order:

| Acquisition order | Raw destination | Whistles | Cursor | Matched-NOOP input response | Accepted |
|---|---:|---:|---:|---:|---:|
| 1-3 then fortress | world 7 | 0 | `(32,80)` | responsive | true |
| fortress then 1-3 | world 7 | 0 | `(128,144)` | unresponsive | false |

Both terminal states satisfy the superficial conditions: raw World 8, overworld
mode, and empty whistle inventory. Acceptance additionally requires directional
input to produce a trace different from a same-length NOOP control. Each probe
restores the exact exit state, records its trace hash, and rolls all diagnostic
frames back. This prevents a naturally moving cursor from being mistaken for
input responsiveness.

The failed order is a result about the **current option implementation and
bounded immediate probe**, not proof that no legal recovery sequence exists.

### 2.4 The option table exposes two useful distinguishing suffixes

The two physical states after the first whistle share a `MetaState`, but the
single-option suffix:

```text
use_whistle_again
```

distinguishes them.

The two states after the second whistle again share a `MetaState`, but:

```text
select_world8_pipe
```

distinguishes them.

These are machine-generated counterexample suffixes from the finite table, not
hand-written explanations. They identify exactly where the current symbolic
quotient loses predictive information.

All 12 resulting refined classes are singletons in this run. Matching block-ID
sets across BFS and uniform-cost are useful implementation evidence, but no
physical-state compression has yet been earned.

### 2.5 What the refinement gate does and does not say

The report's `empirical_conformance_gate=true` means only:

1. all encountered physical representatives have a row for every option in this
   library;
2. every enabled option was run twice;
3. normalized outcomes and finite physical-exit attestations matched within
   each repeated execution;
4. every accepted physical successor is represented in the table;
5. no duplicate `(record, option)` rows conflict; and
6. iterative successor-block refinement reached a fixed point.

It does **not** prove:

- global bisimulation;
- determinism for all future executions;
- equivalence under every primitive input sequence;
- completeness outside the encountered roots, depth, tier, ROM, core, and
  option implementation; or
- that any two currently distinct records may be merged online.

The current refinement is a post-run report. Live frontier dominance remains
per exact in-run physical record, not per refined class.

## 3. Architecture implemented

### 3.1 Strict compatibility mode remains strict

`OptionContext(alias_policy="error")` preserves the historical safety contract.
If raw-distinct candidates claim one `MetaState`, it raises
`StateAliasError`. Legacy `remember()` also remains strict so an old call site
cannot accidentally opt into collapse.

### 3.2 Multi-representative mode

`OptionContext(alias_policy="multi")` maintains:

```text
physical_records[record_id] -> exact snapshot record
record_ids_by_state[MetaState] -> ordered record IDs
exact_record_index[(MetaState, full_artifact_sha256)] -> record ID
record_arrivals[record_id] -> roots and parent/option arrivals
```

Raw-identical candidates deduplicate and add arrival provenance. Raw-distinct
candidates receive separate immutable in-run record IDs and emit an alias
diagnostic with `resolution=retained_both`.

A `None` physical ID explicitly means a symbolic lineage. Physical options fail
closed from it. This is essential: falling back from a symbolic state to the
first snapshot with the same `MetaState` would silently splice histories.

### 3.3 Physical search keys

The new opt-in planner uses:

```text
SearchNodeKey = (MetaState, physical_record_id | None)
```

BFS visits this key directly. Depth-bounded uniform-cost search keeps all
nondominated `(cost, depth)` labels for a physical key. A cheaper but deeper
arrival cannot dominate a costlier shallow arrival when only the shallow path
has enough depth budget left. Expansion results are cached per physical key so
multiple nondominated labels do not rerun and recommit the same physical
options.

Option IDs are sorted before expansion, so dictionary insertion order is not a
semantic input. Negative step costs are rejected.

### 3.4 Repeat conformance

Each enabled live benchmark transition executes twice from the exact parent
record. Equality includes:

- applicability, success, and termination;
- accepted successor `MetaState` and goal label;
- reported and retained duration;
- runtime-effective knowledge tier;
- terminal invariants;
- semantic interventions, excluding file/record identity;
- adapter context;
- emulated RAM evidence; and
- a fixed forward suffix when the executor supports it.

If no forward-suffix API exists, exact snapshot-artifact identity is the
conservative fallback even when a coarse observable exists. For local mGBA,
RAM plus the fixed suffix is used instead of raw serialization equality because
the serialization payload is known to contain noncanonical bytes.

Source/action file hashes and full boundary records remain in provenance. They
are deliberately excluded from behavioral equality: repackaging identical
behavior in a different file must not split a behavioral class.

### 3.5 Fixed-point partition refinement

The initial color is `(MetaState, goal-label)`. For each option, a record's row
contains applicability, repeat conformance, normalized outcome, and the current
successor block. Colors are recomputed until the induced partition stops
changing.

The report includes:

- content-addressed block IDs scoped by option library, tier, repeats, and
  depth;
- missing rows as explicit `unobserved` rows;
- conflict hashes for incompatible duplicate observations;
- closure and repetition gates; and
- recursively generated distinguishing option suffixes.

Content-addressed IDs depend on behavior and declared experiment scope, not
run-local record numbers or raw snapshot hashes.

### 3.6 Report-level reproducibility

The benchmark report now records:

- UTC creation time and exact CLI argument vector;
- Git commit, dirty flag, porcelain status, tracked diff hash, untracked source
  list, and a combined working-source hash;
- Python/platform and Stable-Retro versions;
- SHA-256 for the installed Stable-Retro extension and mGBA core binary; and
- ROM basename and SHA-1.

The final reports must show a clean tree and the commit created by this pass.
Ignored run outputs do not dirty the source tree.

## 4. Why raw mGBA snapshot inequality is not physical inequality

The local stack is Stable-Retro 1.0.1 with its vendored mGBA 0.7.0-era core.
Stable-Retro allocates the Python bytes destination with
`PyBytes_FromStringAndSize(NULL, size)` before asking the core to serialize.
Python documents that this leaves the byte contents uninitialized. The vendored
mGBA serializer writes its declared state fields but does not first clear the
whole destination.

A no-step local reproduction loaded
`runs/sma4_cache/1-fortress_real_entry.pkl` and immediately sampled five
savestates. It produced three raw hashes. Pairwise differences ranged from 256
to 643 bytes, every difference was below offset `0x800`, and there were zero
differences at or above `0x800`.

This supports a narrow implementation diagnosis:

- equal raw serialization bytes are a strong exact-artifact identity witness;
- unequal raw bytes need not mean unequal emulated state;
- full artifact hashes remain useful for safe deduplication and provenance;
- behavioral merging requires independent RAM/context/suffix evidence; and
- finite suffix agreement remains a falsifier, never a universal proof.

Relevant source:

- Stable-Retro 1.0.1
  [`src/retro.cpp`](https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/src/retro.cpp#L51-L55);
- Python
  [`PyBytes_FromStringAndSize`](https://docs.python.org/3/c-api/bytes.html#c.PyBytes_FromStringAndSize);
- vendored mGBA
  [`retro_serialize`](https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/cores/gba/src/platform/libretro/libretro.c#L614-L631);
- vendored mGBA
  [GBA I/O serialization](https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/cores/gba/src/gba/io.c#L918-L945).

## 5. What is established

1. The state-only frontier obstacle is removed in an opt-in planner without
   weakening strict legacy behavior.
2. Both acquisition histories survive and are independently expanded.
3. Symbolic successors cannot borrow physical state.
4. Exact duplicate physical records deduplicate with all arrivals preserved.
5. Repeated normalized outcomes alone are insufficient; physical-exit evidence
   is now part of the gate.
6. The current finite table is complete and closed over 12 encountered
   representatives for this library/tier/depth.
7. The acquisition orders are behaviorally distinct under short option
   suffixes.
8. A viable Tier-3 mixed path exists for 1-3-then-fortress, while the reverse
   order exposes a stuck World-8 map.

## 6. What is not established

1. No representative pair has earned a live-frontier merge; all current classes
   are singletons.
2. Two repeats are evidence of repeat conformance, not a determinism proof.
3. The successful path still restores independent acquisition roots and is not
   a continuous lineage from reset.
4. Acquisition executors still contain declared intervention/context handling.
5. `use_whistle_again` is runtime Tier 3 because four cursor writes remain.
6. `clear_bowser` is symbolic, so 13,320 is not a ROM completion time.
7. The reverse-order failure covers the current script and immediate probe, not
   every legal recovery policy.
8. The greedy baseline is intentionally nonexploring and is stuck; no speedup
   claim follows.
9. Reported legacy option cost is not yet identical to retained physical
   duration.
10. Root manifests hash declared action payloads, not the exact dynamic input
    stream after executor-side settling, truncation, and probes.

## 7. Research synthesis at the July 25 cutoff

### 7.1 Direct mathematical backbone

Ravindran and Barto's
[SMDP homomorphisms](https://www.ijcai.org/Proceedings/03/Papers/145.pdf)
remain the closest formal target: an option-level quotient must preserve the
relevant transition and reward/duration model. The present implementation is a
finite deterministic approximation to that test, not a proof of an SMDP
homomorphism.

Givan, Dean, and Greig's
[MDP equivalence and model minimization](https://www.sciencedirect.com/science/article/pii/S0004370202003764)
clarifies why recursive successor-block agreement matters. Castro, Panangaden,
and Precup's
[trace/bisimulation analysis](https://www.ijcai.org/Proceedings/09/Papers/276.pdf)
also matters: finite traces can be useful tests, but recursive fixed-point
structure carries a stronger claim. That is why this report says finite
empirical conformance rather than bisimulation.

Paige and Tarjan's
[partition-refinement algorithms](https://epubs.siam.org/doi/abs/10.1137/0216062)
are the scaling path if the current simple fixed-point pass becomes expensive.
The current graph has only 12 representatives, so clarity is preferable today.

### 7.2 Skill-conditioned abstraction

Ahmetoglu et al.,
[“Skill-Driven Neurosymbolic State Abstractions”](https://papers.nips.cc/paper_files/paper/2025/hash/0fa694fb9f1e265117e8da75966820fe-Abstract-Conference.html),
derive abstractions around a supplied abstract-action set and explicitly
characterize Markov/model-preserving conditions. This is unusually close to the
repo's actual problem. The next useful comparison is:

- treat each exact local representative as a delta distribution;
- include option initiation, transition, reward/duration, and goal labels;
- ask which features split the two whistle histories; and
- test whether the learned/symbolic feature is stable outside these cached
  roots.

The paper does not license replacing exact local representatives with a learned
encoder before the empirical state contract is sound.

### 7.3 Active tests and counterexamples

Angluin's
[L* query/counterexample framework](https://www.sciencedirect.com/science/article/pii/0890540187900526)
suggests a stronger workflow than blindly lengthening a fixed suffix:

1. hypothesize that two records share a state;
2. search for an option/input suffix that distinguishes them;
3. split on the counterexample; and
4. repeat until the budgeted oracle finds none.

Wißmann, Milius, and Schröder's
[distinguishing-formula construction](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CONCUR.2021.32)
suggests returning a compact human-readable witness with every split. The
current `distinguishing_option_suffixes` field is the smallest first step.

James and Singh's
[predictive state representations with reset](https://icml.cc/Conferences/2004/proceedings/papers/117.pdf)
provide another useful lens: state can be represented by predictions of
observable tests. Here resets and exact snapshots make such tests unusually
cheap. A future predictive signature should be treated as a learned test basis,
not assumed sufficient without held-out counterexamples.

### 7.4 Closest 2026 systems result

Giraud et al.'s
[L-SCALE](https://publikationen.bibliothek.kit.edu/1000195438), published
July 20, 2026, combines L*-style automata learning with an emulator offering
deterministic execution and incremental snapshots. This is the closest new
systems analogue found before the cutoff.

What to borrow:

- incremental snapshot-backed membership tests;
- suffixes generated by counterexamples;
- explicit control over abstraction coarseness; and
- a learned state machine as an inspectable artifact.

What not to borrow yet:

- locality-sensitive hashing as authority to merge physical game states.
  L-SCALE itself notes threshold sensitivity. A false merge here can fabricate
  a route, so approximate hashes may prioritize tests but cannot pass the
  safety gate.

### 7.5 Unknown option effects

Percassi, Saetti, and Scala's
[Planning with Uncertain Action Models](https://ojs.aaai.org/index.php/AAAI/article/view/40954)
(PUMA, AAAI 2026) studies actions whose true effect becomes known after first
execution and can then be reused. This is a valuable control for the
execute-to-observe planner.

Its assumptions are stronger than this repo currently earns. An option's
effect here can depend on hidden physical lineage, and discovery from one
representative does not authorize reuse from another. The direct experiment is
therefore a lineage-indexed PUMA-like control:

- cache an effect only within a refined physical class;
- invalidate or split the class on a counterexample; and
- compare discovery calls and final path cost separately.

## 8. Architectural improvements worth exploring

Ordered by expected scientific value:

1. **Counterexample-guided suffix search.** Generate minimal option or primitive
   input suffixes that split candidate-equivalent records.
2. **Lineage-stable identity.** Add a content-addressed lineage hash over root
   identity, exact executed input transcript, adapter-context changes, and
   interventions. Keep in-run record IDs for references but do not mistake them
   for reproducible identities.
3. **Executed-input provenance.** Hash the actual dynamic input stream, including
   executor-side settling and probes, separately from the source manifest.
4. **Vector option cost.** Record retained physical frames, rolled-back
   evaluation frames, nodes, restores, and intervention tier separately.
   Scalar planner cost should be an explicit projection.
5. **Adaptive repeat policy.** Two repeats are the floor. Increase repetitions
   on conflicts, backend changes, new roots, or proposed merges; preserve every
   outcome rather than majority-voting away a mismatch.
6. **Online refinement only after a holdout gate.** The current report may
   propose merges. A second, independently generated suffix set must fail to
   distinguish the records before refined-class frontier dominance is enabled.
7. **Incremental refinement.** If the graph grows, update only affected blocks
   and then consider Paige–Tarjan-style worklist refinement.
8. **Learned abstraction last.** Once exact representatives and counterexamples
   exist, train a feature predictor to propose partitions; retain exact
   falsification and replay as authority.

## 9. Detailed next-step plan

### P0 — Close this implementation and reproducibility gate

Actions:

1. Run normal and reverse option-insertion Tier-3 benchmarks from the final
   source.
2. Run the same warpless-blocked benchmark at maximum Tier 2.
3. Require BFS/UCS agreement on reachability, selected path, history outcomes,
   content-addressed block set, completeness/closure fields, and absence of
   repeat-conformance failures.
4. Require both acquisition orders to be expanded exactly once at
   `select_world8_pipe`.
5. Run `verify_iteration.py`, the 30/32 stock replay gate, compilation, and
   `git diff --check`.
6. Commit and push code plus documentation. ROMs, runs, roots, and savestates
   remain ignored.

Artifacts:

```text
runs/20260727-sma4-multi-representative-tier3-final/report.json
runs/20260727-sma4-multi-representative-tier3-reverse-final/report.json
runs/20260727-sma4-multi-representative-tier2-final/report.json
runs/tests.json
```

Stop condition: any planner/order disagreement, incomplete table, conflicting
row, repeat mismatch, or dirty-source final report blocks acceptance.

### P1 — Diagnose the two second-whistle lineages

Actions:

1. Capture both representatives before/after `use_whistle_again`.
2. Diff all emulated RAM blocks, adapter/wrapper context, input transcripts,
   menu state, timer/IRQ-sensitive values, and cursor transition traces.
3. Use delta debugging over the preceding input/settle sequence to find the
   shortest history change that flips responsiveness.
4. Run automatically generated primitive suffixes from both states, beginning
   with directional holds, menu open/close, `B`, `A`, and bounded NOOP timings.
5. Record a minimal distinguishing suffix and the first divergent observable.

Acceptance artifact:

```text
runs/<ts>-sma4-second-whistle-lineage-diff/report.json
```

It must include exact parent records, executed-input hashes, per-frame
observable traces, RAM-delta grouping, and a declared search budget.

Decision:

- if legal input makes both histories responsive, incorporate that input-only
  transition;
- if only one history is viable within the budget, retain distinct refined
  classes and plan through the viable one;
- if a stable RAM/context feature predicts the split across new roots, add it
  to `MetaState` only after held-out verification.

### P2 — Remove the Tier-3 cursor repair

Actions:

1. Delete or bypass the four second-whistle cursor writes on an experimental
   branch.
2. Replace them with the P1 input-only settle/navigation sequence.
3. Re-run both histories with `--max-tier 2`.
4. Require zero runtime Tier-3 events on the accepted route.

Success:

```text
use_whistle_again effective_tier <= 2
direct cursor RAM writes = 0
World-8 cursor responsive under matched-NOOP probe
```

If no legal sequence is found under the declared budget, retain Tier 3 and
state the limitation. Do not disguise the write as initialization.

### P3 — Join the live unpowered fortress lineage to a legitimate power source

Actions:

1. Test whether the fortress whistle has any small-Mario route.
2. If flight is required, identify a real P-Wing/leaf inventory source and use
   it through controller input from the live post-1-2 map state.
3. Re-enter the fortress at `(96,96)` with the resulting legitimate power.
4. Search/replay the whistle segment without an unrelated fortress root,
   P-speed seed, or power rehost.

Success:

- one declared earliest root;
- no external root after it on the accepted lineage;
- no direct memory assignment;
- exact executed-input transcript at every option;
- replay-verified whistle inventory transition.

### P4 — Make the whole two-whistle route one physical lineage

Actions:

1. regenerate or recipe-pin the earliest practical root;
2. acquire both whistles from exact predecessor exits;
3. spend both whistles and select World 8 with input-only transitions;
4. remove symbolic physical-state borrowing by construction; and
5. independently replay the complete accepted input history.

Required report fields:

```text
external_roots_after_declared_root = 0
direct_memory_assign_calls = 0
symbolic_edges_before_world8 = 0
alias_collapses = 0
runtime_tier <= declared experiment tier
full executed-input SHA-256
```

### P5 — Replace symbolic endpoints and strengthen planner controls

Actions:

1. Replace `clear_bowser` with a ROM-backed endpoint.
2. Replace/remove symbolic warpless world-advance options.
3. Give greedy, BFS, UCS, an optimistic/UCB explorer, and a PUMA-like
   effect-cache control the same option library and cost projection.
4. Report exploration calls, retained route cost, total evaluation work, and
   wall time separately.
5. Use matched budgets and repeat across roots.

Interpretation:

- if perfect-effect greedy matches UCS, the contribution is unknown-effect
  discovery;
- if history-aware search still wins, quantify why;
- if no positive control gap remains, keep the architecture result and retire
  the speedup thesis.

### P6 — Scale the abstraction experiment

After P1–P5:

1. generate active distinguishing suffixes;
2. hold out some suffixes as merge falsifiers;
3. enable refined-class frontier dominance only after the holdout gate;
4. compare simple fixed point with incremental/Paige–Tarjan refinement;
5. measure representatives, option calls, false proposed merges, nodes, and
   wall time; and
6. test on more roots and another adapter-backed Mario game before calling the
   abstraction reusable.

### P7 — Return to the broader research program

In parallel only when the SMA4 physical contract is stable:

1. solve/quarantine SMB1 6-2/6-3 with phase-aware and variable-duration search;
2. run completeness-safe policy-guided search across level types;
3. preserve checkpoint/data provenance and paired solve-rate gates; and
4. benchmark an alternative GBA core only under the same clone/step/replay
   contract.

## 10. One next action

**Diff the two second-whistle physical lineages, generate the shortest
input-only suffix that explains or repairs the responsiveness split, remove the
Tier-3 cursor writes from the viable route, and rerun the warpless-blocked
benchmark at maximum Tier 2.**
