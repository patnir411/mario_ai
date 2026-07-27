# 2026-07-26 — SMA4 option-boundary integrity and live fortress entry

> Evidence-first follow-up to the July 25 project audit. The requested research
> date was July 25, 2026; execution continued into July 26 in America/Detroit.
> This note records observations before interpretation and planning.
>
> **Correction, 2026-07-27:** the Tier-3 cursor repair and unresponsive reverse
> lineage recorded here were consequences of reading a fixed, abandoned cursor
> object. Following the live pointer removes the repair and makes both orders
> responsive at maximum Tier 2. See
> `2026-07-27-sma4-live-cursor-pointer-tier2.md`. The boundary-provenance and
> live-fortress findings remain valid.

## 1. Questions tested

The July 25 audit left one exact action:

1. identify every physical root, restore, RAM write, and intervention used by the
   SMA4 option planner;
2. hash both sides of each option boundary;
3. fail closed when a physical option has no declared predecessor snapshot or
   when one `MetaState` aliases multiple physical emulator states; and
4. attempt the live post-1-2 overworld route to the real World-1 Fortress at
   `(96,96)` without an unrelated fortress-entry restore.

This pass answers those questions. It does **not** yet establish a continuous
two-whistle World-8 route.

## 2. Evidence summary

Two results matter more than the implementation volume.

### 2.1 The live fortress-entry obstacle is broken

From the single declared local root
`runs/sma4_cache/1-2_entry.pkl`, the diagnostic:

1. replays all 905 actions in `data/solutions/sma4/1-2.json`;
2. settles to a responsive overworld at `(128,32)`;
3. walks the explicit route `DOWN, DOWN, LEFT`;
4. reaches `(96,96)` naturally; and
5. enters the real fortress as small Mario, with no direct emulator-memory
   assignment and no unrelated trajectory root.

The exact command is:

```bash
MARIO_AI_SMA4_ROM=... ./venv/bin/python \
  scripts/diagnose_sma4_fortress_entry.py \
  --forbid-memory-writes --no-promote \
  --out runs/20260726-sma4-live-fortress-entry
```

Primary report:
`runs/20260726-sma4-live-fortress-entry/report.json`.

Observed in two independent repetitions:

| Measurement | Result |
|---|---:|
| semantic success | `true` |
| all seven boundary emulator/full/RAM hashes equal across repeats | `true` |
| final exact emulator SHA-256 | `9173ec00a0145cd36e6a2afc84a376fcf8828844a41eb5e6e98c5663c3bf9762` |
| final full snapshot SHA-256 | `6de602eb346915efde35b2060033907aee6fe37593def9d224159ab6a8e744f9` |
| retained forward-lineage frames | 1,733 |
| rolled-back diagnostic probe frames | 462 |
| total emulator steps | 2,195 |
| restore calls in the measured trajectory | 35 |
| unknown restore targets | 0 |
| direct `Memory.assign` calls | 0 |

The accounting identity is exact:

\[
1{,}733\ \text{retained} + 462\ \text{rolled back}
= 2{,}195\ \text{emulator steps}.
\]

The 35 restores are one declared-root restore plus 34 same-state,
non-destructive probe rollbacks used by map settling and level-entry
controllability checks. The trace is therefore a single committed physical
lineage, but it is not literally restore-free. The diagnostic enforces the
zero-write claim at runtime by replacing `Memory.assign` with a rejecting
wrapper for the whole trajectory.

Final decoded entry invariants are:

```text
mode=level
cursor=(96,96)
x=24
y=112
time=196608
powerup=0
pspeed=0
```

The root file is pinned by SHA-256
`c420e450ab78ba87cbf092f8f79dbd96b4c2858a50b2b741a0e768fe855671f1`;
the 1-2 solution file is pinned by
`380b16d120b47336df9f3b9382f4048198bdc344fac0da6986acfd8637b08cba`.

### 2.2 The old two-acquisition route is not valid under the new state contract

When both cached whistle-acquisition options are admitted at Tier 3, BFS and
uniform-cost search reach the same symbolic warp-zone state through opposite
acquisition orders:

```text
MetaState(
  world=9,
  node=(64,80),
  inventory=("whistle",),
  flags={
    first_whistle_spent,
    two_whistles_acquired,
    warp_zone,
    whistle_acquired_1_3,
    whistle_acquired_fortress,
  },
)
```

The physical successors are not the same. Their exact emulator hashes, adapter
contexts, RAM-block hashes, and screens differ. Their recorded parent states
also expose the acquisition-order distinction: one first-whistle execution
starts from map node `(96,96)`, the other from `(14,135)`. The current report
retains both exact hashes; this durable note does not promote those raw values
to cross-process canonical IDs because mGBA reserialization can change them.
The planner raises `StateAliasError` instead of selecting whichever
representative was inserted first. Uniform-cost finds the same abstraction
failure with the retained/rejected order reversed.

This is a constructive counterexample to the current `MetaState` abstraction:
one symbolic state is not sufficient to determine the physical option process.
The previous 5.18x two-acquisition headline is therefore superseded as an
accepted-route result. Its underlying segments remain useful, but their
composition is not established.

## 3. Fortress-reference comparison

The new live entry is semantically the same kind of unpowered fortress spawn as
`runs/sma4_cache/1-fortress_real_entry.pkl`, but it is not the same physical
state:

| Representation | Emulator SHA-256 |
|---|---|
| historical file bytes | `b4f0c8587edcc16ca98d225e6a8806fb68b9ab4df54bc3da269dfb1b7fc4da12` |
| historical state after restore/resnapshot | `2f5afc3dde0a59bf2f2ab34dada62bee4c501438b62113cbce1372d8e0814963` |
| new live entry | `9173ec00a0145cd36e6a2afc84a376fcf8828844a41eb5e6e98c5663c3bf9762` |

All compared RAM blocks except IWRAM match between the live and restored
historical entries; 256 bytes differ in IWRAM. The report preserves that
mismatch instead of promoting or repairing the cache.

This comparison also exposed a backend property: Stable-Retro/mGBA savestate
bytes are not byte-canonical across load/resnapshot. Raw hashes remain exact
identity witnesses, but raw inequality alone is not proof of behavioral
inequality.

For a restore of the *same recorded source*, the implementation can separately
label `restored_suffix_attested` when all recorded emulated-RAM blocks, wrapper
metadata/context, and a fixed four-NOOP forward suffix agree. The screen hash is
retained as diagnostic evidence but is not an equality gate because
Stable-Retro's display buffer is a host-side cache. The suffix is a falsifier
and finite behavioral attestation, not a proof that every future input sequence
agrees. Cross-route raw/RAM-distinct states are never excused by that
attestation; they remain hard aliases.

## 4. Option benchmark matrix after instrumentation

All reports are local ignored artifacts:

- `runs/20260726-sma4-whistle-provenance-hand-grant-tier1/report.json`
- `runs/20260726-sma4-whistle-provenance-hand-grant-tier3/report.json`
- `runs/20260726-sma4-whistle-provenance-acquire-tier2/report.json`
- `runs/20260726-sma4-whistle-provenance-acquire-tier3/report.json`

| Library / gate | Greedy | BFS / uniform-cost | Interpretation |
|---|---|---|---|
| hand grant, max Tier 1 | symbolic warpless, 69,000 | symbolic warpless, 69,000 | second whistle needs a runtime Tier-3 cursor repair, so no whistle skip is accepted |
| hand grant, max Tier 3 | symbolic warpless, 69,000 | whistle path, 8,966 | 7.7x mixed benchmark, but hand grant + cursor repair + symbolic Bowser remain |
| two cached acquisitions, max Tier 2 | symbolic warpless, 69,000 | symbolic warpless, 69,000 | attempted whistle route exceeds the tier gate |
| two cached acquisitions, max Tier 3 | symbolic warpless, 69,000 | hard `StateAliasError` | acquisition order is physically material and the abstraction collapses it |

The accepted hand-seeded Tier-3 path is:

```text
grant_two_whistles
-> use_whistle
-> use_whistle_again
-> select_world8_pipe
-> clear_bowser
```

The first four transitions execute against the emulator. `clear_bowser` remains
a symbolic planning-layer stand-in. The second-whistle option is declared
Tier 1 but is now correctly measured as effective Tier 3 because it performs
four logged cursor write calls; one actually changes RAM
(`Y: 80 -> 144`). At max Tier 1, the transition fails with
`runtime_knowledge_tier_exceeds_max`.

The report's top-level `found=true` at lower tiers means the symbolic warpless
fallback remains available; it must not be read as a successful whistle route.

## 5. Boundary contract implemented

### 5.1 Deterministic provenance

`mario/provenance.py` adds:

- stable, typed serialization for metadata;
- SHA-256 over exact byte-backed emulator state;
- separate emulator, wrapper-metadata, adapter-context, and combined hashes;
- file hashes for roots and action manifests; and
- a structured `StateAliasError`.

It deliberately has no `repr`, Python `hash()`, generic pickle, or silent
fallback for unsupported snapshot types.

### 5.2 Snapshot ownership and restoration

Physical options now:

- declare that they require a snapshot;
- fail closed before the runner executes when the predecessor root is absent;
- restore exactly once inside `Option.execute`;
- capture the live exit after the runner returns;
- commit snapshots only for accepted transitions; and
- retain producer/source identity for every stored representative.

Failed or unselected trials no longer poison later search by silently replacing
the accepted representative.

### 5.3 Physical versus symbolic evidence

Every selected path edge is labeled as either:

- `physical_executor`, with a boundary report; or
- `symbolic_model`, explicitly without physical boundary evidence.

The planner result separately propagates:

- final-path versus merely attempted injected facts;
- declared versus runtime-effective knowledge tier;
- exact entry and exit hashes;
- decoded physical world/node/inventory/mode;
- symbolic-versus-decoded checks;
- external roots;
- every RAM write, including no-op writes;
- solution file and declared manifest action-payload hashes; and
- adapter metadata/context writes.

`0x03002C52` is no longer compared with `MetaState.cleared`: the address is
exposed only as opaque `panel_slots_raw`, consistent with the earlier RAM-map
correction.

### 5.4 Inventory multiplicity

`MetaState.inventory` is now a sorted multiset. One whistle and two whistles are
different states, and consuming a whistle removes one copy rather than the
entire item kind.

### 5.5 Intervention truth

The executor records each memory write with address, pre-value, requested value,
observed post-value, reason, frame, whether it changed state, and knowledge
tier. Cached-root loads record path/file hash, expected/actual/repeated raw
digests, RAM/display observables, and fixed-suffix attestation.

The fortress manifest's historical label
`pspeed_poke_during_fly` is corrected to
`pspeed_seeded_in_entry_snapshot`: the current executor does not perform a live
P-speed write on that route.

## 6. What is now established

1. The formerly opaque map obstacle was a route-selection error, not a missing
   unlock bit: the natural path from the responsive post-1-2 state is
   `DOWN, DOWN, LEFT`.
2. A deterministic, zero-direct-write committed lineage reaches the real,
   unpowered fortress from one declared 1-2 root.
3. Option restores, roots, action artifacts, RAM writes, and effective
   intervention tiers can now be audited at boundary granularity.
4. The hand-seeded whistle spend reaches World 8 only under a Tier-3 cursor
   intervention in the current implementation.
5. The two cached acquisitions expose a real `MetaState` alias; preserving one
   representative is unsound and can make search order-dependent.

## 7. What is not established

1. The declared 1-2 root has not been regenerated from reset inside this trace.
   Its content hash pins identity but is not a clean-clone generation recipe.
2. The cached root carries stale 1-1/stage-1 wrapper labeling; the diagnostic
   explicitly relabels adapter metadata to 1-2/stage 2 before restore. This is
   not a RAM write, and the complete 1-2 action replay succeeds, but it is
   another reason not to claim reset-to-root provenance.
3. The trace stops at an unpowered fortress spawn. The current fortress-whistle
   solution starts from a different P-Wing/leaf root and cannot consume this
   state.
4. Probe rollbacks mean the diagnostic is not literally restore-free, although
   it has one committed root and no alternate trajectory root.
5. The historical fortress reference is not physically identical to the new
   live entry.
6. A four-NOOP suffix does not prove complete future equivalence of two
   noncanonical savestate serializations.
7. The current World-8 benchmark still has symbolic world-advance/Bowser edges,
   a deliberately myopic greedy construction, and injected hand/cursor facts.
8. No full-ROM speedup or broad “meta-intelligence” result follows from these
   artifacts.
9. The manifest hash pins the declared solution payload, not the exact dynamic
   input stream after executor-side chest truncation, settling, and probes.
   Exact executed-input hashing remains a provenance improvement.

## 8. Architectural diagnosis

The immediate problem is not that the planner needs a larger neural network.
It is that its state quotient is invalid.

Let \(x\) denote a complete physical emulator state and
\(\phi(x)=m\) the current `MetaState`. A safe deterministic option abstraction
needs more than \(\phi(x_1)=\phi(x_2)\). For every admissible option \(o\), the
states must agree on at least:

\[
\bigl(
\text{success/termination},
\phi(x'),
\text{duration/cost},
\text{intervention tier}
\bigr).
\]

The two acquisition-order representatives violate the current one-
representative assumption before that equivalence has been established.

The minimal near-term architecture is therefore:

```text
SearchNode
  meta_state
  physical_record_id
  physical_digest/provenance
  option_history_hash
  accumulated_cost
  parent + boundary
```

`MetaState` remains useful for goal predicates, reporting, and initial
partitioning. It must not remain the sole frontier/deduplication key.

## 9. Detailed next-step plan

### P0 — Preserve physical representatives in search

Implementation:

1. Replace `visited: set[MetaState]` and `best_cost: dict[MetaState, ...]` with
   frontier records keyed by a stable physical record identity.
2. Retain multiple physical representatives under one `MetaState`.
3. Carry producer, parent, option-history hash, boundary digest, and effective
   tier in each node.
4. Merge only exact repeated physical records. Treat same-source mGBA
   load/resnapshot variants through their stable record identity and explicit
   restore attestation, not through raw inequality alone.
5. Make alias discovery diagnostic rather than route-destructive once both
   representatives can safely coexist; keep a hard error if an old one-
   representative code path attempts to collapse them.

Acceptance gates:

- reversing option enumeration cannot change reachability;
- the two acquisition orders both survive the frontier;
- a missing predecessor root never invokes an executor;
- BFS and uniform-cost each restore once per physical option;
- failed trials cannot replace an accepted snapshot;
- identical exact representatives deduplicate deterministically.

### P1 — Test option-bisimulation, then refine the symbolic state

For every concrete representative in an alias block, execute every admissible
option at least twice and record its deterministic option signature:

```text
(success, termination reason, successor MetaState/refined block,
 duration/cost, effective tier, external roots, interventions)
```

Start partition refinement from the current `MetaState`. Split a block whenever
its representatives have different signatures. Only add a new symbolic field
when it explains a stable distinction; otherwise preserve history/digest
identity. This separates harmless mGBA serialization variants from
behaviorally material state differences.

Acceptance gate: a machine-readable abstraction report identifies the minimal
split required by the two acquisition orders and passes generated alias-graph
tests.

### P2 — Remove the Tier-3 second-whistle cursor repair

1. Reproduce the source/destination cursor discrepancy from both acquisition
   orders without writes.
2. Diff complete RAM blocks, wrapper context, recent input history, and menu/map
   timing around the first and second whistle exits.
3. Replace the repair with input-only settling/navigation if possible.
4. If the distinction is genuinely stateful, expose it in the refined option
   state rather than forcing the cursor.

Success: `use_whistle_again` remains at its declared Tier 1 with zero RAM
writes. Narrow condition: retain Tier 3 explicitly if the game state cannot be
reached through legal inputs from the selected root.

### P3 — Supply power legitimately from the live fortress lineage

The live entry is small Mario while the whistle route requires flight. Test in
this order:

1. determine whether the fortress whistle has any small-Mario/input-only route;
2. if not, implement a real map-inventory P-Wing or leaf acquisition and use
   option from the same predecessor lineage;
3. enter `(96,96)` with that legal power state;
4. replay/search the fortress whistle without P-Wing/leaf/P-speed reholds or an
   unrelated entry root.

Every candidate must report initiation state, exact predecessor identity,
primitive action artifact, termination, cost, power-state history, and zero
undeclared writes.

Success: live post-1-2 lineage reaches a whistle-bearing post-fortress map state
with no external root after the declared start. Narrow condition: if a legal
power acquisition is outside the intended scope, describe the result as an
entry-only option study rather than continuous acquisition.

### P4 — Construct the continuous two-whistle route

After P0-P3:

1. regenerate or explicitly recipe-pin the earliest practical root;
2. execute both whistle acquisitions from exact predecessor exits;
3. spend both whistles and select World 8 using input-only transitions;
4. preserve all boundary records and a final replay artifact; and
5. require `external_roots_after_declared_root=0`,
   `direct_memory_assign_calls=0`, no state alias collapse, and no undeclared
   intervention.

The route may use snapshots for branching/probe rollback; the accepted lineage
must remain rooted in the same declared physical history.

### P5 — Replace benchmark stand-ins and strengthen controls

Only after physical composition:

1. replace symbolic `clear_bowser` with a ROM-backed endpoint;
2. replace or remove symbolic warpless world-advance edges;
3. compare planners on the same option library and cost model;
4. add random, optimistic/UCB unknown-effect, cached/perfect-effect greedy, and
   history/belief-aware controls;
5. report discovery calls separately from exploitation cost; and
6. report physical frames separately from symbolic estimates.

If perfect-effect greedy matches uniform-cost, name the contribution
unknown-option exploration rather than general planning intelligence.

### P6 — Performance and learned-search experiments

These remain behind the state-contract work:

1. PHS/PHS* or Levin-style complete-safe policy guidance for SMB1;
2. failed-tree-mined subgoals for 6-2 and fortress search;
3. variable-duration action search for phase-sensitive platform/P-speed states;
4. a QuickerMGBA/JaffarPlus throughput spike on identical clone/step/dedup
   workloads; and
5. paired multi-level evaluation with bootstrap uncertainty and solve-rate
   non-inferiority.

## 10. Research directions as of July 25, 2026

The most important research refresh is older, exact mathematics newly made
operational by the alias counterexample:

1. Balaraman Ravindran and Andrew Barto,
   [“SMDP Homomorphisms: An Algebraic Approach to Abstraction in Semi-Markov
   Decision Processes”](https://www.ijcai.org/Proceedings/03/Papers/145.pdf),
   IJCAI 2003. Direct experiment: partition physical representatives by complete
   option outcome/duration/cost signatures before treating them as one abstract
   state.
2. Pablo Samuel Castro and Doina Precup,
   [“Using Bisimulation for Policy Transfer in
   MDPs”](https://ojs.aaai.org/index.php/AAAI/article/view/7751),
   AAAI 2010. Direct experiment: apply option-bisimulation tests to both
   acquisition-order representatives; raw-hash inequality is a witness to
   inspect, while option outcomes determine whether the distinction is
   behaviorally material.
3. Aijun Bai, Siddharth Srivastava, and Stuart Russell,
   [“Markovian State and Action Abstractions for MDPs via Hierarchical
   MCTS”](https://www.ijcai.org/Proceedings/16/Papers/430.pdf), IJCAI 2016.
   Direct experiment: compare `MetaState`-only, physical-identity, and
   option-history-keyed multi-representative search. Full POMCP is unnecessary
   initially because the emulator exposes each concrete branch.

The highest-value current methods from the July 25 refresh remain:

4. Laurent Orseau and Levi Lelis,
   [“Policy-Guided Heuristic Search with
   Guarantees”](https://arxiv.org/abs/2103.11505), AAAI 2021. Replace unsafe
   hard top-k policy pruning with a completeness-aware policy/heuristic search
   after the physical-state key is sound.
5. Jake Tuero, Michael Buro, and Levi Lelis,
   [“Subgoal-Guided Policy Heuristic Search with Learned
   Subgoals”](https://proceedings.mlr.press/v267/tuero25a.html), ICML 2025.
   Mine the failed 6-2/fortress trees rather than discarding them.
6. Palash Chatterjee and Roni Khardon,
   [“Improving Planning and MBRL with Temporally-Extended
   Actions”](https://papers.nips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html),
   NeurIPS 2025. Treat hold duration as a search variable under equal primitive
   frame and wall-clock budgets.
7. Rishabh Agarwal et al.,
   [“Deep Reinforcement Learning at the Edge of the Statistical
   Precipice”](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html),
   NeurIPS 2021. Use paired runs, IQM/bootstrap intervals, performance profiles,
   and probability of improvement for multi-level claims.
8. The
   [official mGBA scripting API](https://mgba.io/docs/scripting.html) exposes
   savestate buffers, registers, and memory domains but makes no byte-
   canonicalization guarantee. Add several fixed diagnostic suffixes that
   exercise input, timers, DMA/IRQ, and video if restore equivalence becomes a
   blocker; continue to label them finite falsifiers rather than proofs.

No newly fashionable world model outranks these steps. The emulator already
provides exact forward dynamics; the breakthrough is to make the abstraction,
lineage, and experiment contract correct before learning on top of it.

## 11. Verification

```bash
./venv/bin/python -m pytest -q \
  tests/test_option_provenance.py tests/test_options.py tests/test_meta_planner.py
# 27 passed

./venv/bin/python scripts/verify_iteration.py
# determinism/snapshot gate: 6 passed
# remaining repository suite: 143 passed, 10 skipped
# total: 149 passed, 0 failed, 10 skipped

MARIO_AI_SMA4_ROM=... ./venv/bin/python -m pytest -q \
  tests/test_sma4_overworld.py \
  tests/test_acquire_whistle_solution.py \
  tests/test_acquire_whistle_fortress_solution.py
# 7 passed

./venv/bin/python scripts/verify_stock_solutions.py --expect-verified 30
# 30/32; unsolved/quarantined: 6-2, 6-3

./venv/bin/python -m compileall -q mario scripts tests
git diff --check
```

The exact option reports and fortress diagnostic listed above were regenerated
from the final source implementation before documentation reconciliation. Runs,
ROMs, and savestates remain ignored and are not part of the commit.

## 12. One next action

**Refactor the SMA4 frontier to preserve multiple physical representatives per
`MetaState`, add deterministic option-signature partition refinement, and rerun
both whistle-acquisition orders under the hard boundary gates.**
