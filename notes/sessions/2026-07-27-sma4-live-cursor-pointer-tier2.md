# 2026-07-27 — SMA4 live cursor pointer and Tier-2 whistle route

> Corrective, evidence-first continuation of
> `2026-07-27-sma4-physical-representative-refinement.md`. Literature
> availability remains cut off at **2026-07-25**; implementation and
> verification continued on July 27 in America/Detroit.

## 1. Question

The physical-representative pass preserved two whistle-acquisition histories
and appeared to expose a causal asymmetry:

- `1-3 -> fortress` reached an input-responsive World-8 map;
- `fortress -> 1-3` reached raw World 8 but appeared stuck; and
- the viable route still used four Tier-3 writes to fixed cursor addresses.

Before searching for a history-repair suffix, this pass asks a more basic
measurement question:

**Are `0x03003DE0/0x03003DE4` always the live cursor object, or only one
possible storage location?**

They are only one storage location. The prior asymmetry was an observer false
negative, not evidence that the reverse history disabled input.

## 2. Observations before interpretation

### 2.1 The fixed cursor pair becomes stale

The supposedly failed lineage showed the expected map screens, world
transitions, menu behavior, and whistle inventory changes while the legacy
cursor bytes remained `(14, 135)`. That combination was already suspicious:
an off-grid pair stayed fixed while the rendered map and accepted controller
inputs continued to evolve.

A full IWRAM comparison and pointer scan exposed a little-endian cursor-object
base pointer:

```text
cursor object pointer: 0x03007824
logical Y:             *(pointer + 0), low byte
logical X:             *(pointer + 4), low byte
```

Two live bases occur in the retained lineages:

| Last acquisition | Live base | Legacy `(X,Y)` | Logical cursor source |
|---|---:|---:|---|
| fortress | `0x03003DE0` | live | pointer |
| 1-3 | `0x03004EF8` | stale `(14,135)` | pointer |

The 1-3 Toad-house exit relocates the cursor object. It does not break the
cursor. Writing the old pair merely made the observer's assumption appear
true.

### 2.2 Both physical histories traverse the same logical route

With the live pointer followed atomically, both retained histories produce the
same controller-visible sequence:

| Boundary | Expected logical cursor | Both orders |
|---|---:|---|
| 1-3 acquisition exit | `(160,32)` | resolved |
| fortress acquisition exit | `(96,96)` | resolved |
| first whistle settled | `(64,80)` | resolved |
| second whistle settled | `(128,144)` | resolved |
| World-8 entry | `(32,80)` | resolved |
| matched `DOWN` probe | `(32,80) -> (32,112)` | responsive |
| matched `NOOP` probe | `(32,80) -> (32,80)` | stationary |

This is stronger than accepting a direction-only movement test: the matched
NOOP control rules out an autonomous animation being mistaken for controller
response.

During the warp-zone-to-World-8 transition the pointer is zero for one
observed frame. The resolver therefore does **not** cache a prior base across
restores or lineages. In a live-map context it returns an explicitly labeled,
unresolved `(0,0)` for:

- a transient null pointer; or
- an aligned but unrecognized pointer.

Option boundaries fail closed while unresolved. Away from a live-map context,
the adapter retains the legacy pair as a compatibility fallback.

### 2.3 Cursor writes are unnecessary

`SMA4WhistleExecutor` now requires exact, pointer-resolved postconditions:

- acquisition exits are resolved, on the expected World-1 cell, menu closed,
  and have the required inventory increment;
- whistle spends decrement inventory and settle at exact warp-zone cells;
- the second whistle must start at `(64,80)` and end at `(128,144)`;
- World-8 selection starts at `(128,144)`, ends at `(32,80)`, has empty
  whistle inventory, and passes the matched `DOWN`/NOOP response test.

The old `_sync_map_cursor` and off-grid repair path are removed. There are zero
cursor RAM writes in the accepted route and in every explored option boundary.

### 2.4 Dirty-tree diagnostic benchmark

Before final commit, the corrected implementation was exercised in both
option-insertion orders:

```bash
MARIO_AI_SMA4_ROM=... ./venv/bin/python \
  scripts/bench_sma4_whistle_rom.py \
  --no-hand-grant --warpless-blocked --max-tier 2 \
  --out runs/20260727-sma4-pointer-cursor-tier2

MARIO_AI_SMA4_ROM=... ./venv/bin/python \
  scripts/bench_sma4_whistle_rom.py \
  --no-hand-grant --warpless-blocked --max-tier 2 \
  --reverse-option-insertion \
  --out runs/20260727-sma4-pointer-cursor-tier2-reverse
```

These reports intentionally record `git_dirty=true`; they are diagnostic, not
the final clean-source attestations.

| Measurement | BFS | Uniform cost |
|---|---:|---:|
| found mixed goal | true | true |
| hops | 6 | 6 |
| mixed reported frames | 13,266 | 13,194 |
| reported ROM-option frames before symbolic Bowser | 7,266 | 7,194 |
| retained physical records | 13 | 13 |
| refined finite classes | 11 | 11 |
| enabled branches | 14 | 14 |
| option executions, two per branch | 28 | 28 |
| repeat-conformance failures | 0 | 0 |
| cursor RAM writes on selected route | 0 | 0 |
| maximum effective tier | 2 | 2 |

BFS retains the insertion-order path:

```text
1-3 -> fortress -> first whistle -> second whistle -> World 8 -> symbolic Bowser
```

Uniform-cost search selects the 72-frame-cheaper reverse acquisition order:

```text
fortress -> 1-3 -> first whistle -> second whistle -> World 8 -> symbolic Bowser
```

The normal and reverse insertion reports agree on reachability and costs.
Both acquisition histories are expanded once at `select_world8_pipe`, and
both are accepted with exact cursor `(32,80)` and a resolved pointer.

The corrected finite table has 13 physical records in 11 blocks. It joins the
two post-second-whistle records `[9,10]` and the two World-8 records `[11,12]`.
The post-first-whistle records `[7,8]` remain distinct under
`use_whistle_again` because the measured durations are 1,302 versus 1,290
frames, even though both options succeed with the same logical destination.
Thus the correction removes a false reachability distinction; it does not
assert that every costed physical option signature is identical.

Greedy is intentionally nonexploring and remains stuck after two branches.
Therefore no finite speedup ratio is claimed. The 6,000-frame Bowser edge is
symbolic and is separated from the 7,194/7,266 reported ROM-option prefixes.
Those option totals are not yet pure retained-controller time: legacy
acquisition costs can include small rolled-back probes, which Gate D must
separate.

### 2.5 What remains Tier 2

Removing cursor writes does **not** make the route continuous or Tier 1.
Each accepted plan still:

- restores two independent external acquisition roots;
- begins those roots with P-Wing/leaf/P-speed assumptions;
- performs one direct inventory-merge write when composing the second
  acquisition; and
- performs one fortress leaf-rehold write after damage.

Thus each selected route has two direct write calls, neither touching cursor
state, and legitimately remains maximum Tier 2. “Acquisition order” currently
means the order in which independent roots are executed and composed; it is
not yet the history of one uninterrupted playthrough. The small cost
difference is a timing property of those segment executors, not evidence that
one causal game route is strategically better.

## 3. Implementation

### Adapter contract

`mario/adapters.py` now:

1. reads the cursor base from `0x03007824`;
2. accepts the two empirically observed aligned IWRAM bases;
3. resolves `(X,Y)` from `base+4/base+0`;
4. emits pointer, base, source, legacy pair, and resolution status in
   normalized info;
5. labels null/unknown live-map pointers unresolved instead of borrowing
   lineage state; and
6. preserves the old fixed pair only outside a live-map context.

### Option contract

`mario/options.py` now carries cursor provenance through samples, decoded
`MetaState`, boundary comparisons, whistle termination checks, and World-8
responsiveness. A symbolic state cannot match an unresolved physical cursor.

### Reverse-engineering tools

The overworld/whistle probes and fortress reconstruction helpers now read or
inject through the active cursor base. They also report the raw pointer and
legacy pair, so a future relocation cannot silently recreate this error.

### Regression coverage

The new focused tests cover:

- the legacy and relocated live bases;
- exact pointer provenance;
- between-grid cursor animation;
- transient-null and unknown-pointer failure labels;
- off-map legacy fallback;
- aligned IWRAM bounds; and
- rejection of unresolved cursor states at option boundaries.

### Verification before the clean-source benchmark rerun

```bash
./venv/bin/python -m pytest -q \
  tests/test_sma4_cursor_pointer.py \
  tests/test_option_provenance.py \
  tests/test_meta_planner.py
# 44 passed

MARIO_AI_SMA4_ROM=... ./venv/bin/python -m pytest -q \
  tests/test_sma4_overworld.py \
  tests/test_acquire_whistle_solution.py \
  tests/test_acquire_whistle_fortress_solution.py
# 7 passed

./venv/bin/python scripts/verify_iteration.py
# 171 passed, 0 failed, 0 errors, 10 skipped

./venv/bin/python scripts/verify_stock_solutions.py --expect-verified 30
# 30/32; unsolved/quarantined: 6-2, 6-3

./venv/bin/python -m compileall -q mario scripts tests
git diff --check
```

The two clean-source Tier-2 reports remain the post-commit attestation in
Gate A; the measurements in §2.4 deliberately identify themselves as
dirty-tree diagnostics.

## 4. Corrected interpretation

The prior physical-representative machinery remains valuable: it prevented
the two raw-distinct histories from being silently collapsed and exposed a
reproducible discrepancy. But the discrepancy was in the observation
function, not in the environment's controllability.

Formally, the old abstraction used an observation

\[
\phi_{\mathrm{old}}(s) =
  \bigl(\mathrm{RAM}[0x03003DE4],\mathrm{RAM}[0x03003DE0]\bigr),
\]

while the logical cursor is closer to

\[
p(s) = \mathrm{u32le}(\mathrm{RAM}[0x03007824:0x03007828]),
\qquad
\phi_{\mathrm{cursor}}(s) =
  \bigl(\mathrm{RAM}[p(s)+4],\mathrm{RAM}[p(s)]\bigr).
\]

For the relocated lineage,
\(\phi_{\mathrm{old}}\) observed an inactive implementation object.
No input suffix could repair that observer. The correct “minimal suffix” was
the empty suffix after repairing the measurement map.

This is a useful research result in its own right:

- before adding history to a state representation, verify that the observer
  follows indirection and object lifetime;
- preserve provenance for derived observables, not only their values;
- fail closed during transient unresolved frames; and
- distinguish an abstraction counterexample from a sensor/decoder bug.

## 5. Research bridge as of July 25, 2026

The following literature is newly relevant to the corrected diagnosis. These
references motivate experiments; none proves the SMA4 implementation correct.

### 5.1 Skills induce the useful symbols

Konidaris, Kaelbling, and Lozano-Pérez,
[“From Skills to Symbols”](https://doi.org/10.1613/JAIR.5575), JAIR 2018,
construct abstractions around the preconditions and effects of available
skills. That is a close conceptual fit for this repository: logical cursor
cell, inventory multiplicity, menu readiness, and world transition matter
because they determine option availability and effects.

**Repo experiment:** derive candidate option-boundary symbols from the finite
transition table, then hold out primitive and option suffixes as falsifiers.
Keep physical identity as the search key until the held-out gate passes.

### 5.2 Predictive state should survive history changes

Ni et al.,
[“Bridging State and History Representations: Understanding Self-Predictive RL”](https://proceedings.iclr.cc/paper_files/paper/2024/hash/666c1861d709bd84e20b6e0e02a2c223-Abstract-Conference.html),
ICLR 2024, connect state and history representations through
self-predictive objectives.

**Repo experiment:** compare candidate state features by their ability to
predict exact finite option outcomes and costs across independently restored
histories. The pointer-resolved cursor should predict both histories; the
fixed pair should fail immediately.

### 5.3 Causal abstractions must tolerate multiple realizations

Xia and Bareinboim,
[“Causal Abstraction Inference under Lossy Representations”](https://proceedings.mlr.press/v267/xia25a.html),
ICML 2025, study causal abstraction when several low-level interventions or
effects map into a lossy high-level representation.

**Repo experiment:** treat distinct cursor bases as low-level realizations of
one logical map-cell variable and test intervention invariance with matched
controller probes. Do not erase base provenance from diagnostic artifacts even
when the logical effects agree.

### 5.4 Approximate merging belongs after exact finite tests

Abel et al.,
[“Near Optimal Behavior via Approximate State Abstraction”](https://proceedings.mlr.press/v48/abel16.html),
ICML 2016, bound value loss under approximate state abstractions.

**Repo experiment:** first retain exact physical records and finite
option-signature partitions. Only then measure whether a deliberately
approximate merge reduces representatives without exceeding a declared
outcome/cost error budget.

### 5.5 Provenance is part of the scientific object

The W3C
[PROV Data Model](https://www.w3.org/TR/prov-dm/) distinguishes entities,
activities, generation, use, and derivation. Pineau et al.,
[“Improving Reproducibility in Machine Learning Research”](https://jmlr.org/papers/v22/20-303.html),
JMLR 2021, motivate explicit reproducibility checklists and artifact
reporting.

**Repo experiment:** represent root snapshots and successor records as
entities, option executions as activities, and record parents/interventions as
derivations. Final reports should bind source commit, ROM digest, root digest,
executed-input digest, pointer provenance, and clean/dirty source state.

### 5.6 Counterexample minimization remains useful, but is no longer the gate

Zeller and Hildebrandt,
[“Simplifying and Isolating Failure-Inducing Input”](https://doi.org/10.1109/32.988498),
IEEE TSE 2002, introduced delta debugging for minimizing failure-inducing
inputs.

The planned suffix minimization was appropriate while a true behavioral
divergence was plausible. Here it would have optimized around a bad observer.
Retain delta debugging for future replay mismatches, but first validate that
all compared observables refer to the same logical object.

## 6. What is established, and what is not

Established by the current ROM-backed evidence:

1. SMA4's live map cursor can occupy at least two IWRAM object bases.
2. `0x03007824` selects the active base in both retained histories.
3. Both whistle-acquisition orders spend both whistles and select World 8 by
   controller input with zero cursor writes.
4. Both histories pass the matched directional/NOOP response test.
5. The segmented mixed benchmark is viable at maximum Tier 2.

Not established:

1. a continuous route from one earliest root;
2. legitimate acquisition and preservation of all required power;
3. a write-free or root-free acquisition composition;
4. a physical World-8/Bowser endpoint;
5. a planner speedup over a matched exploratory baseline;
6. a globally sufficient `MetaState`;
7. universal cursor-base completeness beyond the two observed bases; or
8. byte-canonical Stable-Retro/mGBA savestate serialization.

## 7. Detailed next-step plan

### Gate A — Close the corrected evidence boundary

Actions:

1. run focused pointer/option/planner tests;
2. run all ROM-backed SMA4 adapter/acquisition tests;
3. run `scripts/verify_iteration.py`;
4. replay the 30/32 stock solution gate;
5. compile every changed Python file and run `git diff --check`;
6. commit the source and documentation;
7. rerun normal and reverse Tier-2 reports from that clean commit; and
8. require both reports to attest `git_dirty=false`, the same commit, zero
   cursor writes, both acquisition orders accepted, and no repeat mismatch.

Stop condition: any unresolved pointer at an accepted boundary, order-dependent
planner result, Tier-3 event, cursor write, or dirty-source report blocks this
gate.

Expected final artifacts:

```text
runs/20260727-sma4-live-cursor-tier2-final/report.json
runs/20260727-sma4-live-cursor-tier2-reverse-final/report.json
runs/tests.json
```

### Gate B — Build one continuous physical acquisition lineage

This is now the highest-value obstacle.

Actions:

1. choose the earliest reproducible live World-1 root already connected to the
   post-1-2 map;
2. enumerate legitimate P-Wing, leaf, and power-preservation routes reachable
   from that root;
3. acquire the 1-3 whistle, fortress whistle, or required power in the order
   favored by actual map topology rather than cached-root timing;
4. carry inventory and power through controller input;
5. remove the second external acquisition restore;
6. remove `prior_whistle_inventory_merged`;
7. remove `fortress_leaf_rehold_after_damage`; and
8. replay the entire accepted input history from the one root.

Required acceptance fields:

```text
external_roots_total = 1
external_roots_after_declared_root = 0
direct_memory_assign_calls = 0
cursor_resolved_at_all_accepted_map_boundaries = true
full_executed_input_sha256 = <present>
independent_replay_verified = true
runtime_effective_tier <= 2
```

If the fortress truly requires flight, document the exact mechanic and build
the legitimate source. Do not silently promote a power write or unrelated
savestate to initialization. The acquisition scripts are currently declared
`TIER2_BLACK_BOX_OPTION`; removing restores and writes does not by itself
reclassify them as Tier 1. Any later tier reduction requires a written
criterion and a corresponding option-contract change.

### Gate C — Replace the symbolic endpoint

Actions:

1. physically traverse the selected World-8 pipe;
2. segment World 8 into replay-verified map/level options;
3. replace `clear_bowser` with a physical endpoint;
4. report deaths, retries, restores, retained frames, evaluation frames, and
   wall time separately; and
5. independently replay the final full-ROM route.

The project should not call the mixed 13,194-frame result a full game solve.

### Gate D — Repair cost accounting and planner controls

Actions:

1. separate retained frames from rolled-back menu probes inside every legacy
   option;
2. define a vector cost
   `(retained_frames, evaluation_frames, option_calls, wall_time)`;
3. compare greedy, BFS, UCS, an optimistic explorer, and a PUMA-like
   effect-cache baseline under the same option library and budgets;
4. remove or rename the current “speedup” contrast when greedy does not find a
   goal; and
5. repeat across roots and option insertion orders.

### Gate E — Test option-induced abstractions without trusting them early

Actions:

1. include pointer source/base in diagnostic provenance but use the logical
   resolved cell in option initiation/termination;
2. generate active distinguishing option and primitive suffixes;
3. reserve held-out suffixes before proposing merges;
4. compare physical-record search, exact refined blocks, and bounded
   approximate abstractions;
5. measure false merges, representative count, option calls, cost error, and
   wall time; and
6. test the learned abstraction on new roots before using it for dominance.

No neural representation should become merge authority merely because it
compresses well. Exact emulator transitions remain the acceptance oracle.

### Gate F — Resume the broader program after the physical contract is stable

1. re-solve SMB1 6-3 and attack 6-2 with platform phase and
   variable-duration actions;
2. test completeness-safe policy-guided search across multiple level types;
3. close checkpoint/training-data provenance before making learned-guidance
   claims; and
4. benchmark alternative GBA cores only under identical clone/step/replay
   contracts.

## 8. One next action

**Starting from one live World-1 root, obtain and preserve the power required
for both whistle acquisitions through controller input, remove the independent
second acquisition root plus inventory/leaf writes, and replay the resulting
continuous physical lineage with zero direct writes at maximum Tier 2.**
