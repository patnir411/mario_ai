# 2026-07-25 — Project audit, claim reconciliation, and next-step program

> This is the durable handoff for the July 25 audit. It records verified state
> before cleanup, separates artifact-backed facts from hypotheses, and defines
> the gates for the next research program. Research citations and the final
> ranked plan are updated in this note as the audit completes.

## 0. Audit question

Reconstruct the project from first principles and its local artifacts; identify
where the implementation has exceeded the older narrative, where claims outrun
verification, what belongs in the next commit, and which experiments have the
best chance of producing a real breakthrough rather than another proxy metric.

The governing standard remains:

1. Search/option results require a replayable action artifact and deterministic
   replay success.
2. Learned results require an evaluation artifact, not validation accuracy.
3. Planner comparisons require the same option library and cost model, with all
   injected facts enumerated.
4. A negative result is retained when it closes a tempting but unproductive
   branch.

## 1. Verified state at the start of the audit

### 1.1 SMB1

- The cached any% route remains 8/8 and `beat_game=True`; the hard water/castle
  results remain the project's strongest systems evidence.
- The stock-level headline needs correction. The working memory says 31/32, but
  `data/solutions/6-3.json` is explicitly quarantined with
  `solved=false`, `invalid_reason=replay_seed0_died_without_flag`, while 6-2 is
  also `solved=false`. By the project's replay standard, the defensible count is
  therefore **30/32 replay-verified**, not 31/32. "31/32" described cached search
  claims before the 6-3 replay gate invalidated one.
- The untracked stock-solver wave added valid small solution JSONs for many
  previously missing stock levels. These are evidence artifacts and should be
  included deliberately in the next commit; the quarantined 6-3 negative should
  also be retained so it cannot silently become a positive again.

### 1.2 Learning/search guidance

- V5/V6 code and findings exist locally but were never coherently committed.
  The later tracked search code already contains policy-prior hooks, while
  `mario/entity.py`, `mario/entity_policy.py`, the then-named
  `mario/stable_bc.py`,
  the training/evaluation drivers, and their tests remain untracked.
- The most defensible learned result is narrow: on SMB1 1-1, one weak entity
  policy prior reduced beam nodes from 7005 to 2770 while preserving a solve
  (60.5% fewer; 2.53 plain/guided ratio). This is a single-level result, not yet evidence of general search
  acceleration.
- The standalone-policy results are useful negatives under the tested data and
  recipes. They do not establish a universal imitation-learning ceiling. The
  July 14 theory audit already narrowed that claim in light of modern BC and
  distillation theory.
- The module then named `mario/stable_bc.py` was not an implementation of
  Mehta et al.'s Stable-BC: it penalized finite-difference changes in policy
  outputs under input noise, but never modeled or constrained the closed-loop
  dynamics Jacobian. It has now been renamed `mario/consistency.py` and is
  described as a perturbation-consistency/sensitivity surrogate.
- `coverage_search` now also has a policy/value guidance interface and new
  untracked tests. That implementation has not yet been paired with a
  multi-level artifact-backed benchmark.

### 1.3 SMA4 / SMB3 Option-SMDP track

- SMA4 1-1 and 1-2 level search, overworld instrumentation, whistle spend, and
  both whistle-acquisition executors are implemented and artifact-backed as
  replayable segments.
- The latest no-door-snapshot benchmark reaches the World-8 skip with no
  whistle hand-grant or warp-zone regrant. It still injects a P-Wing fortress
  entry, leaf/P-speed reholds, map synchronization, and symbolic warpless/Bowser
  endpoints. The 5.18x number is therefore a mixed ROM/symbolic benchmark, not a
  full-ROM any% speedup.
- The acquisitions do not yet compose as one continuous option execution: they
  restore independent cached entry states and merge selected symbolic/inventory
  facts. Runtime interventions reported inside option summaries are not all
  propagated into `MetaSearchResult`, so the current report can undercount
  honesty debt. The defensible label is **segmented option-planning prototype**.
- `OptionContext.remember` is first-wins on compressed `MetaState`. Two distinct
  emulator states that alias to one symbolic state can therefore retrieve the
  wrong snapshot. This is a state-sufficiency hypothesis that needs a hash-based
  contract test before the abstraction is trusted.
- The correct immediate honesty target is live
  overworld -> `(96,96)` -> fortress entry, followed by removal or explicit
  tiering of leaf/P-speed reholds. A second scientific target is a stronger
  baseline: greedy with a learned/cached option-effect model, so the benchmark
  distinguishes *effect discovery* from ordinary cost planning.

### 1.4 Cross-game spine

- The adapter boundary is real, with SML and SMA4 support and substantial tests.
  Several CLI drivers referenced by already-committed notes are nevertheless
  still untracked. The next commit must make tracked documentation and tracked
  executable code agree.
- Cross-game generalization remains a later experiment. Adapter parity is useful;
  a claim of one general Mario controller is not supported.

## 2. Verification performed on July 25

Commands run from the native repository venv:

```bash
./venv/bin/python scripts/update_status.py
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python -m pytest --disable-warnings
./venv/bin/python -m compileall -q mario scripts tests
```

Observed:

- `update_status.py` completed, but exposed stale producer state: its generated
  block still labels V4/V5 TODO and reads the old `runs/tests.json`.
- Full pytest: **83 passed, 10 skipped in 24.21s**.
- Compile-all: clean.
- A clean-source simulation found that tracked
  `tests/test_options.py` expects untracked
  `data/solutions/sma4/{1-1,1-2}.json`. The fresh-clone contract is therefore
  currently broken even though the artifact-rich local checkout passes.
- The documented fresh setup is also broken: `uv venv` does not create
  `./venv/bin/pip`, and the `gba` extra is unsatisfiable because
  `nes-py==9.0.0` declares `pyglet<=1.5.21` while
  `stable-retro==1.0.1` declares `pyglet>=1.5.27`.

The existing `scripts/verify_iteration.py` records only its selected fast-gate
subset as the headline test count. That is why the generated status still says
49 tests. Cleanup should either record the full suite or label the count
explicitly as a core-gate count.

## 3. Commit-readiness findings

The dirty tree is not one undifferentiated change. It contains:

1. **V5/V6 research core:** entity schema/policy, input-consistency experiment,
   held-out split and integrity tests, generalist/DAgger/RL drivers, search
   guidance tests, and V5/V6 findings.
2. **Cross-game command-line surface:** SML/SMA4 solve/probe/replay drivers that
   are already described by tracked notes and modules.
3. **Small evidence artifacts:** newly solved stock-level action JSONs and the
   quarantined 6-3 result, currently hidden by the broad `data/` ignore rule.
4. **Documentation reconciliation:** `DESIGN.md`, `CLAUDE.md`, this note, session
   log, bibliography, and a detailed next-step plan.
5. **Local debris:** `.playwright-mcp/` logs/pages and a machine-specific
   Playwright/Rosetta workaround. These are not research source by default and
   should not be swept into the project commit without an explicit portability
   reason.

The Playwright directory was not merely noisy: its captured browser pages contained
login/OAuth URLs and dashboard text. It was moved recoverably to
`~/.Trash/mario_ai_playwright_mcp_20260725`, added to `.gitignore`,
and excluded from the commit surface. The machine-specific
`PLAYWRIGHT_ARM64_ROSETTA_FIX.md` was likewise moved to
`~/.Trash/mario_ai_PLAYWRIGHT_ARM64_ROSETTA_FIX_20260725.md`.

Cleanup decisions implemented during this pass:

- canonical small solution JSONs are explicitly tracked while datasets,
  checkpoints, videos, savestates, ROMs, and run directories remain ignored;
- NES and GBA emulator stacks are isolated into conflicting optional extras, and
  the shared adapter/search/model imports no longer require NES packages;
- every direct stock publisher now independently replays before a positive can
  become canonical; base SMA4/SML CLI promotions do the same through their
  adapters; custom whistle options have explicit success invariants and all
  snapshot-root manifests now include content and ROM hashes;
- `label_at_state` restores both emulator state and Python-side info/frame caches;
- the hardened generalist trainer, held-out evaluator, policy-guided benchmark,
  and RC-RL prototype record future seed/config/input/checkpoint/environment/
  source provenance. DAgger collection and initial entity-dataset generation
  still need equivalent machine-readable reports and source/solution hashes;
- the 1-1 learned-guidance comparison is preserved at
  `notes/artifacts/2026-07-25-policy-guided-1-1.json`, including both returned
  paths and independent replay success;
- README, DESIGN, V5/V6 findings, CLAUDE working memory, and generated status
  producers now use the same evidence boundaries.

## 4. Ranked next-step program

### P0 — Make the evidence base coherent

Primary outcome: one reviewable commit in which code, claims, and artifacts
refer to the same state.

Gates:

- full suite green and recorded truthfully;
- 30/32 wording everywhere until 6-3 is re-solved;
- all claimed source files tracked;
- no Playwright logs, ROMs, savestates, datasets, or checkpoints staged;
- new stock solution JSONs replay-gated and intentionally included;
- one command lists the exact replay-verified stock set.
- a clean-source checkout passes without depending on ignored local artifacts;
- the documented NES install resolves from a new venv, and GBA dependency
  isolation/conflict handling is explicit and tested.

### P1 — Finish the SMA4 honesty burn

Primary hypothesis:

> The two-whistle World-8 route remains discoverable by execute-to-observe
> option search when fortress entry starts from a live post-clear overworld
> state rather than an injected level snapshot.

Experiment:

1. Begin from one clean boot or a single declared root snapshot.
2. Execute every option from its exact predecessor exit; prohibit unrelated
   entry restores and inventory merging in the flagship run.
3. Reconstruct the natural post-1-2 map state, discover/path to `(96,96)`, and
   enter using `SMA4OverworldAdapter`.
4. Compose that exact state with the fortress executor. If flight power is
   required, add a legitimate acquisition option rather than a power-state poke.
5. Remove `pwing_fortress_entry_snapshot`; then attack leaf/P-speed reholds,
   cursor writes, and any other intervention one at a time.
6. Record full-state hashes and a write/provenance ledger at every boundary.
7. Re-run open and warpless-blocked benches.

Success:

- replay-verified acquire and World-8 selection from one root;
- exact predecessor/entry state hashes match;
- `injected_facts=[]` for the flagship claim, or an explicit segmented/tiered
  claim if that cannot yet be achieved;
- same library/cost model for all planners.

Kill/narrow condition:

- if natural entry cannot preserve the required power state without an explicit
  acquisition option, label that initiation Tier-3 and scope the claim to the
  remaining honesty set instead of hiding the dependency.

### P2 — Turn the planner demo into a stronger experiment

First formalize every option as an initiation set \(I\), executor policy \(\pi\),
success/failure termination \(\beta\), measured cost/duration, and sufficient
successor state. Property-test initiation, termination, replay, and abstraction:
if two emulator states map to one `MetaState` but produce different option
outcomes, the symbolic state is not Markov-sufficient.

Then add controls that can falsify the current interpretation across generated
unknown-option graphs as well as the SMA4 instance:

- myopic greedy with no effect model;
- random and epsilon-greedy trials;
- optimistic/UCB-style unknown effects;
- greedy with cached/perfect option effects;
- uniform-cost/BFS execute-to-observe;
- a small BAMCP-style history/belief baseline.

Report separately:

1. discovery regret (calls before the whistle effect is known);
2. exploitation cost after the effect is known;
3. ROM-measured frames;
4. symbolic endpoint costs;
5. injected facts.
6. option-model accuracy and success versus option-call budget.

If perfect-model greedy matches uniform-cost, the result is an
**unknown-effect exploration result**, not broad "meta-intelligence." That is
still a valid and clearer contribution.

### P3 — Complete SMB1 evidence before calling it 32/32

Order:

1. Re-solve 6-3 at seed 0 and replay-gate it immediately.
2. Treat 6-2 as an athletic/moving-platform state-representation problem, not
   merely "increase beam width."
3. Instrument platform/enemy phase variables; ablate cell keys and fixed versus
   variable action durations such as `{1,2,4,8,16}`.
4. Benchmark unguided coverage, soft policy guidance, and entropy-adaptive
   guidance under equal wall-clock/node budgets.

Success is replay success, not a search-time `solved` flag. Keep 6-3 quarantined
until that gate passes.

### P4 — Generalize the only positive learning result

The next learning question should not be "can the policy play?" It should be:

> Does a calibrated learned prior reduce search work across levels and seeds
> without lowering solve rate?

Minimum benchmark:

- at least six levels spanning linear, athletic, castle, and deceptive routing;
- >=5 deterministic seeds or distinct start snapshots where meaningful;
- paired budgets and bootstrap confidence intervals for nodes and wall time;
- solve-rate non-inferiority gate;
- plain search, fixed top-k, soft log-prior, uniform-mixture prior,
  entropy-adaptive pruning, and a PHS/Levin-style complete-safe alternative;
- a static action-frequency prior and a random prior to separate learned
  information from generic action bias.

Do not train a larger controller until this benchmark reveals where prior error
actually costs nodes.

### P5 — Cross-game parity only after P1/P2

Use the adapter contract to add one second game-level experiment with the same
artifact schema and search metrics. The purpose is to test abstraction quality,
not to claim zero-shot control.

## 5. July 25, 2026 research shortlist

The full annotated citations live in `notes/research-bibliography.md`. The
highest-value experiments are:

1. **PHS/PHS* or Levin-style policy-guided heuristic search** — Orseau and
   Lelis (AAAI 2021), https://arxiv.org/abs/2103.11505. This is the most direct
   mathematical replacement for hard top-k pruning: it targets deterministic
   single-agent search loss while combining a policy and heuristic.
2. **Learn subgoals from failed search trees** — Tuero, Buro, and Lelis
   (ICML 2025), https://proceedings.mlr.press/v267/tuero25a.html. The direct
   local target is 6-2/fortress search debt, not another dense BC dataset.
3. **Variable-duration actions** — Chatterjee and Khardon (NeurIPS 2025),
   https://papers.nips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html.
   Test duration as a planning variable on moving-platform and P-speed states.
4. **Belief/history planning for unknown options** — Guez, Silver, and Dayan's
   BAMCP, https://proceedings.neurips.cc/paper/2012/hash/35051070e572e47d2c26c241ab88307f-Abstract.html.
   Use it as a small-suite control, not as an assumption that a large MCTS stack
   is automatically better.
5. **Selective policy abstention** — Goel, Pei, and Wang (May 2026 preprint),
   https://arxiv.org/abs/2605.09183. This suggests a calibrated policy-to-search
   handoff rather than mandatory standalone action at every state.
6. **Reset access as a formal resource** — Krishnamurthy, Li, and Sekhari
   (COLT 2025), https://proceedings.mlr.press/v291/krishnamurthy25a.html.
   Snapshot/local-reset access does not by itself erase the difficulty of
   agnostic policy learning under their assumptions.
7. **Reliable few-run comparison** — Agarwal et al. (NeurIPS 2021),
   https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html.
   Use IQM, bootstrap intervals, performance profiles, and probability of
   improvement rather than one deterministic run per configuration.
8. **Throughput spike** — JaffarPlus and QuickerMGBA,
   https://github.com/SergioMartin86/jaffarPlus and
   https://github.com/SergioMartin86/quickerMGBA. Compare identical state-clone
   and node-expansion workloads before accepting a new emulator integration.

Secondary architecture lead: Haramati et al.'s ICLR 2026 hierarchical
entity-centric/factored-subgoal model is interesting as an **offline option
subgoal proposer**, not as a replacement for exact execution. It earns a test
only after hand-written and failed-tree-mined subgoals form cheaper controls.

Parked unless the main program demands them:

- modern plasticity/reset methods for reverse-curriculum RL;
- continuous-control action-chunking theory for standalone BC;
- procedural level generation/UED for a separate 10^3–10^4-level program;
- object-centric causal/world models when an exact emulator is already cheap.

## 6. Next-action rule

After cleanup, keep exactly one action in `CLAUDE.md`. The audit changes the
candidate to:

> Add option-boundary state hashes, intervention provenance, and alias tests;
> then replay from one clean root through live post-1-2 overworld ->
> `(96,96)` -> fortress entry, with exact predecessor-exit composition and no
> unrelated snapshot restore.

Only after that invariant passes should the benchmark attack individual
leaf/P-speed/cursor interventions and the planner-class comparison.

## 7. Final cleanup and verification outcome

The starting failures in §2 were repaired rather than hidden.

### Evidence and scientific-contract checks

- `scripts/verify_stock_solutions.py --expect-verified 30` exits green with
  exactly 30/32 positives; only 6-2 and 6-3 remain negative.
- The current 1-1 matched benchmark regenerated the exact 7005 plain / 2770
  guided result. Both returned paths independently replay to the flag. The
  committed report now includes the current source-file hashes, checkpoint hash
  `70d1f0e398e3af462968477b0913241a437fc838c6de7d2e264f1a83de5afcec`,
  and local training-row hash
  `96a09405512b7ea520de21a7f237cb98445b187df795bb48ca7bca29f572c390`.
  This is an artifact-backed outcome, still not a clean-clone training recipe.
- SMA4 1-3 AcquireWhistle replay initially exposed an off-grid map-cursor state
  that was visually on the overworld but classified as `menu`. The adapter now
  uses the observed live-map event (`0x11`) as an additional map invariant.
  Independent replay and the executor both finish with one whistle and
  `final_mode=overworld`; the fortress whistle executor also replays successfully.
- Direct beam, coverage, and Go-Explore-style publishers cannot promote a
  search-time success without an independent replay. Strict manifest validation
  rejects truthy strings, booleans/floats for `chunk_frames`, empty paths, and
  out-of-range actions.
- `SMB1Adapter` snapshots now include emulator state plus wrapper info/frame
  caches. A real rebranch smoke confirmed restored info, observation, cell, and
  the next four-frame child; generic adapter deduplication no longer reads a
  previous candidate's cached cell.
- Held-out policy evaluation now preserves each seed/path/cause/root-RAM hash
  and replay-gates live positives. RC-RL requires at least two critics for its
  clipped double-Q label and records frontier transitions and a success curve.

### Test and portability checks

- Artifact-rich local tree: **134 passed / 10 skipped** in the final
  audit (ROM-backed tests can run locally).
- Exact source-only candidate: **133 passed / 11 skipped**.
- Fresh NES environment: install, one-frame emulator boot, and full source-only
  suite all green at **133 passed / 11 skipped**.
- Fresh GBA+SML environment: install/import green; focused normal suite
  **27 passed / 10 ROM-gated skipped**.
- A combined `[nes,gba]` install fails intentionally with the declared,
  explanatory `pyglet` conflict. Generic imports remain emulator-independent.
- `compileall`, JSON parsing, `git diff --check`, and a candidate secret/ROM/
  savestate/checkpoint/dataset scan are clean.

Portability caveats retained rather than papered over:

- there is no `uv.lock`, so transitive dependency resolution remains
  date-sensitive;
- snapshot-root SMA4 evidence needs the ignored local savestate even though its
  ROM/snapshot hashes are committed;
- one held-out-shard smoke test skips in a clean clone because generated
  training data remains intentionally ignored;
- three small historical contact sheets under `runs/` were already tracked and
  still enter a clone;
- DAgger/entity-dataset collection provenance remains weaker than the hardened
  trainer/evaluator paths.

### Exact next-commit boundary

Handoff state: branch `cursor/bottom-up-research-plan-c3e1`, parent
`8bc8bbc`, with the upstream still 0 ahead / 0 behind before the new commit.
The intended checkpoint is 42 modified tracked paths plus 59 untracked source/
evidence paths. The index remains unstaged; `git add -n --all` enumerates only
the intended surface and no deletion.

Include:

1. replay/manifest, snapshot/cache, checkpoint-loading, optional-emulator, and
   status-contract hardening plus their tests;
2. the deliberately preserved V5/V6 research surface, held-out config,
   consistency/entity/policy/RC-RL drivers, and qualified findings;
3. all 32 SMB1 manifests (including the two negatives), four SMA4 manifests,
   one SML manifest, and the two small status/policy-guidance JSON reports;
4. reconciled README/DESIGN/CLAUDE/session/bibliography documentation and this
   gated P0-P5 program.

Exclude:

- ROMs, snapshots, datasets/shards, checkpoints, videos, new run directories,
  virtual environments, browser captures, OAuth/dashboard residue, and
  machine-specific Playwright notes.

No commit or push was performed during the audit. A defensible checkpoint
message is:

```text
research: harden replay evidence and preserve V5/V6 experiments
```

After that checkpoint, execute P1 in order. Do not start a larger learning or
cross-game architecture until the option-boundary hash/alias contract and live
post-1-2 fortress composition either pass or produce an explicit negative.
