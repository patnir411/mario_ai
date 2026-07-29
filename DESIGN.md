# Mario AI — Design Document

**Goal:** Build a local, from-scratch exact-execution research system, then test whether
query-efficient discovery can recover and certify useful phase-aware, weighted option quotients on
declared finite domains. Exact search remains the solver; Mario is the adversarial case study, not
the entire claim. The primary machine is a **MacBook Pro 14" (2023, M2 Pro, 16 GB unified
memory)**.

**Project intent (chosen):** *Learn the whole stack.* The objective is to implement and genuinely
understand forward-model search, behavior cloning/distillation, DAgger, learned search guidance,
emulator adapters, and option-level planning end to end. Replay-verified game progress is the proof
of correctness; faster routes are an optimization target, not a current universal "speedrun"
claim. Breadth and understanding over peak performance.

This document is the map. Each module below doubles as a learning unit: it states *what* it is, *why* it exists, the *theory* behind it, the *interface*, and the *failure modes* to watch.

Date: 2026-06-03.
Updated: 2026-07-28 after a strategic reassessment, phase-aware option/trace
contract, hardware-accounting correction, and split of the portable science
program from the bounded SMA4 integrity case. The durable contract is
`notes/theory/option-machine-trace-v3.md`; the decision and experiment program
are in
`notes/sessions/2026-07-28-strategic-reassessment-and-option-contract.md`.

---

## 0. The core idea in one paragraph

For a fixed emulator configuration, Mario supplies an exact, resettable transition function:
`state_{t+1} = emulator(state_t, action_t)`. Search therefore remains the low-level solver; there is
no need to approximate known emulator dynamics. Search trajectories can train small policies, but
the strongest current learning use is to **guide** search while retaining exact expansion and
replay verification. Non-NES games enter through a shared adapter contract. Longer routes are
represented as options whose effects are observed by execution, but the current SMA4 system is a
**segmented option-planning prototype**, not yet one continuous Option-SMDP run. The next
representation is an additive phase-aware finite controller plus primitive execution trace; it is
specified but not implemented. Its purpose is to make phase, retained cost, interventions, and
physical lineage falsifiable before any live quotient merge.

```
NES / GB / GBA emulator
        │
   GameAdapter: reset, snapshot/restore, step, progress, terminal
        ├──────────────────────────────────────────────────────────────┐
        │                                                              │
   RAM/tile/entity observation                                exact physical record
        │                                                     + lineage + intervention
   chunked action space                                               │
        │                                              proposed OptionMachine (phase)
   beam / coverage / safe policy search                      + OptionTrace (primitives)
        │                                                              │
   replay-gated solution artifact                    v2 table / suffix diagnostics
        │                                                              │
   self-generated labels → learned proposer/prior    FUTURE: certified finite quotient
```

**Evidence snapshot, 2026-07-28:**

- SMB1 any% is 8/8 with `beat_game=True`.
- SMB1 stock coverage is **30/32 replay-verified**. 6-2 is unsolved; 6-3 is quarantined after a
  seed-0 replay failure.
- The positive learned-guidance result is local to 1-1: 7005 plain-beam nodes versus 2770 guided
  nodes (60.5% fewer; 2.53 plain/guided ratio), with solve preservation in that configuration. The checkpoint/data are
  local and lack a clean-clone training-provenance chain.
- The SMA4 whistle/World-8 route is a segmented mixed emulator/symbolic benchmark with declared
  state and inventory interventions. It is not evidence of a continuous full-ROM route.

**Implementation delta through 2026-07-27:**

- Opt-in SMA4 BFS/UCS now key search by `(MetaState, physical_record_id)` and
  preserve both whistle-acquisition histories. Strict legacy callers still fail
  closed on physical aliases.
- A complete finite-library report repeats enabled transitions, compares
  normalized outcomes plus finite physical-exit attestations, recursively
  refines successor blocks, and returns distinguishing option suffixes.
- The formerly fixed SMA4 cursor pair can name an abandoned object after the
  1-3 exit. Following the live pointer at `0x03007824` makes both acquisition
  histories input-responsive and removes all cursor repair writes. Normal and
  reverse insertion diagnostics both pass the mixed route at maximum Tier 2.
- This is still a segmented result, not a continuous route or a speedup:
  independent power-state roots, inventory/leaf writes, and a symbolic Bowser
  edge remain, while greedy is intentionally nonexploring and finds no goal.
- The 11 refined blocks are stable only within the encountered finite table.
  Refinement starts from exact `MetaState` colors and only splits, so the result
  is not claimed to be the coarsest option-induced quotient.
- The first search-semantics audit fixes are in place: only retained frontiers
  consume novelty, action-vocabulary permutations are regression-tested, and
  sustained pipe macros count their actual chunk calls and scheduled
  chunk-frame path cost. Exact primitive-frame work remains uninstrumented when
  `run_chunk` terminates early. Learned beam guidance still adds an edge prior
  rather than cumulative path probability.

**Strategic/contract delta, 2026-07-28 (documentation only):**

- The current `Option` is explicitly classified as a legacy arbitrary runner;
  it has no finite controller phase or retained primitive action trace.
- `OptionMachine` / `OptionTrace` v3 is an additive target. Current
  `mario-ai.option-observation.v2` artifacts retain their existing name and
  limited boundary-observation meaning.
- The primary science program begins on synthetic systems with known quotients,
  then tests active distinguishing queries, matched search/reset ablations, and
  an unchanged open second domain. One-root SMA4 continuity is a bounded
  parallel integrity case rather than the project-wide blocker.
- The theorem claim is restricted to a declared finite closed domain, fixed
  option alphabet, exact retained weight, and explicit controller phase.

---

## 1. Why this architecture (and not the alternatives)

Grounded in the literature so the choices are defensible, not cargo-culted.

- **Search is a strong fit when exact resets are available.** A* was the strongest historical
  baseline in the Mario AI Competition, and this repository's own artifact-backed successes are
  emulator-search results. This is a project-specific engineering conclusion, not a theorem that
  planning dominates learning in every resettable environment.
- **A learned world model is not the current bottleneck.** MuZero, EfficientZero, and Dreamer learn
  dynamics when the real transition model is unavailable or too costly. Here the emulator is
  exact and locally callable, so model learning is lower priority than search efficiency,
  abstraction quality, and option composition. A learned abstraction could still become useful
  if raw emulator throughput or transfer becomes limiting.
- **Closest prior art:** Tom Murphy VII's `learnfun`/`playfun` (SIGBOVIK 2013): emulator-as-forward-model + searched action sequences + reusable "motifs" (= our action chunks). It beats 1-1 with superhuman timing but never distilled search into a reactive net — that gap is exactly what we add.
- **Pixel-only RL is a different research question.** It discards emulator state and exact reset
  advantages, so it is not the foundation here. Reverse-curriculum RL remains an optional
  specialist experiment, not a substitute for verified search.

---

## 2. Hardware reality & the real bottleneck

The reference 2023 M2 Pro configuration has a 12-core CPU (eight performance,
four efficiency), 19-core GPU, 16 GB unified memory, and 200 GB/s memory
bandwidth. Hardware is supporting infrastructure, not the research claim.

Tracked one-shot local NES diagnostics report 1,375.7 frames/s, a 73.076 µs
isolated snapshot dump/load roundtrip, and 2,968.65 µs per
restore → four steps → child snapshot (336.9 successors/s). The isolated
snapshot fraction is only 2.46%; even removing it entirely has an Amdahl ceiling
of about 1.025x. For this workload, emulator stepping and total exact-query
count dominate. Separately timed SMA4 loops are non-additive and do not identify
a causal bottleneck.

Near-term systems policy:

- parallelize independent conformance/quotient jobs with one persistent
  emulator and opaque snapshot store per worker;
- send compact content-addressed job descriptions and merge results in a
  deterministic order;
- select worker count from repeated full-job throughput, coordinator overhead,
  memory pressure, and thermal state;
- independently replay promoted traces in a fresh serial process;
- treat 16 GB as adequate for bounded current work, not as an unlimited
  snapshot archive; and
- use CPU as the reference inference path. Select MPS only after parity,
  measured batch crossover, and end-to-end benefit. No blanket
  `torch.compile`, MPS, MLX, Core ML, or custom Metal policy is justified.

Snapshot compression, shared-tree concurrency, and custom GPU kernels are
deferred until same-loop profiling identifies them as material.

---

## 3. Environment & toolchain

**Current decision: adapter-first.** The original NES SMB1 lane still uses
`gym-super-mario-bros` 8.0.0 + `nes-py` 9.0.0. Cross-game work now goes through
`mario.adapters.GameAdapter` so each emulator only has to provide reset,
snapshot/restore, action stepping, terminal checks, and progress/cell semantics.

For NES SMB1, `gym-super-mario-bros` / `nes-py` remains the best local tool:

- Prebuilt **arm64 wheels** — `pip install gym-super-mario-bros` just works, no compiler.
- Migrated to **Gymnasium** (`gymnasium>=1.0.0`).
- Exposes **`dump_state()` / `load_state()`** — an *in-memory opaque snapshot* (cheap clone), exactly what beam search needs. (Also legacy single-slot `_backup()`/`_restore()`.)
- Direct RAM access for building observations and reward.
- Headless by default (`render_mode=None`).

For GBA and other non-NES Mario games, **Stable-Retro/mGBA is now accepted and used**.
The SMA4 / SMB3 path runs through `SMA4Adapter` and `SMA4OverworldAdapter`,
sharing one Stable-Retro core between the overworld and level views so savestates
are interchangeable. PyBoy is used for Super Mario Land experiments.

**Caveat:** emulator APIs differ. The adapter boundary is mandatory; do not thread
game-specific RAM/action assumptions through the generic solver.

**Rejected or secondary alternatives:** `FCEUX`+Lua (great TAS tool, but search must be written in Lua, no clean Python headless throughput — keep only for *verifying/replaying* finished runs); `BizHawk` (not native on Apple Silicon); `cynes`/`libretro.py`/`TetaNES` (viable fallbacks but more DIY).

**Toolchain and dependency isolation:**

- Python **3.13** environments created with `uv`.
- The common package contains Gymnasium, NumPy, Torch, image/report dependencies, and the generic
  adapter/search code.
- The NES extra contains `gym-super-mario-bros`/`nes-py`; the GBA extra contains Stable-Retro;
  the SML extra contains PyBoy.
- NES and GBA must use **separate environments**: `nes-py==9.0.0` requires
  `pyglet<=1.5.21`, while `stable-retro==1.0.1` requires `pyglet>=1.5.27,<2`.
- ROMs stay local under ignored `roms/`; FCEUX remains an optional replay viewer.

```bash
# NES / SMB1
uv venv --python 3.13 venv
uv pip install --python venv/bin/python -e '.[nes,dev]'

# GBA SMA4 + GB SML
uv venv --python 3.13 venv-gba
uv pip install --python venv-gba/bin/python -e '.[gba,sml,dev]'
```

The next throughput artifact should preserve warmups, repeated raw samples,
hardware/core/ROM/source identity, thermal state, peak memory, exact primitive
work, and serial versus worker-local full-job scaling. Microbenchmarks are
diagnostics, not an additive performance decomposition.

---

## 4. Observation design

**Decision: RAM/tile features, not pixels.** State features are far more
sample-efficient here. The implemented learners are a compact dense
`MarioPolicy` over stacked observations and entity-token Transformers over
structured player/enemy/terrain features; there is no pixel CNN.

**Observation vector = ego-centric tile grid + scalar features.**

- **Tile grid:** a window around Mario, e.g. `H×W` (start ~13×16) of small integer codes: `0` empty, `1` solid, `2` Mario, `-1` enemy, plus a few for hazards/pipes/coins. Built by reading tile RAM and enemy/Mario positions ÷16. **Use *relative* offsets** (window centered on Mario) so it generalizes across levels — never feed absolute level-x as a feature.
- **Scalars:** horizontal velocity `0x0057`, vertical velocity `0x009F`, powerup state `0x0756`, float/airborne state `0x001D`, on-ground flag, maybe time-bucket.
- **Temporal context:** raw RAM is **not fully Markov** (enemy slots get reused,
  some state is in PPU/timers). The implemented paths use a frame stack or
  `TemporalEntityTransformer`; a GRU remains an untested alternative.

**Key SMB RAM addresses** (from Data Crystal `Super_Mario_Bros./RAM_map`):

| Address | Meaning |
|---|---|
| `0x006D` | player horizontal page (level) |
| `0x0086` | player x on screen |
| `0x00CE` | player y on screen |
| `0x0057` | player horizontal speed (signed) |
| `0x009F` | player vertical velocity (signed) |
| `0x0756` | powerup state (0 small, 1 big, ≥2 fire) |
| `0x001D` | float state (0 ground, 1 jumping, 2 ledge, 3 flagpole) |
| `0x000E` | player state (climbing/pipe/dying/transforming) |
| `0x006E–0x0072` | enemy horizontal position (5 slots) |
| `0x00CF–0x00D3` | enemy y on screen (5 slots) |
| `0x0016–0x001A` | enemy type (5 slots) |
| `0x000F–0x0013` | enemy active flag (5 slots) |
| `0x0500–0x069F` | current tile/level layout grid in RAM |
| `0x071A` | current screen in level |

The env's `info` dict already exposes `x_pos, y_pos, status, life, time, world, stage, coins, score, flag_get` derived from these — use it for reward, build the tile grid ourselves.

**Failure mode:** a too-narrow tile window reproduces Tom7's "walk into the pit" failure — the agent must *see* the pit/enemy ahead before committing. Tune window width on real deaths.

---

## 5. Action representation

**Current contract: nine fixed SMB1 actions + an 8-frame default chunk.**

`SMB1_ACTIONS = [NOOP, right, right+A, right+B, right+A+B, A, left, down, up]`.
These indices are persisted in solution and training artifacts, so they must
never be reordered. `mario/actions.py` is the canonical emulator-free schema.

- **Chunking = action repeat:** the normal search default holds each chosen
  action for `k=8` frames, so a depth-`d` path spans at most `8d` primitive
  frames. Precision specialists and future variable-duration search can use a
  different declared `chunk_frames`; every artifact must record it.
- **Held-action chunks:** one decision selects one button combination and holds
  it for the declared number of frames. The stored path is a sequence of those
  action indices, not a policy-predicted multi-action motif.
- **Caveat:** fixed-duration chunks miss **frame-perfect** inputs and phase
  changes (precise jumps, wall-clips, moving platforms). Completion-oriented
  search already falls back to finer chunks where needed; the next 6-2
  experiment should compare declared durations such as `{1,2,4,8,16}` under
  equal primitive-frame and wall-clock budgets. `left`, `down`, and `up` keep
  backtracking, pipe entry, and door/vertical interactions representable.

---

## 6. The search teacher (the heart of the project)

**Algorithm: beam search over action chunks**, with A\* as an alternative/extension. The emulator is the transition model; `dump_state`/`load_state` clone nodes.

```
beam = [(snapshot_0, inputs=[], score_0)]
for depth in range(max_depth):
    candidates = []
    for (snap, inputs, _) in beam:
        for chunk in action_chunks:
            load_state(snap)
            obs, rew, done, info = run_chunk(chunk)   # k frames each
            s = evaluate(info, done)
            candidates.append((dump_state(), inputs+[chunk], s))
    beam = top_k(candidates, k=beam_width)   # e.g. 100–1000
    # keep best complete trajectory seen so far (playfun's "replay a good future")
```

**Ranking / evaluation — the most important thing to get right.** Tom7's two canonical failures define the rules:
1. **Death/lives MUST be in the objective** as a hard negative — else Mario jumps into pits because respawning one screen back still scores OK.
2. **Ignore counter-style RAM** (score, timer, music, scroll) — they give *fake monotone progress* and cause the agent to get stuck humping a wall forever (the 1-2 coin-ledge trap).

```
score = + W_progress * Δ(level_x)        # true rightward progress — the load-bearing term
        + W_flag     * reached_flag
        + W_milestone* route_milestone    # for warp/branch levels
        - W_death    * died               # HARD penalty
        - W_time     * frames_elapsed     # SMALL, and see framerules below
        - W_stuck    * no_progress_steps
```

The implementation also uses area-aware, potential-inspired progress features so a pipe transition
does not look like a backward jump. This is a practical search-ranking heuristic; the repository
does **not** claim the policy-invariance theorem of formal potential-based reward shaping.

**Framerule subtlety:** SMB rounds level time up to the next **21-frame boundary**, so shaving 1–20 frames *inside* a level saves nothing — **except 8-4** (un-ruled). So `W_time` should be ~0 during the "beat it" phase; only the **final segment** / speed phase optimizes frames.

**Curriculum (don't search full any% from scratch):**
```
1-1 → 1-2 → 1-2 warp entry → 4-1 → 4-2 warp route → 8-1 → 8-2 → 8-3 → 8-4
```
**8-4 is special-cased**: it needs wrong-warp/turnaround routing that pure rightward search won't find — left-moving chunks + milestone rewards for the correct sub-rooms.

**Failure modes:** teacher myopia (shallow beam → confident-but-wrong labels) poisons everything downstream — *fix the teacher before blaming the student*; deceptive geometry (dead-ends) needs width + backtracking chunks; local optima (re-derive Tom7's traps) need the death term + ignoring fake-progress counters.

**Interface:**
```python
class SearchTeacher:
    def solve(self, start_snapshot, max_depth, beam_width) -> Trajectory: ...
    def label(self, snapshot) -> (best_action, soft_targets, value):
        """Run a (possibly shallow) search from an arbitrary state.
        This is the DAgger query — must give GOOD labels off-distribution."""
```

---

## 7. The trajectory buffer (self-generated dataset)

Every successful (and DAgger-corrected) trajectory is logged. **No human demos, no TAS downloads — pure self-generated.**

Record per decision point:
```
{
  obs:          float array (tile grid + scalars),   # what the student sees
  hard_action:  int,                                  # argmax teacher action (hard label)
  soft_targets: float[N_ACTIONS],                     # distribution over the 9 held-action choices
  value:        float,                                # teacher's estimated value (for value-net training)
  level:        (world, stage),
  source:       "search" | "dagger_round_i",
  snapshot:     opaque bytes (optional, for re-labeling / debugging)
}
```

Store as sharded `.npz`/`.parquet` under `data/`. Cap or reservoir-sample the aggregated set so it doesn't grow unbounded across DAgger rounds.

---

## 8. The student policy (distillation)

**Implemented models:** a compact MLP over stacked dense observations and
entity-token Transformers (including a temporal variant), with optional value
heads. Each output is a distribution over the nine button combinations. Search
then holds the selected combination for the artifact's declared
`chunk_frames`; the policy does not currently emit a multi-action sequence.

**Loss: soft, chunked behavior cloning.**
- Don't clone only the single best action—when the teacher exposes comparable action values, clone
  a normalized distribution over surviving chunks. This is a search-derived soft target, not an
  AlphaZero visit-count target unless an iterative tree-search/visit process is actually used.
- A future ACT-style sequence-output policy is a separate, unimplemented
  experiment. Do not use action-chunking theory to describe the present
  single-action-held-for-\(k\)-frames controller.
- `loss = CE(student_logits, soft_targets) [+ λ·MSE(value_head, teacher_value)]`.

**Eval metrics:** completion rate per level, median x_pos reached, best time, **death-location histogram** (drives where DAgger spends budget).

**Failure modes:** non-Markov input → use the frame-stack/temporal Transformer
and test state sufficiency; cross-level distribution shift → train/eval on
multiple levels with relative features; the student can only be as good as the
teacher's labels on the *student's* distribution (→ DAgger).

---

## 9. DAgger correction loop

Plain BC can fail when one bad jump reaches a state absent from the demonstrations. Classical
worst-case analyses show the familiar quadratic-horizon compounding-error bound for naive BC and
a linear-horizon bound for interactive imitation under their assumptions. DAgger addresses that
distribution mismatch by collecting labels on the **learner's induced state distribution**; it
does not itself guarantee cross-level generalization.

```
1. Train π₁ by BC on the search-generated dataset.
2. For i = 1..N:
   a. Roll out a mixed policy β_i·teacher + (1-β_i)·π_i  (β_1=1, geometric decay → 0).
   b. Collect visited states; FOCUS on states where π is uncertain or just died.
   c. Query the teacher (search) for the correct label at those states.   # cheap: deterministic sim
   d. Aggregate D ← D ∪ D_i  (keep all past data — "follow the leader").
   e. Retrain π_{i+1} on D.
3. Return best π_i on a validation set of levels.
```

**Why search-as-expert + DAgger is an operational match:** DAgger needs a queryable expert at
learner-visited states. Snapshot search can label such states (`label(snapshot)`), subject to its
own horizon and search-budget errors.

**Uncertainty gating (spend search budget wisely):** don't re-label every frame. Query the teacher only where the policy is unsure — SafeDAgger (a learned "is the policy safe here?" classifier) or DADAgger (ensemble disagreement). Natural fit: re-label clustered around death-histogram hotspots.

**Pitfalls:** teacher must give *good* off-distribution labels (deepen search at hard states); aggregated set staleness (cap/reservoir); immature policy wandering into junk (cheap in a game — just death — but wastes budget).

**Variants to know (optional reading):** DAgger-by-coaching (label with achievable-better actions when the teacher is far stronger than the tiny net), HG-DAgger (gated intervention).

---

## 10. Learned guidance and route-cost optimization

Policy/value guidance is implemented, but the evidence is deliberately narrow:

- On SMB1 1-1 at width 6/top-3, one local entity-policy prior reduced expansions from 7005 to 2770
  while both configurations solved. That is a **single-level search-guidance observation**, not an
  AlphaZero/Levin/PHS result and not evidence of a universal 2.53
  plain/guided ratio. The implementation adds the current edge's log prior to a
  fresh state score; it does not accumulate path log probability.
- Hard top-k pruning is incomplete: the same weak prior can discard the successful action at a
  tighter top-k. Prefer soft priors, a uniform fallback, entropy-adaptive mixing, or
  completeness-safe policy-guided heuristic search.
- A learned value may order a frontier, but it must be evaluated under paired node/wall-clock
  budgets and a solve-rate non-inferiority gate. A misleading value can prune the only viable
  route.
- Frame cost is a later optimization objective. SMB framerules and emulator-measured duration must
  be reported explicitly; completion or an improved route is not automatically a speedrun record.

The next guidance benchmark spans at least six level types and multiple starts/seeds, comparing
plain search, static/random priors, fixed top-k, soft log-priors, uniform mixtures,
entropy-adaptive pruning, and a PHS/Levin-style safe alternative. Use bootstrap intervals and
solve-rate preservation rather than a single timing.

---

## 11. Cross-game and option architecture

`mario.adapters.GameAdapter` defines reset, snapshot/restore, action stepping, progress/cell
semantics, and terminal reporting. NES imports are lazy so GBA/SML-only environments can import the
shared solver without installing `nes-py`.

At the route layer:

1. A complete physical record binds exact emulator/wrapper state, adapter
   context, source identity, parent lineage, and intervention evidence.
   `MetaState` separately annotates world/map position, clears, inventory, and
   flags; it is neither physical identity nor controller phase.
2. The current `Option` is a **legacy orchestration wrapper**: it combines a
   symbolic precondition, arbitrary Python runner, result, estimated/measured
   cost, and provenance. It does not implement an explicit finite-state
   controller.
3. `OptionContext` has two current contracts:
   - strict compatibility mode maps a symbolic state to one snapshot and raises
     `StateAliasError` on a raw-distinct candidate;
   - opt-in multi mode retains immutable in-run physical records, exact-record
     deduplication, all arrivals, observable/context evidence, and explicit
     record restoration.
4. The proposed additive `OptionMachine`
   \(\Omega=(id,r,I,Q,q_0,Q_{\mathrm{term}},\pi,\delta,\beta)\) makes finite
   controller phase explicit. During execution the Markov state is
   `(physical_state, phase)`. The first serializable profile is a canonically
   expanded open-loop primitive trace whose phase is the primitive step index.
5. A proposed `OptionTrace` v3 records every retained primitive
   `(phase_before, action, phase_after)` transition, exact frames and route
   weight, entry/exit physical records, lineage, evaluation work, reset access,
   interventions, and independent replay attestations. It is specified in
   `notes/theory/option-machine-trace-v3.md` and is **not implemented**.
6. A symbolic transition deliberately drops physical identity. A later
   physical option fails closed rather than borrowing the first snapshot with a
   matching `MetaState`.
7. Legacy greedy/BFS/UCS remain available; opt-in physical BFS/UCS key frontier
   and dominance by `(MetaState, physical_record_id)`. Bounded UCS uses
   nondominated `(cost, depth)` labels.
8. Current post-run refinement builds a v2 finite boundary-observation table, repeats enabled
   transitions, checks normalized behavior plus finite physical-exit evidence,
   propagates successor blocks to a fixed point, and emits content-addressed
   classes and distinguishing option suffixes. It does not contain complete
   primitive action/phase traces and is not a v3-certified quotient.

The SMA4 implementation has real level/overworld executors and replayable 1-1/1-2 segments.
One declared 1-2 root now reaches the real unpowered fortress through live
`DOWN,DOWN,LEFT -> (96,96)` input with zero direct RAM writes. The whistle route still restores
independent power-state roots and applies an inventory merge plus a fortress
leaf rehold. No cursor intervention remains.

The boundary audit's abstraction counterexample is now preserved rather than
route-destructive. Both acquisition orders survive through the second whistle.
The finite option table originally appeared to distinguish their first- and
second-whistle states, but this was a decoder artifact: the 1-3 exit relocates
the live cursor object from `0x03003DE0` to `0x03004EF8`, while the old fixed
pair stays stale. Pointer-resolved observations make both histories agree on
reachability and pass the matched `DOWN`/NOOP World-8 responsiveness test.
Their post-first-whistle records remain cost-distinct (1,302 versus 1,290
frames for the second spend), while the post-second-whistle and World-8 record
pairs share finite refined blocks.

Refinement currently reports classes; it does **not** merge the live frontier.
Completeness and closure mean only the encountered records, finite option
implementation, ROM/core, tier, and depth. They are not global bisimulation or
determinism proofs. The initial coloring includes exact `MetaState`, so
different symbolic states cannot merge even when the finite option alphabet
does not distinguish them; the 11 classes are not a coarsest quotient. Until
the live unpowered entry receives legitimate power
and both acquisitions share one uninterrupted lineage, call this a
**segmented option-planning prototype**. Report retained ROM frames,
rolled-back evaluation work, symbolic endpoint costs, attempted versus
selected interventions, and physical versus symbolic path evidence separately.

The research schema must keep five objects distinct: complete physical state
`x`, task abstraction `z`, lineage `h`, epistemic option-effect knowledge `e`,
and intervention ledger `i`. Route cost is retained controller time; search
cost is the vector of evaluated primitive frames, option calls, wall time, and
memory. One-root, no-write, no-symbolic-edge, physical-terminal, and independent
replay requirements are feasibility constraints, not costs that can be traded
away.

For a future v3 table, let \(D_X\subseteq X\) be the declared finite physical
boundary domain and let \(B_{\mathrm{failure}}\) be disjoint typed failure
symbols. The closed carrier is
\(\widetilde D=D_X\uplus B_{\mathrm{failure}}\); failure symbols are formal
zero-weight absorbing states, not emulator states. Over that carrier and a
fixed terminating machine library, block members must agree on task labels,
enabledness, outcome/termination (including terminal phase only when declared
as output), exact retained weight under a hashed additive algebra, and
successor block or typed failure sink. Each machine's phase is internal; the
ordinary option-boundary domain is physical state, not one global phase space.
Under fixed deterministic dynamics, that yields a well-defined weighted
quotient preserving option-word reachability and accumulated weight inside the
declared carrier. This is not a global ROM, stochastic SMDP, causal abstraction,
or automatic-discovery theorem. Missing rows, conflicting repeats, unknown
boundaries, and replay-invalid lifting fail closed.

---

## 12. Repository layout

```
mario_ai/
  DESIGN.md                  # this file
  pyproject.toml             # common package + isolated NES/GBA/SML extras
  mario/
    env.py                   # NES wrapper and exact cache-aware restore
    adapters.py              # NES, SMA4 level/overworld, and SML boundary
    search.py                # native + adapter beam/coverage search
    solution_verification.py # reusable replay gate
    observation.py           # RAM → tile grid + scalars
    entity.py                # structured entity observation
    policy.py                # compact tile policy
    entity_policy.py         # entity/temporal models and search priors
    consistency.py           # input perturbation consistency (not Stable-BC)
    provenance.py            # deterministic state/artifact hashes and alias errors
    options.py               # option contracts, executors, strict context/BFS
    physical_planner.py      # multi-record BFS/UCS + finite partition refinement
    meta_planner.py          # greedy/uniform-cost SMA4 planning experiments
  scripts/
    solve_all_stock.py
    verify_stock_solutions.py
    policy_guided_search.py
    diagnose_sma4_fortress_entry.py
    solve_sma4*.py / solve_sml.py
  data/solutions/            # small canonical replay manifests
  data/, runs/               # large generated artifacts (ignored)
  tests/                     # deterministic, learning, adapter/option, CLI gates
```

---

## 13. Milestones and current gates

The original V0–V5 sequence is preserved as project history:

| Ver | Historical learning unit | Durable outcome |
|---|---|---|
| **V0** | Beat 1-1 with pure search | exact snapshots, action chunks, ranking |
| **V1** | Generate search-labelled data | trajectory/data contracts |
| **V2** | Train compact policies | distillation and observation design |
| **V3** | Query search on learner states | DAgger and covariate-shift failures |
| **V4** | Solve the full any% route | mechanic-aware routing and replay composition |
| **V5/V6** | Test generalist control and learned guidance | useful standalone negatives; local 1-1 prior win |

The current program has one shared contract and two parallel branches.

### Shared Gate S0 — representation and accounting

1. Preserve passed Gate-0A semantics: retained-only novelty, separate loop
   evidence, real macro accounting, source-bound provenance, and precise
   coverage labels.
2. Add v3 schemas, synthetic adversarial fixtures, and opt-in primitive
   action/phase tracing without changing legacy v2 artifacts.
3. Separate retained route cost from discovery, conformance, attestation, wall
   time, and memory. Label full-snapshot, root-only, and no-arbitrary-reset
   access.
4. Require serial, cached, job-reordered, and worker-local results to be
   identical before accepting parallel evidence.

### Branch A — primary portable science

1. Recover known minimal weighted quotients on blinded generated systems with
   aliases, delayed effects, noncommutative options, variable cost, and
   controller phase.
2. Compare exhaustive, random, and counterexample-guided distinguishing
   suffixes. Count learner membership, hidden benchmark-oracle, and
   certification/conformance queries separately.
3. Run matched BFS/UCS, novelty beam, real Go-Explore, IW/BFWS, restarting
   walks, full-support Levin/PHS, explicit options, structure-induced
   rerooting, and a bounded PUCT control.
4. Repeat under full, root-only, and no-arbitrary-reset access.
5. Run a pre-outcome license/replay/adapter spike on the predeclared MiniHack
   and Crafter candidates. Require redistribution and exact-seed replay; choose
   fewer adapter-specific lines, then lower baseline runtime as tie-breaker.
   Freeze the winner, tasks, and harness before outcome-bearing evaluation.
6. Only after those gates let learned/programmatic/foundation models propose
   predicates, phases, options, or queries; exact execution remains the
   authority.

Pass criteria:

- zero false merges and zero replay-invalid lifted plans inside a declared exact
  finite domain;
- at least 2x query/state compression or 20% end-to-end planning savings;
- active selection uses at least 2x fewer learner membership queries than
  exhaustive construction or, over at least 30 blinded paired seeds, at least
  25% fewer than random with a paired 95% bootstrap interval for the ratio
  below 1.0; hidden scoring/certification queries are reported separately;
- no level IDs, absolute-coordinate patches, or manual predicate per
  counterexample; and
- learned guidance preserves solve rate and saves at least 20% in both primitive
  work and wall time, not nodes alone.

### Branch B — bounded SMA4 integrity case

Attempt one earliest World-1 root, legitimate power, two whistle acquisitions,
zero direct writes, no unrelated restores, a physical World-8/Bowser endpoint,
and fresh-process primitive replay. Stop after two predeclared
legitimate-power approaches or 40 recorded ROM-backed process-hours, whichever
comes first. Success makes SMA4 a strong case study; failure is preserved as a
negative composition result and does not block Branch A.

The finite refinement algorithm remains supporting infrastructure unless
automatic query selection, reset ablations, planning savings, or cross-domain
results establish a nontrivial contribution.

---

## 14. Open questions / decisions deferred

- What is the smallest synthetic fault domain that exposes phase erasure,
  delayed effects, noncommutativity, and representative substitution without
  encoding the answer in hand-written labels?
- Which active query rule finds shortest useful distinguishing suffixes with
  the best exact-interaction complexity?
- Can controller phase or task predicates be proposed automatically while
  exact execution remains a fail-closed authority?
- How much does quotient discovery degrade under root-only and no-arbitrary-
  reset access?
- Which legally redistributable second domain can use the frozen adapter,
  trace, and evaluation contracts without domain-specific repairs?
- Which platform/enemy phase variables make 6-2 search state sufficiently Markov?
- Do variable action durations `{1,2,4,8,16}` improve moving-platform search under equal budgets?
- Can PHS/PHS* or Levin-style guidance retain completeness while capturing the local prior gain?
- Can active counterexample suffixes and held-out tests safely justify any
  online refined-class merge, or should physical identity remain the permanent
  frontier key?
- Which legitimate World-1 item/power route gives one continuous lineage
  through both whistle acquisitions without an external restore or write?
- Which additional cursor-object bases, if any, appear on new roots, and can
  the adapter recognize them from structure rather than a growing allowlist
  without weakening the fail-closed boundary?
- Does unknown-option planning still beat greedy after both receive the same learned/cached effect
  model and cost definition?

---

## 15. Key references

- Baumgarten A\* / Mario AI Competition; Karakovskiy & Togelius survey — historical
  forward-model planning evidence.
- Tom Murphy VII, *learnfun/playfun* (SIGBOVIK 2013), tom7.org/mario — closest prior art; objective-design failure modes.
- Ross, Gordon, Bagnell, *DAgger* (AISTATS 2011) — train on learner's induced distribution.
- Anthony et al., *Expert Iteration* (2017) — iterative search/imitation framework; an analogy,
  not the name of the current one-pass prior experiment.
- Zhao et al., *ACT* (2023) — future sequence-output/action-chunking lead, not
  the architecture currently implemented here.
- Orseau and Lelis, *Policy-guided Heuristic Search with Guarantees* (AAAI 2021) — a direct
  candidate for safe learned guidance.
- Chatterjee and Khardon, *Improving Planning and MBRL with Temporally-Extended
  Actions* (NeurIPS 2025) — relevant to moving platforms and P-speed.
- Sutton, Precup, Singh, *Between MDPs and Semi-MDPs* (AIJ 1999), and
  Ravindran/Barto, *SMDP Homomorphisms* (IJCAI 2003) — exact option and
  abstraction boundaries.
- Abel et al., *Value Preserving State-Action Abstractions* (AISTATS 2020) —
  judge the abstraction jointly with the policies its option set can express.
- Littman, Sutton, Singh, *Predictive Representations of State* (2001), and
  Clarke et al., *CEGAR* (2000) — controlled suffix tests and
  counterexample-guided refinement.
- Vaandrager/Melse (CONCUR 2025), Giraud et al. *L-SCALE* (AST 2026), and
  Turkenburg et al. (CSL 2026) — scoped finite conformance suites, snapshot-
  guided active testing, and quantitative behavioral witnesses.
- Guez, Silver, and Dayan, *BAMCP* (NeurIPS 2012) — belief/history baseline for unknown options.
- Castro and Precup, *Using Bisimulation for Policy Transfer in MDPs* (AAAI
  2010) — direct mathematics for testing whether physical states may safely
  share an option-level abstraction.
- Givan, Dean, and Greig, *Equivalence Notions and Model Minimization in MDPs*
  (AIJ 2003), and Castro, Panangaden, and Precup, *Equivalence Relations in
  Fully and Partially Observable MDPs* (IJCAI 2009) — recursive
  successor-block refinement and the limits of finite traces.
- Ahmetoglu et al., *Skill-Driven Neurosymbolic State Abstractions* (NeurIPS
  2025) — construct state around the supplied option set.
- Konidaris, Kaelbling, and Lozano-Pérez, *From Skills to Symbols* (JAIR
  2018), Ni et al., *Bridging State and History Representations* (ICLR 2024),
  and Xia and Bareinboim, *Causal Abstraction Inference under Lossy
  Representations* (ICML 2025) — option-induced, predictive, and
  multiple-realization views of the corrected cursor abstraction.
- Abel et al., *Near Optimal Behavior via Approximate State Abstraction*
  (ICML 2016) — approximate merging only after exact finite signatures and a
  declared error budget.
- Angluin, *Learning Regular Sets from Queries and Counterexamples* (1987),
  Wißmann et al., *Explaining Behavioural Inequivalence* (CONCUR 2021), and
  Giraud et al., *L-SCALE* (AST 2026) — active distinguishing suffixes over a
  resettable system; approximate hashing is not a safety gate here.
- Fortz et al., *A Research Agenda for Active Automata Learning* (STTT 2026) —
  make the teacher, access, query, fault-domain, and guarantee assumptions
  explicit.
- Zhang, Luo, and Baltieri, *Compositional Behavioral Semantics for State
  Abstraction in Reinforcement Learning* (ICML 2026) — specify which behavior
  a proposed quotient must preserve.
- Tuero et al., *Structure-Induced Information for Rerooting Levin Tree Search*
  (ICML 2026) — matched alternative to reconstructing explicit subgoals.
- Chang et al., *The Surprising Difficulty of Search in Model-Based
  Reinforcement Learning* (ICML 2026) — learned-model accuracy alone does not
  establish search benefit.
- Nixon, *The Myhill-Nerode Theorem for Bounded Interaction* (2026 preprint) —
  close formal lead and novelty-collision warning; do not transfer its
  finite-POMDP claims without a new weighted-option proof.
- Nayyar and Srivastava, *Autonomous Option Invention for Continual
  Hierarchical Reinforcement Learning and Planning* (AAAI 2025), and
  Macfarlane et al., *Gradient-Based Program Synthesis with Neurally Interpreted
  Languages* (ICLR 2026) — controls for manual versus invented symbolic and
  programmatic options.
- Taheri et al., *BarrierBench* (L4DC 2026) — precedent for model proposal plus
  formal validation, not a direct Mario theorem.
- Bai, Srivastava, and Russell, *Markovian State and Action Abstractions for MDPs via Hierarchical
  MCTS* (IJCAI 2016) — history/representative planning when abstraction induces non-Markov state.
- Data Crystal — *Super Mario Bros. RAM map*.
- gym-super-mario-bros / nes-py (Kautenja) — NES environment and snapshot API.

The annotated, date-checked bibliography is `notes/research-bibliography.md`.
