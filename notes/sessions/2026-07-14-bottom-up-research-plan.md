# 2026-07-14 — Bottom-up rethink: theory, challenges, experiment series (SMA4/SMB3 on M2)

> Deep dive from first principles. Question every finding. Plan the next experiment
> ladder for Super Mario Advance 4 / Super Mario Bros. 3 on a MacBook M2, with a
> clean path that also covers SNES-family Mario (SMA2 / SMW ROM already local).
>
> This note is the planning artifact. It does **not** claim new experimental results.
> Every claimed prior result cites an existing artifact or paper.

---

## 0. One-sentence thesis (to stress-test)

**Given a deterministic, resettable emulator, the correct architecture is a
two-tier Option-SMDP: exact low-level search produces verified options; meta-search
over options discovers consumable foresight (whistles) that myopic planners miss.**

Everything below either supports, measures, or tries to falsify that sentence.

---

## 1. Bottom-up stack (what the math actually says)

### 1.1 Layer 0 — Deterministic dynamics

$$
s_{t+1} = f(s_t, a_t)
$$

For NES/GBA Mario via `nes-py` / Stable-Retro+mGBA, \(f\) is known, cheap, and
**resettable** via opaque savestates. This single fact collapses most of modern
"learn a world model" RL (MuZero / Dreamer / Genie) to **out of scope**: those
exist to approximate a missing or expensive \(f\). We already own \(f\).

**Hardware corollary (M2):** the bottleneck is **CPU emulator + snapshot**, not
MPS. Measured on this machine for SMB1/nes-py (`bench/step_rate.json`,
`bench/snapshot_cost.json`):

| Metric | Value |
|---|---|
| step fps | 1375.7 |
| snapshot round-trip | 73.076 µs |
| search nodes/s (chunk=4) | 336.9 |

**No committed SMA4 throughput bench exists.** That is Experiment A0 below.
Stable-Retro is **one core per process** — parallelize *across processes*, never
inside one beam.

### 1.2 Layer 1 — MDP / search as planning

Finite-horizon planning over \(f\):

$$
\pi^\*(s_0) = \arg\max_{a_{0:T}} \; R\!\left(s_0, a_{0:T}\right)
$$

with beam / coverage / Go-Explore as approximate solvers under branching factor
\(|A|^d\). Mario AI Championship history (Baumgarten A\*, Togelius/Karakovskiy)
already settled the empirical claim: **with a forward model, search beats
standalone learning for Mario**.

Local confirmation: SMB1 any% `beat_game=True` via search; 31/32 stock levels
solved; SMA4 1-1 and 1-2 replay-verified via adapter search.

### 1.3 Layer 2 — Reward / potential (why Φ matters)

Ng, Harada, Russell (ICML 1999): potential-based shaping

$$
F(s,a,s') = \gamma\,\Phi(s') - \Phi(s)
$$

is necessary and sufficient for **policy invariance**. Non-potential shaping
(fake counters, score, timer) can create suboptimal attractors — Tom7's classic
"hump the wall forever" failure.

Our SMB1 global progress

$$
\Phi = \texttt{area\_seq}\cdot 10000 + (x - x_{\text{entry}})
$$

is exactly a potential on the area-augmented state. **SMA4 has not yet been
given an equally careful Φ.** Current adapter search mostly uses raw `x_pos`,
which *caps* near the card — hence `goal_suffix_search`. That is a symptom of a
broken progress coordinate, not proof that finisher templates are fundamental.

### 1.4 Layer 3 — Options / SMDPs (the SMB3 object)

Sutton, Precup, Singh (AIJ 1999): an option \(o = \langle \mathcal{I}, \pi, \beta \rangle\)

- \(\mathcal{I} \subseteq S\) — initiation set
- \(\pi(a|s)\) — intra-option policy (here: cached search path / executor)
- \(\beta(s) \in [0,1]\) — termination

A set of options over an MDP induces an **SMDP**. Planning over options is
planning in that SMDP:

$$
Q_\mu(s,o) = r_o(s) + \sum_{s',k} p_o(s',k|s)\, \gamma^k V_\mu(s')
$$

**What we have today:** a pragmatic Option-*scaffold* (`mario/options.py`) with
`MetaState`, knowledge tiers, and opaque-effect discovery by execution.

**What we do *not* yet have (theory gap):**

1. Explicit initiation / termination sets for white-block duck, flight, inventory use.
2. Learned or measured option models \(p_o, r_o\) (multi-time models).
3. Intra-option learning (unused — and probably unnecessary while search fills \(\pi\)).
4. Interruption / call-and-return composition under real ROM costs.

Dietterich's MAXQ and Feudal RL (Dayan/Hinton; FuN, Vezhnevets 2017) are
*learned* hierarchies. Our setting is stronger: **search-defined options with
verified effects**. Prefer the SPS / Options formalism over FuN for this repo —
we do not need a Manager network until option discovery itself is the bottleneck.

### 1.5 Layer 4 — Expert Iteration / "net serves search"

Anthony et al. (ExIt), AlphaZero: iterate

$$
\text{search}(\pi,V) \;\Rightarrow\; \text{targets} \;\Rightarrow\; \text{train }(\pi',V') \;\Rightarrow\; \text{guide search}
$$

V6 measured policy-guided beam on SMB1 1-1: **2.53× node cut** at top-3
(`scripts/policy_guided_search.py`). Value guidance was functional but gave
**no** node savings where Φ already solves.

**Challenge:** this win is (a) SMB1-only, (b) easy level, (c) weak prior.
It has **not** been shown on SMA4, coverage search, or deceptive geometry.
Treat "AlphaZero-with-a-real-sim" as a *program*, not a settled SMA4 fact.

### 1.6 Layer 5 — Imitation theory (what V5/V6 got right and wrong)

| Claim in repo | Theory check (2024–2026) | Verdict |
|---|---|---|
| Offline BC fails at long horizon / death cliffs | Foster, Block, Misra NeurIPS'24: LogLoss BC can be horizon-*independent* when payoff range + SL complexity are controlled; offline/online gap smaller than folklore | **Partially overstated.** Absorbing deaths make payoff range \(\sim H\); that *is* hard. But the paper rehabilitates BC under log loss — our ceiling is partly **data thinness + non-realizability**, not destiny. |
| "Distillation dominates RL under determinism" | Song et al. NeurIPS'25 *To Distill or Decide?*: distillation competitive under **deterministic** latent dynamics; degrades with latent stochasticity; **optimal latent policy is not always best to distill** | Supports preferring distillation over RL for our det. ROMs — but also says **don't distill the optimal brittle expert**; distill a smoother one. V6 never tested that. |
| Procgen ⇒ need \(10^3\)–\(10^4\) levels for zero-shot | Cobbe et al.; scaling laws for *level generalization* | Holds for a **flat generalist controller**. Does **not** apply to the Option-SMDP thesis (different problem). |
| Manipulation scaling law (ICLR'25) | Wrong domain for Mario levels | Correctly retracted in V5. |

**Revised IL stance:** do not revive a generalist Mario policy as the main line.
If a reactive policy is needed, distill a *smoothed* search expert (Song) or
finish reverse-curriculum RL with plasticity fixes (BBF) — both secondary to
SMA4 options.

### 1.7 Layer 6 — Exploration archives (Go-Explore)

Ecoffet et al. (Nature 2021): **first return, then explore**.

1. Archive cells + trajectories (detach-proof memory).
2. Return (reset) before exploring (derailment-proof).
3. Optional Phase 2: robustify under stochasticity.

Our `coverage_search` / `coverage_search_adapter` are Phase-1 cousins.
For deterministic GBA ROMs, **Phase 2 is mostly unnecessary** — cached
replay-verified options *are* the product. The historical mistake (V5) was
substituting BC for Phase 2 on SMB1 when the product should have stayed
search+rescue.

---

## 2. Question every major finding (adversarial audit)

### F1 — "Search is the solver; learning only accelerates"
**Keep, with scope.** True for known deterministic levels with good Φ / cells.
False as a blank check: without the right cell key / subgoal / action set,
search empties (SMA4 1-2 surface at x=629; 1-3 card; white-block). Learning is
optional; **representation of progress and options is not**.

### F2 — "Standalone IL caps ~0.45–0.5 on 1-1; DAgger degrades"
**Keep as local empirical fact; weaken as theory.** Reproduced in V6 under
specific recipes. Foster says offline IL need not be doomed; Song says distill
smoother experts. Do **not** spend M2 cycles re-fighting this until SMA4
AcquireWhistle is done. Mark as "closed for product; open for science."

### F3 — "Policy-guided beam = 2.53×"
**Keep as SMB1 1-1 measurement.** Do not cite as SMA4 readiness. Port only after
SMA4 has enough solved trajectories to train a prior, and only if search wall-clock
dominates (it will).

### F4 — "Value guidance doesn't help when Φ works"
**Keep.** Consistent with potential-based shaping: if Φ already orders the beam,
\(V\) is redundant. Value earns its keep on **deceptive** levels (6-2, fortress
flight, white-block) — untested.

### F5 — "Whistle planner-class gap proves meta-intelligence"
**Challenge hard.** Artifacts:

- Symbolic: `runs/20260629-175809-whistle-planning-bench/report.json` (10.95×)
- ROM-backed spend: `runs/20260629-202831-sma4-whistle-rom-bench/report.json` (7.75×)

Honesty holes still open:

1. Inventory is Tier-1 **hand-granted** (`grant_two_whistles`, re-grant in warp zone).
2. Warpless world advance + Bowser are **symbolic** endpoints.
3. Cost model mixes real spend frames with symbolic 9000-frame world hops.
4. Greedy is defined to **never explore opaque options** — so the gap is partly
   by construction (useful as a baseline, not as a theorem about all myopic agents).

**Thesis still interesting** — but the experiment is incomplete until
`AcquireWhistle` is Tier-2 black-box and warpless/Bowser are ROM-backed or
explicitly scoped out of the claim.

### F6 — "1-2 needed P-speed, not flight"
**Keep.** Ladder of artifacts is clean (surface → physics → pspeed → card return).
Good template for how to pin walls.

### F7 — "1-3 wall is card timing, not whistle"
**Keep for the *normal* route.** The *whistle* route is a different MDP objective
(subgoal: background layer + Toad house), deliberately not pursued. Conflating
"can't finish card" with "can't get whistle" would be a category error.

### F8 — "`(160,64)` is fortress"
**Already corrected in notes; cache name still lies.** Rename mentally to
"side athletic node"; do not build fortress flight experiments on that snapshot.

### F9 — "Zero-shot generalist from 32 levels is open / infeasible"
**Keep for flat policies.** Irrelevant to the Option-SMDP deliverable. Do not let
Procgen dictate SMA4 priorities.

### F10 — "MPS / tiny nets are fine on M2"
**Keep for training.** Search remains CPU-bound. Prefer multiprocess emulator
workers over bigger nets. Avoid Rosetta Python (repo already warns).

---

## 3. Current end-to-end reality (SMA4)

```
boot → overworld discover → enter level → beam → coverage → goal_suffix
         │
         ├─ 1-1 ClearLevel (Tier-0, verified)
         ├─ 1-2 ClearLevel (Tier-0, verified, P-speed + card-return)
         ├─ 1-3: reaches card area, endwalk=0  OR  (separate) white-block whistle
         ├─ side node (160,64): early gap wall (not fortress)
         └─ whistle spend: ROM-backed use×2 + W8 pipe; inventory still poked
```

**Second whistle in World 1** (fortress raccoon flight) is a harder Acquire option
than 1-3 white-block — plan both, sequence 1-3 first.

Local ROMs present: SMA4 (SMB3), SMA2 (SMW), SML — so "SNES → SMB3" can mean
SMA2 adapter later without hunting files.

---

## 4. Experiment series (ordered, falsifiable, M2-shaped)

Design rules for every experiment:

1. **One primary question**, one primary metric, named artifact path under `runs/`.
2. **Falsification condition** written *before* running.
3. Prefer **CPU search + RAM probes**; MPS only for tiny nets if needed.
4. Parallelize with **process pools** (one Stable-Retro env each), never threads.
5. Knowledge tier recorded on every option (`KnowledgeTier`).

### Phase A — Ground truth & throughput (do not skip)

| ID | Question | Method | Success | Fail |
|---|---|---|---|---|
| **A0** | What is SMA4 step/snapshot/node rate on this M2? | Microbench mirroring `bench/{step_rate,snapshot_cost}.json` via `SMA4Adapter` | Commit `bench/sma4_step_rate.json` + `bench/sma4_snapshot_cost.json` | If <~50 nodes/s, redesign macros/chunk before hard levels |
| **A1** | Is `0x03002C52` a real clear bitmap? | Clear 1-1/1-2, diff bits; compare to map nodes | Document bit↔node map or demote field | Keep as opaque progress only |
| **A2** | Background-layer / white-block RAM? | Probe duck-on-white (manual or scripted) vs Data Crystal / Karisa / Southbird | Find durable flag (layer/behind-scenery / duck timer) | Fall back to behavioral detector (Y drop through block + x past end) |
| **A3** | SMA4 Φ that doesn't die at the card | Potential over `(room, x_fixed, endwalk, goalcard_phase)` | Beam without suffix solves 1-1; suffix becomes optional | Keep suffix; document irreducible cap |

**M2 note:** A0–A2 are minutes-to-hours; pure RAM. Highest ROI.

### Phase B — Complete World-1 *control* options

| ID | Question | Method | Success | Fail |
|---|---|---|---|---|
| **B1** | Can 1-3 normal route clear (`endwalk≠0`)? | From `runs/sma4_cache/1-3_entry.pkl`, retune `goal_suffix_search` + P-speed prefix (1-2 recipe) | `data/solutions/sma4/1-3.json` `replay_verified=true` | Classify as timing-oracle need; freeze |
| **B2** | First `AcquireWhistle` (1-3 white-block) | Subgoal search: reach white block → hold DOWN ≥5s → stay in background → run past end → Toad chest; wrap as Tier-2 option | Inventory gains `0x0C` without RAM poke; option replay-verified; knowledge_tier=2 | If only works with Tier-3+ hints, record min tier honestly |
| **B3** | Second W1 whistle (fortress flight) | Need *correct* fortress entry snap + Raccoon + fly-to-hidden-door; Tier-2/3 | Second AcquireWhistle option | Defer; keep one-whistle partial thesis |
| **B4** | Replace hand-grant in ROM bench | Rebuild `build_sma4_whistle_rom_library` with B2(+B3); re-run `bench_sma4_whistle_rom.py` | Same planner-class gap **without** `whistle_hand_granted` / regrant facts | Gap collapses → thesis needs revision |

**This is the critical path.** CLAUDE.md's next action (AcquireWhistle) sits at B2.

White-block mechanics (route data, for option design):

1. Reach white block near end of 1-3 (red Koopa on it).
2. Disable/kill Koopa.
3. Stand on block, hold Down ~5 game seconds → fall into background.
4. Run right behind scenery into Toad house → chest → whistle `0x0C`.

This is **anti-greedy** under x-progress Φ: ducking and delaying look like
negative progress. Option initiation must fire on a **subgoal cell**, not Φ.

### Phase C — Honest meta-benchmark

| ID | Question | Method | Success | Fail |
|---|---|---|---|---|
| **C1** | Planner-class gap with real Acquire options | Greedy vs UCS/BFS on library from B4 | Artifact shows skip discovery + cost ratio; injected_facts ⊆ {option endpoints} | If greedy also finds skip once effects are known, separate "exploration of opaque" from "exploitation" |
| **C2** | ROM-backed warpless stub removal | Either (i) scope claim to "W1→W8 via whistles" only, or (ii) solve minimal warpless prefix | Written claim matches library | Mixed symbolic/ROM claim remains — label as such forever |
| **C3** | Option model quality | Estimate empirical \(k, r_o\) from N≥5 resets per option | Cost variance <20% or documented heavy-tail | Non-stationary costs → keep worst-case UCS |

### Phase D — Search acceleration (only after B stalls on wall-clock)

| ID | Question | Method | Success | Fail |
|---|---|---|---|---|
| **D1** | Multiprocess SMA4 search on M2 | N processes × independent entry snaps (not threads) | Near-linear speedup to ~performance cores | GIL/emu lock → document ceiling |
| **D2** | Policy prior for SMA4 P-speed levels | Distill soft labels from solved 1-1/1-2(/1-3) into tiny net; guide `beam_search_adapter` | ≥1.5× node cut, still solves | No cut → keep macros/action sets |
| **D3** | Cell-key ablation for athletic/fortress | Physics cell vs (x,y,powerup,pspeed,layer) | Solves previously stuck node | Need flight macros / Tier-3 subgoals |
| **D4** | JaffarPlus / QuickerMGBA spike (optional) | Read-only spike: can QuickerMGBA beat Stable-Retro nodes/s on M2? | If ≥3×, consider C++ worker later | Stay on Stable-Retro; not worth port |

### Phase E — Cross-game SNES-family (SMA2 / SMW) — after C1 green

| ID | Question | Method | Success | Fail |
|---|---|---|---|---|
| **E1** | SMA2 adapter parity | Clone SMA4 adapter pattern for SMW GBA ROM already in `roms/` | Boot→level→snapshot→beam smoke | ROM/integration wall |
| **E2** | Shared Option interface across games | Same `Option`/`MetaState` with game-specific executors | One meta-search API, two games | Premature abstraction — keep per-game |

### Phase F — Explicitly deprioritized (do not start until A–C done)

- Flat generalist entity-transformer / held-out-of-32 revival.
- Temporal transformer retries (V6 negative).
- In-loop LLM/VLM control (VideoGameBench).
- World-model RL (Dreamer/MuZero).
- Full warpless SMB3 any% grind without options.
- NES 6-2 (only if returning to SMB1 completionism).
- `rc_rl.py` plasticity engineering (science track only).

---

## 5. Recommended execution order on this MacBook

```
A0 → A2 → B2 → A1/A3 → B1 → B4 → C1
              ↘ B3 when fortress snap is real
D1 in parallel whenever a search exceeds ~30 min wall
D2 only if B-series is solve-bound by nodes, not by missing mechanics
E1 after C1
```

**Why B2 before B1?** The project thesis is Option-MDP / whistle foresight, not
World-1 completionism. A normal 1-3 clear is useful (Tier-0 ClearLevel) but
AcquireWhistle unblocks the *claim*. Run B1 the same day if B2 is blocked on RAM.

**Compute budget intuition (M2 Pro-class):**

- RAM probes / option wiring: interactive.
- Single level solve: minutes–few hours (1-2 history).
- Multiprocess sweeps: overnight, plugged in, avoid thermal throttle.
- Tiny MPS nets: seconds–minutes; never the critical path.

---

## 6. Minimal math checklist for implementers

When adding an option \(o\):

1. **Initiation:** predicate on `MetaState` + RAM (e.g. `at_white_block ∧ big_or_small`).
2. **Policy:** deterministic frame path or closed-loop executor (search).
3. **Termination:** \(\beta=1\) on inventory change / map mode / death.
4. **Effect:** opaque until observed; then cache \((s \mapsto s')\).
5. **Cost:** measured frames (mean + max); no silent symbolic defaults in ROM benches.
6. **Tier:** lowest tier that makes the option findable; record injected facts.

When shaping reward for low-level search:

$$
R \leftarrow R + \big(\Phi(s') - \Phi(s)\big) - W_{\text{death}}\mathbf{1}_{\text{dead}}
$$

with \(\Phi\) monotone in *true* task progress (area/room/card/inventory), never
score/timer/music.

When claiming a planner-class result:

$$
\text{gap} = \frac{\text{cost}(\pi_{\text{greedy}})}{\text{cost}(\pi_{\text{search}})}
\quad\text{only if both libraries and cost models match.}
$$

---

## 7. Updated knowledge map (what we believe on 2026-07-14)

| Belief | Confidence | Needs |
|---|---|---|
| Exact sim ⇒ search > flat RL for known Mario levels | High | — |
| Option-SMDP is the right SMB3 abstraction | High | B2–C1 to complete |
| Planner-class whistle gap is real *under Tier-1 grant* | High | B4 to remove grant |
| Gap remains with real AcquireWhistle | Medium | B2/B3/B4 |
| Policy priors will transfer usefully to SMA4 | Medium | D2 |
| Flat zero-shot generalist from stock levels | Low / wrong goal | PCG moonshot only |
| Standalone IL ceiling is fundamental | Medium-low | Foster/Song reopen science track |

---

## 8. Immediate next action (single)

**Run A0 (SMA4 throughput bench) then B2 scaffolding: probe white-block /
background-layer RAM (`A2`), implement a Tier-2 `AcquireWhistle_1_3` option
skeleton with an explicit duck-hold termination, and attempt a replay-verified
acquisition from `runs/sma4_cache/1-3_entry.pkl`.**

Do not expand the symbolic whistle library further until B2 lands or fails with
a written min-knowledge-tier result.

---

## 9. Key references (added / re-centered this pass)

- Sutton, Precup, Singh — Options / SMDPs (AIJ 1999).
- Ng, Harada, Russell — Potential-based shaping (ICML 1999).
- Anthony et al. — Expert Iteration (NeurIPS 2017); Silver et al. — AlphaZero.
- Ecoffet et al. — Go-Explore (Nature 2021).
- Foster, Block, Misra — "Is BC All You Need?" (NeurIPS 2024) — **nuance on IL horizon**.
- Song, Rohatgi, Singh, Bagnell — "To Distill or Decide?" (NeurIPS 2025) — distill smoother experts under determinism.
- Dietterich — MAXQ; Vezhnevets et al. — FeUdal Networks (secondary for this repo).
- Bandres et al. — IW / width-based planning (cell novelty cousin).
- JaffarPlus / QuickerMGBA — high-throughput savestate search prior art (D4 spike).
- StrategyWiki / TASVideos / Southbird smb3 / Karisa sma4-disasm — whistle & RAM ground truth.
- Local: `V5_FINDINGS.md`, `V6_FINDINGS.md`, `notes/sessions/2026-06-29-option-mdp-whistle-benchmark.md`.

Full bibliography updates land in `notes/research-bibliography.md`.
