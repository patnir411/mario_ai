# Mario AI — Consolidated Findings & Status (V4 + strategy)

_Last consolidated: 2026-06-07. Ground-truth status block lives in `CLAUDE.md`; this is the
narrative deep-dive across the search phase, the V4 distillation experiments, the novelty/
literature assessment, and the cross-game feasibility analysis._

---

## 0. Where we are (one screen)
- **Search side: DONE.** any% full game beaten (8/8, `beat_game=True`); every hard level solved
  from scratch (2-2, 7-2 water side-pipes; 4-4, 7-4, 8-4 castles incl. 7-4's 6-gate multi-loop).
  16 cached solutions; a 15-level showcase video. All artifact-verified (`render.replay` beat=True
  + contact sheets). 43 tests green. Private repo: `github.com/patnir411/mario_ai`.
- **Learning side (V4): the search teacher is excellent; distilling it into a *policy* is the wall.**
  A single learned net does NOT yet beat levels. We diagnosed exactly why (below) — and it is a
  closed-loop / multi-task problem, not a representation problem.
- **Strategy work:** an honest novelty/literature assessment and a cross-game ("any Mario game")
  feasibility + cost/time analysis (below).

---

## 1. Search phase (the part that fully works)
Forward-model search over the deterministic, snapshot-able NES emulator:
- `beam_search` + `coverage_search` (Go-Explore: cell archive `(area,x,y)` + novelty, area-first),
  global progress coordinate Φ (potential-based), value-guided best-first (ExIt §10).
- **The recurring unlock was an instrumentation fix, not difficulty:** gym-super-mario-bros
  fast-forwards pipe/area transitions *inside* `env.step`, so post-step RAM is blind to entries.
  Correct signal = per-frame `info.x_pos`-vs-live-RAM jump (or raw `sim.u._env.frame_advance`).
- **The hard castles fell via disassembly-grounded route decoding** (SMBDIS.ASM): `HandlePipeEntry`
  metatile predicate ($11/$10), `ProcLoopCommand` Y-position loop gates, the 7-4 `MultiLoop`
  counters. This LLM-assisted-reverse-engineering → planner-shaping loop is the most distinctive
  part of the project.

Solved (cf8 = chunk-8, cf1 = frame-precise): 1-1,1-2,1-3,1-4,2-1,4-1,4-2,4-4,8-1,8-2,8-3 (cf8);
2-2,7-2,7-4,8-4 (cf1). Verified, cached, stitched.

---

## 2. V4 — distilling search into a learned policy (the experiments + verdict)

**Goal:** one small net plays many levels (search-teacher → BC → DAgger).

**2a. Fixed the stale pipeline first.** The old dataset was 837-dim / 3 levels; current obs is
1048-dim / 9-action. Regenerated 11 cf8 levels (8,362 states, on-path + perturb-and-recover) via a
new `--from-solution` backbone mode in `gen_dataset.py` (beam can't re-solve 4-2/4-4, so their
cached solutions seed the backbone). The 4 cf1 levels are out of a cf8 net's reach by construction.

**2b. The flat-MLP generalist: 0/11.** BC = 0/11 (dies at first hazard); +1 DAgger round = still
0/11; val action-accuracy stuck ~0.57. The repo's own `train.py` already warned: *"one MLP can't
hold many levels."*

**2c. Codex deep-dive (fresh context, YOLO, adversarial) + my audit.** It ran a CNN probe and
**refuted the easy fix**: a CNN over the tile grid only reaches ~0.60 val-acc — better than the
MLP but nowhere near reliable closed-loop control. It found the real culprits (I verified each):
- The **HAZARD obs channel is unimplemented** (`_hazard_cells` returns `()`), so the net is blind
  to firebars/Podoboos.
- **Severe action imbalance** (actions 0–3 = 97% of labels; `up`=0, `down`=31).
- **Weak 1-ply teacher** soft labels (near-uniform → the soft-KL term is decorative).
- **DAgger too local** (few anchors; corrections were ~4% of data).
- Shared value head likely perturbs the policy.
Literature is sobering: multi-task/platformer policies that work use CNN(+LSTM) on frame stacks
with *lots* of data + scalable online training (DQN/DRQN/IMPALA/Policy-Distillation), and Procgen/
CoinRun show platformer *generalization* overfits badly even so.

**2d. The structured entity-transformer experiment (the user's idea — `scripts/exp_entity.py`).**
Object-centric schema: tokens = `{player}` + 5 enemy slots `{type, dx, dy}` + 8 terrain columns
`{ground-height, pit}`, fed to a tiny **152K-param** transformer (single-token action head).
Distilled from the same 11 solutions + perturb-recover.

| Model | params | val action-acc | plays? | completion |
|---|---|---|---|---|
| flat MLP (tile grid), generalist | 2.28M | 0.57 | stalls/dies immediately | **0/11** |
| **entity-transformer, generalist** | **0.15M** | **0.65** | **yes — runs + jumps** | **0/11** |

Key results, all measured:
- **Representation validated:** the entity-transformer is **13× smaller**, **higher val-acc
  (0.65 > 0.57)**, and unlike the MLP it *actually plays* (reaches ~35–77% of levels from BC).
  (A class-weighting bug initially made it spam DOWN/LEFT; removing aggressive inverse-freq
  weighting was the fix — for a "run-right+jump" policy, up-weighting rare situational actions
  collapses it.)
- **But completion stayed 0**, and DAgger exposed the real wall:
  - **Generalist DAgger *oscillates* (multi-task interference):** 1-1 went **35% → 77% → back to
    35%** across rounds — gains on one level cost another. A single tiny net cannot hold 11 levels.
  - **Even a focused 1-1 *specialist* (no interference) plateaus** at the first-pit region (~28–44%)
    under generic anchor-recovery DAgger — it never reliably learns the frame-precise run-jump over
    the pit when jumps are ~2% of labels at cf=8.

**2e. Honest V4 verdict.**
- The **representation is not the bottleneck** — structured/attention helps (and is tiny/fast).
- The bottleneck is **robust closed-loop control from thin offline data**: BC covariate-shift +
  precise-timing actions + multi-task interference. This needs *tuned* closed-loop training, not a
  better encoder.
- **It is achievable per-level:** the project's *production* pipeline (MLP + engineered
  `gap_ahead`/`x_subtile` timing sensors + 4× jump-correction up-weighting + multi-round DAgger)
  historically cleared **1-1 and 1-2 at 100% as specialists**. My quick experiment deliberately did
  not re-port those tricks, which is why it plateaued.
- **The proven learned-agent deliverable = per-level specialists + search-rescue** (the
  `GameRunner` hybrid), not a single generalist.

---

## 3. Novelty & "is it from-scratch?" (researched, honest)
**Methods are not novel — every block is established prior art, and we built on it explicitly:**
- Forward-model search for Mario = canonical since **Baumgarten's A* (2009 Mario AI Competition)**.
- `coverage_search`+distill ≈ **Go-Explore** verbatim (Ecoffet et al., *First return, then explore*,
  Nature 2021: "exploit determinism, then robustify via imitation").
- Cell-novelty = **width-based planning / Rollout-IW** (Bandres/Bonet/Geffner 2018).
- Φ = **potential-based shaping** (Ng/Harada/Russell 1999); teacher→value→distill = **ExIt**
  (Anthony 2017); warm-start = **DAgger** (Ross 2011). TAS has beaten SMB via savestate search for
  ~20 years.

**From-scratch?** Yes in the ML sense (no pretrained weights, no human demos, no imported input
routes) — *but not tabula-rasa*: the hard castles were cracked with disassembly-derived domain
knowledge, and **7-4 was effectively routed by an LLM reading the 6502 disassembly**, not by
autonomous search.

**Genuinely distinctive (modest):** (1) the **LLM-assisted-reverse-engineering → planner-shaping**
loop for route-data-gated levels (topical: 2024–25 LLM+Ghidra / "Extracting Heuristics from LLMs
for Reward Shaping"); (2) **completeness on the *real* NES** (the academic competition used the
Infinite-Mario clone) with no input imports. Verdict: a strong engineering/systems + learning
project; incremental as novel ML research.

---

## 4. Cross-game general model ("any Mario game") — feasibility, cost, time
**The crux:** our RAM-tile obs is SMB1-specific and does **not** transfer (different RAM/physics/
controllers per game). Two general interfaces:
- **Pixels + SNES-superset actions** — the *only* zero-reverse-engineering universal input (DQN/
  IMPALA/Gato/VPT all use pixels). Needs a deep CNN(+memory) + scale.
- **Structured/symbolic state (the user's text/entity idea)** — *more* sample-efficient and
  generalizes better at the schema level, **but still requires per-game RAM reverse-engineering**
  (eased by existing community RAM maps for SMW/SMB3). Trade per-game RE for huge compute savings.

**Enabler:** `stable-retro` already integrates 1000+ ROMs incl. **Super Mario World**, with reward
variables + **savestates** — so our search infra transfers; we don't build emulators.

**On a "small text model at game speed":** yes — a tiny entity/typed-token transformer with a
**single-token action head** runs in single-digit ms (our cf=8 budget is 133 ms; even 60 fps is
16 ms), and is *faster* than a pixel CNN. Pure character-text encoding is worse than typed-token
embeddings unless you want a pretrained LM's priors. **Trainable on this M2 MacBook** — the
structured route is squarely in laptop range (we trained the 152K transformer in minutes); only
pixel-RL-at-scale or a VPT-style foundation model need cloud.

| Tier | Deliverable | Compute / $ (approx) | Time | Confidence |
|---|---|---|---|---|
| 1 | per-game agents, shared recipe | ~$100–500/game | ~1.5–3 mo (3–4 games) | ~85% decent play |
| 2 | ONE net, many games (moderate) | ~$1k–6k | ~3–5 mo | ~50–60% plays all; ~20–30% beats all |
| 3 | VPT-style Mario foundation model (few-shots new games) | ~$15k–60k (full VPT ≈ $150k+: 0.5B model, ~720×V100×9d) | ~6–12 mo, small team | ~50% genuinely impressive |

**Honest bottom line:** a single *pixel* model that plays multiple Mario games at a moderate level
is feasible (Tier 2). One net that fully *beats* SMB1+SMB3+SMW+SMB2 is research-frontier. The
binding constraint is the same one V4 exposed: **robust closed-loop control from limited data**, and
cross-game **amplifies multi-task interference** rather than removing it.

---

## 5. Recommended next steps (pick by goal)
1. **Bank the proven learned agent:** port the production DAgger tricks (timing-sensor weighting +
   jump up-weighting) into per-level specialists (incl. the entity-transformer) → clear 1-1/1-2/…
   as specialists + the `GameRunner` hybrid. High confidence; delivers a real learned agent.
2. **Lean into the distinctive angle:** formalize the LLM-RE → planner-shaping loop, or crack a
   castle without hand-fed disassembly (stronger from-scratch claim).
3. **Cross-game (frugal):** Phase 0 = stand up `stable-retro` (SMB1 + SMW), prove the search
   teacher + a structured/pixel policy transfer to a *second* game (a few hundred $, ~1 mo) — the
   cheap decisive test before any Tier-2/3 spend.

## Reproduce
`scripts/exp_entity.py {sanity,data,train,eval,dagger}` (entity-transformer experiment; env
`EXP_LEVELS`/`EXP_TAG` for focused runs); `scripts/gen_dataset.py W S [--from-solution]`;
`mario/train.py` (LEVEL_FILTER unset = generalist); `mario/eval.py`. Datasets/checkpoints are
gitignored under `data/`.
