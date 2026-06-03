# Mario AI — Design Document

**Goal:** Build a local AI that plays, beats, and eventually speedruns *Super Mario Bros.* (NES), from scratch, with **no external gameplay data**, running **fully on a MacBook Pro 14" (2023, M2 Pro, 16 GB unified memory)**.

**Project intent (chosen):** *Learn the whole stack.* The objective is to implement and genuinely understand every layer — forward-model search, behavior cloning / distillation, DAgger, and value-guided search — end to end. Beating levels is the proof of correctness, not the only point. Breadth and understanding over peak performance.

This document is the map. Each module below doubles as a learning unit: it states *what* it is, *why* it exists, the *theory* behind it, the *interface*, and the *failure modes* to watch.

Date: 2026-06-03. Author/operator: single developer on one machine.

---

## 0. The core idea in one paragraph

Super Mario Bros. is a **deterministic** game: `state_{t+1} = emulator(state_t, action_t)`. Because we *own a perfect simulator*, we don't need to learn physics — we can **search** the simulator for good action sequences (this is the historically dominant approach for Mario). Search is slow, so we **distill** what it finds into a tiny neural network that plays instantly. The net inevitably wanders into states the search never showed it, so we **correct** it by re-running search from its own failure states and retraining (**DAgger**). Optionally, we make the search itself smarter with a **learned value function**. That loop — search → distill → correct → (guide) — is the whole project.

```
NES emulator (true forward model)
        │
   RAM/tile extractor  ──────────────► compact Markov-ish observation
        │
   chunked action space (SIMPLE_MOVEMENT + frame-skip)
        │
   beam search / A* teacher (death-aware reward)  ◄── [later] learned value net
        │
   self-generated trajectory buffer  (state, action-chunk, value)
        │
   tiny MLP/GRU student policy  (soft, chunked behavior cloning)
        │
   DAgger correction loop  (uncertainty-gated re-labeling)
        │
   [later] speed objective + value-guided search
```

---

## 1. Why this architecture (and not the alternatives)

Grounded in the literature so the choices are defensible, not cargo-culted.

- **Search > RL when you have a forward model.** The 2009–2012 Mario AI Competition was dominated by Robin Baumgarten's **A\*** (cleared every level; best hand-coded controller scored <50%). The Togelius/Karakovskiy survey's conclusion: with a simulator, planning beats learning for this task. We have a perfect simulator.
- **Don't learn a world model — we already have the real one.** MuZero / EfficientZero / DreamerV3 exist to *learn* dynamics from pixels. EfficientZero is sample-efficient but wall-clock-hungry (reference repros: A100 + ~20 CPU cores, 1–2 days *per game*) — hostile to a 16 GB Mac and pointless when the emulator is free. **Out of scope** except as optional reading.
- **Closest prior art:** Tom Murphy VII's `learnfun`/`playfun` (SIGBOVIK 2013): emulator-as-forward-model + searched action sequences + reusable "motifs" (= our action chunks). It beats 1-1 with superhuman timing but never distilled search into a reactive net — that gap is exactly what we add.
- **Pure pixel PPO is the beginner route.** It can learn 1-1 (public repos clear 31/32 levels, but **per-level overfit**, ~4.5 h/level on a *real GPU* — slower here). We keep PPO only as an optional **fine-tuning** step at the very end, never as the foundation.

---

## 2. Hardware reality & the real bottleneck

- **The bottleneck is the emulator search loop on CPU, not the GPU.** Tom7 spent "days of compute" in search. Our tiny nets (100k–2M params) train in *seconds* on the MPS backend.
- **Leverage = parallelize emulator rollouts across cores** and minimize per-node snapshot calls.
- **16 GB is plenty** for tiny nets; watch memory only if replay buffers of tile-grids get large.
- **MPS is beta-but-fine** at this scale. Guardrails: validate one CPU-vs-MPS training run for numerical parity (known silent MPS bugs historically), keep `torch.compile` and `autograd.detect_anomaly` **off**, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

---

## 3. Environment & toolchain

**Decision: `gym-super-mario-bros` 8.0.0 + `nes-py` 9.0.0.** As of the May 2026 revival this is the clear winner on Apple Silicon:

- Prebuilt **arm64 wheels** — `pip install gym-super-mario-bros` just works, no compiler.
- Migrated to **Gymnasium** (`gymnasium>=1.0.0`).
- Exposes **`dump_state()` / `load_state()`** — an *in-memory opaque snapshot* (cheap clone), exactly what beam search needs. (Also legacy single-slot `_backup()`/`_restore()`.)
- Direct RAM access for building observations and reward.
- Headless by default (`render_mode=None`).

**Hard constraint:** requires **Python 3.13+**. This conflicts with `stable-retro` (≤3.12), so we commit to the nes-py lane.

**Rejected alternatives:** `stable-retro` (heavier zlib state clones, Py≤3.12, must hand-author SMB integration); `FCEUX`+Lua (great TAS tool, but search must be written in Lua, no clean Python headless throughput — keep only for *verifying/replaying* finished runs); `BizHawk` (not native on Apple Silicon); `cynes`/`libretro.py`/`TetaNES` (viable fallbacks but more DIY).

**Toolchain:**
- Python **3.13** venv (`uv` or `python -m venv`).
- `gym-super-mario-bros`, `gymnasium`, `numpy`, `torch` (MPS), `tqdm`, `tensorboard` (or simple CSV logging), `pytest`.
- FCEUX (optional, GUI) only to watch/verify replays.

**First thing to do before writing any search:** microbenchmark on *this* machine —
1. headless `env.step` rate, and
2. `dump_state()`/`load_state()` round-trip cost.
Per-node clone cost, not raw fps, dominates beam throughput. Record the numbers in `bench/` so later tuning has a baseline.

---

## 4. Observation design

**Decision: RAM/tile features, not pixels.** State features are far more sample-efficient; we skip the CNN entirely and train a tiny MLP/GRU.

**Observation vector = ego-centric tile grid + scalar features.**

- **Tile grid:** a window around Mario, e.g. `H×W` (start ~13×16) of small integer codes: `0` empty, `1` solid, `2` Mario, `-1` enemy, plus a few for hazards/pipes/coins. Built by reading tile RAM and enemy/Mario positions ÷16. **Use *relative* offsets** (window centered on Mario) so it generalizes across levels — never feed absolute level-x as a feature.
- **Scalars:** horizontal velocity `0x0057`, vertical velocity `0x009F`, powerup state `0x0756`, float/airborne state `0x001D`, on-ground flag, maybe time-bucket.
- **Temporal context:** raw RAM is **not fully Markov** (enemy slots get reused, some state is in PPU/timers). Use a **frame stack (k=2–4)** or a **GRU** so the policy doesn't oscillate.

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

**Decision: `SIMPLE_MOVEMENT` (7 actions) + frame-skip 4.**

`SIMPLE_MOVEMENT = [NOOP, right, right+A, right+B, right+A+B, A, left]`.

- **Frame-skip = action repeat:** each chosen action is held for `k=4` frames, reward accumulated. This is the single biggest lever — keeps beam branching at 7, and a depth-`d` beam covers ~`d·4` frames ≈ `d/15` s.
- **Action chunks ("motifs"):** the search and the policy both operate on **chunks** — short fixed-length action sequences — not per-frame buttons. Align the policy's chunk length to the search granularity.
- **Caveat:** frame-skip misses **frame-perfect** inputs (precise jumps, wall-clips). Fine for "beat the level"; for later speed work, use **adaptive skip** (finer control near hazards). Add `left`-containing and longer-hold chunks so backtracking/routing is *possible* (greedy-right-only search provably fails on deceptive geometry like 4-2/8-4 warps).

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

**Reward / evaluation — THE most important thing to get right.** Tom7's two canonical failures define the rules:
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
    def label(self, snapshot) -> (best_chunk, soft_targets, value):
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
  best_chunk:   int,                                  # argmax teacher action (hard label)
  soft_targets: float[num_chunks],                    # distribution over teacher's surviving top chunks
  value:        float,                                # teacher's estimated value (for value-net training)
  level:        (world, stage),
  source:       "search" | "dagger_round_i",
  snapshot:     opaque bytes (optional, for re-labeling / debugging)
}
```

Store as sharded `.npz`/`.parquet` under `data/`. Cap or reservoir-sample the aggregated set so it doesn't grow unbounded across DAgger rounds.

---

## 8. The student policy (distillation)

**Model: tiny MLP or GRU, 100k–2M params.** No CNN (we use RAM/tile features). Inputs: stacked observation. Output: distribution over action chunks (+ optional value head).

**Loss: soft, chunked behavior cloning.**
- Don't clone only the single best action — clone the **distribution over the teacher's surviving top chunks** (AlphaZero visit-count-style soft targets). Soft targets transfer more info and regularize a tiny student.
- Predict short **action chunks** (ACT-style), aligned to frame-skip, to cut compounding error over the horizon.
- `loss = CE(student_logits, soft_targets) [+ λ·MSE(value_head, teacher_value)]`.

**Eval metrics:** completion rate per level, median x_pos reached, best time, **death-location histogram** (drives where DAgger spends budget).

**Failure modes:** non-Markov input → use frame-stack/GRU; cross-level distribution shift → train/eval on multiple levels with relative features; the student can only be as good as the teacher's labels on the *student's* distribution (→ DAgger).

---

## 9. DAgger correction loop

Plain BC fails because one bad jump lands Mario in a state never seen in training; errors compound (BC error grows ∝ T², DAgger ∝ T). DAgger trains on the **learner's own induced state distribution**.

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

**Why search-as-expert + DAgger is a perfect match:** DAgger needs a queryable expert *at arbitrary states*. A human can't do that; **our beam search can** (`label(snapshot)`).

**Uncertainty gating (spend search budget wisely):** don't re-label every frame. Query the teacher only where the policy is unsure — SafeDAgger (a learned "is the policy safe here?" classifier) or DADAgger (ensemble disagreement). Natural fit: re-label clustered around death-histogram hotspots.

**Pitfalls:** teacher must give *good* off-distribution labels (deepen search at hard states); aggregated set staleness (cap/reservoir); immature policy wandering into junk (cheap in a game — just death — but wastes budget).

**Variants to know (optional reading):** DAgger-by-coaching (label with achievable-better actions when the teacher is far stronger than the tiny net), HG-DAgger (gated intervention).

---

## 10. [Later] Value-guided search & speed objective

Only after fixed-depth search visibly stalls on specific obstacles:

- **Learned value net `V(s)`** to *prune/order* the beam (best-first beam search; ~30% speedups, more at larger beams). Lets a *shallow* beam behave like a deep one — directly attacks Tom7's short-horizon failures.
- **This closes the AlphaZero/ExIt loop in miniature:** search (guided by `V` + the policy as a prior) produces improved targets → distill policy *and* value → better `V` makes the next search better.
- **Bootstrapping fragility:** a bad early `V` prunes the good branch, never learns it's good. Mitigate with a **beam-width floor** (never prune below N), **root exploration noise** (Dirichlet-style), and dense shaping (Δlevel-x) not just sparse flag reward.
- **Speed objective (final phase):** crank `W_flag` + milestone bonuses, add `-W_time·frames` but **respect framerules** (per-frame time only matters on the last segment / 8-4). Optionally a short **PPO fine-tune** at the very end — never as the foundation.

**Pragmatic order:** hand-coded heuristic (progress − danger) for beam ordering first → add the policy net as a cheap **action-ordering prior** → add a learned **value net** last.

---

## 11. Repository layout

```
mario_ai/
  DESIGN.md                  # this file
  pyproject.toml             # Py3.13, deps
  bench/                     # step-rate & snapshot-cost microbenchmarks (run FIRST)
  mario/
    env.py                   # gym-super-mario-bros wrapper, frame-skip, headless
    observation.py           # RAM → tile grid + scalars (Section 4)
    actions.py               # SIMPLE_MOVEMENT chunks / motifs (Section 5)
    reward.py                # death-aware evaluate() (Section 6) — heavily tested
    search.py                # beam search / A* teacher + label() (Section 6)
    buffer.py                # trajectory dataset I/O (Section 7)
    policy.py                # tiny MLP/GRU student (Section 8)
    train.py                 # soft chunked BC (Section 8)
    dagger.py                # correction loop (Section 9)
    value.py                 # [later] learned value net (Section 10)
    eval.py                  # completion rate, death histograms, replay export
  scripts/
    v0_search_1_1.py         # milestone V0
    gen_dataset.py
    run_dagger.py
  data/                      # generated trajectories (gitignored)
  runs/                      # logs, checkpoints, death histograms
  tests/                     # pytest: reward correctness, snapshot determinism, obs shape
```

---

## 12. Milestones (each is a learning unit)

| Ver | Deliverable | What you learn / proves |
|---|---|---|
| **V0** | **Beat 1-1 with pure search** (no net) | Forward-model loop, snapshot save/restore, reward design. *Highest-information first step.* |
| **V1** | Self-generated dataset across 1-1…1-4 | Trajectory logging, curriculum, soft targets |
| **V2** | Tiny student policy clears 1-1 from BC alone | Distillation, observation/action design, MPS training |
| **V3** | DAgger lifts completion rate on its own failures | Distribution shift, expert querying, uncertainty gating |
| **V4** | Multi-world clears via curriculum + DAgger; 8-4 special-cased | Generalization, warp routing |
| **V5** | Value-guided search + speed objective; optional PPO fine-tune | Best-first search, ExIt loop, framerules |

**Definition of done for the educational goal:** every module in §11 implemented, understood, and exercised by a milestone — with `reward.py` and `search.py` covered by tests, and a written retro on where each theoretical failure mode (death-in-objective, fake-progress counters, teacher myopia, distribution shift, MPS parity) actually showed up in practice.

---

## 13. Open questions / decisions deferred

- Exact tile-grid dimensions and code vocabulary (tune empirically against deaths).
- Chunk length and chunk library (which motifs; how many `left`/long-hold chunks).
- Beam width vs. depth budget given measured snapshot cost (decide after `bench/`).
- Whether a GRU is worth it over a frame-stack MLP (try MLP first).
- When exactly to introduce the value net (only when fixed-depth search stalls).

---

## 14. Key references

- Baumgarten A\* / Mario AI Competition; Karakovskiy & Togelius survey — search dominates with a forward model.
- Tom Murphy VII, *learnfun/playfun* (SIGBOVIK 2013), tom7.org/mario — closest prior art; objective-design failure modes.
- Ross, Gordon, Bagnell, *DAgger* (AISTATS 2011) — train on learner's induced distribution.
- AlphaZero/MuZero & *Expert Iteration* (Anthony 2017) — search-as-teacher, visit-count soft targets.
- Zhao et al., *ACT* (2023) — action chunking cuts compounding error.
- Best-First Beam Search (arXiv 2007.03909); MCTS as regularized policy optimization (arXiv 2007.12509).
- Data Crystal — *Super Mario Bros. RAM map*.
- gym-super-mario-bros / nes-py (Kautenja), revived May 2026, arm64 wheels, `dump_state`/`load_state`.
```
