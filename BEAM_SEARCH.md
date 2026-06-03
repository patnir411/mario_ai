# Beam Search — Educational Walkthrough (as applied to V0)

How V0 beat Super Mario Bros 1-1 with **no neural network** — just search over the
emulator. This is the teaching companion to `mario/search.py`. Architecture context is in
`DESIGN.md`; live project status is in `CLAUDE.md`.

All numbers here are from the canonical V0 run `runs/20260603-130253-v0_1_1` on an M2 Pro.

---

## 0. The core idea in one sentence

The NES emulator is a **perfect simulator you can rewind**, so instead of *learning* to
play, V0 **searches** for a winning sequence of button presses — testing moves in the
emulator and undoing them — and keeps only the most promising attempts alive.

No training, no data, no network. Just: simulate futures, score them, keep the best, repeat.

---

## 1. Why search is even possible here

Three properties of NES Mario make search the right first tool (`DESIGN.md` §1):

1. **Deterministic.** `state_{t+1} = emulator(state_t, action_t)` exactly. A sequence of
   inputs replays identically every time. (Verified: `tests/test_determinism.py`.)
2. **Perfect, owned forward model.** We don't approximate physics — we run the real game.
3. **Cheap, exact rewind.** `dump_state()` / `load_state()` clone full emulator state in
   ~60 µs / ~11 µs and restore it bit-for-bit. (Verified: `tests/test_snapshot.py`.)

Without #1 and #3 the whole approach collapses — that's why those were the first tests
written, before any search code.

---

## 2. The algorithm

**Beam search** = breadth-limited best-first search. At every step you hold a fixed number
(`W`, the *beam width*) of the most promising partial solutions, expand each by every
possible action, score all the results, and keep the best `W` again.

Pseudocode (the real version is `mario/search.py::beam_search`):

```
beam = [ root_state ]                      # one node: start of the level
repeat until a node reaches the flag, or max depth:
    candidates = []
    for node in beam:                      # W nodes
        for action in 7 actions:           # B = branching factor
            restore(node.snapshot)         # rewind to this node  (~11 µs)
            info = run action for 8 frames  # frame-skip "chunk"   (~5.8 ms)
            if reached flag:  return success(node.path + [action])
            if died:          discard       # never keep a dead path
            score = reward(info)            # progress + alive, death = huge negative
            candidates.append(child(snapshot(), score, ...))   # ~62 µs to snapshot
    candidates = dedup(candidates)         # merge near-identical states
    beam = top_W(candidates by score)      # PRUNE — keep best W, drop the rest
```

The three load-bearing lines:
- **`restore(node.snapshot)`** — the rewind. Lets us try many futures from the same point.
- **`if died: discard`** — death is removed from the beam entirely (and the reward also
  penalizes it heavily, so it never even ranks).
- **`beam = top_W(...)`** — the **prune**. This single line is what makes search tractable
  (see §4) and also what makes it incomplete (see §6).

### Mapping to the code (`mario/search.py`)

| Concept | In the code |
|---|---|
| beam width `W` | `beam_width` (V0 used **48**) |
| branching `B` | `N_ACTIONS` = **7** (`SIMPLE_MOVEMENT`) |
| action chunk | `chunk_frames` = **8** (hold one action 8 frames) |
| node state | `Node(snap, score, info, path, x_max, stuck, frames)` |
| rewind | `sim.restore(node.snap)` |
| score | `mario.reward.state_score` (death-aware) |
| dedup | `_dedup_key` = `(x//16, y//16, status)`, keep best per bucket |
| anti-stall | `stuck > stuck_cap` (12) → abandon node |
| success | `is_success(info)` → returns immediately |

---

## 3. Worked walkthrough: V0 on 1-1

What actually happened in `runs/20260603-130253-v0_1_1`:

- **Start:** beam = one node, Mario at x≈40.
- **Each round** expands the 48 beam nodes × 7 actions = 336 candidates, each simulated 8
  frames forward, scored, deduped, pruned back to 48.
- **Progress was dead-linear:** ~24 px of rightward progress per round (≈ max run speed),
  with zero deaths surviving in the beam — because the death penalty (`-10000`) instantly
  sinks any path that dies, and "further right + alive" ranks highest.
- **Stopped at depth 134** when a beam node's next chunk landed Mario on the flagpole.

Final result (from `result.json`):

```
beat_level = true     deaths = 0          x reached = 3161 (flagpole)
frames = 1072         framerule_time = 52  (the SMB speed unit)
nodes searched = 42,269                    wall clock = 247 s (~4 min)
```

Visually confirmed: the `contact_sheet.png` last cell shows Mario on the flagpole with the
castle — see `CLAUDE.md` "what works".

---

## 4. Computational requirements

### The formula

```
nodes evaluated  =  W × B × D
```

- `W` = beam width = 48
- `B` = branching = 7
- `D` = depth (decisions to the goal) ≈ 134

→ 48 × 7 × 134 ≈ **45,000** (measured 42,269 — slightly less due to dedup + early stop).

This is **linear** in W, B, and D. That linearity is the whole game.

### Time — dominated by emulator stepping

Cost of one node (measured):

| step | cost |
|---|---|
| restore snapshot | ~11 µs |
| **run 8 frames** | **8 × 727 µs ≈ 5,800 µs** ← ~99% of the cost |
| save snapshot | ~62 µs |
| **per node** | **≈ 5.9 ms** |

```
time ≈ W × B × D × chunk_frames × t_frame
     ≈ 42,269 × 8 × 727 µs ≈ 246 s   (measured 247 s ✓)
```

The bottleneck is **CPU emulator stepping** (`t_frame` ≈ 727 µs; nes-py ≈ 1380 fps/core).
Snapshots, scoring, and pruning are negligible. Search is *step-bound*, not memory-bound.

### Memory — cheap and constant in depth

Beam search keeps only the current frontier, not the whole tree:

```
memory ≈ W × B × snapshot_size ≈ 48 × 7 × tens-of-KB ≈ a few MB
```

This is **O(W)** — independent of how deep the search goes. (Full breadth-first search
would need `B^D` memory; impossible.) Memory is never the constraint here.

---

## 5. Why it stays tractable: exponential → linear

A *complete* search of every input combination is:

```
B^D = 7^134 ≈ 10^113 paths
```

— more than atoms in the universe, with exponential memory to match. The prune-to-`W` step
converts that into:

```
W × B × D ≈ 45,000 nodes
```

**That conversion (exponential → linear) is the entire reason beam search works.** The
price: if the true solution looks unpromising early, the beam can prune it forever (§6).

---

## 6. Design choices in V0 — and the failure modes they fight

Vanilla beam search is naive; `mario/search.py` adds four guards, each targeting a known
failure mode (these lessons trace back to Tom7's *playfun* and the Mario AI literature,
`DESIGN.md` §1, §6):

1. **Action chunking (8 frames/decision).** Cuts the number of decision points ~8×, which
   slashes snapshot/scoring overhead *and* stops the beam from wasting its 48 slots on
   trivially-different one-frame variations. (Subtlety: total *frames simulated* ≈
   `W·B·level_frames` is roughly independent of chunk size; chunking buys search *quality
   per unit compute*, and lets a smaller beam succeed — it does not reduce raw frames.)

2. **Death-aware reward.** Death is a dominant `-10000` and dead nodes are discarded. Fights
   the canonical failure where an agent jumps into a pit because respawning still "scores
   okay." The reward also **ignores score/coins/timer** — counter-style RAM creates *fake
   monotone progress* that traps search on e.g. the 1-2 coin ledge.

3. **State dedup** (`_dedup_key`). Merges near-identical states so the beam holds genuinely
   different trajectories, not 48 clones. Fights **beam diversity collapse**.

4. **Stuck cap.** Abandons nodes that stop making forward progress. Fights local stalls
   where a node lingers without dying or advancing.

These aren't part of "beam search" — they're the difference between a textbook algorithm
and one that actually clears the level.

---

## 7. Tradeoffs

### The central tension: completeness vs. speed (the width dial)

```
W = 1   (greedy)      fast, tiny memory, gets stuck at the first hard spot
W = 48  (V0)          linear cost, usually finds it, can miss it
W = ∞   (full search) guaranteed, but exponential — impossible
```

Beam search is **incomplete and suboptimal by construction** — it can prune the only
winning path. The cheap O(W) memory exists *because* it discards candidates, and discarding
is exactly what risks losing the solution. You can't separate the two.

### Other core tradeoffs (general → Mario)

- **Myopia.** Beam follows the score; it can't see that a locally-bad move (go *left/down*)
  is needed. → Pure rightward search **fails on 8-4** (warp/maze routing) and dead ends.
- **Reward hacking.** Mis-specified score → it games the loophole. → Pit-jumps and
  wall-humping unless death/fake-progress are handled (we did).
- **Diversity collapse.** Survivors descend from a few early nodes and become near-clones;
  effective diversity ≪ W. → Mitigated by dedup + stuck cap.
- **Needs a cheap resettable simulator.** → We have the emulator; **robotics/real world do
  not**, so this approach doesn't transfer there without a *learned* model.
- **Open-loop & deterministic.** Output is a fixed input sequence with no feedback. → Works
  because Mario is deterministic; **any perturbation, new seed, or new level breaks it** —
  zero generalization, re-search every time. Stochastic environments break it entirely.
- **Granularity.** Coarse chunks miss frame-perfect tricks. → Fine for *beating* 1-1; can't
  produce a frame-perfect TAS (wall-jumps, subpixel) without finer control + bigger beam.
- **Tuning is brittle.** No principled way to pick W, chunk size, dedup buckets, stuck cap,
  or reward weights — all interact, all empirical.

### Beam search vs. alternatives

| Method | Optimal? | Memory | Needs | Best when |
|---|---|---|---|---|
| Greedy (W=1) | No | Tiny | Score | Easy, monotone problems |
| **Beam (W=k)** | No | O(W) | Score + sim | Deterministic, decent score, simple+robust |
| A* | Yes (admissible h) | Can blow up | Strong heuristic | You have a good distance estimate |
| BFS / DFS | BFS yes / DFS no | BFS exponential | — | Small problems |
| MCTS | Asymptotically | O(tree) | Sim + rollouts | Huge branching, stochastic, anytime |
| Learned policy | — | O(weights) | Training data | Real-time, generalization, noisy worlds |

- **vs A\*:** with a good heuristic A\* explores *fewer* nodes (prioritizes the single best
  node, not a fixed frontier), but needs that heuristic and can blow up memory. Beam is more
  forgiving of a mediocre score — why we started here.
- **vs MCTS:** MCTS *explores* (tries things that look bad to learn if they're good). Beam
  is **pure exploitation** of the current score — which is exactly why it falls into local
  optima.
- **vs a learned net:** the net generalizes and plays in one instant forward pass but needs
  data and can be wrong off-distribution. Beam is exact-but-brittle; the net is
  approximate-but-flexible.

---

## 8. When beam search is the right tool

**Right when:** deterministic, cheap resettable simulator, a usable scoring function exists,
you want *one good solution* (not a general player), and the space is too big for full
search but structured enough that a frontier can follow it. → **Mario per-level is a near-
perfect fit.**

**Wrong when:** stochastic dynamics, no/expensive simulator, you need real-time reaction,
you need generalization, or success requires long non-greedy detours the score can't see. →
**8-4 routing, real-time play, and "play any level" all fall here.**

---

## 9. How V0 feeds the rest of the project

Beam search's weaknesses map almost one-to-one onto what the later milestones fix, and its
strengths make it the ideal **teacher**:

```
search (V0)            slow, exact, no generalization, can't play live
   │  generates perfect (state → best action) examples
   ▼
distill into a net     instant (one forward pass), generalizes, but can be wrong
   │  the net plays and makes mistakes
   ▼
DAgger                 re-run search from the net's failure states, retrain
```

So the project is **not** "search *or* net." Search is the teacher that produces flawless
data; the net is the student that turns a 4-minute search into microsecond reactions and
generalizes to states no single search ever visited (`DESIGN.md` §8–9).

---

## 10. Where to look in the code

- `mario/search.py` — `beam_search`, `Node`, `_dedup_key` (this doc's subject)
- `mario/reward.py` — `state_score`, death/fake-progress handling (§6.2)
- `mario/env.py` — `MarioSim.snapshot/restore/run_chunk` (the rewind primitive)
- `scripts/v0_search_1_1.py` — runs V0 end-to-end and writes the verifiable artifacts
- `tests/test_determinism.py`, `tests/test_snapshot.py` — the invariants search depends on
```
