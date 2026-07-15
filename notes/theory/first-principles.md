# First principles: the math under Mario AI / Option-SMDP

> Read after (or with) the free textbooks listed in `notes/theory/README.md`.
> This note is a **synthesis from those sources**, mapped onto this repo — not a
> substitute for the books. Last research pass: 2026-07-15.

**One sentence.** With a deterministic, resettable emulator, the justified architecture
is exact search (Bellman lookahead / approximate DP) over savestates, optionally guided
by learned \(\tilde J\) / \(\pi\) priors; temporally extended *options* induce an SMDP
on which the same story repeats one timescale up (whistle foresight).

**Root text.** Richard Bellman, *Dynamic Programming* (Princeton, 1957) — read
front-to-back; digest in `notes/theory/bellman-1957.md`. Everything below is later
packaging of that spine.

---

## 0. Layer map (absolute basics → our code)

| Layer | Math object | Fundamental reference | Repo object |
|---|---|---|---|
| 0 | Deterministic dynamics \(s'=f(s,a)\) | Bertsekas RL course §1.2; Bellman Ch. III §4 | nes-py / Stable-Retro; snapshots |
| 0.5 | Principle of Optimality + functional eq. | **Bellman 1957 Ch. III** | every search recurrence |
| 1 | Finite MDP / Bellman eq. | Sutton Ch. 3–4; Puterman Ch. 1–4 | adapter state, actions, terminals |
| 2 | Planning with known model | Sutton Ch. 8 | `beam_search`, `coverage_search` |
| 3 | Potential \(\Phi\) | Ng–Harada–Russell ICML'99 | global progress \(\Phi\) |
| 4 | Options → SMDP | Sutton–Precup–Singh AIJ'99; Sutton §17.2 | `mario/options.py`, meta search |
| 5 | Approx. value space / Newton | Bertsekas *Lessons from AlphaZero* | policy/value guided beam |
| 6 | ExIt / AlphaZero loop | Anthony et al.; Silver et al. | `policy_guided_search`, ExIt scripts |
| 7 | IL bounds | Ross–Gordon–Bagnell DAgger | V3–V6 BC/DAgger ceiling |
| 8 | Archive exploration | Go-Explore (Ecoffet et al.) | coverage cells / Go-Explore |

World-model RL (MuZero / Dreamer) sits **outside** this stack: those methods exist to
approximate a missing or expensive \(f\). We already own \(f\).

---

## 1. Layer 0 — Deterministic dynamics

\[
s_{t+1}=f(s_t,a_t),\qquad r_t=g(s_t,a_t)\ \text{(or }g(s_t,a_t,s_{t+1})\text{)}.
\]

**Bertsekas (deterministic finite horizon).** Exact DP: compute cost-to-go \(J_k^*\)
backwards; construct an optimal **open-loop** sequence forwards once \(J^*\) is known.
Stochastic MDPs need closed-loop \(\mu(x)\); determinism collapses that distinction for
a *fixed start state*.

**Consequence for us.** Cached action sequences (`data/solutions/*.json`) are valid
replays from the matching entry snapshot. Re-entry desync is a **state mismatch**, not
stochasticity. Snapshots make \(f\) resettable: restore = free teleport for search /
Go-Explore return.

**Hardware corollary.** Bottleneck is CPU emulator + snapshot, not GPU. Measure with
`bench/*` and `scripts/bench_sma4.py`; parallelize **across** processes, never inside
one beam on Stable-Retro’s single-instance emulator.

---

## 1.5 Layer 0.5 — Bellman 1957 (the root)

Bellman defines a **multi-stage decision process**: state \(p\), decision \(q\) selecting
a transformation \(T_q\), policy = \(q\) as a function of state, criterion on the
resulting trajectory. He rejects enumerative maximization over the full decision
sequence (**curse of dimensionality**, Ch. I) in favor of **imbedding** — solve a
*family* of problems and recover policy structure.

**Principle of Optimality (Ch. III §3):**

> An optimal policy has the property that whatever the initial state and initial
> decision are, the remaining decisions must constitute an optimal policy with
> regard to the state resulting from the first decision.

Discrete deterministic transliteration (with stage return \(g\)):

\[
f_N(p)=\max_q\Bigl[g(p,q)+f_{N-1}\bigl(T_q(p)\bigr)\Bigr].
\]

**Approximation in policy space** (Ch. I / III): guess a policy \(q_0\), evaluate its
return, improve — preferred to raw value iteration in applications; monotone when
survival/discount factors are nonnegative. This is the ancestor of policy iteration,
Bertsekas rollout/Newton, ExIt, and our policy-guided beam.

**Dictionary.** \(p\) = savestate; \(q\) = button or option; \(T_q\) = emulator/option
step; \(f_N\) = best \(N\)-stage return; beam = truncated DP with a pruned frontier;
options = the same skeleton on a coarser \(p\).

Full chapter digest (I–XI, inventory through Markovian decision processes):
`notes/theory/bellman-1957.md`. Local PDF:
`notes/theory/pdfs/dynamic programming.pdf`.

---

## 2. Layer 1 — MDPs and Bellman

### 2.1 Definitions (Sutton Ch. 3; Puterman Ch. 2)

- **State** \(s\): Markov sufficient statistic for \(p(s',r\mid s,a)\).
- **Action** \(a\in A(s)\).
- **Transition** \(p(s',r\mid s,a)\). Deterministic emulator ⇒ \(p=\delta_{f(s,a)}\).
- **Return** \(G_t=\sum_{k\ge0}\gamma^k R_{t+k+1}\) (episodic: finite \(T\), absorbing terminal).
- **Policy** \(\pi(a\mid s)\) or deterministic \(\pi(s)\). Puterman: also nonstationary /
  history-dependent *decision rules*; finite-horizon optima exist in Markov deterministic
  class (Puterman Thm. 4.2).
- **Values**
  \[
  v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\quad
  q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
  \]

### 2.2 Bellman equations (Sutton Ch. 3–4)

Policy evaluation:
\[
v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\,[r+\gamma v_\pi(s')].
\]

Optimality (\(v_*\), \(q_*\)):
\[
v_*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\,[r+\gamma v_*(s')].
\]

**Policy improvement theorem** (Sutton Ch. 4): if \(q_\pi(s,\pi'(s))\ge v_\pi(s)\) for all
\(s\), then \(v_{\pi'}\ge v_\pi\).

**Bertsekas operators (cost form).** \(TJ=\min_\mu T_\mu J\); \(J^*=TJ^*\);
discounted \(T\) is a contraction ⇒ unique fixed point (Banach / Shapley–Denardo).

### 2.3 What “having the emulator” does *not* buy

Sutton §3.7: knowing \(p\) does not make the problem computationally easy when \(|S|\) is
enormous. Exact value iteration over all of RAM is impossible. We need **focused**
planning on relevant states (Sutton Ch. 8 RTDP / on-policy trajectory sampling) —
exactly beam from level entry, not tabular VI.

---

## 3. Layer 2 — Planning with a known model

**Sutton Ch. 8.** A *model* predicts next state/reward. *Planning* improves a policy
using the model; *learning* uses real experience. Same backup machinery. Dyna mixes both.

With a perfect deterministic sample model (emulator):

- One visit to \((s,a)\) reveals \(s'\) exactly.
- Sample backup = expected backup (branching factor \(b=1\) on next-state randomness;
  branching remains over \(|A|\)).
- Tree / beam / coverage search **are** approximate DP on relevant states.

**Puterman Ch. 1.** Model-based MDP vs model-free RL are extremes of one continuum.
If a simulator exists, offline planning in sim is the natural primary method.

**Misconception to kill.** “RL = trial-and-error neural nets.” Sutton: RL is the
*problem* of sequential decision-making; DP with a perfect model is inside that problem.

---

## 4. Layer 3 — Potential-based shaping (why \(\Phi\) matters)

**Ng, Harada, Russell (ICML 1999).** Transform rewards by
\[
F(s,a,s')=\gamma\,\Phi(s')-\Phi(s)
\]
for any potential \(\Phi:S\to\mathbb{R}\). Then \(M\) and \(M'\) share the same optimal
policies (**policy invariance**). If \(F\) is *not* of this form, there exist MDPs where
optimal policies change.

Telescoping return:
\[
\sum_{t}\gamma^t F(s_t,a_t,s_{t+1})=\gamma^T\Phi(s_T)-\Phi(s_0).
\]

**Our SMB1 progress**
\[
\Phi=\texttt{area\_seq}\cdot 10000+(x-x_{\mathrm{entry}})
\]
is a potential on the area-augmented state. Used primarily as a **search ranking /
progress coordinate** (heuristic for beam), which is the right spirit: communicate
*what* progress is, without inventing farmable fake rewards (score, coins, wall-humping).

**SMA4 caveat.** Raw `x_pos` that *caps* near the roulette card is a broken progress
coordinate — hence `goal_suffix_search`. Fix \(\Phi\) / terminal detection; do not paper
over with non-potential bonuses.

---

## 5. Layer 4 — Options and SMDPs (the SMB3 object)

### 5.1 Definition (Sutton–Precup–Singh AIJ 1999 §2)

An option \(o=\langle\mathcal{I},\pi,\beta\rangle\):

- \(\mathcal{I}\subseteq S\) — initiation set
- \(\pi(a\mid s)\) — intra-option policy (Markov or semi-Markov)
- \(\beta(s)\in[0,1]\) — termination probability

Primitive actions are one-step options (\(\beta\equiv 1\)).

### 5.2 Theorem 1 — MDP + options = SMDP

Selecting among options and executing each **to termination** yields a semi-Markov
decision process. Multi-time models:
\[
r_s^o=\mathbb{E}\!\left[\sum_{i=0}^{k-1}\gamma^i r_{t+1+i}\mid E(o,s,t)\right],
\quad
p_{ss'}^o=\sum_{k\ge1}p(s',k)\,\gamma^k.
\]

Planning over options uses SMDP Bellman equations for \(V^\mu\), \(Q^\mu\), \(V_O^*\).

**Value achievement.** With correct models, DP finds a policy achieving \(V_O^*\), which
may be \(<V^*\) if \(O\) is incomplete. Planned values are *achieved* if models are true
(unlike many crude abstractions).

### 5.3 Call-and-return vs interruption

- **Call-and-return:** run \(o\) until \(\beta\) (our `search_options` default).
- **Interruption (Thm 2):** may abort when \(Q^\mu(h,o)<V^\mu(s)\); improves value.
  Not implemented at our meta layer yet.

### 5.4 Repo gap analysis

| Theory | `mario/options.py` today |
|---|---|
| \(\mathcal{I},\pi,\beta\) | `precondition` + `runner` + success; \(\beta\) implicit |
| Multi-time \(p_o,r_o\) | Symbolic / measured frame costs |
| Opaque effects | `opaque_effect=True` + execute-to-observe (correct for whistles) |
| Intra-option TD | Unused — appropriate while search fills \(\pi\) |
| Library size | Keep small (Hauskrecht utility problem: more options can *slow* planning) |

**AcquireWhistle** is exactly “build a verified \(\langle\mathcal{I},\pi,\beta\rangle\)”
(white-block route as \(\pi\), terminate on whistle inventory) — not another hand-grant.

---

## 6. Layer 5 — Approximation in value space (Bertsekas)

**Exact DP** sweeps all states. **Approximation in value space:** replace \(J^*\) by
\(\tilde J\); at the current \(x\), solve a short lookahead
\[
\tilde\mu(x)\in\arg\min_u\Bigl[g(x,u)+\tilde J\bigl(f(x,u)\bigr)\Bigr]
\]
(or \(\ell\)-step + terminal \(\tilde J\)).

**Rollout.** \(\tilde J\) from simulating a base heuristic; yields **cost improvement**
vs that heuristic (under stated conditions). Truncated rollout + terminal approx. for
long horizons.

**Newton view (*Lessons from AlphaZero* §3.2).** One-step lookahead / rollout satisfies
\(T_{\tilde\mu}\tilde J=T\tilde J\); solving \(J=T_{\tilde\mu}J\) is a **Newton step**
on Bellman’s equation at \(\tilde J\). Policy iteration ≈ repeated Newton. Local
superlinear improvement when \(\tilde J\approx J^*\).

**Design philosophy (Bertsekas §1.4 / §3.6).** The major determinant of quality is the
**on-line Newton step** (search / lookahead). Off-line training supplies a good starting
\(\tilde J\) or base \(\mu\); a policy network *alone* lacks that exact improvement step.

**Mapping.** `beam_search(policy_prior=, value_guide=)` ≈ limited branching lookahead
with off-line \(\tilde J\) / \(\log\pi\). V6: prior cut 1-1 nodes \(2.53\times\); value
alone saved little where \(\Phi\) already solved — consistent with “Newton helps when
\(\tilde J\) is the scarce ingredient.”

---

## 7. Layer 6 — ExIt and AlphaZero

### 7.1 Expert Iteration (Anthony, Tian, Barber)

Iterate: sample states from apprentice → expert (tree search) labels → train apprentice
→ rebuild expert with new prior/value. **Online ExIt without expert improvement = DAgger.**
No end-to-end regret theorem for ExIt itself; the paper’s claim is architectural +
empirical (Hex). Prefer tree-policy targets \(n(s,a)/n(s)\).

### 7.2 AlphaZero (Silver et al.)

Requires **perfect rules** inside MCTS (successor, terminal, score). Self-play trains
\((p,v)=f_\theta(s)\); play-time strength still comes from **MCTS guided by** \(f_\theta\),
not from \(p\) alone. Loss:
\[
\ell=(z-v)^2-\pi^\top\log p+c\|\theta\|^2.
\]

**For us.** Emulator \(f\) = AlphaZero’s “rules.” Self-play / BC are off-line training.
Do **not** learn a world model (MuZero) when \(f\) is free and exact — that injects model
error into every backup (Bertsekas certainty-equivalence warning).

---

## 8. Layer 7 — Imitation learning bounds (why V6 hurt)

**DAgger (Ross–Gordon–Bagnell).** Supervised BC under expert occupancy: task regret
\(O(T^2\varepsilon)\). Under learner occupancy with cost-to-go gap \(u\): \(O(uT\varepsilon)\).
DAgger reduces imitation to no-regret online learning so that some iterate has small
surrogate loss under **its own** state distribution.

**When it fails (theory + our artifacts).**

- \(\Pi\) too weak ⇒ \(\varepsilon_N\not\to 0\) (single-frame nets at death cliffs).
- Absorbing deaths ⇒ \(u=\Theta(T)\) ⇒ bound collapses toward \(T^2\varepsilon\).
- Surrogate (action mimicry) ≠ sparse success.

V6’s ~0.45–0.5 1-1 ceiling and DAgger degradation are **predicted** by large \(u\) +
non-realizable \(\Pi\), not by a missing DAgger trick. Same weak net can still help
**inside** search (ExIt/AlphaZero prior). Foster/Song (bibliography) refine when BC /
distillation can work — they do **not** rehabilitate flat generalist control as the
main line given an exact sim.

---

## 9. Layer 8 — Go-Explore (when beam detaches)

**Ecoffet et al.** Standard exploration fails via *detachment* (forgetting how to reach
a cell) and *derailment* (noise prevents reliable return). Fix: **first return, then
explore** — archive cells, restore (or goal-condition) to a cell, then explore from it.

**When Φ-beam suffices.** Smooth progress, local expansion finds the goal (open ground).

**When coverage / Go-Explore is necessary.** Non-monotonic or deceptive progress: warp
rooms, height mazes, pipe chains, white-block side routes, inventory bits. Our
`coverage_search` cell archive + snapshot restore is the deterministic-emulator form of
this principle. Whistle acquisition on the overworld graph is the same problem one
timescale up: cells = `(world, node, inventory, …)`.

---

## 10. Unified picture for this repo

```text
                    ┌─────────────────────────────┐
   off-line         │  optional net: π̃, J̃       │  ExIt / AlphaZero / Bertsekas
   training         │  (prior / value / BC)       │  "starting point for Newton"
                    └──────────────┬──────────────┘
                                   │ guides
                                   ▼
   on-line          ┌─────────────────────────────┐
   play             │  search over f (emulator)   │  Bellman lookahead / beam /
                    │  savestate tree / coverage  │  coverage / option search
                    └──────────────┬──────────────┘
                                   │ produces
                                   ▼
                    ┌─────────────────────────────┐
   verified         │  options ⟨I, π, β⟩          │  ClearLevel, AcquireWhistle,
   macros           │  + multi-time costs         │  UseWhistle, …
                    └──────────────┬──────────────┘
                                   │ actions of
                                   ▼
                    ┌─────────────────────────────┐
   meta SMDP        │  MetaState planning         │  greedy misses opaque skip;
                    │  search_options / UC        │  resettable search finds it
                    └─────────────────────────────┘
```

**Thesis stress-test (still holds after this reading).**

1. Deterministic resettable \(f\) ⇒ planning primary (Sutton 8, Puterman 1, Bertsekas).
2. Net serves search, not the reverse (ExIt, AlphaZero, Bertsekas Newton).
3. Options are the right abstraction for SMB3 items/levels (SPS 1999).
4. Opaque consumables require execute-and-observe; myopic greedy can miss skips
   (our whistle benches are the empirical corollary).
5. Standalone IL is a secondary / specialist tool under death cliffs (DAgger bounds + V6).

---

## 11. What the textbooks say NOT to do next

1. Do not treat world-model learning as the main line while \(f\) is exact.
2. Do not replace beam/coverage with a flat generalist policy.
3. Do not expect another DAgger round to unlock standalone 1-1 under current \(\Pi\).
4. Do not grow an option library indiscriminately (utility problem).
5. Do not use non-potential shaping (score farming, fake clear bits) as objectives.
6. Do not call hand-granted inventory an “option” — build verified AcquireWhistle.
7. Do not skip Go-Explore-style archives on deceptive / inventory side-routes.

---

## 12. Textbook / PDF inventory (committed pointers only)

Full URLs and fetch script: `notes/theory/README.md`, `notes/theory/fetch_corpus.py`.
Local PDFs live in `notes/theory/pdfs/` (gitignored). Extracted text in
`notes/theory/extracted/` (gitignored).

**Read end-to-end in the 2026-07-15 pass (via `pdftotext` + chapter study):**

- **Bellman, *Dynamic Programming* (1957) — entire book** (`dynamic programming.pdf`;
  digest `bellman-1957.md`)
- Sutton & Barto Ch. 1–4, 8, 16–17 (options + AlphaGo case material)
- Puterman & Chan draft Ch. 1–4 + Part I preface
- Bertsekas *Lessons from AlphaZero* (full extract chunks)
- Bertsekas RL course early chapters + abstract DP lecture
- Bertsekas *Rollout* early chapters
- Papers: options, shaping, DAgger, ExIt, AlphaZero, Go-Explore (full)

**Not pirated / not used as source:** commercial print-only editions without author PDF
(e.g. older Puterman 1994 hardcover, AIMA). Prefer author-posted free copies above.

---

## 13. Open theory→code gaps (actionable)

1. Explicit \(\beta\) / initiation sets on SMA4 options (white-block, inventory, flight).
2. Measured multi-time models \(p_o,r_o\) from ROM (replace symbolic warpless/Bowser).
3. SMA4 global \(\Phi\) that does not cap at the card.
4. Optional interruption at meta layer once option-values exist.
5. Keep climbing the knowledge ladder: AcquireWhistle_1_3 → second W1 whistle → drop
   Tier-1 re-grants.

These are engineering instantiations of the math above — not new paradigms.
