# Bellman (1957) — *Dynamic Programming* (front-to-back digest)

> Source: `notes/theory/pdfs/dynamic programming.pdf` (Princeton, 1957;
> OCR’d 365 pp.). Read end-to-end 2026-07-15. OCR is noisy; equations below
> are reconstructed from the clearest prose + repeated forms.
>
> This is the **foundational** text. Sutton, Puterman, and Bertsekas are later
> packaging of the same spine. Repo map: `notes/theory/first-principles.md`.

---

## What the book is

Bellman coins **dynamic programming** for the mathematical theory of
**multi-stage decision processes**: a system with state \(p\); at each stage a
**decision** that selects a **transformation** \(T_q\) of that state; a
**policy** = rule for choosing \(q\) given \(p\) (and stage); a **criterion**
measuring the resulting trajectory (final-state return, expected gold, etc.).

The methodological break with “enumerative” maximization over the full
sequence of decisions: instead of one \(MN\)-dimensional search, **imbed** the
problem in a family of similar problems and seek the **structure of the
optimal policy** \(q^*(p)\).

---

## The Principle of Optimality (Ch. III §3)

Quoted (cleaned) from the OCR:

> **PRINCIPLE OF OPTIMALITY.** An optimal policy has the property that
> whatever the initial state and initial decision are, the remaining decisions
> must constitute an optimal policy with regard to the state resulting from
> the first decision.

“A proof by contradiction is immediate.” Every functional equation in the
book is a transliteration of this principle.

### Discrete deterministic functional equation (Ch. III §4)

State \(p\), family \(\{T_q\}\), \(N\)-stage return \(f_N(p)\):

\[
f_N(p)=\max_q\, f_{N-1}\bigl(T_q(p)\bigr),\qquad
f_1(p)=\max_q\, R\bigl(T_q(p)\bigr).
\]

With immediate stage return \(g\):

\[
f_N(p)=\max_q\Bigl[g(p,q)+f_{N-1}\bigl(T_q(p)\bigr)\Bigr].
\]

Infinite horizon / survival factor \(h\):

\[
f(p)=\max_q\bigl[g(p,q)+h(p,q)\,f\bigl(T(p,q)\bigr)\bigr].
\]

### Discrete stochastic (Ch. III §5)

Decision induces a distribution \(dG_q(p,z)\) over next states; maximize expected
return. With observation of the realized \(z\) before the next decision:

\[
f_N(p)=\max_q\int f_{N-1}(z)\,dG_q(p,z).
\]

### Continuous deterministic (Ch. III §6 → Ch. IX)

\(f(p;T)\) over horizon length \(T\); first interval of length \(S\):

\[
f(p;S+T)=\max_{\text{policies on }[0,S]}\, f(p_S;T).
\]

\(S\to 0\) yields a nonlinear PDE; calculus of variations is rewritten as DP
(Ch. IX).

---

## Approximation in policy space (Ch. I §11; Ch. III §10)

Bellman distinguishes:

| Mode | Procedure | Property |
|---|---|---|
| Approximation in **function / value** space | Iterate \(f_{n+1}=\max_q T(f_n,q)\) from any \(f_0\) | Classical successive approximations |
| Approximation in **policy** space | Guess \(q_0(p)\), evaluate \(f_0\) of that policy, improve \(q_1=\arg\max T(f_0,q)\), … | Often **monotone** (\(f_{n+1}\ge f_n\) when \(h\ge 0\)); “by far the more natural” in applications |

This is the conceptual ancestor of **policy iteration** (Howard) and of
Bertsekas’ “rollout / Newton step from a base heuristic,” and of this repo’s
“net serves the search” (prior = \(q_0\); search = improvement).

---

## Chapter map (front to back)

| Ch | Title (OCR-reconstructed) | Core content | Mario relevance |
|---|---|---|---|
| I | Allocation processes | Resource split; curse of dimensionality; \(f_N(x)=\max_y[g+h+f_{N-1}(ay+b(x-y))]\); policy structure (convex→bang-bang) | Finite-horizon DP; dimensionality ⇒ don’t tabulate RAM |
| II | Stochastic gold-mining | Two mines, fragile machine; decision **regions**; expected return FE | Opaque stochastic effects; region structure |
| III | A synthesis | **Principle of Optimality**; discrete/continuous/stochastic FEs; policy-space approx. | **Foundational chapter for this repo** |
| IV | Existence / uniqueness | Types One–Three (contraction / survival); comparison lemma; stability of \(f\) under \(g\)-perturbation | Why discounted / terminating problems are well-posed |
| V | Optimal inventory | Base-stock / (s,S); Arrow–Harris–Marschak lineage; FOCs via \(u'=v_x\) | Analogy only (stochastic demand ≠ emulator) |
| VI–VII | Bottleneck processes | Capacity–stockpile industrial models; continuous PDE; dual verification; homogeneity vs dimensionality | Dimensionality honesty; dual verification ≈ replay checks |
| VIII | Continuous gold-mining | Mixing / singular arcs on switching curve | Discrete buttons: structure analogy only |
| IX | Calculus of variations via DP | \(0=\max_v[F+G f']\); Euler as characteristics; discrete recurrence for computation; policy-space approx. | Prefer discrete recurrence over fragile PDE grids |
| X | Multi-stage games | MaxMin FE; games of survival; pursuit (Isaacs) | Option-SMDP is structured multi-stage **without** adversary |
| XI | Markovian decision processes | Controlled transition matrices; growth / Perron asymptotics; policy-space improvement | Shared DNA with modern MDPs; objective geometry differs from discounted \(V^*\) |

---

## Dictionary: Bellman 1957 → mario_ai

| Bellman | This repo |
|---|---|
| State \(p\) | Savestate / adapter RAM snapshot |
| Decision \(q\) | Button / chunk / **option** |
| Transformation \(T_q(p)\) | `env.step` / option executor |
| \(f_N(p)\) | Best \(N\)-stage return (progress Φ / flag / card / World-8) |
| \(g(p,q)\) | Immediate progress / frame cost |
| Optimal policy \(q^*(p)\) | What search finds; what a prior approximates |
| Enumerative \(MN\)-dim max | Naive full-horizon button sequences |
| Imbedding (family of problems) | Solve from many snapshots / cells |
| Approx. in policy space | Prior / option library → search improvement |
| Approx. in value space | `ValueNet` / Φ as \(\tilde J\) |
| Curse of dimensionality | Why beam/coverage, not tabular VI over RAM |
| Decision regions | Pipe-entry / white-block / inventory menus |

**Beam search** ≈ truncated \(N\)-stage DP that keeps only a width-\(W\) frontier
of partial policies — an engineering approximation, not Bellman’s proved
successive-approximation theorem. Exact emulator + restore is Bellman’s
**discrete deterministic** case exactly (Ch. III §4); expectation apparatus is
unnecessary at the frame level.

**Option-SMDP** (whistle planning) is Bellman’s multi-stage skeleton one
timescale up: meta-state \(p\), option = \(T_q\), opaque effects observed by
execution (his stochastic/opaque mindset with a deterministic \(f\) underneath).

---

## What later chapters do *not* invent for us

- Stochastic inventory (Ch. V), continuous mixing (Ch. VIII), industrial bottleneck
  ODEs (VI–VII), and adversarial MaxMin (Ch. X) are **models**, not templates.
- Ch. XI is not yet the modern discounted MDP textbook equation; that packaging
  arrives with Howard / Blackwell / Puterman. Use Sutton/Puterman/Bertsekas for
  the discrete MDP algorithm layer; use Bellman 1957 for the **principle**, the
  functional-equation viewpoint, policy-space approximation, and the curse of
  dimensionality.

---

## One-line takeaway

Bellman: *imbed, apply the Principle of Optimality, prefer policies over open-loop
sequences, and never pretend to tabulate the full state space.* That is exactly
the spine of emulator search + options + “net serves search.”
