# Theory corpus (first principles)

Legal, author-posted PDFs and drafts for grounding the Mario AI / Option-SMDP stack.
**Do not commit the PDF binaries** (`pdfs/` is gitignored; ~215 MB). Re-fetch with:

```bash
./venv/bin/python notes/theory/fetch_corpus.py
```

Extracted text (for offline reading / search) regenerates via the same script into
`notes/theory/extracted/` (also gitignored). The durable synthesis is
`notes/theory/first-principles.md`.

## Canonical free textbooks (download these first)

| Priority | Work | Why it matters here | URL / local |
|---|---|---|---|
| 0 | **Bellman, *Dynamic Programming* (Princeton, 1957)** | Root text: Principle of Optimality, functional equations, policy-space approx., curse of dimensionality | Local: `pdfs/dynamic programming.pdf` (user-supplied). Digest: `bellman-1957.md` |
| 1 | Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) | MDP, DP, planning vs learning, options (Ch. 17.2), AlphaGo case study | http://incompleteideas.net/book/RLbook2020.pdf |
| 2 | Bertsekas, *A Course in Reinforcement Learning* (2nd ed., free PDF) | Deterministic DP, rollout, approximation in value space | https://web.mit.edu/dimitrib/www/RLCOURSECOMPLETE%202ndEDITION.pdf |
| 3 | Bertsekas, *Lessons from AlphaZero for Optimal, MPC, and Adaptive Control* | Off-line training + on-line Newton/lookahead; why nets must not replace search | https://web.mit.edu/dimitrib/www/LessonsfromAlphazero.pdf |
| 4 | Bertsekas, *Rollout, Policy Iteration, and Distributed RL* | Rollout = one PI / Newton step; fortified / truncated rollout | https://web.mit.edu/dimitrib/www/Rollout_Complete%20Book.pdf |
| 5 | Puterman & Chan, *MDPs and RL* (pre-publication draft chapters) | Classical MDP definitions, finite-horizon optimality of MD policies | https://github.com/martyput/MDP_book |

## Core papers (also fetched)

| Paper | Local name | URL |
|---|---|---|
| Ng, Harada, Russell — potential shaping (ICML 1999) | `NgHaradaRussell-shaping-ICML1999.pdf` | https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf |
| Sutton, Precup, Singh — options (AIJ 1999) | `SuttonPrecupSingh-options-AIJ1999.pdf` | https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf |
| Ross, Gordon, Bagnell — DAgger | `RossGordonBagnell-DAgger.pdf` | https://arxiv.org/pdf/1011.0686.pdf |
| Anthony, Tian, Barber — Expert Iteration | `AnthonyTianBarber-ExIt.pdf` | https://arxiv.org/pdf/1705.08439.pdf |
| Silver et al. — AlphaZero | `Silver-AlphaZero.pdf` | https://arxiv.org/pdf/1712.01815.pdf |
| Ecoffet et al. — Go-Explore | `Ecoffet-GoExplore.pdf` | https://arxiv.org/pdf/2004.12919.pdf |

## Reading order (absolute basics → our stack)

0. **Bellman 1957 Ch. I + III** (Principle of Optimality, curse of dimensionality, policy-space approx.) — then skim II, IV; later chapters as interest
1. Deterministic dynamics / shortest paths (Bertsekas RL course §1.2)
2. Finite MDPs + Bellman eq. (Sutton Ch. 3–4; Puterman Ch. 1–4)
3. Planning with a known model (Sutton Ch. 8)
4. Potential shaping (Ng et al.)
5. Options → SMDP (Sutton–Precup–Singh; Sutton §17.2)
6. Approximation in value space / rollout / Newton (Bertsekas *Lessons* + *Rollout*)
7. ExIt + AlphaZero (search teaches; net guides)
8. DAgger bounds (why standalone IL fails at death cliffs)
9. Go-Explore (when Φ-beam detaches)

See `first-principles.md` for the synthesis and `bellman-1957.md` for the 1957 chapter digest.
