# 2026-07-15 — First-principles theory pass (textbooks + papers)

## Intent

Restart from absolute basics: download legal free textbooks/PDFs, read them
(end-to-end where short; chapter-systematic where long), and synthesize the math
under the Mario AI / SMA4 Option-SMDP stack.

## Corpus assembled (legal / author-posted)

Under `notes/theory/pdfs/` (gitignored; re-fetch via `notes/theory/fetch_corpus.py`):

| Work | Pages | Role |
|---|---|---|
| Sutton & Barto RL book 2e | 548 | MDP, DP, planning, options |
| Bertsekas *Course in RL* 2e | 523 | Deterministic DP, rollout |
| Bertsekas *Lessons from AlphaZero* | 242 | Off-line + on-line Newton view |
| Bertsekas *Rollout / PI / Distributed RL* | 499 | Rollout theory |
| Bertsekas Abstract DP lecture + DP slides | — | Operators / contraction |
| Puterman & Chan MDP draft (GitHub chapters) | ~800 | Classical MDP foundations |
| Ng–Harada–Russell shaping; SPS options; DAgger; ExIt; AlphaZero; Go-Explore | short | Core papers |

Manifest: `notes/theory/README.md`.

## Reading method

`pdftotext -layout` → `notes/theory/extracted/` (gitignored). Short papers read
fully. Long books read by the project-critical chapters (Sutton 1–4, 8, 16–17;
Puterman 1–4; Bertsekas AZ + RL-course early + rollout early), with structured
digests then merged into the synthesis.

## Durable output

**`notes/theory/first-principles.md`** — layer stack 0→8 with equations, theorems,
repo mappings, and “what not to do.”

## Conclusions that survive the reading

1. Perfect deterministic \(f\) ⇒ **planning is primary**; model-free nets are
   accelerators (Sutton 8, Puterman 1, Bertsekas).
2. “Net serves search” is Bertsekas’ **Newton step on Bellman** + ExIt/AlphaZero —
   not a slogan.
3. Options/SMDP (SPS 1999) is the correct abstraction for SMB3 items/levels;
   opaque effects must be executed to be planned over.
4. Ng potentials justify area-aware \(\Phi\); non-potential shaping is how agents
   farm forever.
5. DAgger’s \(uT\varepsilon\) bound + absorbing deaths explain V6’s standalone IL
   ceiling; another DAgger round is the wrong next bet.
6. Go-Explore archive+return is the theory behind coverage search and whistle
   side-routes.

## Follow-up (same day) — Bellman 1957 front-to-back

User supplied `notes/theory/pdfs/dynamic programming.pdf` (Bellman, Princeton 1957,
OCR 365 pp.). Read end-to-end; durable digest `notes/theory/bellman-1957.md`;
folded into `first-principles.md` as Layer 0.5. Reading order in
`notes/theory/README.md` now starts with Bellman Ch. I+III.

## What did *not* change

- Experimental next action remains: climb the AcquireWhistle knowledge ladder
  (second W1 whistle; drop Tier-1 re-grants).
- No new game solves claimed in this pass.

## Verification

- Corpus fetchable via listed URLs; synthesis committed under `notes/theory/`.
- Bibliography updated with textbook section.
