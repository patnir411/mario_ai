# Research Bibliography and Consolidated Field Notes

Research availability cutoff: **2026-07-25**. Repository verification updated
2026-07-27 in America/Detroit; no post-cutoff source is used.

This file consolidates the literature and tool references that informed the V4-V6 direction, the
generalist-policy reassessment, the "net serves search" pivot, the small-LLM/VLM discussion, and
the cross-game Mario adapter work. The July 25 refresh prioritizes methods that
map to falsifiable experiments in this repository rather than architecture
fashion: completeness-safe policy-guided search, unknown-option planning,
variable-duration actions, selective policy handoff, and reliable evaluation.

**First-principles synthesis (2026-07-15):** `notes/theory/first-principles.md`
(corpus fetch: `notes/theory/README.md`).

## High-Level Conclusions

1. Search is the current solver when an exact, resettable emulator is available.
   - Mario AI competition history and planning/distillation work motivate this design; they do
     not prove that search is universally superior to learning.
   - In this repo, the working version is beam/coverage search over emulator snapshots, with learned
     policy/value models treated as accelerators rather than standalone replacements.

2. A zero-shot generalist Mario controller is not supported by the fixed stock-level evidence and
   is not the current near-term deliverable.
   - The ICLR 2025 data-scaling paper that motivated "dozens of environments might be enough" is
     about robotic manipulation, not long-horizon platformer level generalization.
   - Procgen/CoinRun is the closer analogue: hundreds of levels can still overfit, and robust
     train-to-test transfer usually needs far more procedural diversity than the 32 stock SMB levels.

3. Offline BC/DAgger is useful, but the repo's tested recipes hit a local standalone
   closed-loop-control wall.
   - V4/V5/V6 results: structured/entity policies improve validation accuracy and can run/jump, but
     still fail closed-loop completion as generalists; DAgger helped some train levels and did not
   produce held-out transfer. This is an empirical boundary for the tested data,
   architectures, and teacher—not a universal impossibility result for imitation learning.
   - Practical direction: keep per-level cached search solutions/rescue; use
     learned priors inside search where they reduce nodes without reducing
     matched-baseline solve rate or primitive-action support. The bounded beam
     baseline is itself incomplete; PHS-style completeness guarantees require a
     separate algorithm.

4. Reverse-curriculum RL remains the serious option if we ever need a reactive standalone specialist.
   - RFCL/Go-Explore/RLPD/BBF support the recipe: start near successful states, use saved demos and
     online rollouts, and use modern stability/plasticity fixes.
   - The historical partial-suffix observation has no preserved report or checkpoint. The
     hardened `scripts/rc_rl.py` is therefore an experimental driver, not a positive result.

5. Small LLMs/VLMs are not recommended for real-time low-level Mario control in this project.
   - Pretrained language can help some sequential decision problems, but the gains appear to come
     from sequence bias/initialization rather than natural-language knowledge.
   - VideoGameBench reports very limited direct game completion for the evaluated VLM agents.
     No local model-size/latency sweep was run here, so repository-specific thresholds remain
     unmeasured.
   - Better use: offline subgoal proposal, code/RE assistance, or level-generation tooling.

6. Cross-game Mario should stay adapter-first.
   - Stable-Retro/mGBA and PyBoy give usable emulator APIs and savestates.
   - Each game still needs per-game boot, RAM, action, terminal, and progress semantics before any
   learned model or generic search heuristic is meaningful.

7. The next learned-search experiment should preserve action support.
   - Hard top-k pruning produced the repo's only positive learned-search result and
     its clearest failure. Policy-Guided Heuristic Search (PHS), Levin-style search,
     or an explicit nonzero primitive-action fallback is a stronger target than
     simply training a larger prior.
   - Evaluate node/time reduction only subject to solve-rate non-inferiority.

8. The SMA4 whistle work is currently a segmented unknown-option prototype.
   - It has replay-verified low-level segments and a useful execute-to-observe
     planning scaffold, but cached entry restores, state writes, symbolic endpoints,
     and an intentionally nonexploring greedy baseline prevent an end-to-end
     Option-SMDP claim.
   - Exact boundary instrumentation now gives a constructive counterexample to
     `MetaState` sufficiency: opposite whistle-acquisition orders reach one
     symbolic warp state with different physical emulator/RAM states. The
   Opt-in multi-representative physical search and finite option-signature
   refinement now preserve both acquisition histories. They separate under
   short option suffixes: only the 1-3-then-fortress order reaches an
   input-responsive World-8 map in the current Tier-3 implementation. The
   immediate gate is to explain that lineage asymmetry, remove the Tier-3
   cursor repair, and then join legitimate power acquisition to the live
   unpowered fortress entry.

## References

### Foundational textbooks (free / author-posted)

These are the math backbone for the Option-SMDP + search-first stack. Local PDF
corpus is gitignored; re-fetch with `notes/theory/fetch_corpus.py`.
Bellman 1957 digest: `notes/theory/bellman-1957.md`.

- Richard Bellman, *Dynamic Programming*, Princeton University Press, 1957.
  - Local: `notes/theory/pdfs/dynamic programming.pdf` (user-supplied OCR scan, 365 pp.)
  - Repo relevance: **root text** — Principle of Optimality (Ch. III §3), discrete /
    stochastic / continuous functional equations, approximation in policy space,
    curse of dimensionality. Read front-to-back 2026-07-15; all later MDP/RL books
    in this list package this spine.

- *The Bhagavad-Gita* (*The Song Celestial*), tr. Sir Edwin Arnold (1900);
  Project Gutenberg EBook #2388.
  - Local: `notes/theory/pdfs/gutenberg bhagavad gita.pdf`
  - PG: https://www.gutenberg.org/ebooks/2388
  - Repo relevance: read front-to-back for universal agency truths and research ethos;
    **not** a source of MDP theorems. Adversarial bridge (Keep/Kill analogies, revised
    thesis): `notes/theory/gita-and-first-principles.md`.

- Richard S. Sutton, Andrew G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
  - Free PDF: http://incompleteideas.net/book/RLbook2020.pdf
  - Repo relevance: finite MDPs (Ch. 3), DP (Ch. 4), planning with models (Ch. 8),
    options (Ch. 17.2), AlphaGo case study (Ch. 16). Primary undergraduate/grad entry text.

- Dimitri P. Bertsekas, *A Course in Reinforcement Learning* (2nd ed., Athena; free PDF).
  - https://web.mit.edu/dimitrib/www/RLCOURSECOMPLETE%202ndEDITION.pdf
  - Hub: https://www.mit.edu/~dimitrib/RLbook.html
  - Repo relevance: deterministic DP, approximation in value space, rollout algorithms —
    the cleanest statement of “on-line lookahead over a known model.”

- Dimitri P. Bertsekas, *Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control*.
  - https://web.mit.edu/dimitrib/www/LessonsfromAlphazero.pdf
  - Repo relevance: off-line training + on-line play; Newton view of rollout / policy
    iteration; why a policy network alone is weaker than one lookahead/Newton step.

- Dimitri P. Bertsekas, *Rollout, Policy Iteration, and Distributed Reinforcement Learning*.
  - https://web.mit.edu/dimitrib/www/Rollout_Complete%20Book.pdf
  - Repo relevance: fortified/truncated rollout, cost-improvement guarantees, MCTS as
    stochastic rollout relative.

- Martin L. Puterman, Timothy C. Y. Chan, *Markov Decision Processes and Reinforcement Learning*
  (pre-publication draft chapters, personal use).
  - https://github.com/martyput/MDP_book
  - Repo relevance: classical MDP definitions; finite-horizon optimality of Markov
    deterministic policies; model-based vs model-free framing.

- Dimitri P. Bertsekas, abstract DP / contraction lecture notes (free).
  - http://web.mit.edu/dimitrib/www/Abstract_DP_RL_Lecture.pdf
  - Repo relevance: Bellman operators \(T,T_\mu\), contractive vs semicontractive
    (goal/SSP-like) models — closest formal home for “reach the flag / World 8.”

### Mario AI, Search, and Planning

- Julian Togelius, Sergey Karakovskiy, Robin Baumgarten, "The 2009 Mario AI Competition", CEC 2010.
  - PDF: https://julian.togelius.com/Togelius2010The.pdf
  - Repo relevance: establishes the forward-model/A* baseline tradition for Mario agents.

- Sergey Karakovskiy, Julian Togelius, "The Mario AI Benchmark and Competitions", IEEE TCIAIG 2012.
  - Publisher page: https://pure.itu.dk/en/publications/the-mario-ai-benchmark-and-competitions/
  - PDF: https://julian.togelius.com/Karakovskiy2012The.pdf
  - Repo relevance: useful historical context; note this is the Infinite Mario-style benchmark, not
    the original NES ROM environment used here.

- Robin Baumgarten / Mario AI A* agent, competition context.
  - AI Magazine overview: https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2492
  - Repo relevance: validates the core "planning with a forward model beats pure learning" premise.

- Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, "A Reduction of Imitation Learning and
  Structured Prediction to No-Regret Online Learning" (DAgger), AISTATS 2011.
  - PMLR: https://proceedings.mlr.press/v15/ross11a.html
  - arXiv: https://arxiv.org/abs/1011.0686
  - Repo relevance: source for the DAgger/aggregate-correction experiments.

- Andrew Y. Ng, Daishi Harada, Stuart Russell, "Policy invariance under reward transformations:
  Theory and application to reward shaping", ICML 1999.
  - PDF: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
  - Repo relevance: defines potential-based reward shaping. The repo's current
    \(\Phi\) is a potential-inspired search ranking unless it is actually applied
    as \(\gamma\Phi(s')-\Phi(s)\); do not claim the policy-invariance theorem for
    the ranking heuristic.

- Thomas Anthony, Zheng Tian, David Barber, "Thinking Fast and Slow with Deep Learning and Tree
  Search" (Expert Iteration), NeurIPS 2017.
  - arXiv: https://arxiv.org/abs/1705.08439
  - Repo relevance: search-as-teacher and policy/value distillation framing.

- David Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement
  Learning Algorithm" (AlphaZero), 2017.
  - arXiv: https://arxiv.org/abs/1712.01815
  - Repo relevance: learned policy/value guide search over real rules; conceptual analogue for
    "net serves search".

- Ivo Danihelka et al., "Policy improvement by planning with Gumbel" (Gumbel AlphaZero), ICLR 2022.
  - OpenReview: https://openreview.net/forum?id=bERaNdoegnO
  - Repo relevance: policy improvement with search when simulations are limited.

- Adrien Ecoffet et al., "First return, then explore" (Go-Explore), Nature 2021.
  - arXiv: https://arxiv.org/abs/2004.12919
  - Repo relevance: archive/cell-return exploration maps closely to this repo's coverage search and
    saved-state workflow.

- Chris Lu, Shengran Hu, Joel Lehman, Jeff Clune, "Intelligent Go-Explore", ICLR 2025.
  - arXiv: https://arxiv.org/abs/2405.15143
  - Repo relevance: supports using foundation models/offline analysis for route/cell/subgoal
    proposal while keeping frame-level control in emulator search.

- Bandres, Bonet, Geffner, "Planning with pixels in Atari by width-based IW search", 2018.
  - arXiv PDF: https://arxiv.org/pdf/1801.03354
  - Related overview: https://www.ijcai.org/proceedings/2021/0702.pdf
  - Repo relevance: conceptual basis for novelty/cell archive search.

- Clara Meister, Tim Vieira, Ryan Cotterell, "Best-First Beam Search", TACL 2020.
  - arXiv: https://arxiv.org/abs/2007.03909
  - ACL Anthology: https://aclanthology.org/2020.tacl-1.51/
  - Repo relevance: background for beam-style heuristic search, though our search is emulator-state
    planning rather than sequence decoding.

### 2026-07-25 actionable planning and evaluation refresh

Ranked by direct value to the current codebase:

1. **Laurent Orseau, Levi H. S. Lelis, "Policy-Guided Heuristic Search with
   Guarantees" (AAAI 2021).**
   - AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/17469
   - arXiv: https://arxiv.org/abs/2103.11505
   - Repo experiment: replace incomplete fixed `policy_topk` beam pruning with
     PHS/PHS*- or Levin-style best-first enumeration using cumulative path
     probability and a progress/goal heuristic. Compare expansions, wall time,
     and solve rate against plain beam, soft log-prior beam, and hard top-k.
   - Why first: it directly targets this project's metric—single-agent
     deterministic search effort—and supplies search-loss guarantees tied to
     both policy and heuristic quality.

2. **Jake Tuero, Michael Buro, Levi H. S. Lelis, "Subgoal-Guided Policy
   Heuristic Search with Learned Subgoals" (ICML 2025).**
   - PMLR: https://proceedings.mlr.press/v267/tuero25a.html
   - arXiv: https://arxiv.org/abs/2506.07255
   - Repo experiment: retain failed 6-2/fortress search trees and learn
     subgoal-conditioned policies/heuristics from both successful and failed
     expansions instead of discarding unsuccessful runs. Start with explicit
     RAM-derived subgoals before learning the subgoal representation.

3. **Palash Chatterjee, Roni Khardon, "Improving planning and MBRL with
   temporally-extended actions" (NeurIPS 2025).**
   - NeurIPS: https://papers.nips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html
   - arXiv: https://arxiv.org/abs/2505.15754
   - Repo experiment: treat hold duration as a search variable—initially
     \(\{1,2,4,8,16\}\) frames, then a bandit-selected range—rather than the
     current coarse split between fixed cf8 exploration and cf1 specialists.
     Measure whether phase-sensitive 6-2/lift states become reachable at equal
     primitive-frame and wall-clock budgets. The paper studies continuous-time
     planning, so the Mario mapping is a hypothesis, not a transferred theorem.

4. **Arthur Guez, David Silver, Peter Dayan, "Efficient Bayes-Adaptive
   Reinforcement Learning using Sample-Based Search" (NeurIPS 2012) and
   "Scalable and Efficient Bayes-Adaptive Reinforcement Learning" (JAIR 2013).**
   - NeurIPS: https://proceedings.neurips.cc/paper/2012/hash/35051070e572e47d2c26c241ab88307f-Abstract.html
   - JAIR author PDF: https://www.gatsby.ucl.ac.uk/~aguez/files/guez_jair2013.pdf
   - Repo experiment: use a small BAMCP-style history/belief baseline on generated
     unknown-option graphs and the SMA4 whistle instance. This is a control for
     whether execute-to-observe uniform-cost search is solving an exploration
     problem that benefits from an explicit posterior over option effects.

5. **Surbhi Goel, Jonathan Pei, James Wang, "Learning When to Stop:
   Selective Imitation Learning Under Arbitrary Dynamics Shift" (2026 preprint).**
   - arXiv: https://arxiv.org/abs/2605.09183
   - Status: May 2026 preprint / NeurIPS 2026 submission; not a settled result.
   - Repo experiment: calibrate a policy abstention rule that hands uncertain or
     shifted states back to search. This formalizes "net acts, search rescues"
     better than forcing a standalone controller, although the paper's unlabeled
     test-expert-trajectory assumptions do not match Mario exactly.

6. **Akshay Krishnamurthy, Gene Li, Ayush Sekhari, "The Role of Environment
   Access in Agnostic Reinforcement Learning" (COLT 2025).**
   - PMLR: https://proceedings.mlr.press/v291/krishnamurthy25a.html
   - Full paper: https://arxiv.org/abs/2504.05405
   - Repo relevance: makes the reset/snapshot access model a first-class
     experimental variable. Its lower bounds concern agnostic policy learning
     under particular access and representation assumptions; they do not prove
     that search always beats learning in Mario.

7. **Rishabh Agarwal et al., "Deep Reinforcement Learning at the Edge of the
   Statistical Precipice" (NeurIPS 2021).**
   - NeurIPS: https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
   - Repo experiment: report paired solve-rate, censored time-to-solve,
     interquartile mean, bootstrap intervals, performance profiles, and
     probability of improvement across levels/start snapshots. Deterministic
     emulator seeds are not independent evidence when they generate the same
     trajectory.

Secondary, conditional leads:

- Dan Haramati et al., "Hierarchical Entity-centric Reinforcement Learning with
  Factored Subgoal Diffusion" (ICLR 2026).
  - OpenReview: https://openreview.net/forum?id=TimC6hxVHj
  - Repo experiment: compare its factored subgoal idea, used strictly as an
    offline proposal mechanism, against hand-written and failed-tree-mined
    subgoals for SMA4 options. The paper studies offline goal-conditioned RL;
    it is not evidence to replace exact option execution or uniform-cost search.
- Ferdinand Kapl et al., "Are Object-Centric Representations Better At
  Compositional Generalization?" (February 2026 preprint).
  - arXiv: https://arxiv.org/abs/2602.16689
  - Relevance: current evidence that object-centric representations can help
    under constrained data/diversity/compute, but in visual question answering,
    not control. It strengthens the case for an entity-vs-dense ablation while
    warning against treating object tokens as an automatic generalization result.
- Thomas T. C. K. Zhang et al., "Action Chunking and Data Augmentation Yield
  Exponential Improvements in Behavior Cloning for Continuous Spaces" (ICLR
  2026).
  - OpenReview: https://openreview.net/forum?id=jiWXDvw1Lf
  - Relevance: supports testing chunked policy outputs and exploratory
    augmentation if standalone control is reopened. Its guarantees are for
    continuous-space assumptions and should not be quoted as guarantees for
    discrete Mario inputs.
- Shaunak A. Mehta et al., "Stable-BC: Controlling Covariate Shift with Stable
  Behavior Cloning" (IEEE RA-L 2025).
  - Project/paper: https://collab.me.vt.edu/Stable-BC/
  - arXiv: https://arxiv.org/abs/2408.06246
  - Audit finding: the former `mario/stable_bc.py` perturbation-consistency loss,
    now honestly named `mario/consistency.py`,
    does not estimate or constrain the paper's closed-loop error-dynamics
    Jacobian. Preserve it as a sensitivity regularizer, but do not call it an
    implementation of Stable-BC or infer contraction from it.
- Zhou et al., "Stay Hungry, Keep Learning: Sustainable Plasticity for Deep
  Reinforcement Learning" (ICML 2025).
  - PMLR: https://proceedings.mlr.press/v267/zhou25am.html
  - Relevance: neuron regeneration / reset-and-distill methods are worth an
    ablation only if the reverse-curriculum RL track is resumed.

### Generalization and Imitation Learning

- Fanqi Lin et al., "Data Scaling Laws in Imitation Learning for Robotic Manipulation", ICLR 2025.
  - arXiv: https://arxiv.org/abs/2410.18647
  - Project page: https://data-scaling-laws.github.io/
  - Repo relevance: originally motivated more-data/more-level diversity, but the setting is robotic
    manipulation rather than platformer level generalization; do not directly transfer its level-count
    implications to Mario.

- Karl Cobbe et al., "Quantifying Generalization in Reinforcement Learning", ICML 2019.
  - PMLR: https://proceedings.mlr.press/v97/cobbe19a.html
  - arXiv: https://arxiv.org/abs/1812.02341
  - Repo relevance: closer analogue for Mario generalization; procedural train/test splits reveal
    substantial overfitting.

- Dylan J. Foster, Adam Block, Dipendra Misra, "Is Behavior Cloning All You Need? Understanding
  Horizon in Imitation Learning", NeurIPS 2024.
  - OpenReview: https://openreview.net/forum?id=8KPyJm4gt5
  - arXiv: https://arxiv.org/abs/2407.15007
  - Repo relevance: clarifies when offline BC can be strong and when horizon/death-cliff issues still
    matter.

- Mengjiao Yang, Dale Schuurmans, Pieter Abbeel, Ofir Nachum, "Chain of Thought Imitation with
  Procedure Cloning", 2022.
  - arXiv: https://arxiv.org/abs/2205.10816
  - Repo relevance: supports distilling expert computation/procedure rather than only final actions.

- Ev Zisselman et al., "Explore to Generalize in Zero-Shot RL", NeurIPS 2023.
  - arXiv: https://arxiv.org/abs/2306.03072
  - NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2023/file/c793577b644268259b1416464a6cdb8c-Paper-Conference.pdf
  - Repo relevance: supports the finding that reward-optimal policies can memorize while exploratory
    behavior generalizes better.

- Haoqun Cao, Tengyang Xie, "Understanding Behavior Cloning with Action Quantization", 2026.
  - arXiv: https://arxiv.org/abs/2603.20538
  - Repo relevance: a preprint about quantizing continuous actions. Direct relevance
    to an already-discrete Mario controller is weak; keep it as background rather
    than evidence that the local BC recipe should succeed.

- Yuda Song et al., "To Distill or Decide? Understanding the Algorithmic Trade-off in Partially
  Observable Reinforcement Learning", NeurIPS 2025.
  - NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2025/hash/81343f8ebe529de8d0bb654af7523184-Abstract-Conference.html
  - arXiv: https://arxiv.org/abs/2510.03207
  - OpenReview: https://openreview.net/forum?id=iEgaS6wbLa
  - Repo relevance: supports the nuanced view that privileged-state distillation can be efficient but
    is not universally superior to RL under partial observability.

### Reverse Curriculum, RL from Demos, and Stability

- Stone Tao et al., "Reverse Forward Curriculum Learning for Extreme Sample and Demonstration
  Efficiency in Reinforcement Learning", ICLR 2024.
  - OpenReview: https://openreview.net/forum?id=w4rODxXsmM
  - arXiv: https://arxiv.org/abs/2405.03379
  - Project page: https://reverseforward-cl.github.io/
  - Repo relevance: strongest reference for the saved-solution/reverse-curriculum RL direction.

- Tim Salimans, Richard Chen, "Learning Montezuma's Revenge from a Single Demonstration", 2018.
  - arXiv: https://arxiv.org/abs/1812.03381
  - Repo relevance: supports reverse-curriculum/demonstration approaches for hard exploration.

- Philip J. Ball et al., "Efficient Online Reinforcement Learning with Offline Data" (RLPD), ICML 2023.
  - arXiv: https://arxiv.org/abs/2302.02948
  - PMLR PDF: https://proceedings.mlr.press/v202/ball23a/ball23a.pdf
  - Repo relevance: practical online+offline replay recipe for using demos plus new rollouts.

- Max Schwarzer et al., "Bigger, Better, Faster: Human-level Atari with human-level efficiency" (BBF), 2023.
  - arXiv: https://arxiv.org/abs/2305.19452
  - Repo relevance: sample-efficient Atari RL and plasticity/stability reference for any future
    reactive-policy work.

### World Models, LLMs, and VLMs

- Julian Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned
  Model" (MuZero), 2019.
  - arXiv: https://arxiv.org/abs/1911.08265
  - Repo relevance: useful background, but less directly useful here because the emulator already
    supplies exact dynamics.

- Danijar Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), 2023/2025.
  - arXiv: https://arxiv.org/abs/2301.04104
  - Nature: https://www.nature.com/articles/s41586-025-08744-2
  - Repo relevance: world models matter when dynamics are missing or expensive; lower priority here.

- Shuang Li et al., "Pre-Trained Language Models for Interactive Decision-Making", 2022.
  - arXiv: https://arxiv.org/abs/2202.01771
  - Repo relevance: language-model initialization can help sequential decision-making, but their own
    ablation says natural-language formatting itself had little influence; that supports using typed
    entity tokens rather than a text-token control loop.

- Shiro Takagi, "On the Effect of Pre-training for Transformer in Different Modality on Offline
  Reinforcement Learning", NeurIPS 2022.
  - arXiv: https://arxiv.org/abs/2211.09817
  - Repo relevance: modality pretraining effects are mixed; do not assume text pretraining transfers
    directly to Mario control.

- Ruizhe Shi et al., "Unleashing the Power of Pre-trained Language Models for Offline Reinforcement
  Learning" (LaMo), 2023.
  - arXiv: https://arxiv.org/abs/2310.20587
  - Project page: https://lamo2023.github.io/
  - Repo relevance: supports the idea that LM initialization can help low-data offline RL, but still
    not a reason to put an LLM in the per-frame control loop.

- Alex L. Zhang et al., "VideoGameBench: Can Vision-Language Models complete popular video games?",
  2025.
  - arXiv: https://arxiv.org/abs/2505.18134
  - Project page: https://vgbench.com/
  - Repo relevance: strong evidence against relying on frontier VLMs for direct real-time game play;
    even best-reported completion is tiny.

- "The PokéAgent Challenge", 2026.
  - arXiv: https://arxiv.org/abs/2603.15563
  - Repo relevance: current benchmark for Pokémon-class long-horizon game completion; useful
    contrast for the SMB3 thesis because Pokémon agents are often bottlenecked by low-level
    perception/control and harness design.

- "Continual Harness", 2026.
  - arXiv: https://arxiv.org/abs/2605.09998
  - Repo relevance: current harness/memory/sub-agent optimization work for long-horizon agents;
    relevant to future offline route/subgoal discovery, not frame-level Mario control.

### Procedural Content and Cross-Game Mario

- Shyam Sudhakaran et al., "MarioGPT: Open-Ended Text2Level Generation through Large Language Models",
  NeurIPS 2023.
  - arXiv: https://arxiv.org/abs/2302.05981
  - OpenReview: https://openreview.net/forum?id=aa8KsqfTPa
  - GitHub: https://github.com/shyamsn97/mario-gpt
  - Repo relevance: plausible source for procedural SMB-like level diversity if we pursue true
    generalization beyond stock levels.

- Stable-Retro / Farama Foundation.
  - Docs: https://stable-retro.farama.org/
  - Python API: https://stable-retro.farama.org/python/
  - GitHub: https://github.com/Farama-Foundation/stable-retro
  - Repo relevance: current path for GBA/SNES/NES adapter-backed emulator environments.

- PyBoy.
  - Docs: https://docs.pyboy.dk/
  - GitHub: https://github.com/Baekalfen/PyBoy
  - Repo relevance: current path for Super Mario Land experiments; supports save/load state.

- mGBA / libretro core.
  - Libretro docs: https://docs.libretro.com/library/mgba/
  - mGBA FAQ: https://mgba.io/faq.html
  - Repo relevance: current GBA emulator core used through Stable-Retro for SMA4.

### SMB3 / SMA4 reverse-engineering (ground truth for the two-tier SMB3 agent)

- Southbird's NES SMB3 disassembly (reassembles byte-for-byte).
  - GitHub: https://github.com/captainsouthbird/smb3
  - Writeup: https://sonicepoch.com/sm3mix/disassembly.html
  - Repo relevance: the SMB3 analogue of the SMBDIS.ASM used for the SMB1 castle solves;
    authoritative for overworld structure, gating, item/inventory semantics.

- Karisa Advynia's SMA4 (GBA) disassembly.
  - GitHub: https://github.com/KarisaAdvynia/sma4-disasm
  - Repo relevance: cross-reference for GBA EWRAM/IWRAM addresses when reverse-engineering SMA4;
    locally used to confirm the adapter's `0x03003F24` field is a 32-bit player X coordinate, not
    an 8-bit progress cap.

- Data Crystal RAM maps.
  - NES SMB3: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._3/RAM_map
  - SMA4 (GBA): https://datacrystal.tcrf.net/wiki/Super_Mario_Advance_4:_Super_Mario_Bros._3/RAM_map
  - Repo relevance: seeded the confirmed SMA4 map-cursor/world addresses (see
    `notes/sessions/2026-06-26-smb3-overworld-foundation.md` and `scripts/probe_overworld.py`);
    seeded the SMA4 inventory-slot range and warp-whistle item id used by
    `scripts/probe_sma4_whistle.py`.

- Erick Guillen's SMA4 RAM table.
  - URL: https://erick.guillen.com.mx/sma4_ram.html
  - Repo relevance: secondary SMA4 RAM source; useful cross-check but addresses must be probed on
    the exact ROM because community tables can disagree.

- TASVideos SMB3/SMA4 resources and movies.
  - NES SMB3 game resources: https://tasvideos.org/GameResources/NES/SuperMarioBros3
  - NES SMB3 warps TAS movie #3922: https://tasvideos.org/3922M
  - SMA4 warps TAS movie #4246: https://tasvideos.org/4246M
  - SMA4 route notes: https://tasvideos.org/6809S
  - Repo relevance: confirms route-level mechanics such as P-speed management, LEFT+RIGHT
    acceleration in SMA4 TAS contexts, whistle routing, and flight/fortress constraints.

- JaffarPlus, reward-guided BFS / TAS-search tooling.
  - GitHub: https://github.com/SergioMartin86/jaffarPlus
  - Repo relevance: current savestate-search/TAS automation prior art worth studying as a baseline
    for parallel reward-guided emulator search, deduplication, and replay tooling.
    Its current engine documents many-core search, hash-based state deduplication,
    deterministic replay, and 15+ emulator cores.

- QuickerMGBA, a headless/re-recording-oriented mGBA fork used by JaffarPlus.
  - GitHub: https://github.com/SergioMartin86/quickerMGBA
  - Repo experiment: a bounded throughput spike against Stable-Retro/mGBA using
    identical state-clone and step workloads. Adopt only if end-to-end
    nodes/second improves enough to justify a new integration surface.

- Official mGBA scripting API.
  - Docs: https://mgba.io/docs/scripting.html
  - Repo relevance: exposes savestate buffers, CPU registers, and memory
    domains, but does not promise byte-canonical savestate serialization. Keep
    raw SHA-256 as a strong identity witness rather than an equivalence test;
    use RAM/wrapper checks and multiple fixed input suffixes only as explicitly
    finite behavioral falsifiers.

- SMB3 warp whistles / any% route (for the M4 resource-aware skip).
  - StrategyWiki: https://strategywiki.org/wiki/Super_Mario_Bros._3/Warp_Whistles
  - Repo relevance: the W1→W8 whistle route is the planned showcase of consumable foresight;
    avoid the 7-1 ACE (route-data, not search-tractable).

### Options / hierarchical RL (re-centered 2026-07-14)

- Richard S. Sutton, Doina Precup, Satinder Singh, "Between MDPs and semi-MDPs: A framework for
  temporal abstraction in reinforcement learning", Artificial Intelligence 1999.
  - PDF: https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf
  - Repo relevance: formal backbone for `mario/options.py` — initiation set, intra-option policy,
    termination; planning over options is SMDP planning. AcquireWhistle must be a real option
    \(\langle\mathcal{I},\pi,\beta\rangle\), not only a MetaState transform.

- Thomas G. Dietterich, "Hierarchical Reinforcement Learning with the MAXQ Value Function
  Decomposition", JAIR 2000.
  - Repo relevance: classical task decomposition; secondary to options here because our low-level
    \(\pi\) is search-filled, not learned MAXQ subtasks.

- Peter Dayan, Geoffrey E. Hinton, "Feudal Reinforcement Learning", NIPS 1992; Alexander Vezhnevets
  et al., "FeUdal Networks for Hierarchical Reinforcement Learning", ICML 2017.
  - PMLR: https://proceedings.mlr.press/v70/vezhnevets17a.html
  - Repo relevance: Manager/Worker learned hierarchies. Lower priority than search-defined options
    until option *discovery* (not execution) is the bottleneck.

- "Scalable Option Learning in High-Throughput Environments", 2025.
  - arXiv HTML: https://arxiv.org/html/2509.00338v2
  - Repo relevance: modern option-learning at scale; contrast — we need verified few options, not
    billions of samples of option discovery.

### State abstraction at option boundaries (July 25 literature cutoff; July 26–27 implementation)

- Balaraman Ravindran, Andrew G. Barto, "SMDP Homomorphisms: An Algebraic
  Approach to Abstraction in Semi-Markov Decision Processes", IJCAI 2003.
  - Official paper: https://www.ijcai.org/Proceedings/03/Papers/145.pdf
  - Repo experiment: enumerate every concrete boundary representative's
    repeat-conformant option signature—applicability, success/termination,
    successor refined block or symbolic state, retained duration/reported cost,
    effective tier, terminal invariants, and semantic interventions—and
    partition-refine from `MetaState`. Raw/full state digests, record IDs,
    producer/history, display hashes, and source file hashes remain
    identity/provenance witnesses only. Same symbolic label alone does not
    establish an SMDP homomorphism.

- Robert Givan, Thomas Dean, Matthew Greig, "Equivalence Notions and Model
  Minimization in Markov Decision Processes", *Artificial Intelligence* 147,
  2003.
  - Publisher: https://www.sciencedirect.com/science/article/pii/S0004370202003764
  - DOI: https://doi.org/10.1016/S0004-3702(02)00376-4
  - Repo experiment: iteratively compare option outcomes into current successor
    blocks rather than treating one-step or raw-state agreement as sufficient.
    This is the recursive backbone of the new fixed-point report.

- Pablo Samuel Castro, Doina Precup, "Using Bisimulation for Policy Transfer in
  MDPs", AAAI 2010.
  - AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/7751
  - PDF: https://ojs.aaai.org/index.php/AAAI/article/download/7751/7611
  - Repo experiment: execute every admissible option from both representatives
    of the observed whistle-acquisition-order alias. Reject merging when option
    cost/duration, success/termination, intervention tier, or successor refined
    class differs. This separates behaviorally material aliases from
    noncanonical serialization.

- Pablo Samuel Castro, Prakash Panangaden, Doina Precup, "Equivalence Relations
  in Fully and Partially Observable Markov Decision Processes", IJCAI 2009.
  - Official paper: https://www.ijcai.org/Proceedings/09/Papers/276.pdf
  - Repo experiment: distinguish finite trace/suffix evidence from recursive
    bisimulation. Fixed input suffixes are useful falsifiers; finite matching
    traces alone never authorize a global equivalence claim.

- Aijun Bai, Siddharth Srivastava, Stuart Russell, "Markovian State and Action
  Abstractions for MDPs via Hierarchical MCTS", IJCAI 2016.
  - Official paper: https://www.ijcai.org/Proceedings/16/Papers/430.pdf
  - Repo experiment: compare one-representative `MetaState`, `(MetaState,
    physical boundary identity)` multi-representative UCS, and `(MetaState,
    incremental option-history hash)` search on generated alias graphs and
    SMA4. Full POMCP is premature because each emulator branch exposes its
    concrete snapshot.

- Alper Ahmetoglu, Steven James, Cameron Allen, Sam Lobel, David Abel, George
  Konidaris, "Skill-Driven Neurosymbolic State Abstractions", NeurIPS 2025.
  - Official proceedings:
    https://papers.nips.cc/paper_files/paper/2025/hash/0fa694fb9f1e265117e8da75966820fe-Abstract-Conference.html
  - Repo experiment: construct the abstraction around the actual option set,
    including initiation, transition, duration/reward, and goal distinctions.
    Treat each retained local emulator record as a delta distribution first;
    do not place a learned encoder in authority over exact replay.

- Robert Paige, Robert E. Tarjan, "Three Partition Refinement Algorithms",
  *SIAM Journal on Computing* 16(6), 1987.
  - Publisher/DOI: https://epubs.siam.org/doi/abs/10.1137/0216062
  - Repo experiment: retain the current transparent fixed-point pass for the
    12-record graph; move to a worklist/incremental coarsest-partition algorithm
    only when measured graph size makes it necessary.

- Dana Angluin, "Learning Regular Sets from Queries and Counterexamples",
  *Information and Computation* 75(2), 1987.
  - Publisher: https://www.sciencedirect.com/science/article/pii/0890540187900526
  - Repo experiment: treat exact emulator restores as membership-query roots
    and option/input suffixes as counterexamples to candidate state merges.
    The local system is partial, costed, intervention-tiered, and SMDP-like, so
    this is an architectural analogy rather than a direct DFA reduction.

- Thorsten Wißmann, Stefan Milius, Lutz Schröder, "Explaining Behavioural
  Inequivalence Generically in Quasilinear Time", CONCUR 2021.
  - Open proceedings/DOI:
    https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CONCUR.2021.32
  - Repo experiment: attach a compact distinguishing option suffix to every
    refinement split. The current report already returns
    `use_whistle_again` and `select_world8_pipe` witnesses for the two
    acquisition-order alias pairs.

- Michael R. James, Satinder Singh, "Learning and Discovery of Predictive State
  Representations in Dynamical Systems with Reset", ICML 2004.
  - Official paper:
    https://icml.cc/Conferences/2004/proceedings/papers/117.pdf
  - Repo experiment: learn a compact basis of predictive option/input tests
    only after exact records and resettable counterexamples exist. Validate the
    basis on held-out suffixes before using it for live frontier dominance.

- Mark Leon Giraud, Bastian Engel, Lea Nasarek, Yannis Storrer, Philipp Takacs,
  Leon Philipp Wittemund, "L-SCALE: Locality-Sensitive Coverage for Automata
  LEarning", AST 2026.
  - Institutional record:
    https://publikationen.bibliothek.kit.edu/1000195438
  - DOI: https://doi.org/10.1145/3793654.3793755
  - Availability: published online 2026-07-20, inside the July 25 cutoff.
  - Repo experiment: borrow snapshot-backed active suffix testing and an
    inspectable learned automaton. Do not use TLSH/locality-sensitive similarity
    as authority to merge physical game states; threshold-sensitive approximate
    hashes may prioritize tests but a false merge can fabricate a route.

- Francesco Percassi, Alessandro Saetti, Enrico Scala, "Planning with Uncertain
  Action Models", AAAI 2026.
  - Official proceedings:
    https://ojs.aaai.org/index.php/AAAI/article/view/40954
  - DOI: https://doi.org/10.1609/aaai.v40i43.40954
  - Repo experiment: compare execute-to-observe search with a PUMA-like
    effect-cache planner, but cache an observed effect only within a refined
    physical class. PUMA assumes execution reveals a reusable action model;
    SMA4 has not earned reuse across hidden lineages.

#### Local Stable-Retro/mGBA serialization diagnosis

- Stable-Retro 1.0.1, `src/retro.cpp`, savestate allocation:
  https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/src/retro.cpp#L51-L55
- Python C API, `PyBytes_FromStringAndSize(NULL, size)` leaves contents
  uninitialized:
  https://docs.python.org/3/c-api/bytes.html#c.PyBytes_FromStringAndSize
- Stable-Retro's vendored mGBA `retro_serialize`:
  https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/cores/gba/src/platform/libretro/libretro.c#L614-L631
- Vendored mGBA GBA I/O serialization:
  https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/cores/gba/src/gba/io.c#L918-L945
- Local finding: five no-step samples after restoring
  `runs/sma4_cache/1-fortress_real_entry.pkl` produced three raw hashes; 256–643
  bytes differed, all below offset `0x800`, with zero differences at or above
  `0x800`. This supports a narrow diagnosis of noncanonical unused
  serialization slots. Equal bytes remain exact artifact identity; unequal
  bytes alone are not physical inequality.

### IL / distillation nuance (2026-07-14 re-read)

- Foster, Block, Misra, "Is Behavior Cloning All You Need?" (NeurIPS 2024) — see above.
  - **Re-read caveat (2026-07-14):** main result *rehabilitates* offline LogLoss BC (horizon-
    independent under controlled payoff range). V6's "IL ceiling" is a local empirical fact under
    absorbing deaths + thin data, not a theorem that offline IL is impossible.

- Song, Rohatgi, Singh, Bagnell, "To Distill or Decide?" (NeurIPS 2025) — see above.
  - **Re-read caveat (2026-07-14):** distillation is competitive under *deterministic* latent
    dynamics (our setting), but the **optimal** latent expert is not always the best teacher —
    smoother experts can distill better. Untested locally; relevant if a reactive prior returns.

### 2026-07-14 planning note

- Bottom-up experiment ladder for SMA4/SMB3 on M2:
  `notes/sessions/2026-07-14-bottom-up-research-plan.md`.

## How These References Map to Current Repo Decisions

- `beam_search`, `coverage_search`, and adapter-backed search are the core deliverable.
  - Backed by Mario AI competition history, Go-Explore, ExIt, AlphaZero, and our local artifacts.

- `policy_prior` and value guidance should stay optional accelerators.
  - Backed by AlphaZero/ExIt/Gumbel AlphaZero and the local V6 result: policy-guided beam reduced
    1-1 nodes while preserving solve correctness. The replay-backed comparison is
    `notes/artifacts/2026-07-25-policy-guided-1-1.json`. Not yet measured on SMA4.

- The active product thesis is the **Option-SMDP whistle benchmark** (SMA4), not a flat generalist.
  - Backed by Sutton options, the local planner-class contrast artifacts, and the 2026-07-14 plan.
  - Honesty requirement: replace Tier-1 hand-grants with Tier-2 AcquireWhistle before claiming the
    meta-intelligence gap is fully measured.

- The generalist controller track is deprioritized.
  - Backed by Procgen/CoinRun, V4-V6 local failures, and the mismatch between robotic manipulation
    scaling laws and long-horizon platformer levels.

- Reverse-curriculum RL is a plausible standalone-specialist fallback, if needed.
  - Motivated by Go-Explore, RFCL, RLPD, and BBF, but requires a reproducible positive local
    baseline before architectural expansion.

- For SML/SMA4/SMW/SMB3-family expansion, do not start with a generalist policy.
  - First build per-game adapters, RAM/terminal semantics, reliable replay verification, and
    game-specific terminal-object scoring. Only then consider learned priors.
  - Local SMA2 (SMW) ROM enables a later SNES-family adapter parity experiment (plan Phase E).
