# Research Bibliography and Consolidated Field Notes

Last verified: 2026-07-15.

This file consolidates the literature and tool references that informed the V4-V6 direction, the
generalist-policy reassessment, the "net serves search" pivot, the small-LLM/VLM discussion, and
the cross-game Mario adapter work.

**First-principles synthesis (2026-07-15):** `notes/theory/first-principles.md`
(corpus fetch: `notes/theory/README.md`).

## High-Level Conclusions

1. Search is the solver when an exact, resettable emulator is available.
   - The Mario AI competition history and AlphaZero/ExIt lineage both support this: use planning
     over the true dynamics, then optionally distill or use learned priors to guide the search.
   - In this repo, the working version is beam/coverage search over emulator snapshots, with learned
     policy/value models treated as accelerators rather than standalone replacements.

2. A zero-shot generalist Mario controller from the fixed stock level set is not a realistic near-term
   expectation.
   - The ICLR 2025 data-scaling paper that motivated "dozens of environments might be enough" is
     about robotic manipulation, not long-horizon platformer level generalization.
   - Procgen/CoinRun is the closer analogue: hundreds of levels can still overfit, and robust
     train-to-test transfer usually needs far more procedural diversity than the 32 stock SMB levels.

3. Offline BC/DAgger is useful, but the repo's experiments show the standalone policy track hits a
   real closed-loop-control wall.
   - V4/V5/V6 results: structured/entity policies improve validation accuracy and can run/jump, but
     still fail closed-loop completion as generalists; DAgger helped some train levels and did not
     produce held-out transfer.
   - Practical direction: keep per-level cached search solutions/rescue; use learned priors inside
     search where they reduce nodes without losing completeness.

4. Reverse-curriculum RL remains the serious option if we ever need a reactive standalone specialist.
   - RFCL/Go-Explore/RLPD/BBF support the recipe: start near successful states, use saved demos and
     online rollouts, and use modern stability/plasticity fixes.
   - The local `scripts/rc_rl.py` experiment was directionally positive near the end of 1-1 but not
     production-grade.

5. Small LLMs/VLMs are not recommended for real-time low-level Mario control in this project.
   - Pretrained language can help some sequential decision problems, but the gains appear to come
     from sequence bias/initialization rather than natural-language knowledge.
   - VideoGameBench shows frontier VLMs still fail badly at direct real-time game completion, and
     local small models are too slow or too weak for frame-level control.
   - Better use: offline subgoal proposal, code/RE assistance, or level-generation tooling.

6. Cross-game Mario should stay adapter-first.
   - Stable-Retro/mGBA and PyBoy give usable emulator APIs and savestates.
   - Each game still needs per-game boot, RAM, action, terminal, and progress semantics before any
     learned model or generic search heuristic is meaningful.

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
  - Repo relevance: supports potential-based/global-progress shaping logic.

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
  - Repo relevance: recent BC theory around discretized/quantized action spaces; relevant because this
    project discretizes controller actions.

- Yuda Song et al., "To Distill or Decide? Understanding the Algorithmic Trade-off in Partially
  Observable Reinforcement Learning", 2025.
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
    1-1 nodes while preserving solve correctness. Not yet measured on SMA4.

- The active product thesis is the **Option-SMDP whistle benchmark** (SMA4), not a flat generalist.
  - Backed by Sutton options, the local planner-class contrast artifacts, and the 2026-07-14 plan.
  - Honesty requirement: replace Tier-1 hand-grants with Tier-2 AcquireWhistle before claiming the
    meta-intelligence gap is fully measured.

- The generalist controller track is deprioritized.
  - Backed by Procgen/CoinRun, V4-V6 local failures, and the mismatch between robotic manipulation
    scaling laws and long-horizon platformer levels.

- Reverse-curriculum RL is the right standalone-specialist fallback, if needed.
  - Backed by Go-Explore, RFCL, RLPD, and BBF, but requires a dedicated RL-engineering pass.

- For SML/SMA4/SMW/SMB3-family expansion, do not start with a generalist policy.
  - First build per-game adapters, RAM/terminal semantics, reliable replay verification, and
    game-specific terminal-object scoring. Only then consider learned priors.
  - Local SMA2 (SMW) ROM enables a later SNES-family adapter parity experiment (plan Phase E).
