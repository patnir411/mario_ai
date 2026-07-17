# Session Log

Newest first. Detailed notes live under `notes/sessions/`.

## 2026-07-17 — Drop fortress door-entry snapshot

Detailed note: `notes/sessions/2026-07-15-acquire-whistle-fortress.md`

Summary:
- Coverage prefix from leaf fortress spawn → door `x≈1705`; `LEFT+B×50` align + chest script.
- Solution entry is now `1-fortress_pwing_leaf_entry.pkl` (~1794f).
- Dropped `fortress_door_entry_snapshot`. ROM rebench **5.18×**
  (`runs/20260717-sma4-whistle-rom-bench-nodoor-snap-{open,blocked}/`).

## 2026-07-15 — Burn fortress inventory rehost (B-settle exit)

Detailed note: `notes/sessions/2026-07-15-acquire-whistle-fortress.md`

Summary:
- Root cause: treasure-room exit locked L-menu; rehost was a RAM workaround.
- Fix: truncate at chest → `UP` to map → idle → hold `B` (native unlock).
- Dropped `fortress_inventory_rehosted_to_pre_door_map`.
- ROM rebench: **5.38×** open (`…-norehost-open/`); blocked greedy stuck / UC skips.

## 2026-07-15 — Two-acquire ROM `--no-hand-grant` W8 skip (5.23×)

Detailed note: `notes/sessions/2026-07-15-acquire-whistle-fortress.md`

Summary:
- `AcquireWhistle_fortress` replay-verified; fortress exit locks L-menu → inventory
  rehosted onto pre-door map (injected fact recorded).
- ROM open: greedy 69000 warpless / UC 13192 skip = **5.23×**
  (`runs/20260715-sma4-whistle-rom-bench-two-acquire-open/`).
- ROM blocked: greedy stuck / UC still skips
  (`runs/20260715-sma4-whistle-rom-bench-two-acquire-blocked/`).
- No hand-grant or warp-zone regrant. Next: burn fortress rehost / door-entry injections.

## 2026-07-15 — Fortress path blocker + whistle regrant removed

Detailed note: `notes/sessions/2026-07-15-fortress-whistle-blocker.md`

Summary:
- Fortress icon ~`(96,80)` but not in the post-1-2 walk graph (natural + cached);
  cursor poke is cosmetic (`0x03003788` stale); `LEFT` from `(128,64)` hard-blocked.
- Removed `whistle_regranted_in_warp_zone`; second spend needs a real remaining `0x0C`.
- Hand-grant ROM rebench: **7.7×** skip, no regrant fact
  (`runs/20260715-sma4-whistle-rom-bench-noregrant-open/`).
- `--no-hand-grant --warpless-blocked`: all planners `found=False` (honest one-whistle).
- Next: unlock/label fortress entry → `AcquireWhistle_fortress`.

## 2026-07-15 — Next steps plan (theory → Mario AI)

Detailed note: `notes/sessions/2026-07-15-next-steps-after-theory.md`

Summary: Critical path = 2nd AcquireWhistle → burn injections → honest rebench →
SMA4 Φ; accelerators/IL/world-models parked. Claim hygiene: no “meta-intelligence”
until `injected_facts` empty. Next action unchanged (fortress whistle + drop regrant).

## 2026-07-15 — Bhagavad-Gita (Arnold) + adversarial synthesis

Detailed note: `notes/theory/gita-and-first-principles.md`

Summary:
- Read Gutenberg Arnold *Song Celestial* end-to-end (18 chapters); extracted full text.
- Universal truths: fruit vs act, false renunciation, desire as anti-epistemic,
  equanimity as trainable, own-task fidelity, proxy-piety — without forcing theology
  into algorithms.
- Devil’s advocate: almost all Gita↔DP isomorphisms **Kill**ed; Keep Bellman
  policy-space approx. (credit Bellman); Transform = experimenter proxy-hygiene.
- Same pass wounded our own thesis (Newton/Option completeness/“meta-intelligence”
  inflation); wrote narrower defensible thesis. Experimental next action unchanged.

## 2026-07-15 — Bellman 1957 *Dynamic Programming* (front-to-back)

Detailed digest: `notes/theory/bellman-1957.md` (folded into `first-principles.md` §1.5)

Summary:
- Read user-supplied `notes/theory/pdfs/dynamic programming.pdf` (365 pp., OCR)
  end-to-end via chapter extracts.
- Principle of Optimality + \(f_N(p)=\max_q[g+f_{N-1}(T_q(p))]\) is the root of
  emulator search; approximation in policy space is the ancestor of “net serves
  search”; curse of dimensionality justifies beam/coverage over tabular VI.
- Later chapters (inventory, bottlenecks, CoV, games, Markovian DP) deepen the
  same spine; concrete models are mostly analogies for mario_ai.

## 2026-07-15 — First-principles theory pass (textbooks + papers)

Detailed note: `notes/sessions/2026-07-15-first-principles-theory.md`
Synthesis: `notes/theory/first-principles.md` · corpus: `notes/theory/README.md`

Summary:
- Downloaded/read legal free textbooks (Sutton & Barto; Bertsekas RL course,
  Lessons from AlphaZero, Rollout; Puterman & Chan draft) and core papers
  (options, shaping, DAgger, ExIt, AlphaZero, Go-Explore).
- Layer stack 0→8 maps deterministic \(f\) → Bellman → planning → \(\Phi\) →
  options/SMDP → Newton/approx DP → ExIt → IL bounds → Go-Explore onto this repo.
- Verdict unchanged and now textbook-grounded: search solves; net accelerates;
  options are the SMB3 object; standalone IL is secondary under death cliffs.
- PDFs gitignored; re-fetch via `notes/theory/fetch_corpus.py`.

## 2026-07-15 — Tier-2 AcquireWhistle_1_3 + rebench without hand-grant

Detailed note: `notes/sessions/2026-07-15-acquire-whistle-1-3.md`

Summary:
- Replay-verified `data/solutions/sma4/acquire_whistle_1_3.json` (2533 frames): P-Wing 1-3
  entry → white-block duck (`0x03003D06`) → Toad house → chest opens with **B** → map exit
  with inventory `0x0C`.
- A0 bench already present: `bench/sma4_{step_rate,snapshot_cost}.json` (~934 fps / ~230 nodes/s).
- ROM rebench `--no-hand-grant`: search discovers skip via `acquire_whistle_1_3` (5 hops / 12099);
  greedy stays warpless (69000) → **5.7×**. Warpless-blocked: greedy stuck, search still skips.
- Honesty: first whistle is Tier-2 real; P-Wing entry snapshot + second-whistle regrant + cursor
  sync after acquire exit remain injected; warpless/Bowser symbolic.
- Tests: `test_acquire_whistle_solution.py` + meta-planner acquire path; pytest green.

## 2026-07-14 — Bottom-up research plan (SMA4/SMB3 Option-SMDP)

Detailed note: `notes/sessions/2026-07-14-bottom-up-research-plan.md`

Summary:
- Ground-up audit from deterministic dynamics → Ng potential shaping → Sutton options/SMDPs
  → ExIt/AlphaZero → Foster/Song IL theory → Go-Explore, against V5/V6 + whistle artifacts.
- Key challenges: whistle planner gap still incomplete (Tier-1 grant, symbolic warpless/Bowser);
  IL "ceiling" overstated vs Foster LogLossBC; no SMA4 M2 throughput bench; Φ/card-cap unfinished.
- Experiment ladder Phases A–F; critical path **A0 → A2 → B2 AcquireWhistle → B4 rebench**.
- Deprioritized: flat generalist, in-loop LLM/VLM, world models, NES 6-2 until A–C done.
- SNES-family path: local SMA2 (SMW) ROM exists for later adapter parity (Phase E).
- Bibliography updated; CLAUDE next action retargeted to A0+B2.

## 2026-06-29 — Option-MDP scaffold + SMA4 whistle recon + planner contrast + ROM substitution

Detailed note: `notes/sessions/2026-06-29-option-mdp-whistle-benchmark.md`

Phase 4 (Codex pass):
- Added the ROM-backed whistle substitution: `SMA4WhistleExecutor`, a real
  `build_sma4_whistle_rom_library`, and `scripts/bench_sma4_whistle_rom.py`.
- Fixed SMA4 map-mode classification for raw world `7` (World 8) and raw world `8` (warp-zone),
  whose cursor Y positions are off the early World-1 0x20 grid.
- Default ROM benchmark artifact: `runs/20260629-202831-sma4-whistle-rom-bench/report.json`.
  Greedy still takes the symbolic warpless route (8 hops / 69000 frames / skip=False); BFS and
  uniform-cost execute real opaque options and observe `use_whistle` -> raw `8`, `use_whistle_again`
  -> raw `8` at the 5-8 screen, `select_world8_pipe` -> raw `7` World 8. Uniform-cost plan:
  5 hops / 8900 frames / skip=True, a 7.75x reduction vs greedy.
- Blocked-route artifact: `runs/20260629-202846-sma4-whistle-rom-bench/report.json`.
  Greedy is stuck; resettable search still reaches World 8 through the ROM-backed whistle route.
- Verification: `./venv/bin/python -m pytest --disable-warnings` -> 80 passed / 10 skipped;
  ROM-gated `tests/test_sma4_overworld.py` -> 5 passed.

Phase 3 (started, Claude pass):
- Added `mario/meta_planner.py` (myopic `greedy_plan`, cost-optimal `search_options_uniform_cost`,
  symbolic `build_whistle_benchmark_library`, `run_whistle_benchmark`), `scripts/bench_whistle_planning.py`,
  `tests/test_meta_planner.py` (6).
- Planner-class contrast on the SAME effect-opaque whistle library: greedy beats the game the long
  warpless way but NEVER discovers the skip (8 hops / 69000 frames, skip=False); resettable search
  discovers the opaque `use_whistle -> World 8` payoff and returns a 10.95× cheaper plan
  (3 hops / 6300 frames). `--warpless-blocked` → greedy stuck (found=False), search still skips.
  Artifact: `runs/20260629-175809-whistle-planning-bench/report.json`. Costs symbolic; endpoints +
  whistle spend ROM-verified separately. `pytest --disable-warnings` → 78 passed / 10 skipped.

Summary:
- Added `mario/options.py`: `MetaState`, `Option`, `OptionLibrary`, `KnowledgeTier`,
  resettable opaque-effect `search_options`, and an `SMA4SolutionExecutor` bridge to cached
  replay. Existing `data/solutions/sma4/{1-1,1-2}.json` wrap as Tier-0 `ClearLevel` options.
- Added non-ROM tests proving tier gating and an opaque-effect shortcut: the planner executes a
  whistle-like option, observes the post-state, and then exploits the discovered World-8 route.
- Added `scripts/probe_sma4_whistle.py`. Probe artifact:
  `runs/20260629-130158-sma4-whistle-probe/report.json`.
- Whistle recon verdict: two-whistle spend path is executable from savestate with hand-granted
  inventory; one whistle reaches the special warp-zone map, not World 8. `L` opens the item menu,
  `A` uses the selected item, then after the second whistle `RIGHT` + `A` selects the World-8 pipe.
  Raw world byte `8` is warp-zone; raw `7` is World 8.
- Verification: `./venv/bin/python -m pytest --disable-warnings` -> 72 passed / 10 skipped;
  ROM-gated `tests/test_sma4_overworld.py` -> 5 passed.

## 2026-06-27 — SMA4 cached World-1 sweep: 1-3 reached, side node classified

Detailed note: `notes/sessions/2026-06-27-sma4-world1-sweep.md`

Summary:
- Added a cached-solution fast-forward path for SMA4 world sweeps. It restores cached entry
  snapshots, replays `data/solutions/sma4/{1-1,1-2}.json`, settles to the post-clear map, then
  starts discovery/search from there.
- Fast-forward worked for 1-1 and 1-2; entry snapshots now include `runs/sma4_cache/1-1_entry.pkl`,
  `runs/sma4_cache/1-3_entry.pkl`, and `runs/sma4_cache/1-fortress_entry.pkl`.
- 1-3 normal route: beam reaches the roulette-card area (`x_max=2546`, `goalcard=2`, small Mario)
  and misses the card. Artifact/contact: `runs/20260627-113530-sma4_snapshot_1_3_pspeed/`.
  Wall = terminal card timing, not flight and not the whistle route.
- Whistle route assessment: not pursued; the known duck-through-white-block trick is route-data
  behavior and not reachable under the current forward-progress objective without an explicit
  subgoal/option.
- `(160,64)` side node: bounded search stalls/falls around `x_max=329`; contact sheet shows an
  early sky/athletic platform-gap wall, not a fortress/castle screen. Artifact:
  `runs/20260627-115920-sma4_snapshot_1_fortress_pspeed/`.
- Verification: `./venv/bin/python -m pytest --disable-warnings` -> 69 passed / 10 skipped;
  ROM-gated `tests/test_sma4_overworld.py` -> 5 passed; `git diff --check` clean.

## 2026-06-27 — SMA4 1-2 terminal clear + card-return finisher

Detailed note: `notes/sessions/2026-06-27-sma4-1-2-finish.md`

Summary:
- Closed SMA4 1-2 from `runs/sma4_cache/1-2_entry.pkl` using the focused P-speed prefix plus a
  generalized card-return finisher; no Raccoon flight required.
- New solution artifact: `data/solutions/sma4/1-2.json` (`solved=true`, `replay_verified=true`,
  `endwalk=255`). Visual proof: `runs/20260627-104611-sma4_snapshot_1_2_pspeed_finish/solved_contact.png`.
- Finisher change: `goal_suffix_search` now records exact frame-level prefixes, keeps post-cap
  settle snapshots, and tries a cheap left-return/jump template for capped SMB3 card endings.
- Progress-coordinate audit: Karisa SMA4 disassembly confirms `0x03003F24` is a 32-bit player X
  coordinate; RAM diffing found provisional camera-like candidate `0x03003894`, but it is not
  validated enough to use as progress.
- Verification: `./venv/bin/python -m pytest --disable-warnings` -> 67 passed / 10 skipped;
  ROM-gated `tests/test_sma4_overworld.py` -> 5 passed; `git diff --check` clean.

## 2026-06-26 — SMA4 physics-state search: 1-2 wall artifact-backed and broken

Detailed note: `notes/sessions/2026-06-26-sma4-physics-search.md`

Summary:
- Added SMA4 physics telemetry (`pspeed`, speed, powerup, subpixels), richer opt-in action sets,
  frame-macro support, physics-aware adapter coverage cells, JSONL traces, and partial contact
  sheets for failed SMA4 attempts.
- New artifacts pin the 1-2 wall: surface baseline from `runs/sma4_cache/1-2_entry.pkl` fails at
  `x_max=629` with trace/contact sheet in `runs/20260626-195202-sma4_snapshot_1_2_surface/`.
- The focused P-speed/LEFT+RIGHT action set breaks the wall and reaches `x_max=2803`
  (`runs/20260626-201118-sma4_snapshot_1_2_pspeed/`), so 1-2's tall pipe is not a Raccoon-flight
  requirement. It still does not terminal-clear; the remaining blocker is the end-card/finish
  route.
- RAM probe artifact: `runs/20260626-195128-sma4_physics_probe/physics_probe.json`. `0x03002C52`
  is not yet validated as a cleared-level bitmap; observed values differ across 1-1 post-clear and
  cached 1-2 entry.

## 2026-06-26 — SMB3 (SMA4) two-tier foundation: overworld instrumentation + generalized solver

Detailed note: `notes/sessions/2026-06-26-smb3-overworld-foundation.md`

Summary:
- Started the SMB3 end-to-end direction: a two-tier emulator-backed search agent (overworld
  planner + per-level search as an option). Platform: SMA4 / Stable-Retro / mGBA.
- Reverse-engineered the SMA4 overworld RAM (cursor X `0x03003DE4`, cursor Y `0x03003DE0` on a
  0x20 grid, world `0x03002A69` 0-indexed, progress `0x03002C52`); mode detection is behavioral
  (level timer vs map cursor). Reproducible via `scripts/probe_overworld.py`.
- Added `SMA4OverworldAdapter` (shared-core view), `boot_to_level`, `goal_suffix_search`
  (level-agnostic finisher replacing the 1-1 card hack), and adapter-backed
  `replay_adapter`/`make_contact_sheet_adapter` + `replay_video_adapter.py`.
- Tests: `tests/test_sma4_overworld.py` (5, ROM-gated) green; new non-ROM finisher/viz tests green.
- M2 meta-loop PROVEN (`mario/overworld_search.py`, run `runs/20260626-180705-sma4_world/`):
  1-1 solved fresh from overworld entry → map advanced → 1-2 discovered + enterable (gating works).
  Plain beam stalls 1-2 at x=629 → next: port `coverage_search` to adapters for hard levels.
- Fixed: cached frame-precise paths desync on re-entry (solve fresh per node); `enter_level`
  settles past the ~140-frame level-intro lockout. `update_status.py` now game_id-keyed.

## 2026-06-26 — Cross-game adapter push: SML plus SMA4/SMB3 1-1

Detailed note: `notes/sessions/2026-06-26-cross-game-sma4.md`
Research consolidation: `notes/research-bibliography.md`

Summary:
- Added a cross-game adapter path for non-NES Mario experiments.
- Confirmed ROM handling convention: `roms/` is ignored and local only.
- Added/used PyBoy for Super Mario Land and Stable-Retro/mGBA for GBA Mario.
- Created a custom Stable-Retro integration for Super Mario Advance 4 / Super Mario Bros. 3.
- Solved SMA4 World 1-1 from reset to post-level transition.
- Important finding: SMA4 1-1 world `x_pos` caps at the end boundary, so generic x-progress beam search runs under the roulette card. A local frame-level card finisher is currently needed.
- Current SMA4 artifact: `data/solutions/sma4/1-1.json` with `solved=true`, `replay_verified=true`, `chunk_frames=1`, path length 1130.
- Visual artifact: `runs/sma4_1_1_playthrough/play.mp4`; contact sheet: `runs/sma4_1_1_playthrough/contact_sheet.png`.

Verification:
- `./venv/bin/python -m pytest` -> 55 passed, 5 skipped.
- Saved SMA4 replay verified the card hit and post-clear transition after 489 frames.

Next direction:
- Turn the SMA4 1-1 suffix workaround into a more general end-of-level objective using screen-space/object-state scoring, then try the next GBA expansion target.
