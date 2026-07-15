# 2026-07-15 — Next steps: theory → Mario AI experiments

> Connects `notes/theory/{first-principles,bellman-1957,gita-and-first-principles}.md`
> back to the live SMA4 Option-SMDP track and the SMB1 backlog.
>
> **Operating rule (from the adversarial + Gita pass):** do the prescribed
> ROM-complete work; renounce proxy fruit (bench green, poetic isomorphisms,
> “meta-intelligence”); do not call an unfinished honesty ladder a theorem.

---

## 0. Where we are (one screen)

| Layer | Status | Honesty debt |
|---|---|---|
| SMB1 any% / hard castles | DONE by search | 6-2 open; 6-3 false-positive |
| SMA4 1-1 / 1-2 ClearLevel | Tier-0 replay-verified | — |
| `AcquireWhistle_1_3` | Tier-2 replay-verified | P-Wing entry snapshot injected; MAP_CURSOR sync assist |
| Whistle *spend* + W8 pipe | ROM-backed opaque | Regrant **removed**; second spend needs real inventory whistle |
| Warpless / Bowser endpoints | Symbolic costs | Inflates × ratios |
| Theory spine | Written + adversarially narrowed | Vocabulary inflation called out |

**Thesis we are allowed to defend today** (narrow form from `gita-and-first-principles.md` §5):

> Under a deterministic resettable emulator contract, model-based search is the
> reliable solver; nets accelerate; verified temporally extended executors +
> execute-to-observe meta-search handle consumable/gated structure. Myopic
> progress-greedy can miss effect-opaque skips.

**Thesis we are *not* allowed to claim yet:** Option-SMDP completeness, Bellman
optimality of beam, Newton identity of current code, “meta-intelligence.”

---

## 1. Theory → experiment map

| Theory demand | Concrete Mario AI work | Why now |
|---|---|---|
| Bellman: imbed + policy structure; curse of dimensionality | Keep search; don’t revive flat generalist | Already settled V5/V6 |
| SPS options: real \(\langle\mathcal{I},\pi,\beta\rangle\), measured \(p_o,r_o\) | Climb knowledge ladder: 2nd AcquireWhistle; drop re-grant | Closes the biggest honesty wound |
| Ng: potential, not proxy fruit | SMA4 Φ / card terminal that doesn’t farm `x_pos` cap | Stops finisher-as-fundamental myth |
| Opaque effects must be executed | Keep execute-to-observe UC; don’t give greedy a cheat sheet and call the gap “wisdom” | Planner-class claim stays honest |
| Bertsekas/ExIt: prior guides, doesn’t replace | Only after wall-clock stalls: SMA4 policy prior | Phase D — not critical path |
| Go-Explore: return-then-explore | Fortress whistle / deceptive inventory cells | Second acquire is this problem |
| Falsifiability (adversarial F1–F6) | Each step below has a kill condition | Prevents soft claims |
| Ethos: prescribed work > proxy fruit | Prefer ROM-complete options over new narrative docs | This plan’s ranking |

---

## 2. Critical path (do in order)

### Step 1 — Second W1 `AcquireWhistle` (IN PROGRESS)

**What.** Fortress flight / hidden-door whistle as a Tier-2 (or honest Tier-3)
replay-verified option; wire into `build_sma4_whistle_rom_library`.

**Partial done (2026-07-15).** `whistle_regranted_in_warp_zone` **removed**.
Hand-grant path still closes W8 at 7.7× without regrant. Fortress entry itself
is blocked: after 1-1/1-2 the walkable graph never reaches ~`(96,80)`; see
`notes/sessions/2026-07-15-fortress-whistle-blocker.md`.

**Theory link.** Completes the option *library* so the SMDP action set contains
both acquires; opacity is real, not hand-granted mid-plan.

**Done when.**

- `data/solutions/sma4/acquire_whistle_<fortress>.json` with `solved=true`,
  `replay_verified=true`, inventory `0x0C` without RAM poke at acquire time.
- Executor method + test (mirror `test_acquire_whistle_solution.py`).
- `--no-hand-grant` ROM bench closes W8 without inventory poke / regrant.

**Falsify / defer.** If only works with Tier-3+ RAM surgery beyond documented
P-Wing/flight, record min knowledge tier honestly and keep a one-whistle partial
thesis — do not fake Tier-2. (Current defer: fortress **path unlock**, not flight.)

**Likely sub-tasks.**

1. ~~Confirm correct fortress / side-node entry~~ — fortress ~`(96,80)`;
   `1-fortress_entry.pkl` is the wrong room (sky/athletic).
2. Find map path-bit / unlock that wiki’s “after 1-2” implies (our clears do not
   open `LEFT` from `(128,64)`).
3. Power state: Raccoon / P-Wing initiation as its own small option if needed.
4. Flight + hidden door route → chest / whistle; exit to map.

---

### Step 2 — Honesty cleanup of the remaining injections

After both acquires exist, burn down the leftover cheats **one at a time**:

| Debt | Fix | Done when |
|---|---|---|
| P-Wing 1-3 entry snapshot | Real `AcquirePWing` / overworld item option, or start 1-3 from natural power state | Acquire path needs no `pwing_1_3_entry.pkl` injection |
| MAP_CURSOR sync after acquire | Fix exit settle so L-inventory works without poke | Executor has no `_sync_map_cursor` assist (or assist labeled Tier-3 and gated) |
| Symbolic warpless / Bowser | ROM-backed ClearLevel stubs or measured frame costs from real clears | Bench report `cost_model=rom` for those hops, or scoped claim “skip-to-W8 only” |

**Theory link.** Until this is done, × ratios (5.7× / 7.75×) are **mixed-reality** —
useful as planner-class demos, not as finished any%-via-whistle theorems.

**Falsify.** If Bowser/warpless ROM options are intractable this quarter, **narrow
the published claim** to “W1→W8 skip under measured whistle options” rather than
fake full-game costs.

---

### Step 3 — Rebench + claim hygiene (Phase C)

**What.** Single clean report:

- Library: both AcquireWhistle options, both spends, W8 pipe, no regrant, no
  first hand-grant.
- Baselines: greedy vs uniform-cost (and optionally greedy-with-oracle-effects as
  a *control*, not the hero).
- Labels in `report.json`: `injected_facts=[]` or explicit remaining list.

**Done when.** Artifact path under `runs/` + one paragraph in session note stating
exactly what is still symbolic.

**Falsify (good science).** If greedy-with-cached-effect-model matches UC cost,
relabel the gap as “model incompleteness,” not meta-intelligence (adversarial F2).

---

### Step 4 — SMA4 progress coordinate Φ (Ng layer)

**What.** Replace raw capped `x_pos` ranking with an area/mode-aware potential +
honest terminal (`endwalk` / card), so `goal_suffix_search` is a backup, not the
theory of finishing.

**Done when.** At least one level (1-1 or 1-2) re-solved with Φ-only ranking to
card-capable states without a hardcoded return template — or a written negative
that card timing still needs a finisher (honest).

**Why after Step 1–3.** Whistle ladder is the thesis-critical path; Φ is quality
of the level solver, not the Option-SMDP claim.

---

### Step 5 — Optional accelerators (only if wall-clock hurts)

| Work | When | Do not confuse with |
|---|---|---|
| SMA4 policy prior for beam/coverage | Search wall-clock blocks B3/B4 | “Net replaces search” |
| True Bertsekas rollout (evaluate base heuristic from children) vs top-k prune | If we want Newton vocabulary to be earned | Calling current `policy_topk` a Newton step |
| Explicit `Option(I,π,β)` fields + measured multi-time costs | After library is ROM-complete | Claiming SPS completeness early |

---

## 3. Explicitly later / deprioritized

| Item | Why parked |
|---|---|
| Flat generalist / another DAgger push | V6 + DAgger bounds; wrong main line given \(f\) |
| World models / MuZero | \(f\) already free |
| In-loop LLM/VLM control | Latency + VideoGameBench; offline subgoals only |
| SMB1 6-2 / 6-3 quarantine fix | Orthogonal to whistle thesis; pick up only if pursuing 32/32 |
| SNES SMA2 adapter (Phase E) | After C1 green (honest whistle bench) |
| Foster/Song smoothed-expert science track | Secondary; falsifier F1 for “IL can replace search,” not the product path |

---

## 4. Falsifiers to keep on the whiteboard

From the adversarial pass — kill or narrow claims if observed:

1. **F1** — Smoothed ExIt/BC standalone clears a suite w/o search-rescue → reopen IL as peer, not servant.
2. **F2** — Greedy with perfect option-effect table matches UC → gap was missing model rows.
3. **F3** — Growing the option library slows meta-search without better solves → trim macros (Hauskrecht).
4. **F4** — Policy/value guidance increases nodes or fails where unguided coverage works → prior is situational.
5. **F5** — SMA4 replay_verified collapses under re-entry → fix adapter contract before more options.
6. **F6** — At PCG scale search-per-level loses Pareto → scope thesis to stock ROMs.

---

## 5. Recommended cadence (next sessions)

| Session focus | Deliverable |
|---|---|
| **Now** | Fortress / second AcquireWhistle probe → solution JSON → executor → drop regrant |
| **Next** | Rebench `--no-hand-grant --no-regrant`; rewrite claim paragraph |
| **Then** | P-Wing initiation option + kill 1-3 entry injection |
| **Then** | Scope or ROM-back warpless/Bowser costs |
| **Whenever blocked on ROM** | SMA4 Φ experiment (parallelizable, doesn’t need whistle) |

One concrete next action (matches `CLAUDE.md`):

> **Unlock or Tier-3-label a real W1 Fortress entry, then build replay-verified
> `AcquireWhistle_fortress` so `--no-hand-grant` can close the two-whistle W8
> skip without inventory poke.** (Regrant already removed.)

---

## 6. How to talk about results (claim hygiene)

**Say:** “Execute-to-observe option search finds the whistle skip; myopic greedy
does not, under library L with honesty set H.”

**Don’t say:** “We measured meta-intelligence” or “Bellman optimality” or
“Option-SMDP solved” until H is empty (or H is listed in the abstract).

**Proxy fruit to ignore when ranking progress:** new theory prose, val-acc,
node-cut on easy 1-1, symbolic × without ROM costs.

**Real fruit:** replay_verified options, removed injection flags, green rebench
with an honest `injected_facts` list.
