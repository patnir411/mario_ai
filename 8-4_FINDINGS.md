# Mario AI — any% Playthrough & the 8-4 Wall (findings)

_Generated 2026-06-04. Companion to `CLAUDE.md` (status) and `DESIGN.md` (architecture)._

## TL;DR
A from-scratch search→distill→coverage system plays an **any% Super Mario Bros playthrough of 7 of 8 levels, every clip reaching the flagpole** — `runs/playthrough_anypct/full_run.mp4`. The eighth level, **8-4 (the final Bowser castle)**, is the remaining blocker. This document records the full diagnosis.

## UPDATE (Jun-4, 2nd-pass deep dive — corrects earlier wording)
Two independent fresh investigations (an 8-agent research swarm + an independent Codex CLI run reading the disassembly/env) overturned the earlier "proven uncrossable" claim — it was an **instrumentation error**, twice over:
1. **Area-local x (FIXED):** `x_pos = ram[$006D]*256 + ram[$0086]` resets ~40 on a transition, so the old reward scored the correct pipe as a −1000px catastrophe. Fixed with a level-global coordinate Φ (`global_progress` + `state_score(progress=)`), regression-clean.
2. **Wrong success byte (THE big one):** sub-area/pipe transitions in this gym env flip **`$0750` (AREA_POINTER)**, NOT `$0760` (which stays constant). Verified on 1-2 (3 transitions) and 4-2 (5). Every prior 8-4 search watched `$0760`, so it was *touching* transitions and never registering them. Fixing `area_search` to use `$0750` immediately let it **detect the escape** — entering room `(3,229)` via the lift region (x≈1157).

**8-4 room graph (via `$0750`):** `(3,101)` main corridor → `(3,229)` (a dead-end lift detour, entered at x≈1157) and `(3,2)` (the page-16 loop region at x3468+). The disassembly loop-gate (world-8 pages $06/$0b/$10 require `Player_Y ($00CE) == $F0=240` + on-ground) lives in room `(3,2)`.

**CRUX (fully diagnosed):** A room-graph DFS + a 1200s focused search inside room `(3,2)` both stall at **x=3848 (the page-16 wall)** — no forward room, no flag. The reason is now exact: at the gate, grounded Mario stands on a floor at **`$00CE`=128**, but the gate needs **`$00CE`=240** (screen-bottom). The **max grounded Y reachable on this surface is 128 — there is no Y=240 floor** — so the on-foot gate is *physically unsatisfiable* on the reachable surface. The escape therefore requires reaching a **lower vertical section** (where Y=240 is a real floor) via a pipe/drop/lift maneuver — and exhaustive search finds **no `$0750` transition except the 229 dead-end**. So 8-4's escape is beyond action-chunk forward search in this env: it needs reverse-curriculum from a constructed post-gate state, imported route inputs, or a frame-precise lift/pipe controller. The `$0750` fix + room-graph mapping is nonetheless a real, reusable advance (helps 4-4/7-4). New tooling: `scripts/recon_8_4.py`, `scripts/pipe_sweep_8_4.py`, `scripts/solve_area_chain.py` (room-graph DFS), `area_search` `$0750`-aware with `forbid_keys`/`max_x`/`actions`.

## What was built (all from scratch, on-device M2 Pro)
- **Emulator-as-forward-model search** over the deterministic, snapshot-able NES env (`mario/search.py`).
- **`beam_search`** — death-aware, x-greedy beam. Beats linear levels.
- **`coverage_search`** — the key new contribution: beam fused with Go-Explore coverage. Dedup key is the *cell* `(area $0760, x-tile, y-tile)`; each lineage earns a one-time novelty bonus; progress is *area-first* then x. Degrades to plain beam on linear levels; on mazes it keeps "same x, different room/height" states distinct. **Cracked 4-2's vertical/warp-zone maze**, where the plain beam emptied at x=3011.
- **Distillation**: a learned `MarioPolicy` net (search→BC→warm-start DAgger) **beats 1-1 100%**; it plays 1-1 in the stitched video.
- Self-verifying infra: per-level `data/solutions/*.json`, artifact-backed gates, self-healing `CLAUDE.md`, 38/38 tests green.

### Levels solved (10 total)
`1-1` (net), `1-2, 1-3, 1-4, 2-1, 4-1, 8-1, 8-2, 8-3` (beam), `4-2` (coverage_search). any% route = 1-1,1-2,4-1,4-2,8-1,8-2,8-3,**8-4** → **7/8 cleared**.

## The 8-4 wall — what it is (RAM-verified)
8-4's first area is a corridor of piranha-plant pipes, lava pits, and Bullet Bills that **loops**. At the wall:

```
page($0725)=16  x≈3847  Y($00CE)=108  float($001D)=1 (airborne)  area($0760)=3
push right →    page=12  x≈2847   ... (teleported back 4 pages)
```

This is SMB's `ProcLoopCommand` loop-checkpoint: cross the page-16 boundary in the wrong state and you're sent back to page 12. `Player_Y_HighPos ($00B5)` stays =1 throughout (no lower vertical section is reached), and pressing `down` anywhere only drops Mario into death-pits — **no pipe is enterable on the reachable surface.**

## Every from-scratch method tried (all fail)
| Method | Knob | Outcome |
|---|---|---|
| `beam_search` | x-greedy | precision wall x≈2576 (beam empties) |
| `coverage_search` | cov_bonus | reaches loop wall x≈3847 |
| + `area_bonus=6000` | reward any pipe entry | ~1,700 cells explored, **area never leaves 3** |
| + `ground_bonus` | reward grounded x | starves jumps OR grounded crossing still loops |
| + `loop_back_px=600` | kill looped nodes | grounded to x3866, gate never crossed |
| + `pipe_macro` | atomic sustained-down | **no pipe entry** anywhere (depth 250) |
| 64-frame `down` probe | every grounded pos on path | only death-pits, never a pipe |
| `cf=4` fine control | shorter chunks | can't clear lava pits (short jumps), dies x167 |
| **`cf=2` gate probe** | 400-wide, loop-pruned, all Y/jumps at the crossing | **max x=3836, gate NEVER crossed** |
| `go_explore` (random) | down-weighted | stalls at precision x≈2584 |

Artifacts: `runs/*cov_8_4*/contact_sheet.png`, `runs/*probe_8_4*/`, `runs/gate_8_4.png`, `scripts/gate_probe.py`.

## Why it resists from-scratch forward search
The intended human/TAS route exits via a pipe that needs **sub-tile-perfect centered entry** or a **descent that a forward planner reads as death/no-progress**. A reward that's strong enough to chase those starves the precise jumping needed to reach them; a reward weak enough to keep jumping never escapes the loop. The loop is, by design, a trap for greedy forward motion — and coverage/novelty can't help because the escape state is never *reached* to be marked novel.

## Realistic ways to beat 8-4 (not pursued — would break "from scratch")
1. **Authored/imported route**: hand-author 8-4's pipe-entry waypoints + script the centered entry, then verify+replay. (The build plan explicitly allowed external route data for 8-4.)
2. **ROM-extracted pipe affordances**: parse the level's enterable-pipe objects from ROM, navigate, enter.
3. **Bootstrapped value function**: generate beyond-the-loop states some other way, train a value net, guide search. Chicken-and-egg; uncertain.

## How to reproduce
```
# any% video (uses cached solutions):
PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/stitch_solutions.py any%
# re-solve a linear level:   scripts/solve_beam.py <w> <s>
# re-solve a maze level:     scripts/solve_coverage.py <w> <s> [beam cov stuck budget cf area ground loopbk pipemc]
# the 8-4 gate proof:        scripts/gate_probe.py
```
