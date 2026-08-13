# Mario AI — Architecture Diagrams

Two sets. Read the **current** set first; it matches the July 28, 2026 research
contract. The June 2026 set is a historical V0–V4 walkthrough and is stale on
several facts (8-4 is solved; learning is search guidance, not a replacement
controller; SMA4 option work exists).

Rendered with **OpenAI `gpt-image-2`** (2560×1440, high quality).

---

## Current architecture (July 2026)

### 1. System context
![System context](11-current-system-context.png)
Exact search is the solver; Mario is the adversarial case study, not the whole claim.

### 2. Module map
![Module map](12-current-module-map.png)
Three layers: NES search core, GameAdapter cross-game, option planning with unimplemented v3.

### 3. Search → replay lifecycle
![Search replay lifecycle](13-search-replay-lifecycle.png)
A level is solved only after independent seed-0 replay reaches the flag.

### 4. Five scientific objects
![Five scientific objects](14-five-scientific-objects.png)
MetaState is a task label; physical records and lineage cannot be substituted.

### 5. Two-branch research program
![Two-branch roadmap](15-two-branch-roadmap.png)
Next: freeze the v3 schema and make the synthetic known-quotient suite pass invariance checks.

---

## Historical V0–V4 walkthrough (June 2026)

These diagrams document the original NES search → distill → DAgger story. Treat
captions as period evidence. In particular, diagram 10 still says 8-4 is a
holdout; that was later solved and any% is 8/8.

### H1. System Context
![System context](01-system-context.png)
The developer drives search/training; the NES emulator is the forward model.

### H2. Module Map
![Module map](02-module-map.png)
How `mario/` layered up before adapters and option planning.

### H3. Expert Iteration Loop
![Expert iteration loop](03-expert-iteration-loop.png)
Search teaches, the net distills, DAgger corrects learner failures.

### H4. Beam Search Internals
![Beam search internals](04-beam-search-internals.png)
One depth step of `beam_search()`: expand chunks, prune deaths, score Φ, keep top-k.

### H5. The Search Teachers
![Search teacher family](05-search-teacher-family.png)
`beam_search`, `coverage_search`, `area_search`, `go_explore`, plus live rescue.

### H6. Observation Vector
![Observation vector](06-observation-vector.png)
Ego-centric tile grid + scalars; absolute level-x is never a feature.

### H7. Death-aware Reward
![Reward and progress](07-reward-and-progress.png)
`state_score` avoids Tom7's pit-jump and fake-progress-counter failures.

### H8. Distillation Pipeline
![Distillation pipeline](08-distillation-pipeline.png)
Search trajectories become a compact `MarioPolicy`.

### H9. DAgger Correction Loop
![DAgger correction loop](09-dagger-correction-loop.png)
Label the learner's own pre-death and high-entropy states.

### H10. Full Playthrough & the 8-4 Wall (outdated)
![Solve, stitch, and the 8-4 wall](10-solve-stitch-and-8-4-wall.png)
Period caption: 7 of 8 any% levels. **Superseded:** 8-4 is solved; any% is 8/8.
