# Agent Instructions

Scope: this file applies to the whole repository.

## Research partners (who works on this codebase)

Two AI agents collaborate on this project as peer research partners, alongside the human
lead (Niral, repo owner / director):

- **Claude Code** (Anthropic, Opus 4.x) — runs in the primary terminal pane. Strengths: long-context
  synthesis, first-principles framing, writing/curating the narrative (`CLAUDE.md`, `notes/`,
  `V*_FINDINGS.md`), benchmark/experiment design, independent verification, and orchestration.
  Has a Playwright browser open and **monitors usage limits** (its own at `claude.ai #settings/usage`
  and Codex's at `chatgpt.com/codex/.../analytics`).
- **Codex** (OpenAI, gpt-5.5) — runs in an adjacent tmux pane (`node`, pane `0:0.1`), `--yolo`.
  Strengths: deep disassembly dives, RAM reverse-engineering, focused multi-step implementation,
  and independent re-derivation. Historically "owns the working tree" during its passes so the two
  don't edit the same files simultaneously.

Coordination protocol:

- **One writer at a time per file/area.** Before editing files another agent is actively changing,
  coordinate (or work in a separate module). Untracked files are the shared work surface — check
  `git status` and the other agent's pane before large edits.
- **Load-balancing by usage limits.** When one engine is rate-capped (5-hour / weekly window
  maxed), route the next pass to the other until its timer resets; Claude monitors both dashboards
  and picks the engine with more headroom for heavy work.
- **Cross-review.** Each agent independently verifies the other's claims against artifacts (replay
  `beat=True`, passing tests, contact sheets) rather than trusting prose. Disagreement is logged,
  not smoothed over.
- **Briefing handoffs.** Claude relays comprehensive context to Codex (and vice-versa) at the start
  of a handed-off pass — current state, the specific task, conventions, and what NOT to redo.
- Messages to Codex go ONLY to the gpt-5.5 `node` pane in this repo; verify the footer first
  (pane indices can shift). After a large bracketed paste, wait ~2s before pressing Enter.

Both agents follow everything below.

Before substantial work:

- Read `CLAUDE.md` for current working memory.
- Read `notes/session-log.md` and any relevant file under `notes/sessions/`.
- Run `./venv/bin/python scripts/update_status.py` before editing `CLAUDE.md`; never hand-edit the generated STATUS block.

When a session changes what is known:

- Update `notes/session-log.md` with a short entry.
- Add or update a detailed note under `notes/sessions/YYYY-MM-DD-topic.md`.
- Add any new papers, docs, or external technical references to `notes/research-bibliography.md`.
- Keep `CLAUDE.md` concise: durable verified facts, active issues, and one next action only.
- Cite concrete artifacts and verification commands. It is fine to cite local ignored paths such as `data/solutions/...` and `runs/...`.

Repo conventions:

- Use the local venv: `./venv/bin/python`.
- Keep ROMs local. `roms/` is ignored and must not be committed.
- Keep generated datasets, videos, checkpoints, and run outputs out of source notes; cite their paths instead.
- Prefer adapter-level abstractions for non-NES Mario games. The emulator should provide snapshots and forward dynamics; search remains the solver unless a later note explicitly changes that direction.
