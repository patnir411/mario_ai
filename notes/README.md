# Project Notes

This directory holds durable working notes that are too detailed for `CLAUDE.md`.

Use this split:

- `CLAUDE.md`: top-level working memory, current status, active issues, and exactly one next action.
- `notes/session-log.md`: chronological index of meaningful sessions, newest first.
- `notes/sessions/YYYY-MM-DD-topic.md`: detailed experiment notes, findings, artifacts, and follow-up options.
- `notes/research-bibliography.md`: consolidated literature/tool references and how they map to repo decisions.

When a session changes what we know, update the notes before ending the turn. Include:

- what was tested or implemented;
- the strongest artifact paths, including ignored-but-local `data/` or `runs/` outputs;
- verification commands and results;
- known failure modes or false positives;
- the next concrete experiment options.
- any new papers, docs, or external references that should be added to `notes/research-bibliography.md`.

Do not put ROMs, generated videos, checkpoints, or large datasets here. Cite their local paths instead.
