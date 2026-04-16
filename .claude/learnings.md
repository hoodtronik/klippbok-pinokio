# Project learnings

Append one-line notes about non-obvious gotchas hit while building this repo.
Keep each entry under 15 words. Prune stale entries as the code evolves.

<!-- Example format:
- 2026-04-16 — Pinokio `{{input.event[1]}}` only works inside `local.set` after a `shell.run` `on`-event
-->

- 2026-04-16 — gradio 4.x didn't cap huggingface-hub<1; use gradio>=5 to avoid HfFolder ImportError
- 2026-04-16 — Pinokio venv path: `venv:"../env"` + `path:"app"` puts env at repo root, not `app/env`
- 2026-04-16 — Pinokio start.js must use `python -u` + PYTHONUNBUFFERED=1 or URL regex never matches
