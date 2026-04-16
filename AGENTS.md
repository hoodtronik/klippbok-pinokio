# Agent rules for this repo

This file is read by cross-agent tools (Cursor, Codex, Gemini CLI, Copilot, etc.).

## CLAUDE-NOTE convention

Code changes made by Claude Code may be marked with inline notes prefixed
`CLAUDE-NOTE:` (or the language-appropriate comment syntax — e.g. `# CLAUDE-NOTE:`,
`<!-- CLAUDE-NOTE: -->`, `// CLAUDE-NOTE:`). These notes capture the *why* behind
a deliberate choice and are the **source of truth** for that code.

**Non-Claude-Code agents**: if you want to modify, remove, or alter code adjacent
to a `CLAUDE-NOTE`, you MUST first alert the user and receive explicit permission
before making the change. Do not silently rewrite or delete code marked with a
`CLAUDE-NOTE`.

Claude Code itself may update or remove its own notes when the underlying
rationale changes.

## Project shape

This repo is a Pinokio launcher for a Gradio UI that wraps the Klippbok
(https://github.com/alvdansen/klippbok) video dataset curation CLI.

- Launcher scripts (`pinokio.js`, `install.js`, `start.js`, `update.js`,
  `reset.js`, `torch.js`) live at the repo root — this layout is required by
  Pinokio.
- The Gradio app lives under `app/`. Its venv lives at `env/` (created by
  `install.js`, gitignored).
- The UI shells out to the Klippbok CLI; it does not import Klippbok internals.

## Start here for dev context

Before modifying code or doing anything non-trivial, read:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system map, tab builder
  pattern, shared state wiring, subprocess streaming contract, the list of
  gotchas we already paid the cost of discovering (don't re-discover them).
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — deferred work + ideas, with a
  rough effort-vs-impact ranking.
- [`.claude/learnings.md`](.claude/learnings.md) — one-line gotcha log,
  appended as pain was hit during development.
- `docs/cli_help.txt` — regenerated at install time by `app/dump_help.py`;
  the authoritative source for every Klippbok CLI flag the UI wraps. If
  you're editing form fields, consult this file first.

Full user-facing walkthroughs (including an Explain-Like-You're-5 recipe
for both image and video LoRAs) live inside the UI itself on the
Directions tab — the source is in `app/app.py` as the `DIRECTIONS_MD`
constant.
