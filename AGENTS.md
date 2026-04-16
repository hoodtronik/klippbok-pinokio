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
