# Klippbok Pinokio Launcher

A one-click [Pinokio](https://pinokio.co) launcher for a Gradio UI that wraps the
[Klippbok](https://github.com/alvdansen/klippbok) video dataset curation CLI.

Klippbok prepares video datasets for LoRA training. This launcher adds a
friendly local UI with a tab per pipeline stage (scan → triage → ingest →
normalize → caption → score → extract → audit → validate → organize) and a
visual **Manifest Reviewer** so you don't have to hand-edit 1,700-entry triage
manifests in a text editor.

> **Status:** in active development. See [`plans/`](../../.claude/plans) for the
> current build plan.

## Install (via Pinokio)

1. Open Pinokio.
2. Go to **Discover** → **Download from URL** and paste this repo's URL.
3. Click **Install**. Pinokio creates `env/` (a Python 3.11 venv via `uv`) and
   installs `klippbok[all]`, `gradio`, and platform-appropriate PyTorch.
4. Click **Start**. Pinokio launches the Gradio UI and surfaces **Open Web UI**
   once it captures the URL.

You must have `ffmpeg` on PATH. The launcher will not install it automatically
(platform-specific).

## API keys

Copy `.env.example` to `.env` and fill in any providers you use:

- `GEMINI_API_KEY` — for `caption -p gemini` and `audit -p gemini`.
- `REPLICATE_API_TOKEN` — for `caption -p replicate`.

Or paste the keys into the **Settings** tab of the UI; they'll be injected into
the Klippbok subprocess environment without touching disk unless you click
**Save to .env**.

## Repo layout

```
pinokio.js        Pinokio dynamic menu
install.js        venv + deps install
start.js          launch Gradio, capture URL
update.js         git pull + re-sync
reset.js          fs.rm env/
torch.js          GPU-aware PyTorch install
app/
  app.py          Gradio Blocks UI
  runner.py       subprocess streaming
  manifest.py     Manifest Reviewer logic
  requirements.txt
docs/
  cli_help.txt    (generated at install time)
```

## License

MIT (tentative — will align with Klippbok's own license on release).
