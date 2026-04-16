# Klippbok Pinokio Launcher

A one-click [Pinokio](https://pinokio.co) launcher for a Gradio UI that wraps
[Klippbok](https://github.com/alvdansen/klippbok) — the video dataset
curation CLI for LoRA training. Replaces hand-editing 1,700-line triage
manifests in a text editor with a visual reviewer, and turns the ten-command
Klippbok pipeline into tabs a first-timer can follow.

## What it does

Klippbok prepares video datasets for LoRA training through a pipeline of
CLI subcommands. This launcher puts the whole thing behind a Gradio UI:

- **One tab per CLI subcommand** — Scan, Triage, Ingest, Normalize,
  Caption, Score, Extract, Audit, Validate, Organize. Each tab streams
  stdout into a log pane, has a Dry Run toggle, and a Cancel button that
  actually kills the subprocess.
- **Manifest Reviewer** — the payoff tab. Loads either
  `triage_manifest.json` (clip-level) or `scene_triage_manifest.json`
  (scene-level). Renders thumbnails via ffmpeg, paginates at 25 entries
  per page, bulk-gates by CLIP score, and saves a reviewed JSON
  alongside the original so your raw triage output stays untouched.
  Triage's Run completion auto-fills the path.
- **Directions tab** with a full pipeline reference plus an ELI5
  beginner walkthrough for both image and video LoRAs.
- **Project scaffold** — one click creates the `clips/`, `concepts/`,
  `output/` folder tree Klippbok expects, and drops a README into
  `concepts/` explaining what you're supposed to put there.
- **Settings** — API keys persist to `.env` (gitignored) and hydrate
  back on startup; an install check reports Klippbok + ffmpeg + Python
  versions so you can diagnose a broken environment at a glance.

## Install via Pinokio

1. Open Pinokio.
2. **Discover** → **Download from URL**.
3. Paste `https://github.com/hoodtronik/klippbok-pinokio`.
4. Click **Install**. Pinokio will:
   - Create `env/` — a Python 3.11 venv via `uv` (auto-downloads CPython
     3.11 if your system Python doesn't match).
   - `uv pip install -r app/requirements.txt` → `klippbok[all]`,
     `gradio`, `pillow`.
   - Run `torch.js` which swaps in a platform-appropriate PyTorch:
     CUDA 12.4 wheels on NVIDIA, ROCm 6.1 on Linux+AMD, DirectML on
     Windows+AMD, CPU with MPS on Apple Silicon, plain CPU elsewhere.
   - Dump `--help` for every Klippbok subcommand into
     `docs/cli_help.txt` as ground truth for the form fields.
5. Click **Start**. Pinokio captures the Gradio URL from stdout and
   surfaces **Open Web UI** in the menu.
6. In the UI, go to the **Directions** tab first.

**One prerequisite Pinokio won't install for you: `ffmpeg` on PATH.**
Klippbok calls it for scene detection, normalization, and the
Reviewer's thumbnail extraction. Install guides:

- Windows: https://www.ffmpeg.org/download.html (add to PATH after
  extracting).
- macOS: `brew install ffmpeg`.
- Linux: `apt install ffmpeg` / `dnf install ffmpeg`.

Verify with the **Check installation** button in the Settings tab once
the UI is up.

## Manual install (without Pinokio)

If you want to run outside Pinokio:

```bash
# Create a Python 3.11 venv. uv auto-downloads CPython 3.11 if needed.
uv venv --python 3.11 env

# Activate — Windows
env\Scripts\activate

# Activate — macOS / Linux
source env/bin/activate

# Install deps
uv pip install -r app/requirements.txt

# Optional: dump --help for every subcommand
cd app && python dump_help.py && cd ..

# Launch the UI
cd app
python app.py --host 127.0.0.1 --port 7860
```

Then open `http://127.0.0.1:7860`.

## First time using it

Go to the **Directions** tab. It has:

- A 60-second mental model of the Klippbok pipeline.
- "What goes in the Concepts folder" — the one thing nobody can do
  for you, and the usual place first-timers get stuck.
- A tab reference table (purpose / needs / produces, for each tab).
- Two common recipes (full pipeline; re-caption existing dataset).
- An **Explained Like You're 5** section at the bottom with a complete
  beginner walkthrough for both image and video LoRAs — from "I have
  some images" through "I have a folder ready to feed my trainer".

**In a hurry:** **Project tab** → **Scaffold a new project layout** →
pick a parent folder and a project name → **Create project folders**.
Put your content in the `clips/` subfolder that appears, and head back
to the Directions tab for what to do next.

## Manifest Reviewer workflow

After Triage writes a manifest:

1. Go to **Manifest Reviewer**. Path auto-fills from the Triage run
   you just did. (If you restarted the UI, click **Use latest triage
   output** — it scans your Working directory.)
2. **Load**. Thumbnails generate in parallel (8 ffmpeg jobs) on first
   page visit; cached in `cache/thumbs/` so subsequent visits are
   instant.
3. **Bulk first.** Drag the **threshold slider**, click **Include
   where score ≥ threshold** to gate by CLIP confidence. Click
   **Exclude clips with text overlay** to drop watermarked content.
4. **Then clean up by hand.** Page through, look at the thumbnail,
   compare it to the concept label + score, toggle Include on
   anything CLIP got wrong.
5. **Save.** Writes `<name>_reviewed.json` next to the original
   (check "Overwrite original" if you'd rather edit in place).
6. Paste the reviewed path into the **Ingest** tab's `--triage`
   field. Ingest only scene-splits the clips you kept.

## API keys

Caption and Audit need a vision-language model:

- **Gemini** (recommended, free tier): `GEMINI_API_KEY`. Get one at
  https://aistudio.google.com/apikey.
- **Replicate** (pay-per-use): `REPLICATE_API_TOKEN`.
- **Local Ollama / OpenAI-compatible**: set `--provider openai` on
  Caption, then in the advanced accordion set `--base-url
  http://localhost:11434/v1` and `--model llama3.2-vision`. No key
  required.

Paste keys into the **Settings** tab. Click **Save to .env** to
persist to `.env` at repo root (gitignored); the UI auto-loads them
on next startup.

## Troubleshooting

- **UI shows an old version after Update.** Pinokio pulls new files
  but doesn't restart the running Python. Stop → Update → Start, and
  hard-refresh the browser tab (Ctrl+Shift+R).
- **UnicodeDecodeError during Scan / Triage / Ingest on Windows.**
  Fixed — subprocess env forces `PYTHONUTF8=1` and `PYTHONIOENCODING=
  utf-8` so Klippbok's internal ffprobe reader doesn't fall back to
  cp1252. If you hit one anyway, file an issue.
- **Triage refuses to launch with "Concepts folder has no…".**
  Populate `concepts/` per the Directions tab. One subfolder per
  concept, 5–20 jpg/png reference images in each.
- **First Triage run stalls at "Downloading CLIP model".** Expected —
  ~150MB download, cached after first success. Re-run if it was
  interrupted.
- **Caption fails with auth error.** API key not set, or the wrong
  provider selected. Check Settings.
- **Reviewer thumbnails are empty boxes.** Either ffmpeg isn't on
  PATH (Settings → Check installation), or the manifest's file paths
  don't exist (you moved the dataset after Triage). Fix the paths or
  re-run Triage from the current location.

## Repo layout

```
pinokio.js        Pinokio dynamic menu
install.js        uv venv + uv pip + torch.js + dump_help
start.js          daemon launch, capture URL
update.js         git pull + re-sync
reset.js          fs.rm env/
torch.js          platform-aware torch install
icon.png
.env.example
.gitignore
app/
  app.py          Gradio Blocks UI (Directions, Project, 10 command
                  tabs, Manifest Reviewer, Settings)
  runner.py       subprocess streaming + Cancel + UTF-8 env
  manifest.py     Reviewer schema adapter + thumbnail cache
  dump_help.py    called from install.js
  requirements.txt
docs/
  cli_help.txt    (generated at install time; gitignored)
cache/
  thumbs/         (Manifest Reviewer thumbnails; gitignored)
```

## Credits

- Klippbok by [@alvdansen](https://github.com/alvdansen/klippbok) —
  the underlying CLI this launcher wraps.
- Pinokio launcher patterns adapted from
  [pinokiofactory/z-image-turbo](https://github.com/pinokiofactory/z-image-turbo)
  and [cocktailpeanut/ace-step.pinokio](https://github.com/cocktailpeanut/ace-step.pinokio).

## License

MIT. See Klippbok's own license for terms on the underlying CLI.
