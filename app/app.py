"""Klippbok Pinokio launcher — Gradio UI.

Build plan complete (see ~/.claude/plans/clever-sleeping-emerson.md):

    Step 2 - Pinokio skeleton + hello-world Gradio
    Step 3 - klippbok[all] + torch.js + help dump
    Step 4 - Project tab + Scan tab + streaming-log pattern
    Step 5 - Triage/Ingest/Normalize/Caption/Score/Extract/Audit/
             Validate/Organize tabs + minimal Settings for API keys
    Step 6 - Manifest Reviewer (both schemas, thumbnails, pagination,
             bulk actions, auto-link from Triage output)
    Step 7 - Settings polish (.env persist + hydrate on start +
             install check), full README, ELI5 beginner walkthrough
             in Directions tab

Further work, when needed: Python-exe override, per-clip Notes
field in the Reviewer, concept subfolder quick-add in the Project tab.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import gradio as gr

import caption_images as caption_images_mod
import manifest as mft
import project_state as ps_mod
import validate_images as validate_images_mod
import pipeline_installer as pi
import pipeline_setup as pipe
import runner


# Thumbnails live next to the venv, outside `app/`, so they're gitignored and
# don't bloat the source tree even for 1700-entry manifests.
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "thumbs"

# CLAUDE-NOTE: PAGE_SIZE pre-renders this many Gradio rows. On page turn we
# update their values in a single batch. 25 is a compromise: small enough
# that parallel thumbnail extraction completes in a couple of seconds, big
# enough that reviewers can scan quickly without constant paging.
_PAGE_SIZE = 25


# ----- long-form constants (markdown blocks, readme text) ------------------

CONCEPTS_README = """\
Klippbok Concepts Folder
========================

This folder holds reference images that teach Klippbok what each concept
looks like. Each subfolder = one concept. Klippbok's Triage stage matches
your video clips against these references.

Quick start
-----------
1. Create a subfolder for each concept you want to match on — a character,
   a style, a camera motion, an object. Use short, lowercase names — they
   become the labels in the triage manifest.

     concepts/
       my_character/
         ref_01.jpg
         ref_02.jpg
         ...
       cinematic_style/
         ref_01.jpg
         ...

2. Drop 5-20 representative jpg or png images into each subfolder. Variety
   (different poses, lighting, angles) helps CLIP build a robust match.

3. Run the Triage tab in the Klippbok UI. Then Manifest Reviewer to fix
   anything CLIP got wrong.

Klippbok does NOT generate these images. You curate them yourself - that
is why they are called "concepts" and not "samples". The quality of your
training set is bounded by how well these images describe what you want.

See the Directions tab in the Gradio UI for the full pipeline narrative.
"""


# CLAUDE-NOTE: Target-trainer presets. Pure UI sugar — picking a preset
# pushes default values into the Scan / Ingest / Normalize / Caption /
# Validate fields so the user doesn't have to remember per-trainer
# specs. Klippbok's CLI itself is preset-agnostic; it just receives
# whatever flag values are in the fields at Run time.
#
# Spec sources (verified 2026-04-19, see commit message for full cites):
#   * Wan 2.2 — fps=16, frames 4n+1 max 81 from
#     env/Lib/site-packages/klippbok/config/defaults.py +
#     Wan repo wan/configs/shared_config.py.
#   * LTX-Video / LTX-2 — fps=24 (community convention; the trainer
#     itself doesn't pin fps), spatial dims must be divisible by 32,
#     frames 8n+1, max 121-161, from Lightricks/LTX-Video-Trainer
#     docs/dataset-preparation.md.
#   * HunyuanVideo — fps=24 (sample_video.py), frames 4n+1 (inferred
#     from 3D VAE 4x temporal compression), 129 frames recommended,
#     resolutions 544x960 / 720x1280 from Tencent-Hunyuan/HunyuanVideo
#     official requirements table.
#   * Image models — no fps / frame constraints; FLUX / Z-Image /
#     Qwen-Image train at 1024-class resolutions by community
#     convention.
TRAINER_PRESETS: dict[str, dict] = {
    "Wan 2.2": {
        "scan_fps": 16,
        "ingest_max_frames": 81,
        "normalize_fps": 16,
        "caption_fps": 1,
        "frame_count_rule": "4n+1 (Wan / HunyuanVideo)",
        "guidance": (
            "**Wan 2.2** — fps **16**, resolution **480p or 720p (1280×720)**, "
            "frames must satisfy **4n+1** (5, 9, 13, …, 81). Spatial dims "
            "divisible by 16. Resolution isn't a UI field — set it in "
            "`klippbok_data.yaml` if you need to override the trainer's default."
        ),
    },
    "LTX-Video / LTX-2": {
        "scan_fps": 24,
        "ingest_max_frames": 121,
        "normalize_fps": 24,
        "caption_fps": 1,
        "frame_count_rule": "8n+1 (LTX)",
        "guidance": (
            "**LTX-Video / LTX-2** — fps **24** (community convention; the "
            "trainer doesn't pin it), resolution **768×768 / 704×480 / 1216×704**, "
            "frames must satisfy **8n+1** (9, 17, 25, …, 121, 161). Spatial "
            "dims **must be divisible by 32** (hard VAE constraint)."
        ),
    },
    "HunyuanVideo": {
        "scan_fps": 24,
        "ingest_max_frames": 129,
        "normalize_fps": 24,
        "caption_fps": 1,
        "frame_count_rule": "4n+1 (Wan / HunyuanVideo)",
        "guidance": (
            "**HunyuanVideo** — fps **24**, resolution **544×960 or 720×1280**, "
            "frames **129** typical (4n+1 pattern, derived from the 3D VAE's "
            "4× temporal compression). Spatial dims divisible by 16."
        ),
    },
    "Image Models (FLUX / Z-Image / Qwen)": {
        "scan_fps": 16,            # not applicable, but the field needs a number
        "ingest_max_frames": 1,
        "normalize_fps": 16,       # not applicable
        "caption_fps": 1,
        "frame_count_rule": "image-only (skip frame check)",
        "guidance": (
            "**Image-model LoRAs** (FLUX.2 / Z-Image / Qwen-Image) — still "
            "images only. Skip Ingest / Normalize entirely; use the Caption "
            "tab (auto-handles PNG/JPG via the in-process shim) and the "
            "Validate tab (auto-detects image-only mode). 1024-class "
            "resolutions are convention; divisibility by 16 is safe."
        ),
    },
}

DEFAULT_TRAINER_PRESET = "Wan 2.2"


DIRECTIONS_MD = """\
# How Klippbok works

Klippbok turns a pile of raw video into a LoRA training dataset. It's a
**pipeline** — run each stage in order, review the output, adjust, move on.

Every tab in this UI wraps one Klippbok CLI subcommand. Nothing here is
magic — you can run the same commands in a terminal and get the same
results. The UI exists to make the pipeline approachable and to give you
a visual reviewer for the triage manifest (the part that would otherwise
mean hand-editing a 1,700-line JSON file in a text editor).

> **Need deeper detail?** The Klippbok authors maintain authoritative
> docs at
> [PIPELINES](https://github.com/alvdansen/klippbok/blob/main/docs/PIPELINES.md),
> [CAPTIONING](https://github.com/alvdansen/klippbok/blob/main/docs/CAPTIONING.md),
> [COMMANDS](https://github.com/alvdansen/klippbok/blob/main/docs/COMMANDS.md),
> and the
> [WALKTHROUGH](https://github.com/alvdansen/klippbok/blob/main/docs/WALKTHROUGH.md).
> Reach for those when this overview isn't enough.

---

## The 60-second mental model

1. You **hand-curate a concepts folder** — small subfolders of reference
   images, one per thing you want the model to learn (a character, a
   style, a motion, an object).
2. **Triage** uses CLIP to score every clip against those references and
   writes a manifest JSON saying which clip matches which concept.
3. **Manifest Reviewer** lets you visually flip
   `include: true/false` flags where CLIP got things wrong.
4. **Ingest** splits long videos into scene-level clips using the reviewed
   manifest — or skip straight to Normalize if your clips are already cut.
5. **Caption** writes `.txt` sidecar captions via a vision-language model
   (Gemini / Replicate / local Ollama).
6. **Score** and **Audit** spot-check caption quality.
7. **Validate** checks dataset completeness.
8. **Organize** lays it out in the format your trainer expects (musubi,
   aitoolkit, or a flat tree).

---

## What goes in the Concepts folder

This is the one thing you have to do by hand — Klippbok cannot invent it.

```
concepts/
  my_character/      ← folder name = concept label
    ref_01.jpg       ← 5-20 reference images
    ref_02.jpg
    ref_03.jpg
    ...
  another_character/
    ...
  cinematic_style/
    ...
```

- **One subfolder per concept.** Short, lowercase name. It becomes the
  matching label in the triage manifest.
- **5-20 jpg or png images per concept.** More helps up to a point.
- **Vary the references** — different poses, lighting, angles. CLIP
  embeds each image and averages, so diversity protects against
  over-fitting to one specific frame.
- **You curate these yourself.** Pull stills from your best clips, find
  screenshots online, generate references with an image model — whatever
  represents the thing you want.

If your concepts folder is empty or its subfolders have no images, the
Triage tab will refuse to launch and point you back here.

---

## What each tab does

| Tab | Klippbok command | When to use it | Needs | Produces |
|-----|-----------------|----------------|-------|----------|
| **Directions** | — | You are here. | — | — |
| **Project** | — | Set shared working/concepts/output paths once per session. | — | Paths for every other tab |
| **Scan** | `klippbok.video scan` | First thing you run on a new clip dir. Read-only diagnostic. | Clip dir | Text report |
| **Triage** | `klippbok.video triage` | Match clips to concepts via CLIP. | Clip dir + populated concepts dir | `triage_manifest.json` or `scene_triage_manifest.json` |
| **Ingest** | `klippbok.video ingest` | Split long videos into scene-level clips (optionally filtered by a triage manifest). | Raw video file/dir | Clip dir in output |
| **Normalize** | `klippbok.video normalize` | Standardize fps / resolution / frame count on already-cut clips. | Clip dir | Clip dir in output |
| **Caption** | `klippbok.video caption` | Generate .txt captions via VLM (Gemini / Replicate / Ollama). | Clip dir + API key | .txt per clip |
| **Score** | `klippbok.video score` | Local heuristic quality check on captions. No API calls. | Dir of .txt | Report |
| **Extract** | `klippbok.video extract` | Export PNG reference frames. | Clip dir | PNG dir |
| **Audit** | `klippbok.video audit` | Re-caption with VLM, compare to existing. Catch drift. | Clip dir + API key | Report |
| **Validate** | `klippbok.dataset validate` | Dataset-level completeness + quality checks. | Dataset dir | Report |
| **Organize** | `klippbok.dataset organize` | Restructure for trainer (musubi / aitoolkit / flat). | Dataset dir | Trainer-ready dir |
| **Manifest Reviewer** | — (UI-only) | Visually fix include flags in a triage manifest. | A manifest JSON | `*_reviewed.json` |
| **Settings** | — | API keys for Gemini / Replicate. | — | In-memory env for Caption/Audit/Ingest |

---

## Reviewing the triage manifest

Triage writes a JSON manifest marking every clip (or scene) with an
`include: true/false` flag based on CLIP similarity. That's the one
place the pipeline expects human judgment — CLIP gets confused on
edge cases, you know your dataset.

**1. Triage writes the manifest.** Default destination:
- `<clips_dir>/triage_manifest.json` for clip-level runs (when your
  source is already short clips).
- `<clips_dir>/scene_triage_manifest.json` for scene-level runs
  (long source videos Klippbok auto-scene-detected).

If you passed `--output /some/path.json` to Triage, it goes there
instead. Either way, the **Manifest Reviewer's** path box
auto-populates with the latest Triage output as soon as Triage
finishes. If you restart the UI, click **Use latest triage output**
and it'll scan your working directory for a matching file.

**2. First page load extracts thumbnails.** The Reviewer shells out to
ffmpeg (8 jobs in parallel) to grab a frame from the midpoint of each
clip/scene. Expect a few seconds on the first page of each manifest;
subsequent visits are instant thanks to `cache/thumbs/`.

**3. Triage in bulk first, clean up by hand second.** In this order:
- **Threshold slider + "Include where score ≥ threshold"** — gate by
  CLIP confidence. Klippbok's default cutoff is 0.70; raise it for
  more precision (fewer false positives), lower it for more recall
  (fewer missed matches).
- **"Exclude clips with text overlay"** — drops every entry Klippbok
  flagged as having burnt-in text (watermarks, subtitles, UI
  captures). Training on those usually hurts more than it helps.
- **Per-row Include checkboxes** for the ones CLIP got wrong.
  Each row shows a thumbnail + the best-match concept + its
  similarity score — enough context to decide in a glance.

**4. Save.** Writes to `<name>_reviewed.json` alongside the original
by default so your raw Triage output stays untouched. Check
**"Overwrite original"** if you'd rather edit in place.

**5. Feed the reviewed manifest into Ingest.** Paste the reviewed
path into the **Ingest** tab's `--triage` field. Ingest will only
scene-split clips/scenes you kept, leaving everything else on disk
but out of the dataset.

---

## Two common recipes

### A) Full pipeline — raw footage ->trainable dataset

1. **Project tab** ->Scaffold a new project layout (or browse to existing folders).
2. Open the **concepts/** folder (filesystem) and populate it: one subfolder per concept, 5-20 reference images each.
3. **Triage tab** ->run against your raw clips. Produces a manifest.
4. **Manifest Reviewer tab** ->fix any misclassifications CLIP made. Save a reviewed manifest.
5. **Ingest tab** ->point at your raw video (file or directory) with `--triage <reviewed manifest>`. Klippbok scene-splits and writes clips into output/.
6. **Caption tab** ->set --provider, --use-case, --anchor-word. Needs a Gemini or Replicate key in Settings.
7. **Validate tab** ->with --buckets and --quality.
8. **Organize tab** ->with --trainer = musubi or aitoolkit (or both, repeatable).

### B) Re-caption an existing dataset

1. **Project tab** ->point Working directory at your existing clip dir.
2. **Settings tab** ->paste API key.
3. **Caption tab** ->check --overwrite.
4. **Score tab** ->sanity-check.
5. **Audit tab** ->compare before/after.

---

## The Settings tab and API keys

- **Gemini** (`GEMINI_API_KEY`): recommended, free tier available, best caption quality. Get a key at https://aistudio.google.com/apikey.
- **Replicate** (`REPLICATE_API_TOKEN`): pay-per-use. Useful as a fallback.
- **Ollama / OpenAI-compatible**: pick provider `openai` on Caption, then set `--base-url http://localhost:11434/v1` and `--model llama3.2-vision` in the advanced accordion. No API key needed.

Keys live only in memory by default. Nothing is written to disk unless
you explicitly save (coming in the next polish step).

---

## When something breaks

- **"Directory not found"** — the path you typed doesn't resolve. Double-check for typos, use forward slashes, or hit Browse.
- **"ImportError" in Triage** — CLIP's transformers download may have been interrupted. Re-run; it's cached after first success.
- **Caption "auth error"** — API key not set in Settings, or the wrong provider selected.
- **Long runs appear frozen** — check the log pane. Klippbok streams progress. If you see nothing for a minute, the process may actually be stuck — use Cancel.
- **UTF-8 errors on Windows** — already handled via PYTHONUTF8=1 in the subprocess env. If you still see one, it's a bug; please file it.

If anything's unclear, the exact CLI help for every subcommand is in
`docs/cli_help.txt` at the repo root — regenerated on every install.

There's also a full beginner walkthrough further down this page —
**Explained Like You're 5**. Scroll down if any of the above felt
too dense.

---

## Explained Like You're 5 — full beginner walkthrough

No jargon. Step by step. From zero folder on your disk to a folder your
LoRA trainer will accept.

Two recipes below:
- **Image LoRA** — you want the model to learn a face, a style, or a
  specific object in still images.
- **Video LoRA** — you want the model to learn motion (walking,
  dancing, camera moves, a specific animated character's movement).

Before picking a recipe, decide what you're training.

### First: what kind of LoRA are you making?

| You want the model to learn… | Collect… | How many | Image or Video? |
|---|---|---|---|
| A specific person / character | Photos or stills of that person | 15–50 | Either |
| An art style (cinematic, anime, oil-paint) | Images sharing that aesthetic | 20–80 | Either |
| A camera move or body motion | Clips showing the motion clearly | 10–30 | **Video** |
| A specific object (a car model, a logo, a product) | Shots of the object in varied contexts | 15–40 | Either |

**Go with Image LoRA if you're new to this.** It's faster to collect
data for, faster to train, and easier to debug when something goes
wrong. Video LoRAs teach *motion* — you need actual video clips and
you'll spend more time curating. Come back for video once you've
trained a couple of image LoRAs.

---

### Recipe 1 — Image LoRA (person / style / object)

You have a pile of reference images. You want one LoRA that learns
the thing.

#### Step 1 — Make the folders

1. Click the **Project** tab at the top.
2. Open the **Scaffold a new project layout** accordion near the
   bottom.
3. Hit **Browse…** and pick where your LoRA projects live (example:
   `D:/LoRA-Projects`).
4. Give it a name in the **Project name** box (example: `MyCharacter`).
5. Click **Create project folders**.

You now have three folders inside `D:/LoRA-Projects/MyCharacter`:
`clips/`, `concepts/`, `output/`. The three path boxes at the top
of the Project tab auto-filled with the new paths.

#### Step 2 — Put your images in `clips/`

- Open the `clips/` folder in your file explorer.
- Drag your jpg/png reference images in.
- The folder is named "clips" because Klippbok was built for video,
  but it handles images too — no renaming needed.

#### Step 3 — Skip triage, go straight to Caption

For a single-subject image LoRA, triage is overkill. Jump to Step 4.
(If you have one folder with *multiple* characters mixed together and
want to auto-separate them, come back and look at **Triage** + the
**Manifest Reviewer** — that's what they're for. For a clean
single-subject set, skip.)

#### Step 4 — Set your API key

1. Click the **Settings** tab (last tab in the strip).
2. Paste a Gemini API key into the **GEMINI_API_KEY** box.
   Free key here: https://aistudio.google.com/apikey
3. Click **Save to .env** so it sticks between sessions.

If you don't want to use Gemini, Replicate or a local Ollama server
also work — see the Settings/API keys section above.

#### Step 5 — Generate captions

1. Click the **Caption** tab.
2. The Directory auto-fills with your `clips/` path. Leave it.
3. `--provider` = `gemini`.
4. `--use-case` = pick the one that matches:
   - `character` for a person
   - `style` for an aesthetic
   - `object` for a thing
5. `--anchor-word` — a short name you'll type when prompting the
   trained model. Examples: `jane`, `mystyle`, `widget1`.
6. Click **Run**.

Watch the log. You'll see one line per image as Klippbok captions
it. When you see `[exit=0  elapsed=Ns]`, it's done and one `.txt`
file sits next to every image.

#### Step 6 — Sanity-check the captions

1. Click the **Score** tab. Run it. It checks caption length, token
   variety, and anchor-word presence. Read the report.
2. If scores look bad, go back to **Caption** with `--overwrite`
   checked and try a different `--use-case`.

#### Step 7 — Validate the dataset

1. Click the **Validate** tab.
2. Check `--quality` and `--duplicates`.
3. Run.
4. Read the report carefully:
   - **Errors** — fix before moving on (missing captions, unreadable
     files, impossible dimensions).
   - **Warnings** — judgment call. Usually fine to proceed.

#### Step 8 — Organize for your trainer

1. Click the **Organize** tab.
2. `--output` — pick or type a fresh empty folder (like
   `D:/LoRA-Projects/MyCharacter/for-training`).
3. `--trainer` — for image LoRAs, pick `aitoolkit` (most common).
   For musubi-tuner, pick `musubi`. You can pick both; they go into
   separate subfolders.
4. Click **Run**.

Done. The `--output` folder is now laid out exactly the way your
trainer expects. Feed it into ai-toolkit / kohya / musubi-tuner and
hit train.

---

### Recipe 2 — Video LoRA (motion / animated character)

You have raw video files. You want to teach a model how something
moves.

#### Step 1 — Make the folders

Same as image LoRA:
1. **Project** tab ->**Scaffold a new project layout** ->pick parent,
   name it, click **Create project folders**.
2. Three path boxes auto-fill.

#### Step 2 — Put raw videos in `clips/`

- Drop your `.mp4`, `.mov`, or `.mkv` files into `clips/`.
- Long videos are fine — Klippbok will auto-split them into scenes in
  Step 5.
- Aim for source material that shows the motion clearly without much
  other stuff happening. A 30-second clip of someone dancing beats a
  5-minute clip where they dance for 10 seconds.

#### Step 3 — Scan your videos

1. Click the **Scan** tab. Run it.
2. Read the report. The numbers you care about:
   - **Clips scanned** — how many Klippbok saw.
   - **Unusable: N** — clips Klippbok will refuse to process (wrong
     format, zero frames, etc.). If this is most of them, find better
     source.
   - **Normalize recommended: N** — clips that need fps or resolution
     fixes. Klippbok will handle this later; just note the count.
   - Issue lines like `RESOLUTION_BELOW_TARGET` — clips smaller than
     the training target (720p by default). The model can't learn
     detail that isn't in the source, so consider dropping these if
     you can.

**Proceed** to the next step if most of your clips are scannable.
**Stop and find better source** if most are unusable.

#### Step 4 — Populate the Concepts folder

This is the one step nobody can do for you.

1. Open the `concepts/` folder in your file explorer.
2. Make a subfolder for each thing you want to match on. Examples:
   - `jane_dancing/` (one character doing one motion)
   - `cinematic/` (a lighting/camera style)
   - `slow_pan/` (a specific camera move)
3. Drop **5–20 reference images** (jpg/png, not video) into each
   subfolder. These can be:
   - Stills from your best source video (use Snipping Tool, Photoshop,
     anything).
   - Web images.
   - AI-generated reference images.
4. Keep the set varied — different angles, lighting, moments. CLIP
   averages your references when matching, so diversity = better
   matches.

#### Step 5 — Run Triage

1. Click the **Triage** tab.
2. Directory auto-fills from Project. `--concepts` too.
3. Click **Run**.
4. Expect this to take a while. Klippbok:
   - Scene-detects long videos into short scenes (if they're long).
   - Downloads the CLIP model on first run (~150MB, cached after).
   - Compares every scene against every concept reference.
5. When it finishes you'll see a summary in the log and a
   `triage_manifest.json` (short clips) or `scene_triage_manifest.json`
   (long videos, one manifest for all scenes) gets written.

#### Step 6 — Review the manifest

This is the payoff. CLIP is smart but wrong sometimes. You fix it.

1. Click **Manifest Reviewer**. The path auto-fills from Step 5.
2. Click **Load**.
3. Wait a few seconds for the first page's thumbnails.
4. **Do bulk first, then clean up by hand.**
   - Drag the **threshold slider** to ~0.75. Click **Include where
     score ≥ threshold**. High-confidence matches stay. Everything
     else gets excluded.
   - Click **Exclude clips with text overlay**. Drops subtitled /
     watermarked / UI-capture clips (Klippbok flagged them during
     Triage).
5. **Then page through.** For each row:
   - Look at the thumbnail.
   - Does it really show the concept listed (top-right of the
     metadata)?
   - If yes and it's excluded ->tick Include.
   - If no and it's included ->untick Include.
6. Click **Save**. You get `<name>_reviewed.json` alongside the
   original. Your raw Triage output stays untouched.

**Don't skip this step.** Training on bad matches ruins the LoRA.

#### Step 7 — Ingest using the reviewed manifest

1. Click the **Ingest** tab.
2. Directory auto-fills with your raw video location.
3. `--output` auto-fills with your `output/` folder.
4. Paste the path to the **reviewed** manifest into `--triage`.
5. Click **Run**.

Klippbok scene-splits your raw videos — **but only the scenes you
kept in the Reviewer** — and writes trainable clips into `output/`.

#### Step 8 — Caption

1. **Settings** tab: paste API key (Gemini recommended) and Save to
   .env.
2. **Caption** tab:
   - Directory — point at your `output/` folder (where Ingest wrote
     the clips).
   - `--provider` = `gemini`.
   - `--use-case` = match what you're training (`motion`, `character`,
     `style`, or `object`).
   - `--anchor-word` = short identifier you'll use at prompt time.
3. Click **Run**. Watch as Klippbok writes one `.txt` per clip.

#### Step 9 — Score and Audit

1. **Score** tab: run against your `output/` folder. Fast, local.
2. **Audit** tab (optional): run with `--mode save_audit`. Re-captions
   with VLM and compares to existing. Tells you if caption quality
   drifted.

If Score looks bad, re-run Caption with a different provider or
use-case.

#### Step 10 — Validate + Organize

1. **Validate** tab: check `--buckets`, `--quality`, `--duplicates`.
   Run. Fix errors. Judge warnings.
2. **Organize** tab:
   - `--output` — pick a fresh empty folder for the final trainer-ready
     layout.
   - `--trainer` — `musubi` for musubi-tuner (most video trainers) or
     `aitoolkit` if you're using that. Both works.
3. Click **Run**.

Done. The Organize output folder is what you hand to your trainer.

---

### Reading the log output

Every command tab streams stdout into the log pane. You care about:

- Lines like `Scanning: /path` or `Matching scene 3/47…` — progress.
  Keep watching, things are happening.
- `[exit=0  elapsed=N.Ns]` at the end — success. **Proceed** to the
  next tab in the pipeline.
- `[exit=1]` or any non-zero exit — failure. The lines above explain
  why. The three most common reasons:
  - `Directory not found` — typo, or the path needs `/` instead of `\`.
  - `No video files found` — nothing in the folder, or the extensions
    are ones Klippbok doesn't support.
  - `ImportError` / `CUDA out of memory` — GPU or environment issue
    (rare in Pinokio). Try re-running; if persistent, click **Reset**
    in Pinokio and re-install.
- Mid-run `[WARN]` — informational, the run continues.
- Mid-run `[ERROR]` — one item failed but the command may still
  succeed. Check the final summary.

### When to proceed vs re-run

- **Scan** — proceed as long as you have more than zero usable clips.
- **Triage** — **do not** proceed past this without reviewing in the
  Manifest Reviewer. CLIP gets things wrong.
- **Ingest** — open your output folder. If there are clips in it,
  proceed. If it's empty, check the log for errors and fix.
- **Caption** — open a few `.txt` files. Do they make sense? Is the
  anchor word in there? If no, re-run with `--overwrite` and tweak
  use-case or provider.
- **Validate** — fix everything labeled `ERROR`. Warnings are
  judgment calls (a few low-res clips in a 200-clip set is probably
  fine).
- **Organize** — final step. If it succeeds, you're done.

### Common first-timer mistakes

- **Empty `concepts/` folder, then running Triage.** The Triage tab
  refuses to launch with a clear message pointing here. Populate
  concepts first.
- **Using video files as concept references.** Concepts holds
  *images*. Grab stills from your best clips.
- **Skipping the Manifest Reviewer.** Don't. It's the whole reason
  this UI exists.
- **Training on a dataset without Validate passing.** You'll waste
  GPU time debugging things Klippbok would have caught.
- **Forgetting to set `--anchor-word`.** Without it, the model has
  no stable token to learn — prompting it later won't work well.
"""


# ----- .env + install-check helpers ---------------------------------------


_RELEVANT_ENV_KEYS = ("GEMINI_API_KEY", "REPLICATE_API_TOKEN")


def _env_path() -> Path:
    """Location of the .env file — repo root, sibling to pinokio.js."""
    return Path(__file__).resolve().parent.parent / ".env"


def _load_env() -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Returns {} if absent or empty."""
    path = _env_path()
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        # CLAUDE-NOTE: Strip surrounding quotes (single or double). Doesn't
        # handle escaped quotes inside values — that's fine for API tokens,
        # which are opaque alphanumeric strings without shell specials.
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        result[k.strip()] = v
    return result


def _save_env(keys: dict[str, str]) -> str:
    """Write `keys` into .env, preserving any unrelated keys already there.

    Returns the absolute path written to (as a string) for display. Empty
    values in `keys` are skipped rather than deleting an existing entry —
    that way clearing a textbox doesn't nuke a key you meant to keep.
    """
    path = _env_path()
    merged = _load_env() if path.exists() else {}
    for k, v in keys.items():
        if v:
            merged[k] = v
    lines = ["# Klippbok API keys — managed by the Settings tab in the UI."]
    for k, v in sorted(merged.items()):
        if v:
            lines.append(f'{k}="{v}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _check_installation() -> str:
    """Run the klippbok + ffmpeg + Python version checks and format a report."""
    lines: list[str] = []
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import klippbok; print(getattr(klippbok, '__version__', '?'))",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            lines.append(f"[ok]klippbok {r.stdout.strip()}")
        else:
            stderr_tail = (r.stderr or "").strip().splitlines()[-1:] or ["import failed"]
            lines.append(f"[!!]klippbok: {stderr_tail[0]}")
    except Exception as exc:  # noqa: BLE001 — we want every failure surfaced in the UI
        lines.append(f"[!!]klippbok check failed: {exc}")

    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            first = r.stdout.splitlines()[0]
            lines.append(f"[ok]{first}")
        else:
            lines.append("[!!]ffmpeg on PATH but returned non-zero")
    except FileNotFoundError:
        lines.append(
            "[!!]ffmpeg NOT on PATH — install it "
            "(https://www.ffmpeg.org/download.html)"
        )
    except subprocess.TimeoutExpired:
        lines.append("[!!]ffmpeg check timed out")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"[!!]ffmpeg check failed: {exc}")

    lines.append(f"->Python {sys.version.split()[0]} at {sys.executable}")
    env_p = _env_path()
    if env_p.exists():
        lines.append(f"->.env found: {env_p}")
    else:
        lines.append(f"->.env not yet created (would be written to {env_p})")
    return "\n".join(lines)


# ----- filesystem helpers --------------------------------------------------


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _scaffold_project(parent: str, name: str) -> tuple[str, str, str, str]:
    """Create `<parent>/<name>/{clips,concepts,output}` + a concepts README.

    Returns `(message, clips_path, concepts_path, output_path)`. Paths are
    empty strings on error. The operation is idempotent — running it twice
    on the same project name won't clobber anything.
    """
    if not parent or not name:
        return ("[error] Fill both Parent directory and Project name.", "", "", "")
    name = name.strip()
    if any(c in name for c in '/\\:*?"<>|'):
        return ("[error] Project name contains illegal characters.", "", "", "")

    base = Path(parent) / name
    try:
        clips = base / "clips"
        concepts = base / "concepts"
        output = base / "output"
        for p in (clips, concepts, output):
            p.mkdir(parents=True, exist_ok=True)
        readme = concepts / "README.txt"
        if not readme.exists():
            readme.write_text(CONCEPTS_README, encoding="utf-8")
    except Exception as exc:
        return (f"[error] Could not create folders: {exc}", "", "", "")

    lines = [
        f"Created {base}",
        f"  clips/     - put raw videos here (or leave empty if already elsewhere)",
        f"  concepts/  - add reference image subfolders (see README inside)",
        f"  output/    - Klippbok writes processed clips + manifests here",
        "",
        "Open the concepts folder now and create subfolders for each concept.",
        "See the Directions tab for what to put inside.",
    ]
    return ("\n".join(lines), str(clips), str(concepts), str(output))


def _detect_triage_manifest(directory: str, output_override: str) -> str:
    """Best-effort detection of the JSON file Triage just wrote.

    Returns the absolute path as a string, or "" if nothing was found.
    Checked in priority order:
      1. An explicit --output path (if the user set one and it exists).
      2. `<directory>/scene_triage_manifest.json` (long-video mode).
      3. `<directory>/triage_manifest.json` (clip-level mode).
    If both scene and clip files exist, picks whichever is newer — that's the
    one Triage just wrote. The Reviewer tab reads this to auto-populate.
    """
    if output_override:
        p = Path(output_override)
        if p.exists() and p.is_file():
            return str(p)
    if not directory:
        return ""
    dir_p = Path(directory)
    if not dir_p.is_dir():
        return ""
    candidates = [
        dir_p / "scene_triage_manifest.json",
        dir_p / "triage_manifest.json",
    ]
    existing = [c for c in candidates if c.exists() and c.is_file()]
    if not existing:
        return ""
    return str(max(existing, key=lambda p: p.stat().st_mtime))


def _check_concepts_dir(path: str) -> str | None:
    """Return an error string if the concepts dir isn't usable for Triage, else None."""
    if not path:
        return "Concepts folder is required. Set one in the Project tab."
    p = Path(path)
    if not p.exists():
        return f"Concepts folder does not exist: {path}"
    if not p.is_dir():
        return f"Concepts path is not a directory: {path}"
    # CLAUDE-NOTE: Klippbok's triage expects subfolders under concepts/, one
    # per concept. Dotfiles and underscore-prefixed entries are ignored by
    # common conventions — filter them so a README.txt next to real concept
    # folders doesn't make this check pass unintentionally.
    subdirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
    if not subdirs:
        return (
            f"Concepts folder has no concept subfolders. "
            f"Create subfolders inside `{path}` — one per concept "
            f"(character / style / motion / object) — and put 5-20 reference "
            f"images (jpg/png) in each. See the Directions tab for details."
        )
    for d in subdirs:
        try:
            if any(f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir() if f.is_file()):
                return None
        except OSError:
            continue
    return (
        f"Concepts folder has subfolders but no reference images in them. "
        f"Add 5-20 jpg/png reference images to each concept subfolder. "
        f"See the Directions tab."
    )


# ----- shared helpers ------------------------------------------------------


def _pick_folder() -> str:
    """Open a native folder dialog server-side. Empty string if unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory()
        root.destroy()
        return path or ""
    except Exception:
        return ""


def _folder_row(label: str, hint: str) -> gr.Textbox:
    with gr.Row():
        box = gr.Textbox(label=label, info=hint, scale=4, interactive=True)
        btn = gr.Button("Browse…", scale=0, size="sm", min_width=0)
        btn.click(_pick_folder, outputs=box)
    return box


def _command_shell(tab_id: str, title: str, description: str, dir_label: str) -> dict:
    """Build the invariant parts of every command tab.

    Returns a dict of components so each tab can wire its per-command Options
    into the same skeleton without copy-pasting the log pane + button row.
    """
    gr.Markdown(f"### `{title}`\n{description}")
    directory = gr.Textbox(
        label=dir_label,
        info="Defaults to the matching path from the Project tab — override here if needed.",
        interactive=True,
    )
    # Options accordion is appended to by the caller after this function returns.
    options = gr.Accordion("Options", open=True)
    with gr.Row():
        dry_run = gr.Checkbox(label="Dry run (print command only)", value=False, scale=0)
        run_btn = gr.Button("Run", variant="primary", scale=0)
        cancel_btn = gr.Button("Cancel", scale=0)
    log = gr.Textbox(
        label="Output",
        lines=20,
        max_lines=2000,
        autoscroll=True,
        interactive=False,
        show_copy_button=True,
    )
    cancel_btn.click(lambda: runner.cancel(tab_id), outputs=log)
    return {
        "directory": directory,
        "options": options,
        "dry_run": dry_run,
        "run_btn": run_btn,
        "cancel_btn": cancel_btn,
        "log": log,
    }


def _sync(source: gr.Textbox, target: gr.Textbox) -> None:
    """Push source's value into target unless target already has a value.

    CLAUDE-NOTE: Originally this was unconditional (overwrote target on every
    source change), but that made per-tab directory overrides non-persistent
    across project reloads — loading a saved project sets work_dir, which
    fired this handler, which re-clobbered the just-loaded tab directories.
    Now _sync only fills empty targets, so loaded/custom per-tab directories
    survive subsequent Project-tab edits.
    """
    def _fill_if_empty(source_val, target_val):
        return source_val if not target_val else target_val
    source.change(_fill_if_empty, inputs=[source, target], outputs=target)


# CLAUDE-NOTE: Cross-tab path propagation. Each output-producing tab pushes
# its newly-written path into a gr.State; consumer tabs wire a .change()
# handler on that state to fill their own input field unless the user has
# already typed something different. Persistence lives in
# .user_settings.json[detected_paths] (via pipeline_installer), so values
# survive app restarts. Keys kept here so producer + consumer sides agree.
_DETECTED_KEY_TRIAGE_MANIFEST = "last_triage_manifest"
_DETECTED_KEY_INGEST_OUTPUT = "last_ingest_output"
_DETECTED_KEY_CAPTION_DIR = "last_caption_dir"


def _autofill_handler(new_value: str, current: str, last_auto: str) -> tuple:
    """Propagate new_value into a field unless the user has edited it.

    Returns (gr.update_for_field, new_tracker_value). Fills only when the
    field is empty or still holds our prior auto-fill — manual edits are
    preserved forever (the tracker stays pinned to the old auto-fill, so
    current != last_auto stays true on subsequent state changes).
    """
    if not new_value:
        return gr.update(), last_auto
    if current and current != last_auto:
        return gr.update(), last_auto
    return gr.update(value=new_value), new_value


def _wire_autofill(source_state: gr.State, target: gr.Textbox, tracker: gr.State) -> None:
    """Wire source_state.change() to propagate into target via tracker."""
    source_state.change(
        _autofill_handler,
        inputs=[source_state, target, tracker],
        outputs=[target, tracker],
    )


def _persist_detected(key: str, value: str) -> None:
    """Best-effort merge of a single detected_paths entry."""
    if not value:
        return
    try:
        pi.update_detected_path(key, value)
    except Exception:
        pass  # UI keeps working even if disk is read-only.


def _load_detected_paths() -> dict:
    """Best-effort read of detected_paths from .user_settings.json."""
    try:
        return pi.load_user_settings().get("detected_paths", {}) or {}
    except Exception:
        return {}


def _base_cmd(*parts: str) -> list[str]:
    return [runner.python_executable(), "-m", *parts]


def _stream(tab_id: str, cmd: list[str], extra_env: dict | None = None) -> Iterator[str]:
    """Run a command, accumulating output into a string yielded incrementally."""
    buf = ""
    for line in runner.stream_command(tab_id, cmd, extra_env=extra_env):
        buf += line + "\n"
        yield buf


def _dry_preview(cmd: list[str]) -> str:
    return "$ " + runner.format_command(cmd) + "\n[dry run — not executed]"


# ----- Directions + Project tabs -----------------------------------------


def _tab_directions() -> None:
    with gr.Tab("Directions"):
        gr.Markdown(DIRECTIONS_MD)


def _tab_project() -> dict:
    """Project tab with the shared directory rows + scaffold + save/load UI.

    Returns a dict of components because there are now several downstream
    consumers (build_ui wires save/load, the Load Project button feeds a
    file picker, etc.) — returning a tuple would make the call sites
    fragile to reorder.
    """
    with gr.Tab("Project"):
        # CLAUDE-NOTE: Project status lives at the very top of the tab so
        # users see which project is loaded and the last-save timestamp
        # without scrolling. Populated by the save/load wiring in build_ui.
        project_status = gr.Markdown(ps_mod.format_status("", "", []))

        # CLAUDE-NOTE: Target-trainer preset selector. Pure UI sugar —
        # picking a preset emits gr.update for default values across
        # Scan / Ingest / Normalize / Caption / Validate (wiring in
        # _wire_project_persistence). Klippbok's CLI itself stays
        # preset-agnostic.
        with gr.Row():
            trainer_preset = gr.Dropdown(
                label="Target trainer",
                choices=list(TRAINER_PRESETS.keys()),
                value=DEFAULT_TRAINER_PRESET,
                interactive=True,
                scale=2,
                info=(
                    "Sets default fps, max-frames, and frame-count rule "
                    "for the selected trainer. You can still override "
                    "individual settings per-tab."
                ),
            )
        trainer_preset_info = gr.Markdown(
            TRAINER_PRESETS[DEFAULT_TRAINER_PRESET]["guidance"]
        )

        with gr.Row():
            load_project_btn = gr.Button("Load Project", scale=0, size="sm")
            new_project_btn = gr.Button("New Project", scale=0, size="sm")
            recent_dropdown = gr.Dropdown(
                label="Recent projects",
                choices=[],
                value=None,
                interactive=True,
                scale=3,
                info="Pick a recent project directory to switch instantly.",
            )

        gr.Markdown(
            "### Working directories\n"
            "Set these once — every command tab starts with the matching path"
            " as its default, and you can override per-tab as needed.\n\n"
            "**Concepts is something you populate by hand** — see the"
            " **Directions** tab if you're not sure what that means.\n\n"
            "**Auto-save:** every field change is written to"
            " `klippbok_project.json` inside your working directory. Close the"
            " app and reopen — you'll land right back where you left off."
        )
        work_dir = _folder_row(
            "Working directory",
            "Your raw video clips, or a dir of pre-split clips. Also where klippbok_project.json is saved.",
        )
        concepts_dir = _folder_row(
            "Concepts directory",
            "Reference image subfolders — populated by YOU. See Directions tab.",
        )
        output_dir = _folder_row(
            "Output directory",
            "Where Klippbok writes processed clips + manifests.",
        )

        with gr.Accordion("Scaffold a new project layout", open=False):
            gr.Markdown(
                "Don't have the folder structure yet? Pick a parent location and"
                " a project name. Klippbok will create `clips/`, `concepts/`, and"
                " `output/` underneath, drop a README inside `concepts/`"
                " explaining what to put there, and auto-fill the three path"
                " boxes above."
            )
            with gr.Row():
                parent = gr.Textbox(label="Parent directory", scale=4)
                parent_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
                parent_browse.click(_pick_folder, outputs=parent)
            name = gr.Textbox(
                label="Project name",
                value="MyKlippbokProject",
                info="Subfolder created under the parent directory. Use plain characters.",
            )
            scaffold_btn = gr.Button("Create project folders", variant="primary")
            scaffold_log = gr.Code(label="Result", language=None, value="", interactive=False, visible=False)

            def _do_scaffold(parent_v, name_v):
                msg, clips, concepts, output = _scaffold_project(parent_v, name_v)
                # CLAUDE-NOTE: gr.update() (no args) means "don't change this
                # output" — on error we leave the three path textboxes alone.
                return (
                    gr.update(value=msg, visible=True),
                    clips if clips else gr.update(),
                    concepts if concepts else gr.update(),
                    output if output else gr.update(),
                )

            scaffold_btn.click(
                _do_scaffold,
                inputs=[parent, name],
                outputs=[scaffold_log, work_dir, concepts_dir, output_dir],
            )

    return {
        "work_dir": work_dir,
        "concepts_dir": concepts_dir,
        "output_dir": output_dir,
        "project_status": project_status,
        "load_project_btn": load_project_btn,
        "new_project_btn": new_project_btn,
        "recent_dropdown": recent_dropdown,
        "trainer_preset": trainer_preset,
        "trainer_preset_info": trainer_preset_info,
    }


# ----- per-command tab builders -------------------------------------------


def _tab_scan(work_dir, _concepts, _output, _api, ps: "ps_mod.ProjectState") -> None:
    tab_id = "scan"
    with gr.Tab("Scan"):
        s = _command_shell(
            tab_id,
            "klippbok.video scan",
            "Probe a directory of video clips for resolution, fps, frame count, and codec issues. Read-only.",
            "Directory (positional)",
        )
        _sync(work_dir, s["directory"])
        with s["options"]:
            with gr.Row():
                fps = gr.Number(label="--fps", value=16, precision=0, info="Target frame rate. Default 16 (Wan).")
                verbose = gr.Checkbox(label="--verbose", value=False, info="Per-clip details instead of grouped summary.")
            config = gr.Textbox(label="--config (optional)", info="Path to klippbok_data.yaml.", value="")

        ps.register(
            tab_id,
            directory=s["directory"], dry_run=s["dry_run"],
            config=config, fps=fps, verbose=verbose,
        )
        ps.register_run(tab_id, s["run_btn"])

        def _run(directory, config, fps, verbose, dry):
            if not directory:
                yield "[error] Directory is required."
                return
            cmd = _base_cmd("klippbok.video", "scan", directory)
            if config:
                cmd += ["--config", config]
            if fps:
                cmd += ["--fps", str(int(fps))]
            if verbose:
                cmd.append("--verbose")
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(_run, inputs=[s["directory"], config, fps, verbose, s["dry_run"]], outputs=s["log"])


def _tab_triage(work_dir, concepts_dir, _output, _api, last_manifest_state: gr.State, ps: "ps_mod.ProjectState") -> None:
    tab_id = "triage"
    with gr.Tab("Triage"):
        s = _command_shell(
            tab_id,
            "klippbok.video triage",
            "CLIP-based visual matching of clips against reference images. Writes `triage_manifest.json` (or `scene_triage_manifest.json` for long videos) which downstream stages consume.",
            "Directory (positional) — clips to triage",
        )
        _sync(work_dir, s["directory"])
        with s["options"]:
            with gr.Row():
                concepts = gr.Textbox(label="--concepts (required)", info="Folder of reference images with concept subfolders.", scale=3)
                concepts_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
                concepts_browse.click(_pick_folder, outputs=concepts)
            _sync(concepts_dir, concepts)
            with gr.Row():
                threshold = gr.Number(label="--threshold", value=0.70, precision=2, info="Similarity threshold 0.0–1.0.")
                frames = gr.Number(label="--frames", value=5, precision=0, info="Frames sampled per clip.")
                frames_per_scene = gr.Number(label="--frames-per-scene", value=2, precision=0, info="Frames sampled per scene (long videos).")
                scene_threshold = gr.Number(label="--scene-threshold", value=27.0, precision=1, info="Scene detection sensitivity.")
            output = gr.Textbox(label="--output (optional)", info="Custom path for manifest JSON.", value="")
            with gr.Row():
                organize = gr.Textbox(label="--organize DIR (optional)", info="If set, copy/move matched clips into concept-named subfolders here.", scale=3)
                move = gr.Checkbox(label="--move", value=False, info="Move instead of copy when --organize is set.", scale=0)
            clip_model = gr.Textbox(label="--clip-model", value="openai/clip-vit-base-patch32", info="CLIP model identifier.")

        ps.register(
            "triage",
            directory=s["directory"], dry_run=s["dry_run"],
            concepts=concepts, threshold=threshold, frames=frames,
            frames_per_scene=frames_per_scene, scene_threshold=scene_threshold,
            output=output, organize=organize, move=move, clip_model=clip_model,
        )
        ps.register_run("triage", s["run_btn"])

        def _run(directory, concepts, threshold, frames, frames_per_scene, scene_threshold, output, organize, move, clip_model, dry):
            if not directory:
                yield "[error] Directory is required."
                return
            err = _check_concepts_dir(concepts)
            if err:
                yield f"[error] {err}"
                return
            cmd = _base_cmd("klippbok.video", "triage", directory, "--concepts", concepts)
            if threshold is not None:
                cmd += ["--threshold", str(float(threshold))]
            if frames:
                cmd += ["--frames", str(int(frames))]
            if frames_per_scene:
                cmd += ["--frames-per-scene", str(int(frames_per_scene))]
            if scene_threshold is not None:
                cmd += ["--scene-threshold", str(float(scene_threshold))]
            if output:
                cmd += ["--output", output]
            if organize:
                cmd += ["--organize", organize]
                if move:
                    cmd.append("--move")
            if clip_model:
                cmd += ["--clip-model", clip_model]
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        def _after_run(directory: str, output_override: str, dry: bool, current: str, log: str):
            # CLAUDE-NOTE: Don't update state on dry runs — Klippbok wasn't
            # actually invoked so nothing was written. Keep whatever was there.
            if dry:
                return current, log
            detected = _detect_triage_manifest(directory, output_override)
            if not detected:
                return current, log
            _persist_detected(_DETECTED_KEY_TRIAGE_MANIFEST, detected)
            notice = (
                f"\n[propagated] Manifest saved to {detected} — "
                f"auto-loaded into Ingest and Manifest Reviewer tabs."
            )
            return detected, (log or "") + notice

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], concepts, threshold, frames, frames_per_scene, scene_threshold, output, organize, move, clip_model, s["dry_run"]],
            outputs=s["log"],
        ).then(
            _after_run,
            inputs=[s["directory"], output, s["dry_run"], last_manifest_state, s["log"]],
            outputs=[last_manifest_state, s["log"]],
        )


def _tab_ingest(
    work_dir,
    _concepts,
    output_dir,
    _api,
    last_manifest_state: gr.State,
    last_ingest_output_state: gr.State,
    ps: "ps_mod.ProjectState",
) -> None:
    tab_id = "ingest"
    with gr.Tab("Ingest"):
        s = _command_shell(
            tab_id,
            "klippbok.video ingest",
            "Scene-detect, split, and normalize a raw video (or directory of videos) into training clips. Accepts a triage manifest to filter which scenes to keep.",
            "Video (positional) — file or directory",
        )
        _sync(work_dir, s["directory"])
        with s["options"]:
            with gr.Row():
                output = gr.Textbox(label="--output (required)", info="Output directory for split clips.", scale=3)
                output_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
                output_browse.click(_pick_folder, outputs=output)
            _sync(output_dir, output)
            with gr.Row():
                threshold = gr.Number(label="--threshold", value=27.0, precision=1, info="Scene detection threshold.")
                max_frames = gr.Number(label="--max-frames", value=81, precision=0, info="Max frames per clip (0 = no limit). 81 ≈ 5s @ 16fps.")
            config = gr.Textbox(label="--config (optional)", info="Path to klippbok_data.yaml.", value="")
            triage = gr.Textbox(label="--triage MANIFEST (optional)", info="scene_triage_manifest.json — only split scenes marked include:true.", value="")
            with gr.Row():
                caption = gr.Checkbox(label="--caption", value=False, info="Auto-caption each clip after splitting.", scale=0)
                provider = gr.Dropdown(label="--provider (if --caption)", choices=["gemini", "replicate", "openai"], value="gemini", scale=0)

        # CLAUDE-NOTE: Cross-tab auto-fill. `--triage` picks up the manifest
        # Triage just wrote; a tracker state prevents clobbering user edits.
        triage_tracker = gr.State("")
        _wire_autofill(last_manifest_state, triage, triage_tracker)

        ps.register(
            "ingest",
            directory=s["directory"], dry_run=s["dry_run"],
            output=output, config=config, threshold=threshold,
            max_frames=max_frames, triage=triage, caption=caption, provider=provider,
        )
        ps.register_run("ingest", s["run_btn"])

        def _run(video, output, config, threshold, max_frames, triage, caption, provider, dry, api):
            if not video:
                yield "[error] Video path is required."
                return
            if not output:
                yield "[error] --output is required."
                return
            cmd = _base_cmd("klippbok.video", "ingest", video, "--output", output)
            if config:
                cmd += ["--config", config]
            if threshold is not None:
                cmd += ["--threshold", str(float(threshold))]
            if max_frames is not None:
                cmd += ["--max-frames", str(int(max_frames))]
            if triage:
                cmd += ["--triage", triage]
            if caption:
                cmd.append("--caption")
                if provider:
                    cmd += ["--provider", provider]
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd, extra_env=api)

        def _after_run(output_val: str, dry: bool, current: str, log: str):
            # CLAUDE-NOTE: Publish --output to last_ingest_output_state so
            # Normalize/Caption/Extract/Validate auto-fill their directory.
            # Skip dry runs (no write actually happened). We don't inspect
            # the log for success markers — treating any non-dry run as a
            # write is accurate enough, and matches how Triage's _after_run
            # behaves (it just checks disk for the manifest file).
            if dry or not output_val:
                return current, log
            _persist_detected(_DETECTED_KEY_INGEST_OUTPUT, output_val)
            notice = (
                f"\n[propagated] Clips written to {output_val} — "
                f"auto-loaded into Normalize, Caption, Extract, and Validate tabs."
            )
            return output_val, (log or "") + notice

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], output, config, threshold, max_frames, triage, caption, provider, s["dry_run"], _api],
            outputs=s["log"],
        ).then(
            _after_run,
            inputs=[output, s["dry_run"], last_ingest_output_state, s["log"]],
            outputs=[last_ingest_output_state, s["log"]],
        )


def _tab_normalize(work_dir, _concepts, output_dir, _api, last_ingest_output_state: gr.State, ps: "ps_mod.ProjectState") -> None:
    tab_id = "normalize"
    with gr.Tab("Normalize"):
        s = _command_shell(
            tab_id,
            "klippbok.video normalize",
            "Batch-fix fps, resolution, and frame count on pre-split clips to match the Klippbok training targets.",
            "Directory (positional) — clips to normalize",
        )
        _sync(work_dir, s["directory"])
        dir_tracker = gr.State("")
        _wire_autofill(last_ingest_output_state, s["directory"], dir_tracker)
        with s["options"]:
            with gr.Row():
                output = gr.Textbox(label="--output (required)", info="Where normalized clips go.", scale=3)
                output_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
                output_browse.click(_pick_folder, outputs=output)
            _sync(output_dir, output)
            with gr.Row():
                fps = gr.Number(label="--fps", value=16, precision=0, info="Target frame rate.")
                fmt = gr.Dropdown(label="--format", choices=["(source)", ".mp4", ".mov", ".mkv"], value="(source)", info="Force output container.")
            config = gr.Textbox(label="--config (optional)", info="Path to klippbok_data.yaml.", value="")

        ps.register(
            "normalize",
            directory=s["directory"], dry_run=s["dry_run"],
            output=output, fps=fps, fmt=fmt, config=config,
        )
        ps.register_run("normalize", s["run_btn"])

        def _run(directory, output, fps, fmt, config, dry):
            if not directory:
                yield "[error] Directory is required."
                return
            if not output:
                yield "[error] --output is required."
                return
            cmd = _base_cmd("klippbok.video", "normalize", directory, "--output", output)
            if fps:
                cmd += ["--fps", str(int(fps))]
            if fmt and fmt != "(source)":
                cmd += ["--format", fmt]
            if config:
                cmd += ["--config", config]
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(_run, inputs=[s["directory"], output, fps, fmt, config, s["dry_run"]], outputs=s["log"])


def _tab_caption(
    work_dir,
    _concepts,
    _output,
    api_keys,
    last_ingest_output_state: gr.State,
    last_caption_dir_state: gr.State,
    ps: "ps_mod.ProjectState",
) -> None:
    tab_id = "caption"
    with gr.Tab("Caption"):
        s = _command_shell(
            tab_id,
            "klippbok.video caption",
            "Generate `.txt` sidecar captions for each clip (videos AND still images) using a vision-language model. Videos go through `klippbok.video caption`; still images (.png/.jpg/.jpeg/.webp) are sent directly to the same VLM via an in-process shim so image-model LoRA datasets (FLUX.2, Z-Image, Qwen-Image) work in the same Run. API keys live in the Settings tab.",
            "Directory (positional) — clips or images to caption",
        )
        _sync(work_dir, s["directory"])
        dir_tracker = gr.State("")
        _wire_autofill(last_ingest_output_state, s["directory"], dir_tracker)
        with s["options"]:
            with gr.Row():
                provider = gr.Dropdown(label="--provider", choices=["gemini", "replicate", "openai"], value="gemini", info="VLM backend.")
                use_case = gr.Dropdown(label="--use-case", choices=["(auto)", "character", "style", "motion", "object"], value="(auto)", info="Prompt template.")
            with gr.Row():
                anchor = gr.Textbox(label="--anchor-word", info="Prefix prepended to every caption (e.g. character name).", value="")
                caption_fps = gr.Number(label="--caption-fps", value=1, precision=0, info="Frame sampling rate for captioning.")
            tags = gr.Textbox(label="--tags (space-separated)", info="Secondary anchor tags the model should mention when relevant.", value="")
            overwrite = gr.Checkbox(label="--overwrite", value=False, info="Replace existing .txt captions.")
            with gr.Accordion("OpenAI-compatible / Ollama overrides", open=False):
                base_url = gr.Textbox(label="--base-url", value="", info="e.g. http://localhost:11434/v1 for local Ollama.")
                model = gr.Textbox(label="--model", value="", info="e.g. llama3.2-vision.")
            # CLAUDE-NOTE: Image-only knobs — retry_delay only applies to
            # the in-process image-captioning shim (Klippbok's video CLI
            # has its own retry logic baked in). Surfaced here so free-
            # tier users can tune for their provider's quota window.
            with gr.Accordion("Image captioning (PNG/JPG) — rate-limit handling", open=False):
                gr.Markdown(
                    "When the target directory contains PNG / JPG / WEBP "
                    "files, they're captioned in-process via Klippbok's VLM "
                    "backends (video files still go through the "
                    "`klippbok.video caption` CLI). A `.txt` sidecar next to "
                    "each image is the completion marker — **re-running "
                    "Caption just picks up where it left off**, so swapping "
                    "a rate-limited API key in Settings and re-running is "
                    "the supported recovery flow."
                )
                with gr.Row():
                    retry_delay = gr.Number(
                        label="--retry-delay (seconds)",
                        value=60,
                        precision=0,
                        info="Wait this long before retrying after a 429 / quota error.",
                    )
                    max_retries = gr.Number(
                        label="--max-retries",
                        value=3,
                        precision=0,
                        info="Give up after this many retries on the same image.",
                    )

        ps.register(
            "caption",
            directory=s["directory"], dry_run=s["dry_run"],
            provider=provider, use_case=use_case, anchor=anchor,
            caption_fps=caption_fps, tags=tags, overwrite=overwrite,
            base_url=base_url, model=model,
            retry_delay=retry_delay, max_retries=max_retries,
        )
        ps.register_run("caption", s["run_btn"])

        def _run(directory, provider, use_case, anchor, caption_fps, tags, overwrite, base_url, model, retry_delay, max_retries, dry, api):
            if not directory:
                yield "[error] Directory is required."
                return

            # CLAUDE-NOTE: Klippbok's `caption` CLI iterates video files
            # only. Image-model LoRAs (FLUX.2 etc.) train on still frames,
            # so we split the directory here: images go through the
            # in-process shim (caption_images_mod), videos go through the
            # existing subprocess CLI path. Both stream into the same log.
            from pathlib import Path as _P
            dir_p = _P(directory)
            images = caption_images_mod.find_images(dir_p)
            videos = caption_images_mod.find_videos(dir_p)
            if not images and not videos:
                yield f"[error] No images or videos found in {directory}"
                return

            # Build the video CLI cmd once (used for dry preview + run).
            cmd = _base_cmd("klippbok.video", "caption", directory)
            if provider:
                cmd += ["--provider", provider]
            if use_case and use_case != "(auto)":
                cmd += ["--use-case", use_case]
            if anchor:
                cmd += ["--anchor-word", anchor]
            if caption_fps:
                cmd += ["--caption-fps", str(int(caption_fps))]
            if tags:
                cmd += ["--tags", *tags.split()]
            if overwrite:
                cmd.append("--overwrite")
            if base_url:
                cmd += ["--base-url", base_url]
            if model:
                cmd += ["--model", model]

            if dry:
                preview = []
                if images:
                    preview.append(
                        f"[image-caption] Would caption {len(images)} image(s) "
                        f"via {provider or 'gemini'} (in-process shim)"
                    )
                if videos:
                    preview.append("$ " + runner.format_command(cmd))
                preview.append("[dry run — not executed]")
                yield "\n".join(preview)
                return

            # Accumulating log buffer — Gradio replaces the textbox on each
            # yield, so every emit must be the full history to date.
            buf = ""

            if images:
                for line in caption_images_mod.caption_images(
                    directory,
                    provider=provider or "gemini",
                    use_case=None if (not use_case or use_case == "(auto)") else use_case,
                    anchor_word=anchor or None,
                    tags=tags.split() if tags else None,
                    overwrite=bool(overwrite),
                    base_url=base_url or "",
                    model=model or "",
                    caption_fps=int(caption_fps) if caption_fps else 1,
                    retry_delay=int(retry_delay) if retry_delay else caption_images_mod.DEFAULT_RETRY_DELAY,
                    max_retries=int(max_retries) if max_retries else caption_images_mod.DEFAULT_MAX_RETRIES,
                    extra_env=api,
                ):
                    buf += line + "\n"
                    yield buf

            if videos:
                if images:
                    buf += "\n"
                    yield buf
                for line in runner.stream_command(tab_id, cmd, extra_env=api):
                    buf += line + "\n"
                    yield buf
            else:
                # Images-only case — skip the subprocess entirely.
                buf += "[image-caption] no video files in directory — skipping klippbok.video caption CLI\n"
                yield buf

        def _after_run(directory: str, dry: bool, current: str, log: str):
            # CLAUDE-NOTE: Caption writes .txt sidecars in-place, so the
            # "caption output" downstream consumers want IS the input
            # directory. Publishing that here lets Score/Audit auto-fill.
            if dry or not directory:
                return current, log
            _persist_detected(_DETECTED_KEY_CAPTION_DIR, directory)
            notice = (
                f"\n[propagated] Captions written in {directory} — "
                f"auto-loaded into Score and Audit tabs."
            )
            return directory, (log or "") + notice

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], provider, use_case, anchor, caption_fps, tags, overwrite, base_url, model, retry_delay, max_retries, s["dry_run"], api_keys],
            outputs=s["log"],
        ).then(
            _after_run,
            inputs=[s["directory"], s["dry_run"], last_caption_dir_state, s["log"]],
            outputs=[last_caption_dir_state, s["log"]],
        )


def _tab_score(work_dir, _concepts, _output, _api, last_caption_dir_state: gr.State, ps: "ps_mod.ProjectState") -> None:
    tab_id = "score"
    with gr.Tab("Score"):
        s = _command_shell(
            tab_id,
            "klippbok.video score",
            "Local heuristic scoring of existing `.txt` caption files — no API calls.",
            "Directory (positional) — caption .txt files",
        )
        _sync(work_dir, s["directory"])
        dir_tracker = gr.State("")
        _wire_autofill(last_caption_dir_state, s["directory"], dir_tracker)

        ps.register("score", directory=s["directory"], dry_run=s["dry_run"])
        ps.register_run("score", s["run_btn"])

        def _run(directory, dry):
            if not directory:
                yield "[error] Directory is required."
                return
            cmd = _base_cmd("klippbok.video", "score", directory)
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(_run, inputs=[s["directory"], s["dry_run"]], outputs=s["log"])


def _tab_extract(work_dir, _concepts, output_dir, _api, last_ingest_output_state: gr.State, ps: "ps_mod.ProjectState") -> None:
    tab_id = "extract"
    with gr.Tab("Extract"):
        s = _command_shell(
            tab_id,
            "klippbok.video extract",
            "Export reference frames as PNG from a directory of clips (or still images). Use the `--template` field to write a selections template JSON you can edit, then feed back via `--selections`.",
            "Directory (positional) — clips or images",
        )
        _sync(work_dir, s["directory"])
        dir_tracker = gr.State("")
        _wire_autofill(last_ingest_output_state, s["directory"], dir_tracker)
        with s["options"]:
            with gr.Row():
                output = gr.Textbox(label="--output", info="Where PNG references are written.", scale=3)
                output_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
                output_browse.click(_pick_folder, outputs=output)
            _sync(output_dir, output)
            with gr.Row():
                strategy = gr.Dropdown(label="--strategy", choices=["first_frame", "best_frame"], value="first_frame")
                samples = gr.Number(label="--samples", value=10, precision=0, info="Frames to sample for best_frame.")
                overwrite = gr.Checkbox(label="--overwrite", value=False)
            selections = gr.Textbox(label="--selections (JSON path)", info="Custom selections manifest to apply.", value="")
            # CLAUDE-NOTE: --template takes a path, not a boolean. Klippbok writes
            # a template selections JSON to this path and exits without extracting.
            template = gr.Textbox(label="--template (JSON path)", info="Write a selection template to this path (no extraction). Mutually exclusive with --selections.", value="")

        ps.register(
            "extract",
            directory=s["directory"], dry_run=s["dry_run"],
            output=output, strategy=strategy, samples=samples,
            overwrite=overwrite, selections=selections, template=template,
        )
        ps.register_run("extract", s["run_btn"])

        def _run(directory, output, strategy, samples, overwrite, selections, template, dry):
            if not directory:
                yield "[error] Directory is required."
                return
            cmd = _base_cmd("klippbok.video", "extract", directory)
            if output:
                cmd += ["--output", output]
            if strategy:
                cmd += ["--strategy", strategy]
            if samples:
                cmd += ["--samples", str(int(samples))]
            if overwrite:
                cmd.append("--overwrite")
            if selections:
                cmd += ["--selections", selections]
            if template:
                cmd += ["--template", template]
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], output, strategy, samples, overwrite, selections, template, s["dry_run"]],
            outputs=s["log"],
        )


def _tab_audit(work_dir, _concepts, _output, api_keys, last_caption_dir_state: gr.State, ps: "ps_mod.ProjectState") -> None:
    tab_id = "audit"
    with gr.Tab("Audit"):
        s = _command_shell(
            tab_id,
            "klippbok.video audit",
            "Compare existing captions against fresh VLM output to catch low-quality or drifted annotations. Uses the same API keys as Caption.",
            "Directory (positional) — captioned clips",
        )
        _sync(work_dir, s["directory"])
        dir_tracker = gr.State("")
        _wire_autofill(last_caption_dir_state, s["directory"], dir_tracker)
        with s["options"]:
            with gr.Row():
                provider = gr.Dropdown(label="--provider", choices=["gemini", "replicate", "openai"], value="gemini")
                use_case = gr.Dropdown(label="--use-case", choices=["(auto)", "character", "style", "motion", "object"], value="(auto)")
                mode = gr.Dropdown(label="--mode", choices=["report_only", "save_audit"], value="report_only")

        ps.register(
            "audit",
            directory=s["directory"], dry_run=s["dry_run"],
            provider=provider, use_case=use_case, mode=mode,
        )
        ps.register_run("audit", s["run_btn"])

        def _run(directory, provider, use_case, mode, dry, api):
            if not directory:
                yield "[error] Directory is required."
                return
            cmd = _base_cmd("klippbok.video", "audit", directory)
            if provider:
                cmd += ["--provider", provider]
            if use_case and use_case != "(auto)":
                cmd += ["--use-case", use_case]
            if mode:
                cmd += ["--mode", mode]
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd, extra_env=api)

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], provider, use_case, mode, s["dry_run"], api_keys],
            outputs=s["log"],
        )


def _tab_validate(work_dir, _concepts, _output, _api, last_ingest_output_state: gr.State, ps: "ps_mod.ProjectState") -> None:
    tab_id = "validate"
    with gr.Tab("Validate"):
        s = _command_shell(
            tab_id,
            "klippbok.dataset validate",
            "Check dataset completeness and quality. Auto-detects whether the target is a **video**, **image**, or **mixed** dataset — image datasets (PNG/JPG frames + .txt captions) are handled by an in-process launcher shim so you don't get spurious 'no matching target video' errors. Videos still flow through `klippbok.dataset validate`.",
            "Path (positional) — dataset folder (images or videos) or klippbok_data.yaml",
        )
        _sync(work_dir, s["directory"])
        dir_tracker = gr.State("")
        _wire_autofill(last_ingest_output_state, s["directory"], dir_tracker)
        with s["options"]:
            with gr.Row():
                manifest = gr.Checkbox(label="--manifest", value=False, info="Write klippbok_manifest.json to the dataset folder.")
                buckets = gr.Checkbox(label="--buckets", value=False, info="Show training bucket preview.")
                quality = gr.Checkbox(label="--quality", value=False, info="Blur/exposure checks on reference images.")
            with gr.Row():
                duplicates = gr.Checkbox(label="--duplicates", value=False, info="Perceptual duplicate detection.")
                json_out = gr.Checkbox(label="--json", value=False, info="Emit JSON instead of formatted report.")
            config = gr.Textbox(label="--config (optional)", info="Path to klippbok_data.yaml override.", value="")
            # CLAUDE-NOTE: Frame-count rule — set automatically by the
            # Project tab's Target Trainer preset. The shim emits a
            # notice for the chosen rule so users see what the dataset
            # is being held to (enforcement still flows through Klippbok
            # for video-only dirs).
            frame_count_rule = gr.Dropdown(
                label="Frame count rule",
                choices=[
                    "off",
                    "4n+1 (Wan / HunyuanVideo)",
                    "8n+1 (LTX)",
                    "image-only (skip frame check)",
                ],
                value="4n+1 (Wan / HunyuanVideo)",
                info="Auto-set by the Project tab's Target Trainer preset.",
            )

        ps.register(
            "validate",
            directory=s["directory"], dry_run=s["dry_run"],
            manifest=manifest, buckets=buckets, quality=quality,
            duplicates=duplicates, json_out=json_out, config=config,
            frame_count_rule=frame_count_rule,
        )
        ps.register_run("validate", s["run_btn"])

        def _run(path, manifest, buckets, quality, duplicates, json_out, config, frame_count_rule, dry):
            if not path:
                yield "[error] Path is required."
                return

            # CLAUDE-NOTE: Auto-route based on dir contents. Klippbok's
            # CLI hardcodes video extensions when discovering "target"
            # files (discover.py VIDEO_EXTENSIONS), so image-only /
            # mixed datasets falsely report every file as orphaned.
            # We branch:
            #   * images-only or mixed → launcher shim handles pairing +
            #     image quality + duplicates; bucket/frame/fps checks
            #     skip (video-specific — user can re-run on a videos-
            #     only dir for those)
            #   * videos-only → Klippbok CLI unchanged
            #   * yaml config path → Klippbok CLI (config semantics
            #     aren't something the shim reproduces)
            from pathlib import Path as _P
            path_p = _P(path)
            is_yaml = path_p.is_file() and path_p.suffix.lower() in (".yaml", ".yml")
            kind = (
                "config" if is_yaml
                else validate_images_mod.classify_directory(path_p)
                if path_p.is_dir()
                else "unknown"
            )

            cmd = _base_cmd("klippbok.dataset", "validate", path)
            if manifest:
                cmd.append("--manifest")
            if buckets:
                cmd.append("--buckets")
            if quality:
                cmd.append("--quality")
            if duplicates:
                cmd.append("--duplicates")
            if json_out:
                cmd.append("--json")
            if config:
                cmd += ["--config", config]

            if dry:
                preview = [f"[validate] Auto-detected kind: {kind}"]
                if kind in ("images", "mixed"):
                    preview.append(
                        f"[validate-images] Would validate {kind} dataset "
                        f"via launcher shim (quality={bool(quality)}, "
                        f"duplicates={bool(duplicates)}, json={bool(json_out)})"
                    )
                    if buckets:
                        preview.append(
                            "[validate-images] --buckets is video-specific, "
                            "skipping for image/mixed datasets"
                        )
                if kind in ("videos", "config", "unknown"):
                    preview.append("$ " + runner.format_command(cmd))
                preview.append("[dry run — not executed]")
                yield "\n".join(preview)
                return

            if kind in ("images", "mixed"):
                if buckets:
                    yield (
                        "[validate-images] --buckets is video-specific — "
                        "skipping for image/mixed dataset."
                    )
                buf = ""
                for line in validate_images_mod.validate_directory(
                    path,
                    quality=bool(quality),
                    duplicates=bool(duplicates),
                    json_output=bool(json_out),
                    write_manifest=bool(manifest),
                    frame_count_rule=frame_count_rule or "off",
                ):
                    buf += line + "\n"
                    yield buf
                return

            # Videos-only, yaml config, or path that doesn't exist —
            # fall through to Klippbok's CLI and let it report.
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], manifest, buckets, quality, duplicates, json_out, config, frame_count_rule, s["dry_run"]],
            outputs=s["log"],
        )


def _tab_organize(work_dir, _concepts, output_dir, _api, ps: "ps_mod.ProjectState") -> None:
    tab_id = "organize"
    with gr.Tab("Organize"):
        s = _command_shell(
            tab_id,
            "klippbok.dataset organize",
            "Restructure a validated dataset into a trainer-specific layout (musubi, aitoolkit, or flat). Source is the positional path; --output is the destination.",
            "Path (positional) — source dataset folder",
        )
        _sync(work_dir, s["directory"])
        with s["options"]:
            with gr.Row():
                output = gr.Textbox(label="--output (required)", scale=3)
                output_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
                output_browse.click(_pick_folder, outputs=output)
            _sync(output_dir, output)
            with gr.Row():
                layout = gr.Dropdown(label="--layout", choices=["flat", "klippbok"], value="flat", info="Output structure.")
                # CLAUDE-NOTE: Klippbok's --trainer help string names "musubi" and
                # "aitoolkit" but doesn't use argparse choices= — it accepts any
                # string. Using a multiselect with free-add so users aren't
                # locked out if Klippbok adds trainers (e.g. kohya) later.
                trainers = gr.Dropdown(
                    label="--trainer (repeatable)",
                    choices=["musubi", "aitoolkit"],
                    value=[],
                    multiselect=True,
                    allow_custom_value=True,
                    info="Generate trainer configs. Add custom values for newer trainers.",
                )
            concepts = gr.Textbox(label="--concepts (optional)", info="Comma-separated concept folder names to include.", value="")
            with gr.Row():
                # CLAUDE-NOTE: This is Klippbok's own --dry-run flag, distinct
                # from our UI's "Dry run" which just previews the command. Both
                # can be toggled independently.
                klippbok_dry_run = gr.Checkbox(label="--dry-run (Klippbok flag)", value=False, info="Preview what Klippbok would do without touching files.")
                strict = gr.Checkbox(label="--strict", value=False, info="Exclude samples with warnings, not just errors.")
                move = gr.Checkbox(label="--move", value=False, info="Move instead of copy (destructive).")
                manifest = gr.Checkbox(label="--manifest", value=False, info="Write klippbok_manifest.json to the output.")
            config = gr.Textbox(label="--config (optional)", info="Path to klippbok_data.yaml.", value="")

        ps.register(
            "organize",
            directory=s["directory"], dry_run=s["dry_run"],
            output=output, layout=layout, trainers=trainers,
            concepts=concepts, klippbok_dry_run=klippbok_dry_run,
            strict=strict, move=move, manifest=manifest, config=config,
        )
        ps.register_run("organize", s["run_btn"])

        def _run(path, output, layout, trainers, concepts, kb_dry, strict, move, manifest, config, dry):
            if not path:
                yield "[error] Path is required."
                return
            if not output:
                yield "[error] --output is required."
                return
            cmd = _base_cmd("klippbok.dataset", "organize", path, "--output", output)
            if layout:
                cmd += ["--layout", layout]
            for t in trainers or []:
                cmd += ["--trainer", t]
            if concepts:
                cmd += ["--concepts", concepts]
            if kb_dry:
                cmd.append("--dry-run")
            if strict:
                cmd.append("--strict")
            if move:
                cmd.append("--move")
            if manifest:
                cmd.append("--manifest")
            if config:
                cmd += ["--config", config]
            if dry:
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], output, layout, trainers, concepts, klippbok_dry_run, strict, move, manifest, config, s["dry_run"]],
            outputs=s["log"],
        )


def _tab_settings(api_state: gr.State) -> None:
    """API keys, .env persistence, and install check."""
    loaded = _load_env()
    with gr.Tab("Settings"):
        gr.Markdown(
            "### Settings\n"
            "API keys are passed into Caption / Audit / Ingest(`--caption`) subprocesses"
            " via the environment. They live only in memory unless you click"
            f" **Save to .env**, which writes them to `{_env_path()}` — gitignored."
            " On next startup the UI auto-loads from .env so you don't have to re-paste."
        )
        gemini = gr.Textbox(
            label="GEMINI_API_KEY",
            type="password",
            info="Required for `--provider gemini` (free tier: https://aistudio.google.com/apikey).",
            value=loaded.get("GEMINI_API_KEY", ""),
        )
        replicate = gr.Textbox(
            label="REPLICATE_API_TOKEN",
            type="password",
            info="Required for `--provider replicate`.",
            value=loaded.get("REPLICATE_API_TOKEN", ""),
        )

        def _collect(g: str, r: str) -> dict[str, str]:
            keys: dict[str, str] = {}
            if g:
                keys["GEMINI_API_KEY"] = g
            if r:
                keys["REPLICATE_API_TOKEN"] = r
            return keys

        gemini.change(_collect, inputs=[gemini, replicate], outputs=api_state)
        replicate.change(_collect, inputs=[gemini, replicate], outputs=api_state)

        with gr.Row():
            save_btn = gr.Button("Save to .env", variant="primary", scale=0)
            check_btn = gr.Button("Check installation", scale=0)
        action_log = gr.Textbox(
            label="Status",
            lines=6,
            max_lines=20,
            interactive=False,
            show_copy_button=True,
        )

        def _on_save(g: str, r: str) -> str:
            keys = _collect(g, r)
            if not keys:
                return "Nothing to save — both key fields are empty."
            try:
                path = _save_env(keys)
            except Exception as exc:  # noqa: BLE001
                return f"[error] Could not write .env: {exc}"
            return f"Wrote {len(keys)} key(s) to {path}"

        save_btn.click(_on_save, inputs=[gemini, replicate], outputs=action_log)
        check_btn.click(_check_installation, outputs=action_log)


# ----- Agentic Pipeline (workspace scaffolder + guide) -------------------


def _tab_agentic_pipeline() -> None:
    """Front door for the agent-driven LoRA training pipeline.

    This tab scaffolds a project workspace and serves the setup guide.
    It does NOT do any training itself — training runs through MCP
    servers that the user drives from Claude Desktop / Antigravity /
    etc. See pipeline_setup.py for the workspace layout and the
    generated AGENT_INSTRUCTIONS.md / PIPELINE_GUIDE.md / watchdog.py
    templates.
    """
    # CLAUDE-NOTE: Default root path prefers F:/__PROJECTS if it exists
    # (this user's pattern); otherwise falls back to the user's home
    # directory. Works cross-platform since Path.home() is universal.
    default_root_candidates = [
        Path("F:/__PROJECTS"),
        Path.home() / "Projects",
        Path.home(),
    ]
    default_root = str(next((p for p in default_root_candidates if p.exists()), Path.home()))

    # Per-target required-components map (used by the Create pre-flight
    # warning). Every Musubi-supported target maps to musubi_tuner +
    # musubi_mcp; LTX-2.3 uses its own trainer stack. If new targets are
    # added to pipeline_setup.TARGET_MODELS, add matching entries here.
    _musubi_deps = ["musubi_tuner", "musubi_mcp"]
    TARGET_DEPS = {t: _musubi_deps for t in pipe.VIDEO_TARGETS_MUSUBI}
    TARGET_DEPS.update({t: _musubi_deps for t in pipe.IMAGE_TARGETS_MUSUBI})
    TARGET_DEPS["LTX-2.3"] = ["ltx2_trainer", "ltx_trainer_mcp"]

    def _row_display(comp, status, all_statuses):
        """Compute (icon_md, path_text, btn_text, btn_interactive) for one status row."""
        installed = bool(status and status.installed)
        path_text = (status.path or "") if installed else "(not installed)"
        if comp.always_installed:
            return "**[ok]**", path_text, "Installed", False
        ok, reason = pi.can_install(comp, all_statuses)
        if installed:
            # Installed — "Update" if gate allows (uv, cloneable repos),
            # "Installed" (disabled) otherwise (git, which has no in-panel update).
            return (
                ("**[ok]**", path_text, "Update", True)
                if ok
                else ("**[ok]**", path_text, "Installed", False)
            )
        # Not installed — show reason in button text when gated.
        btn_text = "Install" if ok else f"Install ({reason})"
        return "**[!!]**", path_text, btn_text, ok

    with gr.Tab("Agentic Pipeline"):
        gr.Markdown(
            "## Agentic Pipeline\n"
            "Set up a workspace for agent-driven LoRA training that chains"
            " **Klippbok** (this app) → **Musubi Tuner** → **LTX-2.3 Trainer**"
            " through MCP servers. Install the pieces below, then create a"
            " workspace — the generated `AGENT_INSTRUCTIONS.md` will be fully"
            " pre-filled with every path your agent needs.\n\n"
            "*Pipeline designed by [hoodtronik](https://github.com/hoodtronik).*"
        )

        # ---- Shared Root textbox (drives BOTH install location AND workspace parent) -
        # CLAUDE-NOTE: Reused per user's design choice — one textbox at the
        # top of the tab controls where both companion tools get cloned AND
        # where project workspaces get created. Keeps the mental model simple.
        with gr.Row():
            root_path = gr.Textbox(
                label="Root path",
                value=default_root,
                info="Install location for pipeline tools AND parent for project workspaces.",
                scale=4,
            )
            root_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
            root_browse.click(_pick_folder, outputs=root_path)

        # ---- Section: Pipeline Status --------------------------------
        with gr.Accordion("Pipeline Components", open=True):
            gr.Markdown(
                "Install the tools you need before creating a workspace. You only"
                " need trainers for the models you plan to use. Click **Refresh"
                " Paths** after installing something outside this UI."
            )

            # CLAUDE-NOTE: Detect once at build time using only filesystem
            # probes — fast and keeps app boot snappy per the user's choice.
            # Subprocess checks (nvidia-smi, uv --version) only run when
            # Refresh Paths or an install is clicked.
            try:
                _initial_root = Path(default_root)
                _initial_statuses = pi.detect_all(_initial_root)
            except Exception:
                _initial_statuses = {c.id: pi.ComponentStatus(c.id, False, None) for c in pi.COMPONENTS}

            status_rows: dict[str, dict] = {}
            for comp in pi.COMPONENTS:
                s = _initial_statuses.get(comp.id)
                icon, path_text, btn_text, btn_enabled = _row_display(comp, s, _initial_statuses)
                with gr.Row(equal_height=True):
                    # CLAUDE-NOTE: gr.Markdown in Gradio 5 does NOT accept
                    # `scale` or `min_width` kwargs (unlike Textbox / Button).
                    # Gradio 4 allowed them; 5 removed them silently. Rely
                    # on the surrounding Row for layout — the scale-0
                    # Textbox/Button siblings keep the Markdown from eating
                    # too much horizontal space.
                    status_md = gr.Markdown(value=icon)
                    gr.Markdown(
                        value=f"**{comp.display_name}** — {comp.description}"
                        + (f"  \n_{comp.install_warning}_" if comp.install_warning else "")
                    )
                    path_tb = gr.Textbox(
                        value=path_text,
                        interactive=False,
                        show_label=False,
                        container=False,
                        scale=3,
                    )
                    install_btn = gr.Button(
                        value=btn_text,
                        scale=0,
                        size="sm",
                        interactive=btn_enabled,
                        min_width=160,
                    )
                status_rows[comp.id] = {
                    "comp": comp,
                    "status": status_md,
                    "path": path_tb,
                    "button": install_btn,
                }

            with gr.Row():
                refresh_btn = gr.Button("Refresh paths", size="sm")

            install_log = gr.Textbox(
                label="Install log",
                lines=12,
                max_lines=2000,
                interactive=False,
                autoscroll=True,
                show_copy_button=True,
                value="",
            )

        # ---- Section: create workspace ------------------------------
        with gr.Accordion("1. Create training workspace", open=True):
            project_name = gr.Textbox(
                label="Project name",
                value="MyLoRAProject",
                info="Filesystem-safe name. Becomes the top-level folder under Root path.",
            )

            with gr.Accordion("Advanced options", open=False):
                styles_box = gr.Textbox(
                    label="Style subfolders under datasets/",
                    placeholder="blur_studio\nneon_noir\ncinematic_realism",
                    lines=4,
                    info="One per line. Leave blank for a flat datasets/ folder.",
                )
                strategy = gr.Dropdown(
                    label="Training strategy",
                    choices=list(pipe.STRATEGIES),
                    value=pipe.DEFAULT_STRATEGY,
                    info="Layered creates outputs/master_loras/ + outputs/boutique_loras/. "
                         "Single creates outputs/loras/. Custom leaves outputs/ empty.",
                )
                gr.Markdown(
                    "**Target models** — tick every model you plan to train. "
                    "The generated AGENT_INSTRUCTIONS.md gets a trainer-specific "
                    "section for each."
                )
                gr.Markdown("_Video models (via **Musubi Tuner**)_")
                targets_video_musubi = gr.CheckboxGroup(
                    choices=list(pipe.VIDEO_TARGETS_MUSUBI),
                    value=[],
                    show_label=False,
                    container=False,
                )
                gr.Markdown("_Image models (via **Musubi Tuner**)_")
                targets_image_musubi = gr.CheckboxGroup(
                    choices=list(pipe.IMAGE_TARGETS_MUSUBI),
                    value=[],
                    show_label=False,
                    container=False,
                )
                gr.Markdown(
                    "_Other trainers_  \n"
                    "⚠ **LTX-2.3** uses Lightricks' LTX-2 trainer, **not Musubi**. "
                    "Requires `ltx2_trainer` + `ltx_trainer_mcp` instead of the Musubi pair."
                )
                targets_other = gr.CheckboxGroup(
                    choices=list(pipe.VIDEO_TARGETS_OTHER),
                    value=[],
                    show_label=False,
                    container=False,
                )

            with gr.Row():
                refresh_paths_btn = gr.Button("Refresh paths", size="sm")
                create_btn = gr.Button("Create workspace", variant="primary")

            status_box = gr.Code(
                label="Result",
                language=None,
                value="",
                interactive=False,
                visible=False,
            )

            def _on_create(name, root, styles_text, strat, tgts_video, tgts_image, tgts_other):
                # Merge the three grouped CheckboxGroups into a single list
                # the config + downstream template rendering already expect.
                tgts = list(tgts_video or []) + list(tgts_image or []) + list(tgts_other or [])
                styles = [s.strip() for s in (styles_text or "").splitlines() if s.strip()]
                cfg = pipe.WorkspaceConfig(
                    project_name=(name or "").strip(),
                    root_path=(root or "").strip(),
                    styles=styles,
                    strategy=strat or pipe.DEFAULT_STRATEGY,
                    targets=tgts,
                )
                # Pre-flight: which components are required given the target
                # models? klippbok-mcp is always required; each target adds
                # its trainer + MCP. Warn but still proceed so the workspace
                # exists (with placeholders) and the user can re-create it
                # after installing.
                required = {"klippbok_mcp"}
                for t in cfg.targets:
                    for dep in TARGET_DEPS.get(t, []):
                        required.add(dep)
                try:
                    statuses = pi.detect_all(Path(cfg.root_path)) if cfg.root_path else {}
                except Exception:
                    statuses = {}
                missing = [
                    (pi.component_by_id(r).display_name if pi.component_by_id(r) else r)
                    for r in sorted(required)
                    if not (statuses.get(r) and statuses[r].installed)
                ]
                preamble = ""
                if missing:
                    preamble = (
                        "[warning] These components are required by your selected targets but "
                        "are not installed yet:\n"
                        + "\n".join(f"  - {m}" for m in missing)
                        + "\n\nThe workspace will still be created, but paths for the missing "
                        "components will appear as <NOT FOUND> placeholders in AGENT_INSTRUCTIONS.md. "
                        "Install them from Pipeline Components above, then click **Create workspace** "
                        "again to refresh the generated files.\n\n"
                    )
                elif cfg.targets:
                    preamble = "[ok] All required tools detected — workspace will be fully configured.\n\n"

                _ok, message = pipe.create_workspace(cfg)
                return gr.update(value=preamble + message, visible=True)

            create_btn.click(
                _on_create,
                inputs=[
                    project_name, root_path, styles_box, strategy,
                    targets_video_musubi, targets_image_musubi, targets_other,
                ],
                outputs=status_box,
            )

        # ---- Section: pipeline guide --------------------------------
        with gr.Accordion("2. Pipeline guide (also written to the workspace)", open=False):
            gr.Markdown(pipe.PIPELINE_GUIDE_MD)

        # ---- Handlers: refresh + per-component install --------------

        # The `refresh_handler` output list needs to match across every
        # trigger — define once and reuse.
        refresh_outputs: list = []
        for comp in pi.COMPONENTS:
            r = status_rows[comp.id]
            refresh_outputs += [r["status"], r["path"], r["button"]]

        def refresh_handler(root: str):
            install_root = Path(root) if root else Path.cwd()
            try:
                statuses = pi.detect_all(install_root)
            except Exception:
                statuses = {c.id: pi.ComponentStatus(c.id, False, None) for c in pi.COMPONENTS}
            updates: list = []
            for comp in pi.COMPONENTS:
                s = statuses.get(comp.id)
                icon, path_text, btn_text, btn_enabled = _row_display(comp, s, statuses)
                updates.append(gr.update(value=icon))
                updates.append(gr.update(value=path_text))
                updates.append(gr.update(value=btn_text, interactive=btn_enabled))
            return updates

        refresh_btn.click(refresh_handler, inputs=[root_path], outputs=refresh_outputs)
        refresh_paths_btn.click(refresh_handler, inputs=[root_path], outputs=refresh_outputs)

        # Per-component Install / Update handler. Each button gets its own
        # closure binding the component id.
        for comp_id, row in status_rows.items():
            def _make_install(cid: str):
                def _handler(root: str):
                    target_comp = pi.component_by_id(cid)
                    if target_comp is None:
                        yield f"[error] Unknown component: {cid}"
                        return
                    install_root = Path(root) if root else Path.cwd()
                    current = pi.detect_component(target_comp, install_root)
                    is_update = current.installed and not target_comp.always_installed
                    log = f"# {('Updating' if is_update else 'Installing')} {target_comp.display_name}\n"
                    log += f"# Root: {install_root}\n\n"
                    yield log
                    for line in pi.install_component(target_comp, install_root, update=is_update):
                        log += line + "\n"
                        yield log
                return _handler

            row["button"].click(
                _make_install(comp_id),
                inputs=[root_path],
                outputs=install_log,
            ).then(
                refresh_handler,
                inputs=[root_path],
                outputs=refresh_outputs,
            )


# ----- Manifest Reviewer --------------------------------------------------


def _render_entry_meta(e: mft.Entry) -> str:
    """Markdown block shown next to each thumbnail."""
    concept = e.best_concept or "_(no match)_"
    lines = [
        f"**{e.display_label}**",
        f"Score: `{e.score:.3f}` ->`{concept}`",
    ]
    if e.text_overlay:
        lines.append("⚠ text overlay detected")
    if e.use_case:
        lines.append(f"Use case: `{e.use_case}`")
    lines.append(f"Path: `{e.video_path}`")
    return "  \n".join(lines)


def _summary(state: dict) -> str:
    entries = state.get("entries") or []
    if not entries:
        return "_No manifest loaded._"
    included = sum(1 for e in entries if e.include)
    kind = state.get("kind", "?")
    path = state.get("path", "")
    return f"**{included} of {len(entries)} included** · kind=`{kind}` · source `{path}`"


def _page_label(state: dict) -> str:
    entries = state.get("entries") or []
    if not entries:
        return "—"
    page = state.get("page", 0)
    total = max(1, (len(entries) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    return f"Page **{page + 1}** of **{total}**"


def _render_page(state: dict) -> list:
    """Return a flat list of updates: [status, page_label, *25×(row, image, md, checkbox)]."""
    out: list = [gr.update(value=_summary(state)), gr.update(value=_page_label(state))]
    entries = state.get("entries") or []
    page = state.get("page", 0)
    page_entries = entries[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]

    # Parallelize thumbnail extraction — ffmpeg runs are I/O-bound and cached
    # after the first visit, so subsequent page turns are fast.
    thumbs: list = [None] * len(page_entries)
    if page_entries:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, path in enumerate(ex.map(lambda e: mft.generate_thumbnail(e, _CACHE_DIR), page_entries)):
                thumbs[i] = path

    for i in range(_PAGE_SIZE):
        if i < len(page_entries):
            e = page_entries[i]
            out += [
                gr.update(visible=True),
                gr.update(value=thumbs[i]),
                gr.update(value=_render_entry_meta(e)),
                gr.update(value=bool(e.include)),
            ]
        else:
            out += [
                gr.update(visible=False),
                gr.update(value=None),
                gr.update(value=""),
                gr.update(value=True),
            ]
    return out


def _tab_manifest_reviewer(last_manifest_state: gr.State, work_dir: gr.Textbox, ps: "ps_mod.ProjectState") -> None:
    with gr.Tab("Manifest Reviewer"):
        gr.Markdown(
            "### Manifest Reviewer\n"
            "Load a `triage_manifest.json` (clip-level) or `scene_triage_manifest.json` "
            "(scene-level). The path auto-populates as soon as Triage finishes; click "
            "**Use latest triage output** if you restarted the UI and want to pick up the "
            "last run from disk. Thumbnails are extracted on demand via ffmpeg and cached "
            "in `cache/thumbs/`. Save writes to `<name>_reviewed.json` by default so your "
            "original stays untouched."
        )

        # CLAUDE-NOTE: state dict shape = {path, raw, entries, kind, page}.
        # entries is a list[mft.Entry]. raw is the pristine manifest dict —
        # save_manifest writes `include` back into it without touching any
        # other field, so Klippbok's round-trips stay lossless.
        state = gr.State({})

        # ---- Load row --------------------------------------------------
        with gr.Row():
            path_in = gr.Textbox(
                label="Manifest path",
                placeholder="…/triage_manifest.json or …/scene_triage_manifest.json",
                scale=4,
            )
            path_browse = gr.Button("Browse…", scale=0, size="sm", min_width=0)
            use_latest_btn = gr.Button("Use latest triage output", scale=0, size="sm")
            load_btn = gr.Button("Load", variant="primary", scale=0)

        # CLAUDE-NOTE: Triage pushes the written manifest path into
        # last_manifest_state; _wire_autofill tracks what we last auto-set
        # so the user's manual edits survive, but rerunning Triage still
        # refreshes the field (refilling from "" or our own prior value).
        reviewer_tracker = gr.State("")
        _wire_autofill(last_manifest_state, path_in, reviewer_tracker)

        # CLAUDE-NOTE: Only path_in gets persisted — per-clip include flags
        # and pagination state are intentionally excluded (they're large
        # and tied to a manifest file that users re-load each session).
        ps.register("manifest_reviewer", path=path_in)

        # Explicit button for the "I restarted the UI, find the latest from disk"
        # case, which scans the Project tab's working directory too.
        def _use_latest(latest: str, wd: str, current: str) -> str:
            candidate = latest
            if not candidate and wd:
                candidate = _detect_triage_manifest(wd, "")
            if not candidate:
                gr.Warning(
                    "No triage manifest found. Run Triage first, set Working "
                    "directory in the Project tab, or Browse to a manifest manually."
                )
                return current
            gr.Info(f"Loaded path: {candidate}")
            return candidate
        use_latest_btn.click(
            _use_latest,
            inputs=[last_manifest_state, work_dir, path_in],
            outputs=path_in,
        )

        def _pick_manifest_path() -> str:
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                p = filedialog.askopenfilename(
                    title="Select triage manifest",
                    filetypes=[("JSON", "*.json"), ("All", "*.*")],
                )
                root.destroy()
                return p or ""
            except Exception:
                return ""

        path_browse.click(_pick_manifest_path, outputs=path_in)

        status = gr.Markdown(_summary({}))

        # ---- Bulk actions ---------------------------------------------
        with gr.Row():
            include_all = gr.Button("Include all")
            exclude_all = gr.Button("Exclude all")
            threshold = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="Threshold")
            apply_thr = gr.Button("Include where score ≥ threshold")
            exclude_overlays = gr.Button("Exclude clips with text overlay")

        # ---- Pagination -----------------------------------------------
        with gr.Row():
            prev_btn = gr.Button("◀ Prev", scale=0)
            page_label = gr.Markdown("—")
            next_btn = gr.Button("Next ▶", scale=0)

        # ---- 25 fixed slots --------------------------------------------
        rows: list = []
        thumbs: list = []
        metas: list = []
        checks: list = []
        for i in range(_PAGE_SIZE):
            with gr.Row(visible=False, equal_height=True) as row:
                thumb = gr.Image(
                    show_label=False,
                    interactive=False,
                    show_download_button=False,
                    height=120,
                    width=200,
                    container=False,
                )
                meta = gr.Markdown()
                check = gr.Checkbox(label="Include", value=True, scale=0, min_width=120)
            rows.append(row)
            thumbs.append(thumb)
            metas.append(meta)
            checks.append(check)

        # ---- Save ------------------------------------------------------
        with gr.Row():
            overwrite = gr.Checkbox(label="Overwrite original (otherwise write *_reviewed.json)", value=False)
            save_btn = gr.Button("Save", variant="primary")
        save_log = gr.Markdown("")

        # ---- Outputs list reused by handlers that re-render the page ---
        page_outputs = [status, page_label]
        for i in range(_PAGE_SIZE):
            page_outputs += [rows[i], thumbs[i], metas[i], checks[i]]

        # ---- Handlers --------------------------------------------------
        def _on_load(p: str):
            if not p:
                blank = {}
                return [blank, *_render_page(blank)]
            try:
                raw, entries, kind = mft.load_manifest(p)
            except Exception as exc:
                # Reuse blank state so the page clears; surface the error in status.
                err_state = {}
                updates = _render_page(err_state)
                updates[0] = gr.update(value=f"**[error]** {exc}")
                return [err_state, *updates]
            new_state = {"path": str(p), "raw": raw, "entries": entries, "kind": kind, "page": 0}
            return [new_state, *_render_page(new_state)]

        load_btn.click(
            _on_load,
            inputs=[path_in],
            outputs=[state, *page_outputs],
        )

        def _on_prev(st: dict):
            st = dict(st) if st else {}
            st["page"] = max(0, st.get("page", 0) - 1)
            return [st, *_render_page(st)]

        def _on_next(st: dict):
            st = dict(st) if st else {}
            entries = st.get("entries") or []
            total = max(1, (len(entries) + _PAGE_SIZE - 1) // _PAGE_SIZE)
            st["page"] = min(total - 1, st.get("page", 0) + 1)
            return [st, *_render_page(st)]

        prev_btn.click(_on_prev, inputs=[state], outputs=[state, *page_outputs])
        next_btn.click(_on_next, inputs=[state], outputs=[state, *page_outputs])

        def _set_all(st: dict, value: bool):
            if not st or not st.get("entries"):
                return [st, *_render_page(st or {})]
            for e in st["entries"]:
                e.include = value
            return [st, *_render_page(st)]

        include_all.click(lambda st: _set_all(st, True), inputs=[state], outputs=[state, *page_outputs])
        exclude_all.click(lambda st: _set_all(st, False), inputs=[state], outputs=[state, *page_outputs])

        def _on_apply_threshold(st: dict, thr: float):
            if not st or not st.get("entries"):
                return [st, *_render_page(st or {})]
            for e in st["entries"]:
                e.include = e.score >= float(thr)
            return [st, *_render_page(st)]

        apply_thr.click(
            _on_apply_threshold,
            inputs=[state, threshold],
            outputs=[state, *page_outputs],
        )

        def _on_exclude_overlays(st: dict):
            if not st or not st.get("entries"):
                return [st, *_render_page(st or {})]
            for e in st["entries"]:
                if e.text_overlay:
                    e.include = False
            return [st, *_render_page(st)]

        exclude_overlays.click(_on_exclude_overlays, inputs=[state], outputs=[state, *page_outputs])

        # Per-slot checkbox — updates the underlying entry's include flag and
        # the summary, but does NOT re-render other slots. Uses .input (fires
        # only on user interaction) to avoid feedback loops when bulk actions
        # set checkbox values programmatically.
        for i, check in enumerate(checks):
            def _on_toggle(new_value: bool, st: dict, slot_idx: int = i):
                entries = (st or {}).get("entries") or []
                page = (st or {}).get("page", 0)
                global_idx = page * _PAGE_SIZE + slot_idx
                if 0 <= global_idx < len(entries):
                    entries[global_idx].include = bool(new_value)
                return st, gr.update(value=_summary(st))
            check.input(_on_toggle, inputs=[check, state], outputs=[state, status])

        def _on_save(st: dict, overwrite_original: bool):
            if not st or not st.get("entries"):
                return "_Load a manifest first._"
            src = Path(st["path"])
            dest = src if overwrite_original else mft.reviewed_path_for(src)
            try:
                mft.save_manifest(st["raw"], st["entries"], dest)
            except Exception as exc:
                return f"**[error]** {exc}"
            included = sum(1 for e in st["entries"] if e.include)
            return f"Saved {included} / {len(st['entries'])} to `{dest}`"

        save_btn.click(_on_save, inputs=[state, overwrite], outputs=save_log)


# ----- ui ------------------------------------------------------------------


def _wire_project_persistence(
    *,
    demo: gr.Blocks,
    ps: "ps_mod.ProjectState",
    work_dir: gr.Textbox,
    concepts_dir: gr.Textbox,
    output_dir: gr.Textbox,
    tabs_run_state: gr.State,
    last_manifest_state: gr.State,
    last_ingest_output_state: gr.State,
    last_caption_dir_state: gr.State,
    project_status: gr.Markdown,
    recent_dropdown: gr.Dropdown,
    load_project_btn: gr.Button,
    new_project_btn: gr.Button,
    trainer_preset: gr.Dropdown,
    trainer_preset_info: gr.Markdown,
) -> None:
    """Install auto-save, auto-load, and project-management wiring.

    Call exactly once, after every tab has registered its components
    with `ps`. This single function owns the contract between producer
    tabs (which push new values into the registered components) and the
    on-disk `klippbok_project.json` file.
    """
    fields = ps.ordered_components()

    # Order matters — every save handler reads these in this order, and
    # every load handler writes in this same order. Keep the two lists
    # structurally identical (except fields get gr.update(...), states
    # get raw values, and the UI meta goes in the tail).
    _state_inputs = [
        work_dir, concepts_dir, output_dir, tabs_run_state,
        last_manifest_state, last_ingest_output_state, last_caption_dir_state,
    ]
    save_inputs = _state_inputs + list(fields)
    load_outputs = _state_inputs + list(fields) + [project_status, recent_dropdown]
    _load_tail_size = 2  # project_status + recent_dropdown

    def _save_handler(wd, cd, od, tabs_run,
                      last_triage_m, last_ingest_o, last_caption_d,
                      *field_values):
        # No work_dir = no save target; show the empty status and leave
        # the dropdown untouched.
        if not wd:
            return (
                gr.update(value=ps_mod.format_status("", "", [])),
                gr.update(),
            )
        propagation = {
            "last_triage_manifest": last_triage_m or "",
            "last_ingest_output": last_ingest_o or "",
            "last_caption_dir": last_caption_d or "",
        }
        payload = ps_mod.build_payload(
            wd, cd, od, tabs_run or [], propagation,
            ps.pack_values(list(field_values)),
        )
        saved_at, err = ps_mod.save_project(wd, payload)
        status = ps_mod.format_status(wd, saved_at, tabs_run or [], err)
        return (
            gr.update(value=status),
            gr.update(choices=ps_mod.load_recent(), value=None),
        )

    # Trigger save on every registered field + shared dirs + state changes.
    # gr.on bundles multiple triggers onto the same handler, so the 60+
    # events share one save function instead of 60 individual wirings.
    save_triggers = [f.change for f in fields]
    save_triggers += [
        work_dir.change, concepts_dir.change, output_dir.change,
        tabs_run_state.change,
        last_manifest_state.change,
        last_ingest_output_state.change,
        last_caption_dir_state.change,
    ]
    gr.on(
        save_triggers, _save_handler,
        inputs=save_inputs,
        outputs=[project_status, recent_dropdown],
    )

    # Run-button click → mark tab as completed. A second .click() handler
    # on the same button runs alongside the tab's existing _run streamer.
    def _make_mark_ran(tab_id: str):
        def _mark(tabs_run):
            tabs_run = list(tabs_run or [])
            if tab_id not in tabs_run:
                tabs_run.append(tab_id)
            return tabs_run
        return _mark

    for tab_id, btn in ps.run_buttons().items():
        btn.click(
            _make_mark_ran(tab_id),
            inputs=tabs_run_state,
            outputs=tabs_run_state,
        )

    # ----- LOAD pipeline ---------------------------------------------------

    def _no_change_tuple() -> tuple:
        # Used when a load attempt fails — every output stays put.
        return tuple(gr.update() for _ in load_outputs)

    def _load_handler_impl(path: str) -> tuple:
        payload, err = ps_mod.read_project_file(path)
        if err or payload is None:
            gr.Warning(f"Could not load project: {err}")
            return _no_change_tuple()
        wd, cd, od = ps_mod.extract_project_dirs(payload)
        tabs_run = ps_mod.extract_tabs_run(payload)
        paths = ps_mod.extract_paths(payload)
        field_updates = ps.unpack_values(ps_mod.extract_fields(payload))
        saved_at = payload.get("saved_at", "")
        status = ps_mod.format_status(wd, saved_at, tabs_run)
        # Record this open so it shows up in Recent Projects next time.
        if wd:
            try:
                ps_mod._update_recent(wd)
            except Exception:
                pass
        gr.Info(f"Loaded project: {ps_mod.project_name(wd) or wd}")
        return (
            gr.update(value=wd),
            gr.update(value=cd),
            gr.update(value=od),
            tabs_run,
            paths.get("last_triage_manifest", ""),
            paths.get("last_ingest_output", ""),
            paths.get("last_caption_dir", ""),
            *field_updates,
            gr.update(value=status),
            gr.update(choices=ps_mod.load_recent(), value=None),
        )

    # File-picker bridge: button → filedialog → path state → load handler.
    def _pick_project_file() -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            p = filedialog.askopenfilename(
                title="Select klippbok_project.json",
                filetypes=[("Klippbok project", "klippbok_project.json"),
                           ("JSON", "*.json"),
                           ("All files", "*.*")],
            )
            root.destroy()
            return p or ""
        except Exception:
            return ""

    picker_path_state = gr.State("")
    load_project_btn.click(
        _pick_project_file, outputs=picker_path_state,
    ).then(
        _load_handler_impl, inputs=picker_path_state, outputs=load_outputs,
    )

    # Recent Projects dropdown → load matching klippbok_project.json.
    def _load_from_recent(chosen_dir: str) -> tuple:
        if not chosen_dir:
            return _no_change_tuple()
        pf = ps_mod.project_file_in(chosen_dir)
        if not pf or not pf.is_file():
            gr.Warning(f"No klippbok_project.json found in: {chosen_dir}")
            return _no_change_tuple()
        return _load_handler_impl(str(pf))

    recent_dropdown.change(
        _load_from_recent, inputs=recent_dropdown, outputs=load_outputs,
    )

    # New Project → pick a new Working directory. If the picked folder
    # already holds a klippbok_project.json, open it instead of
    # overwriting; otherwise set the working directory and leave every
    # other field alone so the user's current defaults carry over. The
    # subsequent work_dir.change fires the auto-save handler, which
    # writes a fresh project file to the new location.
    #
    # CLAUDE-NOTE: An earlier version returned gr.update(value="") for
    # every registered component; that broke Number / Checkbox /
    # Dropdown fields (each raising on the empty string and rendering
    # the red "Error" badge). Now we only touch work_dir + status, so
    # non-string components are never handed a bad value.
    def _new_project() -> tuple:
        picked = _pick_folder()
        if not picked:
            gr.Info("New project cancelled.")
            return _no_change_tuple()
        pf = ps_mod.project_file_in(picked)
        if pf and pf.is_file():
            gr.Info(f"Opening existing project in {picked}")
            return _load_handler_impl(str(pf))
        gr.Info(
            f"New project folder: {ps_mod.project_name(picked)}. "
            f"Auto-save is now pointed at {picked}/klippbok_project.json."
        )
        updates = list(_no_change_tuple())
        updates[0] = gr.update(value=picked)  # work_dir
        updates[3] = []  # tabs_run — fresh project, no completed steps yet
        updates[-2] = gr.update(value=ps_mod.format_status(picked, "", []))
        return tuple(updates)

    new_project_btn.click(_new_project, outputs=load_outputs)

    # Startup auto-load: prefer the last project file; fall back to
    # per-user detected_paths if no project file exists (so the
    # cross-tab propagation feature still has sane defaults).
    def _initial_load() -> tuple:
        last = ps_mod.load_last_project_dir()
        if last:
            pf = ps_mod.project_file_in(last)
            if pf and pf.is_file():
                return _load_handler_impl(str(pf))
        # Fallback path: no project file → only fill propagation states
        # from detected_paths; leave every field untouched.
        dp = _load_detected_paths()
        updates = list(_no_change_tuple())
        updates[4] = dp.get(_DETECTED_KEY_TRIAGE_MANIFEST, "")
        updates[5] = dp.get(_DETECTED_KEY_INGEST_OUTPUT, "")
        updates[6] = dp.get(_DETECTED_KEY_CAPTION_DIR, "")
        # Tail: status + dropdown choices.
        updates[-2] = gr.update(value=ps_mod.format_status("", "", []))
        updates[-1] = gr.update(choices=ps_mod.load_recent(), value=None)
        return tuple(updates)

    demo.load(_initial_load, outputs=load_outputs)

    # Ensure _load_tail_size matches expectations (compile-time sanity
    # check — kept explicit so future edits notice if they drift).
    assert _load_tail_size == 2, "load_outputs tail size mismatch"

    # ----- TRAINER-PRESET wiring ------------------------------------------
    # CLAUDE-NOTE: Pure UI-side feature. Picking a Target Trainer in the
    # Project tab pushes default values across Scan / Ingest / Normalize /
    # Caption / Validate. Klippbok's CLI is preset-agnostic — it just
    # receives whatever flag values are in the fields at Run time. The
    # auto-save pipeline picks up these field changes the same way it
    # would for a manual edit, so the preset survives across reloads.
    preset_targets = [
        ("scan", "fps", "scan_fps"),
        ("ingest", "max_frames", "ingest_max_frames"),
        ("normalize", "fps", "normalize_fps"),
        ("caption", "caption_fps", "caption_fps"),
        ("validate", "frame_count_rule", "frame_count_rule"),
    ]
    preset_components: list[gr.components.Component] = []
    preset_keys: list[str] = []
    for tab_id, field_name, preset_key in preset_targets:
        comp = ps.get_field(tab_id, field_name)
        if comp is not None:
            preset_components.append(comp)
            preset_keys.append(preset_key)

    def _apply_preset(preset_name: str) -> tuple:
        # Defensive: unknown preset name leaves every field alone but
        # still updates the guidance markdown to match.
        spec = TRAINER_PRESETS.get(preset_name) or {}
        updates: list = []
        for key in preset_keys:
            if key in spec:
                updates.append(gr.update(value=spec[key]))
            else:
                updates.append(gr.update())
        guidance = spec.get("guidance", "")
        return (*updates, gr.update(value=guidance))

    trainer_preset.change(
        _apply_preset,
        inputs=trainer_preset,
        outputs=[*preset_components, trainer_preset_info],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Klippbok", analytics_enabled=False) as demo:
        gr.Markdown("# Klippbok\n_Video dataset curation — Pinokio launcher_")

        # CLAUDE-NOTE: Shared state declared up front so the command tabs
        # (built in display order below) can reference it as an input while
        # other tabs write into it via .change() handlers. Initial value
        # is hydrated from .env if present, so Caption/Audit work without
        # the user having to visit Settings first.
        _persisted = _load_env()
        api_state = gr.State(
            {k: v for k, v in _persisted.items() if k in _RELEVANT_ENV_KEYS and v}
        )
        # CLAUDE-NOTE: Cross-tab path-propagation states. Producers
        # (Triage / Ingest / Caption) push their just-written path here on
        # success; consumers wire _wire_autofill on these states to fill
        # their input fields unless the user has typed something else.
        # Values are hydrated from .user_settings.json at demo.load() so
        # they survive app restarts — see _hydrate_propagation_states below.
        last_manifest_state = gr.State("")
        last_ingest_output_state = gr.State("")
        last_caption_dir_state = gr.State("")

        # CLAUDE-NOTE: Per-session registry + tabs-run tracker for
        # project-level persistence. `ps` is populated as each tab builds
        # (ps.register / ps.register_run); the wiring pass at the end of
        # build_ui installs a save handler on every registered component.
        ps = ps_mod.ProjectState()
        tabs_run_state = gr.State([])

        # Landing / orientation.
        _tab_directions()

        # Shared directory state + scaffold helper + Load/New/Recent UI.
        project_ui = _tab_project()
        work_dir = project_ui["work_dir"]
        concepts_dir = project_ui["concepts_dir"]
        output_dir = project_ui["output_dir"]

        # The Klippbok command tabs, in pipeline order. Each tab receives
        # only the propagation states it produces or consumes; `ps` is
        # always last so registrations happen during tab construction.
        _tab_scan(work_dir, concepts_dir, output_dir, api_state, ps)
        _tab_triage(work_dir, concepts_dir, output_dir, api_state, last_manifest_state, ps)
        _tab_ingest(
            work_dir, concepts_dir, output_dir, api_state,
            last_manifest_state, last_ingest_output_state, ps,
        )
        _tab_normalize(work_dir, concepts_dir, output_dir, api_state, last_ingest_output_state, ps)
        _tab_caption(
            work_dir, concepts_dir, output_dir, api_state,
            last_ingest_output_state, last_caption_dir_state, ps,
        )
        _tab_score(work_dir, concepts_dir, output_dir, api_state, last_caption_dir_state, ps)
        _tab_extract(work_dir, concepts_dir, output_dir, api_state, last_ingest_output_state, ps)
        _tab_audit(work_dir, concepts_dir, output_dir, api_state, last_caption_dir_state, ps)
        _tab_validate(work_dir, concepts_dir, output_dir, api_state, last_ingest_output_state, ps)
        _tab_organize(work_dir, concepts_dir, output_dir, api_state, ps)

        # Manifest Reviewer — the whole reason this launcher exists.
        _tab_manifest_reviewer(last_manifest_state, work_dir, ps)

        # API keys, install-check, python-exe override (polished in step 7).
        _tab_settings(api_state)

        # Front door to the agent-driven training pipeline (Klippbok ->
        # Musubi/LTX trainers via MCP). Scaffolds a workspace + docs; does
        # NOT do any training itself. Pipeline designed by hoodtronik.
        _tab_agentic_pipeline()

        # CLAUDE-NOTE: Install the project-persistence pipeline AFTER every
        # tab has registered its components. This wires auto-save on every
        # registered field + the project-management buttons + the
        # demo.load hook that auto-loads the last project on startup.
        _wire_project_persistence(
            demo=demo,
            ps=ps,
            work_dir=work_dir,
            concepts_dir=concepts_dir,
            output_dir=output_dir,
            tabs_run_state=tabs_run_state,
            last_manifest_state=last_manifest_state,
            last_ingest_output_state=last_ingest_output_state,
            last_caption_dir_state=last_caption_dir_state,
            project_status=project_ui["project_status"],
            recent_dropdown=project_ui["recent_dropdown"],
            load_project_btn=project_ui["load_project_btn"],
            new_project_btn=project_ui["new_project_btn"],
            trainer_preset=project_ui["trainer_preset"],
            trainer_preset_info=project_ui["trainer_preset_info"],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Klippbok Gradio UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
