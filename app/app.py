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

import manifest as mft
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


DIRECTIONS_MD = """\
# How Klippbok works

Klippbok turns a pile of raw video into a LoRA training dataset. It's a
**pipeline** — run each stage in order, review the output, adjust, move on.

Every tab in this UI wraps one Klippbok CLI subcommand. Nothing here is
magic — you can run the same commands in a terminal and get the same
results. The UI exists to make the pipeline approachable and to give you
a visual reviewer for the triage manifest (the part that would otherwise
mean hand-editing a 1,700-line JSON file in a text editor).

---

## The 60-second mental model

1. You **hand-curate a concepts folder** — small subfolders of reference
   images, one per thing you want the model to learn (a character, a
   style, a motion, an object).
2. **Triage** uses CLIP to score every clip against those references and
   writes a manifest JSON saying which clip matches which concept.
3. **Manifest Reviewer** (under construction) lets you visually flip
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
    """Push source's value into target whenever source changes."""
    source.change(lambda v: v, inputs=source, outputs=target)


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


def _tab_project() -> tuple[gr.Textbox, gr.Textbox, gr.Textbox]:
    """Project tab with the three shared directory rows + a scaffold helper."""
    with gr.Tab("Project"):
        gr.Markdown(
            "### Working directories\n"
            "Set these once — every command tab starts with the matching path"
            " as its default, and you can override per-tab as needed.\n\n"
            "**Concepts is something you populate by hand** — see the"
            " **Directions** tab if you're not sure what that means."
        )
        work_dir = _folder_row(
            "Working directory",
            "Your raw video clips, or a dir of pre-split clips.",
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

    return work_dir, concepts_dir, output_dir


# ----- per-command tab builders -------------------------------------------


def _tab_scan(work_dir, _concepts, _output, _api) -> None:
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


def _tab_triage(work_dir, concepts_dir, _output, _api, last_manifest_state: gr.State) -> None:
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

        def _after_run(directory: str, output_override: str, dry: bool, current: str) -> str:
            # CLAUDE-NOTE: Don't update state on dry runs — Klippbok wasn't
            # actually invoked so nothing was written. Keep whatever was there.
            if dry:
                return current
            detected = _detect_triage_manifest(directory, output_override)
            return detected or current

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], concepts, threshold, frames, frames_per_scene, scene_threshold, output, organize, move, clip_model, s["dry_run"]],
            outputs=s["log"],
        ).then(
            _after_run,
            inputs=[s["directory"], output, s["dry_run"], last_manifest_state],
            outputs=last_manifest_state,
        )


def _tab_ingest(work_dir, _concepts, output_dir, _api) -> None:
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

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], output, config, threshold, max_frames, triage, caption, provider, s["dry_run"], _api],
            outputs=s["log"],
        )


def _tab_normalize(work_dir, _concepts, output_dir, _api) -> None:
    tab_id = "normalize"
    with gr.Tab("Normalize"):
        s = _command_shell(
            tab_id,
            "klippbok.video normalize",
            "Batch-fix fps, resolution, and frame count on pre-split clips to match the Klippbok training targets.",
            "Directory (positional) — clips to normalize",
        )
        _sync(work_dir, s["directory"])
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


def _tab_caption(work_dir, _concepts, _output, api_keys) -> None:
    tab_id = "caption"
    with gr.Tab("Caption"):
        s = _command_shell(
            tab_id,
            "klippbok.video caption",
            "Generate `.txt` sidecar captions for each clip using a vision-language model. API keys live in the Settings tab.",
            "Directory (positional) — clips to caption",
        )
        _sync(work_dir, s["directory"])
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

        def _run(directory, provider, use_case, anchor, caption_fps, tags, overwrite, base_url, model, dry, api):
            if not directory:
                yield "[error] Directory is required."
                return
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
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd, extra_env=api)

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], provider, use_case, anchor, caption_fps, tags, overwrite, base_url, model, s["dry_run"], api_keys],
            outputs=s["log"],
        )


def _tab_score(work_dir, _concepts, _output, _api) -> None:
    tab_id = "score"
    with gr.Tab("Score"):
        s = _command_shell(
            tab_id,
            "klippbok.video score",
            "Local heuristic scoring of existing `.txt` caption files — no API calls.",
            "Directory (positional) — caption .txt files",
        )
        _sync(work_dir, s["directory"])

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


def _tab_extract(work_dir, _concepts, output_dir, _api) -> None:
    tab_id = "extract"
    with gr.Tab("Extract"):
        s = _command_shell(
            tab_id,
            "klippbok.video extract",
            "Export reference frames as PNG from a directory of clips (or still images). Use the `--template` field to write a selections template JSON you can edit, then feed back via `--selections`.",
            "Directory (positional) — clips or images",
        )
        _sync(work_dir, s["directory"])
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


def _tab_audit(work_dir, _concepts, _output, api_keys) -> None:
    tab_id = "audit"
    with gr.Tab("Audit"):
        s = _command_shell(
            tab_id,
            "klippbok.video audit",
            "Compare existing captions against fresh VLM output to catch low-quality or drifted annotations. Uses the same API keys as Caption.",
            "Directory (positional) — captioned clips",
        )
        _sync(work_dir, s["directory"])
        with s["options"]:
            with gr.Row():
                provider = gr.Dropdown(label="--provider", choices=["gemini", "replicate", "openai"], value="gemini")
                use_case = gr.Dropdown(label="--use-case", choices=["(auto)", "character", "style", "motion", "object"], value="(auto)")
                mode = gr.Dropdown(label="--mode", choices=["report_only", "save_audit"], value="report_only")

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


def _tab_validate(work_dir, _concepts, _output, _api) -> None:
    tab_id = "validate"
    with gr.Tab("Validate"):
        s = _command_shell(
            tab_id,
            "klippbok.dataset validate",
            "Check dataset completeness and quality. Takes either a dataset folder or a `klippbok_data.yaml` config path as the positional argument.",
            "Path (positional) — dataset folder or klippbok_data.yaml",
        )
        _sync(work_dir, s["directory"])
        with s["options"]:
            with gr.Row():
                manifest = gr.Checkbox(label="--manifest", value=False, info="Write klippbok_manifest.json to the dataset folder.")
                buckets = gr.Checkbox(label="--buckets", value=False, info="Show training bucket preview.")
                quality = gr.Checkbox(label="--quality", value=False, info="Blur/exposure checks on reference images.")
            with gr.Row():
                duplicates = gr.Checkbox(label="--duplicates", value=False, info="Perceptual duplicate detection.")
                json_out = gr.Checkbox(label="--json", value=False, info="Emit JSON instead of formatted report.")
            config = gr.Textbox(label="--config (optional)", info="Path to klippbok_data.yaml override.", value="")

        def _run(path, manifest, buckets, quality, duplicates, json_out, config, dry):
            if not path:
                yield "[error] Path is required."
                return
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
                yield _dry_preview(cmd)
                return
            yield from _stream(tab_id, cmd)

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], manifest, buckets, quality, duplicates, json_out, config, s["dry_run"]],
            outputs=s["log"],
        )


def _tab_organize(work_dir, _concepts, output_dir, _api) -> None:
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


def _tab_manifest_reviewer(last_manifest_state: gr.State, work_dir: gr.Textbox) -> None:
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

        # CLAUDE-NOTE: When Triage finishes it pushes the written manifest path
        # into last_manifest_state. Auto-fill path_in only if the user hasn't
        # typed anything — don't clobber their in-progress work.
        def _on_triage_output(latest: str, current: str) -> str:
            if current:
                return current
            return latest or ""
        last_manifest_state.change(
            _on_triage_output, inputs=[last_manifest_state, path_in], outputs=path_in
        )

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
        # Populated by the Triage tab on successful run; consumed by the
        # Manifest Reviewer to auto-fill its path box.
        last_manifest_state = gr.State("")

        # Landing / orientation.
        _tab_directions()

        # Shared directory state + scaffold helper.
        work_dir, concepts_dir, output_dir = _tab_project()

        # The Klippbok command tabs, in pipeline order. Triage and the
        # Reviewer are the only tabs that need last_manifest_state, so the
        # others stay on the uniform four-arg signature.
        _tab_scan(work_dir, concepts_dir, output_dir, api_state)
        _tab_triage(work_dir, concepts_dir, output_dir, api_state, last_manifest_state)
        _tab_ingest(work_dir, concepts_dir, output_dir, api_state)
        _tab_normalize(work_dir, concepts_dir, output_dir, api_state)
        _tab_caption(work_dir, concepts_dir, output_dir, api_state)
        _tab_score(work_dir, concepts_dir, output_dir, api_state)
        _tab_extract(work_dir, concepts_dir, output_dir, api_state)
        _tab_audit(work_dir, concepts_dir, output_dir, api_state)
        _tab_validate(work_dir, concepts_dir, output_dir, api_state)
        _tab_organize(work_dir, concepts_dir, output_dir, api_state)

        # Manifest Reviewer — the whole reason this launcher exists.
        _tab_manifest_reviewer(last_manifest_state, work_dir)

        # API keys, install-check, python-exe override (polished in step 7).
        _tab_settings(api_state)

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
