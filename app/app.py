"""Klippbok Pinokio launcher — Gradio UI.

# TODO (build plan — ~/.claude/plans/clever-sleeping-emerson.md):
#   [x] Step 2 — Pinokio skeleton + hello-world Gradio
#   [x] Step 3 — klippbok[all] + torch.js + help dump
#   [x] Step 4 — Project tab + Scan tab
#   [x] Step 5 — all command tabs + minimal Settings
#   [>] Step 6 — Manifest Reviewer (this checkpoint)
#   [ ] Step 7 — Settings polish + README walkthrough
"""
from __future__ import annotations

import argparse
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

## Two common recipes

### A) Full pipeline — raw footage → trainable dataset

1. **Project tab** → Scaffold a new project layout (or browse to existing folders).
2. Open the **concepts/** folder (filesystem) and populate it: one subfolder per concept, 5-20 reference images each.
3. **Triage tab** → run against your raw clips. Produces a manifest.
4. **Manifest Reviewer tab** → fix any misclassifications CLIP made. Save a reviewed manifest.
5. **Ingest tab** → point at your raw video (file or directory) with `--triage <reviewed manifest>`. Klippbok scene-splits and writes clips into output/.
6. **Caption tab** → set --provider, --use-case, --anchor-word. Needs a Gemini or Replicate key in Settings.
7. **Validate tab** → with --buckets and --quality.
8. **Organize tab** → with --trainer = musubi or aitoolkit (or both, repeatable).

### B) Re-caption an existing dataset

1. **Project tab** → point Working directory at your existing clip dir.
2. **Settings tab** → paste API key.
3. **Caption tab** → check --overwrite.
4. **Score tab** → sanity-check.
5. **Audit tab** → compare before/after.

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
"""


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


def _tab_triage(work_dir, concepts_dir, _output, _api) -> None:
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

        s["run_btn"].click(
            _run,
            inputs=[s["directory"], concepts, threshold, frames, frames_per_scene, scene_threshold, output, organize, move, clip_model, s["dry_run"]],
            outputs=s["log"],
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
    """Minimal Settings tab — just API keys for now.

    Writes into the api_state gr.State provided by build_ui so command tabs
    can read the keys during their handlers. Step 7 will add env persistence,
    python-exe override, and install-check tooling.
    """
    with gr.Tab("Settings"):
        gr.Markdown(
            "### Settings\n"
            "API keys live here and are passed into Caption/Audit/Ingest(`--caption`) subprocesses "
            "via the environment. They are **not** saved to disk — keep them alive by leaving this "
            "tab configured for the session. Persistence and an install check land in the final polish step."
        )
        gemini = gr.Textbox(
            label="GEMINI_API_KEY",
            type="password",
            info="Required for `--provider gemini`.",
            value="",
        )
        replicate = gr.Textbox(
            label="REPLICATE_API_TOKEN",
            type="password",
            info="Required for `--provider replicate`.",
            value="",
        )

        def _update_keys(g, r):
            keys = {}
            if g:
                keys["GEMINI_API_KEY"] = g
            if r:
                keys["REPLICATE_API_TOKEN"] = r
            return keys

        gemini.change(_update_keys, inputs=[gemini, replicate], outputs=api_state)
        replicate.change(_update_keys, inputs=[gemini, replicate], outputs=api_state)


# ----- Manifest Reviewer --------------------------------------------------


def _render_entry_meta(e: mft.Entry) -> str:
    """Markdown block shown next to each thumbnail."""
    concept = e.best_concept or "_(no match)_"
    lines = [
        f"**{e.display_label}**",
        f"Score: `{e.score:.3f}` → `{concept}`",
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


def _tab_manifest_reviewer() -> None:
    with gr.Tab("Manifest Reviewer"):
        gr.Markdown(
            "### Manifest Reviewer\n"
            "Load a `triage_manifest.json` (clip-level) or `scene_triage_manifest.json` "
            "(scene-level). Thumbnails are extracted on demand via ffmpeg and cached in "
            "`cache/thumbs/`. Toggle `Include` per row, or use bulk actions below. Save "
            "writes to `<name>_reviewed.json` by default so your original is untouched."
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
            load_btn = gr.Button("Load", variant="primary", scale=0)

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

        # CLAUDE-NOTE: api_state is declared up front so the command tabs
        # (built in display order below) can reference it as an input while
        # the Settings tab (built last, to appear last in the tab strip)
        # writes into it via .change() handlers.
        api_state = gr.State({})

        # Landing / orientation.
        _tab_directions()

        # Shared directory state + scaffold helper.
        work_dir, concepts_dir, output_dir = _tab_project()

        # The Klippbok command tabs, in pipeline order.
        for builder in (
            _tab_scan,
            _tab_triage,
            _tab_ingest,
            _tab_normalize,
            _tab_caption,
            _tab_score,
            _tab_extract,
            _tab_audit,
            _tab_validate,
            _tab_organize,
        ):
            builder(work_dir, concepts_dir, output_dir, api_state)

        # Manifest Reviewer — the whole reason this launcher exists.
        _tab_manifest_reviewer()

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
