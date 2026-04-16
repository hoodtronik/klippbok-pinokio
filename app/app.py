"""Klippbok Pinokio launcher — Gradio UI.

# TODO (build plan — ~/.claude/plans/clever-sleeping-emerson.md):
#   [x] Step 2 — Pinokio skeleton + hello-world Gradio
#   [x] Step 3 — klippbok[all] + torch.js + help dump
#   [x] Step 4 — Project tab + Scan tab
#   [>] Step 5 — all command tabs + minimal Settings (this checkpoint)
#   [ ] Step 6 — Manifest Reviewer (both schemas, thumbnails, bulk actions)
#   [ ] Step 7 — Settings polish + README walkthrough
"""
from __future__ import annotations

import argparse
from typing import Iterator

import gradio as gr

import runner


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
            if not concepts:
                yield "[error] --concepts is required for triage."
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


def _tab_settings() -> gr.State:
    """Minimal Settings tab — just API keys for now.

    Returns a gr.State holding {"GEMINI_API_KEY": str, "REPLICATE_API_TOKEN": str}.
    Step 7 will flesh this out with env persistence, python-exe override,
    and install-check tooling.
    """
    api_state = gr.State({})
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

    return api_state


# ----- ui ------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Klippbok", analytics_enabled=False) as demo:
        gr.Markdown("# Klippbok\n_Video dataset curation — Pinokio launcher_")

        # Project tab — shared state for every downstream tab.
        with gr.Tab("Project"):
            gr.Markdown(
                "### Working directories\n"
                "Set these once — every command tab starts with the matching path"
                " as its default. Override per-tab as needed."
            )
            work_dir = _folder_row("Working directory", "Where your raw video clips live.")
            concepts_dir = _folder_row("Concepts directory", "Reference images for triage (subfolders per concept).")
            output_dir = _folder_row("Output directory", "Where Klippbok writes processed clips + manifests.")

        # Settings built early so API key state is available to caption/audit/ingest tabs.
        # CLAUDE-NOTE: Tab display order is insertion order, so Settings is
        # visually last — we render the other tabs first but keep a forward
        # reference to the api state. Simplest way: build Settings first, keep
        # its state, then build the rest. Gradio will order them by code order.
        api_state = _tab_settings()

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

        # Manifest Reviewer placeholder (step 6).
        with gr.Tab("Manifest Reviewer"):
            gr.Markdown(
                "_**Manifest Reviewer** — under construction. "
                "Will load `triage_manifest.json` or `scene_triage_manifest.json`, "
                "render thumbnails per entry, and let you toggle `include` flags in bulk. "
                "Arriving in the next checkpoint._"
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
