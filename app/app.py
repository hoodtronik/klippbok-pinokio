"""Klippbok Pinokio launcher — Gradio UI.

# TODO (build plan — ~/.claude/plans/clever-sleeping-emerson.md):
#   [x] Step 2 — Pinokio skeleton + hello-world Gradio
#   [x] Step 3 — klippbok[all] + torch.js + help dump
#   [>] Step 4 — Project tab + Scan tab (this checkpoint)
#   [ ] Step 5 — replicate pattern for Triage/Ingest/Normalize/Caption/Score/
#                 Extract/Audit/Validate/Organize
#   [ ] Step 6 — Manifest Reviewer (both schemas, thumbnails, bulk actions)
#   [ ] Step 7 — Settings polish + README walkthrough
"""
from __future__ import annotations

import argparse
from typing import Iterator

import gradio as gr

import runner


# --------------------------------------------------------------------- helpers


def _pick_folder() -> str:
    """Open a native folder dialog server-side. Empty string if unavailable.

    In Pinokio the server == the user's machine, so the dialog opens locally.
    If tkinter isn't importable (some headless Python builds), users can paste
    a path manually — the Browse button is a convenience, not a requirement.
    """
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
    """A [textbox] [Browse…] row. Returns the textbox for further wiring."""
    with gr.Row():
        box = gr.Textbox(label=label, info=hint, scale=4, interactive=True)
        btn = gr.Button("Browse…", scale=0, size="sm", min_width=0)
        btn.click(_pick_folder, outputs=box)
    return box


# --------------------------------------------------------------------------- ui


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Klippbok", analytics_enabled=False) as demo:
        gr.Markdown("# Klippbok\n_Video dataset curation — Pinokio launcher_")

        # ----- Project tab (shared directory state) ------------------------
        with gr.Tab("Project"):
            gr.Markdown(
                "### Working directories\n"
                "Set these once — every command tab starts with these paths as"
                " defaults. Override per-tab as needed."
            )
            work_dir = _folder_row(
                "Working directory",
                "Where your raw video clips live.",
            )
            concepts_dir = _folder_row(
                "Concepts directory",
                "Reference images for triage (subfolders per concept).",
            )
            output_dir = _folder_row(
                "Output directory",
                "Where Klippbok writes processed clips + manifests.",
            )

        # ----- Scan tab ----------------------------------------------------
        with gr.Tab("Scan"):
            gr.Markdown(
                "### `klippbok.video scan`\n"
                "Probe a directory of video clips for resolution, fps, frame"
                " count, and codec issues. Read-only — never modifies files."
            )
            scan_dir = gr.Textbox(
                label="Directory (positional)",
                info="Defaults to the Working directory from the Project tab.",
                interactive=True,
            )
            # CLAUDE-NOTE: The project tab's work_dir pushes its value to every
            # per-command tab's directory textbox on change. Users can still
            # override per-tab afterwards — Gradio only re-syncs when the
            # Project textbox itself changes, not the per-tab one.
            work_dir.change(lambda v: v, inputs=work_dir, outputs=scan_dir)

            with gr.Accordion("Options", open=True):
                with gr.Row():
                    scan_fps = gr.Number(
                        label="--fps",
                        value=16,
                        precision=0,
                        info="Target frame rate. Default 16 (Wan models).",
                    )
                    scan_verbose = gr.Checkbox(
                        label="--verbose",
                        value=False,
                        info="Show full per-clip details instead of grouped summary.",
                    )
                scan_config = gr.Textbox(
                    label="--config (optional)",
                    info="Path to klippbok_data.yaml.",
                    value="",
                )

            with gr.Row():
                scan_dry = gr.Checkbox(label="Dry run", value=False, scale=0)
                scan_run = gr.Button("Run", variant="primary", scale=0)
                scan_cancel = gr.Button("Cancel", scale=0)

            scan_log = gr.Textbox(
                label="Output",
                lines=20,
                max_lines=2000,
                autoscroll=True,
                interactive=False,
                show_copy_button=True,
            )

            def _run_scan(directory, config, fps, verbose, dry) -> Iterator[str]:
                if not directory:
                    yield "[error] Directory is required. Set one in the Project tab or here."
                    return
                cmd = [
                    runner.python_executable(),
                    "-m",
                    "klippbok.video",
                    "scan",
                    directory,
                ]
                if config:
                    cmd += ["--config", config]
                if fps:
                    cmd += ["--fps", str(int(fps))]
                if verbose:
                    cmd.append("--verbose")

                if dry:
                    yield "$ " + runner.format_command(cmd) + "\n[dry run — not executed]"
                    return

                buf = ""
                for line in runner.stream_command("scan", cmd):
                    buf += line + "\n"
                    yield buf

            scan_run.click(
                _run_scan,
                inputs=[scan_dir, scan_config, scan_fps, scan_verbose, scan_dry],
                outputs=scan_log,
            )
            scan_cancel.click(lambda: runner.cancel("scan"), outputs=scan_log)

        # ----- Placeholder tabs for the other commands --------------------
        for name in (
            "Triage",
            "Ingest",
            "Normalize",
            "Caption",
            "Score",
            "Extract",
            "Audit",
            "Validate",
            "Organize",
            "Manifest Reviewer",
            "Settings",
        ):
            with gr.Tab(name):
                gr.Markdown(
                    f"_**{name}** — under construction. "
                    f"Ships alongside the other command tabs in the next checkpoint._"
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
