"""Klippbok Pinokio launcher — Gradio UI.

Step-2 placeholder. Verifies the Pinokio Install -> Start -> Open Web UI loop
works before wiring up any Klippbok subcommand tabs.

# TODO (build plan, see /.claude/plans/clever-sleeping-emerson.md):
#   [ ] Step 3 — add klippbok[all] to requirements.txt + dump --help
#   [ ] Step 4 — Project tab + Scan tab with streaming-log pattern (pause here)
#   [ ] Step 5 — replicate pattern for Triage/Ingest/Normalize/Caption/Score/
#                Extract/Audit/Validate/Organize
#   [ ] Step 6 — Manifest Reviewer (both schemas, thumbnails, bulk actions)
#   [ ] Step 7 — Settings polish + README walkthrough
"""

import argparse

import gradio as gr


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Klippbok", analytics_enabled=False) as demo:
        gr.Markdown(
            """
            # Klippbok
            ### Video dataset curation — Pinokio launcher smoke test

            If you can see this page, the Pinokio **Install → Start → Open Web UI**
            loop works end-to-end. The real tabbed UI ships in the next checkpoint:

            - **Project** — working dir, concepts dir, output dir (global state)
            - **Scan / Triage / Ingest / Normalize / Caption / Score / Extract / Audit / Validate / Organize** — one tab per Klippbok subcommand
            - **Manifest Reviewer** — visual review of `scene_triage_manifest.json`
            - **Settings** — API keys, install check

            See the build plan at
            `~/.claude/plans/clever-sleeping-emerson.md` (local to this machine).
            """
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Klippbok Gradio UI")
    # CLAUDE-NOTE: Pinokio's start.js passes --host / --port explicitly using
    # the {{port}} template var. inbrowser=False because Pinokio handles
    # surfacing the URL itself via the captured stdout match.
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
