# Roadmap — deferred work + ideas

Things the build plan deliberately left for later. Nothing here is
a commitment — they're candidates for future sessions, ranked by
rough effort × user impact.

## Small, clearly useful

- **Per-clip Notes field in the Manifest Reviewer.** The `Entry`
  dataclass already carries a notes slot; `save_manifest` doesn't
  write it back. Adding: one `gr.Textbox` per slot + a `.input`
  handler + a writeback branch in `save_manifest`. ~30 lines.
- **"Re-dump help" button in Settings.** Currently only
  `install.js` / `update.js` run `dump_help.py`. A button that
  invokes it on demand would let users verify the UI matches a
  manually-upgraded Klippbok. `subprocess.run([sys.executable, "dump_help.py"], cwd=...)`
  + show the result in the Settings status textbox. ~20 lines.
- **Concept subfolder quick-add in the Project scaffold.** Today
  the scaffold creates `concepts/` with just a README. Adding a
  "Concepts to pre-create (one per line)" textbox that spawns
  subfolders would shorten the new-user path. ~30 lines.
- **Open concepts folder button.** After scaffolding, a button
  that runs `os.startfile(concepts_path)` (Windows) /
  `subprocess.Popen(["open", path])` (macOS) /
  `xdg-open` (Linux). Takes the user directly to where they need
  to drop reference images.

## Medium effort, nice to have

- **Python executable override.** Settings tab already has a
  textbox for `sys.executable` (read-only). Wiring it to
  `runner.python_executable()` via a module-level shared var would
  let advanced users point at a different venv without editing
  code. Gotcha: mid-run changes should not affect in-flight
  subprocesses. ~60 lines.
- **Progress bars per tab.** Gradio 5 has `gr.Progress()` as a
  streaming component. For long-running Triage / Ingest, parsing a
  line like `Processing 13/47` out of Klippbok stdout and feeding
  it to `gr.Progress()` would give users something visual. Needs
  a regex per subcommand since Klippbok's progress format varies.
- **Cancel timeout as a setting.** Currently hardcoded at 3s in
  `runner.cancel()`. Some long-running triage runs need longer to
  unwind cleanly. Make it configurable from Settings.
- **Rerun button on a failed command tab.** Currently users re-click
  Run. A "Rerun with same args" button after a non-zero exit would
  save clicks during debugging. Stash the last `cmd` list per tab.

## Big, speculative

- **Manifest Reviewer: video preview on click.** Clicking a
  thumbnail opens a modal that plays the clip / scene. Needs a
  ffmpeg → WebM / HLS pipeline or just `gr.Video` with the source
  file. Might be too heavy for a reviewer whose job is quick
  triage.
- **Session state persistence.** Today a browser refresh loses
  `gr.State`. Saving state to a JSON in `cache/session/` on any
  state change and restoring on load would smooth restarts.
  Scope creep risk: "save everything" rarely stays simple.
- **Multi-user mode.** Currently the launcher assumes one user
  because it's a local Pinokio app. Not a real need, but if
  someone wanted to host it, `gr.State` per session + queue
  tuning would be the starting point.
- **Pipeline runner tab.** A single "Run full pipeline" tab that
  chains Scan → Triage → Reviewer (pause for human) → Ingest →
  Caption → Validate → Organize with configurable parameters per
  stage. Replaces the "click Run on 8 tabs in sequence" flow.
  Needs the Reviewer to emit a programmatic "done" signal the
  pipeline can wait on, which would be new infrastructure.

## Definitely not doing unless asked

- Rewriting this as something other than a Gradio wrapper. The
  whole premise is "wrap the CLI minimally". A from-scratch UI
  would replicate Klippbok's argparse surface for no benefit.
- Importing Klippbok internals. See
  [ARCHITECTURE.md](./ARCHITECTURE.md#the-one-invariant).
- Adding features Klippbok itself doesn't expose. This is a
  launcher, not a fork.
