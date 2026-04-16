# Architecture — for agents and contributors

This is the dev-facing map of the launcher. The user-facing doc is
[README.md](../README.md); the user-facing tutorial lives in the
Directions tab inside the UI itself.

## The one invariant

The UI **shells out** to the Klippbok CLI. It never imports Klippbok
internals. This keeps the launcher decoupled from Klippbok's API
churn — when Klippbok upgrades, the only thing that might break is
form-field names, which are authoritatively listed in
[`cli_help.txt`](./cli_help.txt) (regenerated on every install by
`app/dump_help.py`, invoked from `install.js`).

If you find yourself writing `import klippbok` anywhere outside of
`dump_help.py` or a diagnostic script, you're doing it wrong.

## File map

```
/
├── pinokio.js         dynamic Pinokio menu (installed / running / idle)
├── install.js         uv venv --python 3.11 env
│                      → uv pip install -r app/requirements.txt
│                      → script.start torch.js
│                      → python dump_help.py
├── start.js           daemon: python -u app.py --host 127.0.0.1 --port {{port}}
│                      on /(http:\/\/[0-9.:]+)/ → local.set({url})
├── update.js          git pull + re-sync deps + re-torch + re-dump help
├── reset.js           fs.rm env/ (keeps app/, cache/, .env)
├── torch.js           platform-aware torch install (NVIDIA/AMD/Mac/CPU)
├── icon.png           Pinokio menu icon
├── README.md          user-facing install + workflow + troubleshooting
├── AGENTS.md          cross-agent CLAUDE-NOTE convention
├── .env.example       API key template
├── .gitignore         env/, cache/, .env, docs/cli_help.txt, *.pyc, etc.
├── app/
│   ├── app.py         the whole Gradio UI
│   ├── runner.py      subprocess streaming + cancel + UTF-8 env
│   ├── manifest.py    Manifest Reviewer schema adapter + thumbnail cache
│   ├── dump_help.py   called from install.js; writes docs/cli_help.txt
│   └── requirements.txt
└── docs/
    ├── ARCHITECTURE.md    this file
    ├── ROADMAP.md         deferred work + ideas
    └── cli_help.txt       (generated at install; gitignored)
```

## Pinokio launcher conventions

Ported from [`pinokiofactory/z-image-turbo`](https://github.com/pinokiofactory/z-image-turbo)
with the uv pattern from [`cocktailpeanut/ace-step.pinokio`](https://github.com/cocktailpeanut/ace-step.pinokio).
When conventions change upstream, these are the authoritative sources — re-fetch via
`gh api repos/OWNER/REPO/contents/FILE --jq .content | base64 -d`, don't paraphrase from a web summary.

- **venv location.** `env/` lives at the repo root (not `app/env/`).
  Achieved by passing `venv: "../env"` with `path: "app"` in every
  `shell.run`. `pinokio.js`'s `info.exists("env")` gates the
  "installed" menu state.
- **URL capture.** `start.js` uses `on: [{event: "/(http:\\/\\/[0-9.:]+)/", done: true}]`
  to match Gradio's "Running on local URL:" line, then
  `local.set({url: "{{input.event[1]}}"})`. Pinokio surfaces **Open
  Web UI** pointing at the captured URL.
- **Buffering.** `start.js` runs `python -u app.py` **and** sets
  `env: { PYTHONUNBUFFERED: "1" }`. Both are needed — without them
  Python buffers stdout when captured by a non-TTY parent and the URL
  regex never matches until the process exits. Caught during initial
  development; do not remove either.
- **Port.** `start.js` passes `--port {{port}}` (Pinokio allocates a
  free one). `app.py` accepts `--host` and `--port` and honors them
  in `demo.launch(server_name=..., server_port=...)`. Never hardcode
  7860 in either file.
- **Torch.** `torch.js` is called by `install.js` after the main
  `uv pip install` so it can overwrite torch with a GPU-appropriate
  build. Branches on `{{gpu}}` + `{{platform}}` + `{{arch}}`.
  Versions are pinned (currently torch 2.4.1 on CUDA 12.4 /
  ROCm 6.1 / CPU; torch 2.2.2 on Intel Mac). Upgrade deliberately.

## Gradio app architecture (`app/app.py`)

Single file by design — the whole UI fits in one readable module
(~1,400 lines). If you ever split it, keep `runner.py` and
`manifest.py` separate as they are; they're independently testable.

### Tab order (matters — this is the user-facing mental sequence)

```
Directions → Project → Scan → Triage → Ingest → Normalize
          → Caption → Score → Extract → Audit → Validate
          → Organize → Manifest Reviewer → Settings
```

Directions is first because first-timers land there. Settings is last
so returning users can reach their API keys without scrolling.

### Shared state wiring

Three `gr.State` objects live at the top of `build_ui()` and are
handed to the tabs that need them:

- **`work_dir`, `concepts_dir`, `output_dir`** — these are
  `gr.Textbox` components (not `gr.State`, since the user types into
  them). They live on the Project tab. Every command tab has its own
  directory textbox, and calls `_sync(project_textbox, tab_textbox)`
  which wires a `.change()` handler pushing the project value into
  the tab box. Per-tab overrides stick because Gradio's `.change`
  doesn't refire on programmatic updates coming from the tab box
  itself (the Project textbox only fires when the user edits it
  directly).

- **`api_state`** — `gr.State(dict)` hydrated from `.env` on startup
  via `_load_env()`. Settings tab textboxes update it on `.change`.
  Caption / Audit / Ingest (when `--caption` is checked) pass it into
  `runner.stream_command`'s `extra_env=` param so API keys land in
  the subprocess environment without touching `os.environ`.

- **`last_manifest_state`** — `gr.State(str)`. Triage's Run button
  chains `.then(_after_run)` which calls `_detect_triage_manifest`
  (looks at `--output` override or picks the newer of
  `triage_manifest.json` / `scene_triage_manifest.json` in the clips
  dir) and writes the path into the state. Manifest Reviewer
  subscribes via `last_manifest_state.change`, auto-filling its path
  textbox but only if the user hasn't typed anything (don't clobber
  in-flight work).

### Tab builder signatures

Every command tab is a function:

```python
def _tab_foo(work_dir, concepts_dir, output_dir, api_state) -> None: ...
```

Uniform signature by convention so `build_ui()` could loop, but the
loop was unrolled so Triage and the Reviewer can take the extra
`last_manifest_state`:

```python
_tab_triage(work_dir, concepts_dir, output_dir, api_state, last_manifest_state)
_tab_manifest_reviewer(last_manifest_state, work_dir)
```

### The `_command_shell` helper

All ten command tabs share the same outer frame:

```python
s = _command_shell(tab_id, title, description, dir_label)
# s = {"directory", "options", "dry_run", "run_btn", "cancel_btn", "log"}
_sync(work_dir, s["directory"])
with s["options"]:
    # per-command form fields (flags)
def _run(...):
    # build cmd list
    if dry: yield _dry_preview(cmd); return
    yield from _stream(tab_id, cmd, extra_env=...)  # or without extra_env
s["run_btn"].click(_run, inputs=[...], outputs=s["log"])
```

This is why the code is copy-pasty but low per-tab — each command
tab is about 30-50 lines, mostly the specific flag list.

### Subprocess pattern (`app/runner.py`)

```python
stream_command(tab_id, cmd, extra_env=None) -> Iterator[str]
```

- `Popen(stdout=PIPE, stderr=STDOUT, bufsize=1, text=True, encoding="utf-8", errors="replace")`.
  All four encoding-related settings matter on Windows; do not omit
  any.
- `env` merges `os.environ` + `PYTHONUNBUFFERED=1` + `PYTHONUTF8=1` +
  `PYTHONIOENCODING=utf-8` + the caller's `extra_env`. The UTF-8
  trio is what keeps Klippbok's internal `ffprobe` reader from
  crashing with a `UnicodeDecodeError: 'charmap' codec` when a
  filename contains non-ASCII characters.
- Yields per line as they arrive, then a final
  `[exit=N  elapsed=Ns]` footer.
- Active `Popen` is stored in a module-level `_ACTIVE` dict keyed by
  `tab_id`, guarded by a `threading.Lock`. `cancel(tab_id)` calls
  `terminate()` then `kill()` after 3s if still alive.
- **Never** use `subprocess.run(..., capture_output=True)` here. It
  buffers the entire run and defeats the whole streaming pattern.

### Manifest Reviewer (`app/manifest.py` + `_tab_manifest_reviewer`)

**Schema adapter.** Klippbok writes two shapes:

- `triage_manifest.json` — clip-level, flat `clips[]`, each with
  `include` and `matches[]`.
- `scene_triage_manifest.json` — scene-level, `videos[]` each with
  a nested `scenes[]` array and `triage_mode: "scene"` at the top.

`load_manifest()` detects via `triage_mode == "scene"` or presence
of `videos`, flattens either into a list of `Entry` dataclasses
with writeback pointers (`clip_idx` OR `video_idx + scene_idx`).
`save_manifest()` walks entries and writes `include` back into the
original `raw` dict — **any field the reviewer doesn't touch is
preserved byte-for-byte**. Round-trip losslessness is a feature;
don't break it.

**Thumbnails.** `generate_thumbnail(entry, cache_dir)` shells out
to `ffmpeg -ss <mid> -i <path> -frames:v 1 -vf scale=...`. Fast
seek (`-ss` before `-i`) trades frame accuracy for ~20x speed —
fine for thumbnails. Cache key is
`sha1(video_path + start + end + kind)[:20]`. Cached files live in
`<repo>/cache/thumbs/` (gitignored).

**UI pagination.** 25 pre-rendered slots. Each slot is
`[gr.Image, gr.Markdown, gr.Checkbox]` inside a `gr.Row(visible=False)`.
On page change, `_render_page(state)` returns a flat list of 2 +
25×4 = 102 updates:
`[status, page_label, row_0_visibility, image_0, md_0, checkbox_0, row_1_visibility, ...]`.
Thumbnails for the new page are generated in a
`ThreadPoolExecutor(max_workers=8)` — page turns show the new text
immediately and fill in images as ffmpeg returns.

**Per-slot checkbox handlers.** Use `.input(...)` not `.change(...)`.
`.change` fires on any value update including programmatic ones from
bulk actions; `.input` fires only on user clicks. Using `.change`
would feedback-loop every time "Include all" set 25 checkbox values.

### API surface and form fields

**Authoritative source: `docs/cli_help.txt`.** Regenerated on every
install by `dump_help.py`. When Klippbok adds / renames / removes a
flag, diff this file (`git diff cli_help.txt` won't work since it's
gitignored — run `python app/dump_help.py` before and after, diff by
hand) and update the matching form fields in `app.py`.

Drift caught during initial build:
- `extract --template` takes a path argument; it is NOT a boolean
  flag. UI uses a textbox for it.
- `organize --trainer` is free-form text, not argparse `choices=`.
  UI uses `gr.Dropdown(multiselect=True, allow_custom_value=True)`
  with `musubi` / `aitoolkit` as suggestions.
- `caption --tags` is `nargs="+"` — space-separated in the textbox,
  split with `.split()` at cmd-assembly time.

## Key gotchas (don't re-discover)

Also see [`.claude/learnings.md`](../.claude/learnings.md) for the
one-liner log of these as they were hit during development.

1. **Gradio 5, not 4.** Gradio 4.44 didn't cap `huggingface-hub<1`,
   so it resolved to 1.x which had dropped `HfFolder`, crashing
   Gradio's import. Requirements pin `gradio>=5,<6`.
2. **`python -u` AND `PYTHONUNBUFFERED=1`.** Required in `start.js`
   or the URL regex never matches. Belt and suspenders — keep both.
3. **UTF-8 everywhere in subprocess.** `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`
   in the env, plus explicit `encoding="utf-8" errors="replace"` on
   the Popen itself. Windows' default cp1252 decoder crashes on any
   non-ASCII byte in ffprobe output.
4. **`gr.Markdown` doesn't accept `scale=` in Gradio 5.** The Row's
   other components (Image / Checkbox) use `scale=0` and Markdown
   fills the remainder.
5. **`.input` vs `.change` for checkboxes.** Bulk actions set
   checkbox values programmatically; `.change` would feedback-loop.
6. **`_sync(source, target)` is one-way.** Source is the Project
   tab's textbox; target is the per-tab box. The per-tab box
   setting itself doesn't push back to the source (intentional —
   per-tab overrides shouldn't overwrite global defaults).
7. **Triage's concepts folder must be populated by hand.** The UI
   pre-flight check in `_tab_triage._run` rejects empty / imageless
   concepts dirs with a message pointing at the Directions tab.
   Don't silently fall through.

## How to add a new command tab (recipe)

Say Klippbok gains a new subcommand `python -m klippbok.video foo`.

1. Regenerate `cli_help.txt` (via `python app/dump_help.py`) and add
   `klippbok.video foo` to the `SUBCOMMANDS` list in `dump_help.py`
   so future installs pick it up.
2. In `app.py`, add `_tab_foo(work_dir, concepts_dir, output_dir, api_state)`:
   ```python
   def _tab_foo(work_dir, _concepts, _output, _api) -> None:
       tab_id = "foo"
       with gr.Tab("Foo"):
           s = _command_shell(tab_id, "klippbok.video foo", "description...", "Directory")
           _sync(work_dir, s["directory"])
           with s["options"]:
               # form fields matching --help output
           def _run(directory, ..., dry):
               cmd = _base_cmd("klippbok.video", "foo", directory)
               # conditionally append flags
               if dry: yield _dry_preview(cmd); return
               yield from _stream(tab_id, cmd)
           s["run_btn"].click(_run, inputs=[...], outputs=s["log"])
   ```
3. Add to `build_ui()` at the right pipeline position.
4. Add a row to the Directions tab reference table in
   `DIRECTIONS_MD` (search for "## What each tab does").

## How to change a form field safely

1. Run `python app/dump_help.py` to refresh `docs/cli_help.txt`.
2. Compare the relevant block with what's in `app.py`.
3. Update the form field. Mirror flag name, type, default, and
   choices from the help output verbatim.
4. Check whether Directions tab references the flag; update there
   too if so.

## Memory / persistence model

- `gr.State` objects are session-local. Browser refresh loses them.
  This is acceptable — users reload manifests and re-pick directories
  after a refresh. If this becomes a real pain point, see
  [ROADMAP.md](./ROADMAP.md).
- `.env` is the only disk persistence the UI does. Written by the
  Settings tab's Save button, read on startup. All other state
  (working directories, loaded manifest, caption progress) is
  in-memory.
- `cache/thumbs/` survives across sessions but is keyed by
  `(path, start, end, kind)` — if the underlying video file moves,
  the thumbnail cache still exists but is effectively orphaned. Safe
  to `rm -rf cache/` at any time.
