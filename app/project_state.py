"""Project-state persistence for the Klippbok Gradio UI.

Users close/reopen the launcher constantly. Without persistence, every
reopen is a from-scratch ceremony (retype the paths, re-pick the
provider, rewire the thresholds). This module snapshots the UI's scalar
form values to `<work_dir>/klippbok_project.json` whenever something
meaningful changes, and restores them on app start.

Design choices:

* Project file lives in the user's **working directory**, not the app
  install folder, so the settings travel with the dataset if they zip
  and share it.
* `.user_settings.json` (managed by `pipeline_installer`) gets two new
  keys: `last_project_dir` and `recent_projects` (capped at 5) — these
  are user-specific, not per-project.
* Tabs opt in by calling `ProjectState.register(tab_id, **components)`
  during build. Pagination state, logs, API keys, and the Manifest
  Reviewer's per-clip `include` flags are intentionally NOT persisted;
  the project file is a human-readable snapshot of the form, not a
  session replay.
* Tab "completion" = Run button was clicked at least once (tracked via
  `register_run` + a single `tabs_run` list in the project file).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

import pipeline_installer as pi


SCHEMA_VERSION = 1
PROJECT_FILE_NAME = "klippbok_project.json"
USER_SETTINGS_RECENT_KEY = "recent_projects"
USER_SETTINGS_LAST_KEY = "last_project_dir"
RECENT_MAX = 5


class ProjectState:
    """Registry of the components that participate in save/load.

    Populated during tab construction, consumed by the wiring pass in
    `app.build_ui` once every tab has been built. Deterministic key
    ordering (sorted tab_id → sorted field_name) keeps the save-handler
    output indices stable across Python runs.
    """

    def __init__(self) -> None:
        self._fields: dict[str, dict[str, gr.components.Component]] = {}
        self._run_buttons: dict[str, gr.Button] = {}

    def register(self, tab_id: str, **components: gr.components.Component) -> None:
        self._fields.setdefault(tab_id, {}).update(components)

    def register_run(self, tab_id: str, run_btn: gr.Button) -> None:
        self._run_buttons[tab_id] = run_btn

    def tab_ids(self) -> list[str]:
        return sorted(self._fields)

    def run_buttons(self) -> dict[str, gr.Button]:
        return dict(self._run_buttons)

    def get_field(self, tab_id: str, field_name: str) -> gr.components.Component | None:
        """Look up a registered component by (tab_id, field_name).

        Returns None if either the tab or the field hasn't been
        registered yet. Used by build_ui's preset wiring to grab the
        cross-tab handles it needs to update on dropdown change.
        """
        return self._fields.get(tab_id, {}).get(field_name)

    def ordered_components(self) -> list[gr.components.Component]:
        out: list[gr.components.Component] = []
        for tab_id in sorted(self._fields):
            for field_name in sorted(self._fields[tab_id]):
                out.append(self._fields[tab_id][field_name])
        return out

    def ordered_keys(self) -> list[tuple[str, str]]:
        """(tab_id, field_name) pairs in the same order as ordered_components."""
        out: list[tuple[str, str]] = []
        for tab_id in sorted(self._fields):
            for field_name in sorted(self._fields[tab_id]):
                out.append((tab_id, field_name))
        return out

    def pack_values(self, values: list[object]) -> dict[str, dict[str, object]]:
        """values (from a Gradio event) -> {tab_id: {field_name: value}}."""
        packed: dict[str, dict[str, object]] = {}
        for (tab_id, field_name), val in zip(self.ordered_keys(), values):
            packed.setdefault(tab_id, {})[field_name] = val
        return packed

    def unpack_values(self, packed: dict[str, dict[str, object]]) -> list[object]:
        """Return values in ordered_components order; missing fields -> gr.update()."""
        out: list[object] = []
        packed = packed or {}
        for tab_id, field_name in self.ordered_keys():
            tab = packed.get(tab_id, {}) or {}
            if field_name in tab:
                out.append(gr.update(value=tab[field_name]))
            else:
                out.append(gr.update())
        return out


# --------------------------------------------------------------- file paths


def project_file_in(work_dir: str) -> Path | None:
    """Return the project-file path inside work_dir, or None if invalid."""
    if not work_dir:
        return None
    p = Path(work_dir)
    if not p.is_dir():
        return None
    return p / PROJECT_FILE_NAME


def project_file_exists(work_dir: str) -> bool:
    pf = project_file_in(work_dir)
    return bool(pf and pf.is_file())


# --------------------------------------------------------------- persistence


def _now_iso() -> str:
    # Trimmed to whole seconds — minute-level granularity is all the UI shows.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _update_recent(work_dir: str) -> list[str]:
    """Prepend work_dir into recent_projects (dedup, cap RECENT_MAX).

    Also updates `last_project_dir` so the next app start auto-loads.
    """
    s = pi.load_user_settings()
    recent = s.get(USER_SETTINGS_RECENT_KEY, []) or []
    recent = [p for p in recent if p and p != work_dir]
    recent.insert(0, work_dir)
    recent = recent[:RECENT_MAX]
    s[USER_SETTINGS_RECENT_KEY] = recent
    s[USER_SETTINGS_LAST_KEY] = work_dir
    pi.save_user_settings(s)
    return recent


def load_recent() -> list[str]:
    """Best-effort read of recent_projects from .user_settings.json."""
    try:
        s = pi.load_user_settings()
    except Exception:
        return []
    return [p for p in (s.get(USER_SETTINGS_RECENT_KEY, []) or []) if p]


def load_last_project_dir() -> str:
    """Best-effort read of last_project_dir from .user_settings.json."""
    try:
        s = pi.load_user_settings()
    except Exception:
        return ""
    return s.get(USER_SETTINGS_LAST_KEY, "") or ""


def project_name(work_dir: str) -> str:
    """Derive a display name from the working directory path."""
    if not work_dir:
        return ""
    return Path(work_dir).name or work_dir


def format_status(
    work_dir: str,
    saved_at_iso: str,
    tabs_run: list[str],
    error: str = "",
) -> str:
    """Build the Markdown shown in the Project-tab status row."""
    if error:
        return f"**Project:** error — {error}"
    if not work_dir:
        return "**No project loaded** — pick a working directory below."
    name = project_name(work_dir)
    time_part = f" — last saved `{saved_at_iso}`" if saved_at_iso else " — not yet saved"
    completed = (
        f"  \nCompleted steps: {', '.join(tabs_run)}"
        if tabs_run
        else "  \nCompleted steps: _none yet_"
    )
    return f"**Project:** `{name}`{time_part}{completed}"


def build_payload(
    work_dir: str,
    concepts_dir: str,
    output_dir: str,
    tabs_run: list[str],
    propagation: dict[str, str],
    fields_by_tab: dict[str, dict[str, object]],
) -> dict:
    """Shape the JSON we write. Schema version lives here so future
    migrations can branch on it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": _now_iso(),
        "project": {
            "work_dir": work_dir or "",
            "concepts_dir": concepts_dir or "",
            "output_dir": output_dir or "",
        },
        "paths": {k: (v or "") for k, v in (propagation or {}).items()},
        "tabs_run": list(tabs_run or []),
        "fields": fields_by_tab or {},
    }


def save_project(work_dir: str, payload: dict) -> tuple[str, str]:
    """Write payload to `<work_dir>/klippbok_project.json`.

    Returns (saved_at, error). On failure saved_at is "" and error
    describes why — callers show this in the status row without
    raising (the UI should never crash from a disk glitch).
    """
    target = project_file_in(work_dir)
    if not target:
        return "", "no working directory set"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return "", f"write failed: {exc}"
    try:
        _update_recent(work_dir)
    except Exception:
        pass  # recent_projects is a nicety, not worth failing saves over.
    return payload.get("saved_at", _now_iso()), ""


def read_project_file(path: str) -> tuple[dict | None, str]:
    """Load JSON from `path`. Returns (payload, error). Never raises.

    Forward compatibility: extra fields in the file are preserved (we
    hand the caller the raw dict), missing fields fall through to the
    UI's defaults because unpack_values emits `gr.update()` for them.
    """
    if not path:
        return None, "no path"
    p = Path(path)
    if not p.is_file():
        return None, f"not a file: {path}"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read project file: {exc}"
    if not isinstance(raw, dict):
        return None, "project file is not a JSON object"
    return raw, ""


def read_project_in_dir(work_dir: str) -> tuple[dict | None, str]:
    pf = project_file_in(work_dir)
    if not pf or not pf.is_file():
        return None, ""
    return read_project_file(str(pf))


# --------------------------------------------------------------- field helpers


def extract_paths(payload: dict) -> dict[str, str]:
    """Read the propagation paths (triage / ingest / caption) from a payload."""
    return {k: (v or "") for k, v in (payload.get("paths") or {}).items()}


def extract_project_dirs(payload: dict) -> tuple[str, str, str]:
    prj = payload.get("project") or {}
    return (
        prj.get("work_dir", "") or "",
        prj.get("concepts_dir", "") or "",
        prj.get("output_dir", "") or "",
    )


def extract_tabs_run(payload: dict) -> list[str]:
    return [t for t in (payload.get("tabs_run") or []) if isinstance(t, str)]


def extract_fields(payload: dict) -> dict[str, dict[str, object]]:
    f = payload.get("fields") or {}
    if not isinstance(f, dict):
        return {}
    return f
