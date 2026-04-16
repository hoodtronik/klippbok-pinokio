"""Subprocess orchestration for the Klippbok UI.

Each command tab calls `stream_command` as a generator from its Gradio handler.
The generator yields stdout lines as they arrive so Gradio can incrementally
update the log textbox without blocking the event loop. Active Popen handles
are kept in a module-level dict keyed by tab id so the Cancel button can find
and terminate them.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Optional

# CLAUDE-NOTE: Keyed by tab id so two tabs can run commands concurrently
# without clobbering each other's Cancel handles. Guard with a Lock — the
# generator and Cancel handler run on different Gradio worker threads.
_ACTIVE: dict[str, subprocess.Popen] = {}
_LOCK = threading.Lock()


def format_command(cmd: list[str]) -> str:
    """Shell-safe one-line preview of a command list for the dry-run display."""
    return " ".join(shlex.quote(x) for x in cmd)


def stream_command(
    tab_id: str,
    cmd: list[str],
    extra_env: Optional[dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Iterator[str]:
    """Run `cmd`, yielding stdout lines as they arrive.

    Final yield is a `[exit=N  elapsed=Ss]` footer. Merges stderr into stdout
    so both streams are interleaved in display order. Never use
    capture_output=True here — it would buffer the entire run.
    """
    # CLAUDE-NOTE: PYTHONUTF8=1 puts the Klippbok child process (and any
    # subprocess it spawns — ffprobe, CLIP downloads, etc.) into Python's
    # UTF-8 Mode. Without this on Windows, Python defaults to cp1252 and any
    # non-ASCII byte in a video filename or ffmpeg output crashes an internal
    # reader thread with UnicodeDecodeError: 'charmap' codec can't decode ...
    # PYTHONIOENCODING is belt-and-suspenders for older paths that don't
    # consult PYTHONUTF8.
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v})

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            # CLAUDE-NOTE: Explicit encoding="utf-8" errors="replace" so OUR
            # read of the child's stdout survives any rogue bytes (e.g. an
            # ffmpeg progress line written in a different codec). Without
            # these, `text=True` falls back to locale.getpreferredencoding()
            # which is cp1252 on Windows — same crash as the child.
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        yield f"[error] {exc}"
        return

    with _LOCK:
        _ACTIVE[tab_id] = proc
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
        proc.wait()
    finally:
        with _LOCK:
            _ACTIVE.pop(tab_id, None)

    elapsed = time.monotonic() - start
    yield f"\n[exit={proc.returncode}  elapsed={elapsed:.1f}s]"


def cancel(tab_id: str) -> str:
    """Terminate the running Popen for `tab_id`, escalating to kill on timeout."""
    with _LOCK:
        proc = _ACTIVE.get(tab_id)
    if proc is None or proc.poll() is not None:
        return "(nothing running)"
    proc.terminate()
    try:
        proc.wait(timeout=3)
        return f"[canceled — exit={proc.returncode}]"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return f"[killed after 3s — exit={proc.returncode}]"


def python_executable() -> str:
    """The python binary that runs Klippbok. Defaults to the launcher's own venv."""
    # CLAUDE-NOTE: When launched from Pinokio's start.js, sys.executable points
    # at env/Scripts/python.exe (Windows) or env/bin/python (POSIX). Settings
    # tab exposes an override for advanced users with a different venv.
    return sys.executable
