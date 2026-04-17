"""Pipeline Status panel — detection probes, install runners, persistence.

Extends the Agentic Pipeline tab with one-click install/update for every
companion tool the pipeline needs. Lives alongside `pipeline_setup.py`:

  * `pipeline_setup.py` owns the **workspace scaffold** (folder tree,
    watchdog, guide, agent instructions).
  * `pipeline_installer.py` (this module) owns the **install/detection
    surface** (what's on disk, what to clone, user-settings persistence).

Persistence: `.user_settings.json` at the repo root (gitignored) caches
the install root + last-known detected paths so the panel renders with
the right state immediately on app boot — no subprocess calls needed for
the fast path. Heavier probes (nvidia-smi CUDA version, `uv --version`,
`git --version`) run on "Refresh" or before an install.

Everything is **stdlib-only** (pathlib, json, os, re, shutil, subprocess)
— no new pip deps. Install subprocesses stream via `runner.stream_command`
so the existing log pattern (line-by-line yield into a gr.Textbox) is
reused verbatim.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import runner


# --------------------------------------------------------------- settings file


def _settings_path() -> Path:
    """Location of the user-settings JSON — repo root, sibling to pinokio.js."""
    return Path(__file__).resolve().parent.parent / ".user_settings.json"


def load_user_settings() -> dict:
    """Parse `.user_settings.json` or return an empty skeleton. Never raises."""
    p = _settings_path()
    if not p.exists():
        return {"pipeline_install_root": "", "detected_paths": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pipeline_install_root": "", "detected_paths": {}}
    # Fill in missing keys defensively — callers assume the skeleton exists.
    raw.setdefault("pipeline_install_root", "")
    raw.setdefault("detected_paths", {})
    return raw


def save_user_settings(data: dict) -> str:
    """Write settings and return the path written (for display)."""
    p = _settings_path()
    # CLAUDE-NOTE: indent=2 + sorted keys so git-diffs (if the file ever
    # gets accidentally committed) are readable. ensure_ascii=False keeps
    # non-ASCII path characters intact.
    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(p)


def update_detected_path(key: str, path: Optional[str]) -> None:
    """Merge a single detected path into the stored settings."""
    s = load_user_settings()
    detected = s.setdefault("detected_paths", {})
    if path:
        detected[key] = path
    else:
        detected.pop(key, None)
    save_user_settings(s)


# --------------------------------------------------------------- tool probes


def has_uv() -> bool:
    """`uv` binary on PATH (used by every install command)."""
    return shutil.which("uv") is not None


def has_git() -> bool:
    return shutil.which("git") is not None


def uv_version() -> Optional[str]:
    """`uv --version` -> '0.10.12' or None. Subprocess; don't call on hot path."""
    if not has_uv():
        return None
    try:
        r = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, timeout=5
        )
        m = re.search(r"uv\s+([\d.]+)", r.stdout or "")
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def git_version() -> Optional[str]:
    if not has_git():
        return None
    try:
        r = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        )
        m = re.search(r"git\s+version\s+([\d.]+)", r.stdout or "")
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_cuda_version() -> Optional[str]:
    """Parse 'CUDA Version: X.Y' from the nvidia-smi header. None if unavailable.

    `nvidia-smi --query-gpu=cuda_version` is NOT a valid query field — CUDA
    version only appears in the default text header. Parse that directly.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return None
        m = re.search(r"CUDA Version:\s+([\d.]+)", r.stdout or "")
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def musubi_cuda_extras() -> list[str]:
    """Map the detected CUDA version to the right `uv sync --extra` flags.

    Musubi's pyproject has extras named by CUDA minor version. As of writing:
    `cu128`, `cu124`. We pick the nearest match at-or-below the detected
    version, falling back to cu124 if detection fails (broadest compat).
    """
    cuda = detect_cuda_version()
    if not cuda:
        return ["--extra", "cu124", "--extra", "gui"]
    try:
        major, minor = cuda.split(".")[:2]
        v = int(major) * 10 + int(minor)
    except (ValueError, IndexError):
        return ["--extra", "cu124", "--extra", "gui"]
    if v >= 128:
        return ["--extra", "cu128", "--extra", "gui"]
    # 12.4 and below: cu124 (cu121 / cu118 historically existed but we
    # aren't generating variants we haven't verified against musubi).
    return ["--extra", "cu124", "--extra", "gui"]


# --------------------------------------------------------------- components


@dataclass
class Component:
    """One row in the Pipeline Status panel.

    `folder_name` is the directory name used when cloning into the install
    root. `clone_url` is the git remote. `install_cmds` is a list of
    subprocess arg-lists to run *inside* the cloned directory after clone
    — the runner joins them sequentially, so a failure in one skips the
    rest. `detect_path_under_root(root)` returns where this component
    lives *when installed via this panel* (i.e. `<root>/<folder_name>`);
    the broader `detect(...)` function also checks env vars and classic
    alternate locations, for components users might have installed by
    hand.
    """

    id: str
    display_name: str
    description: str
    folder_name: Optional[str] = None
    clone_url: Optional[str] = None
    install_cmds_static: list[list[str]] = field(default_factory=list)
    env_var: Optional[str] = None
    requires: list[str] = field(default_factory=list)
    always_installed: bool = False
    install_warning: Optional[str] = None

    def install_cmds(self) -> list[list[str]]:
        """Hook so musubi-tuner can swap CUDA extras in at install time."""
        return [list(cmd) for cmd in self.install_cmds_static]


def _musubi_install_cmds() -> list[list[str]]:
    return [["uv", "sync", *musubi_cuda_extras()]]


# Declared in render order for the panel. `uv` first — spec requires it
# before any other install can succeed.
COMPONENTS: list[Component] = [
    Component(
        id="uv",
        display_name="uv",
        description="Python package manager used by every install action.",
    ),
    Component(
        id="git",
        display_name="git",
        description="Required for cloning the companion repos. Install from https://git-scm.com/ if missing.",
    ),
    Component(
        id="klippbok_launcher",
        display_name="Klippbok (this app)",
        description="You are here. Auto-detected; no install needed.",
        always_installed=True,
    ),
    Component(
        id="klippbok_mcp",
        display_name="klippbok-mcp",
        description="MCP server exposing the Klippbok CLI as tools.",
        folder_name="klippbok-mcp",
        clone_url="https://github.com/hoodtronik/klippbok-mcp.git",
        install_cmds_static=[["uv", "sync"]],
    ),
    Component(
        id="musubi_tuner",
        display_name="Musubi Tuner",
        description="Trainer for Wan 2.2, FLUX.2, Z-Image, HunyuanVideo, FramePack, Qwen-Image.",
        folder_name="musubi-tuner",
        clone_url="https://github.com/kohya-ss/musubi-tuner.git",
        env_var="MUSUBI_TUNER_DIR",
        install_warning=(
            "Heavy install: torch + CUDA kernels. Expect 5-15 minutes and "
            "3-6 GB of disk + downloads."
        ),
    ),
    Component(
        id="musubi_mcp",
        display_name="musubi-mcp",
        description="MCP server for Musubi Tuner.",
        folder_name="musubi-mcp",
        clone_url="https://github.com/hoodtronik/musubi-mcp.git",
        install_cmds_static=[["uv", "sync"]],
        requires=["musubi_tuner"],
    ),
    Component(
        id="ltx2_trainer",
        display_name="LTX-2 Trainer",
        description="Lightricks' official trainer for LTX-2.3.",
        folder_name="LTX-2",
        clone_url="https://github.com/Lightricks/LTX-2.git",
        env_var="LTX_TRAINER_DIR",
        install_cmds_static=[["uv", "sync", "--frozen"]],
        install_warning=(
            "LTX-2 training targets Linux/WSL2. Installing the repo on "
            "Windows works; running the trainer on Windows native does not."
        ),
    ),
    Component(
        id="ltx_trainer_mcp",
        display_name="ltx-trainer-mcp",
        description="MCP server for the LTX-2 trainer.",
        folder_name="ltx-trainer-mcp",
        clone_url="https://github.com/hoodtronik/ltx-trainer-mcp.git",
        install_cmds_static=[["uv", "sync"]],
        requires=["ltx2_trainer"],
    ),
]


def component_by_id(cid: str) -> Optional[Component]:
    for c in COMPONENTS:
        if c.id == cid:
            return c
    return None


# --------------------------------------------------------------- detection


@dataclass
class ComponentStatus:
    id: str
    installed: bool
    path: Optional[str]
    # Why not installed / extra info:
    note: str = ""


def _launcher_root() -> Path:
    """Repo root of this Klippbok launcher (repo/app/_.py -> repo)."""
    return Path(__file__).resolve().parent.parent


def _launcher_python() -> Optional[str]:
    """`env/Scripts/python.exe` (Windows) or `env/bin/python` (POSIX)."""
    root = _launcher_root()
    candidate = root / "env" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    return str(candidate) if candidate.exists() else None


def _find_venv_python(install_dir: Path) -> Optional[str]:
    """Probe common uv / venv layouts inside an installed repo."""
    candidates = [
        install_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python"),
        install_dir / "venv" / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def detect_component(comp: Component, install_root: Path) -> ComponentStatus:
    """Fast filesystem-only detection. Subprocess probes live elsewhere."""
    if comp.id == "klippbok_launcher":
        return ComponentStatus(comp.id, True, str(_launcher_root()))
    if comp.id == "uv":
        return ComponentStatus(comp.id, has_uv(), shutil.which("uv"))
    if comp.id == "git":
        return ComponentStatus(comp.id, has_git(), shutil.which("git"))

    # Everything else: look under install_root/<folder_name> first, then
    # $ENV_VAR, then load_user_settings-stored override (user may have
    # installed via "Refresh Paths" after a hand-install elsewhere).
    if comp.folder_name:
        candidate = install_root / comp.folder_name
        if candidate.exists() and candidate.is_dir():
            return ComponentStatus(comp.id, True, str(candidate))

    if comp.env_var and os.environ.get(comp.env_var):
        env_path = Path(os.environ[comp.env_var])
        if env_path.exists():
            return ComponentStatus(
                comp.id, True, str(env_path), note=f"via ${comp.env_var}"
            )

    stored = load_user_settings().get("detected_paths", {}).get(comp.id)
    if stored and Path(stored).exists():
        return ComponentStatus(comp.id, True, stored, note="from .user_settings.json")

    return ComponentStatus(comp.id, False, None)


def detect_all(install_root: Path) -> dict[str, ComponentStatus]:
    """Run `detect_component` over every component. Persists results."""
    results: dict[str, ComponentStatus] = {}
    settings = load_user_settings()
    detected = settings.setdefault("detected_paths", {})
    for comp in COMPONENTS:
        status = detect_component(comp, install_root)
        results[comp.id] = status
        if status.installed and status.path:
            detected[comp.id] = status.path
        else:
            detected.pop(comp.id, None)
    # Also persist the launcher Python — used by generated MCP config blocks.
    lp = _launcher_python()
    if lp:
        detected["klippbok_python"] = lp
    save_user_settings(settings)
    return results


# --------------------------------------------------------------- install runners


def can_install(comp: Component, statuses: dict[str, ComponentStatus]) -> tuple[bool, str]:
    """Gate: returns (ok, reason).

    `ok=True` means the Install/Update button should be enabled.
    `reason` explains why it's disabled (used as the button label
    suffix when ok=False, since Gradio has no native tooltip).

    Semantics:
      * ``always_installed`` components (klippbok_launcher): always False.
        They're auto-detected; nothing to install.
      * ``git``: always False. Git has to be installed out-of-band
        (https://git-scm.com/). We can't pip-install it.
      * ``uv``: always True when missing (we install via the launcher
        venv's pip) or when present (Update == pip install --upgrade).
      * Everything else: requires uv + git on PATH, plus any
        ``requires=[...]`` deps already installed.
    """
    if comp.always_installed:
        return False, "Auto-detected; no install needed."
    if comp.id == "git":
        # Git is never installable from this panel — surface the
        # external-install path in the disabled button's label.
        if has_git():
            return False, "installed"
        return False, "install manually from git-scm.com"
    if comp.id == "uv":
        return True, ""
    if not has_uv():
        return False, "install uv first"
    if not has_git():
        return False, "install git first"
    for req in comp.requires:
        req_status = statuses.get(req)
        if not req_status or not req_status.installed:
            req_comp = component_by_id(req)
            req_name = req_comp.display_name if req_comp else req
            return False, f"install {req_name} first"
    return True, ""


def install_component(
    comp: Component, install_root: Path, *, update: bool = False
) -> Iterator[str]:
    """Run the git clone + post-install subprocesses, yielding log lines.

    Yields each line of subprocess stdout/stderr as it arrives, then an
    `[exit=N  elapsed=Ns]` footer for each step. At end of sequence emits
    a `[done]` or `[failed at step N]` line so callers can update the
    status row.
    """
    # Special case: `uv` itself bootstraps via the launcher's Python.
    if comp.id == "uv":
        yield from _install_uv()
        return
    if comp.id == "git":
        yield "git is installed outside this tab — download the installer "
        yield "from https://git-scm.com/download/win and re-run Refresh."
        yield "[failed at step 0]"
        return
    if comp.always_installed:
        yield f"{comp.display_name} is always installed — nothing to do."
        yield "[done]"
        return
    if not comp.clone_url or not comp.folder_name:
        yield f"[error] Component {comp.id} has no install definition."
        yield "[failed at step 0]"
        return

    target = install_root / comp.folder_name

    # Decide between install and update paths.
    if target.exists() and not update:
        yield f"[error] {target} already exists. Use Update to refresh it."
        yield "[failed at step 0]"
        return

    try:
        install_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        yield f"[error] Could not create install root {install_root}: {exc}"
        yield "[failed at step 0]"
        return

    # Build the command sequence.
    steps: list[tuple[str, list[str], Optional[str]]] = []  # (label, cmd, cwd)
    if update and target.exists():
        steps.append((f"git pull in {target}", ["git", "-C", str(target), "pull"], None))
    else:
        steps.append(
            (
                f"git clone into {target}",
                ["git", "clone", comp.clone_url, str(target)],
                None,
            )
        )

    post_cmds = _musubi_install_cmds() if comp.id == "musubi_tuner" else comp.install_cmds()
    for cmd in post_cmds:
        steps.append((" ".join(cmd), cmd, str(target)))

    # Run each step serially, yielding output. Bail on first non-zero exit.
    tab_id = f"install-{comp.id}"
    for step_idx, (label, cmd, cwd) in enumerate(steps, start=1):
        yield f"\n=== Step {step_idx}/{len(steps)} — {label} ==="
        yield "$ " + runner.format_command(cmd)
        exit_marker = None
        for line in runner.stream_command(tab_id, cmd, cwd=cwd):
            yield line
            if line.startswith("[exit="):
                exit_marker = line
        # runner.stream_command's final footer includes "[exit=N  elapsed=Ns]".
        # We need to know N to decide whether to continue.
        if exit_marker:
            m = re.search(r"\[exit=(-?\d+)", exit_marker)
            code = int(m.group(1)) if m else -1
            if code != 0:
                yield f"[failed at step {step_idx}]"
                return
        else:
            # If we never saw an exit marker something went wrong upstream.
            yield f"[failed at step {step_idx} — no exit line seen]"
            return

    yield "[done]"


def _install_uv() -> Iterator[str]:
    """Bootstrap uv using the launcher venv's pip.

    We deliberately avoid the cross-platform `astral.sh/uv/install.sh`
    shell script because a GUI app shouldn't be running arbitrary curl |
    sh flows without clear consent. Installing into the launcher venv is
    enough to make every subsequent `uv` subprocess find it on PATH so
    long as the launcher venv's Scripts dir is on PATH at process start.
    On Pinokio this is handled automatically; outside Pinokio, users may
    need to activate the venv or add its Scripts dir to PATH.
    """
    py = _launcher_python()
    if not py:
        yield (
            "[error] Could not find the launcher's Python interpreter. "
            "Install uv manually: see https://docs.astral.sh/uv/getting-started/installation/"
        )
        yield "[failed at step 0]"
        return
    cmd = [py, "-m", "pip", "install", "--upgrade", "uv"]
    yield f"=== Installing uv via {py} -m pip ==="
    yield "$ " + runner.format_command(cmd)
    exit_marker = None
    for line in runner.stream_command("install-uv", cmd):
        yield line
        if line.startswith("[exit="):
            exit_marker = line
    if exit_marker and "[exit=0" in exit_marker:
        yield "[done]"
    else:
        yield "[failed at step 1]"


# --------------------------------------------------------------- MCP config JSON


def _needs_api_env(comp_id: str) -> dict[str, str]:
    """Env vars that go into the MCP config for a given component, with placeholders."""
    if comp_id == "klippbok_mcp":
        return {"GEMINI_API_KEY": "<your-gemini-key>"}
    return {}


def render_mcp_config(statuses: dict[str, ComponentStatus]) -> str:
    """JSON block users paste into Claude Desktop / Antigravity / Cursor / etc.

    Only includes entries for MCP servers whose trainer/base is installed
    AND whose MCP server itself is installed. A user who installed only
    Klippbok+klippbok-mcp gets a one-server block, not five empty entries.
    """
    servers: dict[str, dict] = {}
    settings = load_user_settings()
    detected = settings.get("detected_paths", {})

    def _p(id_: str) -> Optional[str]:
        s = statuses.get(id_)
        return s.path if s and s.installed and s.path else None

    klippbok_python = detected.get("klippbok_python") or _launcher_python()

    if _p("klippbok_mcp"):
        servers["klippbok"] = {
            "command": "uv",
            "args": ["run", "--directory", _p("klippbok_mcp"), "klippbok-mcp"],
            "env": {
                "KLIPPBOK_PYTHON": klippbok_python or "<path to klippbok python>",
                "GEMINI_API_KEY": "<your-gemini-key>",
            },
        }

    if _p("musubi_mcp") and _p("musubi_tuner"):
        musubi_python = _find_venv_python(Path(_p("musubi_tuner")))
        servers["musubi"] = {
            "command": "uv",
            "args": ["run", "--directory", _p("musubi_mcp"), "musubi-mcp"],
            "env": {
                "MUSUBI_TUNER_DIR": _p("musubi_tuner"),
                "MUSUBI_PYTHON": musubi_python or "<path to musubi python>",
            },
        }

    if _p("ltx_trainer_mcp") and _p("ltx2_trainer"):
        servers["ltx-trainer"] = {
            "command": "uv",
            "args": ["run", "--directory", _p("ltx_trainer_mcp"), "ltx-trainer-mcp"],
            "env": {"LTX_TRAINER_DIR": _p("ltx2_trainer")},
        }

    if not servers:
        return (
            "No MCP servers installed yet. Install at least klippbok-mcp "
            "from the Pipeline Status panel, then re-create the workspace."
        )
    return json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False)
