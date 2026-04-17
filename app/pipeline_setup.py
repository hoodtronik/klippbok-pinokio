"""Agentic pipeline workspace scaffolder + embedded docs + watchdog source.

Backing module for the "Agentic Pipeline" tab in the Klippbok UI. Keeps
the long string constants (pipeline guide, agent instructions template,
watchdog script) out of app.py so the UI file stays focused on Gradio
wiring.

Design invariants:
  * stdlib only — no new pip deps. pathlib, os, platform, shutil, subprocess.
  * Auto-detection of companion tools fails gracefully. Missing tools get
    a `<NOT FOUND — set this path>` placeholder so the user knows to fill
    it in manually rather than crash on workspace creation.
  * All generated files are UTF-8. LF line endings for portability
    (Git will normalize on checkout per repo .gitattributes).
  * The workspace creator checks for pre-existing directories and returns
    a warning rather than overwriting. Re-running on an existing project
    tops up missing files but never clobbers user edits.

Credited to hoodtronik — the pipeline design is theirs; this module
implements the scaffold.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pipeline_installer as pi  # fresh import; no circular risk


# --------------------------------------------------------------- targets


# Target models grouped by trainer + modality so the UI can render them
# in meaningful clusters (video-via-Musubi / image-via-Musubi / other).
#
# VIDEO_TARGETS_MUSUBI and IMAGE_TARGETS_MUSUBI are the Musubi-Tuner
# supported lineup. VIDEO_TARGETS_OTHER is for trainers that are NOT
# Musubi (currently just LTX-2.3 on the Lightricks LTX-2 trainer).
# TARGET_MODELS is the flat concatenation used by callers that don't
# care about the grouping.
VIDEO_TARGETS_MUSUBI: tuple[str, ...] = (
    "Wan 2.2",
    "HunyuanVideo",
    "HunyuanVideo 1.5",
    "FramePack",
)

IMAGE_TARGETS_MUSUBI: tuple[str, ...] = (
    "FLUX.2 Klein 9B",
    "FLUX.2 Dev",
    "FLUX.1 Kontext",
    "Z-Image",
    "Qwen-Image",
    "Kandinsky 5",
)

VIDEO_TARGETS_OTHER: tuple[str, ...] = (
    "LTX-2.3",
)

TARGET_MODELS: tuple[str, ...] = (
    *VIDEO_TARGETS_MUSUBI,
    *IMAGE_TARGETS_MUSUBI,
    *VIDEO_TARGETS_OTHER,
)

STRATEGIES: tuple[str, ...] = (
    "Layered (Master + Boutique LoRAs)",
    "Single LoRA",
    "Custom",
)

DEFAULT_STRATEGY = STRATEGIES[0]


# --------------------------------------------------------------- auto-detect


@dataclass
class DetectedPaths:
    """What we could find for companion tools on the current machine."""

    klippbok_pinokio: Optional[str] = None
    klippbok_python: Optional[str] = None
    musubi_tuner: Optional[str] = None
    musubi_python: Optional[str] = None
    ltx_trainer: Optional[str] = None
    klippbok_mcp: Optional[str] = None
    musubi_mcp: Optional[str] = None
    ltx_trainer_mcp: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[int] = None
    platform_label: str = ""

    def render(self, key: str) -> str:
        """Return the value or a clearly-marked placeholder for templates."""
        value = getattr(self, key, None)
        return value if value else "<NOT FOUND — set this path manually>"


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def _pinokio_roots() -> list[Path]:
    """Probable Pinokio install locations, in rough priority order."""
    candidates = [
        Path(os.environ.get("PINOKIO_HOME", "")) if os.environ.get("PINOKIO_HOME") else None,
        Path.home() / "pinokio",
        Path("C:/pinokio"),
        Path("C:/Users") / os.environ.get("USERNAME", "") / "pinokio" if os.name == "nt" else None,
    ]
    return [c for c in candidates if c is not None and c.exists()]


def _detect_gpu() -> tuple[Optional[str], Optional[int]]:
    """Best-effort NVIDIA GPU name + VRAM via nvidia-smi. Returns (None, None) if not found."""
    # CLAUDE-NOTE: shelling out to nvidia-smi is the only zero-dep way to
    # get GPU info on Windows. torch.cuda would be more portable but would
    # require importing torch from the user's venv — we avoid that dep
    # chain per the "stdlib only" rule for this scaffolding module.
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None, None
        first = r.stdout.strip().splitlines()[0]
        # "NVIDIA GeForce RTX 4090, 24564"
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 2:
            return None, None
        name = parts[0]
        try:
            # VRAM comes back in MiB; convert to GB (round up so 24564 -> 24).
            vram_mib = int(parts[1])
            vram_gb = max(1, vram_mib // 1024)
        except ValueError:
            vram_gb = None
        return name, vram_gb
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, None


def detect_paths() -> DetectedPaths:
    """Best-effort detection of all companion tools. Never raises."""
    out = DetectedPaths(platform_label=f"{platform.system()} {platform.release()}")

    # 1. Klippbok Pinokio launcher itself. We're running inside app/, so
    #    parent of our package path is the repo root.
    klippbok_app_dir = Path(__file__).resolve().parent
    klippbok_root = klippbok_app_dir.parent
    out.klippbok_pinokio = str(klippbok_root)
    venv_python = klippbok_root / "env" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    if venv_python.exists():
        out.klippbok_python = str(venv_python)

    # 2. Musubi Tuner — check Pinokio, F:/__PROJECTS/, and MUSUBI_TUNER_DIR.
    musubi_candidates: list[Path] = []
    if os.environ.get("MUSUBI_TUNER_DIR"):
        musubi_candidates.append(Path(os.environ["MUSUBI_TUNER_DIR"]))
    for proot in _pinokio_roots():
        musubi_candidates.append(proot / "api" / "musubi-tuner.pinokio.git" / "app")
    musubi_candidates += [
        Path("F:/__PROJECTS/musubi-tuner.pinokio/app"),
        Path("F:/__PROJECTS/musubi-tuner/app"),
        Path.home() / "musubi-tuner",
    ]
    if (m := _first_existing(musubi_candidates)):
        out.musubi_tuner = str(m)
        vp = m / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )
        if vp.exists():
            out.musubi_python = str(vp)

    # 3. LTX Trainer — several possible layouts.
    ltx_candidates: list[Path] = []
    if os.environ.get("LTX_TRAINER_DIR"):
        ltx_candidates.append(Path(os.environ["LTX_TRAINER_DIR"]))
    ltx_candidates += [
        Path("F:/__PROJECTS/LTXTrainer/LTX-2"),
        Path("F:/__PROJECTS/LTX-2"),
        Path.home() / "LTX-2",
    ]
    if (ltx := _first_existing(ltx_candidates)):
        out.ltx_trainer = str(ltx)

    # 4. MCP server repos — look in parent dir of this repo + F:/__PROJECTS/.
    project_parent = klippbok_root.parent
    for attr, folder_names in (
        ("klippbok_mcp", ["klippbok-mcp"]),
        ("musubi_mcp", ["musubi-mcp"]),
        ("ltx_trainer_mcp", ["ltx-trainer-mcp"]),
    ):
        for name in folder_names:
            for base in (project_parent, Path("F:/__PROJECTS")):
                candidate = base / name
                if candidate.exists():
                    setattr(out, attr, str(candidate))
                    break
            if getattr(out, attr):
                break

    # 5. GPU info — best effort; None on non-NVIDIA or if nvidia-smi missing.
    out.gpu_name, out.gpu_vram_gb = _detect_gpu()
    return out


# --------------------------------------------------------------- scaffolder


@dataclass
class WorkspaceConfig:
    project_name: str
    root_path: str
    styles: list[str] = field(default_factory=list)
    strategy: str = DEFAULT_STRATEGY
    targets: list[str] = field(default_factory=list)

    def workspace_dir(self) -> Path:
        return Path(self.root_path) / self.project_name / "workspace"


ILLEGAL_FS_CHARS = set('/\\:*?"<>|')


def _validate(cfg: WorkspaceConfig) -> Optional[str]:
    if not cfg.project_name.strip():
        return "Project name is required."
    if not cfg.root_path.strip():
        return "Root path is required."
    if any(c in cfg.project_name for c in ILLEGAL_FS_CHARS):
        return "Project name contains filesystem-illegal characters."
    root = Path(cfg.root_path)
    if not root.exists():
        return f"Root path does not exist: {cfg.root_path}"
    if not root.is_dir():
        return f"Root path is not a directory: {cfg.root_path}"
    for style in cfg.styles:
        if any(c in style for c in ILLEGAL_FS_CHARS):
            return f"Style folder name contains illegal characters: {style!r}"
    return None


def _output_subdirs(strategy: str) -> list[str]:
    """Per-strategy subfolders under outputs/."""
    if strategy.startswith("Layered"):
        return ["master_loras", "boutique_loras"]
    if strategy.startswith("Single"):
        return ["loras"]
    # Custom — caller organizes outputs/ themselves.
    return []


def _discover_knowledge_source() -> Optional[Path]:
    """Find the LoRA training knowledge base files, or None if unavailable.

    Priority:
      1. ``$KNOWLEDGE_BASE_DIR`` env var (user's override).
      2. ``F:/__PROJECTS/LoRAKnowledgeBase/reviewed/`` (this machine's pattern).
      3. ``~/LoRAKnowledgeBase/reviewed/`` (portable fallback).
    """
    candidates: list[Path] = []
    if os.environ.get("KNOWLEDGE_BASE_DIR"):
        candidates.append(Path(os.environ["KNOWLEDGE_BASE_DIR"]))
    candidates += [
        Path("F:/__PROJECTS/LoRAKnowledgeBase/reviewed"),
        Path.home() / "LoRAKnowledgeBase" / "reviewed",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _copy_knowledge_files(source: Path, dest: Path) -> tuple[int, list[str]]:
    """Copy every .md file from ``source`` into ``dest``.

    Overwrites existing files in ``dest`` — the knowledge base is the
    source of truth, and agents consult these via markdown read, not
    via user edits. Returns ``(count, errors)``.

    NOTE: files flagged "INSUFFICIENT DATA" in their content are copied
    verbatim. The orchestration agent is expected to see the flag and
    ask the user to fill in the gap rather than guessing. Don't filter.
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    errors: list[str] = []
    for md in source.glob("*.md"):
        try:
            shutil.copy2(md, dest / md.name)
            count += 1
        except OSError as exc:
            errors.append(f"{md.name}: {exc}")
    return count, errors


def create_workspace(cfg: WorkspaceConfig) -> tuple[bool, str]:
    """Create the workspace tree and populate the three template files.

    Returns ``(success, message)``. On success, ``message`` is a
    multi-line summary suitable for display in the UI. Idempotent:
    re-running on an existing project tops up missing files and
    subfolders without clobbering anything already there.
    """
    err = _validate(cfg)
    if err:
        return False, f"[error] {err}"

    workspace = cfg.workspace_dir()
    is_new = not workspace.exists()

    try:
        # Core tree. mkdir(exist_ok=True) is why this is idempotent.
        (workspace / "datasets").mkdir(parents=True, exist_ok=True)
        (workspace / "configs").mkdir(parents=True, exist_ok=True)
        (workspace / "manifests").mkdir(parents=True, exist_ok=True)
        (workspace / "outputs").mkdir(parents=True, exist_ok=True)
        (workspace / "knowledge").mkdir(parents=True, exist_ok=True)

        # Per-strategy outputs.
        for sub in _output_subdirs(cfg.strategy):
            (workspace / "outputs" / sub).mkdir(exist_ok=True)

        # Style subfolders under datasets/.
        for style in cfg.styles:
            style = style.strip()
            if style:
                (workspace / "datasets" / style).mkdir(exist_ok=True)
    except OSError as exc:
        return False, f"[error] Could not create workspace directories: {exc}"

    # Populate knowledge/ from the shared reviewed/ folder. Non-fatal
    # if unavailable — the workspace still works, agent guidance is just
    # less specific.
    knowledge_source = _discover_knowledge_source()
    knowledge_count = 0
    knowledge_errors: list[str] = []
    if knowledge_source is not None:
        knowledge_count, knowledge_errors = _copy_knowledge_files(
            knowledge_source, workspace / "knowledge"
        )
    else:
        # CLAUDE-NOTE: Write a placeholder README so the agent finds something
        # in knowledge/ and knows what to do rather than hitting a missing-dir
        # error. The README tells the user how to populate the folder.
        _knowledge_placeholder = workspace / "knowledge" / "README.md"
        if not _knowledge_placeholder.exists():
            _knowledge_placeholder.write_text(
                "# Knowledge Base\n\n"
                "This folder should contain training knowledge files for the orchestration agent.\n\n"
                "To populate it, place `.md` files here covering your target architectures\n"
                "(e.g. `wan22_training.md`, `flux2_klein_training.md`, `ltx23_training.md`,\n"
                "`common_failures.md`, `hardware_profiles.md`). Each file should start with\n"
                "a confidence rating line: `CONFIDENCE: HIGH / MEDIUM / LOW / INSUFFICIENT DATA`.\n\n"
                "See the Pipeline Guide for how to build a knowledge base using the research\n"
                "agent workflow. You can also set the `KNOWLEDGE_BASE_DIR` environment variable\n"
                "to point at an existing reviewed/ folder and re-create this workspace.\n",
                encoding="utf-8",
            )

    # Always write these; they're templates that should refresh on each run
    # so path auto-detection picks up newly-installed companions.
    detected = detect_paths()
    # CLAUDE-NOTE: Install Root == Workspace Root per user's design choice,
    # so we detect under cfg.root_path (not workspace.parent). This feeds
    # the MCP Server Configuration JSON block in AGENT_INSTRUCTIONS.md.
    try:
        statuses = pi.detect_all(Path(cfg.root_path))
    except Exception:
        statuses = {}
    try:
        (workspace / "PIPELINE_GUIDE.md").write_text(PIPELINE_GUIDE_MD, encoding="utf-8")
        (workspace / "AGENT_INSTRUCTIONS.md").write_text(
            render_agent_instructions(cfg, detected, statuses=statuses), encoding="utf-8"
        )
        (workspace / "watchdog.py").write_text(WATCHDOG_PY, encoding="utf-8")
    except OSError as exc:
        return False, f"[error] Could not write template files: {exc}"

    status_header = "Created new workspace" if is_new else "Refreshed existing workspace (files updated, nothing deleted)"
    lines = [
        status_header,
        f"  {workspace}",
        "",
        "Structure:",
        "  datasets/    <- put your source clips/images here",
    ]
    if cfg.styles:
        lines.append("    (style subfolders: " + ", ".join(s for s in cfg.styles if s.strip()) + ")")
    lines += [
        "  configs/     <- training configs (created by the agent)",
        "  manifests/   <- Klippbok triage manifests",
        "  outputs/     <- training outputs",
    ]
    for sub in _output_subdirs(cfg.strategy):
        lines.append(f"    {sub}/")
    lines.append("  knowledge/   <- LoRA training knowledge base (agent reference)")
    if knowledge_source is not None:
        lines.append(f"    <- {knowledge_count} file(s) copied from {knowledge_source}")
        if knowledge_errors:
            lines.append(f"    <- {len(knowledge_errors)} error(s): " + "; ".join(knowledge_errors[:3]))
    else:
        lines.append("    <- source folder not found (set KNOWLEDGE_BASE_DIR to populate)")
    lines += [
        "",
        "Generated files:",
        "  AGENT_INSTRUCTIONS.md  <- paste into your agent as system/workspace instructions",
        "  PIPELINE_GUIDE.md      <- full setup walkthrough",
        "  watchdog.py            <- training watchdog (no external deps)",
        "",
        "Next: add your source material to datasets/ and follow the Pipeline Guide.",
    ]
    return True, "\n".join(lines)


# ------------------------------------------------------ agent instructions

# CLAUDE-NOTE: Each block is a regular string (not an f-string). The render
# function calls .format(root_path=cfg.root_path) on each block so {root_path}
# is substituted with the actual workspace root before insertion into the
# f-string template. Keep all other literal braces escaped as {{ }}.
_PHASE0_DOWNLOAD_BLOCKS: dict[str, str] = {
    "Wan 2.2": """\
### Wan 2.2

**Source repos:**
- `Wan-AI/Wan2.2-T2V-A14B` (high/low-noise DiTs)
- `Wan-AI/Wan2.1-T2V-14B` (VAE + T5 encoder)

Required files:
- `wan2.2_t2v_high_noise_14B_fp16.safetensors` (~28 GB) — high-noise DiT
- `wan2.2_t2v_low_noise_14B_fp16.safetensors` (~28 GB) — low-noise DiT
- `wan_2.1_vae.safetensors` — VAE (shared with Wan 2.1)
- `umt5-xxl-enc-bf16.pth` — T5 text encoder

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir {root_path}/models/wan22/
huggingface-cli download Wan-AI/Wan2.1-T2V-14B wan_2.1_vae.safetensors umt5-xxl-enc-bf16.pth --local-dir {root_path}/models/wan22/
```

⚠️ Do NOT use quantized files from Wan2GP (filenames ending in `_mbf16` or `_gguf`).
Training requires the full fp16 checkpoints.
""",

    "HunyuanVideo": """\
### HunyuanVideo

Read the `musubi://docs/hunyuan_video` MCP resource for current repo names and
file lists — these change with model updates.

Required components (approximate sizes):
- HunyuanVideo DiT checkpoint (~14 GB)
- VAE
- LLaMA-based LLM text encoder (~7 GB)
- CLIP text encoder

⚠️ Budget ~30–40 GB disk space total.
""",

    "HunyuanVideo 1.5": """\
### HunyuanVideo 1.5

Read the `musubi://docs/hunyuan_video` MCP resource for current repo names and
file lists. HunyuanVideo 1.5 uses updated weights with the same training pipeline.
""",

    "FramePack": """\
### FramePack

FramePack reuses Wan 2.1/2.2 base model files with additional FramePack-specific
weights. Download Wan 2.2 first (see above), then read the `musubi://docs/framepack`
MCP resource for any FramePack-specific additions.
""",

    "FLUX.2 Klein 9B": """\
### FLUX.2 Klein 9B

Read the `musubi://docs/flux_2` MCP resource for the exact HuggingFace repo paths —
they may change between releases.

Required files (approximate):
- Klein **base** DiT checkpoint (`klein-base-9b`, ~18 GB)
- VAE (`ae.safetensors`, ~335 MB)
- Qwen3 8B text encoder (~16 GB, or fp8 quantized ~8 GB)

```bash
huggingface-cli download <klein-base-9b-repo> --local-dir {root_path}/models/flux2_klein/
```

⚠️ CRITICAL: Training MUST use `klein-base-9b`. The distilled `klein-9b` (inference
model) will NOT train correctly — it lacks the residual structure needed for LoRA.
Set `--model_version klein-base-9b` in every training command.
""",

    "FLUX.2 Dev": """\
### FLUX.2 Dev

Read the `musubi://docs/flux_2` MCP resource for the exact HuggingFace repo paths.

Required files (approximate):
- Dev DiT checkpoint (~18 GB)
- VAE (`ae.safetensors`) — same file as Klein
- Mistral 3 Small text encoder (~7 GB)

⚠️ Official Musubi docs recommend training on `klein-base-9b` instead. Use FLUX.2 Dev
only if you specifically need the full-size model's parameter count.
⚠️ `--fp8_text_encoder` is NOT supported for Dev (Mistral 3 doesn't support FP8).
""",

    "FLUX.1 Kontext": """\
### FLUX.1 Kontext

Read the `musubi://docs/flux_2` MCP resource or Musubi Tuner documentation for
current download instructions. Limited community training data — check the
`INSUFFICIENT DATA` flag in the knowledge base before proceeding.
""",

    "Z-Image": """\
### Z-Image Base

Read the `musubi://docs/zimage` MCP resource for the exact HuggingFace repo paths.

Required files (approximate):
- Z-Image **Base** DiT checkpoint (NOT Turbo)
- Z-Image VAE
- Z-Image text encoder

```bash
huggingface-cli download <zimage-base-repo> --local-dir {root_path}/models/zimage/
```

⚠️ Use the Base model for training, not Turbo. Base produces more stable LoRA gradients;
Turbo is distilled for inference speed and is not suitable for training.
""",

    "Qwen-Image": """\
### Qwen-Image

Read Musubi Tuner documentation for current download instructions.
Limited community training data — check the `INSUFFICIENT DATA` flag before proceeding.
""",

    "Kandinsky 5": """\
### Kandinsky 5

Read Musubi Tuner documentation for current download instructions.
Limited community training data — check the `INSUFFICIENT DATA` flag before proceeding.
""",

    "LTX-2.3": """\
### LTX-2.3

**Source repo:** `Lightricks/LTX-Video` on HuggingFace

Required files:
- `ltx-2.3-22b-dev.safetensors` (~44 GB) — full BF16 base checkpoint (training)
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (~2 GB) — inference upscaler
- `ltx-2.3-22b-distilled-lora-384.safetensors` (~500 MB) — distilled LoRA for fast inference
- Gemma 3 12B text encoder — full directory `google/gemma-3-12b-it-qat-q4_0-unquantized` (~24 GB)

```bash
huggingface-cli download Lightricks/LTX-Video ltx-2.3-22b-dev.safetensors --local-dir {root_path}/models/ltx23/
huggingface-cli download Lightricks/LTX-Video ltx-2.3-spatial-upscaler-x2-1.0.safetensors --local-dir {root_path}/models/ltx23/
huggingface-cli download Lightricks/LTX-Video ltx-2.3-22b-distilled-lora-384.safetensors --local-dir {root_path}/models/ltx23/
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir {root_path}/models/ltx23/gemma/
```

After downloading, update paths via the LTX-Trainer Pinokio UI:
- **Project tab → Scan folder** — paste `{root_path}/models/ltx23/` and click Scan to auto-detect
- Or run `ltx_check_installation` via the MCP tool to verify paths

⚠️ LTX-2.3 uses its own trainer (ltx-trainer), NOT Musubi Tuner.
⚠️ Training requires Linux or WSL2 (triton/bitsandbytes are Linux-only).
⚠️ Do NOT use FP8-quantized or community-reposted variants with the same filename —
   they are inference-only and will fail silently during training. The scanner in the
   LTX-Trainer UI detects quantized models automatically via the safetensors header.
""",
}


_TARGET_TRAINING_BLOCKS: dict[str, str] = {
    "Wan 2.2": """\
### Wan 2.2 (via Musubi Tuner — video)

Video LoRA. Uses `musubi-tuner` with the Wan 2.2 cache + training scripts.
Call via the `musubi-mcp` tool set.

Key flags the agent should consider:
  - Target rank: Master 128, Boutique 32-64 (Layered strategy)
  - fp8: yes on <24GB VRAM
  - blocks_to_swap: 20-30 on 12-16GB VRAM
  - learning_rate: 1e-4 for Master, 5e-5 for Boutique
  - max_train_steps: 2000-3000 Master, 500-800 Boutique
""",
    "HunyuanVideo": """\
### HunyuanVideo (via Musubi Tuner — video)

Tencent HunyuanVideo LoRA training. Uses musubi-tuner's hunyuan scripts.
See musubi-tuner docs for the correct cache prep — HunyuanVideo dataset
format differs from Wan 2.2.
""",
    "HunyuanVideo 1.5": """\
### HunyuanVideo 1.5 (via Musubi Tuner — video)

Updated Hunyuan. Cache + training harness similar to HunyuanVideo but
check musubi-tuner's current hunyuan_1_5 branch for the latest scripts —
flag surface and resolution caps shift between minor releases.
""",
    "FramePack": """\
### FramePack (via Musubi Tuner — video, frame-sequence)

FramePack is frame-sequence-based rather than video-clip-based. Dataset
layout differs from Wan/Hunyuan — consult musubi-tuner's FramePack docs
before running. Klippbok's Ingest output usually needs reshaping before
feeding it to FramePack.
""",
    "FLUX.2 Klein 9B": """\
### FLUX.2 Klein 9B (via Musubi Tuner — image)

Image LoRA. Requires the FLUX.2 cache + training scripts. Image-only
dataset (skip Ingest; go Scan -> Triage -> Caption -> Validate -> Organize).

Key flags:
  - Resolution: 1024 or 1440 depending on source quality
  - Target rank: 64 typical
  - fp8 + gradient_checkpointing: on <24GB
  - learning_rate: 1e-4 to 5e-5
""",
    "FLUX.2 Dev": """\
### FLUX.2 Dev (via Musubi Tuner — image)

Image LoRA on the developer-tier FLUX.2 checkpoint. Larger base than
Klein 9B. Training knobs resemble FLUX.2 Klein 9B but expect higher VRAM
floors. Check musubi-tuner docs for the current flux_2_dev config.
""",
    "FLUX.1 Kontext": """\
### FLUX.1 Kontext (via Musubi Tuner — image)

FLUX.1 Kontext context-editing LoRA. Image input pairs (source + target
images). Classical FLUX.1 LoRA flags apply (rank 32-64, fp8 below 24GB);
dataset layout requires paired captions so Klippbok's Caption tab needs
configuration attention.
""",
    "Z-Image": """\
### Z-Image (via Musubi Tuner — image)

Tongyi-MAI/Z-Image fine-tuning pipeline. See musubi-tuner docs for the
Z-Image specific cache prep.
""",
    "Qwen-Image": """\
### Qwen-Image (via Musubi Tuner — image)

Alibaba Qwen-Image LoRA. Image dataset. Consult musubi-tuner's qwen_image
config for resolution, rank, and fp8 recommendations current to your
installed version.
""",
    "Kandinsky 5": """\
### Kandinsky 5 (via Musubi Tuner — image)

Kandinsky 5 LoRA. Image dataset. Musubi-tuner's kandinsky_5 scripts
handle the cache + training. Newer addition to musubi's lineup — verify
flag surface against the repo's current docs.
""",
    "LTX-2.3": """\
### LTX-2.3 (via LTX-2 trainer — video, **NOT Musubi**)

Lightricks' official trainer, driven through `ltx-trainer-mcp`. Separate
from Musubi entirely — different dataset format, different cache pipeline.
Do NOT use `musubi-mcp` for LTX-2.3.

Key flags:
  - INT8 low-VRAM config for <32GB cards
  - Standard config wants 80GB+
  - Video clips only; image datasets not supported
  - Run after Klippbok has produced a normalized clip set
  - Linux / WSL2 only for training itself (repo installs fine on Windows)
""",
}


def render_agent_instructions(
    cfg: WorkspaceConfig,
    detected: DetectedPaths,
    statuses: Optional[dict] = None,
) -> str:
    """Fill in the AGENT_INSTRUCTIONS.md template from the config + detected paths.

    ``statuses`` is the dict returned by ``pipeline_installer.detect_all``.
    When provided, the generated file includes a ready-to-paste MCP Server
    Configuration JSON block — only servers whose trainer + MCP server are
    both installed appear. When omitted (back-compat path), that section
    is skipped and the user has to assemble the JSON by hand.
    """
    workspace = cfg.workspace_dir()
    targets = cfg.targets or ["(none selected — add target blocks manually)"]

    # Phase 0 — one download block per selected target, with {root_path} substituted.
    phase0_sections: list[str] = []
    for t in cfg.targets:
        block = _PHASE0_DOWNLOAD_BLOCKS.get(t)
        if block:
            phase0_sections.append(block.format(root_path=cfg.root_path))
    if not phase0_sections:
        phase0_sections.append(
            "### No target models selected\n\n"
            "Re-scaffold the workspace with target models ticked to get\n"
            "architecture-specific download instructions here."
        )

    target_sections: list[str] = []
    for t in cfg.targets:
        block = _TARGET_TRAINING_BLOCKS.get(t)
        if block:
            target_sections.append(block)
    if not target_sections:
        target_sections.append(
            "### No target models selected\n\n"
            "Re-scaffold the workspace with target models ticked, or add "
            "trainer-specific sections here manually. The agent will ask "
            "clarifying questions if this stays blank."
        )

    gpu_line = (
        f"{detected.gpu_name} ({detected.gpu_vram_gb} GB VRAM)"
        if detected.gpu_name else "GPU: nvidia-smi not found — fill in manually"
    )

    styles_line = (
        "  - " + "\n  - ".join(s for s in cfg.styles if s.strip())
        if any(s.strip() for s in cfg.styles)
        else "  (no style subfolders configured)"
    )

    if statuses:
        mcp_json = pi.render_mcp_config(statuses)
        mcp_config_section = f"""
## MCP Server Configuration

Copy-paste this JSON into your agent's MCP configuration (Antigravity
settings, Claude Desktop `claude_desktop_config.json`, Cursor's
`.cursor/mcp.json`, or the Claude Code CLI via `claude mcp add`):

```json
{mcp_json}
```

Only servers whose trainer + MCP server are both installed appear above.
If a server is missing, install it from the Pipeline Status panel in the
Agentic Pipeline tab, then re-create this workspace — the JSON will
expand automatically.
"""
    else:
        mcp_config_section = ""

    return f"""\
# Agent Instructions — {cfg.project_name}

*Pipeline designed by [hoodtronik](https://github.com/hoodtronik). This workspace
was scaffolded by the Klippbok Pinokio launcher's Agentic Pipeline tab.*
{mcp_config_section}

## Workspace

- **Project**: {cfg.project_name}
- **Workspace root**: `{workspace}`
- **Strategy**: {cfg.strategy}
- **Target models**: {", ".join(targets)}

### Style subfolders under datasets/
{styles_line}

### Knowledge base
A copy of the LoRA training knowledge base lives at `{workspace}/knowledge/`.
Before picking training hyperparameters, read the file matching your target
architecture (e.g. `wan22_training.md`, `flux2_klein_training.md`,
`ltx23_training.md`, …). Each file has a confidence rating at the top
(HIGH / MEDIUM / LOW).

**If any file you need shows an `INSUFFICIENT DATA` flag — ask the user
before guessing.** The flag is there specifically because community data
couldn't nail the values down; the user knows their machine and their
training intent better than you do at that point. Also consult
`common_failures.md` when loss misbehaves and `hardware_profiles.md`
before picking rank / blocks_to_swap.

The same knowledge is also exposed via `knowledge://` MCP resources on
`musubi-mcp` and `ltx-trainer-mcp` — use whichever surface is more
convenient in your agent.

## Environment (auto-detected)

- Platform: {detected.platform_label}
- {gpu_line}
- Klippbok launcher: `{detected.render("klippbok_pinokio")}`
- Klippbok Python: `{detected.render("klippbok_python")}`
- Musubi Tuner: `{detected.render("musubi_tuner")}`
- Musubi Python: `{detected.render("musubi_python")}`
- LTX-2 Trainer: `{detected.render("ltx_trainer")}`
- klippbok-mcp repo: `{detected.render("klippbok_mcp")}`
- musubi-mcp repo: `{detected.render("musubi_mcp")}`
- ltx-trainer-mcp repo: `{detected.render("ltx_trainer_mcp")}`

Anything marked `<NOT FOUND>` above needs to be set manually — either in this
file or via environment variables before the agent starts.

## Recommended execution order

1. **Phase 0: Download models** — get the correct full-precision base models for your target architectures
2. **Phase 1: Dataset curation** — scan, caption, validate your source material
3. **Phase 2: Training** — cache latents, cache text encoder, train
4. **Phase 3: Watchdog** — activate before walking away from any training run
5. **Phase 4: Evaluation** — check results, pick best checkpoint, iterate

For a first-time test, start with your smallest dataset on the fastest architecture
(image models like FLUX.2 Klein train faster than video models like Wan 2.2).

## Pipeline execution plan

Follow these phases in order. Each is driven through MCP tools.

### Phase 0 — Download base models

**Do this first.** Training requires full-precision (bf16/fp16) model checkpoints.
Quantized inference models from Wan2GP, ComfyUI, or similar tools will NOT work
for training even when they share the exact same filename as the correct checkpoint.

Verify the environment first:
- For Musubi-based models: `musubi_check_installation`
- For LTX-2.3: `ltx_check_installation`

**Suggested model directory layout:**

```
{cfg.root_path}/models/
├── wan22/         ← Wan 2.2 DiTs + VAE + T5 encoder
├── flux2_klein/   ← Klein-base-9b + VAE + Qwen3 encoder
├── zimage/        ← Z-Image Base + VAE + text encoder
└── ltx23/         ← LTX-2.3 checkpoint + Gemma encoder
```

{chr(10).join(phase0_sections)}

After downloading, tell the agent the model paths:

> "Models downloaded to `{cfg.root_path}/models/`. Here are the locations:
> - [architecture]: [path]
> Update model paths in training configurations accordingly."

The agent will use these paths when creating latent-cache configs and training commands.

### Phase 1 — Dataset curation (Klippbok)

Goal: turn raw source material in `datasets/` into a captioned, validated,
organized dataset ready for a trainer.

1. `klippbok_check_installation` — sanity-check the environment before doing
   anything else.
2. For each style subfolder in `datasets/`:
   - `klippbok_scan` → inventory the clips / images.
   - Populate `datasets/<style>/concepts/` by hand with 5-20 reference images
     per concept if triage is needed (see PIPELINE_GUIDE.md).
   - `klippbok_triage` → produces a manifest in `manifests/`.
   - `klippbok_read_manifest` + `klippbok_update_manifest` to gate by score
     and surgical-fix include flags.
   - `klippbok_ingest` (video only) using the reviewed manifest.
   - `klippbok_caption` with `--provider gemini` (or replicate) using a
     short `--anchor-word` you'll reuse at prompt time.
   - `klippbok_score` + optional `klippbok_audit` for quality checks.
3. `klippbok_validate` with `quality=True, duplicates=True`.
4. `klippbok_organize` → lays out the dataset for the trainer target.

### Phase 2 — Training setup

For each selected target model:

{chr(10).join(target_sections)}

### Phase 3 — Activate the watchdog

Before kicking off any training run, start `watchdog.py` in parallel:

```bash
python {workspace}/watchdog.py \\
    --log-file <training log path> \\
    --pid <training process pid> \\
    --max-loss 2.0 \\
    --plateau-steps 500
```

The watchdog will terminate the training process and write
`watchdog_report.json` if loss explodes, hits NaN/Inf, or plateaus. Safe to
walk away from a long run once the watchdog is active.

**Run the watchdog in the same environment as the trainer.** If training
runs inside WSL2, launch the watchdog from inside WSL2 too — Windows and
WSL2 have separate PID namespaces, so a Windows-side watchdog can't see
or kill a WSL2 training process. The watchdog auto-picks the right kill
primitive on each side (taskkill /F /T on Windows for process-tree kill,
SIGTERM then SIGKILL on POSIX).

### Phase 4 — Evaluation

After each LoRA finishes:

1. Log `steps_trained`, `final_loss`, and `watchdog_report.json` status to
   `configs/<run_name>.json`.
2. Render a small comparison grid against the base model using the trainer's
   inference harness.
3. If the result is usable, copy to `outputs/<tier>/` with a descriptive name.
4. If not, iterate on config + retry.

## Hard rules

- **Run ONE training job at a time.** Don't queue two trainers on the same GPU.
- **Don't run local LLMs during training.** They compete for VRAM and will OOM
  the trainer mid-step.
- **Always use cloud captioning** (Gemini or Replicate) — local Ollama works
  but is slower and contends for GPU with the trainer.
- **Always activate the watchdog** before kicking off training. Non-negotiable.
- **Commit configs to `configs/`** so failed runs are reproducible.

## If you hit a snag

- Agent stuck: dump the last 50 lines of the trainer's log, summarize, and
  ask the user before deciding.
- OOM: reduce `rank`, turn on `fp8`, increase `blocks_to_swap`, or drop
  resolution. Don't just retry the same config.
- Loss NaN'd out: the watchdog killed the run. Check `watchdog_report.json`
  — probably a learning-rate issue. Drop LR 2-5x and retry.
- Caption quality poor: re-run `klippbok_caption --overwrite` with a different
  `use_case` or provider.

---

*Everything above is a plan, not a script. The agent should confirm with the
user before destructive operations (overwriting a manifest, deleting a
dataset subfolder, re-captioning with --overwrite, etc.).*
"""


# ---------------------------------------------------------------- watchdog

# CLAUDE-NOTE: Embedded as a string constant (not a companion .py file) so
# there's exactly one source of truth — regenerate from here and the file
# stays synced when the template evolves. Uses stdlib only per spec; keep
# that constraint (no numpy, no rich, no tqdm) so the script runs in any
# venv without extra installs.
WATCHDOG_PY = '''\
#!/usr/bin/env python
"""Training watchdog — kill a training process if loss misbehaves.

Tails a log file, parses loss values, and terminates the target process
(by PID) when:

  * Loss > --max-loss for 10 consecutive observations.
  * Loss is NaN or Inf.
  * Loss has not decreased for --plateau-steps observations.

Writes `watchdog_report.json` to the log file's parent directory (or
cwd) on exit so the calling agent can inspect why the run stopped.

**Run this in the same environment as the trainer.** Windows cannot see
WSL2 PIDs and vice versa — if training runs inside WSL2, launch this
watchdog inside WSL2 too. Otherwise the PID lookup will fail or, worse,
hit an unrelated Windows process with the same number.

Process-kill strategy:

  * Windows native: `taskkill /F /T /PID <pid>` (tree kill — catches
    torchrun / accelerate / DataLoader worker children that a plain
    TerminateProcess would orphan, leaving GPU memory wired up). Falls
    back to direct TerminateProcess via ctypes if `taskkill` is not on
    PATH (stripped Windows setups).
  * POSIX (incl. WSL2): SIGTERM, wait 3s, escalate to SIGKILL. Signals
    hit only the target PID — trainers that fork worker pools should
    install their own SIGTERM handler to tear them down, or the run
    script should use `setsid` so the group can be killed by PGID.

Stdlib only: argparse, json, math, os, pathlib, re, signal, subprocess,
sys, time.

Usage:
  python watchdog.py --log-file <path> --pid <pid> \\
      [--max-loss 2.0] [--plateau-steps 500] \\
      [--poll-interval 2.0] [--report-dir <path>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# Matches common patterns: "loss=1.234", "loss: 1.234", '"loss": 1.234'.
LOSS_RE = re.compile(r'loss["\\s:=]+([\\-\\d\\.eE+]+|nan|inf)', re.IGNORECASE)


def parse_loss(line: str) -> Optional[float]:
    m = LOSS_RE.search(line)
    if not m:
        return None
    raw = m.group(1).lower()
    if raw == "nan":
        return float("nan")
    if raw in ("inf", "+inf", "-inf"):
        return float("inf") if not raw.startswith("-") else float("-inf")
    try:
        return float(raw)
    except ValueError:
        return None


def kill_pid(pid: int) -> tuple[bool, str]:
    """Terminate the target process (and on Windows its children) cross-platform.

    Returns ``(killed, method)``. Method is a short string describing what
    actually landed so `watchdog_report.json` is diagnosable after the fact.
    """
    try:
        if os.name == "nt":
            # Primary path: taskkill /F /T walks the process tree and kills
            # the parent + all descendants. Training runs spawn worker
            # processes (torchrun, accelerate launch, DataLoader workers
            # holding GPU memory); a direct TerminateProcess would orphan
            # them. We prefer taskkill because it's tree-aware.
            try:
                r = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    return True, "taskkill /F /T"
                # Exit 128 = process not found — likely already exited.
                if r.returncode == 128:
                    return False, f"pid {pid} not found (already exited?)"
                # Anything else: try the fallback path below.
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Fallback: direct TerminateProcess via ctypes. Single PID only;
            # worker children may survive and hold GPU memory — manual
            # cleanup required if this path is taken.
            import ctypes

            PROCESS_TERMINATE = 1
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if not handle:
                return False, f"OpenProcess failed for pid={pid}"
            ok = kernel32.TerminateProcess(handle, 1)
            kernel32.CloseHandle(handle)
            return bool(ok), "TerminateProcess (fallback, single PID only)"
        else:
            os.kill(pid, signal.SIGTERM)
            # Grace period — well-behaved trainers handle SIGTERM and clean
            # up worker pools. Escalate if they don't exit within 3s.
            for _ in range(30):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True, "SIGTERM"
            os.kill(pid, signal.SIGKILL)
            return True, "SIGKILL"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes

            SYNCHRONIZE = 0x00100000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def write_report(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[watchdog] Could not write report to {path}: {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Training watchdog.")
    ap.add_argument("--log-file", required=True, help="Path to the trainer log file to tail.")
    ap.add_argument("--pid", type=int, required=True, help="PID of the training process.")
    ap.add_argument("--max-loss", type=float, default=2.0,
                    help="Loss values above this (for 10 consecutive observations) trigger a kill.")
    ap.add_argument("--plateau-steps", type=int, default=500,
                    help="Kill if loss has not decreased in this many observations.")
    ap.add_argument("--poll-interval", type=float, default=2.0,
                    help="Seconds between log-file polls.")
    ap.add_argument("--report-dir", default=None,
                    help="Directory to write watchdog_report.json. Default: log file's parent.")
    args = ap.parse_args()

    log_path = Path(args.log_file).resolve()
    report_dir = Path(args.report_dir) if args.report_dir else log_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "watchdog_report.json"

    print(f"[watchdog] Tailing {log_path}, monitoring pid={args.pid}, "
          f"max_loss={args.max_loss}, plateau_steps={args.plateau_steps}")

    start_time = time.time()
    offset = 0
    if log_path.exists():
        offset = log_path.stat().st_size  # Start at end — only new lines matter.
    losses: list[float] = []
    best_loss: Optional[float] = None
    steps_since_best = 0
    consecutive_high = 0
    total_loss_lines = 0

    def finish(reason: str, exit_code: int = 0) -> int:
        last_loss = losses[-1] if losses else None
        duration = round(time.time() - start_time, 2)
        killed, how = (False, "") if reason == "process_exited" else kill_pid(args.pid)
        report = {
            "reason": reason,
            "last_loss": last_loss,
            "best_loss": best_loss,
            "total_loss_lines_seen": total_loss_lines,
            "steps_since_best": steps_since_best,
            "duration_seconds": duration,
            "pid": args.pid,
            "log_file": str(log_path),
            "killed": killed,
            "kill_method": how if killed else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_report(report_path, report)
        print(f"[watchdog] Exiting — reason={reason}, last_loss={last_loss}, "
              f"killed={killed} ({how}), report={report_path}")
        return exit_code

    while True:
        if not pid_alive(args.pid):
            return finish("process_exited")

        if not log_path.exists():
            time.sleep(args.poll_interval)
            continue

        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                chunk = fh.read()
                offset = fh.tell()
        except OSError as exc:
            print(f"[watchdog] Log read error: {exc}", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        for line in chunk.splitlines():
            loss = parse_loss(line)
            if loss is None:
                continue
            total_loss_lines += 1
            losses.append(loss)

            # NaN / Inf is instant death.
            if math.isnan(loss) or math.isinf(loss):
                return finish("loss_nan_or_inf", exit_code=1)

            # Consecutive high-loss check.
            if loss > args.max_loss:
                consecutive_high += 1
                if consecutive_high >= 10:
                    return finish("loss_too_high", exit_code=1)
            else:
                consecutive_high = 0

            # Plateau check.
            if best_loss is None or loss < best_loss:
                best_loss = loss
                steps_since_best = 0
            else:
                steps_since_best += 1
                if steps_since_best >= args.plateau_steps:
                    return finish("loss_plateau", exit_code=1)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------- pipeline guide


PIPELINE_GUIDE_MD = """\
# Agentic LoRA Training Pipeline
*Designed by [hoodtronik](https://github.com/hoodtronik)*

## What this is

A fully automated pipeline for training production-quality LoRAs using AI
agents. The pipeline combines three tools — each handling one stage of the
workflow — connected through MCP servers that any AI agent can operate.

**The pipeline**:

1. **Klippbok** → Curate and prepare your training dataset (scan, triage,
   caption, validate).
2. **Musubi Tuner** → Train LoRAs for Wan2.2, FLUX.2 Klein, Z-Image,
   HunyuanVideo, FramePack, and more.
3. **LTX-2.3 Trainer** → Train LoRAs for Lightricks' LTX-2.3 video model.

An AI agent (Claude, Gemini, etc.) orchestrates all three through MCP tool
calls — no manual CLI commands needed.

## Required tools

### Dataset curation

| Tool | Purpose | Install |
|------|---------|---------|
| **Klippbok** | Video dataset curation | You already have it (this app) |
| **klippbok-mcp** | MCP server for Klippbok | [github.com/hoodtronik/klippbok-mcp](https://github.com/hoodtronik/klippbok-mcp) |

### Training (install the ones you need)

| Tool | Models supported | Install |
|------|------------------|---------|
| **Musubi Tuner** | Wan2.2, FLUX.2, Z-Image, HunyuanVideo, FramePack, Qwen-Image | [Pinokio launcher](https://github.com/hoodtronik/musubi-tuner.pinokio) or [manual](https://github.com/kohya-ss/musubi-tuner) |
| **musubi-mcp** | MCP server for Musubi | [github.com/hoodtronik/musubi-mcp](https://github.com/hoodtronik/musubi-mcp) |
| **LTX-2.3 Trainer** | LTX-2.3 | [Pinokio launcher](https://github.com/hoodtronik/ltx-trainer-pinokio) or [manual](https://github.com/Lightricks/LTX-2) |
| **ltx-trainer-mcp** | MCP server for LTX trainer | [github.com/hoodtronik/ltx-trainer-mcp](https://github.com/hoodtronik/ltx-trainer-mcp) |

### Agent (pick one)

| Agent | Notes |
|-------|-------|
| **Antigravity (Gemini)** | Recommended — cloud-based, no local VRAM usage |
| **Claude Desktop** | Works great, uses Claude tokens |
| **Claude Code** | Best for debugging, overkill for routine runs |
| **Cursor** | Works with any MCP server |

## Step 1 — Install the MCP servers

Each MCP server is a lightweight Python package. Install with uv:

```bash
# Klippbok MCP
git clone https://github.com/hoodtronik/klippbok-mcp
cd klippbok-mcp && uv sync

# Musubi MCP (if training Wan2.2 / FLUX.2 / Z-Image / Hunyuan / FramePack)
git clone https://github.com/hoodtronik/musubi-mcp
cd musubi-mcp && uv sync

# LTX Trainer MCP (if training LTX-2.3)
git clone https://github.com/hoodtronik/ltx-trainer-mcp
cd ltx-trainer-mcp && uv sync
```

## Step 2 — Configure your agent

Add the MCP servers to your agent's configuration.

**For Antigravity / Claude Desktop** — add to your MCP config:

```json
{
  "mcpServers": {
    "klippbok": {
      "command": "uv",
      "args": ["run", "--directory", "<path-to>/klippbok-mcp", "klippbok-mcp"],
      "env": {
        "KLIPPBOK_PYTHON": "<path-to>/klippbok-pinokio/env/Scripts/python.exe",
        "GEMINI_API_KEY": "<your-gemini-key>"
      }
    },
    "musubi": {
      "command": "uv",
      "args": ["run", "--directory", "<path-to>/musubi-mcp", "musubi-mcp"],
      "env": {
        "MUSUBI_TUNER_DIR": "<path-to>/musubi-tuner.pinokio/app",
        "MUSUBI_PYTHON": "<path-to>/musubi-tuner.pinokio/app/.venv/Scripts/python.exe"
      }
    },
    "ltx-trainer": {
      "command": "uv",
      "args": ["run", "--directory", "<path-to>/ltx-trainer-mcp", "ltx-trainer-mcp"],
      "env": {
        "LTX_TRAINER_DIR": "<path-to>/LTX-2"
      }
    }
  }
}
```

Replace `<path-to>` with your actual install paths. The workspace's
`AGENT_INSTRUCTIONS.md` auto-detects these if it can; anything left as
`<NOT FOUND>` needs to be filled in.

## Step 3 — Create a workspace

Use the **Create Training Workspace** button above to scaffold a project
folder. Then add your source clips/images to the `datasets/` folder.

**Note:** After creating your workspace, the first thing the agent will do is
help you download the correct base models for training. Quantized models from
inference tools (Wan2GP, ComfyUI, etc.) do NOT work for training — you need the
full-precision checkpoints. Budget 30–80 GB of downloads per architecture depending
on model size. The generated `AGENT_INSTRUCTIONS.md` includes architecture-specific
download commands with exact file names and HuggingFace repo paths.

## Step 4 — Start the agent

Open your agent and point it at the workspace. The generated
`AGENT_INSTRUCTIONS.md` in your workspace has a complete prompt template.
Paste it as your system/workspace instructions and start with:

> "Scan the datasets folder. For each style subfolder, triage the clips,
> caption them, and validate the dataset. Then show me a training plan."

The agent will use the MCP tools to work through the pipeline.

## The Layered LoRA strategy (optional)

For cinematic and animation projects, consider training two tiers of LoRAs:

**Master LoRAs** — high-rank (128), 2000-3000 steps. Captures the overall
rendering style (lighting, materials, proportions). Trained on 100-150
diverse clips from a studio or visual style.

**Boutique LoRAs** — low-rank (32-64), 500-800 steps. Captures specific
textures, color palettes, or moods from a single scene or episode. Trained
on 15-25 "hero" clips. Meant to be stacked on top of a Master.

At inference, stack them with phased denoising: Master at 0.6-0.7 weight
during structure phase, Boutique at 0.8-1.1 during detail phase.

## Training watchdog

The workspace includes a `watchdog.py` script that monitors training and
kills the process if loss explodes, hits NaN, or plateaus. The agent
activates it automatically when starting training. You can safely walk
away from a training run — the watchdog handles catastrophic failures.

Run it alongside training like so (from inside the workspace):

```bash
python watchdog.py \\
    --log-file path/to/trainer.log \\
    --pid <training-pid> \\
    --max-loss 2.0 \\
    --plateau-steps 500
```

The script writes `watchdog_report.json` on exit describing why it
stopped. Stdlib only — no extra deps.

**Run the watchdog in the same environment as the trainer.** If your
trainer runs inside WSL2, launch the watchdog from inside WSL2 too —
Windows cannot see WSL2 process IDs and vice versa. The watchdog
auto-detects its host OS and picks the right kill primitive:

- **Windows native**: `taskkill /F /T /PID <pid>` (process-tree kill,
  catches torchrun / accelerate / DataLoader workers that would
  otherwise orphan and keep GPU memory wired up). Falls back to direct
  TerminateProcess via ctypes if `taskkill` is unavailable.
- **WSL2 / Linux / macOS**: SIGTERM → wait 3s → SIGKILL on the target
  PID. Well-behaved trainers install their own SIGTERM handler to clean
  up worker pools; trainers that don't should be launched via `setsid`
  so the whole process group can be killed explicitly.

## GPU requirements

- **Dataset curation (Klippbok)**: Minimal GPU needed; captioning uses cloud APIs.
- **Musubi training**: 12GB+ for images, 24GB+ for video. fp8 +
  `blocks_to_swap` help on smaller cards.
- **LTX-2.3 training**: 32GB+ with low-VRAM config (INT8), 80GB+ for the
  standard config.
- **Important**: only run ONE training job at a time. Don't run local LLMs
  during training (they'll contend for VRAM).

## Links

- Klippbok: https://github.com/alvdansen/klippbok
- Musubi Tuner: https://github.com/kohya-ss/musubi-tuner
- LTX-2 Trainer: https://github.com/Lightricks/LTX-2
- Pipeline by hoodtronik: https://github.com/hoodtronik
"""
