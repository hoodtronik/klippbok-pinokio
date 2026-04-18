"""Launcher-side still-image captioning shim.

Klippbok's `klippbok.video caption` CLI iterates **video** files only, but
every VLM backend in `klippbok.caption.*` already ships a working
`caption_image(path, prompt)` method. This shim wires those in-process
backends directly so the Caption tab can caption PNG / JPG / WEBP files
without waiting for upstream to expose an `images` subcommand.

Robustness features (free-tier / rate-limit friendly):

* **Resumable.** A `.txt` sidecar next to an image is the completion
  marker — present + non-empty ⇒ skip. Re-running Caption picks up
  exactly where the previous run stopped, whether it stopped because
  of a crash, a rate-limit, or a user cancel.
* **Progress file.** `caption_progress.json` written next to the images
  after every successful caption (atomic replace). Holds running
  counts + last-processed name + status so the Run log can say
  "Resuming — X of Y already done." on the next invocation.
* **Rate-limit aware.** 429 / quota / RESOURCE_EXHAUSTED errors are
  caught, the loop pauses for `retry_delay` seconds, then tries again
  up to `max_retries` times. If still blocked, the shim returns with
  a user-friendly hint and leaves everything else on disk intact.
* **API-key swap mid-run.** Because the completion check is "does
  `<image>.txt` exist?", the user can update their key in Settings,
  re-click Run, and the loop just continues past the captioned files.

Design choice: zero changes to upstream Klippbok. We only IMPORT from
it. If Klippbok is reinstalled or updated, this shim keeps working as
long as the `VLMBackend.caption_image(path, prompt) -> str` contract
holds.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")

PROGRESS_FILE_NAME = "caption_progress.json"
PROGRESS_SCHEMA_VERSION = 1

DEFAULT_RETRY_DELAY = 60
DEFAULT_MAX_RETRIES = 3


def find_images(directory: Path) -> list[Path]:
    """Return image files directly inside `directory` (non-recursive, sorted).

    Non-recursive on purpose — matches upstream Klippbok's `caption`
    behavior of operating on a flat directory of clips. Recursing would
    risk captioning reference images in nested concept subfolders.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def find_videos(directory: Path) -> list[Path]:
    """Return video files directly inside `directory` (non-recursive, sorted).

    We use this to decide whether to invoke the video-captioning CLI at
    all; if the directory is images-only, skipping the subprocess saves
    several seconds of torch / transformers import time.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


# --------------------------------------------------------------- progress


def _progress_path(directory: Path) -> Path:
    return directory / PROGRESS_FILE_NAME


def read_progress(directory: Path) -> dict | None:
    """Return the progress dict if a valid one exists, else None."""
    p = _progress_path(directory)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_progress(directory: Path, progress: dict) -> None:
    """Atomic progress write. Best-effort — never raises to caller."""
    p = _progress_path(directory)
    tmp = p.with_suffix(".json.tmp")
    progress["updated_at"] = _now_iso()
    try:
        tmp.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(p)  # atomic on both POSIX and Windows
    except OSError:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- backends


def _make_backend(
    provider: str,
    *,
    model: str,
    base_url: str,
    caption_fps: int,
):
    """Construct the matching Klippbok VLMBackend.

    CLAUDE-NOTE: Each provider has a slightly different constructor
    (Replicate uses `api_token`, others `api_key`; OpenAICompat needs
    `base_url`). We lean on each backend's env-var fallback so API keys
    don't flow through this shim — the Caption tab already places them
    in os.environ via `extra_env`.
    """
    provider = (provider or "gemini").lower()
    if provider == "gemini":
        from klippbok.caption.gemini import GeminiBackend
        return GeminiBackend(
            model=model or "gemini-2.5-flash",
            caption_fps=caption_fps or 1,
        )
    if provider == "replicate":
        from klippbok.caption.replicate import ReplicateBackend
        return ReplicateBackend(
            model=model or "google/gemini-2.5-flash",
        )
    if provider == "openai":
        from klippbok.caption.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(
            base_url=base_url or "http://localhost:11434/v1",
            model=model or "llama3.2-vision",
            caption_fps=caption_fps or 1,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


# --------------------------------------------------------------- rate-limit


_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota",
    "resource_exhausted",
    "resource exhausted",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """String-match rate-limit signatures across provider SDKs.

    Each provider raises a different exception type; catching by string
    keeps this shim independent of Gemini / Replicate / OpenAI SDK
    version drift. False positives (e.g. "quota" in an unrelated
    message) would just cause a harmless wait-retry.
    """
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


# --------------------------------------------------------------- main loop


def caption_images(
    directory: str,
    *,
    provider: str = "gemini",
    use_case: str | None = None,
    anchor_word: str | None = None,
    tags: list[str] | None = None,
    overwrite: bool = False,
    base_url: str = "",
    model: str = "",
    caption_fps: int = 1,
    retry_delay: int = DEFAULT_RETRY_DELAY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    extra_env: dict | None = None,
) -> Iterator[str]:
    """Caption every still image in `directory`, yielding progress lines.

    Skip-if-exists is the default: an image with a non-empty `.txt`
    sidecar is never re-captioned unless `overwrite=True`. So running
    this function twice is a pickup, not a redo.

    Rate-limit handling: if the VLM raises an error whose message
    matches one of the `_RATE_LIMIT_MARKERS`, we sleep `retry_delay`
    seconds and retry the SAME image up to `max_retries` times. If the
    block persists, the function returns with a message that reminds
    the user they can just re-run to continue.
    """
    # Promote API keys into os.environ so the backend's constructor can
    # find them without having to plumb them through. Mirrors how the
    # video CLI subprocess inherits its environment via `extra_env`.
    if extra_env:
        for k, v in extra_env.items():
            if v:
                os.environ[k] = v

    directory_p = Path(directory)
    images = find_images(directory_p)
    if not images:
        return  # Silent no-op; caller logs "no images" if it wants to.

    # Accounting — distinguish "already done before this run" from the
    # work we do in this call so the progress file and final tally
    # report something meaningful to the user.
    total = len(images)
    already_done = sum(
        1 for p in images
        if p.with_suffix(".txt").exists() and p.with_suffix(".txt").stat().st_size > 0
    )
    to_do = total - already_done

    if already_done:
        yield (
            f"[image-caption] Resuming — {already_done} of {total} already done; "
            f"{to_do} remaining."
        )
    else:
        yield f"[image-caption] Found {total} image(s) in {directory}"

    if to_do == 0 and not overwrite:
        yield "[image-caption] Nothing to do (all images already captioned)."
        return

    try:
        from klippbok.caption.prompts import get_image_prompt
    except Exception as exc:
        yield f"[image-caption] ERROR: could not import Klippbok prompts: {exc}"
        return

    prompt = get_image_prompt(
        use_case=use_case or None,
        anchor_word=anchor_word or None,
        secondary_anchors=tags or None,
    )

    try:
        backend = _make_backend(
            provider,
            model=model or "",
            base_url=base_url or "",
            caption_fps=int(caption_fps) if caption_fps else 1,
        )
    except Exception as exc:
        yield f"[image-caption] ERROR initializing '{provider}' backend: {exc}"
        return

    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "directory": str(directory_p),
        "provider": provider,
        "total_images": total,
        "already_captioned": already_done,
        "captioned_this_run": 0,
        "skipped_this_run": 0,
        "failed_this_run": 0,
        "remaining": to_do,
        "last_processed": "",
        "last_error": "",
        "status": "running",
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _write_progress(directory_p, progress)

    captioned = skipped = failed = 0
    for i, image_path in enumerate(images, start=1):
        sidecar = image_path.with_suffix(".txt")
        if sidecar.exists() and sidecar.stat().st_size > 0 and not overwrite:
            skipped += 1
            progress["skipped_this_run"] = skipped
            continue

        # Rate-limit-aware inner loop: retry the SAME image up to
        # max_retries times with `retry_delay` between attempts. Any
        # non-rate-limit exception fails this image and moves on.
        caption: str | None = None
        for attempt in range(max_retries + 1):
            try:
                caption = backend.caption_image(image_path, prompt)
                break
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    if attempt < max_retries:
                        yield (
                            f"[image-caption] rate limit hit on {image_path.name} — "
                            f"waiting {retry_delay}s (retry {attempt + 1}/{max_retries})"
                        )
                        progress["status"] = "rate_limited_retrying"
                        progress["last_error"] = str(exc)[:500]
                        _write_progress(directory_p, progress)
                        time.sleep(retry_delay)
                        continue
                    # Exhausted retries: stop the whole run cleanly.
                    remaining = total - already_done - captioned - skipped
                    progress["status"] = "rate_limited"
                    progress["failed_this_run"] = failed
                    progress["last_error"] = str(exc)[:500]
                    progress["remaining"] = remaining
                    _write_progress(directory_p, progress)
                    yield (
                        f"[image-caption] API rate limit hit after "
                        f"{captioned} images this run. {remaining} images remaining. "
                        f"Re-run this step to continue from where it left off. "
                        f"If you're on the free tier, wait ~60 seconds or switch "
                        f"to a different API key in Settings."
                    )
                    return
                # Non-rate-limit error → log and keep going.
                yield f"[image-caption] FAILED {image_path.name}: {exc}"
                failed += 1
                progress["failed_this_run"] = failed
                progress["last_error"] = str(exc)[:500]
                _write_progress(directory_p, progress)
                break

        if caption is None:
            continue  # failure already logged above

        try:
            sidecar.write_text(caption, encoding="utf-8")
        except OSError as exc:
            yield f"[image-caption] FAILED writing {sidecar.name}: {exc}"
            failed += 1
            progress["failed_this_run"] = failed
            progress["last_error"] = str(exc)[:500]
            _write_progress(directory_p, progress)
            continue

        captioned += 1
        progress["captioned_this_run"] = captioned
        progress["last_processed"] = image_path.name
        progress["remaining"] = total - already_done - captioned - skipped
        progress["status"] = "running"
        _write_progress(directory_p, progress)
        yield f"[{already_done + captioned}/{total}] Captioned {image_path.name}"

    # Final tally + terminal status.
    progress["status"] = "completed" if failed == 0 else "completed_with_failures"
    progress["remaining"] = total - already_done - captioned - skipped
    _write_progress(directory_p, progress)
    yield (
        f"[image-caption] done — captioned_this_run={captioned} "
        f"skipped={skipped} failed={failed} "
        f"total_done={already_done + captioned}/{total}"
    )
