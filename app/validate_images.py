"""Launcher-side image-dataset validation shim.

Klippbok's `klippbok.dataset validate` CLI hardcodes video extensions
when looking for "target" files (see `klippbok/dataset/discover.py`
VIDEO_EXTENSIONS), so image-model LoRA datasets — PNG/JPG frames with
matching `.txt` captions — get flagged "no matching target video" for
every sample even though the pair is structurally valid.

This shim reproduces the parts of `validate` that actually matter for
still-image datasets:

* **Stem-based pairing** of captions ↔ images (↔ videos, for mixed dirs).
* **Orphan detection** — images without captions, captions without media.
* **--quality** — reuses Klippbok's `compute_sharpness` (blur via
  Laplacian variance), `compute_exposure`, `is_blank`.
* **--duplicates** — reuses Klippbok's `find_duplicates` (dHash, no
  extra deps — already numpy/opencv).

What we deliberately DON'T reimplement: video-specific checks (bucket
preview, frame count, fps, duration). If the directory is videos-only
the launcher delegates to Klippbok's CLI unchanged; if it's mixed, the
shim handles the pairing (both video and image stems count as valid
targets) and notes that bucket/frame checks skip for image entries.

Design choice mirrors caption_images.py: zero upstream modification,
just imports — so a Klippbok reinstall or version bump can't silently
revert this feature.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")
VIDEO_EXTS = (
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
    ".mpg", ".mpeg", ".flv", ".wmv", ".ts", ".mts",
)

# Launcher-generated .json / .txt files that aren't caption sidecars —
# excluded from caption detection so they don't register as orphans.
KNOWN_NON_CAPTION_STEMS = frozenset({
    "caption_progress",
    "klippbok_manifest",
    "klippbok_project",
    "klippbok_data",
    "scene_triage_manifest",
    "triage_manifest",
})


# --------------------------------------------------------------- scanning


def find_by_ext(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    ext_set = {e.lower() for e in extensions}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ext_set
    )


def find_images(directory: Path) -> list[Path]:
    return find_by_ext(directory, IMAGE_EXTS)


def find_videos(directory: Path) -> list[Path]:
    return find_by_ext(directory, VIDEO_EXTS)


def find_captions(directory: Path) -> list[Path]:
    """.txt files that look like caption sidecars (by stem filter)."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
        and p.stem not in KNOWN_NON_CAPTION_STEMS
    )


def classify_directory(directory: Path) -> str:
    """Return "images", "videos", "mixed", or "empty" for routing."""
    imgs = bool(find_images(directory))
    vids = bool(find_videos(directory))
    if imgs and vids:
        return "mixed"
    if imgs:
        return "images"
    if vids:
        return "videos"
    return "empty"


# --------------------------------------------------------------- pairing


def pair_media_captions(
    media: list[Path],
    captions: list[Path],
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    """Stem-match media files to caption files.

    Returns (pairs, media_without_captions, captions_without_media).
    media can mix images and videos — the pairing logic is ext-agnostic.
    """
    media_by_stem = {p.stem: p for p in media}
    caption_by_stem = {p.stem: p for p in captions}

    pairs: list[tuple[Path, Path]] = []
    media_orphans: list[Path] = []
    caption_orphans: list[Path] = []

    for stem, m in media_by_stem.items():
        c = caption_by_stem.get(stem)
        if c is None:
            media_orphans.append(m)
        else:
            pairs.append((m, c))

    for stem, c in caption_by_stem.items():
        if stem not in media_by_stem:
            caption_orphans.append(c)

    return pairs, media_orphans, caption_orphans


# --------------------------------------------------------------- checks


def _safe_sharpness(path: Path) -> float | None:
    try:
        from klippbok.video.image_quality import compute_sharpness
        return float(compute_sharpness(path))
    except Exception:
        return None


def _safe_exposure(path: Path) -> tuple[float, float] | None:
    try:
        from klippbok.dataset.quality import compute_exposure
        return compute_exposure(path)
    except Exception:
        return None


def _safe_is_blank(path: Path, threshold: float = 5.0) -> bool | None:
    try:
        from klippbok.video.image_quality import is_blank
        return bool(is_blank(path, threshold=threshold))
    except Exception:
        return None


def _safe_duplicates(paths: list[Path]) -> list[list[Path]] | None:
    try:
        from klippbok.dataset.quality import find_duplicates
        return find_duplicates(paths)
    except Exception:
        return None


# --------------------------------------------------------------- report model


def _empty_report(directory: str) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "klippbok-pinokio launcher validate shim",
        "directory": directory,
        "dir_classification": "empty",
        "counts": {
            "images": 0, "videos": 0, "captions": 0,
            "pairs": 0, "media_orphans": 0, "caption_orphans": 0,
        },
        "pairs": [],
        "media_orphans": [],
        "caption_orphans": [],
        "quality_issues": [],
        "duplicate_groups": [],
        "errors": [],
        "warnings": [],
        "status": "PASS",
    }


def _format_plaintext(report: dict) -> str:
    lines: list[str] = []
    lines.append("Dataset Validation Report (image-aware launcher shim)")
    lines.append("=" * 54)
    lines.append(f"Directory:        {report['directory']}")
    lines.append(f"Classification:   {report['dir_classification']}")
    c = report["counts"]
    lines.append(f"Images:           {c['images']}")
    lines.append(f"Videos:           {c['videos']}")
    lines.append(f"Captions (.txt):  {c['captions']}")
    lines.append(f"Pairs:            {c['pairs']}")
    lines.append(f"Media orphans:    {c['media_orphans']}")
    lines.append(f"Caption orphans:  {c['caption_orphans']}")
    lines.append(f"Quality issues:   {len(report['quality_issues'])}")
    lines.append(f"Duplicate groups: {len(report['duplicate_groups'])}")
    lines.append(f"Status:           {report['status']}")

    if report["media_orphans"]:
        lines.append("")
        lines.append("Media without captions:")
        for p in report["media_orphans"][:50]:
            lines.append(f"  [WARN ] {p}")
        if len(report["media_orphans"]) > 50:
            lines.append(f"  ... and {len(report['media_orphans']) - 50} more")

    if report["caption_orphans"]:
        lines.append("")
        lines.append("Captions without media:")
        for p in report["caption_orphans"][:50]:
            lines.append(f"  [WARN ] {p}")
        if len(report["caption_orphans"]) > 50:
            lines.append(f"  ... and {len(report['caption_orphans']) - 50} more")

    if report["quality_issues"]:
        lines.append("")
        lines.append("Quality issues:")
        for issue in report["quality_issues"][:100]:
            sev = issue.get("severity", "WARN")
            path = issue.get("path", "?")
            msg = issue.get("message", "")
            lines.append(f"  [{sev:5s}] {path} — {msg}")
        if len(report["quality_issues"]) > 100:
            lines.append(f"  ... and {len(report['quality_issues']) - 100} more")

    if report["duplicate_groups"]:
        lines.append("")
        lines.append("Duplicate groups:")
        for i, group in enumerate(report["duplicate_groups"], start=1):
            lines.append(f"  Group {i}: {', '.join(Path(p).name for p in group)}")

    if report["errors"]:
        lines.append("")
        lines.append("Errors:")
        for e in report["errors"]:
            lines.append(f"  [ERROR] {e}")

    return "\n".join(lines)


# --------------------------------------------------------------- main


def validate_directory(
    directory: str,
    *,
    quality: bool = False,
    duplicates: bool = False,
    json_output: bool = False,
    write_manifest: bool = False,
    blur_threshold: float = 100.0,
    exposure_min: float = 0.15,
    exposure_max: float = 0.85,
    frame_count_rule: str = "off",
) -> Iterator[str]:
    """Validate an image (or image+video) dataset, yielding log lines.

    The last yield is the final formatted report (text or JSON). The
    generator yields intermediate progress lines so long runs give
    visible feedback in the Gradio log pane.
    """
    dir_p = Path(directory)
    if not dir_p.is_dir():
        yield f"[validate-images] ERROR: not a directory: {directory}"
        return

    images = find_images(dir_p)
    videos = find_videos(dir_p)
    captions = find_captions(dir_p)
    kind = classify_directory(dir_p)

    yield (
        f"[validate-images] Scanning {directory} — "
        f"images={len(images)} videos={len(videos)} captions={len(captions)} "
        f"(kind: {kind})"
    )

    report = _empty_report(directory)
    report["dir_classification"] = kind
    report["counts"]["images"] = len(images)
    report["counts"]["videos"] = len(videos)
    report["counts"]["captions"] = len(captions)

    # Pair media (images + videos) to captions. In mixed directories a
    # caption is satisfied if EITHER a video or an image shares its stem.
    media = [*images, *videos]
    pairs, media_orphans, caption_orphans = pair_media_captions(media, captions)
    report["pairs"] = [
        {"media": str(m), "caption": str(c)} for m, c in pairs
    ]
    report["media_orphans"] = [str(p) for p in media_orphans]
    report["caption_orphans"] = [str(p) for p in caption_orphans]
    report["counts"]["pairs"] = len(pairs)
    report["counts"]["media_orphans"] = len(media_orphans)
    report["counts"]["caption_orphans"] = len(caption_orphans)

    yield (
        f"[validate-images] Paired {len(pairs)} sample(s); "
        f"{len(media_orphans)} media orphans, {len(caption_orphans)} caption orphans"
    )

    # CLAUDE-NOTE: Frame-count rule is informational in this shim — we
    # surface what the dataset is being held to but don't probe video
    # frame counts here (that requires ffprobe and matches what
    # Klippbok's CLI already enforces for video-only dirs). For image
    # entries the rule is trivially satisfied (1 frame each).
    if frame_count_rule and frame_count_rule != "off":
        report["frame_count_rule"] = frame_count_rule
        if videos:
            yield (
                f"[validate-images] Frame-count rule: {frame_count_rule}. "
                f"Image entries trivially comply (1 frame). For the "
                f"{len(videos)} video file(s), Klippbok's own validate "
                f"enforces this when --config klippbok_data.yaml has the "
                f"matching constraint set."
            )
        else:
            yield (
                f"[validate-images] Frame-count rule: {frame_count_rule}. "
                f"Image-only dataset — every entry trivially complies."
            )

    if quality and images:
        yield f"[validate-images] Running quality checks on {len(images)} image(s)..."
        for i, img in enumerate(images, start=1):
            if i % 25 == 1:
                yield f"[validate-images] quality: {i}/{len(images)}"
            blank = _safe_is_blank(img)
            if blank:
                report["quality_issues"].append({
                    "path": str(img),
                    "severity": "ERROR",
                    "code": "BLANK_IMAGE",
                    "message": "image is blank or near-blank",
                })
                continue
            sharp = _safe_sharpness(img)
            if sharp is not None and sharp < blur_threshold:
                report["quality_issues"].append({
                    "path": str(img),
                    "severity": "WARN",
                    "code": "BLUR_BELOW_THRESHOLD",
                    "message": f"sharpness={sharp:.1f} < {blur_threshold}",
                    "actual": sharp,
                    "expected_min": blur_threshold,
                })
            exp = _safe_exposure(img)
            if exp is not None:
                mean, _std = exp
                if mean < exposure_min:
                    report["quality_issues"].append({
                        "path": str(img),
                        "severity": "WARN",
                        "code": "EXPOSURE_TOO_DARK",
                        "message": f"mean brightness={mean:.2f} < {exposure_min}",
                        "actual": mean,
                    })
                elif mean > exposure_max:
                    report["quality_issues"].append({
                        "path": str(img),
                        "severity": "WARN",
                        "code": "EXPOSURE_TOO_BRIGHT",
                        "message": f"mean brightness={mean:.2f} > {exposure_max}",
                        "actual": mean,
                    })
        yield f"[validate-images] Quality issues: {len(report['quality_issues'])}"
    elif quality and not images:
        report["warnings"].append(
            "--quality requested but no images found; the launcher shim "
            "only runs image-level quality checks."
        )

    if duplicates and images:
        yield f"[validate-images] Running perceptual duplicate check on {len(images)} image(s)..."
        groups = _safe_duplicates(images)
        if groups is None:
            report["errors"].append("duplicate check failed to import or run")
        else:
            report["duplicate_groups"] = [[str(p) for p in g] for g in groups]
            yield f"[validate-images] Duplicate groups: {len(groups)}"
    elif duplicates and not images:
        report["warnings"].append(
            "--duplicates requested but no images found; perceptual "
            "dedup is only wired for images in this shim."
        )

    # Status: FAIL if any errors, any caption orphans, or any blank-image
    # quality issues. Blur/exposure alone are warnings, not fails.
    hard_failures = (
        bool(report["errors"])
        or len(report["caption_orphans"]) > 0
        or any(
            q.get("severity") == "ERROR"
            for q in report["quality_issues"]
        )
    )
    report["status"] = "FAIL" if hard_failures else "PASS"

    # Optional manifest write — matches the placement Klippbok uses:
    # inside the dataset directory, json next to the clips/images.
    if write_manifest:
        manifest_path = dir_p / "klippbok_manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(
                    {
                        "generator": report["generator"],
                        "generated_at": report["generated_at"],
                        "directory": report["directory"],
                        "pairs": report["pairs"],
                        "counts": report["counts"],
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            yield f"[validate-images] Wrote manifest: {manifest_path}"
        except OSError as exc:
            report["errors"].append(f"manifest write failed: {exc}")
            yield f"[validate-images] FAILED writing manifest: {exc}"

    # Final yield: the rendered report. JSON dumps the full dict;
    # text rendering mimics Klippbok's plain-text report header so the
    # Gradio log looks consistent between image and video runs.
    if json_output:
        yield json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    else:
        yield ""  # blank line for visual separation
        yield _format_plaintext(report)
