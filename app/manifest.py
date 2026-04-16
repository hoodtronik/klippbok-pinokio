"""Manifest Reviewer helpers — schema adapter + thumbnail cache.

Klippbok's triage stage writes one of two JSON shapes:

  * `triage_manifest.json` — clip-level, flat list of clips each with
    `include: true|false`.
  * `scene_triage_manifest.json` — scene-level, per source video nests a
    `scenes[]` array where each scene has its own `include` flag plus
    `start_time` / `end_time`.

The reviewer UI doesn't want to know the difference. `load_manifest` flattens
either shape into a list of `Entry` objects; `save_manifest` writes the edited
`include` values back to the original structure without touching any other
field. This way we don't care if Klippbok adds new metadata later — we only
read/write the one flag the reviewer edits.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


ManifestKind = Literal["clip", "scene"]


# CLAUDE-NOTE: Entry is the UI-facing row model. Kept deliberately thin — the
# original manifest dict is the source of truth for any field the reviewer
# doesn't touch (triage metadata, match lists, concept lookup tables). On save
# we walk entries and write `include` back to the original structure using the
# stored *_idx pointers.
@dataclass
class Entry:
    idx: int                       # position in the flattened list
    kind: ManifestKind
    video_path: str                # absolute path for ffmpeg thumbnail
    display_label: str             # short label shown in the UI
    start: Optional[float]         # scene start (seconds); None for clip-level
    end: Optional[float]           # scene end (seconds); None for clip-level
    score: float                   # similarity of the best match, 0.0 if none
    best_concept: str              # label of the top match, "" if none
    include: bool
    text_overlay: bool
    use_case: Optional[str]
    # Writeback pointers — exactly one pair is populated per kind.
    clip_idx: Optional[int] = None
    video_idx: Optional[int] = None
    scene_idx: Optional[int] = None


def load_manifest(path: str | Path) -> tuple[dict, list[Entry], ManifestKind]:
    """Parse either manifest schema into (raw_dict, flat_entries, kind)."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    # Scene-level detection: prefer the explicit marker, fall back to the
    # presence of `videos[]`. Klippbok writes `triage_mode: "scene"` when
    # processing long source videos.
    if raw.get("triage_mode") == "scene" or "videos" in raw:
        entries: list[Entry] = []
        for vi, v in enumerate(raw.get("videos", [])):
            vpath = v.get("path") or v.get("file", "")
            vname = Path(vpath).name or "<unknown>"
            for si, s in enumerate(v.get("scenes", [])):
                matches = s.get("matches") or []
                best = matches[0] if matches else {}
                start = float(s.get("start_time", 0.0))
                end = s.get("end_time")
                entries.append(
                    Entry(
                        idx=len(entries),
                        kind="scene",
                        video_path=vpath,
                        display_label=f"{vname}  [scene {si}  {start:.1f}s–{float(end):.1f}s]"
                        if end is not None
                        else f"{vname}  [scene {si}  {start:.1f}s+]",
                        start=start,
                        end=float(end) if end is not None else None,
                        score=float(best.get("similarity", 0.0)),
                        best_concept=str(best.get("concept", "")),
                        include=bool(s.get("include", True)),
                        text_overlay=bool(s.get("text_overlay", False)),
                        use_case=None,
                        video_idx=vi,
                        scene_idx=si,
                    )
                )
        return raw, entries, "scene"

    if "clips" in raw:
        entries = []
        for ci, c in enumerate(raw.get("clips", [])):
            matches = c.get("matches") or []
            best = matches[0] if matches else {}
            cpath = c.get("path") or c.get("file", "")
            entries.append(
                Entry(
                    idx=len(entries),
                    kind="clip",
                    video_path=cpath,
                    display_label=Path(cpath).name or c.get("file", "<unknown>"),
                    start=None,
                    end=None,
                    score=float(best.get("similarity", 0.0)),
                    best_concept=str(best.get("concept", "")),
                    include=bool(c.get("include", True)),
                    text_overlay=bool(c.get("text_overlay", False)),
                    use_case=c.get("use_case"),
                    clip_idx=ci,
                )
            )
        return raw, entries, "clip"

    raise ValueError(
        f"Unrecognized manifest schema. Expected either `clips` (clip-level) "
        f"or `videos` / `triage_mode: scene` (scene-level). Got top-level "
        f"keys: {sorted(raw.keys())}"
    )


def save_manifest(raw: dict, entries: list[Entry], path: str | Path) -> None:
    """Write `include` values from entries back into raw, then serialize."""
    for e in entries:
        if e.kind == "clip" and e.clip_idx is not None:
            raw["clips"][e.clip_idx]["include"] = bool(e.include)
        elif e.kind == "scene" and e.video_idx is not None and e.scene_idx is not None:
            raw["videos"][e.video_idx]["scenes"][e.scene_idx]["include"] = bool(e.include)
    # CLAUDE-NOTE: ensure_ascii=False preserves non-ASCII chars in filenames
    # (common in video filenames). indent=2 matches Klippbok's own writer so a
    # round-trip load/save is a no-op diff for reviewers using `git diff`.
    Path(path).write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def reviewed_path_for(original: str | Path) -> Path:
    """`foo.json` -> `foo_reviewed.json`. Stable so the user can re-save."""
    p = Path(original)
    return p.with_name(p.stem + "_reviewed" + p.suffix)


# ----- thumbnail extraction ------------------------------------------------


def _seek_seconds(entry: Entry) -> float:
    """Midpoint of the scene, or 0.5s into clip-level entries."""
    if entry.start is not None and entry.end is not None and entry.end > entry.start:
        return (entry.start + entry.end) / 2.0
    if entry.start is not None:
        return entry.start
    return 0.5


def thumbnail_key(entry: Entry) -> str:
    """Stable filesystem-safe id for the cached thumbnail of this entry."""
    seed = f"{entry.video_path}|{entry.start}|{entry.end}|{entry.kind}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


def thumbnail_path(entry: Entry, cache_dir: Path) -> Path:
    return cache_dir / f"{thumbnail_key(entry)}.jpg"


def generate_thumbnail(entry: Entry, cache_dir: Path, timeout: float = 8.0) -> Optional[str]:
    """Extract one JPG frame at the midpoint into `cache_dir`.

    Returns the cached path on success, None if the file is missing or ffmpeg
    fails. Idempotent: existing cached thumbnails are returned without
    re-running ffmpeg. Safe to call from worker threads.
    """
    out = thumbnail_path(entry, cache_dir)
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    if not entry.video_path:
        return None
    source = Path(entry.video_path)
    if not source.exists() or not source.is_file():
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    seek = _seek_seconds(entry)

    # CLAUDE-NOTE: -ss BEFORE -i enables fast seek (container-level) — frame
    # accuracy is worse but thumbnails don't need it and this is ~20x faster
    # on long files. -vf scale keeps the cached file tiny (< 20KB typically).
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{seek:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(320,iw)':-2",
        "-q:v",
        "5",
        "-y",
        str(out),
    ]
    try:
        subprocess.run(
            cmd,
            timeout=timeout,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    return None
