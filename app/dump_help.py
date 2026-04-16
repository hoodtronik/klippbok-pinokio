"""Dump `--help` output for every Klippbok subcommand into docs/cli_help.txt.

Run from inside app/ during the Pinokio install step (see install.js). The
Gradio form fields in app.py must match the flags in this file verbatim —
re-run after upgrading Klippbok to catch any drift.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# CLAUDE-NOTE: argparse respects $COLUMNS when wrapping help output. Pinning
# to 200 keeps the dumped file stable across terminals so a diff of cli_help.txt
# across Klippbok versions shows real flag changes, not cosmetic reflow.
SUBCOMMANDS: list[tuple[str, str]] = [
    ("klippbok.video", "scan"),
    ("klippbok.video", "triage"),
    ("klippbok.video", "ingest"),
    ("klippbok.video", "normalize"),
    ("klippbok.video", "caption"),
    ("klippbok.video", "score"),
    ("klippbok.video", "extract"),
    ("klippbok.video", "audit"),
    ("klippbok.dataset", "validate"),
    ("klippbok.dataset", "organize"),
]


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "docs" / "cli_help.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "COLUMNS": "200", "PYTHONUNBUFFERED": "1"}
    missing: list[str] = []

    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"# Klippbok --help dump (python {sys.version.split()[0]})\n\n")
        for module, cmd in SUBCOMMANDS:
            header = f"=== {module} {cmd} ==="
            print(header)
            fh.write(header + "\n")
            fh.flush()
            proc = subprocess.run(
                [sys.executable, "-m", module, cmd, "--help"],
                capture_output=True,
                text=True,
                env=env,
            )
            fh.write(proc.stdout)
            if proc.stderr:
                fh.write("\n--- stderr ---\n")
                fh.write(proc.stderr)
            if proc.returncode != 0:
                missing.append(f"{module} {cmd} (exit {proc.returncode})")
            fh.write("\n")

    print(f"\nWrote {out}")
    if missing:
        print("WARNING: the following subcommands failed --help (API drift?):")
        for m in missing:
            print(f"  - {m}")
        return 0  # non-fatal; surface in the log but don't block install
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
