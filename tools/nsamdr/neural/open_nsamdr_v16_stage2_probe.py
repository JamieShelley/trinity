#!/usr/bin/env python3
"""Locate and optionally open the latest V16 Stage 2 visual probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _active_run(root: Path) -> Path | None:
    pointer = root / "stage2_resume_pointer.json"
    if pointer.is_file():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            run_dir = Path(str(payload.get("runDir") or ""))
            if run_dir.is_dir():
                return run_dir
        except (OSError, ValueError, TypeError):
            pass

    candidates = sorted(
        (path for path in root.glob("multiregion_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _open(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    command = ["open", str(path)] if sys.platform == "darwin" else ["xdg-open", str(path)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open the latest NSAMDR V16 Stage 2 live visual probe")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--open", action="store_true", dest="open_probe")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.repo_root.resolve() / "artifacts/nsamdr/diagnostics/v16_mini"
    run_dir = _active_run(root)
    if run_dir is None:
        print("[stage2-probe] No Stage 2 run directory found.")
        return 2

    latest = run_dir / "probes" / "latest"
    overview = latest / "ALL_FAMILIES.png"
    if not overview.is_file():
        print(f"[stage2-probe] Live probe not written yet: {overview}")
        print("[stage2-probe] It is created at the next validation point.")
        return 1

    print(f"[stage2-probe] Run      : {run_dir}")
    print(f"[stage2-probe] Overview : {overview}")
    index = latest / "probe_index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            print(f"[stage2-probe] Step     : {int(payload.get('step') or 0)}")
            for entry in list(payload.get("files") or []):
                print(f"[stage2-probe] Family   : {entry.get('family')} -> {latest / str(entry.get('file') or '')}")
        except (OSError, ValueError, TypeError):
            pass

    if args.open_probe:
        _open(overview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
