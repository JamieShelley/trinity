#!/usr/bin/env python3
"""Print the latest durable V16 Stage 2 heartbeat/resume state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_heartbeat(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.glob("multiregion_*/stage2_heartbeat.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _fmt_minutes(value: Any) -> str:
    try:
        return f"{float(value) / 60.0:.1f} min"
    except (TypeError, ValueError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect durable V16 Stage 2 state")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    root = args.repo_root.resolve() / "artifacts/nsamdr/diagnostics/v16_mini"
    pointer_path = root / "stage2_resume_pointer.json"
    pointer = _read_json(pointer_path)

    heartbeat_path: Path | None = None
    if pointer:
        run_dir = Path(str(pointer.get("runDir") or ""))
        candidate = run_dir / "stage2_heartbeat.json"
        if candidate.is_file():
            heartbeat_path = candidate
    if heartbeat_path is None:
        heartbeat_path = _latest_heartbeat(root)

    if heartbeat_path is None:
        print("[v16-stage2-status] no durable Stage 2 heartbeat found")
        return 1

    heartbeat = _read_json(heartbeat_path) or {}
    print(f"[v16-stage2-status] heartbeat : {heartbeat_path}")
    print(f"[v16-stage2-status] status    : {heartbeat.get('status', 'unknown')}")
    print(
        f"[v16-stage2-status] progress  : "
        f"{heartbeat.get('step', '?')}/{heartbeat.get('maximumSteps', '?')}"
    )
    print(f"[v16-stage2-status] elapsed   : {_fmt_minutes(heartbeat.get('elapsedSeconds'))}")
    print(f"[v16-stage2-status] ETA       : {_fmt_minutes(heartbeat.get('etaSeconds'))}")

    gpu = heartbeat.get("gpu") if isinstance(heartbeat.get("gpu"), dict) else {}
    if gpu:
        print(
            "[v16-stage2-status] GPU       : "
            f"temp={gpu.get('temperatureC', '?')}C "
            f"power={gpu.get('powerW', '?')}/{gpu.get('powerLimitW', '?')}W "
            f"util={gpu.get('utilizationPercent', '?')}% "
            f"VRAM={gpu.get('vramUsedMiB', '?')}/{gpu.get('vramTotalMiB', '?')} MiB"
        )

    progress = heartbeat_path.parent / "stage2_progress.jsonl"
    if progress.is_file():
        print(f"[v16-stage2-status] progress   : {progress}")

    if pointer:
        checkpoint = Path(str(pointer.get("checkpoint") or ""))
        print(
            f"[v16-stage2-status] resume     : "
            f"{'READY' if checkpoint.is_file() else 'MISSING'} {checkpoint}"
        )
    else:
        print("[v16-stage2-status] resume     : no active resume pointer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
