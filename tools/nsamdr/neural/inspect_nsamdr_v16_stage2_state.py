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


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "?"


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
    run_dir = heartbeat_path.parent
    print(f"[v16-stage2-status] run        : {run_dir}")
    print(f"[v16-stage2-status] heartbeat  : {heartbeat_path}")
    print(f"[v16-stage2-status] status     : {heartbeat.get('status', 'unknown')}")
    print(
        f"[v16-stage2-status] progress   : "
        f"{heartbeat.get('step', '?')}/{heartbeat.get('maximumSteps', '?')}"
    )
    print(f"[v16-stage2-status] elapsed    : {_fmt_minutes(heartbeat.get('elapsedSeconds'))}")
    print(f"[v16-stage2-status] ETA        : {_fmt_minutes(heartbeat.get('etaSeconds'))}")

    gpu = heartbeat.get("gpu") if isinstance(heartbeat.get("gpu"), dict) else {}
    if gpu:
        util_avg = gpu.get("utilizationAveragePercent")
        util_now = gpu.get("utilizationPercent")
        util_peak = gpu.get("utilizationPeakPercent")
        power_avg = gpu.get("powerAverageW")
        power_now = gpu.get("powerW")
        temp_now = gpu.get("temperatureC")
        temp_peak = gpu.get("temperaturePeakC")
        vram_now = gpu.get("vramUsedMiB")
        vram_total = gpu.get("vramTotalMiB")
        vram_peak = gpu.get("vramPeakMiB")

        if util_avg is not None:
            print(
                "[v16-stage2-status] GPU load   : "
                f"avg={_fmt_num(util_avg)}% now={_fmt_num(util_now)}% "
                f"peak={_fmt_num(util_peak)}%"
            )
        else:
            print(
                "[v16-stage2-status] GPU load   : "
                f"instantaneous={_fmt_num(util_now)}% (rolling sampler unavailable)"
            )
        print(
            "[v16-stage2-status] GPU therm  : "
            f"temp={_fmt_num(temp_now)}C rolling-peak={_fmt_num(temp_peak)}C "
            f"power-avg={_fmt_num(power_avg)}W now={_fmt_num(power_now)}W "
            f"limit={_fmt_num(gpu.get('powerLimitW'))}W"
        )
        print(
            "[v16-stage2-status] GPU VRAM   : "
            f"now={_fmt_num(vram_now)}/{_fmt_num(vram_total)} MiB "
            f"rolling-peak={_fmt_num(vram_peak)} MiB"
        )

    progress = run_dir / "stage2_progress.jsonl"
    if progress.is_file():
        print(f"[v16-stage2-status] progress   : {progress}")

    checkpoint = run_dir / "resume_checkpoint.pt"
    sidecar = _read_json(run_dir / "resume_checkpoint.json") or {}
    if checkpoint.is_file():
        durable_step = sidecar.get("step")
        durable_max = sidecar.get("maximumSteps")
        if durable_step is None and pointer:
            try:
                if Path(str(pointer.get("runDir") or "")).resolve() == run_dir.resolve():
                    durable_step = pointer.get("step")
                    durable_max = pointer.get("maximumSteps")
            except OSError:
                pass
        progress_text = (
            f" step={durable_step}/{durable_max}"
            if durable_step is not None and durable_max is not None
            else ""
        )
        print(f"[v16-stage2-status] resume      : READY{progress_text} {checkpoint}")
    elif pointer:
        pointer_checkpoint = Path(str(pointer.get("checkpoint") or ""))
        print(
            f"[v16-stage2-status] resume      : "
            f"{'READY' if pointer_checkpoint.is_file() else 'MISSING'} {pointer_checkpoint}"
        )
    else:
        print("[v16-stage2-status] resume      : no retained checkpoint")

    probe = run_dir / "probes/latest/ALL_FAMILIES.png"
    detail_probe = run_dir / "probes/latest/ALL_FAMILIES_DETAIL.png"
    if probe.is_file():
        print(f"[v16-stage2-status] probe       : {probe}")
    if detail_probe.is_file():
        print(f"[v16-stage2-status] probe-all   : {detail_probe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
