#!/usr/bin/env python3
"""V16 low-impact GUI with monitored Stage 2 runtime and live status banner."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import nsamdr_v16_lowimpact_workflow_gui as low

base = low.base
multifamily = low.multifamily
_original_dispatcher_argv = base.App._dispatcher_argv
_original_build = base.App._build
_original_preview_refresh_tick = base.App._preview_refresh_tick


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dispatcher_argv(self: base.App, command: tuple[str, ...], args: list[str]) -> list[str]:
    if command == ("v14-mini-multiregion",):
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/v14/safe_live_resume_monitored_fourfamily_multiregion_diagnostic.py"),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _stage2_resume_choices(self: base.App) -> list[str]:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    mapping: dict[str, Path] = {}
    if not root.is_dir():
        self._stage2_resume_paths = mapping
        return []
    pointer = _read_json(root / "stage2_resume_pointer.json") or {}
    pointer_run = str(pointer.get("runDir") or "")
    labels: list[str] = []
    for run_dir in sorted((path for path in root.glob("multiregion_*") if path.is_dir()), key=lambda path: path.name, reverse=True):
        checkpoint = run_dir / "resume_checkpoint.pt"
        if not checkpoint.is_file():
            continue
        sidecar = _read_json(run_dir / "resume_checkpoint.json") or {}
        heartbeat = _read_json(run_dir / "stage2_heartbeat.json") or {}
        step = int(sidecar.get("step") or 0)
        maximum = int(sidecar.get("maximumSteps") or 0)
        exact = bool(step and maximum)
        if not exact and pointer_run:
            try:
                if Path(pointer_run).resolve() == run_dir.resolve():
                    step = int(pointer.get("step") or 0)
                    maximum = int(pointer.get("maximumSteps") or 0)
                    exact = bool(step and maximum)
            except OSError:
                pass
        if not step:
            step = int(heartbeat.get("step") or 0)
            maximum = int(heartbeat.get("maximumSteps") or 0)
        status = str(heartbeat.get("status") or "saved")
        short_name = run_dir.name.removeprefix("multiregion_")
        if step > 0 and maximum > 0:
            progress = f"{'' if exact else '~'}{step}/{maximum}"
        else:
            progress = "checkpoint"
        label = f"{short_name} | {progress} | {status}"
        mapping[label] = checkpoint.resolve()
        labels.append(label)
    self._stage2_resume_paths = mapping
    return labels


def _latest_stage2_paths(self: base.App) -> tuple[Path | None, dict[str, object], dict[str, object]]:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    pointer = _read_json(root / "stage2_resume_pointer.json") or {}
    run_dir: Path | None = None
    if pointer.get("runDir"):
        candidate = Path(str(pointer["runDir"]))
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None and root.is_dir():
        candidates = [path for path in root.glob("multiregion_*") if path.is_dir()]
        if candidates:
            run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    heartbeat = _read_json(run_dir / "stage2_heartbeat.json") if run_dir else None
    return run_dir, pointer, heartbeat or {}


def _fmt_eta(seconds: object) -> str:
    try:
        minutes = max(0.0, float(seconds)) / 60.0
    except (TypeError, ValueError):
        return "?"
    return f"{minutes / 60.0:.1f}h" if minutes >= 60.0 else f"{minutes:.0f}m"


def _refresh_stage2_status(self: base.App) -> None:
    if not hasattr(self, "stage2_live_status_var"):
        return
    run_dir, pointer, heartbeat = _latest_stage2_paths(self)
    if run_dir is None or not heartbeat:
        self.stage2_live_status_var.set("Stage 2: no durable run")
        return
    step = heartbeat.get("step", "?")
    maximum = heartbeat.get("maximumSteps", "?")
    status = str(heartbeat.get("status") or "unknown")
    eta = _fmt_eta(heartbeat.get("etaSeconds"))
    resume_step = pointer.get("step") if pointer else None
    gpu = heartbeat.get("gpu") if isinstance(heartbeat.get("gpu"), dict) else {}
    temp = gpu.get("temperatureC") if gpu else None
    util_avg = gpu.get("utilizationAveragePercent") if gpu else None
    util_now = gpu.get("utilizationPercent") if gpu else None
    vram = gpu.get("vramUsedMiB") if gpu else None
    vram_total = gpu.get("vramTotalMiB") if gpu else None
    parts = [f"Stage 2 {step}/{maximum} {status}", f"ETA {eta}"]
    if temp is not None:
        parts.append(f"{float(temp):.0f}C")
    if util_avg is not None:
        parts.append(f"GPU {float(util_avg):.0f}%avg" + (f"/{float(util_now):.0f}%now" if util_now is not None else ""))
    elif util_now is not None:
        parts.append(f"GPU {float(util_now):.0f}%now")
    if vram is not None and vram_total is not None:
        parts.append(f"VRAM {float(vram)/1024.0:.1f}/{float(vram_total)/1024.0:.1f}G")
    if resume_step is not None:
        parts.append(f"resume@{resume_step}")
    self.stage2_live_status_var.set(" | ".join(parts))


def _run_stage2_audit(self: base.App) -> None:
    existing = getattr(self, "_stage2_audit_process", None)
    if existing is not None and existing.poll() is None:
        messagebox.showinfo("Stage 2 family audit", "The CPU family-difficulty audit is already running.")
        return

    script = self.repo / "tools/nsamdr/neural/audit_nsamdr_v16_stage2_family_difficulty.py"
    if not script.is_file():
        messagebox.showerror("Stage 2 family audit", f"Audit script is missing:\n{script}")
        return

    command = [
        sys.executable,
        "-u",
        str(script),
        "--repo-root",
        str(self.repo),
    ]
    kwargs: dict[str, object] = {"cwd": str(self.repo)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    try:
        self._stage2_audit_process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        messagebox.showerror("Stage 2 family audit", f"Could not start audit:\n{exc}")
        return

    self.stage2_audit_status_var.set("Stage 2 audit: running (CPU)")


def _refresh_stage2_audit_status(self: base.App) -> None:
    if not hasattr(self, "stage2_audit_status_var"):
        return
    process = getattr(self, "_stage2_audit_process", None)
    if process is not None:
        code = process.poll()
        if code is None:
            self.stage2_audit_status_var.set("Stage 2 audit: running (CPU)")
            return
        self._stage2_audit_process = None
        self.stage2_audit_status_var.set(
            "Stage 2 audit: complete" if code == 0 else f"Stage 2 audit: failed ({code})"
        )
        return

    run_dir, _pointer, _heartbeat = _latest_stage2_paths(self)
    if run_dir is not None and (run_dir / "difficulty_audit.json").is_file():
        self.stage2_audit_status_var.set(f"Stage 2 audit: available ({run_dir.name.removeprefix('multiregion_')})")
    else:
        self.stage2_audit_status_var.set("Stage 2 audit: ready")


def _build(self: base.App) -> None:
    _original_build(self)
    self.stage2_live_status_var = tk.StringVar(value="Stage 2: checking...")
    ttk.Label(self.footer, textvariable=self.stage2_live_status_var, font=("Segoe UI", 9, "bold")).pack(side="right", padx=(6, 8))

    self.stage2_audit_status_var = tk.StringVar(value="Stage 2 audit: ready")
    ttk.Button(
        self.footer,
        text="Run Stage 2 Audit",
        command=lambda: _run_stage2_audit(self),
    ).pack(side="left", padx=(10, 4))
    ttk.Label(self.footer, textvariable=self.stage2_audit_status_var).pack(side="left", padx=(0, 8))

    _refresh_stage2_status(self)
    _refresh_stage2_audit_status(self)


def _preview_refresh_tick(self: base.App) -> None:
    try:
        _refresh_stage2_status(self)
        _refresh_stage2_audit_status(self)
    except (OSError, ValueError, TypeError, tk.TclError):
        pass
    _original_preview_refresh_tick(self)


multifamily._stage2_resume_choices = _stage2_resume_choices
base.App._dispatcher_argv = _dispatcher_argv
base.App._build = _build
base.App._preview_refresh_tick = _preview_refresh_tick


if __name__ == "__main__":
    raise SystemExit(base.main())
