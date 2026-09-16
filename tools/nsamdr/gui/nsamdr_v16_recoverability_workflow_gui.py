#!/usr/bin/env python3
"""V16.1 monitored GUI plus CPU-only Stage 2 recoverability audit."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import nsamdr_v16_lowimpact_monitored_workflow_gui as monitored

base = monitored.base
_original_build = base.App._build
_original_preview_refresh_tick = base.App._preview_refresh_tick


def _latest_completed_run(self: base.App) -> Path | None:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for run_dir in root.glob("multiregion_*"):
        if run_dir.is_dir() and (run_dir / "report.json").is_file():
            candidates.append(run_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _run_recoverability_audit(self: base.App) -> None:
    existing = getattr(self, "_stage2_recoverability_process", None)
    if existing is not None and existing.poll() is None:
        messagebox.showinfo(
            "Stage 2 recoverability audit",
            "The CPU recoverability audit is already running.",
        )
        return

    script = self.repo / "tools/nsamdr/neural/audit_nsamdr_v16_stage2_recoverability.py"
    if not script.is_file():
        messagebox.showerror(
            "Stage 2 recoverability audit",
            f"Recoverability audit script is missing:\n{script}",
        )
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
        self._stage2_recoverability_process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        messagebox.showerror(
            "Stage 2 recoverability audit",
            f"Could not start recoverability audit:\n{exc}",
        )
        return
    self.stage2_recoverability_status_var.set("Recoverability: running (CPU)")


def _refresh_recoverability_status(self: base.App) -> None:
    if not hasattr(self, "stage2_recoverability_status_var"):
        return
    process = getattr(self, "_stage2_recoverability_process", None)
    if process is not None:
        code = process.poll()
        if code is None:
            self.stage2_recoverability_status_var.set("Recoverability: running (CPU)")
            return
        self._stage2_recoverability_process = None
        self.stage2_recoverability_status_var.set(
            "Recoverability: complete"
            if code == 0
            else f"Recoverability: failed ({code})"
        )
        return

    run_dir = _latest_completed_run(self)
    if run_dir is not None and (run_dir / "recoverability_audit.json").is_file():
        stamp = run_dir.name.removeprefix("multiregion_")
        self.stage2_recoverability_status_var.set(f"Recoverability: available ({stamp})")
    else:
        self.stage2_recoverability_status_var.set("Recoverability: ready")


def _build(self: base.App) -> None:
    _original_build(self)
    self._stage2_recoverability_process = None
    self.stage2_recoverability_status_var = tk.StringVar(value="Recoverability: ready")
    ttk.Button(
        self.footer,
        text="Run Recoverability Audit",
        command=lambda: _run_recoverability_audit(self),
    ).pack(side="left", padx=(8, 4))
    ttk.Label(
        self.footer,
        textvariable=self.stage2_recoverability_status_var,
    ).pack(side="left", padx=(0, 8))
    _refresh_recoverability_status(self)


def _preview_refresh_tick(self: base.App) -> None:
    try:
        _refresh_recoverability_status(self)
    except (OSError, ValueError, TypeError, tk.TclError):
        pass
    _original_preview_refresh_tick(self)


base.App._build = _build
base.App._preview_refresh_tick = _preview_refresh_tick


if __name__ == "__main__":
    raise SystemExit(base.main())
