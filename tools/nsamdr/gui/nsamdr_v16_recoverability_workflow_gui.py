#!/usr/bin/env python3
"""V16.1 monitored GUI plus CPU-only Stage 2 recoverability audit."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import nsamdr_v16_lowimpact_monitored_workflow_gui as monitored

base = monitored.base
_original_build = base.App._build
_original_preview_refresh_tick = base.App._preview_refresh_tick
_original_select_multiregion = monitored.multifamily.v16.legacy._select_multiregion


def _latest_completed_run(self: base.App) -> Path | None:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for run_dir in root.glob("multiregion_*"):
        if run_dir.is_dir() and (run_dir / "report.json").is_file():
            candidates.append(run_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _open_path(path: Path) -> None:
    target = path.resolve()
    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        messagebox.showerror("Stage 2 recoverability audit", f"Could not open:\n{target}\n\n{exc}")


def _recoverability_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "recoverability_audit_summary.txt",
        run_dir / "recoverability_audit.json",
        run_dir / "recoverability_audit_console.log",
    )


def _result_message(run_dir: Path, code: int) -> str:
    summary_path, json_path, log_path = _recoverability_paths(run_dir)
    lines = [
        f"Recoverability audit {'completed' if code == 0 else 'failed'} (exit {code}).",
        "",
        f"Run: {run_dir}",
        f"Summary: {summary_path}",
        f"JSON: {json_path}",
        f"Console log: {log_path}",
    ]
    if summary_path.is_file():
        try:
            summary = summary_path.read_text(encoding="utf-8", errors="replace")
            marker = "Decision\n"
            index = summary.find(marker)
            if index >= 0:
                tail = summary[index + len(marker):].splitlines()
                decision = next(
                    (line.strip() for line in tail if line.strip() and set(line.strip()) != {"-"}),
                    "",
                )
                if decision:
                    lines.extend(("", f"Decision: {decision}"))
        except OSError:
            pass
    return "\n".join(lines)


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

    run_dir = _latest_completed_run(self)
    if run_dir is None:
        messagebox.showerror(
            "Stage 2 recoverability audit",
            "No completed Stage 2 run with report.json was found.",
        )
        return

    summary_path, json_path, log_path = _recoverability_paths(run_dir)
    try:
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    except OSError as exc:
        messagebox.showerror(
            "Stage 2 recoverability audit",
            f"Could not create audit log:\n{log_path}\n\n{exc}",
        )
        return

    command = [
        sys.executable,
        "-u",
        str(script),
        "--repo-root",
        str(self.repo),
        "--run-dir",
        str(run_dir),
    ]
    kwargs: dict[str, object] = {
        "cwd": str(self.repo),
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
    }
    # GUI-launched diagnostics no longer depend on a transient console window.
    # Every line is persisted in recoverability_audit_console.log instead.
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        log_handle.close()
        messagebox.showerror(
            "Stage 2 recoverability audit",
            f"Could not start recoverability audit:\n{exc}\n\nLog: {log_path}",
        )
        return

    self._stage2_recoverability_process = process
    self._stage2_recoverability_log_handle = log_handle
    self._stage2_recoverability_run_dir = run_dir
    self._stage2_recoverability_notified_pid = None
    if hasattr(self, "stage2_recoverability_status_var"):
        self.stage2_recoverability_status_var.set(
            f"Recoverability: running (CPU) -> {run_dir.name.removeprefix('multiregion_')}"
        )
    messagebox.showinfo(
        "Stage 2 recoverability audit",
        "Recoverability audit started.\n\n"
        f"Run: {run_dir}\n"
        f"Live/persistent log: {log_path}\n\n"
        "The GUI will show the result and output paths when it finishes.",
    )


def _open_recoverability_results(self: base.App) -> None:
    run_dir = getattr(self, "_stage2_recoverability_run_dir", None) or _latest_completed_run(self)
    if run_dir is None:
        messagebox.showinfo("Stage 2 recoverability audit", "No completed Stage 2 run was found.")
        return
    summary_path, json_path, log_path = _recoverability_paths(Path(run_dir))
    for candidate in (summary_path, json_path, log_path, Path(run_dir)):
        if candidate.exists():
            _open_path(candidate)
            return


def _refresh_recoverability_status(self: base.App) -> None:
    if not hasattr(self, "stage2_recoverability_status_var"):
        return
    process = getattr(self, "_stage2_recoverability_process", None)
    if process is not None:
        code = process.poll()
        if code is None:
            self.stage2_recoverability_status_var.set("Recoverability: running (CPU)")
            return

        run_dir = Path(getattr(self, "_stage2_recoverability_run_dir", _latest_completed_run(self) or self.repo))
        log_handle = getattr(self, "_stage2_recoverability_log_handle", None)
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except OSError:
                pass
            self._stage2_recoverability_log_handle = None

        self._stage2_recoverability_process = None
        if code == 0:
            summary_path, json_path, _log_path = _recoverability_paths(run_dir)
            complete = summary_path.is_file() and json_path.is_file()
            self.stage2_recoverability_status_var.set(
                "Recoverability: complete" if complete else "Recoverability: incomplete output"
            )
        else:
            self.stage2_recoverability_status_var.set(f"Recoverability: failed ({code})")

        if getattr(self, "_stage2_recoverability_notified_pid", None) != process.pid:
            self._stage2_recoverability_notified_pid = process.pid
            message = _result_message(run_dir, int(code))
            if code == 0:
                messagebox.showinfo("Stage 2 recoverability audit", message)
            else:
                messagebox.showerror("Stage 2 recoverability audit", message)
        return

    run_dir = _latest_completed_run(self)
    if run_dir is not None and (run_dir / "recoverability_audit.json").is_file():
        stamp = run_dir.name.removeprefix("multiregion_")
        self.stage2_recoverability_status_var.set(f"Recoverability: available ({stamp})")
        self._stage2_recoverability_run_dir = run_dir
    else:
        self.stage2_recoverability_status_var.set("Recoverability: ready")


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    _original_select_multiregion(self, stage, status)
    self._label_row(
        "Recoverability audit",
        "CPU-only check of whether authored residual detail is statistically recoverable from the 4x LR evidence before another GPU run.",
    )
    action_row = ttk.Frame(self.form)
    action_row.pack(fill="x", pady=(8, 4))
    ttk.Label(action_row, text="CPU diagnostic", width=27).pack(side="left")
    ttk.Button(
        action_row,
        text="Run Recoverability Audit",
        command=lambda: _run_recoverability_audit(self),
    ).pack(side="left")
    ttk.Button(
        action_row,
        text="Open Results",
        command=lambda: _open_recoverability_results(self),
    ).pack(side="left", padx=(6, 0))
    if hasattr(self, "stage2_recoverability_status_var"):
        ttk.Label(
            action_row,
            textvariable=self.stage2_recoverability_status_var,
        ).pack(side="left", padx=(8, 0))
    run_dir = _latest_completed_run(self)
    if run_dir is not None:
        _summary, _json, log = _recoverability_paths(run_dir)
        self._label_row("Recoverability output", str(run_dir))
        self._label_row("Recoverability log", str(log))


def _build(self: base.App) -> None:
    _original_build(self)
    self._stage2_recoverability_process = None
    self._stage2_recoverability_log_handle = None
    self._stage2_recoverability_run_dir = None
    self._stage2_recoverability_notified_pid = None
    self.stage2_recoverability_status_var = tk.StringVar(value="Recoverability: ready")
    # Keep the footer shortcut for wide windows, but the Stage 2 form also contains
    # the action and a result opener so output cannot be lost with a closed console.
    ttk.Button(
        self.footer,
        text="Run Recoverability Audit",
        command=lambda: _run_recoverability_audit(self),
    ).pack(side="left", padx=(8, 4))
    ttk.Button(
        self.footer,
        text="Open Recoverability Results",
        command=lambda: _open_recoverability_results(self),
    ).pack(side="left", padx=(0, 4))
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
monitored.multifamily.v16.legacy._select_multiregion = _select_multiregion


if __name__ == "__main__":
    raise SystemExit(base.main())
