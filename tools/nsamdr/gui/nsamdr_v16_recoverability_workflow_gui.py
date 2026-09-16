#!/usr/bin/env python3
"""V16.1 monitored GUI plus CPU-only Stage 2 recovery diagnostics."""
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

_DIAGNOSTICS = {
    "recoverability": {
        "title": "Stage 2 recoverability audit",
        "script": "audit_nsamdr_v16_stage2_recoverability.py",
        "summary": "recoverability_audit_summary.txt",
        "json": "recoverability_audit.json",
        "log": "recoverability_audit_console.log",
        "statusPrefix": "Recoverability",
    },
    "recovery_sweep": {
        "title": "Stage 2 recovery evidence sweep",
        "script": "audit_nsamdr_v16_stage2_recovery_sweep.py",
        "summary": "recovery_sweep_summary.txt",
        "json": "recovery_sweep.json",
        "log": "recovery_sweep_console.log",
        "statusPrefix": "Recovery sweep",
    },
}


def _latest_completed_run(self: base.App) -> Path | None:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in root.glob("multiregion_*")
        if path.is_dir() and (path / "report.json").is_file()
    ]
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
        messagebox.showerror("NSAMDR Stage 2 diagnostic", f"Could not open:\n{target}\n\n{exc}")


def _paths(run_dir: Path, kind: str) -> tuple[Path, Path, Path]:
    spec = _DIAGNOSTICS[kind]
    return (
        run_dir / str(spec["summary"]),
        run_dir / str(spec["json"]),
        run_dir / str(spec["log"]),
    )


def _decision_from_summary(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for marker in ("DECISION\n", "Decision\n"):
        index = text.find(marker)
        if index < 0:
            continue
        for line in text[index + len(marker) :].splitlines():
            value = line.strip()
            if not value or set(value) == {"-"} or value in {"Signals:", "Recommended development directions:"}:
                continue
            return value.removeprefix("- ")
    return ""


def _result_message(run_dir: Path, kind: str, code: int) -> str:
    spec = _DIAGNOSTICS[kind]
    summary_path, json_path, log_path = _paths(run_dir, kind)
    lines = [
        f"{spec['statusPrefix']} {'completed' if code == 0 else 'failed'} (exit {code}).",
        "",
        f"Run: {run_dir}",
        f"Summary: {summary_path}",
        f"JSON: {json_path}",
        f"Console log: {log_path}",
    ]
    decision = _decision_from_summary(summary_path)
    if decision:
        lines.extend(("", f"Decision: {decision}"))
    return "\n".join(lines)


def _run_cpu_diagnostic(self: base.App, kind: str) -> None:
    spec = _DIAGNOSTICS[kind]
    process_attr = f"_stage2_{kind}_process"
    existing = getattr(self, process_attr, None)
    if existing is not None and existing.poll() is None:
        messagebox.showinfo(str(spec["title"]), f"{spec['statusPrefix']} is already running.")
        return

    script = self.repo / "tools/nsamdr/neural" / str(spec["script"])
    if not script.is_file():
        messagebox.showerror(str(spec["title"]), f"Diagnostic script is missing:\n{script}")
        return
    run_dir = _latest_completed_run(self)
    if run_dir is None:
        messagebox.showerror(str(spec["title"]), "No completed Stage 2 run with report.json was found.")
        return

    _summary_path, _json_path, log_path = _paths(run_dir, kind)
    try:
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    except OSError as exc:
        messagebox.showerror(str(spec["title"]), f"Could not create diagnostic log:\n{log_path}\n\n{exc}")
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
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        log_handle.close()
        messagebox.showerror(str(spec["title"]), f"Could not start diagnostic:\n{exc}\n\nLog: {log_path}")
        return

    setattr(self, process_attr, process)
    setattr(self, f"_stage2_{kind}_log_handle", log_handle)
    setattr(self, f"_stage2_{kind}_run_dir", run_dir)
    setattr(self, f"_stage2_{kind}_notified_pid", None)
    status_var = getattr(self, f"stage2_{kind}_status_var", None)
    if status_var is not None:
        status_var.set(f"{spec['statusPrefix']}: running (CPU)")
    messagebox.showinfo(
        str(spec["title"]),
        f"{spec['statusPrefix']} started.\n\nRun: {run_dir}\nPersistent log: {log_path}\n\n"
        "The GUI will report the output paths when it finishes.",
    )


def _open_results(self: base.App, kind: str) -> None:
    spec = _DIAGNOSTICS[kind]
    run_dir = getattr(self, f"_stage2_{kind}_run_dir", None) or _latest_completed_run(self)
    if run_dir is None:
        messagebox.showinfo(str(spec["title"]), "No completed Stage 2 run was found.")
        return
    summary_path, json_path, log_path = _paths(Path(run_dir), kind)
    for candidate in (summary_path, json_path, log_path, Path(run_dir)):
        if candidate.exists():
            _open_path(candidate)
            return


def _refresh_status(self: base.App, kind: str) -> None:
    spec = _DIAGNOSTICS[kind]
    status_var = getattr(self, f"stage2_{kind}_status_var", None)
    if status_var is None:
        return
    process = getattr(self, f"_stage2_{kind}_process", None)
    if process is not None:
        code = process.poll()
        if code is None:
            status_var.set(f"{spec['statusPrefix']}: running (CPU)")
            return

        run_dir = Path(getattr(self, f"_stage2_{kind}_run_dir", _latest_completed_run(self) or self.repo))
        log_handle = getattr(self, f"_stage2_{kind}_log_handle", None)
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except OSError:
                pass
            setattr(self, f"_stage2_{kind}_log_handle", None)
        setattr(self, f"_stage2_{kind}_process", None)

        summary_path, json_path, _log_path = _paths(run_dir, kind)
        if code == 0 and summary_path.is_file() and json_path.is_file():
            status_var.set(f"{spec['statusPrefix']}: complete")
        elif code == 0:
            status_var.set(f"{spec['statusPrefix']}: incomplete output")
        else:
            status_var.set(f"{spec['statusPrefix']}: failed ({code})")

        if getattr(self, f"_stage2_{kind}_notified_pid", None) != process.pid:
            setattr(self, f"_stage2_{kind}_notified_pid", process.pid)
            message = _result_message(run_dir, kind, int(code))
            if code == 0:
                messagebox.showinfo(str(spec["title"]), message)
            else:
                messagebox.showerror(str(spec["title"]), message)
        return

    run_dir = _latest_completed_run(self)
    if run_dir is not None and (run_dir / str(spec["json"])).is_file():
        status_var.set(
            f"{spec['statusPrefix']}: available ({run_dir.name.removeprefix('multiregion_')})"
        )
        setattr(self, f"_stage2_{kind}_run_dir", run_dir)
    else:
        status_var.set(f"{spec['statusPrefix']}: ready")


def _diagnostic_row(self: base.App, *, label: str, kind: str, run_text: str, open_text: str) -> None:
    row = ttk.Frame(self.form)
    row.pack(fill="x", pady=(6, 2))
    ttk.Label(row, text=label, width=27).pack(side="left")
    ttk.Button(row, text=run_text, command=lambda: _run_cpu_diagnostic(self, kind)).pack(side="left")
    ttk.Button(row, text=open_text, command=lambda: _open_results(self, kind)).pack(side="left", padx=(6, 0))
    status_var = getattr(self, f"stage2_{kind}_status_var", None)
    if status_var is not None:
        ttk.Label(row, textvariable=status_var).pack(side="left", padx=(8, 0))


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    _original_select_multiregion(self, stage, status)
    self._label_row(
        "Recovery diagnostics",
        "CPU-only diagnostics test 4x information recoverability before another GPU experiment.",
    )
    _diagnostic_row(
        self,
        label="CPU recoverability",
        kind="recoverability",
        run_text="Run Recoverability Audit",
        open_text="Open Audit Results",
    )
    _diagnostic_row(
        self,
        label="CPU evidence sweep",
        kind="recovery_sweep",
        run_text="Run Scale / Context / Prior Sweep",
        open_text="Open Sweep Results",
    )
    self._label_row(
        "Evidence sweep",
        "Same held-out records; tests 2x/~3x/4x evidence, 5/9/17/33 LR context, and wider prepared-train priors.",
    )
    self._label_row(
        "Goal alignment",
        "README goal remains genuine 4x aligned albedo/normal/material recovery, no unsupported invention, unchanged gates.",
    )
    run_dir = _latest_completed_run(self)
    if run_dir is not None:
        _r_summary, _r_json, recover_log = _paths(run_dir, "recoverability")
        _s_summary, _s_json, sweep_log = _paths(run_dir, "recovery_sweep")
        self._label_row("Diagnostic run", str(run_dir))
        self._label_row("Recoverability log", str(recover_log))
        self._label_row("Evidence sweep log", str(sweep_log))


def _build(self: base.App) -> None:
    _original_build(self)
    for kind, spec in _DIAGNOSTICS.items():
        setattr(self, f"_stage2_{kind}_process", None)
        setattr(self, f"_stage2_{kind}_log_handle", None)
        setattr(self, f"_stage2_{kind}_run_dir", None)
        setattr(self, f"_stage2_{kind}_notified_pid", None)
        setattr(
            self,
            f"stage2_{kind}_status_var",
            tk.StringVar(value=f"{spec['statusPrefix']}: ready"),
        )

    # Keep the original recoverability shortcut in the footer.  The larger sweep
    # stays in the scrollable Stage 2 form so footer crowding cannot hide it.
    ttk.Button(
        self.footer,
        text="Run Recoverability Audit",
        command=lambda: _run_cpu_diagnostic(self, "recoverability"),
    ).pack(side="left", padx=(8, 4))
    ttk.Button(
        self.footer,
        text="Open Recoverability Results",
        command=lambda: _open_results(self, "recoverability"),
    ).pack(side="left", padx=(0, 4))
    ttk.Label(
        self.footer,
        textvariable=self.stage2_recoverability_status_var,
    ).pack(side="left", padx=(0, 8))
    for kind in _DIAGNOSTICS:
        _refresh_status(self, kind)


def _preview_refresh_tick(self: base.App) -> None:
    try:
        for kind in _DIAGNOSTICS:
            _refresh_status(self, kind)
    except (OSError, ValueError, TypeError, tk.TclError):
        pass
    _original_preview_refresh_tick(self)


base.App._build = _build
base.App._preview_refresh_tick = _preview_refresh_tick
monitored.multifamily.v16.legacy._select_multiregion = _select_multiregion


if __name__ == "__main__":
    raise SystemExit(base.main())
