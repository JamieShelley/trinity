#!/usr/bin/env python3
"""Canonical NSAMDR V17.6 operator GUI: train, verify and render Raven.

Only the active V17.6 dense-specialist research workflow is exposed. Historical
proof implementations are retained as authenticated runtime dependencies, not
as selectable legacy training engines. Quadtree is not yet implemented.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "tools/nsamdr/nsamdr_cli.py"
CAPACITY_ROOT = ROOT / "artifacts/nsamdr/diagnostics/v176_stage10a1_capacity"
DETAIL_ROOT = ROOT / "artifacts/nsamdr/diagnostics/v176_stage10a_detail"
PREVIEW_ROOT = ROOT / "artifacts/nsamdr/render_preview"
PHASES = ("diagnose", "stability", "capacity", "compare-features", "attainable")


def latest_report(folder: Path, phase: str = "") -> Path | None:
    options = []
    for path in folder.glob("proof_*/report.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not phase or obj.get("phase") == phase:
                options.append(path)
        except (OSError, ValueError):
            continue
    return max(options, key=lambda p: p.stat().st_mtime) if options else None


def training_command(phase: str, resume: str, audit: bool, amp: str) -> list[str]:
    if phase not in PHASES or resume not in ("fresh", "auto") or amp not in ("auto", "bf16", "fp16"):
        raise ValueError("invalid V17.6 training selection")
    return ["v176-capacity-proof", "--phase", phase, "--resume", resume,
            "--amp-precision", amp] + (["--audit-only"] if audit else [])


def raven_command(cache: str, size: int, authority: int, mode: str) -> list[str]:
    if authority not in (1, 10) or mode not in ("output10", "hr_features58"):
        raise ValueError("invalid authenticated V17.6 training arm")
    if not 512 <= size <= 2048:
        raise ValueError("native Raven preview size must be 512..2048")
    return ["v176-raven-preview", "--shared-cache", cache, "--target-size", str(size),
            "--authority-index", str(authority), "--mode", mode]


class Operator:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("NSAMDR V17.6 — Training and Raven Preview")
        root.geometry("1080x750")
        root.minsize(880, 570)
        self.phase = tk.StringVar(value="compare-features")
        self.resume = tk.StringVar(value="fresh")
        self.audit = tk.BooleanVar(value=False)
        self.precision = tk.StringVar(value="auto")
        self.cache = tk.StringVar(value=r"C:\CCP\EVE")
        self.size = tk.StringVar(value="1024")
        self.authority = tk.StringVar(value="10")
        self.mode = tk.StringVar(value="output10")
        self.status = tk.StringVar(value="Ready — V17.6 dense specialist, research only")
        self.command = tk.StringVar()
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.controls: list[ttk.Button] = []

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="NSAMDR V17.6", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Active architecture only · frozen V17.2 backbone · train-only specialist · quadtree not implemented").pack(anchor="w", pady=(2, 10))
        notebook = ttk.Notebook(frame)
        notebook.pack(fill="x")

        train = ttk.Frame(notebook, padding=12)
        preview = ttk.Frame(notebook, padding=12)
        notebook.add(train, text="V17.6 training")
        notebook.add(preview, text="Raven visual preview")

        self._field(train, 0, "Phase", self.phase, PHASES)
        self._field(train, 1, "Resume", self.resume, ("fresh", "auto"))
        self._field(train, 2, "AMP", self.precision, ("auto", "bf16", "fp16"))
        ttk.Checkbutton(train, text="Audit only (no training)", variable=self.audit,
                        command=self._refresh_command).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(train, text="Train the actual V17.6 specialist with 'compare-features'. 'attainable' does NOT update weights.").grid(
            row=4, column=0, columnspan=4, sticky="w", pady=5)
        self._run_button(train, "Run selected phase", self.start_training, 5, 0)
        self._run_button(train, "Open latest report", self.open_report, 5, 1)

        self._field(preview, 0, "Authority", self.authority, ("10", "1"))
        self._field(preview, 1, "Specialist", self.mode, ("output10", "hr_features58"))
        self._field(preview, 2, "Target (px)", self.size, ("512", "1024", "2048"))
        ttk.Label(preview, text="EVE ResFiles root").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(preview, textvariable=self.cache, width=55).grid(row=3, column=1, sticky="ew")
        ttk.Label(preview, text="Uses the latest authenticated V17.6 comparison checkpoint, NOT V16 or a target-assisted fit.").grid(
            row=4, column=0, columnspan=4, sticky="w", pady=5)
        self._run_button(preview, "Render Raven (DX11)", self.start_preview, 5, 0)
        self._run_button(preview, "Open preview output", self.open_preview, 5, 1)
        preview.columnconfigure(1, weight=1)

        ttk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(12, 3))
        ttk.Label(frame, textvariable=self.command, wraplength=1000, foreground="#64748b").pack(anchor="w")
        output = ttk.Frame(frame)
        output.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(output, wrap="word", font=("Consolas", 9))
        bar = ttk.Scrollbar(output, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self._refresh_command()
        for state in (self.phase, self.resume, self.precision, self.audit, self.mode, self.size, self.authority):
            state.trace_add("write", lambda *_: self._refresh_command())
        self.root.after(125, self._poll)

    def _field(self, parent, row, label, var, options):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Combobox(parent, textvariable=var, values=options, width=24, state="readonly").grid(
            row=row, column=1, sticky="w", pady=5)

    def _run_button(self, parent, label, callback, row, column):
        b = ttk.Button(parent, text=label, command=callback)
        b.grid(row=row, column=column, sticky="w", padx=(0, 8), pady=(7, 3))
        if label.startswith(("Run", "Render")):
            self.controls.append(b)

    def _refresh_command(self):
        try:
            self.command.set("CLI: scripts\\build\\nsamdr.bat " +
                             " ".join(training_command(self.phase.get(), self.resume.get(),
                                                       self.audit.get(), self.precision.get())))
        except ValueError:
            self.command.set("Invalid phase configuration")

    def _start(self, args: list[str], activity: str):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("NSAMDR", "An operator process is already running.")
            return
        argv = [sys.executable, "-u", str(CLI), *args]
        self.log.insert("end", "\n> " + subprocess.list2cmdline(argv) + "\n")
        self.log.see("end")
        try:
            self.process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            bufsize=1, errors="replace")
        except OSError as exc:
            messagebox.showerror("NSAMDR", str(exc))
            return
        for b in self.controls:
            b.configure(state="disabled")
        self.status.set(activity + " — running")
        process = self.process
        def reader():
            assert process.stdout is not None
            for line in process.stdout:
                self.messages.put(("line", line))
            self.messages.put(("done", (activity, process.wait())))
        threading.Thread(target=reader, daemon=True).start()

    def start_training(self):
        self._start(training_command(self.phase.get(), self.resume.get(),
                                     self.audit.get(), self.precision.get()), "V17.6 " + self.phase.get())

    def start_preview(self):
        try:
            argv = raven_command(self.cache.get(), int(self.size.get()),
                                 int(self.authority.get()), self.mode.get())
            self._start(argv, "Raven DX11 preview")
        except ValueError as exc:
            messagebox.showerror("NSAMDR", str(exc))

    def open_report(self):
        path = latest_report(CAPACITY_ROOT, self.phase.get())
        if path is None:
            path = latest_report(DETAIL_ROOT)
        if path is None:
            messagebox.showinfo("NSAMDR", "No completed V17.6 report found.")
        else:
            os.startfile(str(path))

    def open_preview(self):
        label = f"V176_RAVEN_TRAIN{int(self.authority.get()):02d}_{self.mode.get().upper()}"
        folder = PREVIEW_ROOT / label
        if not folder.is_dir():
            messagebox.showinfo("NSAMDR", "No V17.6 Raven render output yet.")
        else:
            os.startfile(str(folder))

    def _poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "line":
                    self.log.insert("end", str(value))
                    self.log.see("end")
                elif kind == "done":
                    activity, code = value
                    self.status.set(f"{activity}: {'completed' if code == 0 else 'failed'} (exit {code})")
                    for b in self.controls:
                        b.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(125, self._poll)


def main() -> None:
    root = tk.Tk()
    Operator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
