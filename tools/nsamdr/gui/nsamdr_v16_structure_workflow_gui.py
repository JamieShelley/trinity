#!/usr/bin/env python3
"""Canonical NSAMDR V16.2 workflow GUI.

This is intentionally standalone. Historical GUI inheritance layers were removed;
only the current Stage 2 runtime, structural evidence tools and authored-corpus
census are exposed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from typing import IO


REPO_ROOT = Path(__file__).resolve().parents[3]
NEURAL_ROOT = REPO_ROOT / "tools/nsamdr/neural"
DIAGNOSTIC_ROOT = REPO_ROOT / "artifacts/nsamdr/diagnostics/v16_mini"
CENSUS_ROOT = REPO_ROOT / "artifacts/nsamdr/diagnostics/eve_census"
STAGE2_RUNTIME = (
    NEURAL_ROOT
    / "v14/safe_live_resume_monitored_fourfamily_multiregion_diagnostic.py"
)

LOW_IMPACT_ARGS = (
    "--safe-mode",
    "--cooldown-every", "4",
    "--cooldown-seconds", "1.5",
    "--max-gpu-temperature", "75",
    "--temperature-resume-margin", "7",
    "--thermal-poll-seconds", "3",
    "--validation-cooldown-seconds", "5",
)

DIAGNOSTICS = {
    "structure": (
        "audit_nsamdr_v16_structure_support.py",
        "structure_support_audit_summary.txt",
        "structure_support_audit.json",
        "structure_support_audit_console.log",
    ),
    "profiles": (
        "audit_nsamdr_v16_boundary_profiles.py",
        "boundary_profile_audit_summary.txt",
        "boundary_profile_audit.json",
        "boundary_profile_audit_console.log",
    ),
}


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


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
        messagebox.showerror("NSAMDR", f"Could not open:\n{target}\n\n{exc}")


def _latest_completed_stage2() -> Path | None:
    if not DIAGNOSTIC_ROOT.is_dir():
        return None
    candidates: list[Path] = []
    for run_dir in DIAGNOSTIC_ROOT.glob("multiregion_*"):
        report = _read_json(run_dir / "report.json")
        if report is None:
            continue
        if int(report.get("completedSteps") or 0) > 0:
            candidates.append(run_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _latest_resume_checkpoint() -> Path | None:
    pointer = _read_json(DIAGNOSTIC_ROOT / "stage2_resume_pointer.json") or {}
    raw = str(pointer.get("checkpoint") or pointer.get("checkpointPath") or "")
    if raw:
        value = Path(raw)
        if not value.is_absolute():
            value = REPO_ROOT / value
        if value.is_file():
            return value.resolve()
    if DIAGNOSTIC_ROOT.is_dir():
        candidates = [
            path / "resume_checkpoint.pt"
            for path in DIAGNOSTIC_ROOT.glob("multiregion_*")
            if (path / "resume_checkpoint.pt").is_file()
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime).resolve()
    return None


def _latest_census() -> Path | None:
    latest = CENSUS_ROOT / "LATEST.txt"
    if latest.is_file():
        try:
            value = Path(latest.read_text(encoding="utf-8").strip())
            if not value.is_absolute():
                value = REPO_ROOT / value
            if value.is_dir():
                return value.resolve()
        except OSError:
            pass
    if not CENSUS_ROOT.is_dir():
        return None
    runs = [path for path in CENSUS_ROOT.glob("census_*") if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NSAMDR V16.2 Advanced Diagnostics")
        self.root.geometry("1040x620")
        self.processes: dict[str, subprocess.Popen[object]] = {}
        self.log_handles: dict[str, IO[str]] = {}
        self.status = tk.StringVar(value="Ready")
        self.stage2_status = tk.StringVar(value="Stage 2: checking")
        self.shared_cache = tk.StringVar(value=r"C:\CCP\EVE")
        self.max_steps = tk.StringVar(value="5120")
        self.validate_every = tk.StringVar(value="256")
        self._build()
        self._refresh()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="NSAMDR V16.2 Advanced Diagnostics",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Research/diagnostic tools. Use the main NSAMDR Workflow GUI for the "
                "operator training and preview pipeline."
            ),
        ).pack(anchor="w", pady=(2, 12))

        evidence = ttk.LabelFrame(outer, text="CPU structural evidence", padding=10)
        evidence.pack(fill="x", pady=(0, 10))
        self._button_row(
            evidence,
            "Structure support",
            "Run Structure Support Audit",
            lambda: self._run_diagnostic("structure"),
            "Open Results",
            lambda: self._open_diagnostic("structure"),
        )
        self._button_row(
            evidence,
            "Boundary profiles",
            "Run Boundary Profile Audit",
            lambda: self._run_diagnostic("profiles"),
            "Open Results",
            lambda: self._open_diagnostic("profiles"),
        )
        ttk.Label(
            evidence,
            text=(
                "Both audits are CPU-only and use the latest completed Stage 2 report. "
                "They do not mutate training data or model state."
            ),
        ).pack(anchor="w", pady=(7, 0))

        census = ttk.LabelFrame(outer, text="Authored EVE corpus", padding=10)
        census.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(census)
        row.pack(fill="x")
        ttk.Button(row, text="Scan Full EVE Corpus", command=self._run_census).pack(side="left")
        ttk.Button(row, text="Open Census Results", command=self._open_census).pack(side="left", padx=(6, 0))
        ttk.Label(row, text="Index + DDS-header scan only; no CUDA training.").pack(side="left", padx=(12, 0))

        stage2 = ttk.LabelFrame(outer, text="V16 Stage 2 runtime", padding=10)
        stage2.pack(fill="x", pady=(0, 10))
        fields = ttk.Frame(stage2)
        fields.pack(fill="x", pady=(0, 8))
        ttk.Label(fields, text="Shared cache").grid(row=0, column=0, sticky="w")
        ttk.Entry(fields, textvariable=self.shared_cache, width=34).grid(row=0, column=1, sticky="w", padx=(6, 18))
        ttk.Label(fields, text="Max steps").grid(row=0, column=2, sticky="w")
        ttk.Entry(fields, textvariable=self.max_steps, width=8).grid(row=0, column=3, sticky="w", padx=(6, 18))
        ttk.Label(fields, text="Validate every").grid(row=0, column=4, sticky="w")
        ttk.Entry(fields, textvariable=self.validate_every, width=8).grid(row=0, column=5, sticky="w", padx=(6, 0))

        actions = ttk.Frame(stage2)
        actions.pack(fill="x")
        ttk.Button(actions, text="Run / Auto-Resume Stage 2", command=self._run_stage2).pack(side="left")
        ttk.Button(actions, text="Resume Latest Checkpoint", command=lambda: self._run_stage2(explicit_resume=True)).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Open Latest Probe", command=self._open_probe).pack(side="left", padx=(12, 0))
        ttk.Button(actions, text="Open Stage 2 Summary", command=self._open_stage2_summary).pack(side="left", padx=(6, 0))
        ttk.Label(stage2, textvariable=self.stage2_status).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            stage2,
            text=(
                "Low-impact profile: 8 train regions, 4 held-out regions, 16/4 prepared caps, "
                "0.0002 LR, 75 C thermal pause, durable resume checkpoints."
            ),
        ).pack(anchor="w", pady=(3, 0))

        footer = ttk.Frame(outer)
        footer.pack(fill="x", side="bottom")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        ttk.Button(footer, text="Open Diagnostics Folder", command=lambda: _open_path(DIAGNOSTIC_ROOT)).pack(side="right")

    def _button_row(
        self,
        parent: ttk.Widget,
        label: str,
        run_text: str,
        run_command: object,
        open_text: str,
        open_command: object,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Button(row, text=run_text, command=run_command).pack(side="left")
        ttk.Button(row, text=open_text, command=open_command).pack(side="left", padx=(6, 0))

    def _spawn(
        self,
        key: str,
        command: list[str],
        *,
        log_path: Path | None = None,
        new_console: bool = False,
    ) -> None:
        existing = self.processes.get(key)
        if existing is not None and existing.poll() is None:
            messagebox.showinfo("NSAMDR", f"{key} is already running.")
            return
        kwargs: dict[str, object] = {"cwd": str(REPO_ROOT)}
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8", buffering=1)
            self.log_handles[key] = handle
            kwargs.update(stdout=handle, stderr=subprocess.STDOUT)
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        elif new_console and sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        try:
            self.processes[key] = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            handle = self.log_handles.pop(key, None)
            if handle is not None:
                handle.close()
            messagebox.showerror("NSAMDR", f"Could not start {key}:\n{exc}")
            return
        self.status.set(f"Running: {key}")

    def _run_diagnostic(self, kind: str) -> None:
        run_dir = _latest_completed_stage2()
        if run_dir is None:
            messagebox.showerror("NSAMDR", "No completed Stage 2 run with report.json was found.")
            return
        script_name, _summary, _json_name, log_name = DIAGNOSTICS[kind]
        self._spawn(
            kind,
            [
                sys.executable,
                "-u",
                str(NEURAL_ROOT / script_name),
                "--repo-root",
                str(REPO_ROOT),
                "--run-dir",
                str(run_dir),
            ],
            log_path=run_dir / log_name,
        )

    def _open_diagnostic(self, kind: str) -> None:
        run_dir = _latest_completed_stage2()
        if run_dir is None:
            messagebox.showinfo("NSAMDR", "No completed Stage 2 run was found.")
            return
        _script, summary, json_name, log_name = DIAGNOSTICS[kind]
        for candidate in (run_dir / summary, run_dir / json_name, run_dir / log_name, run_dir):
            if candidate.exists():
                _open_path(candidate)
                return
        messagebox.showinfo("NSAMDR", f"No {kind} result exists yet.")

    def _run_census(self) -> None:
        CENSUS_ROOT.mkdir(parents=True, exist_ok=True)
        self._spawn(
            "eve-census",
            [
                sys.executable,
                "-u",
                str(NEURAL_ROOT / "scan_eve_authored_corpus.py"),
                "--repo-root",
                str(REPO_ROOT),
            ],
            log_path=CENSUS_ROOT / "eve_census_console.log",
        )

    def _open_census(self) -> None:
        run_dir = _latest_census()
        if run_dir is None:
            log = CENSUS_ROOT / "eve_census_console.log"
            if log.is_file():
                _open_path(log)
            else:
                messagebox.showinfo("NSAMDR", "No corpus census result exists yet.")
            return
        for candidate in (
            run_dir / "eve_authored_corpus_census_summary.txt",
            run_dir / "eve_authored_corpus_census.json",
            run_dir,
        ):
            if candidate.exists():
                _open_path(candidate)
                return

    def _stage2_command(self) -> list[str]:
        return [
            sys.executable,
            "-u",
            str(STAGE2_RUNTIME),
            "--repo-root", str(REPO_ROOT),
            "--shared-cache", self.shared_cache.get().strip() or r"C:\CCP\EVE",
            "--train-regions", "8",
            "--validation-regions", "4",
            "--prepare-train-regions", "16",
            "--prepare-validation-regions", "4",
            "--max-steps", self.max_steps.get().strip() or "5120",
            "--validate-every", self.validate_every.get().strip() or "256",
            "--learning-rate", "0.0002",
            "--required-edge-recovery", "0.60",
            "--required-global-recovery", "0.45",
            "--required-gradient-recovery", "0.35",
            "--device", "cuda",
            "--amp-precision", "auto",
            "--resume-stage2",
            *LOW_IMPACT_ARGS,
        ]

    def _run_stage2(self, explicit_resume: bool = False) -> None:
        command = self._stage2_command()
        if explicit_resume:
            checkpoint = _latest_resume_checkpoint()
            if checkpoint is None:
                messagebox.showerror("NSAMDR", "No retained Stage 2 resume checkpoint was found.")
                return
            command.extend(("--resume-checkpoint", str(checkpoint)))
        self._spawn("stage2", command, new_console=True)

    def _open_probe(self) -> None:
        subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(NEURAL_ROOT / "open_nsamdr_v16_stage2_probe.py"),
                "--repo-root",
                str(REPO_ROOT),
                "--open",
            ],
            cwd=REPO_ROOT,
        )

    def _open_stage2_summary(self) -> None:
        run_dir = _latest_completed_stage2()
        if run_dir is None:
            messagebox.showinfo("NSAMDR", "No completed Stage 2 run was found.")
            return
        summary = run_dir / "stage2_summary.txt"
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(NEURAL_ROOT / "summarize_nsamdr_v16_stage2_run.py"),
                "--repo-root",
                str(REPO_ROOT),
                "--run-dir",
                str(run_dir),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        if summary.exists():
            _open_path(summary)
        else:
            _open_path(run_dir)

    def _refresh_stage2(self) -> None:
        pointer = _read_json(DIAGNOSTIC_ROOT / "stage2_resume_pointer.json") or {}
        run_dir: Path | None = None
        raw_run = str(pointer.get("runDir") or "")
        if raw_run:
            candidate = Path(raw_run)
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if candidate.is_dir():
                run_dir = candidate
        if run_dir is None and DIAGNOSTIC_ROOT.is_dir():
            runs = [path for path in DIAGNOSTIC_ROOT.glob("multiregion_*") if path.is_dir()]
            if runs:
                run_dir = max(runs, key=lambda path: path.stat().st_mtime)
        heartbeat = _read_json(run_dir / "stage2_heartbeat.json") if run_dir else None
        if heartbeat is None:
            self.stage2_status.set("Stage 2: no durable run")
            return
        step = heartbeat.get("step", "?")
        maximum = heartbeat.get("maximumSteps", "?")
        state = heartbeat.get("status", "unknown")
        gpu = heartbeat.get("gpu") if isinstance(heartbeat.get("gpu"), dict) else {}
        suffix: list[str] = []
        if gpu and gpu.get("temperatureC") is not None:
            suffix.append(f"{float(gpu['temperatureC']):.0f} C")
        if gpu and gpu.get("utilizationAveragePercent") is not None:
            suffix.append(f"GPU {float(gpu['utilizationAveragePercent']):.0f}% avg")
        self.stage2_status.set(
            f"Stage 2: {step}/{maximum} {state}" + (" | " + " | ".join(suffix) if suffix else "")
        )

    def _refresh(self) -> None:
        completed: list[str] = []
        for key, process in list(self.processes.items()):
            code = process.poll()
            if code is None:
                continue
            handle = self.log_handles.pop(key, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            self.processes.pop(key, None)
            completed.append(f"{key}: exit {code}")
        if completed:
            self.status.set(" | ".join(completed))
        self._refresh_stage2()
        self.root.after(1000, self._refresh)


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
