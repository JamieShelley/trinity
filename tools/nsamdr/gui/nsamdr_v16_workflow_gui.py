#!/usr/bin/env python3
"""Canonical NSAMDR V16.2 operator workflow GUI.

This restores the operator-oriented workflow surface used before the temporary
structure-only diagnostic GUI became the default. The diagnostic GUI remains
available as an advanced tool; this file owns the normal setup -> development
training -> main training -> preview workflow.

Main V16 production training is intentionally locked until the current
broad-authority D4 recipe is promoted from diagnostics into the canonical
training workflow. The GUI must not silently route that button to an obsolete
V9/V14 full-training path.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "tools/nsamdr/nsamdr_cli.py"
ADVANCED_GUI = REPO_ROOT / "tools/nsamdr/gui/nsamdr_v16_structure_workflow_gui.py"
EXPERIMENT_ROOT = REPO_ROOT / "artifacts/nsamdr/experiments"
GUI_STATE_ROOT = REPO_ROOT / "artifacts/nsamdr/gui"
STATE_PATH = GUI_STATE_ROOT / "v16_operator_workflow_state.json"
DIAGNOSTIC_ROOT = REPO_ROOT / "artifacts/nsamdr/diagnostics"
APP_TITLE = "NSAMDR V16.2 Workflow"
STATE_SCHEMA = "nsamdr-v16-operator-workflow-v1"

EXPERIMENT_RE = re.compile(r"^EXP_\d{4,}$", re.I)
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+phase=([^\s]+)", re.I)
BATCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+total=", re.I)
PROGRESS_RE = re.compile(r"total=(\d+)/(\d+)")
STEP_MS_RE = re.compile(r"\bstep=([0-9]+(?:\.[0-9]+)?)ms\b", re.I)


@dataclass(frozen=True)
class Stage:
    id: str
    number: str
    label: str
    description: str


STAGES = (
    Stage(
        "setup",
        "P0",
        "Prepare / validate environment",
        "Create or verify the managed CUDA environment and active NSAMDR checkout.",
    ),
    Stage(
        "quick",
        "1",
        "Raven Quick",
        "Run the legacy deterministic Raven qualification baseline; useful for comparison, but it does not include the current D4/broad-authority research recipe.",
    ),
    Stage(
        "train",
        "2",
        "Main V16 Training",
        "Train the promoted broad-authority V16 production candidate.",
    ),
    Stage(
        "preview",
        "3",
        "Preview",
        "Generate and launch a preview from a completed qualified experiment.",
    ),
)
BY_ID = {stage.id: stage for stage in STAGES}


def _read_json(path: Path) -> dict[str, Any] | None:
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


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _qualified_final(path: Path) -> bool:
    """Return True only for a completed immutable production final."""
    final = _read_json(path / "final_manifest.json")
    experiment = _read_json(path / "experiment.json")
    if final is None or experiment is None:
        return False
    checkpoint = final.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return False
    raw_path = str(checkpoint.get("path") or "")
    if not raw_path:
        return False
    checkpoint_path = (path / raw_path).resolve()
    try:
        checkpoint_path.relative_to((path / "checkpoints/final").resolve())
    except ValueError:
        return False
    participation = _read_json(path / "architecture_participation.json")
    return bool(
        final.get("qualified") is True
        and final.get("status") == "completed"
        and final.get("selectionKind") == "production-final"
        and experiment.get("qualified") is True
        and experiment.get("status") == "completed"
        and checkpoint.get("immutable") is True
        and checkpoint.get("selectionKind") == "production-final"
        and re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256") or ""))
        and participation is not None
        and participation.get("pass") is True
        and checkpoint_path.is_file()
    )


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1320x820")
        self.root.minsize(980, 680)

        self.process: subprocess.Popen[str] | None = None
        self.process_thread: threading.Thread | None = None
        self.process_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.active_stage: str | None = None
        self.process_started_at: float | None = None
        self.current_epoch: tuple[int, int, str] | None = None

        self.state = self._load_state()
        self.vars: dict[str, tk.Variable] = {}
        self.description = tk.StringVar()
        self.command_preview = tk.StringVar()
        self.progress_text = tk.StringVar(value="Idle")
        self.scope_text = tk.StringVar(value="Qualified production final: checking")
        self.recipe_text = tk.StringVar(
            value=(
                "Main V16 training recipe selected: D4 augmentation + broader authority-balanced exposure. "
                "Production launch remains locked until this recipe is wired into the canonical workflow."
            )
        )
        self.preview_target = tk.StringVar(value="")

        self._build()
        self.detect(silent=True)
        self.root.after(100, self._poll)

    def _load_state(self) -> dict[str, Any]:
        default = {
            "schema": STATE_SCHEMA,
            "current": "quick",
            "status": {stage.id: "pending" for stage in STAGES},
        }
        loaded = _read_json(STATE_PATH)
        if loaded is not None and loaded.get("schema") == STATE_SCHEMA:
            default.update(loaded)
        statuses = default.setdefault("status", {})
        for stage in STAGES:
            if statuses.get(stage.id) in {"running", "stopping"}:
                statuses[stage.id] = "interrupted"
        return default

    def _save_state(self) -> None:
        GUI_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def _experiment_ids(self, *, qualified_only: bool = False) -> list[str]:
        if not EXPERIMENT_ROOT.is_dir():
            return []
        values: list[str] = []
        for path in EXPERIMENT_ROOT.iterdir():
            if not path.is_dir() or EXPERIMENT_RE.fullmatch(path.name) is None:
                continue
            if qualified_only and not _qualified_final(path):
                continue
            values.append(path.name.upper())
        return sorted(values, key=lambda value: int(value.split("_")[1]))

    def _latest_qualified(self) -> str | None:
        values = self._experiment_ids(qualified_only=True)
        return values[-1] if values else None

    def _latest_experiment(self, *, training_mode: str | None = None) -> str | None:
        values = self._experiment_ids()
        if training_mode is None:
            return values[-1] if values else None
        wanted = training_mode.lower()
        for experiment_id in reversed(values):
            payload = _read_json(EXPERIMENT_ROOT / experiment_id / "experiment.json") or {}
            if str(payload.get("trainingMode") or "").lower() == wanted:
                return experiment_id
        return None

    def _latest_training_preview(self) -> Path | None:
        experiment_id = self._latest_experiment(training_mode="quick")
        if experiment_id is None:
            return None
        candidate = EXPERIMENT_ROOT / experiment_id / "previews/live/latest_ABCF.png"
        return candidate if candidate.is_file() else None

    def _cli_argv(self, *arguments: str) -> list[str]:
        return [sys.executable, "-u", str(CLI), *arguments]

    def _command_for_stage(self, stage_id: str) -> list[str] | None:
        if stage_id == "setup":
            command = self._cli_argv("setup", "cuda")
            force = self.vars.get("force")
            if force is not None and bool(force.get()):
                command.append("--force")
            return command

        if stage_id == "quick":
            command = self._cli_argv("raven-quick")
            command += [
                "--shared-cache",
                self._value("cache", r"C:\CCP\EVE"),
                "--max-train-regions",
                self._value("train_crops", "16"),
                "--max-validation-regions",
                self._value("validation_crops", "4"),
                "--experiment",
                self._value("experiment", "new"),
                "--control",
                self._value("control", "auto"),
                "--preview-target-size",
                self._value("target", "4096"),
                "--preview-device",
                self._value("device", "cuda"),
                "--performance-profile",
                self._value("profile", "fast"),
                "--workers",
                self._value("workers", "4"),
                "--prefetch-factor",
                self._value("prefetch", "2"),
                "--amp-precision",
                self._value("amp", "auto"),
            ]
            rebuild = self.vars.get("rebuild")
            if rebuild is not None and bool(rebuild.get()):
                command.append("--rebuild-dataset")
            return command

        if stage_id == "train":
            return None

        if stage_id == "preview":
            experiment = self._value("experiment", "")
            if not experiment or experiment == "<none>":
                return None
            return self._cli_argv(
                "preview",
                experiment,
                "--shared-cache",
                self._value("cache", r"C:\CCP\EVE"),
                "--target-size",
                self._value("target", "4096"),
                "--device",
                self._value("device", "cuda"),
            )
        raise KeyError(stage_id)

    def _value(self, key: str, default: str = "") -> str:
        variable = self.vars.get(key)
        return str(variable.get()) if variable is not None else default

    def _stage_lock_reason(self, stage_id: str) -> str | None:
        if stage_id == "train":
            return (
                "The V16 broad-authority/D4 training recipe is still being qualified. "
                "No canonical production full-train command exists yet."
            )
        if stage_id == "preview" and not self._experiment_ids(qualified_only=True):
            return "No completed qualified EXP_#### final is available yet."
        return None

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        top = ttk.Frame(self.root, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text=APP_TITLE, font=("Segoe UI", 17, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(top, text=str(REPO_ROOT)).grid(row=0, column=1, sticky="e")

        scope = ttk.Frame(self.root, padding=(10, 0, 10, 4))
        scope.grid(row=1, column=0, sticky="ew")
        scope.columnconfigure(0, weight=1)
        ttk.Label(scope, textvariable=self.scope_text, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            scope,
            text="Advanced diagnostics",
            command=self._open_advanced_gui,
        ).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(
            scope,
            text="Open diagnostics folder",
            command=lambda: _open_path(DIAGNOSTIC_ROOT),
        ).grid(row=0, column=2, padx=(6, 0))

        ttk.Label(
            self.root,
            textvariable=self.recipe_text,
            padding=(10, 0, 10, 8),
            foreground="#555555",
        ).grid(row=2, column=0, sticky="ew")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=5)

        self.tree = ttk.Treeview(
            left,
            columns=("number", "stage", "status"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("number", text="#")
        self.tree.heading("stage", text="Stage")
        self.tree.heading("status", text="Status")
        self.tree.column("number", width=50, anchor="w")
        self.tree.column("stage", width=270, anchor="w")
        self.tree.column("status", width=105, anchor="w")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selected())

        for stage in STAGES:
            self.tree.insert(
                "",
                "end",
                iid=stage.id,
                values=(
                    stage.number,
                    stage.label,
                    self.state.get("status", {}).get(stage.id, "pending"),
                ),
            )

        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        ttk.Label(
            right,
            textvariable=self.description,
            font=("Segoe UI", 12, "bold"),
            wraplength=850,
        ).grid(row=0, column=0, sticky="ew", pady=(4, 6))

        controls = ttk.LabelFrame(right, text="Stage controls", padding=10)
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        self.form = ttk.Frame(controls)
        self.form.grid(row=0, column=0, sticky="ew")

        runtime = ttk.LabelFrame(right, text="Runtime", padding=10)
        runtime.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        runtime.columnconfigure(0, weight=1)
        runtime.rowconfigure(6, weight=1)

        ttk.Label(runtime, text="Command preview").grid(row=0, column=0, sticky="w")
        ttk.Entry(
            runtime,
            textvariable=self.command_preview,
            state="readonly",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 6))

        self.progress = ttk.Progressbar(runtime, maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew")
        ttk.Label(runtime, textvariable=self.progress_text).grid(
            row=3, column=0, sticky="w", pady=(3, 5)
        )
        ttk.Separator(runtime, orient="horizontal").grid(
            row=4, column=0, sticky="ew", pady=(1, 5)
        )
        ttk.Label(runtime, text="Runtime log", font=("Segoe UI", 9, "bold")).grid(
            row=5, column=0, sticky="w"
        )

        log_frame = ttk.Frame(runtime)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(3, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.output = tk.Text(log_frame, wrap="none", height=16)
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.output.yview)
        xscroll = ttk.Scrollbar(log_frame, orient="horizontal", command=self.output.xview)
        self.output.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        footer = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        footer.grid(row=4, column=0, sticky="ew")
        self.run_button = ttk.Button(footer, text="Run selected", command=self.run_selected)
        self.run_button.pack(side="left")
        ttk.Button(footer, text="Stop current process", command=self.stop).pack(
            side="left", padx=(5, 0)
        )
        ttk.Button(footer, text="Detect artifacts", command=lambda: self.detect()).pack(
            side="right"
        )
        ttk.Label(footer, text="Preview:").pack(side="left", padx=(14, 4))
        self.preview_combo = ttk.Combobox(
            footer,
            textvariable=self.preview_target,
            width=14,
            state="readonly",
            postcommand=self._refresh_preview_choices,
        )
        self.preview_combo.pack(side="left")
        ttk.Button(
            footer,
            text="Render selected preview",
            command=self.preview_selected,
        ).pack(side="left", padx=(5, 0))

        current = str(self.state.get("current") or "quick")
        if current not in BY_ID:
            current = "quick"
        self.tree.selection_set(current)
        self._selected()

    def _clear_form(self) -> None:
        for child in self.form.winfo_children():
            child.destroy()
        self.vars = {}

    def _row(
        self,
        label: str,
        key: str,
        default: str,
        choices: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=26).pack(side="left")
        variable = tk.StringVar(value=default)
        self.vars[key] = variable
        if choices is None:
            widget: tk.Widget = ttk.Entry(row, textvariable=variable)
        else:
            widget = ttk.Combobox(
                row,
                textvariable=variable,
                values=tuple(choices),
                state="readonly",
            )
        widget.pack(side="left", fill="x", expand=True)
        variable.trace_add("write", lambda *_args: self._update_command_preview())

    def _check(self, label: str, key: str, default: bool = False) -> None:
        variable = tk.BooleanVar(value=default)
        self.vars[key] = variable
        ttk.Checkbutton(
            self.form,
            text=label,
            variable=variable,
            command=self._update_command_preview,
        ).pack(anchor="w", pady=2)

    def _label(self, label: str, value: str) -> None:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=26).pack(side="left")
        ttk.Label(row, text=value, wraplength=700).pack(side="left", fill="x", expand=True)

    def _selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        stage_id = str(selection[0])
        stage = BY_ID[stage_id]
        self.state["current"] = stage_id
        self._save_state()
        self._clear_form()

        lock = self._stage_lock_reason(stage_id)
        suffix = f" — LOCKED: {lock}" if lock else ""
        self.description.set(
            f"{stage.number}. {stage.label} — {stage.description}{suffix}"
        )

        if stage_id == "setup":
            self._label("Environment", "Managed NSAMDR CUDA Python environment")
            self._check("Recreate CUDA environment", "force", False)
            self._label("Validation", "Use Detect artifacts or scripts\\build\\nsamdr.bat validate")

        elif stage_id == "quick":
            experiments = ["new", *self._experiment_ids()]
            self._row("Experiment", "experiment", "new", experiments)
            self._label("Model", "Legacy V16.0 Raven qualification baseline")
            self._label("Dataset", "Raven deterministic development set")
            self._label(
                "Current relevance",
                "This path predates the D4 + broad-authority recipe now being qualified. A qualification rejection here is expected evidence, not a software crash.",
            )
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Training regions", "train_crops", "16")
            self._row("Held-out regions", "validation_crops", "4")
            self._check("Rebuild Raven dataset", "rebuild", False)
            self._row("Training control", "control", "auto", ("auto", "resume"))
            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))
            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._row(
                "Performance profile",
                "profile",
                "fast",
                ("optimized", "fast", "balanced", "compatibility"),
            )
            self._row("Workers", "workers", "0" if os.name == "nt" else "4")
            self._row("Prefetch", "prefetch", "2")
            self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
            ttk.Button(
                self.form,
                text="Open latest Raven training preview",
                command=self._open_latest_training_preview,
            ).pack(anchor="w", pady=(8, 2))
            if os.name == "nt":
                self._label(
                    "Windows safety",
                    "Workers=0 avoids the multiprocessing/pinned-transfer path during Raven Quick.",
                )

        elif stage_id == "train":
            self._label("Model", "V16.2 broad-authority production candidate")
            self._label(
                "Current state",
                (
                    "The 32-authority matched-budget proof is complete. D4 augmentation + "
                    "broader authority exposure is now the selected training direction. "
                    "Canonical Main V16 Training integration is the next engineering step."
                ),
            )
            self._label(
                "Safety",
                (
                    "This stage stays locked until the recipe is promoted into the "
                    "canonical workflow. The GUI will not call an obsolete full-train path."
                ),
            )
            ttk.Button(
                self.form,
                text="Open advanced diagnostics / research controls",
                command=self._open_advanced_gui,
            ).pack(anchor="w", pady=(8, 2))
            ttk.Button(
                self.form,
                text="Open NSAMDR README",
                command=lambda: _open_path(REPO_ROOT / "tools/nsamdr/README.md"),
            ).pack(anchor="w", pady=2)

        elif stage_id == "preview":
            choices = list(reversed(self._experiment_ids(qualified_only=True)))
            self._row(
                "Experiment",
                "experiment",
                choices[0] if choices else "<none>",
                choices or ["<none>"],
            )
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Target size", "target", "4096", ("1024", "2048", "4096"))
            self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._label(
                "Contract",
                "Only a completed qualified immutable production final may be previewed.",
            )

        self._update_command_preview()
        self._update_run_state()

    def _update_command_preview(self) -> None:
        selection = self.tree.selection()
        if not selection:
            self.command_preview.set("")
            return
        stage_id = str(selection[0])
        command = self._command_for_stage(stage_id)
        if command is None:
            lock = self._stage_lock_reason(stage_id)
            self.command_preview.set(
                f"[locked] {lock}" if lock else "[no runnable command]"
            )
        else:
            self.command_preview.set(subprocess.list2cmdline(command))
        self._update_run_state()

    def _update_run_state(self) -> None:
        selection = self.tree.selection()
        stage_id = str(selection[0]) if selection else ""
        locked = bool(stage_id and self._stage_lock_reason(stage_id))
        running = self.process is not None and self.process.poll() is None
        self.run_button.configure(
            state=("disabled" if locked or running else "normal")
        )

    def _refresh_preview_choices(self) -> None:
        choices = list(reversed(self._experiment_ids(qualified_only=True)))
        self.preview_combo.configure(values=choices)
        current = self.preview_target.get().strip().upper()
        if current not in choices:
            self.preview_target.set(choices[0] if choices else "")

    def _refresh_scope(self) -> None:
        latest = self._latest_qualified()
        if latest is None:
            self.scope_text.set("Qualified production final: none")
            return
        experiment = _read_json(EXPERIMENT_ROOT / latest / "experiment.json") or {}
        mode = str(experiment.get("trainingMode") or "unknown").upper()
        self.scope_text.set(f"Qualified production final: {latest} | {mode} training")

    def detect(self, silent: bool = False) -> None:
        self._refresh_preview_choices()
        self._refresh_scope()
        statuses = self.state.setdefault("status", {})

        cuda_python = REPO_ROOT / "artifacts/nsamdr/python-env/Scripts/python.exe"
        if self.active_stage != "setup":
            statuses["setup"] = "ready" if cuda_python.is_file() else "pending"

        modes: set[str] = set()
        preview_done = False
        for experiment_id in self._experiment_ids(qualified_only=True):
            directory = EXPERIMENT_ROOT / experiment_id
            experiment = _read_json(directory / "experiment.json") or {}
            modes.add(str(experiment.get("trainingMode") or "").lower())
            preview = _read_json(directory / "previews/preview_manifest.json") or {}
            preview_done = preview_done or preview.get("status") == "launched"

        latest_quick = self._latest_experiment(training_mode="quick")
        latest_quick_status = ""
        if latest_quick is not None:
            quick_payload = _read_json(EXPERIMENT_ROOT / latest_quick / "experiment.json") or {}
            latest_quick_status = str(quick_payload.get("status") or "").lower()

        if self.active_stage != "quick":
            if "quick" in modes:
                statuses["quick"] = "completed"
            elif latest_quick_status == "training-rejected":
                statuses["quick"] = "rejected"
            elif latest_quick_status == "failed":
                statuses["quick"] = "failed"
            else:
                statuses["quick"] = "pending"
        if self.active_stage != "train":
            statuses["train"] = "locked"
        if self.active_stage != "preview":
            statuses["preview"] = "completed" if preview_done else (
                "ready" if self._experiment_ids(qualified_only=True) else "locked"
            )

        for stage in STAGES:
            if self.tree.exists(stage.id):
                self.tree.set(stage.id, "status", statuses.get(stage.id, "pending"))
        self._save_state()
        self._update_run_state()
        if not silent:
            self.progress_text.set("Artifacts and stage locks refreshed")

    def _set_status(self, stage_id: str, status: str) -> None:
        self.state.setdefault("status", {})[stage_id] = status
        if self.tree.exists(stage_id):
            self.tree.set(stage_id, "status", status)
        self._save_state()

    def run_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        stage_id = str(selection[0])
        lock = self._stage_lock_reason(stage_id)
        if lock:
            messagebox.showinfo("NSAMDR", lock)
            return
        command = self._command_for_stage(stage_id)
        if command is None:
            return
        self._start_process(stage_id, command)

    def preview_selected(self) -> None:
        experiment = self.preview_target.get().strip().upper()
        if not experiment:
            messagebox.showinfo("NSAMDR", "No qualified preview experiment is available.")
            return
        self.tree.selection_set("preview")
        self._selected()
        variable = self.vars.get("experiment")
        if variable is not None:
            variable.set(experiment)
        command = self._command_for_stage("preview")
        if command is not None:
            self._start_process("preview", command)

    def _start_process(self, stage_id: str, command: list[str]) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("NSAMDR", "A workflow process is already running.")
            return
        self.output.delete("1.0", "end")
        self.progress["value"] = 0
        self.progress_text.set("Starting...")
        self.active_stage = stage_id
        self.process_started_at = time.monotonic()
        self._set_status(stage_id, "running")
        self._update_run_state()

        def worker() -> None:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.process = process
                self.process_queue.put(("started", process.pid))
                assert process.stdout is not None
                for line in process.stdout:
                    self.process_queue.put(("line", line))
                code = process.wait()
                self.process_queue.put(("done", code))
            except Exception as exc:  # GUI must surface subprocess launch failures.
                self.process_queue.put(("error", repr(exc)))

        self.process_thread = threading.Thread(target=worker, daemon=True)
        self.process_thread.start()

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            self.progress_text.set("No workflow process is running")
            return
        self.progress_text.set("Stopping process...")
        try:
            process.terminate()
        except OSError as exc:
            messagebox.showerror("NSAMDR", f"Could not stop process:\n{exc}")

    def _update_progress_from_line(self, line: str) -> None:
        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            current = int(progress_match.group(1))
            maximum = int(progress_match.group(2))
            if maximum > 0:
                percent = 100.0 * current / maximum
                self.progress["value"] = max(0.0, min(100.0, percent))
                elapsed = (
                    time.monotonic() - self.process_started_at
                    if self.process_started_at is not None
                    else 0.0
                )
                self.progress_text.set(
                    f"{current}/{maximum} ({percent:.1f}%) | elapsed {_format_duration(elapsed)}"
                )
                return

        epoch = EPOCH_RE.search(line)
        if epoch:
            self.current_epoch = (int(epoch.group(1)), int(epoch.group(2)), epoch.group(3))
            self.progress_text.set(
                f"Epoch {epoch.group(1)}/{epoch.group(2)} — {epoch.group(3)}"
            )
            return

        batch = BATCH_RE.search(line)
        if batch and self.current_epoch:
            epoch_index, epoch_total, phase = self.current_epoch
            item = int(batch.group(1))
            item_total = int(batch.group(2))
            if epoch_total > 0 and item_total > 0:
                percent = 100.0 * ((epoch_index - 1) + item / item_total) / epoch_total
                self.progress["value"] = max(0.0, min(100.0, percent))
                step = STEP_MS_RE.search(line)
                eta = ""
                if step:
                    remaining = max(0, item_total - item)
                    eta = f" | phase ETA ~{_format_duration(remaining * float(step.group(1)) / 1000.0)}"
                self.progress_text.set(
                    f"Epoch {epoch_index}/{epoch_total} — {phase} | "
                    f"batch {item}/{item_total} | {percent:.1f}%{eta}"
                )

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.process_queue.get_nowait()
                if kind == "started":
                    self.output.insert("end", f"[GUI] Process started: PID {payload}\n")
                    self.output.see("end")
                    self.progress_text.set(f"Process running — PID {payload}")
                elif kind == "line":
                    line = str(payload)
                    self.output.insert("end", line)
                    self.output.see("end")
                    self._update_progress_from_line(line)
                elif kind == "done":
                    code = int(payload)
                    stage_id = self.active_stage
                    self.process = None
                    self.active_stage = None
                    self.progress["value"] = 100

                    rejected = False
                    rejected_experiment = None
                    if stage_id == "quick" and code == 2:
                        rejected_experiment = self._latest_experiment(training_mode="quick")
                        if rejected_experiment is not None:
                            manifest = _read_json(
                                EXPERIMENT_ROOT / rejected_experiment / "experiment.json"
                            ) or {}
                            rejected = str(manifest.get("status") or "").lower() == "training-rejected"

                    if stage_id:
                        if rejected:
                            self._set_status(stage_id, "rejected")
                        else:
                            self._set_status(stage_id, "completed" if code == 0 else "failed")

                    if rejected:
                        self.progress_text.set(
                            f"Training completed; {rejected_experiment} was rejected by candidate qualification"
                        )
                    else:
                        self.progress_text.set(
                            "Completed successfully" if code == 0 else f"Failed, exit code {code}"
                        )
                    self.detect(silent=True)
                    self._update_run_state()
                elif kind == "error":
                    stage_id = self.active_stage
                    self.process = None
                    self.active_stage = None
                    if stage_id:
                        self._set_status(stage_id, "failed")
                    self.output.insert("end", f"[GUI] ERROR: {payload}\n")
                    self.output.see("end")
                    self.progress_text.set("Could not start workflow process")
                    self._update_run_state()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _open_latest_training_preview(self) -> None:
        preview = self._latest_training_preview()
        if preview is None:
            messagebox.showinfo(
                "NSAMDR",
                "No Raven diagnostic training preview is available yet.",
            )
            return
        _open_path(preview)

    def _open_advanced_gui(self) -> None:
        if not ADVANCED_GUI.is_file():
            messagebox.showerror("NSAMDR", f"Missing advanced GUI:\n{ADVANCED_GUI}")
            return
        cuda_python = REPO_ROOT / "artifacts/nsamdr/python-env/Scripts/python.exe"
        python = cuda_python if cuda_python.is_file() else Path(sys.executable)
        try:
            subprocess.Popen(
                [str(python), "-u", str(ADVANCED_GUI)],
                cwd=REPO_ROOT,
            )
        except OSError as exc:
            messagebox.showerror("NSAMDR", f"Could not open diagnostics GUI:\n{exc}")


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
