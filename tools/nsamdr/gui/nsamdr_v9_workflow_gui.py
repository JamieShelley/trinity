#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

APP_TITLE = "NSAMDR Workflow"
STATE_SCHEMA = "nsamdr-workflow-gui-v6"
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+phase=([^\s]+)", re.I)
BATCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+total=", re.I)
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)tile/s\b", re.I)
STEP_RE = re.compile(r"\bstep=([0-9]+(?:\.[0-9]+)?)ms\b", re.I)
VALIDATION_PROGRESS_RE = re.compile(
    r"^\s*\[validation\] label=([^\s]+) item=(\d+)/(\d+) elapsed=([0-9.]+)s eta=([0-9.]+)s",
    re.I,
)
VRAM_MODE_RE = re.compile(r"\bvram=([^\s]+)\s+free=([0-9]+(?:\.[0-9]+)?)GiB", re.I)
EXPERIMENT_RE = re.compile(r"^EXP_\d{4,}$", re.I)


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


def _schedule_progress(
    epoch_index: int,
    epoch_total: int,
    item_index: int,
    item_total: int,
) -> float:
    """Return schedule completion only; this is deliberately not a time estimate."""
    if epoch_index < 1 or epoch_total < 1 or item_total < 1:
        return 0.0
    epoch_index = min(epoch_index, epoch_total)
    fraction = min(max(float(item_index) / float(item_total), 0.0), 1.0)
    return 100.0 * ((epoch_index - 1) + fraction) / float(epoch_total)


def _phase_eta_from_step_ms(item_index: int, item_total: int, step_ms: float) -> float:
    """Estimate only the remainder of the current homogeneous training loop."""
    if item_total < 1 or item_index < 0 or step_ms <= 0.0:
        return math.inf
    remaining = max(0, int(item_total) - int(item_index))
    return remaining * float(step_ms) / 1000.0


@dataclass(frozen=True)
class Stage:
    id: str
    number: str
    label: str
    command: tuple[str, ...]
    description: str
    pipeline: bool = True


STAGES = (
    Stage("setup", "P0", "Prepare CUDA environment", ("setup", "cuda"), "Create or verify the CUDA Python environment.", False),
    Stage("quick", "1", "Raven Quick", ("raven-quick",), "Train and qualify the complete production NSAMDR model on a deterministic feature-stratified Raven development set."),
    Stage("train", "2", "Full Training", ("full-train",), "Train and qualify the same production model on the full production dataset."),
    Stage("preview", "3", "Preview", ("preview",), "Preview only a completed qualified experiment from its immutable final checkpoint.", False),
)
BY_ID = {stage.id: stage for stage in STAGES}
PIPELINE = [stage.id for stage in STAGES if stage.pipeline]

class App:
    def __init__(self, root: tk.Tk, repo: Path) -> None:
        self.root = root
        self.repo = repo.resolve()
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.queue: queue.Queue[tuple] = queue.Queue()
        self.active_stage: str | None = None
        self.pending: list[str] = []
        self.current_epoch: tuple[int, int, str] | None = None
        self.current_rate: float | None = None
        self.state_dir = self.repo / "artifacts/nsamdr/gui"
        self.log_dir = self.state_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "nsamdr_v9_workflow_state.json"
        self.state = self._load_state()
        self.recovered_interrupted: list[str] = []
        self._normalize_recovered_process_state()
        self.vars: dict[str, tk.Variable] = {}
        # Preview selector follows newly-created experiments until the user
        # explicitly chooses an older one.  This keeps the default on the
        # newest EXP_* while preserving deliberate manual selection.
        self._preview_user_selected = False
        self._preview_last_latest: str | None = None

        # Selected-preview preparation is deliberately tracked separately from
        # the main pipeline worker.  Preview generation can spend several
        # minutes in 4x CUDA inference before the renderer window exists; a
        # dedicated progress/log window keeps that work visible to the user.
        self.preview_process: subprocess.Popen[str] | None = None
        self.preview_thread: threading.Thread | None = None
        self.preview_queue: queue.Queue[tuple] = queue.Queue()
        self.preview_window: tk.Toplevel | None = None
        self.preview_output: tk.Text | None = None
        self.preview_progress: ttk.Progressbar | None = None
        self.preview_status_var = tk.StringVar(value="Preview idle")
        self.preview_elapsed_var = tk.StringVar(value="")
        self.preview_started_at: float | None = None
        self.preview_last_output_at: float | None = None
        self.preview_last_heartbeat_at: float | None = None
        self.preview_phase = "Waiting to start"
        self.preview_log_path: Path | None = None
        self.preview_experiment: str | None = None

        root.title(APP_TITLE)
        root.geometry("1580x940")
        root.minsize(1180, 760)
        self._build()
        self.detect(silent=True)
        if self.recovered_interrupted:
            recovered = ", ".join(
                f"{BY_ID[stage_id].number}. {BY_ID[stage_id].label}"
                for stage_id in self.recovered_interrupted
                if stage_id in BY_ID
            )
            self.output.insert(
                "end",
                "[GUI] Recovered workflow state from the previous session.\n"
                f"[GUI] Previously-running stage(s) are now interrupted: {recovered}\n"
                "[GUI] No stage is running until this GUI launches a new child process.\n",
            )
        self._poll()
        self._poll_preview()
        self.root.after(1000, self._preview_refresh_tick)

    def _load_state(self) -> dict:
        state = {
            "schema": STATE_SCHEMA,
            "status": {stage.id: "pending" for stage in STAGES},
            "current": "quick",
        }
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if loaded.get("schema") == STATE_SCHEMA:
                state.update(loaded)
        except Exception:
            pass
        return state

    def _normalize_recovered_process_state(self) -> None:
        statuses = self.state.setdefault("status", {})
        changed = False
        for stage in STAGES:
            previous = statuses.get(stage.id, "pending")
            if previous in {"running", "stopping"}:
                statuses[stage.id] = "interrupted"
                self.recovered_interrupted.append(stage.id)
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def _build(self) -> None:
        # Root rows are explicit so an oversized stage form can never push the
        # footer or runtime log outside the visible window.
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text=APP_TITLE, font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=str(self.repo)).grid(row=0, column=1, sticky="e")

        self.scope_text = tk.StringVar(value="Model scope: no completed model")
        ttk.Label(
            self.root,
            textvariable=self.scope_text,
            padding=(10, 0, 10, 6),
            font=("Segoe UI", 10, "bold"),
        ).grid(row=1, column=0, sticky="ew")

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=5)

        self.tree = ttk.Treeview(
            left,
            columns=("n", "stage", "status"),
            show="headings",
            selectmode="browse",
        )
        tree_scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        for column, title, width in (("n", "#", 55), ("stage", "Stage", 330), ("status", "Status", 115)):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selected())
        for stage in STAGES:
            self.tree.insert(
                "",
                "end",
                iid=stage.id,
                values=(stage.number, stage.label, self.state["status"].get(stage.id, "pending")),
            )

        self.description = tk.StringVar()
        ttk.Label(
            right,
            textvariable=self.description,
            font=("Segoe UI", 12, "bold"),
            wraplength=980,
        ).pack(anchor="w", fill="x", pady=(4, 6))

        # The right side is split vertically: controls above, runtime below.
        # The controls live inside a Canvas so Stage 2 can grow arbitrarily
        # without consuming the runtime/log/footer area.
        self.right_split = ttk.Panedwindow(right, orient="vertical")
        self.right_split.pack(fill="both", expand=True)

        controls_panel = ttk.Frame(self.right_split)
        runtime_panel = ttk.Frame(self.right_split)
        self.right_split.add(controls_panel, weight=3)
        self.right_split.add(runtime_panel, weight=2)

        controls_header = ttk.Frame(controls_panel)
        controls_header.pack(fill="x", pady=(0, 3))
        ttk.Label(controls_header, text="Stage controls", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(controls_header, text="scroll for additional training controls").pack(side="right")

        form_host = ttk.Frame(controls_panel)
        form_host.pack(fill="both", expand=True)
        form_host.columnconfigure(0, weight=1)
        form_host.rowconfigure(0, weight=1)

        self.form_canvas = tk.Canvas(
            form_host,
            highlightthickness=0,
            borderwidth=0,
            height=330,
        )
        self.form_scroll = ttk.Scrollbar(form_host, orient="vertical", command=self.form_canvas.yview)
        self.form_canvas.configure(yscrollcommand=self.form_scroll.set)
        self.form_canvas.grid(row=0, column=0, sticky="nsew")
        self.form_scroll.grid(row=0, column=1, sticky="ns")

        self.form = ttk.Frame(self.form_canvas, padding=(0, 0, 4, 0))
        self.form_window = self.form_canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", self._form_content_configured)
        self.form_canvas.bind("<Configure>", self._form_canvas_configured)
        self.root.bind_all("<MouseWheel>", self._form_mousewheel, add="+")

        runtime_panel.columnconfigure(0, weight=1)
        runtime_panel.rowconfigure(7, weight=1)
        ttk.Label(runtime_panel, text="Command preview").grid(row=0, column=0, sticky="w", pady=(6, 2))
        self.command = tk.StringVar()
        ttk.Entry(runtime_panel, textvariable=self.command, state="readonly").grid(row=1, column=0, sticky="ew")
        self.progress = ttk.Progressbar(runtime_panel, maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(8, 2))
        self.progress_text = tk.StringVar(value="Idle")
        ttk.Label(runtime_panel, textvariable=self.progress_text).grid(row=3, column=0, sticky="w")
        ttk.Separator(runtime_panel, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=(5, 4))
        ttk.Label(runtime_panel, text="Runtime log", font=("Segoe UI", 9, "bold")).grid(row=5, column=0, sticky="w")

        log_host = ttk.Frame(runtime_panel)
        log_host.grid(row=7, column=0, sticky="nsew", pady=(2, 0))
        log_host.columnconfigure(0, weight=1)
        log_host.rowconfigure(0, weight=1)
        self.output = tk.Text(log_host, height=12, wrap="none")
        output_y = ttk.Scrollbar(log_host, orient="vertical", command=self.output.yview)
        output_x = ttk.Scrollbar(log_host, orient="horizontal", command=self.output.xview)
        self.output.configure(yscrollcommand=output_y.set, xscrollcommand=output_x.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        output_y.grid(row=0, column=1, sticky="ns")
        output_x.grid(row=1, column=0, sticky="ew")

        footer = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        footer.grid(row=3, column=0, sticky="ew")
        ttk.Button(footer, text="Run selected", command=self.run_selected).pack(side="left")
        ttk.Button(footer, text="Run next incomplete", command=self.run_next).pack(side="left", padx=5)
        ttk.Button(footer, text="Run remaining pipeline", command=self.run_remaining).pack(side="left")
        ttk.Button(footer, text="Stop current process", command=self.stop).pack(side="left", padx=5)
        ttk.Label(footer, text="Preview:").pack(side="left", padx=(6, 2))
        self.preview_target = tk.StringVar(value="")
        self.preview_combo = ttk.Combobox(
            footer,
            textvariable=self.preview_target,
            width=14,
            state="readonly",
            postcommand=self._refresh_preview_selector,
        )
        self.preview_combo.bind("<<ComboboxSelected>>", self._preview_selection_changed)
        self.preview_combo.pack(side="left", padx=(0, 3))
        ttk.Button(footer, text="Render selected preview", command=self.preview_available_model).pack(side="left")
        ttk.Button(footer, text="Detect artifacts", command=self.detect).pack(side="right")
        self.footer = footer
        self._refresh_preview_selector()

        current = self.state.get("current", "quick")
        if current not in BY_ID:
            current = "quick"
        self.tree.selection_set(current)
        self._selected()

        # Give the runtime pane useful space on first display. Users can drag
        # the sash afterwards; these values are advisory and scale with window.
        self.root.after_idle(self._set_initial_vertical_split)

    def _form_content_configured(self, _event: tk.Event | None = None) -> None:
        self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))

    def _form_canvas_configured(self, event: tk.Event) -> None:
        self.form_canvas.itemconfigure(self.form_window, width=max(1, event.width))
        self._form_content_configured()

    def _form_mousewheel(self, event: tk.Event) -> None:
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
            if widget is None:
                return
            cursor = widget
            inside = False
            while cursor is not None:
                if cursor == self.form_canvas or cursor == self.form:
                    inside = True
                    break
                parent_name = cursor.winfo_parent()
                if not parent_name:
                    break
                cursor = cursor._nametowidget(parent_name)
            if inside:
                delta = int(-event.delta / 120) if event.delta else 0
                if delta:
                    self.form_canvas.yview_scroll(delta, "units")
        except tk.TclError:
            pass

    def _set_initial_vertical_split(self) -> None:
        try:
            height = self.right_split.winfo_height()
            if height > 200:
                # Controls get about 54%; runtime/log remains clearly visible.
                self.right_split.sashpos(0, int(height * 0.54))
        except (tk.TclError, IndexError):
            pass

    def _clear_form(self) -> None:
        for widget in self.form.winfo_children():
            widget.destroy()
        self.vars = {}

    def _row(self, label: str, key: str, default: str, choices: tuple[str, ...] | list[str] | None = None) -> None:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=27).pack(side="left")
        variable = tk.StringVar(value=default)
        self.vars[key] = variable
        if choices is not None:
            ttk.Combobox(row, textvariable=variable, values=tuple(choices), state="readonly").pack(side="left", fill="x", expand=True)
        else:
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        variable.trace_add("write", lambda *_args: self._update_command())

    def _label_row(self, label: str, value: str) -> None:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=27).pack(side="left")
        ttk.Label(row, text=value).pack(side="left", fill="x", expand=True)

    def _check(self, label: str, key: str, default: bool = False) -> None:
        variable = tk.BooleanVar(value=default)
        self.vars[key] = variable
        ttk.Checkbutton(self.form, text=label, variable=variable, command=self._update_command).pack(anchor="w", pady=2)

    def _experiments_root(self) -> Path:
        return self.repo / "artifacts/nsamdr/experiments"

    @staticmethod
    def _qualified_final(path: Path) -> bool:
        manifest_path = path / "final_manifest.json"
        experiment_path = path / "experiment.json"
        if not manifest_path.is_file() or not experiment_path.is_file():
            return False
        try:
            final = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            checkpoint = final.get("checkpoint", {})
            checkpoint_path = (path / str(checkpoint.get("path") or "")).resolve()
            checkpoint_path.relative_to((path / "checkpoints/final").resolve())
            participation = json.loads(
                (path / "architecture_participation.json").read_text(encoding="utf-8")
            )
            return bool(
                final.get("qualified") is True
                and final.get("status") == "completed"
                and final.get("selectionKind") == "production-final"
                and experiment.get("qualified") is True
                and experiment.get("status") == "completed"
                and bool(re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256") or "")))
                and checkpoint.get("immutable") is True
                and checkpoint.get("selectionKind") == "production-final"
                and participation.get("pass") is True
                and checkpoint_path.is_file()
            )
        except (OSError, ValueError, TypeError):
            return False

    def _experiment_ids(self, *, completed_only: bool = False) -> list[str]:
        root = self._experiments_root()
        if not root.is_dir():
            return []
        values = []
        for path in root.iterdir():
            if not path.is_dir() or EXPERIMENT_RE.fullmatch(path.name) is None:
                continue
            if completed_only and not self._qualified_final(path):
                continue
            values.append(path.name.upper())
        return sorted(values, key=lambda value: int(value.split("_")[1]))

    def _latest_completed_experiment(self) -> str | None:
        values = self._experiment_ids(completed_only=True)
        return values[-1] if values else None

    def _previewable_experiment_ids(self) -> list[str]:
        return self._experiment_ids(completed_only=True)

    def _preview_choices(self) -> list[str]:
        return list(reversed(self._previewable_experiment_ids()))

    def _preview_selection_changed(self, _event: tk.Event | None = None) -> None:
        # Once the user deliberately chooses a historical experiment, periodic
        # discovery must not yank the selector back to latest.
        self._preview_user_selected = True

    def _refresh_preview_selector(self, *, force_latest: bool = False) -> None:
        if not hasattr(self, "preview_combo") or not hasattr(self, "preview_target"):
            return
        choices = self._preview_choices()
        latest = choices[0] if choices else None
        previous_latest = self._preview_last_latest
        new_latest_appeared = latest is not None and latest != previous_latest

        self.preview_combo.configure(values=choices)
        current = self.preview_target.get().strip().upper()
        choose_latest = (
            force_latest
            or not current
            or current not in choices
            or (new_latest_appeared and not self._preview_user_selected)
        )
        if choose_latest:
            self.preview_target.set(latest or "")

        self._preview_last_latest = latest

    def _preview_refresh_tick(self) -> None:
        # Experiments are allocated by a child process while the GUI remains
        # open. Refresh independently of stage completion so EXP_000N appears
        # in the selector as soon as its directory exists.
        try:
            self._refresh_preview_selector()
        finally:
            try:
                self.root.after(1000, self._preview_refresh_tick)
            except tk.TclError:
                pass

    def _refresh_scope(self) -> None:
        latest = self._latest_completed_experiment()
        if latest:
            mode = "unknown"
            try:
                manifest = json.loads((self._experiments_root() / latest / "experiment.json").read_text(encoding="utf-8"))
                mode = str(manifest.get("trainingMode") or "unknown").upper()
            except Exception:
                pass
            self.scope_text.set(f"Qualified production final: {latest} | {mode} training")
        else:
            self.scope_text.set("Qualified production final: none")

    def _stage_lock_reason(self, stage_id: str) -> str | None:
        if stage_id == "preview" and not self._previewable_experiment_ids():
            return "Preview requires a completed qualified EXP_#### final."
        return None

    def _selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        stage_id = selection[0]
        stage = BY_ID[stage_id]
        self.state["current"] = stage_id
        self._save()
        status = self.state["status"].get(stage_id, "pending")
        lock = self._stage_lock_reason(stage_id)
        note = f" — LOCKED: {lock}" if lock else (" — interrupted; auto/resume available" if status == "interrupted" else "")
        self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
        self._clear_form()

        if stage_id == "setup":
            self._check("Recreate environment", "force", False)
        elif stage_id == "quick":
            experiments = ["new", *self._experiment_ids()]
            self._row("Experiment", "experiment", "new", experiments)
            self._label_row("Model", "Complete production NSAMDR architecture")
            self._label_row("Dataset", "Raven Navy Issue - deterministic, feature-stratified, disjoint")
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Training regions", "train_crops", "16")
            self._row("Max held-out regions", "validation_crops", "4")
            self._check("Rebuild fixed Raven dataset", "rebuild", False)
            self._row("Training control", "control", "auto", ("auto", "resume"))
            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))
            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._training_performance_rows(default_workers="4")
        elif stage_id == "train":
            experiments = ["new", *self._experiment_ids()]
            self._row("Experiment", "experiment", "new", experiments)
            self._label_row("Model", "Complete production NSAMDR architecture")
            self._label_row("Dataset", "Full production authored EVE dataset")
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._check("Rebuild full dataset", "rebuild", False)
            self._row("Training control", "control", "auto", ("auto", "resume"))
            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))
            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._training_performance_rows(default_workers="8")
        elif stage_id == "preview":
            completed = list(reversed(self._previewable_experiment_ids()))
            self._row("Experiment", "experiment", completed[0] if completed else "<none>", completed or ("<none>",))
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Target size", "target", "4096", ("1024", "2048", "4096"))
            self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
        self._update_command()
        self.form_canvas.yview_moveto(0.0)
        self.root.after_idle(self._form_content_configured)

    def _training_performance_rows(self, *, default_workers: str) -> None:
        self._row("Performance profile", "profile", "fast", ("optimized", "fast", "balanced", "compatibility"))
        self._row("Workers", "workers", default_workers)
        self._row("Prefetch", "prefetch", "2")
        self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))

    def _value(self, key: str, default: str = "") -> str:
        variable = self.vars.get(key)
        return str(variable.get()) if variable is not None else default

    def _args(self, stage_id: str) -> list[str]:
        args: list[str] = []
        if stage_id == "setup":
            if self.vars.get("force") and bool(self.vars["force"].get()):
                args.append("--force")
        elif stage_id == "quick":
            experiment = self._value("experiment", "new")
            args = [
                "--shared-cache", self._value("cache"),
                "--max-train-regions", self._value("train_crops", "16"),
                "--max-validation-regions", self._value("validation_crops", "4"),
                "--experiment", experiment,
                "--control", self._value("control", "auto"),
                "--preview-target-size", self._value("target", "4096"),
                "--preview-device", self._value("device", "cuda"),
                "--performance-profile", self._value("profile", "fast"),
                "--workers", self._value("workers", "4"),
                "--prefetch-factor", self._value("prefetch", "2"),
                "--amp-precision", self._value("amp", "auto"),
            ]
            if bool(self.vars["rebuild"].get()):
                args.append("--rebuild-dataset")
        elif stage_id == "train":
            args = [
                "--shared-cache", self._value("cache"),
                "--experiment", self._value("experiment", "new"),
                "--control", self._value("control", "auto"),
                "--performance-profile", self._value("profile", "fast"),
                "--workers", self._value("workers", "8"),
                "--prefetch-factor", self._value("prefetch", "2"),
                "--amp-precision", self._value("amp", "auto"),
                "--preview-target-size", self._value("target", "4096"),
                "--preview-device", self._value("device", "cuda"),
            ]
            if bool(self.vars["rebuild"].get()):
                args.append("--rebuild-dataset")
        elif stage_id == "preview":
            args = [
                self._value("experiment"),
                "--shared-cache", self._value("cache"),
                "--target-size", self._value("target"),
                "--device", self._value("device"),
            ]
        return args

    def _training_artifact_paths(self, stage_id: str) -> tuple[Path | None, Path | None]:
        if stage_id in {"quick", "train"}:
            experiment = self._value("experiment", "new")
            if experiment.lower() == "new":
                return None, None
            output = self._experiments_root() / experiment
        else:
            return None, None
        return output / "nsamdr_v9_training_state.pt", output / "nsamdr_v9_fidelity.pt"

    def _validate_before_launch(self, stage_id: str) -> bool:
        lock = self._stage_lock_reason(stage_id)
        if lock:
            messagebox.showwarning("Stage locked", lock)
            self.progress_text.set(f"Locked — {lock}")
            return False
        if stage_id in {"quick", "train"}:
            control = self._value("control", "auto")
            state_path, checkpoint_path = self._training_artifact_paths(stage_id)
            if self._value("experiment", "new").lower() != "new":
                experiment_id = self._value("experiment").upper()
                manifest_path = self._experiments_root() / experiment_id / "experiment.json"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    stored_mode = str(manifest.get("trainingMode") or "full").lower()
                    selected_mode = "quick" if stage_id == "quick" else "full"
                    if stored_mode != selected_mode:
                        messagebox.showerror(
                            APP_TITLE,
                            f"{experiment_id} is an existing {stored_mode.upper()} experiment. "
                            "Its immutable training mode cannot be changed while resuming.",
                        )
                        return False
                    if manifest.get("status") == "completed" or manifest.get("qualified") is True:
                        messagebox.showinfo(
                            APP_TITLE,
                            "Completed experiments are immutable. Select 'new' to allocate a new EXP_####.",
                        )
                        return False
                except OSError:
                    pass
            if control == "resume" and (state_path is None or not state_path.is_file()):
                messagebox.showerror(APP_TITLE, f"Resume selected but no resumable state exists:\n\n{state_path or '<new experiment>'}")
                return False
            if control == "auto" and state_path is not None and not state_path.is_file() and checkpoint_path and checkpoint_path.is_file():
                messagebox.showerror(APP_TITLE, "A final checkpoint exists without a resumable state. Auto refuses to overwrite it.")
                return False
        return True

    def _dispatcher_argv(self, command: tuple[str, ...], args: list[str]) -> list[str]:
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/nsamdr_cli.py"),
            *command,
            *args,
        ]

    def _command_line(self, stage_id: str, args: list[str] | None = None) -> str:
        arguments = args if args is not None else self._args(stage_id)
        return subprocess.list2cmdline(self._dispatcher_argv(BY_ID[stage_id].command, arguments))

    def _process_command(self, stage_id: str, args: list[str] | None = None) -> list[str]:
        arguments = args if args is not None else self._args(stage_id)
        return self._dispatcher_argv(BY_ID[stage_id].command, arguments)

    def _update_command(self) -> None:
        selection = self.tree.selection()
        if selection:
            self.command.set(self._command_line(selection[0]))

    def _set_status(self, stage_id: str, status: str) -> None:
        self.state["status"][stage_id] = status
        if self.tree.exists(stage_id):
            self.tree.set(stage_id, "status", status)
        self._save()

    def run_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        stage_id = selection[0]
        if self._validate_before_launch(stage_id):
            self._run(stage_id)

    def run_next(self) -> None:
        for stage_id in PIPELINE:
            if self.state["status"].get(stage_id) != "completed":
                self.tree.selection_set(stage_id)
                self._selected()
                if self._validate_before_launch(stage_id):
                    self._run(stage_id)
                return

    def run_remaining(self) -> None:
        self.pending = [stage_id for stage_id in PIPELINE if self.state["status"].get(stage_id) != "completed"]
        self._run_pending()

    def _run_pending(self) -> None:
        if self.process is not None or not self.pending:
            return
        stage_id = self.pending.pop(0)
        self.tree.selection_set(stage_id)
        self._selected()
        if not self._validate_before_launch(stage_id):
            self.pending.clear()
            return
        self._run(stage_id, queue_next=True)

    def _run(self, stage_id: str, queue_next: bool = False, args: list[str] | None = None) -> None:
        if self.process is not None:
            messagebox.showwarning(APP_TITLE, "A stage is already running.")
            return
        command = self._process_command(stage_id, args)
        display_command = self._command_line(stage_id, args)
        if stage_id in {"quick", "train"}:
            experiment_var = self.vars.get("experiment")
            experiment_value = str(experiment_var.get()).strip().lower() if experiment_var is not None else ""
            if experiment_value == "new":
                # A newly allocated EXP_* becomes the default preview target as
                # soon as the child process creates its directory.
                self._preview_user_selected = False
        self.active_stage = stage_id
        self.current_epoch = None
        self.current_rate = None
        self._set_status(stage_id, "running")
        self.progress["value"] = 0
        self.progress_text.set(f"Launching stage {BY_ID[stage_id].number}...")
        self.output.delete("1.0", "end")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{stamp}_{stage_id}.log"
        self.output.insert(
            "end",
            f"[GUI] Starting {BY_ID[stage_id].number}. {BY_ID[stage_id].label}\n"
            f"[GUI] Command: {display_command}\n"
            f"[GUI] Log: {log_path}\n"
            "[GUI] Launching command process...\n",
        )
        self.output.see("end")
        self.thread = threading.Thread(
            target=self._worker,
            args=(stage_id, command, log_path, queue_next),
            daemon=True,
        )
        self.thread.start()

    def _worker(self, stage_id: str, command: list[str], log_path: Path, queue_next: bool) -> None:
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.repo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self.queue.put(("started", stage_id, self.process.pid, str(log_path)))
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    self.queue.put(("line", stage_id, line))
                return_code = self.process.wait()
            self.queue.put(("done", stage_id, return_code, queue_next))
        except Exception as exc:
            self.queue.put(("error", stage_id, repr(exc), queue_next))

    def _poll(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                kind, stage_id = item[0], item[1]
                if kind == "started":
                    self.output.insert("end", f"[GUI] Process started: PID {item[2]}\n")
                    self.output.see("end")
                    self.progress_text.set(f"Process running — PID {item[2]}")
                elif kind == "line":
                    line = item[2]
                    self.output.insert("end", line)
                    self.output.see("end")
                    if line.startswith("[startup] "):
                        self.progress_text.set(line[len("[startup] "):].strip())
                    epoch = EPOCH_RE.search(line)
                    if epoch:
                        self.current_epoch = (int(epoch.group(1)), int(epoch.group(2)), epoch.group(3))
                        self.current_rate = None
                        self.progress_text.set(f"Epoch {epoch.group(1)}/{epoch.group(2)} — {epoch.group(3)} — waiting for live rate...")
                    if line.lstrip().startswith("[VRAM] safety wait:") and self.current_epoch:
                        e, total, phase = self.current_epoch
                        self.progress_text.set(f"Epoch {e}/{total} — {phase} | ETA paused — waiting for GPU/RAM safety envelope")
                    validation = VALIDATION_PROGRESS_RE.search(line)
                    if validation and self.current_epoch:
                        e, total, phase = self.current_epoch
                        label = validation.group(1)
                        item, item_total = int(validation.group(2)), int(validation.group(3))
                        elapsed = float(validation.group(4))
                        eta = float(validation.group(5))
                        self.progress_text.set(
                            f"Epoch {e}/{total} — {phase} | {label} {item}/{item_total} | "
                            f"Elapsed {_format_duration(elapsed)} | Phase ETA ~{_format_duration(eta)}"
                        )
                    batch = BATCH_RE.search(line)
                    rate = RATE_RE.search(line)
                    step = STEP_RE.search(line)
                    if batch and rate and self.current_epoch:
                        e, total, phase = self.current_epoch
                        b, btotal = int(batch.group(1)), int(batch.group(2))
                        live_rate = float(rate.group(1))
                        progress = _schedule_progress(e, total, b, btotal)
                        self.progress["value"] = progress
                        phase_eta = (
                            _phase_eta_from_step_ms(b, btotal, float(step.group(1)))
                            if step else math.inf
                        )
                        vram = VRAM_MODE_RE.search(line)
                        vram_text = f" | VRAM {vram.group(1)} {float(vram.group(2)):.2f}GiB free" if vram else ""
                        self.progress_text.set(
                            f"Epoch {e}/{total} — {phase} | Batch {b}/{btotal} | schedule {progress:.1f}% | "
                            f"Rate {live_rate:.2f} tile/s | Phase ETA ~{_format_duration(phase_eta)}{vram_text}"
                        )
                elif kind == "done":
                    return_code, queue_next = item[2], item[3]
                    self.process = None
                    # Epoch-derived progress can legitimately stop at 30-40% when
                    # Quick mode exits its staged training schedule early. Once the
                    # process exits, the operation is 100% finished regardless of
                    # success/failure; never leave a stale partial green bar.
                    self.progress["value"] = 100
                    self.detect(silent=True)
                    # Artifact detection is historical and may find an older
                    # successful experiment. The explicit result of the stage that
                    # just ran must win over that historical state.
                    self._set_status(stage_id, "completed" if return_code == 0 else "failed")
                    if stage_id in {"quick", "train"}:
                        result_path = self.repo / "artifacts/nsamdr/gui/last_nsamdr_workflow_result.json"
                        if result_path.is_file():
                            try:
                                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                                latest_experiment = str(result_payload.get("experiment") or "").strip().upper()
                                if latest_experiment:
                                    self._refresh_preview_selector()
                                    if latest_experiment in self._preview_choices():
                                        self.preview_target.set(latest_experiment)
                            except (OSError, ValueError, TypeError):
                                pass
                        if return_code == 0:
                            self.progress_text.set("Completed - qualified final renderer launched")
                        else:
                            self.progress_text.set(
                                f"Training or qualified preview failed (exit code {return_code})"
                            )
                    else:
                        self.progress_text.set(
                            "Completed successfully" if return_code == 0 else f"Failed, exit code {return_code}"
                        )
                    if queue_next and return_code == 0:
                        self.root.after(150, self._run_pending)
                else:
                    self.process = None
                    self.progress["value"] = 100
                    self._set_status(stage_id, "failed")
                    self.progress_text.set(item[2])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True)
        except Exception:
            self.process.terminate()
        if self.active_stage:
            self._set_status(self.active_stage, "interrupted")

    def _preview_phase_from_line(self, line: str) -> str | None:
        """Translate noisy preview subprocess output into a user-facing phase."""
        text = line.strip()
        lower = text.lower()
        if not text:
            return None
        if "reading eve resource indexes" in lower or "indexed resources:" in lower:
            return "Reading EVE resource indexes and Raven source assets"
        if "selected model:" in lower or "selected albedo:" in lower or "selected normal:" in lower:
            return "Resolving Raven model, albedo and normal-map inputs"
        if "direct production inference" in lower:
            return "Running direct production-model inference from the immutable final"
        if lower.startswith("[candidate] ["):
            return "Generating final Raven physical-map textures"
        if "[candidate] verified" in lower:
            return "Candidate textures generated; preparing the renderer"
        if "preview=pass" in lower:
            return "Candidate provenance verified; renderer launch approved"
        if "cmake" in lower and ("configure" in lower or "build" in lower):
            return "Configuring the standalone DX11 preview renderer"
        if "msbuild" in lower or "nsamdrpreview_dx11" in lower and ("build" in lower or "link" in lower):
            return "Building/reusing the standalone DX11 preview renderer"
        if "launching" in lower and ("renderer" in lower or "preview" in lower):
            return "Launching the Raven comparison renderer"
        if "preview failed" in lower or "renderer preview" in lower and "failed" in lower:
            return "Preview preparation failed"
        return None

    def _append_preview_log(self, text: str) -> None:
        if self.preview_output is None:
            return
        try:
            self.preview_output.insert("end", text)
            self.preview_output.see("end")
        except tk.TclError:
            pass

    def _show_preview_progress_window(self, experiment: str, log_path: Path) -> None:
        if self.preview_window is not None:
            try:
                if self.preview_window.winfo_exists():
                    self.preview_window.deiconify()
                    self.preview_window.lift()
                    return
            except tk.TclError:
                pass

        win = tk.Toplevel(self.root)
        self.preview_window = win
        win.title(f"NSAMDR Preview Preparation — {experiment}")
        win.geometry("1120x650")
        win.minsize(760, 420)

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=f"Preparing renderer preview for {experiment}", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.preview_status_var).pack(anchor="w", pady=(6, 2))
        ttk.Label(outer, textvariable=self.preview_elapsed_var).pack(anchor="w", pady=(0, 8))

        progress = ttk.Progressbar(outer, mode="indeterminate")
        self.preview_progress = progress
        progress.pack(fill="x", pady=(0, 10))
        progress.start(10)

        log_frame = ttk.Frame(outer)
        log_frame.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical")
        output = tk.Text(log_frame, wrap="none", yscrollcommand=scroll.set, font=("Consolas", 9))
        self.preview_output = output
        scroll.configure(command=output.yview)
        scroll.pack(side="right", fill="y")
        output.pack(side="left", fill="both", expand=True)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, text=f"Detailed log: {log_path}").pack(side="left", anchor="w")
        ttk.Button(footer, text="Hide", command=win.withdraw).pack(side="right")

        self._append_preview_log(
            f"[GUI PREVIEW] Experiment: {experiment}\n"
            f"[GUI PREVIEW] Detailed log: {log_path}\n"
            "[GUI PREVIEW] The renderer window will open only after candidate preparation completes.\n"
            "[GUI PREVIEW] This window remains live during long CUDA inference steps.\n\n"
        )

    def _preview_worker(self, experiment: str, command: list[str], log_path: Path) -> None:
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                log.write(f"[GUI PREVIEW] Experiment: {experiment}\n")
                log.write(f"[GUI PREVIEW] Command: {subprocess.list2cmdline(command)}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.repo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
                self.preview_process = process
                self.preview_queue.put(("started", experiment, process.pid))
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.preview_queue.put(("line", experiment, line))
                return_code = process.wait()
            self.preview_queue.put(("done", experiment, return_code))
        except Exception as exc:
            self.preview_queue.put(("error", experiment, repr(exc)))

    def _poll_preview(self) -> None:
        try:
            while True:
                item = self.preview_queue.get_nowait()
                kind, experiment = item[0], item[1]
                now = time.monotonic()
                if kind == "started":
                    self.preview_started_at = now
                    self.preview_last_output_at = now
                    self.preview_last_heartbeat_at = now
                    self.preview_phase = "Preview subprocess started; resolving Raven inputs"
                    self.preview_status_var.set(f"{self.preview_phase} — PID {item[2]}")
                    self._append_preview_log(f"[GUI PREVIEW] Process started: PID {item[2]}\n")
                elif kind == "line":
                    line = item[2]
                    self.preview_last_output_at = now
                    self._append_preview_log(line)
                    phase = self._preview_phase_from_line(line)
                    if phase:
                        self.preview_phase = phase
                        self.preview_status_var.set(phase)
                elif kind == "done":
                    return_code = int(item[2])
                    elapsed = now - self.preview_started_at if self.preview_started_at is not None else 0.0
                    self.preview_process = None
                    if self.preview_progress is not None:
                        try:
                            self.preview_progress.stop()
                        except tk.TclError:
                            pass
                    if return_code == 0:
                        self.preview_phase = "Preview preparation completed / renderer launched"
                        self.preview_status_var.set(self.preview_phase)
                        self.progress_text.set(f"Renderer preview prepared for {experiment}")
                    else:
                        self.preview_phase = f"Preview preparation failed (exit code {return_code})"
                        self.preview_status_var.set(self.preview_phase)
                        self.progress_text.set(self.preview_phase)
                    self.preview_elapsed_var.set(f"Elapsed: {_format_duration(elapsed)}")
                    self._append_preview_log(
                        f"\n[GUI PREVIEW] Process exited with code {return_code} after {_format_duration(elapsed)}.\n"
                    )
                elif kind == "error":
                    self.preview_process = None
                    if self.preview_progress is not None:
                        try:
                            self.preview_progress.stop()
                        except tk.TclError:
                            pass
                    self.preview_phase = "Preview preparation worker failed"
                    self.preview_status_var.set(self.preview_phase)
                    self._append_preview_log(f"\n[GUI PREVIEW] ERROR: {item[2]}\n")
        except queue.Empty:
            pass

        now = time.monotonic()
        if self.preview_process is not None and self.preview_started_at is not None:
            elapsed = now - self.preview_started_at
            silent_for = now - (self.preview_last_output_at or self.preview_started_at)
            self.preview_elapsed_var.set(
                f"Elapsed: {_format_duration(elapsed)} | Last subprocess output: {_format_duration(silent_for)} ago | Process active"
            )
            # Long CUDA kernels may emit no lines for minutes.  Emit a sparse
            # GUI-owned heartbeat into the dedicated log so silence can never
            # look like a frozen/idle tool.
            if self.preview_last_heartbeat_at is None or now - self.preview_last_heartbeat_at >= 10.0:
                self.preview_last_heartbeat_at = now
                self._append_preview_log(
                    f"[GUI PREVIEW] {_format_duration(elapsed)} elapsed — {self.preview_phase}; process still active.\n"
                )

        try:
            self.root.after(250, self._poll_preview)
        except tk.TclError:
            pass

    def _launch_preview_with_detailed_log(self, experiment: str, command: list[str]) -> None:
        if self.preview_process is not None:
            self._show_preview_progress_window(experiment, self.preview_log_path or self.log_dir / "preview.log")
            messagebox.showinfo(APP_TITLE, "A renderer preview is already being prepared. Its detailed progress window has been brought forward.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{stamp}_preview_{experiment.lower()}.log"
        self.preview_log_path = log_path
        self.preview_experiment = experiment
        self.preview_started_at = None
        self.preview_last_output_at = None
        self.preview_last_heartbeat_at = None
        self.preview_phase = "Launching preview preparation"
        self.preview_status_var.set(self.preview_phase)
        self.preview_elapsed_var.set("")
        self._show_preview_progress_window(experiment, log_path)
        if self.preview_output is not None:
            try:
                self.preview_output.delete("1.0", "end")
            except tk.TclError:
                pass
        self._append_preview_log(
            f"[GUI PREVIEW] Experiment: {experiment}\n"
            f"[GUI PREVIEW] Detailed log: {log_path}\n"
            f"[GUI PREVIEW] Command: {subprocess.list2cmdline(command)}\n"
            "[GUI PREVIEW] The renderer window will open only after candidate preparation completes.\n"
            "[GUI PREVIEW] Long CUDA inference steps receive a GUI heartbeat every 10 seconds.\n\n"
        )
        self.preview_thread = threading.Thread(
            target=self._preview_worker,
            args=(experiment, command, log_path),
            daemon=True,
        )
        self.preview_thread.start()

    def preview_available_model(self) -> None:
        self._refresh_preview_selector()
        target = self.preview_target.get().strip().upper() if hasattr(self, "preview_target") else ""
        if not target:
            messagebox.showinfo(APP_TITLE, "No completed qualified experiment exists yet.")
            return

        experiment = target
        if experiment not in self._previewable_experiment_ids():
            self._refresh_preview_selector()
            messagebox.showinfo(
                APP_TITLE,
                f"{experiment} is not a completed qualified immutable final.",
            )
            return

        preview_args = [
            experiment, "--shared-cache", self._value("cache", r"C:\CCP\EVE"),
            "--target-size", self._value("target", "4096"),
            "--device", self._value("device", "cuda"),
        ]
        command = self._dispatcher_argv(("preview",), preview_args)
        self._launch_preview_with_detailed_log(experiment, command)
        self.progress_text.set(f"Preparing diagnostic Raven renderer preview for {experiment} — see detailed preview log")

    def detect(self, silent: bool = False) -> None:
        self._refresh_preview_selector()
        statuses = self.state.setdefault("status", {})
        completed_experiments = self._experiment_ids(completed_only=True)
        completed_modes: set[str] = set()
        preview_completed = False
        for experiment_id in completed_experiments:
            directory = self._experiments_root() / experiment_id
            try:
                experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
                completed_modes.add(str(experiment.get("trainingMode") or "").lower())
                preview_path = directory / "previews/preview_manifest.json"
                if preview_path.is_file():
                    preview = json.loads(preview_path.read_text(encoding="utf-8"))
                    preview_completed = preview_completed or preview.get("status") == "launched"
            except (OSError, ValueError, TypeError):
                continue
        if self.active_stage != "quick":
            statuses["quick"] = "completed" if "quick" in completed_modes else "pending"
        if self.active_stage != "train":
            statuses["train"] = "completed" if "full" in completed_modes else "pending"
        if self.active_stage != "preview":
            statuses["preview"] = "completed" if preview_completed else "pending"
        for stage in STAGES:
            if self.process is not None and stage.id == self.active_stage:
                continue
            lock = self._stage_lock_reason(stage.id)
            if lock and statuses.get(stage.id) != "completed":
                statuses[stage.id] = "locked"
            elif statuses.get(stage.id) == "locked":
                statuses[stage.id] = "pending"
            if self.tree.exists(stage.id):
                self.tree.set(stage.id, "status", statuses.get(stage.id, "pending"))
        self._save()
        self._refresh_scope()
        if not silent:
            self.progress_text.set("Artifacts detected; locks/capabilities refreshed")
        selection = self.tree.selection()
        if selection:
            self._selected()


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    root = tk.Tk()
    App(root, repository)
    root.mainloop()


if __name__ == "__main__":
    main()
