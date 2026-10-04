#!/usr/bin/env python3
"""Canonical NSAMDR V17 operator workflow GUI.

V17 is the active reconstruction architecture. The normal operator path is now
environment -> legacy Raven comparison -> V17 unseen-sibling architecture proof
-> preview/review. Broad V16 Main training is historical and is not exposed as
the active training stage.

The V17 architecture gate is deliberately cheap: train one authored crop,
evaluate a different never-trained crop from the same authority, and inspect the
A/B/C sibling sheet. Broad training is not allowed until that transfer is
visually credible.
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
STATE_PATH = GUI_STATE_ROOT / "v17_operator_workflow_state.json"
DIAGNOSTIC_ROOT = REPO_ROOT / "artifacts/nsamdr/diagnostics"
V17_PROOF_ROOT = DIAGNOSTIC_ROOT / "v17_sibling_architecture"
MAIN_TRAINING_ROOT = REPO_ROOT / "artifacts/nsamdr/main_training"
APP_TITLE = "NSAMDR V17 Workflow"
STATE_SCHEMA = "nsamdr-v17-operator-workflow-v1"
V17_PREVIEW_CHOICE = "V17_SIBLING_LATEST"
MAIN_PREVIEW_CHOICE = "MAIN_V16_LATEST"

EXPERIMENT_RE = re.compile(r"^EXP_\d{4,}$", re.I)
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+phase=([^\s]+)", re.I)
BATCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+total=", re.I)
PROGRESS_RE = re.compile(r"total=(\d+)/(\d+)")
FULL_BROAD_STEP_RE = re.compile(r"\[full-broad\]\s+step\s+(\d+)\s*/\s*(\d+)", re.I)
V17_UPDATE_RE = re.compile(r"\[v17-sibling\]\s+update\s+(\d+)\s*/\s*(\d+)", re.I)
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
        "Legacy Raven Quick",
        "Historical V16.0 Raven baseline retained for comparison and regression checks.",
    ),
    Stage(
        "train",
        "2",
        "V17 Sibling Architecture Proof",
        "Train one authored crop, evaluate an unseen sibling crop from the same authority, and require visual A/B/C review before any broad training.",
    ),
    Stage(
        "preview",
        "3",
        "Preview / visual gate",
        "Open the latest V17 unseen-sibling A/B/C sheet or historical preview artifacts.",
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
        self.progress_value = tk.DoubleVar(value=0.0)
        self.scope_text = tk.StringVar(value="Qualified production final: checking")
        self.recipe_text = tk.StringVar(
            value=(
                "Active architecture: V17 LR physical encoder + relative-coordinate "
                "subpixel residual decoder. Current gate: unseen same-authority sibling crop."
            )
        )
        self.preview_target = tk.StringVar(value="")

        self._build()
        self.detect(silent=True)
        self.root.after(100, self._poll)

    def _load_state(self) -> dict[str, Any]:
        default = {
            "schema": STATE_SCHEMA,
            "current": "train",
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

    def _experiment_live_preview(self, experiment_id: str) -> Path | None:
        candidate = (
            EXPERIMENT_ROOT
            / str(experiment_id).upper()
            / "previews/live/latest_ABCF.png"
        )
        return candidate.resolve() if candidate.is_file() else None

    def _latest_training_preview(self) -> Path | None:
        experiment_id = self._latest_experiment(training_mode="quick")
        return (
            self._experiment_live_preview(experiment_id)
            if experiment_id is not None
            else None
        )

    def _latest_v17_proof_report(self) -> tuple[Path, dict[str, Any]] | None:
        if not V17_PROOF_ROOT.is_dir():
            return None
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for report_path in V17_PROOF_ROOT.glob("proof_*/report.json"):
            payload = _read_json(report_path)
            if payload is None:
                continue
            if payload.get("schema") not in {"NSAMDR_V17_SIBLING_ARCHITECTURE_PROOF_V1", "NSAMDR_V17_SIBLING_ARCHITECTURE_PROOF_V2"}:
                continue
            try:
                stamp = report_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((stamp, report_path, payload))
        if not candidates:
            return None
        _stamp, report_path, payload = max(candidates, key=lambda item: item[0])
        return report_path.resolve(), payload

    def _latest_v17_proof_dir(self) -> Path | None:
        latest = self._latest_v17_proof_report()
        if latest is not None:
            return latest[0].parent
        if not V17_PROOF_ROOT.is_dir():
            return None
        candidates: list[tuple[float, Path]] = []
        for path in V17_PROOF_ROOT.glob("proof_*"):
            if not path.is_dir():
                continue
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            candidates.append((stamp, path))
        return max(candidates, key=lambda item: item[0])[1].resolve() if candidates else None

    def _latest_v17_preview(self, *, role: str = "sibling") -> Path | None:
        latest = self._latest_v17_proof_report()
        if latest is not None:
            report_path, payload = latest
            snapshots = list(payload.get("snapshots") or [])
            for snapshot in reversed(snapshots):
                previews = dict(snapshot.get("previews") or {})
                raw = str(previews.get(role) or "").strip()
                if raw:
                    path = Path(raw)
                    if not path.is_absolute():
                        path = REPO_ROOT / path
                    if path.is_file():
                        return path.resolve()
            run_dir = report_path.parent
        else:
            run_dir = self._latest_v17_proof_dir()
            if run_dir is None:
                return None

        matches = sorted(
            run_dir.glob(f"stage_*/{role}_ABC.png"),
            key=lambda path: path.parent.name,
            reverse=True,
        )
        return matches[0].resolve() if matches else None

    def _live_main_training_pointer(self) -> dict[str, Any] | None:
        root = DIAGNOSTIC_ROOT / "v16_full_broad"
        if not root.is_dir():
            return None
        candidates: list[tuple[float, dict[str, Any]]] = []
        for pointer in root.glob("probe_*/previews/latest.json"):
            payload = _read_json(pointer)
            if payload is None:
                continue
            if payload.get("schema") != "NSAMDR_V16_FULL_BROAD_LIVE_PREVIEW_V1":
                continue
            if str(payload.get("augmentationPolicy") or "") != "d4-cyclic":
                continue
            if str(payload.get("spatialPolicy") or "") != "balanced-detail":
                continue
            try:
                stamp = pointer.stat().st_mtime
            except OSError:
                continue
            candidates.append((stamp, payload))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _main_training_pointer(self) -> dict[str, Any] | None:
        return _read_json(MAIN_TRAINING_ROOT / "latest.json")

    def _latest_main_checkpoint(self) -> Path | None:
        pointer = self._main_training_pointer() or {}
        raw = str(pointer.get("resumeCheckpoint") or "").strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.is_file():
                return path.resolve()

        # A run can have published its immutable epoch checkpoint/live pointer
        # before the final main-training pointer is written. Recover the sibling
        # resume checkpoint so an already-completed epoch never needs retraining.
        live = self._live_main_training_pointer() or {}
        raw = str(live.get("checkpoint") or "").strip()
        if not raw:
            return None
        stage = Path(raw)
        if not stage.is_absolute():
            stage = REPO_ROOT / stage
        if not stage.is_file():
            return None
        run_dir = stage.parent.parent if stage.parent.name == "checkpoints" else stage.parent
        resume = run_dir / "resume_checkpoint.pt"
        return resume.resolve() if resume.is_file() else None

    def _latest_main_preview(self) -> Path | None:
        live = self._live_main_training_pointer() or {}
        raw = str(live.get("albedoComparison") or "").strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.is_file():
                return path.resolve()
        live_root = str(live.get("previewRoot") or "").strip()
        if live_root:
            root = Path(live_root)
            if not root.is_absolute():
                root = REPO_ROOT / root
            matches = sorted(root.glob("sample_*/albedo_comparison.png"))
            if matches:
                return matches[0].resolve()

        pointer = self._main_training_pointer() or {}
        raw = str(pointer.get("previewAlbedoComparison") or "").strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.is_file():
                return path.resolve()
        root_raw = str(pointer.get("previewRoot") or "").strip()
        if root_raw:
            root = Path(root_raw)
            if not root.is_absolute():
                root = REPO_ROOT / root
            matches = sorted(root.glob("sample_*/albedo_comparison.png"))
            if matches:
                return matches[0].resolve()
        return None

    def _preview_choices(self) -> list[str]:
        return [
            V17_PREVIEW_CHOICE,
            MAIN_PREVIEW_CHOICE,
            *list(reversed(self._experiment_ids())),
        ]

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
            command = self._cli_argv(
                "v17-sibling-proof",
                "--device",
                self._value("device", "cuda"),
                "--amp-precision",
                self._value("amp", "auto"),
                "--authority-id",
                self._value("authority_id", "13006d2b807f89ac"),
                "--hr-size",
                self._value("hr_size", "512"),
                "--stages",
                self._value("stages", "64,128,256,384"),
                "--learning-rate",
                self._value("learning_rate", "0.0002"),
                "--minimum-target-residual",
                self._value("minimum_target_residual", "0.01"),
                "--encoder-channels",
                self._value("encoder_channels", "96"),
                "--neighbourhood-channels",
                self._value("neighbourhood_channels", "128"),
                "--decoder-hidden-channels",
                self._value("decoder_hidden_channels", "192"),
            )
            manifest = self._value("manifest", "").strip()
            if manifest:
                command += ["--manifest", manifest]
            return command

        if stage_id == "preview":
            experiment = self._value("experiment", "")
            if (
                not experiment
                or experiment == "<none>"
                or experiment == MAIN_PREVIEW_CHOICE
                or not _qualified_final(EXPERIMENT_ROOT / experiment)
            ):
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

        controls = ttk.LabelFrame(right, text="Stage controls", padding=6)
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(0, weight=1)

        # Keep the runtime/status area visible even for stages with many controls.
        # The old direct frame let Raven Quick's long form consume the entire
        # right pane and push the progress bar/log below the visible window.
        self.controls_canvas = tk.Canvas(
            controls,
            height=300,
            highlightthickness=0,
            borderwidth=0,
        )
        controls_scroll = ttk.Scrollbar(
            controls,
            orient="vertical",
            command=self.controls_canvas.yview,
        )
        self.controls_canvas.configure(yscrollcommand=controls_scroll.set)
        self.controls_canvas.grid(row=0, column=0, sticky="ew")
        controls_scroll.grid(row=0, column=1, sticky="ns")

        self.form = ttk.Frame(self.controls_canvas)
        self.controls_window = self.controls_canvas.create_window(
            (0, 0),
            window=self.form,
            anchor="nw",
        )
        self.form.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            ),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(
                self.controls_window,
                width=event.width,
            ),
        )

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

        self.progress = ttk.Progressbar(
            runtime,
            maximum=100,
            variable=self.progress_value,
        )
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
        self.output = tk.Text(log_frame, wrap="none", height=8)
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
        ttk.Label(footer, text="Status:").pack(side="left", padx=(14, 4))
        ttk.Progressbar(
            footer,
            maximum=100,
            variable=self.progress_value,
            length=170,
        ).pack(side="left", padx=(0, 6))
        ttk.Label(
            footer,
            textvariable=self.progress_text,
        ).pack(side="left")
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
            text="Texture preview",
            command=self.preview_selected,
        ).pack(side="left", padx=(5, 0))
        ttk.Button(
            footer,
            text="Render selected preview",
            command=self.render_selected_preview,
        ).pack(side="left", padx=(5, 0))

        current = str(self.state.get("current") or "train")
        if current not in BY_ID:
            current = "train"
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
            self._label("Model", "V17.1 local-ensemble implicit physical-map reconstruction")
            self._label(
                "Architecture",
                (
                    "128 LR physical maps -> LR encoder + analytic gradients -> four surrounding "
                    "LR anchor queries using relative dx/dy -> bilinear local-ensemble blend -> 512 HR residual -> B + residual."
                ),
            )
            self._label(
                "Architecture gate",
                (
                    "Train only crop A until it passes the unchanged candidate fit gates. Only then "
                    "evaluate crop B from the same authority. The sibling A/B/C image must "
                    "visibly restore thin manufactured features rather than merely sharpen B."
                ),
            )
            self._row("Authority", "authority_id", "13006d2b807f89ac")
            self._row("HR target size", "hr_size", "512", ("512",))
            self._row(
                "Checkpoint updates",
                "stages",
                "64,128,256,384",
                ("64,128,256,384", "64,128,256", "32,64,128"),
            )
            self._row("Learning rate", "learning_rate", "0.0002")
            self._row("Minimum target residual", "minimum_target_residual", "0.01")
            self._row("Encoder channels", "encoder_channels", "96", ("64", "96", "128"))
            self._row(
                "Neighbourhood channels",
                "neighbourhood_channels",
                "128",
                ("96", "128", "160"),
            )
            self._row(
                "Decoder hidden channels",
                "decoder_hidden_channels",
                "192",
                ("128", "192", "256"),
            )
            self._row("Device", "device", "cuda", ("cuda", "auto", "cpu"))
            self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
            self._row(
                "Manifest override",
                "manifest",
                "",
            )
            self._label(
                "Broad training",
                "BLOCKED. Do not run historical Main V16 or a broad V17 job until this sibling visual gate passes.",
            )

            latest = self._latest_v17_proof_report()
            if latest is not None:
                _path, report = latest
                snapshots = list(report.get("snapshots") or [])
                final = snapshots[-1] if snapshots else {}
                trained = dict(final.get("trained") or {})
                trained_metrics = dict(trained.get("metrics") or {})
                trained_pass = bool(report.get("trainedCandidateGatePass"))
                self._label(
                    "Latest trained result",
                    (
                        f"global {float(trained_metrics.get('global_recovery') or 0.0)*100:.2f}% | "
                        f"edge {float(trained_metrics.get('edge_recovery') or 0.0)*100:.2f}% | "
                        f"gradient {float(trained_metrics.get('gradient_recovery') or 0.0)*100:.2f}% | "
                        f"1px {float(trained_metrics.get('detail_recovery_1px') or 0.0)*100:.2f}% | "
                        f"fit gate {'PASS' if trained_pass else 'FAIL'}"
                    ),
                )
                if bool(report.get("siblingEvaluated")):
                    sibling = dict(final.get("sibling") or {})
                    metrics = dict(sibling.get("metrics") or {})
                    early = dict(report.get("finalSiblingEarlyTransferGate") or {})
                    self._label(
                        "Latest sibling result",
                        (
                            f"global {float(metrics.get('global_recovery') or 0.0)*100:.2f}% | "
                            f"edge {float(metrics.get('edge_recovery') or 0.0)*100:.2f}% | "
                            f"gradient {float(metrics.get('gradient_recovery') or 0.0)*100:.2f}% | "
                            f"1px {float(metrics.get('detail_recovery_1px') or 0.0)*100:.2f}% | "
                            f"early gate {'PASS' if early and all(early.values()) else 'FAIL'}"
                        ),
                    )
                else:
                    self._label(
                        "Sibling status",
                        "Not evaluated: trained crop did not pass the architecture fit gate.",
                    )
            ttk.Button(
                self.form,
                text="Open latest unseen sibling A/B/C",
                command=self._open_latest_v17_sibling_preview,
            ).pack(anchor="w", pady=(8, 2))
            ttk.Button(
                self.form,
                text="Open latest trained-crop A/B/C",
                command=self._open_latest_v17_trained_preview,
            ).pack(anchor="w", pady=2)
            ttk.Button(
                self.form,
                text="Open latest V17 proof folder",
                command=self._open_latest_v17_run,
            ).pack(anchor="w", pady=2)
            ttk.Button(
                self.form,
                text="Open historical V16 advanced diagnostics",
                command=self._open_advanced_gui,
            ).pack(anchor="w", pady=2)

        elif stage_id == "preview":
            choices = self._preview_choices()
            self._row(
                "Preview source",
                "experiment",
                choices[0] if choices else "<none>",
                choices or ["<none>"],
            )
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Target size", "target", "4096", ("1024", "2048", "4096"))
            self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._label(
                "Preview behavior",
                (
                    "V17_SIBLING_LATEST opens the current architecture-gate A/B/C sheet. "
                    "Native rendering is not yet enabled for V17 proof checkpoints. "
                    "MAIN_V16_LATEST and EXP sources remain historical preview paths."
                ),
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
        choices = self._preview_choices()
        self.preview_combo.configure(values=choices)
        current = self.preview_target.get().strip().upper()
        if current not in choices:
            self.preview_target.set(choices[0] if choices else "")

    def _refresh_scope(self) -> None:
        latest = self._latest_qualified()
        if latest is None:
            self.scope_text.set("Active architecture: V17 | Qualified production final: none")
            return
        experiment = _read_json(EXPERIMENT_ROOT / latest / "experiment.json") or {}
        mode = str(experiment.get("trainingMode") or "unknown").upper()
        self.scope_text.set(
            f"Active architecture: V17 | Qualified production final: {latest} | {mode} training"
        )

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
            latest_v17 = self._latest_v17_proof_report()
            statuses["train"] = "visual-review" if latest_v17 is not None else "ready"
        if self.active_stage != "preview":
            research_preview = (
                self._latest_v17_preview() is not None
                or self._latest_main_preview() is not None
                or any(
                    self._experiment_live_preview(experiment_id) is not None
                    for experiment_id in self._experiment_ids()
                )
            )
            statuses["preview"] = (
                "completed" if preview_done else
                "ready" if research_preview or self._experiment_ids() else
                "pending"
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
        if stage_id == "preview":
            target = self._value("experiment", "")
            if target and target != "<none>":
                self.preview_target.set(target)
            self.render_selected_preview()
            return
        lock = self._stage_lock_reason(stage_id)
        if lock:
            messagebox.showinfo("NSAMDR", lock)
            return
        command = self._command_for_stage(stage_id)
        if command is None:
            return
        self._start_process(stage_id, command)

    def preview_selected(self) -> None:
        target = self.preview_target.get().strip().upper()
        if not target:
            messagebox.showinfo("NSAMDR", "No preview source is selected.")
            return

        if target == V17_PREVIEW_CHOICE:
            preview = self._latest_v17_preview(role="sibling")
            if preview is None:
                messagebox.showinfo(
                    "NSAMDR",
                    "No V17 unseen-sibling preview exists yet. Run the V17 Sibling Architecture Proof first.",
                )
                return
            _open_path(preview)
            return

        if target == MAIN_PREVIEW_CHOICE:
            preview = self._latest_main_preview()
            if preview is None:
                messagebox.showinfo(
                    "NSAMDR",
                    (
                        "No Main V16 epoch preview exists yet. "
                        "During Main V16 Training it becomes available after the first "
                        "596-crop corpus epoch completes."
                    ),
                )
                return
            _open_path(preview)
            return

        experiment_dir = EXPERIMENT_ROOT / target
        if not experiment_dir.is_dir():
            messagebox.showinfo("NSAMDR", f"Experiment is missing: {target}")
            return

        if _qualified_final(experiment_dir):
            self.tree.selection_set("preview")
            self._selected()
            variable = self.vars.get("experiment")
            if variable is not None:
                variable.set(target)
            command = self._command_for_stage("preview")
            if command is not None:
                self._start_process("preview", command)
            return

        preview = self._experiment_live_preview(target)
        if preview is None:
            messagebox.showinfo(
                "NSAMDR",
                (
                    f"{target} has no training preview yet. Raven Quick publishes "
                    "A/B/C/F immediately after each completed SR epoch."
                ),
            )
            return
        _open_path(preview)

    def render_selected_preview(self) -> None:
        target = self.preview_target.get().strip().upper()
        if not target:
            messagebox.showinfo("NSAMDR", "No preview source is selected.")
            return

        if target == V17_PREVIEW_CHOICE:
            messagebox.showinfo(
                "NSAMDR",
                (
                    "V17 is currently at the texture architecture gate. "
                    "Inspect the unseen sibling A/B/C sheet first; the native EVE renderer "
                    "will be wired only after this architecture passes."
                ),
            )
            return

        if target.startswith("EXP_") and _qualified_final(EXPERIMENT_ROOT / target):
            command = self._cli_argv(
                "preview",
                target,
                "--shared-cache",
                self._value("cache", r"C:\CCP\EVE"),
                "--target-size",
                self._value("target", "4096"),
                "--device",
                self._value("device", "cuda"),
            )
        else:
            command = self._cli_argv(
                "render-preview",
                target,
                "--shared-cache",
                self._value("cache", r"C:\CCP\EVE"),
                "--target-size",
                "1024",
                "--device",
                "cuda",
                "--watch",
            )

        log_dir = REPO_ROOT / "artifacts/nsamdr/render_preview" / target
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "launcher.log"
        try:
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            handle.close()
        except Exception as exc:
            messagebox.showerror("NSAMDR render preview", str(exc))
            return

        self.output.insert(
            "end",
            f"[GUI] Native render preview launched: {target} PID {process.pid}\n"
            f"[GUI] Render log: {log_path}\n",
        )
        self.output.see("end")

    def _start_process(self, stage_id: str, command: list[str]) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("NSAMDR", "A workflow process is already running.")
            return
        self.output.delete("1.0", "end")
        self.progress_value.set(0.0)
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
        v17 = V17_UPDATE_RE.search(line)
        if v17:
            current = int(v17.group(1))
            maximum = int(v17.group(2))
            if maximum > 0:
                percent = 100.0 * current / maximum
                self.progress_value.set(max(0.0, min(100.0, percent)))
                elapsed = (
                    time.monotonic() - self.process_started_at
                    if self.process_started_at is not None
                    else 0.0
                )
                self.progress_text.set(
                    f"V17 sibling proof {current}/{maximum} ({percent:.1f}%) | "
                    f"elapsed {_format_duration(elapsed)}"
                )
                return

        broad = FULL_BROAD_STEP_RE.search(line)
        if broad:
            current = int(broad.group(1))
            maximum = int(broad.group(2))
            if maximum > 0:
                percent = 100.0 * current / maximum
                self.progress_value.set(max(0.0, min(100.0, percent)))
                elapsed = (
                    time.monotonic() - self.process_started_at
                    if self.process_started_at is not None
                    else 0.0
                )
                self.progress_text.set(
                    f"Historical Main V16 {current}/{maximum} ({percent:.1f}%) | "
                    f"elapsed {_format_duration(elapsed)}"
                )
                return

        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            current = int(progress_match.group(1))
            maximum = int(progress_match.group(2))
            if maximum > 0:
                percent = 100.0 * current / maximum
                self.progress_value.set(max(0.0, min(100.0, percent)))
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
                self.progress_value.set(max(0.0, min(100.0, percent)))
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
                    self.progress_value.set(100.0)

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
                        elif stage_id == "train" and code == 0:
                            self._set_status(stage_id, "visual-review")
                        else:
                            self._set_status(stage_id, "completed" if code == 0 else "failed")

                    if rejected:
                        self.progress_text.set(
                            f"Training completed; {rejected_experiment} was rejected by candidate qualification"
                        )
                    elif stage_id == "train" and code == 0:
                        self.progress_text.set(
                            "V17 sibling proof complete; visual review of unseen sibling A/B/C is required"
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

    def _open_latest_v17_sibling_preview(self) -> None:
        preview = self._latest_v17_preview(role="sibling")
        if preview is None:
            messagebox.showinfo(
                "NSAMDR",
                "No V17 unseen-sibling A/B/C preview is available yet.",
            )
            return
        _open_path(preview)

    def _open_latest_v17_trained_preview(self) -> None:
        preview = self._latest_v17_preview(role="trained")
        if preview is None:
            messagebox.showinfo(
                "NSAMDR",
                "No V17 trained-crop A/B/C preview is available yet.",
            )
            return
        _open_path(preview)

    def _open_latest_v17_run(self) -> None:
        run_dir = self._latest_v17_proof_dir()
        if run_dir is None or not run_dir.is_dir():
            messagebox.showinfo("NSAMDR", "No V17 sibling proof run is available yet.")
            return
        _open_path(run_dir)

    def _open_latest_main_preview(self) -> None:
        preview = self._latest_main_preview()
        if preview is None:
            messagebox.showinfo(
                "NSAMDR",
                "No Main V16 research preview is available yet.",
            )
            return
        _open_path(preview)

    def _open_latest_main_run(self) -> None:
        pointer = self._main_training_pointer() or {}
        raw = str(pointer.get("runDirectory") or "").strip()
        if not raw:
            messagebox.showinfo("NSAMDR", "No Main V16 research run is available yet.")
            return
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_dir():
            messagebox.showinfo("NSAMDR", f"Main V16 run folder is missing:\n{path}")
            return
        _open_path(path)

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
