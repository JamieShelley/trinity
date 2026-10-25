#!/usr/bin/env python3
"""Canonical NSAMDR V17.6 operator GUI.

Restores the rich stage-tree workflow UI: stage forms, runtime log,
live progress, stop, source/report detection, and native Raven previews.
Only V17.6 experiments are executable. Historical proofs remain read-only
sources for authenticated frozen weights; never exposed as train commands.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "tools/nsamdr/nsamdr_cli.py"
GUI_STATE_ROOT = REPO_ROOT / "artifacts/nsamdr/gui"
STATE_PATH = GUI_STATE_ROOT / "v176_operator_workflow_state.json"
DIAGNOSTIC_ROOT = REPO_ROOT / "artifacts/nsamdr/diagnostics"
CAPACITY_ROOT = DIAGNOSTIC_ROOT / "v176_stage10a1_capacity"
DETAIL_ROOT = DIAGNOSTIC_ROOT / "v176_stage10a_detail"
PREVIEW_ROOT = REPO_ROOT / "artifacts/nsamdr/render_preview"
APP_TITLE = "NSAMDR — V17.7 Baseline / V17.6 Neural Research"
STATE_SCHEMA = "nsamdr-v176-operator-workflow-v2"
PHASES = ("diagnose", "stability", "capacity", "compare-features", "attainable")
PREVIEW_CHOICES = (
    "TRAIN10_OUTPUT10", "TRAIN10_HR_FEATURES58",
    "TRAIN01_OUTPUT10", "TRAIN01_HR_FEATURES58",
)
UPDATE_RE = re.compile(r"\[v176-(?:capacity|detail)\](?:\s+update)?\s*(?:train_\d+_\w+\s+)?(\d+)/(\d+)")
ATTAINABLE_RE = re.compile(r"\[v176-attainable\]\s+authority=(\d+)\s+probe=(\w+)\s+objective=(\w+)\s+update=(\d+)/(\d+)")

@dataclass(frozen=True)
class Stage:
    id: str
    number: str
    label: str
    description: str

STAGES = (
    Stage("setup", "P0", "Environment / CUDA", "Manage Python environment and verify CUDA capability."),
    Stage("train", "10A", "Dense specialist training", "Train V17.6 detail specialist with frozen Stage 9 backbone."),
    Stage("compare", "10A.1", "Capacity and features", "Diagnose, run stability, train capacity, or compare 10/58-channel specialists."),
    Stage("attainable", "10A.2", "Attainability diagnostic", "Target-assisted training-only pixel fit. Does not update neural weights."),
    Stage("enhance", "11", "V17.7 native enhancement", "Train a bounded RGB/normal/material correction over B using clean synthetic geometry."),
    Stage("golden", "G0–G1", "Raven Golden Patch", "Capture verified native Raven pixels and independently trace clean structural contours."),
    Stage("contour", "G2", "Structural contour proof", "Train structural field from synthetic clean geometry; hold out real Raven until independently traced."),
    Stage("preview", "VIS", "Raven 3D A/B/C0/C1 preview", "Compare authored A, semantic B, frozen C0, and trained V17.6 C1 on the Raven."),
    Stage("evidence", "REP", "Reports / provenance", "Inspect completed V17.6 proof and native render artifacts."),
    Stage("tests", "T", "Regression tests", "Run the current supported V17.6 verification suite."),
    Stage("quadtree", "Q", "Adaptive quadtree", "Planned extension; not implemented in this branch. Dense model first."),
)
BY_ID = {s.id: s for s in STAGES}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def latest_report(folder: Path, phase: str = "") -> Path | None:
    options = []
    for path in folder.glob("proof_*/report.json"):
        data = _read_json(path)
        if data is None or (phase and data.get("phase") != phase):
            continue
        options.append(path)
    return max(options, key=lambda path: path.stat().st_mtime) if options else None


def training_command(phase: str, resume: str, audit: bool, amp: str) -> list[str]:
    if phase not in PHASES or resume not in ("fresh", "auto") or amp not in ("auto", "bf16", "fp16"):
        raise ValueError("invalid V17.6 training selection")
    return ["v176-capacity-proof", "--phase", phase, "--resume", resume,
            "--amp-precision", amp] + (["--audit-only"] if audit else [])


def raven_command(cache: str, authority: int, mode: str) -> list[str]:
    if authority not in (1, 10) or mode not in ("output10", "hr_features58"):
        raise ValueError("invalid authenticated V17.6 training arm")
    return ["v176-raven-preview", "--shared-cache", cache,
            "--authority-index", str(authority), "--mode", mode]


def preview_label(authority: int, mode: str) -> str:
    if authority not in (1, 10) or mode not in ("output10", "hr_features58"):
        raise ValueError("invalid V17.6 preview selection")
    return f"V176_RAVEN_TRAIN{authority:02d}_{mode.upper()}"


def _open_path(path: Path) -> None:
    try:
        target = path.resolve()
        if not target.exists():
            raise FileNotFoundError(target)
        if sys.platform == "win32":
            os.startfile(str(target))
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except (OSError, ValueError) as exc:
        messagebox.showerror("NSAMDR", str(exc))


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
        self.state = self._load_state()
        self.vars: dict[str, tk.Variable] = {}
        self.description = tk.StringVar()
        self.command_preview = tk.StringVar()
        self.progress_text = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0.0)
        self.scope_text = tk.StringVar(value="V17.7 baseline · V17.6 neural checkpoints · reading evidence")
        self.recipe_text = tk.StringVar(value=(
            "Current: dense V17.6 scalar-albedo detail specialist. "
            "Frozen Stage 9 backbone; original physical caps and training-only supervision."
        ))
        self.preview_target = tk.StringVar(value="TRAIN10_OUTPUT10")
        self._build()
        self.detect(silent=True)
        self.root.after(100, self._poll)

    def _load_state(self) -> dict[str, Any]:
        default = {"schema": STATE_SCHEMA, "current": "preview",
                   "status": {s.id: "pending" for s in STAGES},
                   "forms": {}}
        loaded = _read_json(STATE_PATH)
        if loaded is not None and loaded.get("schema") == STATE_SCHEMA:
            default.update(loaded)
        for key, value in list(default["status"].items()):
            if value in ("running", "stopping"):
                default["status"][key] = "interrupted"
        return default

    def _save_state(self) -> None:
        GUI_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")

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
            text="Open diagnostics",
            command=self._open_diagnostics,
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
            text="Open preview files",
            command=self.preview_selected,
        ).pack(side="left", padx=(5, 0))
        ttk.Button(
            footer,
            text="Render Raven (DX11)",
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

    def _form_default(self, key: str, default: str) -> str:
        return str((self.state.get("forms", {}).get(self.state.get("current"), {}) or {}).get(key, default))

    def _form_changed(self) -> None:
        stage = self.state.get("current", "preview")
        saved = self.state.setdefault("forms", {}).setdefault(stage, {})
        for key, variable in self.vars.items():
            saved[key] = variable.get()
        if stage == "preview" and "authority" in self.vars and "mode" in self.vars:
            try:
                candidate = f"TRAIN{int(self.vars['authority'].get()):02d}_{str(self.vars['mode'].get()).upper()}"
                if candidate in PREVIEW_CHOICES:
                    self.preview_target.set(candidate)
            except (ValueError, TypeError):
                pass
        self._save_state()
        self._update_command_preview()

    def _row(self, label: str, key: str, default: str, choices=None, *, editable=False) -> None:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=26).pack(side="left")
        variable = tk.StringVar(value=self._form_default(key, default))
        self.vars[key] = variable
        if choices is None:
            widget = ttk.Entry(row, textvariable=variable)
        else:
            widget = ttk.Combobox(
                row, textvariable=variable, values=tuple(choices),
                state="normal" if editable else "readonly",
            )
        widget.pack(side="left", fill="x", expand=True)
        variable.trace_add("write", lambda *_: self._form_changed())

    def _check(self, label: str, key: str, default: bool = False) -> None:
        variable = tk.BooleanVar(value=bool((self.state.get("forms", {}).get(
            self.state.get("current"), {}) or {}).get(key, default)))
        self.vars[key] = variable
        ttk.Checkbutton(self.form, text=label, variable=variable,
                        command=self._form_changed).pack(anchor="w", pady=3)

    def _label(self, label: str, value: str) -> None:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=26).pack(side="left")
        ttk.Label(row, text=value, wraplength=690).pack(side="left", fill="x", expand=True)

    def _selected(self) -> None:
        choices = self.tree.selection()
        if not choices:
            return
        selected = str(choices[0])
        if selected not in BY_ID:
            return
        self.state["current"] = selected
        self._save_state()
        self._clear_form()
        stage = BY_ID[selected]
        self.description.set(f"{stage.number}. {stage.label} — {stage.description}")
        if selected == "setup":
            self._label("Environment", "Managed CUDA Python for training / Raven.")
            self._row("Setup target", "kind", "cuda", ("cuda", "cpu"))
            self._check("Recreate environment", "force", False)
            self._label("Current architecture", "V17.6 only. Older scripts are not operator commands.")
        elif selected == "train":
            self._label("Source", "Authenticated frozen A12 Stage 9 backbone and original Stage10A loss")
            self._row("Checkpoint mode", "resume", "fresh", ("fresh", "auto"))
            self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
            self._check("Audit only", "audit", False)
            self._row("Stage 9 report", "stage9_report", "")
            self._row("V3 source report", "v3_report", "")
        elif selected == "compare":
            self._label("Training authorities", "Frozen A12 indices 1 and 10 (no final-test access)")
            self._row("Phase", "phase", "compare-features", PHASES[:-1])
            self._row("Checkpoint mode", "resume", "fresh", ("fresh", "auto"))
            self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
            self._row("Device", "device", "cuda", ("cuda", "cpu"))
            self._check("Audit only (no updates)", "audit", False)
            self._row("Stage10A report", "stage10a_report", "")
            self._row("V3 source report", "v3_report", "")
        elif selected == "attainable":
            self._label("Purpose", "Target-assisted free-pixel feasibility diagnostic; no neural updates")
            self._row("Checkpoint mode", "resume", "fresh", ("fresh", "auto"))
            self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
            self._row("Device", "device", "cuda", ("cuda", "cpu"))
            self._check("Audit only", "audit", False)
            self._row("Stage10A report", "stage10a_report", "")
            self._row("Comparison report", "comparison_report", "")
            self._row("V3 source report", "v3_report", "")
        elif selected == "enhance":
            self._label("Goal", "Clean 4x synthetic line/curve supervision only; never real EVE holdout data")
            self._row("Updates", "steps", "300")
            self._row("Batch size", "batch_size", "4")
            self._row("Native patch width", "patch", "32")
            self._row("Seed", "seed", "178")
            self._row("Learning rate", "lr", "0.001")
            self._row("Device", "device", "cuda", ("cuda", "cpu"))
            self._row("Resume", "resume", "fresh", ("fresh", "auto"))
            self._label("Gate", "Synthetic validation improvement required; no real-Raven qualification")
        elif selected == "golden":
            self._label("Contract", "Native Raven 256×256 → 4× structural reference. No fabricated clean target.")
            self._row("EVE ResFiles cache", "cache", r"C:\CCP\EVE")
            self._row("Native ROI X", "x", "320")
            self._row("Native ROI Y", "y", "448")
            self._row("Native ROI size", "size", "256", ("128", "256", "384", "512"))
            self._row("Exact source filename", "source_name", "")
            self._label("Next action", "Prepare evidence, then trace panel centre-lines independently.")
            ttk.Button(self.form, text="Open latest Golden Patch / trace editor",
                       command=self.open_golden_editor).pack(anchor="w", pady=5)
            ttk.Button(self.form, text="Open latest golden evidence folder",
                       command=self.open_golden_folder).pack(anchor="w", pady=5)
        elif selected == "contour":
            self._label("Objective", "Clean 4× structural coverage: tangent, curvature, continuity. Synthetic gate only.")
            self._row("Updates", "steps", "400")
            self._row("Batch size", "batch_size", "4")
            self._row("Native patch width", "patch", "32")
            self._row("Seed", "seed", "17802")
            self._row("Learning rate", "lr", "0.0007")
            self._row("Device", "device", "cuda", ("cuda", "cpu"))
            self._row("Resume", "resume", "fresh", ("fresh", "auto"))
            self._label("Gate", "At least 20% relative contour recovery on unseen analytic geometry.")
        elif selected == "preview":
            self._label("Ship", "Caldari Raven — real EVE textures / DX11 native viewer")
            self._row("Training authority", "authority", "10", ("10", "1"))
            self._row("Specialist", "mode", "output10", ("output10", "hr_features58"))
            self._label("Output scale", "Native full-resolution source × 4 (no downsampling)")
            self._row("EVE ResFiles cache", "cache", r"C:\CCP\EVE")
            self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._row("Comparison report", "comparison_report", "")
            self._row("C1 source", "candidate_source", "v176", ("v176", "v177"))
            self._row("V17.7 checkpoint", "enhancer_checkpoint", "")
            self._label("Source", "V17.7 option requires an explicitly gated candidate.pt from V17.7 training")
            self._label("Qualification", "B = conservative semantic reference; C0 = frozen V17.2; C1 = selected V17.6 or synthetic-trained V17.7. Research only.")
            self._label("Comparison panes", "A / B / C0 / C1 in synchronized 2×2 native DX11 view.")
            ttk.Button(self.form, text="Open current preview files",
                       command=self.preview_selected).pack(anchor="w", pady=5)
            ttk.Button(self.form, text="Audit native texture channels (no training)",
                       command=self.audit_native_channels).pack(anchor="w", pady=5)
            ttk.Button(self.form, text="Open latest channel audit",
                       command=self.open_native_channel_audit).pack(anchor="w", pady=5)
            self._label("Channel audit",
                        "Exports nearest vs bilinear crops of native albedo, normals, "
                        "material, glow and roughness. It does not alter the model.")
        elif selected == "evidence":
            self._label("V17.6 report", str(latest_report(CAPACITY_ROOT) or "No completed capacity report"))
            self._label("Original detail", str(latest_report(DETAIL_ROOT) or "No completed detail report"))
            self._label("Frozen Stage 9", "Retained solely as immutable model/provenance source")
            ttk.Button(self.form, text="Open V17.6 report",
                       command=self.open_latest_report).pack(anchor="w", pady=3)
            ttk.Button(self.form, text="Open diagnostics folder",
                       command=self._open_diagnostics).pack(anchor="w", pady=3)
            ttk.Button(self.form, text="Open Raven preview folder",
                       command=lambda: _open_path(PREVIEW_ROOT)).pack(anchor="w", pady=3)
            ttk.Button(self.form, text="Recheck artifacts",
                       command=self.detect).pack(anchor="w", pady=3)
        elif selected == "tests":
            self._label("Suite", "Active V17.6 CLI/GUI, specialist, stability, feasibility, and renderer tests")
            self._label("Command", "scripts\\build\\nsamdr.bat test")
        elif selected == "quadtree":
            self._label("Status", "Not implemented in the checked-in V17.6 dense architecture")
            self._label("Gate", "Prove dense fit, full-density equivalence and seam handling first")
            self._label("Execution", "No quadtree training action exists; never mislabel dense preview")
        if selected == "preview":
            try:
                candidate = f"TRAIN{int(self._value('authority', '10')):02d}_{self._value('mode', 'output10').upper()}"
                if candidate in PREVIEW_CHOICES:
                    self.preview_target.set(candidate)
            except (ValueError, TypeError):
                pass
        self._update_command_preview()

    def _value(self, key: str, default: str = "") -> str:
        var = self.vars.get(key)
        return str(var.get()) if var is not None else default

    def _checked(self, key: str) -> bool:
        variable = self.vars.get(key)
        return bool(variable.get()) if variable is not None else False

    def _stage_lock_reason(self, stage: str) -> str | None:
        return "Quadtree routing not yet implemented" if stage == "quadtree" else None

    def _command_for_stage(self, stage_id: str) -> list[str] | None:
        command = [sys.executable, "-u", str(CLI)]
        if stage_id == "setup":
            args = ["setup", self._value("kind", "cuda")]
            if self._checked("force"):
                args.append("--force")
        elif stage_id == "train":
            args = ["v176-detail-proof", "--resume", self._value("resume", "fresh"),
                    "--amp-precision", self._value("amp", "auto")]
            if self._checked("audit"):
                args.append("--audit-only")
            for key, flag in (("stage9_report", "--stage9-report"),
                              ("v3_report", "--v3-report")):
                if self._value(key).strip():
                    args.extend((flag, self._value(key).strip()))
        elif stage_id in ("compare", "attainable"):
            phase = ("attainable" if stage_id == "attainable"
                     else self._value("phase", "compare-features"))
            args = training_command(phase, self._value("resume", "fresh"),
                                    self._checked("audit"),
                                    self._value("amp", "auto"))
            args += ["--device", self._value("device", "cuda")]
            for key, flag in (("stage10a_report", "--stage10a-report"),
                              ("comparison_report", "--comparison-report"),
                              ("v3_report", "--v3-report")):
                if self._value(key).strip():
                    args.extend((flag, self._value(key).strip()))
        elif stage_id == "enhance":
            args = ["v177-enhancement-train", "--steps", self._value("steps", "300"),
                    "--batch-size", self._value("batch_size", "4"),
                    "--patch-size", self._value("patch", "32"),
                    "--seed", self._value("seed", "178"),
                    "--learning-rate", self._value("lr", "0.001"),
                    "--device", self._value("device", "cuda"),
                    "--resume", self._value("resume", "fresh")]
        elif stage_id == "golden":
            args = [
                "v178-golden-patch", "--action", "prepare",
                "--shared-cache", self._value("cache", r"C:\CCP\EVE"),
                "--crop-x", self._value("x", "320"),
                "--crop-y", self._value("y", "448"),
                "--size", self._value("size", "256"),
            ]
            if self._value("source_name").strip():
                args.extend(("--source-name",self._value("source_name").strip()))
        elif stage_id == "contour":
            args = [
                "v178-contour-train", "--steps", self._value("steps", "400"),
                "--batch-size", self._value("batch_size", "4"),
                "--patch-size", self._value("patch", "32"),
                "--seed", self._value("seed", "17802"),
                "--learning-rate", self._value("lr", "0.0007"),
                "--device", self._value("device", "cuda"),
                "--resume", self._value("resume", "fresh"),
            ]
        elif stage_id == "preview":
            args = raven_command(self._value("cache", r"C:\CCP\EVE"),
                                 int(self._value("authority", "10")),
                                 self._value("mode", "output10"))
            args.extend(("--device", self._value("device", "cuda")))
            if self._value("comparison_report").strip():
                args.extend(("--comparison-report", self._value("comparison_report").strip()))
            if self._value("candidate_source", "v176") == "v177":
                checkpoint = self._value("enhancer_checkpoint").strip()
                if not checkpoint:
                    raise ValueError("Select the validated V17.7 candidate.pt checkpoint")
                args.extend(("--enhancer-checkpoint", checkpoint))
        elif stage_id == "tests":
            args = ["test"]
        else:
            return None
        return command + args

    def _update_command_preview(self) -> None:
        selected = self.tree.selection()
        stage = str(selected[0]) if selected else ""
        try:
            command = self._command_for_stage(stage)
            self.command_preview.set(
                subprocess.list2cmdline(command) if command else "[read-only stage]"
            )
        except (ValueError, TypeError) as exc:
            self.command_preview.set(f"[invalid options] {exc}")
        self._update_run_state()

    def _update_run_state(self) -> None:
        selected = self.tree.selection()
        stage = str(selected[0]) if selected else ""
        busy = self.process is not None and self.process.poll() is None
        try:
            command = self._command_for_stage(stage)
        except (ValueError, TypeError):
            command = None
        self.run_button.configure(
            state=("disabled" if busy or command is None else "normal")
        )

    def _refresh_preview_choices(self) -> None:
        self.preview_combo.configure(values=PREVIEW_CHOICES)
        if self.preview_target.get() not in PREVIEW_CHOICES:
            self.preview_target.set(PREVIEW_CHOICES[0])

    def _open_diagnostics(self) -> None:
        if not DIAGNOSTIC_ROOT.exists():
            messagebox.showinfo("NSAMDR", "No diagnostic artifacts found.")
            return
        _open_path(DIAGNOSTIC_ROOT)

    def _refresh_scope(self) -> None:
        report = latest_report(CAPACITY_ROOT, "compare-features")
        decision = (_read_json(report) or {}).get("decision", "no report") if report else "no report"
        self.scope_text.set(f"Preview: V17.7 semantic B + frozen C0 + trained V17.6 C1 | Latest comparison: {decision} | Production: unqualified")

    def detect(self, silent: bool = False) -> None:
        lookup = {
            "train": latest_report(DETAIL_ROOT),
            "compare": latest_report(CAPACITY_ROOT, "compare-features"),
            "attainable": latest_report(CAPACITY_ROOT, "attainable"),
        }
        self._refresh_scope()
        for stage in STAGES:
            if stage.id in lookup:
                row = lookup[stage.id]
                self.state["status"][stage.id] = (
                    "completed" if row and (_read_json(row) or {}).get("decision") not in
                    ("RUNNING", "PARENT_PROVENANCE_VERIFIED_ONLY") else
                    "audit only" if row else "pending"
                )
            elif stage.id == "golden":
                self.state["status"][stage.id] = (
                    "reference pending" if self._latest_golden() is not None
                    else "pending"
                )
            elif stage.id == "contour":
                proofs = DIAGNOSTIC_ROOT / "v178_golden_contour"
                self.state["status"][stage.id] = (
                    "report available" if any(proofs.glob("seed_*/report.json"))
                    else "pending"
                )
            elif stage.id == "enhance":
                synthetic = DIAGNOSTIC_ROOT / "v177_native_enhancement"
                self.state["status"][stage.id] = (
                    "report available" if any(synthetic.glob("seed_*/report.json"))
                    else "pending"
                )
            elif stage.id == "preview":
                self.state["status"][stage.id] = (
                    "rendered" if any(PREVIEW_ROOT.glob("V176_RAVEN_*/current_candidate.txt"))
                    else "pending"
                )
            elif stage.id == "quadtree":
                self.state["status"][stage.id] = "not implemented"
            else:
                self.state["status"].setdefault(stage.id, "pending")
            if self.tree.exists(stage.id):
                self.tree.set(stage.id, "status", self.state["status"][stage.id])
        self._save_state()
        if not silent:
            self.progress_text.set("Artifact detection completed")
            self._selected()

    def _set_status(self, stage: str, value: str) -> None:
        self.state["status"][stage] = value
        if self.tree.exists(stage):
            self.tree.set(stage, "status", value)
        self._save_state()

    def run_selected(self) -> None:
        selected = self.tree.selection()
        stage = str(selected[0]) if selected else ""
        if not stage or self._stage_lock_reason(stage):
            return
        if stage == "evidence":
            self.open_latest_report()
            return
        try:
            command = self._command_for_stage(stage)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("NSAMDR options", str(exc))
            return
        if command is None:
            return
        self._start_process(stage, command)

    def _preview_id(self) -> tuple[int, str]:
        choice = self.preview_target.get()
        if choice not in PREVIEW_CHOICES:
            raise ValueError("unsupported V17.6 preview")
        authority = int(choice[5:7])
        mode = choice.split("_", 1)[1].lower()
        return authority, mode

    def preview_selected(self) -> None:
        try:
            authority, mode = self._preview_id()
            path = PREVIEW_ROOT / preview_label(authority, mode)
            if not path.is_dir():
                messagebox.showinfo("NSAMDR", "No native V17.6 preview exists for this training arm.")
                return
            _open_path(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("NSAMDR", str(exc))

    def render_selected_preview(self) -> None:
        if not (self.process is None or self.process.poll() is not None):
            messagebox.showinfo("NSAMDR", "A process is already running.")
            return
        try:
            authority, mode = self._preview_id()
            self.tree.selection_set("preview")
            self._selected()
            self.vars["authority"].set(str(authority))
            self.vars["mode"].set(mode)
            command = self._command_for_stage("preview")
            if command is not None:
                self._start_process("preview", command)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("NSAMDR preview", str(exc))

    def audit_native_channels(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("NSAMDR", "Another operator process is running.")
            return
        command = [sys.executable, "-u", str(CLI),
                   "v176-raven-channel-audit", "--shared-cache",
                   self._value("cache", r"C:\CCP\EVE")]
        self._start_process("preview", command)

    def open_native_channel_audit(self) -> None:
        base = DIAGNOSTIC_ROOT / "v176_raven_channel_audit"
        reports = sorted(base.glob("audit_*/report.json"),
                         key=lambda item: item.stat().st_mtime, reverse=True)
        if not reports:
            messagebox.showinfo("NSAMDR", "No Raven channel audit exists yet.")
            return
        _open_path(reports[0].parent)

    def _latest_golden(self) -> Path | None:
        root = REPO_ROOT / "artifacts/nsamdr/golden_patch"
        reports = list(root.glob("*/evidence.json"))
        return max(reports,key=lambda p:p.stat().st_mtime).parent if reports else None

    def open_golden_folder(self) -> None:
        folder = self._latest_golden()
        if folder is None:
            messagebox.showinfo("Golden Patch", "Run Golden Patch preparation first.")
            return
        _open_path(folder)

    def open_golden_editor(self) -> None:
        folder = self._latest_golden()
        if folder is None:
            messagebox.showinfo("Golden Patch", "Prepare the Raven native patch first.")
            return
        from tools.nsamdr.gui.golden_trace_editor import open_trace_editor
        try:
            open_trace_editor(self.root,folder)
        except (OSError,ValueError,RuntimeError) as exc:
            messagebox.showerror("Golden Patch",str(exc))

    def open_latest_report(self) -> None:
        report = latest_report(CAPACITY_ROOT) or latest_report(DETAIL_ROOT)
        if report is None:
            messagebox.showinfo("NSAMDR", "No V17.6 report has been completed.")
        else:
            _open_path(report)

    def _start_process(self, stage: str, command: list[str]) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("NSAMDR", "A workflow process is already running.")
            return
        self.output.insert("end", "\n> " + subprocess.list2cmdline(command) + "\n")
        self.output.see("end")
        self.progress_value.set(0.0)
        self.progress_text.set(f"{BY_ID[stage].label}: starting")
        try:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace",
            )
        except OSError as exc:
            messagebox.showerror("NSAMDR", str(exc))
            return
        self.process = process
        self.active_stage = stage
        self._set_status(stage, "running")
        self._update_run_state()

        def reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self.process_queue.put(("line", line))
            self.process_queue.put(("done", (stage, process.wait())))

        self.process_thread = threading.Thread(target=reader, daemon=True)
        self.process_thread.start()

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        process = self.process
        if messagebox.askyesno("Stop NSAMDR", "Terminate the active command and its subprocesses?"):
            self._set_status(self.active_stage or "train", "stopping")
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False)
            else:
                process.terminate()

    def _update_progress_from_line(self, line: str) -> None:
        found = UPDATE_RE.search(line)
        if found:
            done, total = int(found.group(1)), int(found.group(2))
            if total > 0:
                self.progress_value.set(100.0 * done / total)
                self.progress_text.set(f"Update {done}/{total}")
            return
        extra = ATTAINABLE_RE.search(line)
        if extra:
            authority, probe, objective, done, total = extra.groups()
            if int(total) > 0:
                self.progress_value.set(100.0 * int(done) / int(total))
            self.progress_text.set(
                f"Authority {authority} {probe} {objective}: {done}/{total}"
            )

    def _poll(self) -> None:
        try:
            while True:
                kind, data = self.process_queue.get_nowait()
                if kind == "line":
                    line = str(data)
                    self.output.insert("end", line)
                    self.output.see("end")
                    self._update_progress_from_line(line)
                elif kind == "done":
                    stage, code = data
                    self._set_status(stage, "completed" if code == 0 else f"failed ({code})")
                    self.progress_text.set(f"{BY_ID[stage].label}: exit {code}")
                    if code == 0:
                        self.progress_value.set(100.0)
                    self.process = None
                    self.active_stage = None
                    self._update_run_state()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
