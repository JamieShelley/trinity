#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

APP_TITLE = "NSAMDR V9 Workflow Controller 4.9.3"
STATE_SCHEMA = "nsamdr-v9-workflow-gui-v5"
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)\s+phase=([^\s]+)", re.I)
BATCH_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+total=", re.I)
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)tile/s\b", re.I)
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


def _live_eta(
    epoch_index: int,
    epoch_total: int,
    batch_index: int,
    batch_total: int,
    rate_tiles_per_second: float,
) -> tuple[float, float, float]:
    """Return progress %, this-epoch remaining time and full-run ETA."""
    if (
        epoch_index < 1
        or epoch_total < 1
        or batch_index < 0
        or batch_total < 1
        or rate_tiles_per_second <= 0.0
    ):
        return 0.0, math.inf, math.inf
    epoch_index = min(epoch_index, epoch_total)
    batch_index = min(max(batch_index, 0), batch_total)
    completed_tiles = (epoch_index - 1) * batch_total + batch_index
    total_tiles = epoch_total * batch_total
    remaining_epoch_tiles = max(0, batch_total - batch_index)
    remaining_total_tiles = max(0, total_tiles - completed_tiles)
    progress = 100.0 * completed_tiles / max(1, total_tiles)
    return (
        progress,
        remaining_epoch_tiles / rate_tiles_per_second,
        remaining_total_tiles / rate_tiles_per_second,
    )


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
    Stage("validate", "0", "Pipeline validation", ("validate",), "Verify source layout, fidelity contract and CUDA architecture."),
    Stage("tune", "1", "Raven tune + instrumented preview", ("tune",), "Build/reuse the fixed Raven set, train V9.8 coarse-to-fine geometry fields, run the G0-G5 renderer/SDF/gate proof first, and launch the Raven candidate/preview only after synthetic PASS."),
    Stage("tune_compare", "C1", "Compare tuning experiments", ("compare",), "Open deterministic linked-pan/zoom comparisons for two or three completed Raven experiments.", False),
    Stage("tune_promote", "2", "Promote best configuration", ("promote",), "Promotion is locked during the V9.8 geometry-convergence boundary-only proof; later promotes an appearance-enabled Full proof.", False),
    Stage("index", "3", "Index full V9 dataset", ("index", "eve"), "Build the all-supported-assets dataset using the promoted configuration."),
    Stage("train", "4", "Train full production", ("train", "full"), "Run full production from the promoted experiment configuration."),
    Stage("preview", "5", "Preview full production", ("preview", "production"), "Unlock all-ship production preview after full training completes."),
    Stage("all", "U1", "Run promoted production pipeline", ("run",), "Validate/index/train/preview using the selected promoted experiment.", False),
)
BY_ID = {stage.id: stage for stage in STAGES}
PIPELINE = [stage.id for stage in STAGES if stage.pipeline]


PRESET_VALUES = {
    "Baseline": {
        "learning_rate": "0.0001", "weight_decay": "0.00001", "optimizer": "adamw", "scheduler": "phase",
        "batch_size": "1", "tiles_per_epoch": "320", "augmentation_strength": "1.0",
        "regret_weight": "2.50", "normal_regret_weight": "1.25", "edge_weight": "2.00",
        "detail_laplacian_weight": "0.28", "geometric_alignment_weight": "0.48",
        "tangent_coherence_weight": "0.36", "curvature_coherence_weight": "0.30",
        "synthetic_geometry_probability": "0.82", "boundary_sampling_probability": "0.68",
        "boundary_renderer_band_pixels": "3.5", "boundary_renderer_sample_pixels": "3.75",
        "boundary_renderer_hard_width_pixels": "0.70", "boundary_renderer_soft_width_pixels": "1.80",
        "boundary_renderer_gate_gain": "1.60", "boundary_renderer_far_sample_multiplier": "1.70", "boundary_renderer_far_sample_weight": "0.22", "boundary_gate_need_scale": "0.075", "boundary_gate_exact_floor": "0.35", "boundary_sdf_zero_weight": "3.00", "boundary_edge_sdf_consistency_weight": "1.50", "boundary_pixel_regret_weight": "3.00", "boundary_profile_weight": "1.65", "boundary_regret_weight": "5.00", "sdf_surface_weight": "8.00", "sdf_sign_weight": "2.00", "sdf_eikonal_weight": "8.00", "sdf_gradient_alignment_weight": "2.00", "sdf_metric_gradient_weight": "6.00", "sdf_metric_band_pixels": "12.0", "sdf_coarse_init_std": "0.0005", "sdf_synthetic_validation_tiles": "12", "sdf_zero_band_pixels": "0.50", "sdf_bootstrap_residual_pixels": "0.00", "sdf_proof_residual_pixels": "1.00", "sdf_proof_renderer_weight": "2.50", "implicit_sdf_hidden_channels": "48", "implicit_sdf_residual_pixels": "2.00", "coarse_sdf_surface_weight": "6.00", "sdf_residual_l1_weight": "0.30", "boundary_fuzz_weight": "2.50", "boundary_halo_weight": "1.75", "boundary_renderer_plateau_samples": "5", "boundary_renderer_plateau_max_multiplier": "2.20", "boundary_renderer_plateau_stability_scale": "14.0", "seed": "1337",
    },
    "High Detail": {
        "learning_rate": "0.0001", "weight_decay": "0.00001", "optimizer": "adamw", "scheduler": "cosine-phase",
        "batch_size": "1", "tiles_per_epoch": "384", "augmentation_strength": "1.0",
        "regret_weight": "2.60", "normal_regret_weight": "1.30", "edge_weight": "2.30",
        "detail_laplacian_weight": "0.36", "geometric_alignment_weight": "0.52",
        "tangent_coherence_weight": "0.40", "curvature_coherence_weight": "0.34",
        "synthetic_geometry_probability": "0.88", "boundary_sampling_probability": "0.72",
        "boundary_renderer_band_pixels": "3.75", "boundary_renderer_sample_pixels": "4.0",
        "boundary_renderer_hard_width_pixels": "0.65", "boundary_renderer_soft_width_pixels": "1.65",
        "boundary_renderer_gate_gain": "1.70", "boundary_renderer_far_sample_multiplier": "1.75", "boundary_renderer_far_sample_weight": "0.25", "boundary_gate_need_scale": "0.070", "boundary_gate_exact_floor": "0.40", "boundary_sdf_zero_weight": "3.25", "boundary_edge_sdf_consistency_weight": "1.65", "boundary_pixel_regret_weight": "3.25", "boundary_profile_weight": "1.80", "boundary_regret_weight": "5.25", "sdf_surface_weight": "9.00", "sdf_sign_weight": "2.25", "sdf_eikonal_weight": "9.00", "sdf_gradient_alignment_weight": "2.25", "sdf_metric_gradient_weight": "7.00", "sdf_metric_band_pixels": "12.0", "sdf_coarse_init_std": "0.0005", "sdf_synthetic_validation_tiles": "12", "sdf_zero_band_pixels": "0.45", "sdf_bootstrap_residual_pixels": "0.00", "sdf_proof_residual_pixels": "1.00", "sdf_proof_renderer_weight": "2.75", "implicit_sdf_hidden_channels": "64", "implicit_sdf_residual_pixels": "2.25", "coarse_sdf_surface_weight": "7.00", "sdf_residual_l1_weight": "0.32", "boundary_fuzz_weight": "2.75", "boundary_halo_weight": "2.00", "boundary_renderer_plateau_samples": "6", "boundary_renderer_plateau_max_multiplier": "2.35", "boundary_renderer_plateau_stability_scale": "15.0", "seed": "1337",
    },
    "Conservative": {
        "learning_rate": "0.00008", "weight_decay": "0.00001", "optimizer": "adamw", "scheduler": "phase",
        "batch_size": "1", "tiles_per_epoch": "320", "augmentation_strength": "0.85",
        "regret_weight": "3.00", "normal_regret_weight": "1.45", "edge_weight": "1.80",
        "detail_laplacian_weight": "0.24", "geometric_alignment_weight": "0.50",
        "tangent_coherence_weight": "0.40", "curvature_coherence_weight": "0.34",
        "synthetic_geometry_probability": "0.82", "boundary_sampling_probability": "0.65",
        "boundary_renderer_band_pixels": "3.25", "boundary_renderer_sample_pixels": "3.5",
        "boundary_renderer_hard_width_pixels": "0.85", "boundary_renderer_soft_width_pixels": "2.0",
        "boundary_renderer_gate_gain": "1.45", "boundary_renderer_far_sample_multiplier": "1.60", "boundary_renderer_far_sample_weight": "0.18", "boundary_gate_need_scale": "0.085", "boundary_gate_exact_floor": "0.30", "boundary_sdf_zero_weight": "3.00", "boundary_edge_sdf_consistency_weight": "1.50", "boundary_pixel_regret_weight": "3.50", "boundary_profile_weight": "1.35", "boundary_regret_weight": "5.50", "sdf_surface_weight": "8.50", "sdf_sign_weight": "2.00", "sdf_eikonal_weight": "8.00", "sdf_gradient_alignment_weight": "2.25", "sdf_metric_gradient_weight": "6.50", "sdf_metric_band_pixels": "12.0", "sdf_coarse_init_std": "0.0005", "sdf_synthetic_validation_tiles": "12", "sdf_zero_band_pixels": "0.45", "sdf_bootstrap_residual_pixels": "0.00", "sdf_proof_residual_pixels": "0.75", "sdf_proof_renderer_weight": "2.25", "implicit_sdf_hidden_channels": "48", "implicit_sdf_residual_pixels": "1.50", "coarse_sdf_surface_weight": "7.00", "sdf_residual_l1_weight": "0.35", "boundary_fuzz_weight": "2.25", "boundary_halo_weight": "2.25", "boundary_renderer_plateau_samples": "5", "boundary_renderer_plateau_max_multiplier": "2.00", "boundary_renderer_plateau_stability_scale": "16.0", "seed": "1337",
    },
}


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

    def _load_state(self) -> dict:
        state = {
            "schema": STATE_SCHEMA,
            "status": {stage.id: "pending" for stage in STAGES},
            "current": "validate",
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
        ttk.Label(controls_header, text="scroll for additional tuning controls").pack(side="right")

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
        ttk.Button(footer, text="Preview available model", command=self.preview_available_model).pack(side="left")
        ttk.Button(footer, text="Detect artifacts", command=self.detect).pack(side="right")
        self.footer = footer

        current = self.state.get("current", "validate")
        if current not in BY_ID:
            current = "validate"
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

    def _experiment_ids(self, *, completed_only: bool = False) -> list[str]:
        root = self._experiments_root()
        if not root.is_dir():
            return []
        values = []
        for path in root.iterdir():
            if not path.is_dir() or EXPERIMENT_RE.fullmatch(path.name) is None:
                continue
            if completed_only and not (path / "checkpoint_best.pt").is_file():
                continue
            values.append(path.name.upper())
        return sorted(values, key=lambda value: int(value.split("_")[1]))

    def _latest_completed_experiment(self) -> str | None:
        values = self._experiment_ids(completed_only=True)
        return values[-1] if values else None

    def _previewed_experiment_ids(self) -> list[str]:
        return [
            experiment_id
            for experiment_id in self._experiment_ids(completed_only=True)
            if (self._experiments_root() / experiment_id / "previews/preview_manifest.json").is_file()
        ]

    def _promotion_eligible_experiment_ids(self) -> list[str]:
        eligible: list[str] = []
        for experiment_id in self._previewed_experiment_ids():
            manifest_path = self._experiments_root() / experiment_id / "experiment.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(manifest.get("trainingMode") or "").lower() == "full" and bool(manifest.get("promotionEligible")):
                eligible.append(experiment_id)
        return eligible

    def _promotion_record(self) -> dict | None:
        path = self.repo / "artifacts/nsamdr/promoted/selected_experiment.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _promoted_config(self) -> Path | None:
        record = self._promotion_record()
        if not record:
            return None
        raw = str(record.get("promotedConfig") or "")
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.repo / path
        return path if path.is_file() else None

    def _production_checkpoint(self) -> Path:
        return self.repo / "artifacts/nsamdr/neural_v9/nsamdr_v9_fidelity.pt"

    def _production_state(self) -> Path:
        return self.repo / "artifacts/nsamdr/neural_v9/nsamdr_v9_training_state.pt"

    def _promotion_pointer(self) -> Path:
        return self.repo / "artifacts/nsamdr/promoted/selected_experiment.json"

    def _production_artifact_is_current(self, path: Path) -> bool:
        if not path.is_file():
            return False
        pointer = self._promotion_pointer()
        # A legacy/full checkpoint remains usable until a new tuning config is
        # explicitly promoted. Once promotion occurs, older production state is
        # stale and must not satisfy the new workflow's completion lock.
        if not pointer.is_file():
            return True
        try:
            return path.stat().st_mtime_ns >= pointer.stat().st_mtime_ns
        except OSError:
            return False

    def _production_checkpoint_is_current(self) -> bool:
        return self._production_artifact_is_current(self._production_checkpoint())

    def _preview_dataset_manifest(self) -> Path:
        return self.repo / "artifacts/nsamdr/training_v9_preview_raven/dataset_manifest.json"

    def _refresh_scope(self) -> None:
        production = self._production_checkpoint()
        promotion = self._promotion_record()
        if self._production_checkpoint_is_current():
            source = str((promotion or {}).get("sourceExperiment") or "legacy/unrecorded")
            self.scope_text.set(f"Model scope: FULL — all supported ships enabled | tuning source: {source}")
            return
        latest = self._latest_completed_experiment()
        if latest:
            mode = "unknown"
            try:
                manifest = json.loads((self._experiments_root() / latest / "experiment.json").read_text(encoding="utf-8"))
                mode = str(manifest.get("trainingMode") or "unknown").upper()
            except Exception:
                pass
            self.scope_text.set(
                f"Model scope: TUNING — Raven Navy Issue only | {latest} {mode} | all-ship production preview LOCKED"
            )
        else:
            self.scope_text.set("Model scope: V9.4 geometry-only | Stage 1 Quick Raven tune is the next feedback gate")

    def _stage_lock_reason(self, stage_id: str) -> str | None:
        completed = self._experiment_ids(completed_only=True)
        if stage_id == "tune_compare" and len(completed) < 2:
            return "At least two completed Raven experiments are required."
        if stage_id == "tune_promote" and not self._promotion_eligible_experiment_ids():
            return "V9.4 geometry-only experiments cannot be promoted. Promotion remains locked until geometry A/B passes and the later frozen-geometry appearance stage is enabled and proven."
        if stage_id in {"index", "train", "all"} and self._promoted_config() is None:
            return "Full-dataset work is locked until a Full Raven proof experiment is promoted."
        if stage_id == "preview" and not self._production_checkpoint_is_current():
            return "Full production preview is locked until the selected promoted configuration completes full training."
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
        elif stage_id == "tune":
            experiments = ["new", *self._experiment_ids()]
            self._row("Training mode", "training_mode", "Quick (~10-15 min)", ("Quick (~10-15 min)", "Full / promotion proof"))
            self._row("Experiment", "experiment", "new", experiments)
            self._row("Preset", "preset", "Baseline", ("Current Best", "Baseline", "High Detail", "Conservative", "Custom"))
            self._label_row("Dataset", "Raven Navy Issue — fixed deterministic train + held-out regions")
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Max training regions", "train_crops", "12")
            self._row("Max held-out regions", "validation_crops", "4")
            self._check("Rebuild fixed Raven dataset", "rebuild", False)
            self._row("Training control", "control", "auto", ("auto", "resume", "restart"))
            self._row("Learning rate", "learning_rate", "0.0001")
            self._row("Optimiser", "optimizer", "adamw", ("adamw", "adam"))
            self._row("Scheduler", "scheduler", "phase", ("phase", "cosine-phase"))
            self._row("Batch size", "batch_size", "1", ("1", "2", "4", "8"))
            self._row("Weight decay", "weight_decay", "0.00001")
            self._row("Tiles / epoch", "tiles_per_epoch", "96")
            self._row("Validation tiles", "validation_tiles", "16")
            self._row("Augmentation strength", "augmentation_strength", "1.0")
            self._row("Regret loss", "regret_weight", "2.50")
            self._row("Normal regret loss", "normal_regret_weight", "1.25")
            self._row("Edge loss", "edge_weight", "2.00")
            self._row("Detail/Laplacian", "detail_laplacian_weight", "0.28")
            self._row("Geometric alignment", "geometric_alignment_weight", "0.48")
            self._row("Tangent coherence", "tangent_coherence_weight", "0.36")
            self._row("Curvature coherence", "curvature_coherence_weight", "0.30")
            self._row("Exact geometry fraction", "synthetic_geometry_probability", "0.82")
            self._row("Boundary sampling", "boundary_sampling_probability", "0.68")
            self._row("Boundary band (HR px)", "boundary_renderer_band_pixels", "3.5")
            self._row("Side sample distance (HR px)", "boundary_renderer_sample_pixels", "3.75")
            self._row("Hard edge width (HR px)", "boundary_renderer_hard_width_pixels", "0.70")
            self._row("Soft edge width (HR px)", "boundary_renderer_soft_width_pixels", "1.80")
            self._row("Boundary gate gain", "boundary_renderer_gate_gain", "1.60")
            self._row("Far plateau multiplier", "boundary_renderer_far_sample_multiplier", "1.70")
            self._row("Far plateau weight", "boundary_renderer_far_sample_weight", "0.22")
            self._row("Gate need scale", "boundary_gate_need_scale", "0.075")
            self._row("Exact gate floor", "boundary_gate_exact_floor", "0.35")
            self._row("SDF zero-set loss", "boundary_sdf_zero_weight", "3.00")
            self._row("Edge/SDF consistency", "boundary_edge_sdf_consistency_weight", "1.50")
            self._row("Pixel boundary regret", "boundary_pixel_regret_weight", "3.00")
            self._row("Boundary profile loss", "boundary_profile_weight", "1.65")
            self._row("Boundary regret loss", "boundary_regret_weight", "5.00")
            self._row("SDF surface loss", "sdf_surface_weight", "8.00")
            self._row("SDF sign loss", "sdf_sign_weight", "2.00")
            self._row("SDF Eikonal loss", "sdf_eikonal_weight", "8.00")
            self._row("SDF gradient alignment", "sdf_gradient_alignment_weight", "2.00")
            self._row("SDF metric gradient", "sdf_metric_gradient_weight", "6.00")
            self._row("SDF metric band px", "sdf_metric_band_pixels", "12.0")
            self._row("SDF coarse init std", "sdf_coarse_init_std", "0.0005")
            self._row("Synthetic SDF validation tiles", "sdf_synthetic_validation_tiles", "12")
            self._row("SDF zero band px", "sdf_zero_band_pixels", "0.50")
            self._row("Bootstrap residual px", "sdf_bootstrap_residual_pixels", "0.00")
            self._row("SDF-proof residual px", "sdf_proof_residual_pixels", "1.00")
            self._row("SDF forced-gate renderer", "sdf_proof_renderer_weight", "2.50")
            self._row("Implicit SDF hidden", "implicit_sdf_hidden_channels", "48")
            self._row("Bounded SDF residual px", "implicit_sdf_residual_pixels", "2.00")
            self._row("Coarse SDF loss", "coarse_sdf_surface_weight", "6.00")
            self._row("SDF residual L1", "sdf_residual_l1_weight", "0.30")
            self._row("Hard-edge fuzz loss", "boundary_fuzz_weight", "2.50")
            self._row("Halo suppression loss", "boundary_halo_weight", "1.75")
            self._row("Plateau samples", "boundary_renderer_plateau_samples", "5")
            self._row("Plateau max multiplier", "boundary_renderer_plateau_max_multiplier", "2.20")
            self._row("Plateau stability", "boundary_renderer_plateau_stability_scale", "14.0")
            self._row("Seed", "seed", "1337")
            self._check("Randomise seed (advanced)", "randomise_seed", False)
            self._row("Advanced overrides", "advanced_overrides", "")
            self._label_row("Renderer", "launches automatically after successful training")
            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))
            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))
            self._check("Force candidate regeneration", "force_candidate", True)
            self._label_row("Automated evidence", "G0-G5 renderer/SDF/gate proof + fuzz/halo/topology; Raven only after PASS")
            self._check("Automatic geometry audit + captured evidence", "geometry_audit", True)
            self._row("Geometry critic", "geometry_critic", "auto", ("auto", "off", "required"))
            self._row("Audit policy", "geometry_audit_policy", "report", ("report", "strict"))
            self._row("Evidence regions", "geometry_evidence_regions", "12")
            self._row("Critic calibration steps", "critic_steps", "120")
            self._label_row("Feedback output", r"EXP_####\previews\EXP_####_geometry_feedback.zip")
            self._row("Early-stop patience (Full)", "early_stop_patience", "3")
            self._row("Early-stop min delta (Full)", "early_stop_min_delta", "0.0005")
            self._training_performance_rows(default_workers="4")
            self.vars["preset"].trace_add("write", lambda *_args: self._preset_changed())
            self.vars["training_mode"].trace_add("write", lambda *_args: self._tuning_mode_changed())
            self.vars["experiment"].trace_add("write", lambda *_args: self._experiment_changed())
            self._apply_preset("Baseline")
            self._tuning_mode_changed()
        elif stage_id == "tune_compare":
            completed = self._experiment_ids(completed_only=True)
            a = completed[-1] if completed else "<none>"
            b = completed[-2] if len(completed) >= 2 else "<none>"
            self._row("Experiment A", "exp_a", a, completed or ("<none>",))
            self._row("Experiment B", "exp_b", b, completed or ("<none>",))
            self._row("Experiment C", "exp_c", "<none>", ["<none>", *completed])
            self._label_row("Comparison", "same held-out Raven tiles; linked pan/zoom; immutable config diff")
        elif stage_id == "tune_promote":
            eligible = self._promotion_eligible_experiment_ids()
            default = eligible[-1] if eligible else "<none>"
            self._row("Experiment", "experiment", default, eligible or ("<none>",))
            self._label_row("Promotion policy", "semantic hyperparameters copied exactly; only full-data/work scope changes")
        elif stage_id == "index":
            config = self._promoted_config()
            self._label_row("Promoted config", str(config or "<locked: promote experiment first>"))
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._check("Rebuild full dataset", "rebuild", False)
        elif stage_id == "train":
            promotion = self._promotion_record() or {}
            config = self._promoted_config()
            self._label_row("Tuning source", str(promotion.get("sourceExperiment") or "<locked>"))
            self._label_row("Promoted config", str(config or "<locked>"))
            self._row("Shared cache", "cache", r"C:\CCP\EVE")
            self._row("Training control", "control", "auto", ("auto", "resume", "restart"))
            self._check("Skip existing full dataset", "skip", True)
            self._training_performance_rows(default_workers="8")
        elif stage_id == "preview":
            self._label_row("Checkpoint", str(self._production_checkpoint()))
            self._row("Target size", "target", "4096", ("1024", "2048", "4096"))
            self._row("Device", "device", "cuda", ("cuda", "cpu"))
            self._check("Force candidate regeneration", "force_candidate", True)
            self._label_row("Capability", "FULL — all supported ships unlocked")
        elif stage_id == "all":
            promotion = self._promotion_record() or {}
            self._label_row("Tuning source", str(promotion.get("sourceExperiment") or "<locked>"))
            self._label_row("Promoted config", str(self._promoted_config() or "<locked>"))
        self._update_command()
        self.form_canvas.yview_moveto(0.0)
        self.root.after_idle(self._form_content_configured)

    def _training_performance_rows(self, *, default_workers: str) -> None:
        self._row("Performance profile", "profile", "fast", ("optimized", "fast", "balanced", "compatibility"))
        self._row("Workers", "workers", default_workers)
        self._row("Prefetch", "prefetch", "2")
        self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))

    def _preset_changed(self) -> None:
        preset = self._value("preset")
        if preset != "Custom":
            self._apply_preset(preset)
            if "training_mode" in self.vars:
                self._tuning_mode_changed()

    def _apply_preset(self, preset: str) -> None:
        values = PRESET_VALUES.get(preset)
        if preset == "Current Best":
            source = None
            promotion = self._promotion_record()
            if promotion:
                source = str(promotion.get("sourceExperiment") or "")
            if not source:
                source = self._latest_completed_experiment()
            path = self._experiments_root() / source / "resolved_config.json" if source else None
            if path and path.is_file():
                try:
                    config = json.loads(path.read_text(encoding="utf-8"))
                    values = {
                        "learning_rate": str(config.get("learning_rate", 0.0001)),
                        "weight_decay": str(config.get("weight_decay", 0.00001)),
                        "optimizer": str(config.get("optimizer_name", "adamw")),
                        "scheduler": str(config.get("scheduler_name", "phase")),
                        "batch_size": str(config.get("batch_size", 1)),
                        "tiles_per_epoch": "320",
                        "augmentation_strength": "1.0",
                        "regret_weight": str(config.get("regret_weight", 2.5)),
                        "normal_regret_weight": str(config.get("normal_regret_weight", 1.25)),
                        "edge_weight": str(config.get("edge_weight", 2.00)),
                        "detail_laplacian_weight": str(config.get("detail_laplacian_weight", 0.28)),
                        "geometric_alignment_weight": str(config.get("geometric_alignment_weight", 0.48)),
                        "tangent_coherence_weight": str(config.get("tangent_coherence_weight", 0.36)),
                        "curvature_coherence_weight": str(config.get("curvature_coherence_weight", 0.30)),
                        "synthetic_geometry_probability": str(config.get("synthetic_geometry_probability", 0.82)),
                        "boundary_sampling_probability": str(config.get("boundary_sampling_probability", 0.68)),
                        "boundary_renderer_band_pixels": str(config.get("boundary_renderer_band_pixels", 3.5)),
                        "boundary_renderer_sample_pixels": str(config.get("boundary_renderer_sample_pixels", 3.75)),
                        "boundary_renderer_hard_width_pixels": str(config.get("boundary_renderer_hard_width_pixels", 0.70)),
                        "boundary_renderer_soft_width_pixels": str(config.get("boundary_renderer_soft_width_pixels", 1.80)),
                        "boundary_renderer_gate_gain": str(config.get("boundary_renderer_gate_gain", 1.60)),
                        "boundary_renderer_far_sample_multiplier": str(config.get("boundary_renderer_far_sample_multiplier", 1.70)),
                        "boundary_renderer_far_sample_weight": str(config.get("boundary_renderer_far_sample_weight", 0.22)),
                        "boundary_gate_need_scale": str(config.get("boundary_gate_need_scale", 0.075)),
                        "boundary_gate_exact_floor": str(config.get("boundary_gate_exact_floor", 0.35)),
                        "boundary_sdf_zero_weight": str(config.get("boundary_sdf_zero_weight", 3.00)),
                        "boundary_edge_sdf_consistency_weight": str(config.get("boundary_edge_sdf_consistency_weight", 1.50)),
                        "boundary_pixel_regret_weight": str(config.get("boundary_pixel_regret_weight", 3.00)),
                        "boundary_profile_weight": str(config.get("boundary_profile_weight", 1.65)),
                        "boundary_regret_weight": str(config.get("boundary_regret_weight", 5.00)),
                        "sdf_surface_weight": str(config.get("sdf_surface_weight", 8.00)),
                        "sdf_sign_weight": str(config.get("sdf_sign_weight", 2.00)),
                        "sdf_eikonal_weight": str(config.get("sdf_eikonal_weight", 8.00)),
                        "sdf_gradient_alignment_weight": str(config.get("sdf_gradient_alignment_weight", 2.00)),
                        "sdf_metric_gradient_weight": str(config.get("sdf_metric_gradient_weight", 6.00)),
                        "sdf_metric_band_pixels": str(config.get("sdf_metric_band_pixels", 12.0)),
                        "sdf_coarse_init_std": str(config.get("sdf_coarse_init_std", 0.0005)),
                        "sdf_synthetic_validation_tiles": str(config.get("sdf_synthetic_validation_tiles", 12)),
                        "sdf_zero_band_pixels": str(config.get("sdf_zero_band_pixels", 0.50)),
                        "sdf_bootstrap_residual_pixels": str(config.get("sdf_bootstrap_residual_pixels", 0.00)),
                        "sdf_proof_residual_pixels": str(config.get("sdf_proof_residual_pixels", 1.00)),
                        "sdf_proof_renderer_weight": str(config.get("sdf_proof_renderer_weight", 2.50)),
                        "implicit_sdf_hidden_channels": str(config.get("implicit_sdf_hidden_channels", 48)),
                        "implicit_sdf_residual_pixels": str(config.get("implicit_sdf_residual_pixels", 2.0)),
                        "coarse_sdf_surface_weight": str(config.get("coarse_sdf_surface_weight", 6.0)),
                        "sdf_residual_l1_weight": str(config.get("sdf_residual_l1_weight", 0.30)),
                        "boundary_fuzz_weight": str(config.get("boundary_fuzz_weight", 2.50)),
                        "boundary_halo_weight": str(config.get("boundary_halo_weight", 1.75)),
                        "boundary_renderer_plateau_samples": str(config.get("boundary_renderer_plateau_samples", 5)),
                        "boundary_renderer_plateau_max_multiplier": str(config.get("boundary_renderer_plateau_max_multiplier", 2.20)),
                        "boundary_renderer_plateau_stability_scale": str(config.get("boundary_renderer_plateau_stability_scale", 14.0)),
                        "seed": str(config.get("seed", 1337)),
                    }
                except Exception:
                    values = PRESET_VALUES["Baseline"]
            else:
                values = PRESET_VALUES["Baseline"]
        if not values:
            return
        for key, value in values.items():
            variable = self.vars.get(key)
            if variable is not None:
                variable.set(value)

    def _tuning_mode_changed(self) -> None:
        mode = self._value("training_mode", "Quick (~10-15 min)")
        quick = mode.startswith("Quick")
        if "tiles_per_epoch" in self.vars:
            self.vars["tiles_per_epoch"].set("96" if quick else "128")
        if "validation_tiles" in self.vars:
            self.vars["validation_tiles"].set("16" if quick else "32")
        self._update_command()

    def _experiment_changed(self) -> None:
        experiment = self._value("experiment", "new").upper()
        if experiment in {"", "NEW"}:
            return
        manifest_path = self._experiments_root() / experiment / "experiment.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return
        mode = str(manifest.get("trainingMode") or "full").lower()
        variable = self.vars.get("training_mode")
        if variable is not None:
            variable.set("Quick (~10-15 min)" if mode == "quick" else "Full / promotion proof")

    def _value(self, key: str, default: str = "") -> str:
        variable = self.vars.get(key)
        return str(variable.get()) if variable is not None else default

    def _args(self, stage_id: str) -> list[str]:
        args: list[str] = []
        if stage_id == "setup":
            if self.vars.get("force") and bool(self.vars["force"].get()):
                args.append("--force")
        elif stage_id == "tune":
            experiment = self._value("experiment", "new")
            mode = "quick" if self._value("training_mode", "Quick").startswith("Quick") else "full"
            args = [
                "--base-config", "tools/nsamdr/neural/configs/v9_preview_raven.json",
                "--shared-cache", self._value("cache"),
                "--max-train-regions", self._value("train_crops", "12"),
                "--max-validation-regions", self._value("validation_crops", "4"),
                "--experiment", experiment,
                "--control", self._value("control", "auto"),
                "--training-mode", mode,
                "--preset", self._value("preset", "Baseline"),
                "--preview-target-size", self._value("target", "4096"),
                "--preview-device", self._value("device", "cuda"),
                "--performance-profile", self._value("profile", "fast"),
                "--workers", self._value("workers", "4"),
                "--prefetch-factor", self._value("prefetch", "2"),
                "--amp-precision", self._value("amp", "auto"),
                "--early-stop-patience", self._value("early_stop_patience", "3"),
                "--early-stop-min-delta", self._value("early_stop_min_delta", "0.0005"),
                "--geometry-critic", self._value("geometry_critic", "auto"),
                "--geometry-audit-policy", self._value("geometry_audit_policy", "report"),
                "--geometry-evidence-regions", self._value("geometry_evidence_regions", "12"),
                "--critic-steps", self._value("critic_steps", "120"),
            ]
            if self.vars.get("geometry_audit") is not None and not bool(self.vars["geometry_audit"].get()):
                args.append("--no-geometry-audit")
            if bool(self.vars["rebuild"].get()):
                args.append("--rebuild-dataset")
            if bool(self.vars["force_candidate"].get()):
                args.append("--force-candidate")
            if experiment.lower() == "new":
                args.extend([
                    "--learning-rate", self._value("learning_rate"),
                    "--optimizer-name", self._value("optimizer"),
                    "--scheduler-name", self._value("scheduler"),
                    "--batch-size", self._value("batch_size"),
                    "--weight-decay", self._value("weight_decay"),
                    "--tiles-per-epoch", self._value("tiles_per_epoch"),
                    "--validation-tiles", self._value("validation_tiles"),
                    "--augmentation-strength", self._value("augmentation_strength"),
                    "--regret-weight", self._value("regret_weight"),
                    "--normal-regret-weight", self._value("normal_regret_weight"),
                    "--edge-weight", self._value("edge_weight"),
                    "--detail-laplacian-weight", self._value("detail_laplacian_weight"),
                    "--geometric-alignment-weight", self._value("geometric_alignment_weight"),
                    "--tangent-coherence-weight", self._value("tangent_coherence_weight"),
                    "--curvature-coherence-weight", self._value("curvature_coherence_weight"),
                    "--synthetic-geometry-probability", self._value("synthetic_geometry_probability"),
                    "--boundary-sampling-probability", self._value("boundary_sampling_probability"),
                    "--boundary-renderer-band-pixels", self._value("boundary_renderer_band_pixels"),
                    "--boundary-renderer-sample-pixels", self._value("boundary_renderer_sample_pixels"),
                    "--boundary-renderer-hard-width-pixels", self._value("boundary_renderer_hard_width_pixels"),
                    "--boundary-renderer-soft-width-pixels", self._value("boundary_renderer_soft_width_pixels"),
                    "--boundary-renderer-gate-gain", self._value("boundary_renderer_gate_gain"),
                    "--boundary-renderer-far-sample-multiplier", self._value("boundary_renderer_far_sample_multiplier"),
                    "--boundary-renderer-far-sample-weight", self._value("boundary_renderer_far_sample_weight"),
                    "--boundary-gate-need-scale", self._value("boundary_gate_need_scale"),
                    "--boundary-gate-exact-floor", self._value("boundary_gate_exact_floor"),
                    "--boundary-sdf-zero-weight", self._value("boundary_sdf_zero_weight"),
                    "--boundary-edge-sdf-consistency-weight", self._value("boundary_edge_sdf_consistency_weight"),
                    "--boundary-pixel-regret-weight", self._value("boundary_pixel_regret_weight"),
                    "--boundary-profile-weight", self._value("boundary_profile_weight"),
                    "--boundary-regret-weight", self._value("boundary_regret_weight"),
                    "--sdf-surface-weight", self._value("sdf_surface_weight"),
                    "--sdf-sign-weight", self._value("sdf_sign_weight"),
                    "--sdf-eikonal-weight", self._value("sdf_eikonal_weight"),
                    "--sdf-gradient-alignment-weight", self._value("sdf_gradient_alignment_weight"),
                    "--sdf-metric-gradient-weight", self._value("sdf_metric_gradient_weight"),
                    "--sdf-metric-band-pixels", self._value("sdf_metric_band_pixels"),
                    "--sdf-coarse-init-std", self._value("sdf_coarse_init_std"),
                    "--sdf-synthetic-validation-tiles", self._value("sdf_synthetic_validation_tiles"),
                    "--sdf-zero-band-pixels", self._value("sdf_zero_band_pixels"),
                    "--sdf-bootstrap-residual-pixels", self._value("sdf_bootstrap_residual_pixels"),
                    "--sdf-proof-residual-pixels", self._value("sdf_proof_residual_pixels"),
                    "--sdf-proof-renderer-weight", self._value("sdf_proof_renderer_weight"),
                    "--implicit-sdf-hidden-channels", self._value("implicit_sdf_hidden_channels"),
                    "--implicit-sdf-residual-pixels", self._value("implicit_sdf_residual_pixels"),
                    "--coarse-sdf-surface-weight", self._value("coarse_sdf_surface_weight"),
                    "--sdf-residual-l1-weight", self._value("sdf_residual_l1_weight"),
                    "--boundary-fuzz-weight", self._value("boundary_fuzz_weight"),
                    "--boundary-halo-weight", self._value("boundary_halo_weight"),
                    "--boundary-renderer-plateau-samples", self._value("boundary_renderer_plateau_samples"),
                    "--boundary-renderer-plateau-max-multiplier", self._value("boundary_renderer_plateau_max_multiplier"),
                    "--boundary-renderer-plateau-stability-scale", self._value("boundary_renderer_plateau_stability_scale"),
                    "--seed", self._value("seed"),
                ])
                if bool(self.vars["randomise_seed"].get()):
                    args.append("--randomise-seed")
                advanced = self._value("advanced_overrides").strip()
                if advanced:
                    args.extend(["--advanced-overrides", advanced])
        elif stage_id == "tune_compare":
            values = [self._value("exp_a"), self._value("exp_b"), self._value("exp_c")]
            ids = [value for value in values if value and value != "<none>"]
            args = ids
        elif stage_id == "tune_promote":
            args = [self._value("experiment")]
        elif stage_id == "index":
            config = self._promoted_config()
            args = ["--config", str(config or ""), "--shared-cache", self._value("cache")]
            if bool(self.vars["rebuild"].get()):
                args.append("--rebuild")
        elif stage_id == "train":
            config = self._promoted_config()
            args = ["--config", str(config or ""), "--shared-cache", self._value("cache")]
            if bool(self.vars["skip"].get()):
                args.append("--skip-dataset")
            args.extend([
                "--control", self._value("control", "auto"),
                "--performance-profile", self._value("profile", "fast"),
                "--workers", self._value("workers", "8"),
                "--prefetch-factor", self._value("prefetch", "2"),
                "--amp-precision", self._value("amp", "auto"),
            ])
        elif stage_id == "preview":
            args = [
                "--checkpoint-dir", "artifacts/nsamdr/neural_v9",
                "--preview-strength", "1.0",
                "--target-size", self._value("target"),
                "--device", self._value("device"),
            ]
            if bool(self.vars["force_candidate"].get()):
                args.append("--force-candidate")
        return args

    def _training_artifact_paths(self, stage_id: str) -> tuple[Path | None, Path | None]:
        if stage_id == "tune":
            experiment = self._value("experiment", "new")
            if experiment.lower() == "new":
                return None, None
            output = self._experiments_root() / experiment
        elif stage_id == "train":
            output = self.repo / "artifacts/nsamdr/neural_v9"
        else:
            return None, None
        return output / "nsamdr_v9_training_state.pt", output / "nsamdr_v9_fidelity.pt"

    def _validate_before_launch(self, stage_id: str) -> bool:
        lock = self._stage_lock_reason(stage_id)
        if lock:
            messagebox.showwarning("Stage locked", lock)
            self.progress_text.set(f"Locked — {lock}")
            return False
        if stage_id == "tune_compare":
            ids = [self._value("exp_a"), self._value("exp_b"), self._value("exp_c")]
            ids = [value for value in ids if value and value != "<none>"]
            if len(set(ids)) < 2:
                messagebox.showerror(APP_TITLE, "Select at least two distinct completed experiments.")
                return False
        if stage_id in {"tune", "train"}:
            control = self._value("control", "auto")
            state_path, checkpoint_path = self._training_artifact_paths(stage_id)
            if stage_id == "tune" and self._value("experiment", "new").lower() != "new":
                experiment_id = self._value("experiment").upper()
                manifest_path = self._experiments_root() / experiment_id / "experiment.json"
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    stored_mode = str(manifest.get("trainingMode") or "full").lower()
                    selected_mode = "quick" if self._value("training_mode", "Quick").startswith("Quick") else "full"
                    if stored_mode != selected_mode:
                        messagebox.showerror(
                            APP_TITLE,
                            f"{experiment_id} is an existing {stored_mode.upper()} experiment. "
                            "Its immutable training mode cannot be changed while resuming.",
                        )
                        return False
                except OSError:
                    pass
            if (
                stage_id == "tune"
                and self._value("experiment", "new").lower() != "new"
                and checkpoint_path is not None
                and (checkpoint_path.parent / "checkpoint_best.pt").is_file()
            ):
                messagebox.showinfo(
                    APP_TITLE,
                    "Completed experiments are immutable. Select 'new' to allocate "
                    "a new EXP_####. Existing completed checkpoints/configs are never "
                    "overwritten by Stage 1.",
                )
                return False
            if (
                stage_id == "train"
                and control in {"auto", "resume"}
                and state_path is not None
                and state_path.is_file()
                and not self._production_artifact_is_current(state_path)
            ):
                messagebox.showwarning(
                    "New promoted configuration",
                    "The existing production training state predates the selected "
                    "Raven experiment promotion and cannot be resumed into the new "
                    "semantic configuration.\n\nSelect 'restart' once to begin the new "
                    "full training run. The old state/checkpoint will be backed up "
                    "before deletion.",
                )
                self.progress_text.set(
                    "New promotion requires one explicit backed-up production restart"
                )
                return False
            if control == "resume" and (state_path is None or not state_path.is_file()):
                messagebox.showerror(APP_TITLE, f"Resume selected but no resumable state exists:\n\n{state_path or '<new experiment>'}")
                return False
            if control == "auto" and state_path is not None and not state_path.is_file() and checkpoint_path and checkpoint_path.is_file():
                messagebox.showerror(APP_TITLE, "A final checkpoint exists without a resumable state. Auto refuses to overwrite it.")
                return False
            if control == "restart" and state_path is not None:
                existing = [path for path in (state_path, checkpoint_path) if path and path.is_file()]
                if existing:
                    text = "\n".join(str(path) for path in existing)
                    if not messagebox.askyesno(
                        "Confirm destructive restart",
                        "Restart begins again at epoch 1. A timestamped backup is created first.\n\n"
                        f"Existing artifact(s):\n{text}\n\nContinue?",
                        icon="warning",
                    ):
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
                    batch = BATCH_RE.search(line)
                    rate = RATE_RE.search(line)
                    if batch and rate and self.current_epoch:
                        e, total, phase = self.current_epoch
                        b, btotal = int(batch.group(1)), int(batch.group(2))
                        live_rate = float(rate.group(1))
                        progress, epoch_eta, total_eta = _live_eta(e, total, b, btotal, live_rate)
                        self.progress["value"] = progress
                        vram = VRAM_MODE_RE.search(line)
                        vram_text = f" | VRAM {vram.group(1)} {float(vram.group(2)):.2f}GiB free" if vram else ""
                        self.progress_text.set(
                            f"Epoch {e}/{total} — {phase} | Batch {b}/{btotal} | {progress:.1f}% | "
                            f"Rate {live_rate:.2f} tile/s | ETA {_format_duration(total_eta)} | "
                            f"This epoch {_format_duration(epoch_eta)}{vram_text}"
                        )
                elif kind == "done":
                    return_code, queue_next = item[2], item[3]
                    self.process = None
                    self._set_status(stage_id, "completed" if return_code == 0 else "failed")
                    self.progress_text.set("Completed successfully" if return_code == 0 else f"Failed, exit code {return_code}")
                    self.detect(silent=True)
                    if queue_next and return_code == 0:
                        self.root.after(150, self._run_pending)
                else:
                    self.process = None
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

    def preview_available_model(self) -> None:
        if self._production_checkpoint_is_current():
            args = [
                "--checkpoint-dir", "artifacts/nsamdr/neural_v9",
                "--preview-strength", "1.0", "--target-size", "1024",
                "--device", "cuda", "--force-candidate",
            ]
            self._run("preview", args=args)
            return
        experiment = self._latest_completed_experiment()
        if experiment:
            preview_args = [
                experiment, "--shared-cache", self._value("cache", r"C:\CCP\EVE"),
                "--target-size", self._value("target", "4096"),
                "--device", self._value("device", "cuda"),
                "--geometry-critic", self._value("geometry_critic", "auto"),
                "--geometry-audit-policy", self._value("geometry_audit_policy", "report"),
                "--geometry-evidence-regions", self._value("geometry_evidence_regions", "12"),
                "--force-candidate",
            ]
            command = self._dispatcher_argv(("preview", "experiment"), preview_args)
            subprocess.Popen(command, cwd=self.repo)
            self.progress_text.set(f"Launched Raven renderer preview for {experiment}")
            return
        messagebox.showinfo(APP_TITLE, "No completed tuning or production checkpoint exists yet.")

    def detect(self, silent: bool = False) -> None:
        statuses = self.state.setdefault("status", {})
        artifacts = {
            "validate": None,
            "tune": None,
            "tune_compare": None,
            "tune_promote": self.repo / "artifacts/nsamdr/promoted/selected_experiment.json",
            "index": self.repo / "artifacts/nsamdr/training_v9/dataset_manifest.json",
            "train": self._production_checkpoint() if self._production_checkpoint_is_current() else None,
            "preview": self._production_checkpoint() if self._production_checkpoint_is_current() else None,
        }
        if self._promotion_pointer().is_file() and not self._production_checkpoint_is_current():
            if statuses.get("train") == "completed":
                statuses["train"] = "pending"
            if statuses.get("preview") == "completed":
                statuses["preview"] = "pending"
        completed_experiments = self._experiment_ids(completed_only=True)
        if any((self._experiments_root() / exp / "previews/preview_manifest.json").is_file() for exp in completed_experiments):
            statuses["tune"] = "completed"
        compare_root = self.repo / "artifacts/nsamdr/experiments/compare"
        if compare_root.is_dir() and any(compare_root.glob("compare_*.html")):
            statuses["tune_compare"] = "completed"
        for stage_id, path in artifacts.items():
            if path is not None and path.is_file():
                statuses[stage_id] = "completed"
        if statuses.get("validate") == "locked":
            statuses["validate"] = "pending"
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
