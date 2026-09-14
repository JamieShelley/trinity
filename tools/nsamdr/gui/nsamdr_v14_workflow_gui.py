#!/usr/bin/env python3
"""NSAMDR V14.3 GUI surface.

The diagnostic ladder uses the exact V14.3 production model. Historical V9-V13
geometry, SDF, seam, and profile implementations are not restored.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import nsamdr_v9_workflow_gui as base

NEURAL_ROOT = Path(__file__).resolve().parent.parent / "neural"
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14.config import MODEL_SCHEMA


VERSION = "V14.3"
EXPERIMENT_SCHEMA = "NSAMDR_V14_EXPERIMENT_V1"
FINAL_SCHEMA = "NSAMDR_V14_FINAL_MANIFEST_V1"
MINI_SCHEMA = "NSAMDR_V14_MINI_DIAGNOSTIC_V1"

base.APP_TITLE = f"NSAMDR {VERSION} HR-First Workflow"
_existing = {stage.id: stage for stage in base.STAGES}

_setup = base.Stage(
    "setup",
    "P0",
    "Prepare CUDA environment",
    _existing["setup"].command,
    "Create or verify the CUDA Python environment.",
    False,
)
_capacity = base.Stage(
    "capacity",
    "1",
    f"{VERSION} HR Residual Capacity",
    ("v14-mini-capacity",),
    (
        "Fast non-promotable overfit proof on one deterministic high-detail Raven "
        "region. Tests the stable V14.3 multi-scale candidate C."
    ),
    False,
)
_multiregion = base.Stage(
    "multiregion",
    "2",
    f"{VERSION} Multi-Region SR Mini",
    ("v14-mini-multiregion",),
    (
        "Small disjoint-region generalisation proof for the exact V14.3 candidate C. "
        "Requires Capacity first."
    ),
    False,
)
_selector = base.Stage(
    "selector",
    "3",
    f"{VERSION} Selector Retention Mini",
    ("v14-mini-selector",),
    "Train only BenefitSelector from the latest passing V14.3 multi-region candidate.",
    False,
)
_quick = base.Stage(
    "quick",
    "4",
    f"{VERSION} HR-First Raven Quick",
    _existing["quick"].command,
    (
        "Train the stable multi-scale phase-neutral 4x candidate C on representative "
        "Raven evidence. Train BenefitSelector only after C qualifies."
    ),
    True,
)
_train = base.Stage(
    "train",
    "5",
    "Full Training (disabled)",
    _existing["train"].command,
    "Disabled until V14.3 Raven Quick meets the README qualification gates.",
    False,
)
_preview = base.Stage(
    "preview",
    "6",
    f"Qualified {VERSION} Preview",
    _existing["preview"].command,
    "Bake and inspect 4x physical maps from a qualified immutable V14.3 checkpoint.",
    False,
)

base.STAGES = (
    _setup,
    _capacity,
    _multiregion,
    _selector,
    _quick,
    _train,
    _preview,
)
base.BY_ID = {stage.id: stage for stage in base.STAGES}
base.PIPELINE = ["quick"]

_original_stage_lock_reason = base.App._stage_lock_reason
_original_selected = base.App._selected
_original_args = base.App._args
_original_dispatcher_argv = base.App._dispatcher_argv


class V14ArtifactInspector:
    """Read V14.3 experiment and diagnostic artifacts for GUI state."""

    def __init__(self, app: base.App) -> None:
        self.app = app

    def is_v14_experiment(self, directory: Path) -> bool:
        try:
            payload = json.loads(
                (directory / "experiment.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return False
        return bool(
            payload.get("schema") == EXPERIMENT_SCHEMA
            and payload.get("modelSchema") == MODEL_SCHEMA
        )

    def is_qualified_final(self, directory: Path) -> bool:
        try:
            experiment = json.loads(
                (directory / "experiment.json").read_text(encoding="utf-8")
            )
            final = json.loads(
                (directory / "final_manifest.json").read_text(encoding="utf-8")
            )
            checkpoint = dict(final.get("checkpoint") or {})
            checkpoint_path = (
                directory / str(checkpoint.get("path") or "")
            ).resolve()
            checkpoint_path.relative_to(
                (directory / "checkpoints" / "final").resolve()
            )
        except (OSError, ValueError, TypeError):
            return False
        return bool(
            experiment.get("schema") == EXPERIMENT_SCHEMA
            and experiment.get("modelSchema") == MODEL_SCHEMA
            and experiment.get("status") == "completed"
            and experiment.get("qualified") is True
            and final.get("schema") == FINAL_SCHEMA
            and final.get("status") == "completed"
            and final.get("qualified") is True
            and final.get("selectionKind") == "production-final"
            and checkpoint.get("schema") == MODEL_SCHEMA
            and checkpoint.get("immutable") is True
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(checkpoint.get("sha256") or ""),
                )
            )
            and checkpoint_path.is_file()
        )

    def mini_passed(self, mode: str) -> bool:
        root = self.app.repo / "artifacts/nsamdr/diagnostics/v14_mini"
        if not root.is_dir():
            return False
        for path in root.glob(f"{mode}_*/report.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if (
                payload.get("schema") == MINI_SCHEMA
                and payload.get("revision") == VERSION
                and payload.get("mode") == mode
                and payload.get("modelSchema") == MODEL_SCHEMA
                and payload.get("passed") is True
            ):
                return True
        return False


def _experiment_ids(
    self: base.App,
    *,
    completed_only: bool = False,
) -> list[str]:
    if not completed_only:
        return []
    inspector = V14ArtifactInspector(self)
    root = self._experiments_root()
    if not root.is_dir():
        return []
    result: list[str] = []
    for directory in root.iterdir():
        if (
            not directory.is_dir()
            or base.EXPERIMENT_RE.fullmatch(directory.name) is None
        ):
            continue
        if (
            inspector.is_v14_experiment(directory)
            and inspector.is_qualified_final(directory)
        ):
            result.append(directory.name.upper())
    return sorted(result, key=lambda value: int(value.split("_")[1]))


def _stage_lock_reason(self: base.App, stage_id: str) -> str | None:
    inspector = V14ArtifactInspector(self)
    if stage_id == "multiregion" and not inspector.mini_passed("capacity"):
        return f"Run and pass {VERSION} HR Residual Capacity first."
    if stage_id == "selector" and not inspector.mini_passed("multiregion"):
        return f"Run and pass {VERSION} Multi-Region SR Mini first."
    if stage_id == "train":
        return f"Full Training is disabled until {VERSION} Raven Quick qualifies."
    if stage_id == "preview" and not self._previewable_experiment_ids():
        return f"Preview requires a completed qualified {VERSION} EXP_#### final."
    return _original_stage_lock_reason(self, stage_id)


def _common_mini_args(self: base.App) -> list[str]:
    values = [
        "--shared-cache",
        self._value("cache", r"C:\CCP\EVE"),
        "--device",
        self._value("device", "cuda"),
        "--amp-precision",
        self._value("amp", "auto"),
    ]
    rebuild = self.vars.get("rebuild")
    if rebuild is not None and bool(rebuild.get()):
        values.append("--rebuild-dataset")
    return values


def _args(self: base.App, stage_id: str) -> list[str]:
    if stage_id == "capacity":
        return [
            *_common_mini_args(self),
            "--steps",
            self._value("capacity_steps", "3072"),
            "--learning-rate",
            self._value("capacity_lr", "0.001"),
            "--report-every",
            self._value("capacity_report_every", "32"),
            "--required-edge-recovery",
            self._value("edge_recovery", "0.60"),
            "--required-global-recovery",
            self._value("global_recovery", "0.45"),
            "--required-gradient-recovery",
            self._value("gradient_recovery", "0.35"),
        ]
    if stage_id == "multiregion":
        return [
            *_common_mini_args(self),
            "--train-regions",
            self._value("mini_train_regions", "4"),
            "--validation-regions",
            self._value("mini_validation_regions", "4"),
            "--prepare-train-regions",
            self._value("prepare_train_regions", "16"),
            "--prepare-validation-regions",
            self._value("prepare_validation_regions", "4"),
            "--epochs",
            self._value("mini_epochs", "3"),
            "--tiles-per-epoch",
            self._value("mini_tiles", "64"),
            "--learning-rate",
            self._value("mini_lr", "0.0002"),
            "--required-edge-recovery",
            self._value("edge_recovery", "0.60"),
            "--required-global-recovery",
            self._value("global_recovery", "0.45"),
            "--required-gradient-recovery",
            self._value("gradient_recovery", "0.35"),
        ]
    if stage_id == "selector":
        return [
            *_common_mini_args(self),
            "--selector-epochs",
            self._value("selector_epochs", "2"),
            "--tiles-per-epoch",
            self._value("selector_tiles", "64"),
            "--selector-learning-rate",
            self._value("selector_lr", "0.0002"),
            "--required-retention",
            self._value("selector_retention", "0.90"),
            "--protected-preservation",
            self._value("protected_preservation", "0.99"),
        ]

    values = _original_args(self, stage_id)
    if stage_id != "quick":
        return values
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] == "--control" and index + 1 < len(values):
            cleaned.extend(("--control", "auto"))
            index += 2
            continue
        cleaned.append(values[index])
        index += 1
    return cleaned


def _dispatcher_argv(
    self: base.App,
    command: tuple[str, ...],
    args: list[str],
) -> list[str]:
    if command == ("v14-mini-capacity",):
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/v14/capacity_diagnostic.py"),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    mode_by_command = {
        ("v14-mini-multiregion",): "multiregion",
        ("v14-mini-selector",): "selector",
    }
    mode = mode_by_command.get(command)
    if mode is not None:
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/v14/mini_diagnostics.py"),
            "--repo-root",
            str(self.repo),
            "--mode",
            mode,
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _select_capacity(self: base.App, stage: base.Stage, status: str) -> None:
    note = (
        " - rerun starts a fresh non-promotable diagnostic"
        if status in {"failed", "interrupted", "completed"}
        else ""
    )
    self.description.set(f"{stage.number}. {stage.label} - {stage.description}{note}")
    self._clear_form()
    self._label_row(
        "Authority",
        "DIAGNOSTIC ONLY - cannot create or promote a production final",
    )
    self._label_row(
        "Model",
        "Exact V14.3 stable phase-neutral multi-scale RCAN candidate C",
    )
    self._label_row(
        "Stability",
        "Identity-initialized residual groups + softsign residuals + pre-projection residual supervision",
    )
    self._label_row(
        "Divergence guard",
        "Stops on non-finite values, persistent zero gradient, or persistent >95% residual saturation",
    )
    self._label_row("Forbidden paths", "No PixelShuffle and no ConvTranspose2d")
    self._label_row("Geometry", "128 LR -> 512 authored HR")
    self._label_row("Purpose", "C must materially beat B without LR-grid imprint")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Maximum steps", "capacity_steps", "3072", ("1024", "2048", "3072"))
    self._row("Learning rate", "capacity_lr", "0.001", ("0.0003", "0.001"))
    self._row("Report interval", "capacity_report_every", "32", ("16", "32", "64"))
    self._row("Required edge recovery", "edge_recovery", "0.60", ("0.45", "0.60", "0.70"))
    self._row("Required global recovery", "global_recovery", "0.45", ("0.30", "0.45", "0.55"))
    self._row("Required gradient recovery", "gradient_recovery", "0.35", ("0.20", "0.35", "0.45"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._label_row(
        "Telemetry",
        "1/2/4-pixel detail recovery, residual saturation, gradient norm, and VRAM",
    )
    self._label_row(
        "Artifacts",
        "A/B/C images, curve, best checkpoint, last stable checkpoint, report, and diagnostics ZIP",
    )


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    lock = _stage_lock_reason(self, "multiregion")
    note = f" - LOCKED: {lock}" if lock else ""
    self.description.set(f"{stage.number}. {stage.label} - {stage.description}{note}")
    self._clear_form()
    self._label_row(
        "Authority",
        "DIAGNOSTIC ONLY - cannot create or promote a production final",
    )
    self._label_row("Prerequisite", f"{VERSION} HR Residual Capacity must pass")
    self._label_row("Model", f"Same exact {VERSION} candidate C as Raven Quick")
    self._label_row(
        "Purpose",
        "Check whether local SR gain survives across disjoint Raven regions",
    )
    self._label_row("Training", "Clean SR only; robustness is excluded from this mini proof")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Training regions", "mini_train_regions", "4", ("2", "4", "8"))
    self._row("Held-out regions", "mini_validation_regions", "4", ("2", "4"))
    self._row("Dataset train cap", "prepare_train_regions", "16", ("8", "16"))
    self._row("Dataset held-out cap", "prepare_validation_regions", "4", ("2", "4"))
    self._row("Epochs", "mini_epochs", "3", ("2", "3", "4"))
    self._row("Tiles per epoch", "mini_tiles", "64", ("32", "64", "96", "128"))
    self._row("Learning rate", "mini_lr", "0.0002", ("0.0001", "0.0002", "0.0003"))
    self._row("Required edge recovery", "edge_recovery", "0.60", ("0.45", "0.60", "0.70"))
    self._row("Required global recovery", "global_recovery", "0.45", ("0.30", "0.45", "0.55"))
    self._row("Required gradient recovery", "gradient_recovery", "0.35", ("0.20", "0.35", "0.45"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)


def _select_selector(self: base.App, stage: base.Stage, status: str) -> None:
    lock = _stage_lock_reason(self, "selector")
    note = f" - LOCKED: {lock}" if lock else ""
    self.description.set(f"{stage.number}. {stage.label} - {stage.description}{note}")
    self._clear_form()
    self._label_row(
        "Authority",
        "DIAGNOSTIC ONLY - cannot create or promote a production final",
    )
    self._label_row("Prerequisite", f"Latest passing {VERSION} Multi-Region SR Mini")
    self._label_row("Candidate", f"Frozen {VERSION} C; only BenefitSelector trains")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Selector epochs", "selector_epochs", "2", ("1", "2", "3"))
    self._row("Tiles per epoch", "selector_tiles", "64", ("32", "64", "96"))
    self._row("Selector learning rate", "selector_lr", "0.0002", ("0.0001", "0.0002", "0.0003"))
    self._row("Required recovery retention", "selector_retention", "0.90", ("0.85", "0.90", "0.95"))
    self._row("Protected-B preservation", "protected_preservation", "0.99", ("0.98", "0.99", "0.995"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)


def _selected(self: base.App) -> None:
    selection = self.tree.selection()
    if not selection:
        return
    stage_id = selection[0]
    if stage_id not in {"capacity", "multiregion", "selector"}:
        _original_selected(self)
        if stage_id == "quick":
            self.description.set(
                "4. V14.3 HR-First Raven Quick - stable phase-neutral LR context, "
                "multi-scale RCAN HR reconstruction, held-out qualification, then "
                "BenefitSelector only after C passes."
            )
            if "control" in self.vars:
                self.vars["control"].set("auto")
        return

    stage = base.BY_ID[stage_id]
    self.state["current"] = stage_id
    self._save()
    status = self.state["status"].get(stage_id, "pending")
    if stage_id == "capacity":
        _select_capacity(self, stage, status)
    elif stage_id == "multiregion":
        _select_multiregion(self, stage, status)
    else:
        _select_selector(self, stage, status)
    self._update_command()
    self.form_canvas.yview_moveto(0.0)
    self.root.after_idle(self._form_content_configured)


base.App._experiment_ids = _experiment_ids
base.App._stage_lock_reason = _stage_lock_reason
base.App._args = _args
base.App._dispatcher_argv = _dispatcher_argv
base.App._selected = _selected


if __name__ == "__main__":
    raise SystemExit(base.main())
