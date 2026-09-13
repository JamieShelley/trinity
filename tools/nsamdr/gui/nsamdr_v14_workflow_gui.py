#!/usr/bin/env python3
"""NSAMDR V14 GUI surface.

Historical V9-V13 diagnostic stages are deliberately absent. The UI exposes only the
production V14 path: Raven Quick, disabled Full Training, and immutable V14 Preview.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import nsamdr_v9_workflow_gui as base


V14_MODEL_SCHEMA = "NSAMDR_HR_FIRST_MULTI_MAP_SR_4X_V14_0"
V14_EXPERIMENT_SCHEMA = "NSAMDR_V14_EXPERIMENT_V1"
V14_FINAL_SCHEMA = "NSAMDR_V14_FINAL_MANIFEST_V1"

base.APP_TITLE = "NSAMDR V14 HR-First Workflow"
_existing = {stage.id: stage for stage in base.STAGES}

_setup = base.Stage(
    "setup", "P0", "Prepare CUDA environment", _existing["setup"].command,
    "Create or verify the CUDA Python environment.", False,
)
_quick = base.Stage(
    "quick", "1", "V14 HR-First Raven Quick", _existing["quick"].command,
    (
        "Train the clean HR-first 4x SR candidate C on native authored Raven evidence, "
        "require held-out candidate qualification, then train BenefitSelector F only if C passes."
    ), True,
)
_train = base.Stage(
    "train", "2", "Full Training (disabled)", _existing["train"].command,
    (
        "Disabled until the V14 HR-first Raven candidate meets the README qualification gates. "
        "Full will use the same model/checkpoint contract with only a larger authored dataset and work budget."
    ), False,
)
_preview = base.Stage(
    "preview", "3", "Qualified V14 Preview", _existing["preview"].command,
    "Bake and inspect 4x physical maps from an immutable qualified V14 checkpoint.", False,
)

base.STAGES = (_setup, _quick, _train, _preview)
base.BY_ID = {stage.id: stage for stage in base.STAGES}
base.PIPELINE = ["quick"]

_original_stage_lock_reason = base.App._stage_lock_reason
_original_selected = base.App._selected
_original_args = base.App._args


def _is_v14_experiment(directory: Path) -> bool:
    try:
        payload = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        payload.get("schema") == V14_EXPERIMENT_SCHEMA
        and payload.get("modelSchema") == V14_MODEL_SCHEMA
    )


def _qualified_v14_final(directory: Path) -> bool:
    try:
        experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        final = json.loads((directory / "final_manifest.json").read_text(encoding="utf-8"))
        checkpoint = dict(final.get("checkpoint") or {})
        checkpoint_path = (directory / str(checkpoint.get("path") or "")).resolve()
        checkpoint_path.relative_to((directory / "checkpoints" / "final").resolve())
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        experiment.get("schema") == V14_EXPERIMENT_SCHEMA
        and experiment.get("modelSchema") == V14_MODEL_SCHEMA
        and experiment.get("status") == "completed"
        and experiment.get("qualified") is True
        and final.get("schema") == V14_FINAL_SCHEMA
        and final.get("status") == "completed"
        and final.get("qualified") is True
        and final.get("selectionKind") == "production-final"
        and checkpoint.get("schema") == V14_MODEL_SCHEMA
        and checkpoint.get("immutable") is True
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256") or "")))
        and checkpoint_path.is_file()
    )


def _experiment_ids(self: base.App, *, completed_only: bool = False) -> list[str]:
    # V14 experiments cannot resume. The non-completed listing is used only by the
    # Quick experiment picker, which must therefore expose "new" only. Qualified
    # immutable finals remain discoverable for Preview/status detection.
    if not completed_only:
        return []
    root = self._experiments_root()
    if not root.is_dir():
        return []
    result: list[str] = []
    for directory in root.iterdir():
        if not directory.is_dir() or base.EXPERIMENT_RE.fullmatch(directory.name) is None:
            continue
        if _is_v14_experiment(directory) and _qualified_v14_final(directory):
            result.append(directory.name.upper())
    return sorted(result, key=lambda value: int(value.split("_")[1]))


def _stage_lock_reason(self: base.App, stage_id: str) -> str | None:
    if stage_id == "train":
        return "Full Training is disabled until V14 Raven Quick qualifies."
    if stage_id == "preview" and not self._previewable_experiment_ids():
        return "Preview requires a completed qualified V14 EXP_#### final."
    return _original_stage_lock_reason(self, stage_id)


def _args(self: base.App, stage_id: str) -> list[str]:
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


def _selected(self: base.App) -> None:
    _original_selected(self)
    selection = self.tree.selection()
    if not selection:
        return
    stage_id = selection[0]
    if stage_id == "quick":
        self.description.set(
            "1. V14 HR-First Raven Quick — 128→512 clean/robust SR training plus native 256→1024 "
            "Raven qualification; BenefitSelector trains only after candidate C passes."
        )
        if "control" in self.vars:
            self.vars["control"].set("auto")
    elif stage_id == "train":
        self.description.set(
            "2. Full Training (disabled) — no legacy geometry/profile/seam curriculum remains available. "
            "Full will be enabled only after V14 Raven Quick qualifies."
        )


base.App._experiment_ids = _experiment_ids
base.App._stage_lock_reason = _stage_lock_reason
base.App._args = _args
base.App._selected = _selected


if __name__ == "__main__":
    raise SystemExit(base.main())
