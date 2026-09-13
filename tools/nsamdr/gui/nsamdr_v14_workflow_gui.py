#!/usr/bin/env python3
"""NSAMDR V14 GUI surface.

The historical V9-V13 diagnostic stages are intentionally not exposed here. The GUI
shows only the production V14 path: Raven Quick, disabled Full Training, and Preview.
"""
from __future__ import annotations

import nsamdr_v9_workflow_gui as base


base.APP_TITLE = "NSAMDR V14 HR-First Workflow"
_existing = {stage.id: stage for stage in base.STAGES}

_setup = base.Stage(
    "setup",
    "P0",
    "Prepare CUDA environment",
    _existing["setup"].command,
    "Create or verify the CUDA Python environment.",
    False,
)
_quick = base.Stage(
    "quick",
    "1",
    "V14 HR-First Raven Quick",
    _existing["quick"].command,
    (
        "Train the clean HR-first 4x SR candidate C on native authored Raven evidence, "
        "require held-out candidate qualification, then train BenefitSelector F only if C passes."
    ),
    True,
)
_train = base.Stage(
    "train",
    "2",
    "Full Training (disabled)",
    _existing["train"].command,
    (
        "Disabled until the V14 HR-first Raven candidate meets the README qualification gates. "
        "Full will use the same model/checkpoint contract with only a larger authored dataset and work budget."
    ),
    False,
)
_preview = base.Stage(
    "preview",
    "3",
    "Qualified V14 Preview",
    _existing["preview"].command,
    "Bake and inspect 4x physical maps from an immutable qualified V14 checkpoint.",
    False,
)

base.STAGES = (_setup, _quick, _train, _preview)
base.BY_ID = {stage.id: stage for stage in base.STAGES}
base.PIPELINE = ["quick"]

_original_stage_lock_reason = base.App._stage_lock_reason
_original_selected = base.App._selected


def _stage_lock_reason(self: base.App, stage_id: str) -> str | None:
    if stage_id == "train":
        return "Full Training is disabled until V14 Raven Quick qualifies."
    return _original_stage_lock_reason(self, stage_id)


def _selected(self: base.App) -> None:
    _original_selected(self)
    selection = self.tree.selection()
    if not selection:
        return
    stage_id = selection[0]
    if stage_id == "quick":
        # Override legacy wording only; controls remain the proven base GUI controls.
        self.description.set(
            "1. V14 HR-First Raven Quick — 128→512 clean/robust SR training, native 256→1024 Raven "
            "qualification, then BenefitSelector training only after candidate C passes."
        )
        if "control" in self.vars:
            try:
                self.vars["control"].set("auto")
            except Exception:
                pass
    elif stage_id == "train":
        self.description.set(
            "2. Full Training (disabled) — no legacy geometry/profile/seam curriculum remains available. "
            "Full will be enabled only after V14 Raven Quick qualifies."
        )


base.App._stage_lock_reason = _stage_lock_reason
base.App._selected = _selected


if __name__ == "__main__":
    raise SystemExit(base.main())
