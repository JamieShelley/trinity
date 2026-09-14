#!/usr/bin/env python3
"""NSAMDR V16.0 GUI surface.

This wrapper reuses the mature workflow shell and exposes the active V16.0 SwinIR-style
candidate contract. Historical V14/V15 candidates cannot unlock V16 stages.
"""
from __future__ import annotations

import json

import nsamdr_v14_workflow_gui as legacy


VERSION = "V16.0"
MINI_SCHEMA = "NSAMDR_V16_MINI_DIAGNOSTIC_V1"
legacy.VERSION = VERSION
legacy.MINI_SCHEMA = MINI_SCHEMA
base = legacy.base
base.APP_TITLE = f"NSAMDR {VERSION} HR-First Workflow"

_existing = {stage.id: stage for stage in base.STAGES}
base.STAGES = (
    base.Stage(
        "setup",
        "P0",
        "Prepare CUDA environment",
        _existing["setup"].command,
        "Create or verify the CUDA Python environment.",
        False,
    ),
    base.Stage(
        "capacity",
        "1",
        f"{VERSION} HR Residual Capacity",
        _existing["capacity"].command,
        (
            "Fast non-promotable overfit proof on one deterministic high-detail Raven "
            "region. Tests the fixed-HR SwinIR-style candidate C."
        ),
        False,
    ),
    base.Stage(
        "multiregion",
        "2",
        f"{VERSION} Multi-Region SR Mini",
        _existing["multiregion"].command,
        "Disjoint-region generalisation proof for the exact V16.0 candidate C.",
        False,
    ),
    base.Stage(
        "selector",
        "3",
        f"{VERSION} Selector Retention Mini",
        _existing["selector"].command,
        "Train only BenefitSelector from the latest passing V16.0 candidate.",
        False,
    ),
    base.Stage(
        "quick",
        "4",
        f"{VERSION} HR-First Raven Quick",
        _existing["quick"].command,
        (
            "Train the phase-neutral fixed-HR SwinIR-style candidate C on Raven evidence. "
            "Train BenefitSelector only after C qualifies."
        ),
        True,
    ),
    base.Stage(
        "train",
        "5",
        "Full Training (disabled)",
        _existing["train"].command,
        "Disabled until V16.0 Raven Quick meets the README qualification gates.",
        False,
    ),
    base.Stage(
        "preview",
        "6",
        f"Qualified {VERSION} Preview",
        _existing["preview"].command,
        "Bake and inspect 4x physical maps from a qualified immutable V16.0 checkpoint.",
        False,
    ),
)
base.BY_ID = {stage.id: stage for stage in base.STAGES}


def _mini_passed(self: legacy.V14ArtifactInspector, mode: str) -> bool:
    root = self.app.repo / "artifacts/nsamdr/diagnostics/v16_mini"
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
            and payload.get("modelSchema") == legacy.MODEL_SCHEMA
            and payload.get("passed") is True
        ):
            return True
    return False


legacy.V14ArtifactInspector.mini_passed = _mini_passed


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
        "V16.0 fixed-HR SwinIR-style candidate C",
    )
    self._label_row(
        "Backbone",
        "96 HR channels, 6 residual Swin groups x 6 layers, window 8, 6 heads",
    )
    self._label_row(
        "Attention",
        "Alternating regular and shifted-window self-attention with relative position bias",
    )
    self._label_row(
        "Spatial rule",
        "No HR feature pyramid or decoder. All learned candidate reconstruction stays at HR.",
    )
    self._label_row(
        "Optimizer",
        "Adam at the production SR learning rate: 0.0002",
    )
    self._label_row(
        "Residual output",
        "Zero-initialized physical-map heads with tanh residual bounds",
    )
    self._label_row(
        "Divergence guard",
        "Stops on non-finite values, persistent zero gradient, or persistent >95% saturation",
    )
    self._label_row(
        "Forbidden paths",
        "No PixelShuffle, ConvTranspose2d, BatchNorm, GAN, or HR encoder/decoder pyramid",
    )
    self._label_row("Geometry", "128 LR -> 512 authored HR")
    self._label_row("Purpose", "C must materially beat B without LR-grid imprint")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Maximum steps", "capacity_steps", "3072", ("1024", "2048", "3072"))
    self._row("Learning rate", "capacity_lr", "0.0002", ("0.0001", "0.0002", "0.0003"))
    self._row("Report interval", "capacity_report_every", "32", ("16", "32", "64"))
    self._row("Required edge recovery", "edge_recovery", "0.60", ("0.45", "0.60", "0.70"))
    self._row("Required global recovery", "global_recovery", "0.45", ("0.30", "0.45", "0.55"))
    self._row("Required gradient recovery", "gradient_recovery", "0.35", ("0.20", "0.35", "0.45"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._label_row(
        "Telemetry",
        "1/2/4-pixel detail, residual saturation, raw residual magnitude, gradient norm, and VRAM",
    )


legacy._select_capacity = _select_capacity
_legacy_selected = legacy._selected


def _selected(self: base.App) -> None:
    _legacy_selected(self)
    selection = self.tree.selection()
    if selection and selection[0] == "quick":
        self.description.set(
            "4. V16.0 HR-First Raven Quick - phase-neutral LR context, "
            "fixed-HR SwinIR-style deep feature extraction, held-out qualification, "
            "then BenefitSelector only after C passes."
        )


base.App._selected = _selected


if __name__ == "__main__":
    raise SystemExit(base.main())
