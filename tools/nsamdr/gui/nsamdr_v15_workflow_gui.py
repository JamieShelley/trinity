#!/usr/bin/env python3
"""NSAMDR V15.0 GUI surface.

This wrapper reuses the mature workflow UI shell. It replaces the active stage labels and
Capacity description with the V15.0 single-resolution EDSR candidate contract.
"""
from __future__ import annotations

import nsamdr_v14_workflow_gui as legacy


VERSION = "V15.0"
legacy.VERSION = VERSION
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
            "region. Tests the single-resolution EDSR-style candidate C."
        ),
        False,
    ),
    base.Stage(
        "multiregion",
        "2",
        f"{VERSION} Multi-Region SR Mini",
        _existing["multiregion"].command,
        "Disjoint-region generalisation proof for the exact V15.0 candidate C.",
        False,
    ),
    base.Stage(
        "selector",
        "3",
        f"{VERSION} Selector Retention Mini",
        _existing["selector"].command,
        "Train only BenefitSelector from the latest passing V15.0 candidate.",
        False,
    ),
    base.Stage(
        "quick",
        "4",
        f"{VERSION} HR-First Raven Quick",
        _existing["quick"].command,
        (
            "Train the single-resolution phase-neutral 4x candidate C on Raven evidence. "
            "Train BenefitSelector only after C qualifies."
        ),
        True,
    ),
    base.Stage(
        "train",
        "5",
        "Full Training (disabled)",
        _existing["train"].command,
        "Disabled until V15.0 Raven Quick meets the README qualification gates.",
        False,
    ),
    base.Stage(
        "preview",
        "6",
        f"Qualified {VERSION} Preview",
        _existing["preview"].command,
        "Bake and inspect 4x physical maps from a qualified immutable V15.0 checkpoint.",
        False,
    ),
)
base.BY_ID = {stage.id: stage for stage in base.STAGES}


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
        "V15.0 single-resolution EDSR-style HR candidate C",
    )
    self._label_row(
        "Backbone",
        "64 HR channels, 24 Conv-ReLU-Conv residual blocks, residual scale 0.10, no BN",
    )
    self._label_row(
        "Spatial rule",
        "No feature downsampling or decoder. All learned candidate reconstruction stays at HR.",
    )
    self._label_row(
        "Optimizer",
        "Adam at the production SR learning rate: 0.0002",
    )
    self._label_row(
        "Residual output",
        "Zero-initialized map heads with tanh physical residual bounds",
    )
    self._label_row(
        "Divergence guard",
        "Stops on non-finite values, persistent zero gradient, or persistent >95% saturation",
    )
    self._label_row("Forbidden paths", "No PixelShuffle, ConvTranspose2d, BN, or multi-scale feature decoder")
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
            "4. V15.0 HR-First Raven Quick - phase-neutral LR context, "
            "single-resolution EDSR-style HR reconstruction, held-out qualification, "
            "then BenefitSelector only after C passes."
        )


base.App._selected = _selected


if __name__ == "__main__":
    raise SystemExit(base.main())
