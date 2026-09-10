#!/usr/bin/env python3
"""GUI extension exposing Raven direct-residual and staged micro diagnostics."""
from __future__ import annotations

import sys

import nsamdr_v9_workflow_gui as base


_existing = {stage.id: stage for stage in base.STAGES}


def _renumber(stage: base.Stage, number: str) -> base.Stage:
    return base.Stage(
        stage.id,
        number,
        stage.label,
        stage.command,
        stage.description,
        stage.pipeline,
    )


_direct = base.Stage(
    "direct",
    "1",
    "Direct Residual Capacity",
    ("raven-direct-residual",),
    (
        "Fast non-promotable proof that the production detail decoder can independently "
        "recover authored Raven detail directly over deterministic baseline B. Geometry, "
        "seam authority, profile gating and BenefitSelector are deliberately bypassed."
    ),
    False,
)
_micro = base.Stage(
    "micro",
    "2",
    "Raven Staged Micro",
    ("raven-micro",),
    (
        "Non-promotable staged production integration proof on one deterministic edge-dense "
        "Raven region. Run only after Direct Residual Capacity passes."
    ),
    False,
)
_quick = _renumber(_existing["quick"], "3")
_train = _renumber(_existing["train"], "4")
_preview = base.Stage(
    "preview",
    "5",
    "Preview",
    ("preview",),
    "Preview only a completed qualified experiment from its immutable final checkpoint.",
    False,
)
base.STAGES = (
    _existing["setup"],
    _direct,
    _micro,
    _quick,
    _train,
    _preview,
)
base.BY_ID = {stage.id: stage for stage in base.STAGES}
base.PIPELINE = [stage.id for stage in base.STAGES if stage.pipeline]

_original_args = base.App._args
_original_dispatcher_argv = base.App._dispatcher_argv
_original_selected = base.App._selected


def _common_flags(self: base.App) -> list[str]:
    values = [
        "--shared-cache",
        self._value("cache", r"C:\CCP\EVE"),
    ]
    rebuild = self.vars.get("rebuild")
    if rebuild is not None and bool(rebuild.get()):
        values.append("--rebuild-dataset")
    return values


def _args(self: base.App, stage_id: str) -> list[str]:
    if stage_id == "direct":
        values = [
            *_common_flags(self),
            "--tile-size",
            self._value("direct_tile", "32"),
            "--steps",
            self._value("direct_steps", "1536"),
            "--learning-rate",
            self._value("direct_lr", "0.001"),
            "--required-edge-recovery",
            self._value("direct_edge_recovery", "0.50"),
            "--required-global-recovery",
            self._value("direct_global_recovery", "0.25"),
            "--device",
            self._value("device", "cuda"),
            "--amp-precision",
            self._value("amp", "auto"),
        ]
        open_result = self.vars.get("open_result")
        if open_result is not None and bool(open_result.get()):
            values.append("--open-result")
        return values

    if stage_id == "micro":
        values = [
            *_common_flags(self),
            "--tile-size",
            self._value("micro_tile", "32"),
            "--steps-per-epoch",
            self._value("micro_steps", "32"),
            "--required-recovery",
            self._value("micro_recovery", "0.50"),
            "--device",
            self._value("device", "cuda"),
            "--amp-precision",
            self._value("amp", "auto"),
        ]
        open_result = self.vars.get("open_result")
        if open_result is not None and bool(open_result.get()):
            values.append("--open-result")
        return values

    return _original_args(self, stage_id)


def _dispatcher_argv(
    self: base.App,
    command: tuple[str, ...],
    args: list[str],
) -> list[str]:
    if command == ("raven-direct-residual",):
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/run_nsamdr_v9_raven_direct_residual_diagnostic.py"),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    if command == ("raven-micro",):
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _select_direct(self: base.App, stage: base.Stage, status: str) -> None:
    note = " — interrupted; rerun starts a fresh diagnostic" if status == "interrupted" else ""
    self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
    self._clear_form()
    self._label_row("Authority", "DIAGNOSTIC ONLY — cannot create/promote a production final")
    self._label_row("Network", "Existing production GeometryConditionedDetailNet")
    self._label_row("Input", "Native 17-channel LR evidence + deterministic production baseline B")
    self._label_row("Output", "B + bounded albedo residual; geometry/seam/selector authority bypassed")
    self._label_row("Oracle", "Sweeps residual amplitude first and reports whether the production 0.20 cap is mathematically sufficient")
    self._label_row("Capacity cap", "Automatically uses the smallest bounded residual amplitude capable of meeting the requested thresholds")
    self._label_row("Optimisation", "Identity-safe zero head gets 3x LR; direct residual target supervision removes the previous 250-step dead start")
    self._label_row("Stop rule", "Pass as soon as edge and global recovery thresholds are both reached")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Direct LR tile", "direct_tile", "32", ("32", "48", "64"))
    self._row("Maximum steps", "direct_steps", "1536", ("512", "1024", "1536", "2048", "3072"))
    self._row("Learning rate", "direct_lr", "0.001", ("0.0003", "0.001", "0.003"))
    self._row("Required edge recovery", "direct_edge_recovery", "0.50", ("0.35", "0.50", "0.70", "0.85"))
    self._row("Required global recovery", "direct_global_recovery", "0.25", ("0.10", "0.25", "0.50", "0.70"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._check("Open final lightweight probe image", "open_result", True)
    self._label_row(
        "Artifacts",
        "A/B/R/error probe PNG + cap-oracle sweep + step metrics + direct_residual_report.json + DIRECT_*_DIAGNOSTICS.zip",
    )


def _select_micro(self: base.App, stage: base.Stage, status: str) -> None:
    note = " — interrupted; rerun starts a fresh diagnostic" if status == "interrupted" else ""
    self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
    self._clear_form()
    self._label_row("Authority", "DIAGNOSTIC ONLY — cannot create/promote a production final")
    self._label_row("Prerequisite", "Direct Residual Capacity should pass first")
    self._label_row("Model", "Exact production NSAMDR architecture + current production losses")
    self._label_row("Region", "Deterministic highest edge-energy authored Raven patch")
    self._label_row("Probe", "geometry -> seam -> raw detail -> final selector authority")
    self._label_row("Epoch schedule", "Full production phase schedule on one repeated tiny patch")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Micro LR tile", "micro_tile", "32", ("32", "48", "64"))
    self._row("Repeated steps / epoch", "micro_steps", "32", ("16", "32", "64", "128"))
    self._row("Required edge recovery", "micro_recovery", "0.50", ("0.25", "0.50", "0.70", "0.85"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._check("Open final lightweight probe image", "open_result", True)
    self._label_row(
        "Output",
        "A/B/G/D/F probe PNG + per-epoch candidate/gate metrics + MICRO_*_DIAGNOSTICS.zip",
    )


def _selected(self: base.App) -> None:
    selection = self.tree.selection()
    if not selection or selection[0] not in {"direct", "micro"}:
        _original_selected(self)
        return

    stage_id = selection[0]
    stage = base.BY_ID[stage_id]
    self.state["current"] = stage_id
    self._save()
    status = self.state["status"].get(stage_id, "pending")
    if stage_id == "direct":
        _select_direct(self, stage, status)
    else:
        _select_micro(self, stage, status)
    self._update_command()
    self.form_canvas.yview_moveto(0.0)
    self.root.after_idle(self._form_content_configured)


base.App._args = _args
base.App._dispatcher_argv = _dispatcher_argv
base.App._selected = _selected


if __name__ == "__main__":
    base.main()
