#!/usr/bin/env python3
"""GUI extension exposing the non-promotable Raven micro-capacity proof."""
from __future__ import annotations

import sys

import nsamdr_v9_workflow_gui as base


_existing = {stage.id: stage for stage in base.STAGES}
_micro = base.Stage(
    "micro",
    "3",
    "Raven Micro Capacity",
    ("raven-micro",),
    (
        "Fast non-promotable overfit proof on one deterministic edge-dense Raven region. "
        "Uses the production architecture/loss path and full production phase schedule "
        "before the expensive whole-Raven preview."
    ),
    False,
)
_preview = base.Stage(
    "preview",
    "4",
    "Preview",
    ("preview",),
    "Preview only a completed qualified experiment from its immutable final checkpoint.",
    False,
)
base.STAGES = (
    _existing["setup"],
    _existing["quick"],
    _existing["train"],
    _micro,
    _preview,
)
base.BY_ID = {stage.id: stage for stage in base.STAGES}
base.PIPELINE = [stage.id for stage in base.STAGES if stage.pipeline]

_original_args = base.App._args
_original_dispatcher_argv = base.App._dispatcher_argv
_original_selected = base.App._selected


def _args(self: base.App, stage_id: str) -> list[str]:
    if stage_id != "micro":
        return _original_args(self, stage_id)
    values = [
        "--shared-cache",
        self._value("cache", r"C:\CCP\EVE"),
        "--tile-size",
        self._value("micro_tile", "32"),
        "--steps-per-epoch",
        self._value("micro_steps", "64"),
        "--required-recovery",
        self._value("micro_recovery", "0.50"),
        "--device",
        self._value("device", "cuda"),
        "--amp-precision",
        self._value("amp", "auto"),
    ]
    rebuild = self.vars.get("rebuild")
    if rebuild is not None and bool(rebuild.get()):
        values.append("--rebuild-dataset")
    open_result = self.vars.get("open_result")
    if open_result is not None and bool(open_result.get()):
        values.append("--open-result")
    return values


def _dispatcher_argv(
    self: base.App,
    command: tuple[str, ...],
    args: list[str],
) -> list[str]:
    if command == ("raven-micro",):
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_overfit.py"),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _selected(self: base.App) -> None:
    selection = self.tree.selection()
    if not selection or selection[0] != "micro":
        _original_selected(self)
        return

    stage_id = "micro"
    stage = base.BY_ID[stage_id]
    self.state["current"] = stage_id
    self._save()
    status = self.state["status"].get(stage_id, "pending")
    note = " — interrupted; rerun starts a fresh diagnostic" if status == "interrupted" else ""
    self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
    self._clear_form()

    self._label_row("Authority", "DIAGNOSTIC ONLY — cannot create/promote a production final")
    self._label_row("Model", "Exact production NSAMDR architecture + current production losses")
    self._label_row("Region", "Deterministic highest edge-energy authored Raven patch")
    self._label_row("Epoch schedule", "Full production: B1a/B1b/seam/gate/detail/physical-finetune")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Micro LR tile", "micro_tile", "32", ("32", "48", "64"))
    self._row("Repeated steps / epoch", "micro_steps", "64", ("32", "64", "128", "256"))
    self._row("Required edge recovery", "micro_recovery", "0.50", ("0.25", "0.50", "0.70", "0.85"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._check("Open final lightweight A/B/C image", "open_result", True)
    self._label_row(
        "Output",
        "A/B/C PNG + per-epoch metrics + MICRO_*_DIAGNOSTICS.zip",
    )
    self._update_command()
    self.form_canvas.yview_moveto(0.0)
    self.root.after_idle(self._form_content_configured)


base.App._args = _args
base.App._dispatcher_argv = _dispatcher_argv
base.App._selected = _selected


if __name__ == "__main__":
    base.main()
