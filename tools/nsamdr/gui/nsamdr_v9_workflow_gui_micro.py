#!/usr/bin/env python3
"""GUI extension exposing focused Raven capacity/integration diagnostics."""
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
_parallel = base.Stage(
    "parallel",
    "2",
    "Parallel Detail Integration",
    ("raven-parallel-detail",),
    (
        "Current production-forward proof that the successful direct-detail candidate survives "
        "composition. Structure/seam support is forced to zero so fused U must equal D exactly "
        "before BenefitSelector retention is measured with the current support evidence."
    ),
    False,
)
_micro = base.Stage(
    "micro",
    "3",
    "Raven Staged Micro",
    ("raven-micro",),
    (
        "V12.9 hard-gated qualification ladder on one deterministic edge-dense Raven region. "
        "G topology/geometry, rendered structure, profile, forced seam reconstruction, learned "
        "seam authority, D, U and F must each pass before downstream training may continue."
    ),
    False,
)
_quick = _renumber(_existing["quick"], "4")
_train = _renumber(_existing["train"], "5")
_preview = base.Stage(
    "preview",
    "6",
    "Preview",
    ("preview",),
    "Preview only a completed qualified experiment from its immutable final checkpoint.",
    False,
)
base.STAGES = (
    _existing["setup"],
    _direct,
    _parallel,
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
            self._value("direct_steps", "3072"),
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

    if stage_id == "parallel":
        values = [
            *_common_flags(self),
            "--tile-size",
            self._value("parallel_tile", "32"),
            "--detail-steps",
            self._value("parallel_detail_steps", "3072"),
            "--selector-steps",
            self._value("parallel_selector_steps", "512"),
            "--required-edge-recovery",
            self._value("parallel_edge_recovery", "0.50"),
            "--required-global-recovery",
            self._value("parallel_global_recovery", "0.25"),
            "--required-retention",
            self._value("parallel_retention", "0.85"),
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
            "--geometry-topology-steps",
            self._value("micro_geometry_topology_steps", "512"),
            "--geometry-steps",
            self._value("micro_geometry_steps", "3072"),
            "--profile-steps",
            self._value("micro_profile_steps", "1536"),
            "--seam-capacity-steps",
            self._value("micro_seam_capacity_steps", "1536"),
            "--seam-authority-steps",
            self._value("micro_seam_authority_steps", "1024"),
            "--detail-capacity-steps",
            self._value("micro_detail_capacity_steps", "3072"),
            "--selector-capacity-steps",
            self._value("micro_selector_capacity_steps", "1024"),
            "--required-recovery",
            self._value("micro_recovery", "0.50"),
            "--required-fusion-retention",
            self._value("micro_fusion_retention", "0.95"),
            "--required-final-retention",
            self._value("micro_final_retention", "0.85"),
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
    if command == ("raven-parallel-detail",):
        return [
            sys.executable,
            str(self.repo / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic.py"),
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
    self._label_row("Optimisation", "Identity-safe zero head gets 3x LR; run stops as soon as the requested recovery threshold is reached")
    self._label_row("Stop rule", "Pass as soon as edge and global recovery thresholds are both reached")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Direct LR tile", "direct_tile", "32", ("32", "48", "64"))
    self._row("Maximum steps", "direct_steps", "3072", ("1024", "1536", "2048", "3072", "4096"))
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


def _select_parallel(self: base.App, stage: base.Stage, status: str) -> None:
    note = " — interrupted; rerun starts a fresh diagnostic" if status == "interrupted" else ""
    self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
    self._clear_form()
    self._label_row("Authority", "DIAGNOSTIC ONLY — cannot create/promote a production final")
    self._label_row("Prerequisite", "Direct Residual Capacity must pass first")
    self._label_row("Production change", "Structure/seam support is forced to zero for this proof, therefore fused U must equal direct D exactly")
    self._label_row("Selector evidence", "B + isolated U(=D) + observable LR support + calibrated detail support / zero conflict")
    self._label_row("Geometry/seam", "Execute in the production graph but have zero fusion authority during D integration qualification")
    self._label_row("Pass 1", "Direct detail candidate reaches the same 50% edge / 25% global recovery target")
    self._label_row("Pass 2", "Production U equals D and BenefitSelector retains at least 85% of both recoveries")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Parallel LR tile", "parallel_tile", "32", ("32", "48", "64"))
    self._row("Detail steps", "parallel_detail_steps", "3072", ("1280", "1536", "2048", "3072", "4096"))
    self._row("Selector steps", "parallel_selector_steps", "512", ("128", "256", "384", "512", "768"))
    self._row("Required edge recovery", "parallel_edge_recovery", "0.50", ("0.35", "0.50", "0.70"))
    self._row("Required global recovery", "parallel_global_recovery", "0.25", ("0.10", "0.25", "0.50"))
    self._row("Required selector retention", "parallel_retention", "0.85", ("0.70", "0.85", "0.95"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._check("Open final lightweight probe image", "open_result", True)
    self._label_row(
        "Output",
        "A/B/D/F probe PNG + step metrics + parallel_detail_report.json + PARALLEL_*_DIAGNOSTICS.zip",
    )


def _select_micro(self: base.App, stage: base.Stage, status: str) -> None:
    note = " — interrupted; rerun starts a fresh diagnostic" if status == "interrupted" else ""
    self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
    self._clear_form()
    self._label_row("Authority", "DIAGNOSTIC ONLY — cannot create/promote a production final")
    self._label_row("Prerequisite", "Direct Residual and Parallel Detail Integration should pass first")
    self._label_row("Model", "Current V12.8 production graph; V12.9 owns diagnostic qualification only")
    self._label_row("Region", "Deterministic highest edge-energy authored Raven patch")
    self._label_row("Ladder", "G topology -> G metric/render -> profile -> forced S -> S authority -> D -> U -> F")
    self._label_row("Hard stop", "A failed stage stops immediately; downstream specialists cannot hide it")
    self._label_row("Best state", "Each stage saves its best metrics/probe/checkpoint and restores best state on failure")
    self._label_row("G qualification", "Production contour gain/win/regression/topology criteria + rendered G non-regression")
    self._label_row("Profile qualification", "Both target-SDF teacher geometry and learned-G profile must reach configured recovery")
    self._label_row("S qualification", "Forced seam recovery >= configured requirement, then learned authority IoU >= configured requirement")
    self._label_row("Final qualification", "D capacity + U retention + F retention + protected-B preservation >=99%")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Micro LR tile", "micro_tile", "32", ("32", "48", "64"))
    self._row("G topology max steps", "micro_geometry_topology_steps", "512", ("256", "512", "768", "1024"))
    self._row("G metric/render max steps", "micro_geometry_steps", "3072", ("1536", "2048", "3072", "4096"))
    self._row("Profile max steps", "micro_profile_steps", "1536", ("768", "1024", "1536", "2048"))
    self._row("Forced seam max steps", "micro_seam_capacity_steps", "1536", ("768", "1024", "1536", "2048"))
    self._row("Seam authority max steps", "micro_seam_authority_steps", "1024", ("512", "768", "1024", "1536"))
    self._row("Detail max steps", "micro_detail_capacity_steps", "3072", ("1536", "2048", "3072", "4096"))
    self._row("Selector max steps", "micro_selector_capacity_steps", "1024", ("512", "768", "1024", "1536"))
    self._row("Required D edge recovery", "micro_recovery", "0.50", ("0.35", "0.50", "0.70", "0.85"))
    self._row("Required U/D retention", "micro_fusion_retention", "0.95", ("0.85", "0.90", "0.95", "0.98"))
    self._row("Required F/U retention", "micro_final_retention", "0.85", ("0.70", "0.85", "0.90", "0.95"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._check("Open final/failure probe image", "open_result", True)
    self._label_row(
        "Output",
        "stage_summary.json + per-stage best_state/best_metrics/best_probe + A/B/G/S/D/U/F final/failure probe + MICRO_*_DIAGNOSTICS.zip",
    )


def _selected(self: base.App) -> None:
    selection = self.tree.selection()
    if not selection or selection[0] not in {"direct", "parallel", "micro"}:
        _original_selected(self)
        return

    stage_id = selection[0]
    stage = base.BY_ID[stage_id]
    self.state["current"] = stage_id
    self._save()
    status = self.state["status"].get(stage_id, "pending")
    if stage_id == "direct":
        _select_direct(self, stage, status)
    elif stage_id == "parallel":
        _select_parallel(self, stage, status)
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
