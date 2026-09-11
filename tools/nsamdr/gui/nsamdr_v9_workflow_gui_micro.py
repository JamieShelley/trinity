#!/usr/bin/env python3
"""GUI extension exposing focused Raven SR diagnostics and V13 production flow."""
from __future__ import annotations

import json
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
    "V13.1 SR Visual Fidelity",
    ("raven-micro",),
    (
        "Fast SR-first proof on one deterministic edge-dense Raven region. The production "
        "candidate is deterministic B plus a learned 4x multi-map residual conditioned only "
        "on observable LR evidence; BenefitSelector provides the final conservative B-to-C blend."
    ),
    False,
)
_quick = base.Stage(
    "quick",
    "4",
    "V13.2 SR Raven Quick",
    _existing["quick"].command,
    (
        "Train only the SR candidate C and BenefitSelector F on representative Raven crops, "
        "then require a deterministic 32-patch held-out visual-fidelity qualification. "
        "The retired G/profile/seam curriculum cannot run or be resumed in Quick."
    ),
    True,
)
_train = base.Stage(
    "train",
    "5",
    "Full Training (V13 conversion pending)",
    _existing["train"].command,
    (
        "Temporarily disabled because Full still contains the retired geometry/profile/seam "
        "curriculum. It will be re-enabled only after the V13.2 Quick proof passes and Full "
        "is converted to the same SR-first authority."
    ),
    False,
)
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
_original_stage_lock_reason = base.App._stage_lock_reason


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
            "--steps",
            self._value("micro_sr_steps", "3072"),
            "--selector-steps",
            self._value("micro_selector_steps", "768"),
            "--learning-rate",
            self._value("micro_sr_lr", "0.001"),
            "--selector-learning-rate",
            self._value("micro_selector_lr", "0.001"),
            "--required-edge-recovery",
            self._value("micro_edge_recovery", "0.70"),
            "--required-global-recovery",
            self._value("micro_global_recovery", "0.50"),
            "--required-retention",
            self._value("micro_retention", "0.90"),
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
    self._label_row("Oracle", "Sweeps residual amplitude first and reports whether the current production cap is mathematically sufficient")
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
    self._label_row("Authority", "DIAGNOSTIC ONLY — fast V13.1 visual-fidelity/capacity proof")
    self._label_row("Region", "Deterministic highest edge-energy authored Raven patch")
    self._label_row("Production candidate", "C = deterministic B + bounded 4x multi-map SR residual")
    self._label_row("Albedo residual cap", "0.40")
    self._label_row("Conditioning", "17-channel LR evidence + observable LR SDF/gradients/normal/material edges; no learned G/P/S pixel authority")
    self._label_row("Outputs trained", "Albedo RGB + normal XY + material/emissive/roughness residuals")
    self._label_row("Visual objective", "Global + edge-weighted reconstruction, gradients, Laplacian/high-frequency fidelity and baseline regret")
    self._label_row("Quality target", ">=70% edge, >=50% global, >=40% gradient recovery")
    self._label_row("Final authority", "BenefitSelector chooses a conservative per-pixel blend B <-> C")
    self._label_row("Protection", ">=99% of baseline-correct protected pixels must remain within the one-code-value drift contract")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("SR LR tile", "micro_tile", "32", ("32", "48", "64"))
    self._row("SR maximum steps", "micro_sr_steps", "3072", ("1024", "1536", "2048", "3072", "4096"))
    self._row("Selector maximum steps", "micro_selector_steps", "768", ("256", "512", "768", "1024"))
    self._row("SR learning rate", "micro_sr_lr", "0.001", ("0.0003", "0.001", "0.003"))
    self._row("Selector learning rate", "micro_selector_lr", "0.001", ("0.0003", "0.001", "0.003"))
    self._row("Required edge recovery", "micro_edge_recovery", "0.70", ("0.50", "0.60", "0.70", "0.80"))
    self._row("Required global recovery", "micro_global_recovery", "0.50", ("0.35", "0.45", "0.50", "0.60"))
    self._row("Required selector retention", "micro_retention", "0.90", ("0.85", "0.90", "0.95"))
    self._row("Device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._row("AMP precision", "amp", "auto", ("auto", "bf16", "fp16"))
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._check("Open final visual-fidelity probe", "open_result", True)
    self._label_row(
        "Output",
        "TARGET/B/C/F/error probe + albedo/gradient/normal/material metrics + sr_first_report.json + SR_*_DIAGNOSTICS.zip",
    )


def _config_value(payload: dict, snake: str, camel: str, default: object = None) -> object:
    if snake in payload:
        return payload[snake]
    return payload.get(camel, default)


def _is_v132_quick_experiment(self: base.App, experiment_id: str) -> bool:
    directory = self._experiments_root() / experiment_id
    try:
        experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
        if str(experiment.get("trainingMode") or "").lower() != "quick":
            return False
        zero_fields = (
            ("identity_epochs", "identityEpochs"),
            ("residual_epochs", "residualEpochs"),
            ("seam_proof_epochs", "seamProofEpochs"),
            ("seam_authority_epochs", "seamAuthorityEpochs"),
            ("boundary_epochs", "boundaryEpochs"),
        )
        if any(int(_config_value(config, snake, camel, 1)) != 0 for snake, camel in zero_fields):
            return False
        if int(_config_value(config, "detail_epochs", "detailEpochs", 0)) <= 0:
            return False
        if int(_config_value(config, "physical_finetune_epochs", "physicalFinetuneEpochs", 0)) <= 0:
            return False
        if float(_config_value(config, "detail_albedo_max_delta", "detailAlbedoMaxDelta", 0.0)) < 0.40 - 1.0e-8:
            return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def _v132_quick_experiment_ids(self: base.App) -> list[str]:
    return [
        experiment_id
        for experiment_id in self._experiment_ids()
        if _is_v132_quick_experiment(self, experiment_id)
    ]


def _select_quick(self: base.App, stage: base.Stage, status: str) -> None:
    note = " — interrupted; SR-first resume available" if status == "interrupted" else ""
    self.description.set(f"{stage.number}. {stage.label} — {stage.description}{note}")
    self._clear_form()
    experiments = ["new", *_v132_quick_experiment_ids(self)]
    self._row("Experiment", "experiment", "new", experiments)
    self._label_row("Production authority", "B -> multi-map SR candidate C -> BenefitSelector F")
    self._label_row("Retired specialists", "G/profile/seam are frozen compatibility/evidence only; zero Quick optimiser authority")
    self._label_row("SR work budget", "8 x 384 Raven patches = 3072 maximum SR updates at 32x32 LR")
    self._label_row("Selector budget", "3 x 384 patches = 1152 maximum selector updates")
    self._label_row("Held-out qualification", "32 deterministic Raven patches; median edge/global/gradient + catastrophe + preservation gates")
    self._label_row("Resume policy", "Only V13.2 SR-first Quick experiments are listed; older Quick experiments are rejected")
    self._row("Shared cache", "cache", r"C:\CCP\EVE")
    self._row("Training regions", "train_crops", "16")
    self._row("Max held-out regions", "validation_crops", "4")
    self._check("Rebuild fixed Raven dataset", "rebuild", False)
    self._row("Training control", "control", "auto", ("auto", "resume"))
    self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))
    self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))
    self._check("Live EVE A/B preview while training", "live_preview", True)
    self._row("Live preview target", "live_target", "1024", ("512", "1024", "2048"))
    self._training_performance_rows(default_workers="4")


def _stage_lock_reason(self: base.App, stage_id: str) -> str | None:
    if stage_id == "train":
        return (
            "Full Training is temporarily disabled because it still contains the retired "
            "geometry/profile/seam curriculum. Run V13.2 SR Raven Quick first; Full will be "
            "re-enabled only after it is converted to SR-first."
        )
    return _original_stage_lock_reason(self, stage_id)


def _selected(self: base.App) -> None:
    selection = self.tree.selection()
    if not selection or selection[0] not in {"direct", "parallel", "micro", "quick"}:
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
    elif stage_id == "micro":
        _select_micro(self, stage, status)
    else:
        _select_quick(self, stage, status)
    self._update_command()
    self.form_canvas.yview_moveto(0.0)
    self.root.after_idle(self._form_content_configured)


base.App._args = _args
base.App._dispatcher_argv = _dispatcher_argv
base.App._stage_lock_reason = _stage_lock_reason
base.App._selected = _selected


if __name__ == "__main__":
    base.main()
