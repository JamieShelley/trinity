#!/usr/bin/env python3
"""Train one canonical NSAMDR experiment with a Quick or Full work budget.

Both modes resolve from the production config. Quick may replace only dataset
scope and work-budget/runtime fields; it cannot select alternate semantic flags,
modules, losses, checkpoints, or forward graphs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.config import V9Config
from v9.experiments import (
    DEFAULT_TUNING_ASSET_NAME,
    DEFAULT_TUNING_ASSET_QUERY,
    experiment_dir,
    finalise_experiment,
    initialise_experiment,
    load_experiment_manifest,
    load_resolved_config,
    write_experiment_manifest,
)
import v9.training as _v9_training
from v9.local_boundary_production_contract import install_local_boundary_training_contract
from v9.evolutionary_recovery import EvolutionaryRecoveryController, FailureKind, classify_failure

# Both Quick and Full call the same current V11 trainer. The installer swaps
# only the production structural authority and its existing local losses.
install_local_boundary_training_contract(_v9_training)
train_v9 = _v9_training.train_v9


DATASET_SCOPE_FIELDS = (
    "dataset_manifest",
    "dataset_root",
    "max_families",
    "crops_per_family",
    "source_crop_size",
    "min_source_dimension",
    "min_auxiliary_dimension",
    "validation_fraction",
    "require_complete_pbr_family",
)

# These are production semantic invariants, not Quick-only behavior. The current
# JSON predates the complete staged route, so both modes resolve the same values
# explicitly until the production config is updated.
CANONICAL_SEMANTIC_OVERRIDES: dict[str, Any] = {
    "appearance_enabled": True,
    "detail_reconstruction_enabled": True,
    "seam_directional_enabled": True,
    "raven_full_pipeline_preview_enabled": True,
    "raven_representative_preview_enabled": False,
    "raven_train_only_enabled": False,
    "preview_allow_unqualified_downstream": False,
}

# Raven Quick uses the same pass-driven production curriculum as Full. The
# reduced budget must still leave room for local geometry -> parameters ->
# integration; four joint B1b epochs were proven insufficient by EXP_0001.
#
# These remain work-budget changes only. Quick still uses the exact production
# model, losses, forward graph, qualification gates and checkpoint schema.
QUICK_WORK_BUDGET: dict[str, int] = {
    # V11.4 Quick proves the real local analytic geometry in one compact stage.
    # V9Config requires residual_epochs >= 1. That slot is a retired B1b slot
    # and is skipped only AFTER the real local geometry + redraw gate passes.
    "identity_epochs": 3,
    "residual_epochs": 1,
    "seam_proof_epochs": 1,
    "seam_authority_epochs": 1,
    "boundary_epochs": 1,
    "detail_epochs": 1,
    "physical_finetune_epochs": 1,
    "tiles_per_epoch": 64,
    "validation_tiles": 8,
    "raven_downstream_tiles_per_epoch": 16,
    # V9Config also requires this to be at least the micro-batch size even
    # though the retired global primitive loader is never entered by V11.4.
    "parametric_primitive_train_tiles_per_epoch": 14,
}

FULL_MINIMUM_WORK_BUDGET: dict[str, int] = {
    "identity_epochs": 12,
    "residual_epochs": 1,
    "seam_proof_epochs": 3,
    "seam_authority_epochs": 2,
    "boundary_epochs": 2,
    "detail_epochs": 5,
    "physical_finetune_epochs": 3,
    "raven_downstream_tiles_per_epoch": 128,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set_values(config: V9Config, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(config, key):
            raise RuntimeError(f"canonical workflow references unknown config field: {key}")
        setattr(config, key, value)


def _canonical_overrides(
    args: argparse.Namespace,
    base: V9Config,
    dataset_config: V9Config,
) -> dict[str, Any]:
    resolved = copy.deepcopy(base)
    if args.training_mode == "quick":
        _set_values(
            resolved,
            {field: getattr(dataset_config, field) for field in DATASET_SCOPE_FIELDS},
        )
    _set_values(resolved, CANONICAL_SEMANTIC_OVERRIDES)

    if args.training_mode == "quick":
        _set_values(resolved, QUICK_WORK_BUDGET)
    else:
        full_values = {
            field: max(int(getattr(resolved, field)), minimum)
            for field, minimum in FULL_MINIMUM_WORK_BUDGET.items()
        }
        # V9Config requires one residual slot. V11.4 never trains the retired
        # whole-tile classifier in that slot; the local-geometry promotion moves
        # the resume cursor across it after the real structural gate passes.
        full_values["residual_epochs"] = 1
        _set_values(resolved, full_values)

    # The current trainer still constructs the retired B1b DataLoader before the
    # phase loop. Keep its bank at the legal micro-batch minimum; it is never
    # iterated by the V11.4 pass-driven curriculum.
    resolved.parametric_primitive_train_tiles_per_epoch = max(
        int(getattr(resolved, "parametric_primitive_batch_size", 14)), 14
    )
    if args.tiles_per_epoch is not None:
        resolved.tiles_per_epoch = max(1, int(args.tiles_per_epoch))
    if args.validation_tiles is not None:
        resolved.validation_tiles = max(1, int(args.validation_tiles))

    resolved.apply_performance_profile(args.performance_profile)
    resolved.data_loader_workers = max(0, int(args.workers))
    resolved.data_loader_prefetch_factor = max(1, int(args.prefetch_factor))
    resolved.amp_dtype = args.amp_precision
    resolved.validate()

    base_payload = base.to_dict()
    resolved_payload = resolved.to_dict()
    return {
        key: value
        for key, value in resolved_payload.items()
        if base_payload.get(key) != value
    }


def _result_path(repo_root: Path, requested: Path | None) -> Path | None:
    if requested is None:
        return None
    return requested if requested.is_absolute() else repo_root / requested


def _write_result(repo_root: Path, requested: Path | None, payload: dict[str, Any]) -> None:
    path = _result_path(repo_root, requested)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the complete production NSAMDR model with a Quick or Full work budget"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("tools/nsamdr/neural/configs/v9_fidelity_full.json"),
        help="Production semantic config used by both Quick and Full",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="Dataset scope config; Quick normally uses v9_preview_raven.json",
    )
    parser.add_argument("--experiment", default="new", help="new or an in-progress EXP_####")
    parser.add_argument("--control", choices=("auto", "resume"), default="auto")
    parser.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--preset", default="Production")
    parser.add_argument("--tiles-per-epoch", type=int)
    parser.add_argument("--validation-tiles", type=int)
    parser.add_argument(
        "--performance-profile",
        choices=("optimized", "fast", "balanced", "compatibility"),
        default="optimized",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--allocate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop-after-phase",
        choices=(
            "sdf-bootstrap", "sdf-proof", "seam-proof", "seam-authority",
            "gate-proof", "detail-reconstruction", "boundary-hardening", "physical-finetune",
        ),
        help=argparse.SUPPRESS,
    )
    return parser



def _metric_float(metrics: dict[str, Any], key: str, default: float) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _gate_candidate_passed(metadata: dict[str, Any], config: V9Config) -> bool:
    """Re-evaluate the canonical gate-proof promotion rule from exported metrics.

    This duplicates only the boolean orchestration decision. The authoritative
    metrics and checkpoint are still produced by v9.training, and final
    qualification remains uncached/strict-load/fail-closed.
    """
    metrics = metadata.get("bestSyntheticSdfValidation")
    if not isinstance(metrics, dict) or not metrics:
        return False

    contour_gain = _metric_float(metrics, "sdf_zero_contour_relative_gain_mean", -1.0)
    contour_wins = _metric_float(metrics, "sdf_zero_contour_relative_win_fraction", 0.0)
    contour_regress = _metric_float(metrics, "sdf_zero_contour_relative_regression_fraction", 1.0)
    source_missing = _metric_float(metrics, "sdf_source_missing_contour_fraction", 1.0)
    predicted_missing = _metric_float(metrics, "sdf_predicted_missing_contour_fraction", 1.0)
    contour_chamfer = _metric_float(metrics, "sdf_zero_contour_chamfer_pixels", 999.0)
    topology_regression = _metric_float(metrics, "sdf_stageb_topology_regression_fraction", 1.0)
    line_jitter = _metric_float(metrics, "sdf_line_perpendicular_jitter_pixels_mean", 999.0)
    curve_roughness = _metric_float(metrics, "sdf_circle_radial_roughness_pixels_mean", 999.0)
    staircase_recovery = _metric_float(metrics, "sdf_line_staircase_recovery_mean", -1.0)

    hard_structure_gate = (
        contour_gain >= float(config.sdf_relative_gain_required)
        and contour_wins >= float(config.sdf_relative_win_fraction)
        and contour_regress <= float(config.sdf_relative_regression_fraction)
        and predicted_missing
        <= source_missing + float(config.sdf_missing_contour_tolerance)
        and contour_chamfer <= float(config.sdf_catastrophic_chamfer_pixels)
        and topology_regression == 0.0
        and line_jitter <= float(config.structural_line_jitter_required_pixels)
        and curve_roughness <= float(config.structural_curve_roughness_required_pixels)
        and staircase_recovery
        >= float(config.structural_line_staircase_recovery_required)
    )

    oracle_render_mae = _metric_float(metrics, "sdf_oracle_render_band_mae_mean", 999.0)
    oracle_global_mae = _metric_float(metrics, "sdf_oracle_global_mae_mean", 999.0)
    oracle_global_mae_max = _metric_float(metrics, "sdf_oracle_global_mae_case_max", 999.0)
    oracle_gradient_mae = _metric_float(metrics, "sdf_oracle_gradient_mae_mean", 999.0)
    oracle_width_error = _metric_float(metrics, "sdf_oracle_profile_width_relative_error_mean", 999.0)
    oracle_profile_corr = _metric_float(metrics, "sdf_oracle_profile_correlation_mean", -1.0)
    oracle_core_halo_delta = _metric_float(metrics, "sdf_oracle_core_halo_delta_8bit_max", 999.0)

    legacy_profile_render_gate = (
        oracle_render_mae <= float(config.sdf_oracle_render_band_mae_required)
        and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
        and oracle_width_error <= float(config.sdf_oracle_profile_width_error_required)
        and oracle_profile_corr >= float(config.sdf_oracle_profile_correlation_required)
        and oracle_core_halo_delta
        <= float(config.sdf_oracle_core_halo_delta_required_8bit)
    )
    direct_pixel_render_gate = (
        oracle_global_mae <= float(config.sdf_oracle_global_mae_required)
        and oracle_global_mae_max
        <= float(config.sdf_oracle_global_mae_case_max_required)
        and oracle_render_mae
        <= float(config.sdf_oracle_render_band_mae_preview_required)
        and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
    )

    profile_teacher_recovery = _metric_float(
        metrics, "sdf_profile_teacher_recovery_mean", -1.0
    )
    return bool(
        hard_structure_gate
        and (legacy_profile_render_gate or direct_pixel_render_gate)
        and profile_teacher_recovery >= float(config.sdf_teacher_recovery_required)
    )


def _state_snapshot(directory: Path, config: V9Config) -> dict[str, Any]:
    """Read only persisted qualification state used to resume the stage plan."""
    snapshot: dict[str, Any] = {
        "topologyBootstrapped": False,
        "geometryQualified": False,
        "renderQualified": False,
        "seamReconstructionQualified": False,
        "seamAuthorityQualified": False,
        "detailQualified": False,
    }
    state_path = directory / config.training_state_name
    if state_path.is_file():
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception:
            state = {}
        if isinstance(state, dict):
            snapshot.update(
                {
                    "topologyBootstrapped": bool(state.get("topology_bootstrapped", False)),
                    "geometryQualified": bool(state.get("structure_qualified", False)),
                    "renderQualified": bool(state.get("render_qualified", False)),
                    "seamReconstructionQualified": bool(
                        state.get("seam_reconstruction_qualified", False)
                    ),
                    "seamAuthorityQualified": bool(
                        state.get("seam_authority_qualified", False)
                    ),
                    "detailQualified": bool(state.get("detail_qualified", False)),
                }
            )

    metadata_path = directory / config.metadata_name
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if isinstance(previous, dict):
            # Preserve the exact exported metric bundle so gate-proof can be
            # resumed between stage invocations without replaying earlier work.
            snapshot["bestSyntheticSdfValidation"] = previous.get(
                "bestSyntheticSdfValidation"
            )
    return snapshot


StageGate = Callable[[dict[str, Any], V9Config], bool]


def _simple_gate(key: str) -> StageGate:
    return lambda metadata, _config: bool(metadata.get(key, False))


def _local_geometry_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = metadata.get("bestSyntheticSdfValidation")
    if isinstance(metrics, dict) and metrics:
        return metrics
    metrics = metadata.get("syntheticSdfValidation")
    return metrics if isinstance(metrics, dict) else {}


def _local_geometry_gate(metadata: dict[str, Any], config: V9Config) -> bool:
    """Same hard structural/redraw thresholds, without obsolete class gating."""
    metrics = _local_geometry_metrics(metadata)
    if not metrics:
        return False
    contour_gain = _metric_float(metrics, "sdf_zero_contour_relative_gain_mean", -1.0)
    contour_wins = _metric_float(metrics, "sdf_zero_contour_relative_win_fraction", 0.0)
    contour_regress = _metric_float(metrics, "sdf_zero_contour_relative_regression_fraction", 1.0)
    source_missing = _metric_float(metrics, "sdf_source_missing_contour_fraction", 1.0)
    predicted_missing = _metric_float(metrics, "sdf_predicted_missing_contour_fraction", 1.0)
    contour_chamfer = _metric_float(metrics, "sdf_zero_contour_chamfer_pixels", 999.0)
    topology_regression = _metric_float(metrics, "sdf_stageb_topology_regression_fraction", 1.0)
    line_jitter = _metric_float(metrics, "sdf_line_perpendicular_jitter_pixels_mean", 999.0)
    curve_roughness = _metric_float(metrics, "sdf_circle_radial_roughness_pixels_mean", 999.0)
    staircase_recovery = _metric_float(metrics, "sdf_line_staircase_recovery_mean", -1.0)

    structure = (
        contour_gain >= float(config.sdf_relative_gain_required)
        and contour_wins >= float(config.sdf_relative_win_fraction)
        and contour_regress <= float(config.sdf_relative_regression_fraction)
        and predicted_missing <= source_missing + float(config.sdf_missing_contour_tolerance)
        and contour_chamfer <= float(config.sdf_catastrophic_chamfer_pixels)
        and topology_regression == 0.0
        and line_jitter <= float(config.structural_line_jitter_required_pixels)
        and curve_roughness <= float(config.structural_curve_roughness_required_pixels)
        and staircase_recovery >= float(config.structural_line_staircase_recovery_required)
    )
    if not structure:
        return False

    oracle_render_mae = _metric_float(metrics, "sdf_oracle_render_band_mae_mean", 999.0)
    oracle_global_mae = _metric_float(metrics, "sdf_oracle_global_mae_mean", 999.0)
    oracle_global_mae_max = _metric_float(metrics, "sdf_oracle_global_mae_case_max", 999.0)
    oracle_gradient_mae = _metric_float(metrics, "sdf_oracle_gradient_mae_mean", 999.0)
    oracle_width_error = _metric_float(metrics, "sdf_oracle_profile_width_relative_error_mean", 999.0)
    oracle_profile_corr = _metric_float(metrics, "sdf_oracle_profile_correlation_mean", -1.0)
    oracle_core_halo_delta = _metric_float(metrics, "sdf_oracle_core_halo_delta_8bit_max", 999.0)
    profile_render = (
        oracle_render_mae <= float(config.sdf_oracle_render_band_mae_required)
        and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
        and oracle_width_error <= float(config.sdf_oracle_profile_width_error_required)
        and oracle_profile_corr >= float(config.sdf_oracle_profile_correlation_required)
        and oracle_core_halo_delta <= float(config.sdf_oracle_core_halo_delta_required_8bit)
    )
    direct_render = (
        oracle_global_mae <= float(config.sdf_oracle_global_mae_required)
        and oracle_global_mae_max <= float(config.sdf_oracle_global_mae_case_max_required)
        and oracle_render_mae <= float(config.sdf_oracle_render_band_mae_preview_required)
        and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
    )
    return bool(profile_render or direct_render)


def _promote_local_geometry_state(directory: Path, config: V9Config, metadata: dict[str, Any]) -> None:
    """Persist the direct local structure/redraw proof for the canonical resume."""
    state_path = directory / config.training_state_name
    if not state_path.is_file():
        raise RuntimeError(f"local geometry stage produced no training state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError("local geometry training state is not a mapping")
    state["topology_bootstrapped"] = True
    state["structure_qualified"] = True
    state["render_qualified"] = True

    # Keep the checkpoint's true learned epoch for provenance, then advance the
    # RESUME CURSOR across the validator-mandated residual/B1b slot.  That slot
    # belonged exclusively to the retired whole-tile classifier. It receives no
    # fake qualification and no optimiser step in the local-boundary curriculum.
    learned_epoch = int(state.get("completed_epoch", 0))
    structural_end = int(config.identity_epochs)
    if learned_epoch < structural_end:
        raise RuntimeError(
            f"local geometry promotion occurred too early: epoch={learned_epoch} "
            f"expected>={structural_end}"
        )
    retired_end = int(config.identity_epochs) + int(config.residual_epochs)
    retired_skipped = max(0, retired_end - learned_epoch)
    state["completed_epoch"] = max(learned_epoch, retired_end)
    state["retired_b1b_epochs_skipped"] = retired_skipped
    state["retired_b1b_reason"] = (
        "whole-tile primitive classifier removed from V11.4 production structure"
    )
    torch.save(state, state_path)

    # The current trainer restores best_b2_redraw.pt when downstream starts.
    # Seed that checkpoint from the exact promoted local-geometry weights.
    metrics = _local_geometry_metrics(metadata)
    payload = {
        "schema": str(state.get("schema", "")),
        "config": config.to_dict(),
        "phase": "sdf-bootstrap",
        "epoch": learned_epoch,
        "selection_kind": "v114-evolutionary-local-boundary-structure-redraw",
        "metrics": metrics,
        "qualified": True,
        "retired_b1b_epochs_skipped": retired_skipped,
        "state_dict": {key: value.detach().cpu() for key, value in state["state_dict"].items()},
    }
    torch.save(payload, directory / "best_b2_redraw.pt")
    torch.save(payload, directory / "best_b1b_geometry.pt")
    print(
        f"[pipeline] retired whole-tile B1b slot skipped: {retired_skipped} epoch(s); "
        f"resume cursor={int(state['completed_epoch'])}",
        flush=True,
    )


_PASS_DRIVEN_STAGE_PLAN: tuple[tuple[str, str, StageGate], ...] = (
    (
        "sdf-bootstrap",
        "B1 local analytic geometry + B2 same-renderer redraw",
        _local_geometry_gate,
    ),
    (
        "seam-proof",
        "B3 forced-authority seam reconstruction",
        _simple_gate("seamReconstructionQualified"),
    ),
    (
        "seam-authority",
        "B4 learned seam authority",
        _simple_gate("seamAuthorityQualified"),
    ),
    ("gate-proof", "boundary/profile candidate", _gate_candidate_passed),
    (
        "detail-reconstruction",
        "geometry-conditioned physical detail",
        _simple_gate("detailQualified"),
    ),
)


def _stage_already_qualified(
    phase: str,
    snapshot: dict[str, Any],
    config: V9Config,
) -> bool:
    for candidate_phase, _label, gate in _PASS_DRIVEN_STAGE_PLAN:
        if candidate_phase == phase:
            return bool(gate(snapshot, config))
    return False


def _mark_stage_rejected(
    repo_root: Path,
    directory: Path,
    experiment_id: str,
    args: argparse.Namespace,
    *,
    phase: str,
    gate_label: str,
    metadata: dict[str, Any],
) -> int:
    manifest = load_experiment_manifest(repo_root, experiment_id)
    manifest.update(
        {
            "status": "training-rejected",
            "failedStage": phase,
            "failedGate": gate_label,
            "epochsCompleted": int(metadata.get("epochsCompleted", 0)),
            "lastStoppedUtc": _utc_now(),
        }
    )
    write_experiment_manifest(directory / "experiment.json", manifest)
    _write_result(
        repo_root,
        args.result_file,
        {
            "experiment": experiment_id,
            "directory": str(directory),
            "trainingMode": args.training_mode,
            "status": "training-rejected",
            "failedStage": phase,
            "failedGate": gate_label,
            "epochsCompleted": int(metadata.get("epochsCompleted", 0)),
        },
    )
    print(
        f"[pipeline] REJECTED at {phase}: {gate_label} did not qualify "
        "within its bounded production budget. Downstream stages were not run.",
        flush=True,
    )
    # Non-zero is deliberate: the outer canonical workflow stops before
    # freeze_final_checkpoint instead of attempting to promote an unqualified state.
    return 2


def _archive_and_reset_structural_attempt(
    directory: Path,
    config: V9Config,
    *,
    attempt: int,
) -> Path:
    """Archive a failed B1/B2 attempt and reset only in-progress training state.

    The experiment identity and immutable resolved config are preserved. No
    downstream stage can have run before this helper is called. The next
    train_v9 invocation therefore starts from the same deterministic seed with
    a different checkpointed genome inside the same production supernet.
    """
    archive = directory / "evolution" / f"failed_structural_attempt_{attempt:02d}"
    archive.mkdir(parents=True, exist_ok=True)
    names = {
        str(getattr(config, "training_state_name", "nsamdr_v9_training_state.pt")),
        str(getattr(config, "metadata_name", "nsamdr_v9_fidelity.json")),
        str(getattr(config, "checkpoint_name", "nsamdr_v9_fidelity.pt")),
        "training_log.csv",
        "best_b1a_topology.pt",
        "best_b1b_geometry.pt",
        "best_b2_redraw.pt",
    }
    for name in sorted(names):
        source = directory / name
        if not source.is_file():
            continue
        target = archive / source.name
        shutil.copy2(source, target)
        source.unlink()
    return archive


def _train_pass_driven_pipeline(
    *,
    config: V9Config,
    repo_root: Path,
    directory: Path,
    experiment_id: str,
    args: argparse.Namespace,
    initial_resume: bool,
    evolution: EvolutionaryRecoveryController,
) -> tuple[dict[str, Any] | None, int]:
    """Run one click as a sequence of gated canonical trainer stages.

    Each stage receives its existing configured maximum work budget. A stage may
    advance only after the canonical exported qualification says PASS. Failure
    returns immediately and the outer workflow never freezes or previews an
    unqualified checkpoint.
    """
    resume_now = bool(initial_resume)
    snapshot = _state_snapshot(directory, config)
    latest: dict[str, Any] | None = None

    for phase, gate_label, gate in _PASS_DRIVEN_STAGE_PLAN:
        if _stage_already_qualified(phase, snapshot, config):
            print(
                f"[pipeline] {phase}: already qualified in persisted state; skipping replay.",
                flush=True,
            )
            continue

        print("=" * 72, flush=True)
        print(f"PASS-DRIVEN STAGE       : {phase}", flush=True)
        print(f"Promotion gate          : {gate_label}", flush=True)
        print("Failure policy          : stop here; do not run downstream", flush=True)
        print("=" * 72, flush=True)

        while True:
            latest = train_v9(
                config,
                repo_root,
                args.device,
                resume=resume_now,
                restart=False,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                stop_after_phase=phase,
            )
            resume_now = True

            if gate(latest, config):
                break

            # Only a structural/representation failure is eligible for genome
            # evolution. Software, contract and numerical failures never reach
            # this branch because train_v9 raises them; learning-only failures
            # retain the ordinary fail-closed behaviour.
            metrics = _local_geometry_metrics(latest) if phase == "sdf-bootstrap" else {}
            failure_kind = classify_failure(metrics=metrics) if phase == "sdf-bootstrap" else FailureKind.LEARNING
            if (
                phase == "sdf-bootstrap"
                and failure_kind == FailureKind.REPRESENTATION
                and evolution.can_recover(metrics)
            ):
                print(
                    "[evolution] structural gate failed; breeding a bounded "
                    "production-supernet generation instead of continuing downstream.",
                    flush=True,
                )
                recovery = evolution.recover_after_structural_failure(metrics)
                if recovery.passed:
                    archive = _archive_and_reset_structural_attempt(
                        directory, config, attempt=evolution.recovery_count
                    )
                    print(
                        f"[evolution] recovery generation passed microproof; archived "
                        f"failed structural state at {archive} and restarting B1/B2 "
                        "from the deterministic seed with the evolved genome.",
                        flush=True,
                    )
                    resume_now = False
                    snapshot = _state_snapshot(directory, config)
                    continue
                print(
                    "[evolution] recovery generation produced no viable structural "
                    "candidate; fail closed.",
                    flush=True,
                )

            return latest, _mark_stage_rejected(
                repo_root,
                directory,
                experiment_id,
                args,
                phase=phase,
                gate_label=gate_label,
                metadata=latest,
            )

        if phase == "sdf-bootstrap":
            _promote_local_geometry_state(directory, config, latest)
            latest["topologyBootstrapped"] = True
            latest["geometryQualified"] = True
            latest["renderQualified"] = True
            locked = evolution.lock_production_genome(
                experiment_id=experiment_id,
                metrics=_local_geometry_metrics(latest),
            )
            print(f"[evolution] production genome locked: {locked}", flush=True)
        print(f"[pipeline] PASS {phase}: {gate_label}", flush=True)
        snapshot.update(latest)

    print("=" * 72, flush=True)
    print("PASS-DRIVEN FINAL STAGE : physical-finetune / BenefitSelector", flush=True)
    print("Promotion gate          : production-final + full final qualification", flush=True)
    print("=" * 72, flush=True)

    latest = train_v9(
        config,
        repo_root,
        args.device,
        resume=True,
        restart=False,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        stop_after_phase=None,
    )
    final_pass = bool(latest.get("trainingSafetyPass", False)) and str(
        latest.get("selectionKind") or ""
    ) == "production-final"
    if not final_pass:
        return latest, _mark_stage_rejected(
            repo_root,
            directory,
            experiment_id,
            args,
            phase="physical-finetune",
            gate_label="production final selector + strict training safety",
            metadata=latest,
        )

    print("[pipeline] PASS physical-finetune: production-final selected.", flush=True)
    return latest, 0

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    base_config_path = args.base_config if args.base_config.is_absolute() else repo_root / args.base_config
    base_config_path = base_config_path.resolve()
    dataset_config_path = args.dataset_config or args.base_config
    if not dataset_config_path.is_absolute():
        dataset_config_path = repo_root / dataset_config_path
    dataset_config_path = dataset_config_path.resolve()
    base = V9Config.load(base_config_path)
    data_scope = V9Config.load(dataset_config_path)

    requested = str(args.experiment).strip().upper()
    if requested in {"", "NEW"}:
        overrides = _canonical_overrides(args, base, data_scope)
        resolved_probe = copy.deepcopy(base)
        _set_values(resolved_probe, overrides)
        resolved_probe.validate()
        dataset_manifest = Path(resolved_probe.dataset_manifest)
        if not dataset_manifest.is_absolute():
            dataset_manifest = repo_root / dataset_manifest
        if not dataset_manifest.is_file() and not args.allocate_only:
            raise RuntimeError(f"prepared dataset manifest is missing: {dataset_manifest}")
        dataset = (
            json.loads(dataset_manifest.read_text(encoding="utf-8"))
            if dataset_manifest.is_file()
            else {}
        )
        asset = dataset.get("asset", {}) if isinstance(dataset, dict) else {}
        asset_name = str(asset.get("displayName") or DEFAULT_TUNING_ASSET_NAME)
        asset_query = str(asset.get("query") or DEFAULT_TUNING_ASSET_QUERY)
        selection_key = str(asset.get("selectionKey") or "")
        experiment_id, directory, config = initialise_experiment(
            repo_root,
            base_config_path,
            overrides,
            preset=args.preset,
            asset_name=asset_name,
            asset_query=asset_query,
            selection_key=selection_key,
            training_mode=args.training_mode,
        )
        resume = False
        print(f"[experiment] Allocated {experiment_id}: {directory}", flush=True)
    else:
        experiment_id = requested
        directory = experiment_dir(repo_root, experiment_id)
        config = load_resolved_config(repo_root, experiment_id)
        manifest = load_experiment_manifest(repo_root, experiment_id)
        stored_mode = str(manifest.get("trainingMode") or "").lower()
        if stored_mode != args.training_mode:
            raise RuntimeError(
                f"{experiment_id} is {stored_mode!r}, not requested {args.training_mode!r}"
            )
        status = str(manifest.get("status") or "").lower()
        if status in {
            "trained-pending-qualification", "completed",
        }:
            raise RuntimeError(
                f"{experiment_id} already produced a final training result and is immutable; "
                "resume the orchestrator qualification or allocate a new experiment"
            )
        dataset_manifest = Path(config.dataset_manifest)
        if not dataset_manifest.is_absolute():
            dataset_manifest = repo_root / dataset_manifest
        if not dataset_manifest.is_file():
            raise RuntimeError(f"experiment dataset manifest is missing: {dataset_manifest}")
        asset = manifest.get("asset", {}) if isinstance(manifest, dict) else {}
        asset_name = str(asset.get("displayName") or DEFAULT_TUNING_ASSET_NAME)
        asset_query = str(asset.get("query") or DEFAULT_TUNING_ASSET_QUERY)
        selection_key = str(asset.get("selectionKey") or "")
        state_path = directory / config.training_state_name
        if args.control == "resume" and not state_path.is_file():
            raise RuntimeError(f"resume requested but experiment state is missing: {state_path}")
        resume = state_path.is_file()
        print(f"[experiment] Reusing immutable resolved config: {directory / 'resolved_config.json'}", flush=True)

    manifest = load_experiment_manifest(repo_root, experiment_id)
    if args.allocate_only:
        manifest.update({"status": "allocated", "lastStoppedUtc": _utc_now()})
        write_experiment_manifest(directory / "experiment.json", manifest)
        _write_result(
            repo_root,
            args.result_file,
            {
                "experiment": experiment_id,
                "directory": str(directory),
                "trainingMode": args.training_mode,
                "allocatedOnly": True,
            },
        )
        print(f"[experiment] Allocation complete: {experiment_id}", flush=True)
        return 0

    manifest.update(
        {
            "status": "running",
            "trainingMode": args.training_mode,
            "lastStartedUtc": _utc_now(),
        }
    )
    write_experiment_manifest(directory / "experiment.json", manifest)

    print("=" * 72, flush=True)
    print(f"NSAMDR COMPLETE PRODUCTION MODEL - {args.training_mode.upper()} WORK BUDGET", flush=True)
    print(f"Experiment               : {experiment_id}", flush=True)
    print(f"Dataset manifest         : {repo_root / config.dataset_manifest}", flush=True)
    print(f"Epochs                   : {config.total_epochs}", flush=True)
    print(f"Tiles / validation       : {config.tiles_per_epoch} / {config.validation_tiles}", flush=True)
    print(f"Resolved config          : {directory / 'resolved_config.json'}", flush=True)
    print("Semantic model config    : production (identical for Quick and Full)", flush=True)
    print("=" * 72, flush=True)

    evolution = EvolutionaryRecoveryController(
        repo_root=repo_root,
        experiment_dir=directory,
        config=config,
        device=args.device,
        population=4 if args.training_mode == "quick" else 6,
        micro_steps=3 if args.training_mode == "quick" else 5,
        max_recoveries=2,
    )
    discovery = evolution.discover_before_training()
    if not discovery.passed:
        rejected = load_experiment_manifest(repo_root, experiment_id)
        rejected.update({
            "status": "training-rejected",
            "failedStage": "evolution-capacity-microproof",
            "failedGate": "real Raven local-boundary capacity",
            "lastStoppedUtc": _utc_now(),
        })
        write_experiment_manifest(directory / "experiment.json", rejected)
        _write_result(
            repo_root, args.result_file,
            {
                "experiment": experiment_id,
                "directory": str(directory),
                "trainingMode": args.training_mode,
                "status": "training-rejected",
                "failedStage": "evolution-capacity-microproof",
                "evolutionReport": str(directory / "evolution"),
            },
        )
        print(
            "[evolution] no candidate passed after two bounded generations; "
            "expensive training was not started.",
            flush=True,
        )
        return 2
    print(
        f"[evolution] capacity microproof PASS; candidate genome="
        f"{discovery.winner.fingerprint()[:12]} (not production-locked until B1/B2 passes)",
        flush=True,
    )

    try:
        if args.stop_after_phase is not None:
            # Hidden/manual diagnostic mode remains available as one canonical
            # staged trainer invocation.
            train_metadata = train_v9(
                config,
                repo_root,
                args.device,
                resume=resume,
                restart=False,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                stop_after_phase=args.stop_after_phase,
            )
            pipeline_code = 0
        else:
            train_metadata, pipeline_code = _train_pass_driven_pipeline(
                config=config,
                repo_root=repo_root,
                directory=directory,
                experiment_id=experiment_id,
                args=args,
                initial_resume=resume,
                evolution=evolution,
            )
            if pipeline_code != 0:
                return pipeline_code
            if train_metadata is None:
                raise RuntimeError("pass-driven trainer returned no final metadata")
    except BaseException:
        failed = load_experiment_manifest(repo_root, experiment_id)
        failed.update({"status": "interrupted-or-failed", "lastStoppedUtc": _utc_now()})
        write_experiment_manifest(directory / "experiment.json", failed)
        raise

    if args.stop_after_phase is not None and bool(train_metadata.get("stagedStopReached")):
        paused = load_experiment_manifest(repo_root, experiment_id)
        paused.update(
            {
                "status": "stage-paused",
                "stagedStopPhase": args.stop_after_phase,
                "epochsCompleted": int(train_metadata.get("epochsCompleted", 0)),
                "lastStoppedUtc": _utc_now(),
            }
        )
        write_experiment_manifest(directory / "experiment.json", paused)
        _write_result(
            repo_root,
            args.result_file,
            {
                "experiment": experiment_id,
                "directory": str(directory),
                "trainingMode": args.training_mode,
                "partialTraining": True,
                "stagedStopPhase": args.stop_after_phase,
            },
        )
        return 0

    completed = finalise_experiment(
        repo_root,
        experiment_id,
        asset_name=asset_name,
        asset_query=asset_query,
        selection_key=selection_key,
        training_mode=args.training_mode,
    )
    write_experiment_manifest(directory / "experiment.json", completed)
    _write_result(
        repo_root,
        args.result_file,
        {
            "experiment": experiment_id,
            "directory": str(directory),
            "trainingMode": args.training_mode,
            "status": "trained-pending-qualification",
        },
    )
    print(f"[experiment] Training complete; qualification pending: {experiment_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
