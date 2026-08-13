#!/usr/bin/env python3
"""Train a complete V9 tuning experiment on the fixed Raven preview set."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

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
from v9.training import train_v9


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if "," in value:
        parts = [item.strip() for item in value.split(",")]
        try:
            return [int(item) for item in parts]
        except ValueError:
            try:
                return [float(item) for item in parts]
            except ValueError:
                return parts
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _advanced_overrides(raw: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"advanced override must be key=value: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"advanced override has empty key: {item!r}")
        result[key] = _parse_scalar(value)
    return result


def _build_overrides(args: argparse.Namespace, base: V9Config) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if args.learning_rate is not None:
        requested = float(args.learning_rate)
        ratio = requested / max(base.learning_rate, 1.0e-12)
        values.update(
            {
                "identity_learning_rate": base.identity_learning_rate * ratio,
                "learning_rate": requested,
                "boundary_learning_rate": base.boundary_learning_rate * ratio,
                "detail_learning_rate": base.detail_learning_rate * ratio,
                "finetune_learning_rate": base.finetune_learning_rate * ratio,
            }
        )
    for arg_name, config_name in (
        ("weight_decay", "weight_decay"),
        ("optimizer_name", "optimizer_name"),
        ("scheduler_name", "scheduler_name"),
        ("scheduler_min_lr_ratio", "scheduler_min_lr_ratio"),
        ("batch_size", "batch_size"),
        ("tiles_per_epoch", "tiles_per_epoch"),
        ("validation_tiles", "validation_tiles"),
        ("regret_weight", "regret_weight"),
        ("normal_regret_weight", "normal_regret_weight"),
        ("edge_weight", "edge_weight"),
        ("detail_laplacian_weight", "detail_laplacian_weight"),
        ("geometric_alignment_weight", "geometric_alignment_weight"),
        ("tangent_coherence_weight", "tangent_coherence_weight"),
        ("curvature_coherence_weight", "curvature_coherence_weight"),
        ("synthetic_geometry_probability", "synthetic_geometry_probability"),
        ("boundary_sampling_probability", "boundary_sampling_probability"),
        ("boundary_renderer_band_pixels", "boundary_renderer_band_pixels"),
        ("boundary_renderer_sample_pixels", "boundary_renderer_sample_pixels"),
        ("boundary_renderer_hard_width_pixels", "boundary_renderer_hard_width_pixels"),
        ("boundary_renderer_soft_width_pixels", "boundary_renderer_soft_width_pixels"),
        ("boundary_renderer_gate_gain", "boundary_renderer_gate_gain"),
        ("boundary_renderer_far_sample_multiplier", "boundary_renderer_far_sample_multiplier"),
        ("boundary_renderer_far_sample_weight", "boundary_renderer_far_sample_weight"),
        ("boundary_gate_need_scale", "boundary_gate_need_scale"),
        ("boundary_gate_exact_floor", "boundary_gate_exact_floor"),
        ("boundary_sdf_zero_weight", "boundary_sdf_zero_weight"),
        ("boundary_edge_sdf_consistency_weight", "boundary_edge_sdf_consistency_weight"),
        ("boundary_pixel_regret_weight", "boundary_pixel_regret_weight"),
        ("boundary_profile_weight", "boundary_profile_weight"),
        ("boundary_regret_weight", "boundary_regret_weight"),
        ("sdf_surface_weight", "sdf_surface_weight"),
        ("sdf_sign_weight", "sdf_sign_weight"),
        ("sdf_eikonal_weight", "sdf_eikonal_weight"),
        ("sdf_gradient_alignment_weight", "sdf_gradient_alignment_weight"),
        ("sdf_metric_gradient_weight", "sdf_metric_gradient_weight"),
        ("sdf_metric_band_pixels", "sdf_metric_band_pixels"),
        ("sdf_coarse_init_std", "sdf_coarse_init_std"),
        ("sdf_synthetic_validation_tiles", "sdf_synthetic_validation_tiles"),
        ("sdf_zero_band_pixels", "sdf_zero_band_pixels"),
        ("sdf_bootstrap_residual_pixels", "sdf_bootstrap_residual_pixels"),
        ("sdf_proof_residual_pixels", "sdf_proof_residual_pixels"),
        ("sdf_proof_renderer_weight", "sdf_proof_renderer_weight"),
        ("implicit_sdf_hidden_channels", "implicit_sdf_hidden_channels"),
        ("implicit_sdf_residual_pixels", "implicit_sdf_residual_pixels"),
        ("coarse_sdf_surface_weight", "coarse_sdf_surface_weight"),
        ("sdf_residual_l1_weight", "sdf_residual_l1_weight"),
        ("boundary_fuzz_weight", "boundary_fuzz_weight"),
        ("boundary_halo_weight", "boundary_halo_weight"),
        ("boundary_renderer_plateau_samples", "boundary_renderer_plateau_samples"),
        ("boundary_renderer_plateau_max_multiplier", "boundary_renderer_plateau_max_multiplier"),
        ("boundary_renderer_plateau_stability_scale", "boundary_renderer_plateau_stability_scale"),
        ("seed", "seed"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            values[config_name] = value

    if args.randomise_seed:
        values["seed"] = random.SystemRandom().randrange(1, 2**31 - 1)

    if args.augmentation_strength is not None:
        strength = max(0.0, min(float(args.augmentation_strength), 2.0))
        # 1.0 is exactly the production degradation distribution. The scalar
        # expands/contracts probabilities without creating a second hidden
        # augmentation implementation.
        for key in (
            "anisotropic_blur_probability",
            "bc_block_probability",
            "chroma_loss_probability",
            "ringing_probability",
            "halo_probability",
        ):
            values[key] = max(0.0, min(1.0, float(getattr(base, key)) * strength))
        values["lod_bias_max"] = max(base.lod_bias_min, base.lod_bias_min + (base.lod_bias_max - base.lod_bias_min) * strength)

    values.update(_advanced_overrides(args.advanced_overrides or ""))
    return values


def _apply_runtime(config: V9Config, args: argparse.Namespace) -> None:
    if args.performance_profile:
        config.apply_performance_profile(args.performance_profile)
    if args.workers is not None:
        config.data_loader_workers = int(args.workers)
    if args.prefetch_factor is not None:
        config.data_loader_prefetch_factor = int(args.prefetch_factor)
    if args.amp_precision:
        config.amp_dtype = args.amp_precision
    config.validate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Quick or Full V9 tuning experiment on fixed Raven set")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-config", type=Path, default=Path("tools/nsamdr/neural/configs/v9_preview_raven.json"))
    parser.add_argument("--experiment", default="new", help="new or EXP_####")
    parser.add_argument("--control", choices=("auto", "resume", "restart"), default="auto")
    parser.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--preset", default="Baseline")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--optimizer-name", choices=("adamw", "adam"))
    parser.add_argument("--scheduler-name", choices=("phase", "cosine-phase"))
    parser.add_argument("--scheduler-min-lr-ratio", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tiles-per-epoch", type=int)
    parser.add_argument("--validation-tiles", type=int)
    parser.add_argument("--regret-weight", type=float)
    parser.add_argument("--normal-regret-weight", type=float)
    parser.add_argument("--edge-weight", type=float)
    parser.add_argument("--detail-laplacian-weight", type=float)
    parser.add_argument("--geometric-alignment-weight", type=float)
    parser.add_argument("--tangent-coherence-weight", type=float)
    parser.add_argument("--curvature-coherence-weight", type=float)
    parser.add_argument("--synthetic-geometry-probability", type=float)
    parser.add_argument("--boundary-sampling-probability", type=float)
    parser.add_argument("--boundary-renderer-band-pixels", type=float)
    parser.add_argument("--boundary-renderer-sample-pixels", type=float)
    parser.add_argument("--boundary-renderer-hard-width-pixels", type=float)
    parser.add_argument("--boundary-renderer-soft-width-pixels", type=float)
    parser.add_argument("--boundary-renderer-gate-gain", type=float)
    parser.add_argument("--boundary-renderer-far-sample-multiplier", type=float)
    parser.add_argument("--boundary-renderer-far-sample-weight", type=float)
    parser.add_argument("--boundary-gate-need-scale", type=float)
    parser.add_argument("--boundary-gate-exact-floor", type=float)
    parser.add_argument("--boundary-sdf-zero-weight", type=float)
    parser.add_argument("--boundary-edge-sdf-consistency-weight", type=float)
    parser.add_argument("--boundary-pixel-regret-weight", type=float)
    parser.add_argument("--boundary-profile-weight", type=float)
    parser.add_argument("--boundary-regret-weight", type=float)
    parser.add_argument("--sdf-surface-weight", type=float)
    parser.add_argument("--sdf-sign-weight", type=float)
    parser.add_argument("--sdf-eikonal-weight", type=float)
    parser.add_argument("--sdf-gradient-alignment-weight", type=float)
    parser.add_argument("--sdf-metric-gradient-weight", type=float)
    parser.add_argument("--sdf-metric-band-pixels", type=float)
    parser.add_argument("--sdf-coarse-init-std", type=float)
    parser.add_argument("--sdf-synthetic-validation-tiles", type=int)
    parser.add_argument("--sdf-zero-band-pixels", type=float)
    parser.add_argument("--sdf-bootstrap-residual-pixels", type=float)
    parser.add_argument("--sdf-proof-residual-pixels", type=float)
    parser.add_argument("--sdf-proof-renderer-weight", type=float)
    parser.add_argument("--implicit-sdf-hidden-channels", type=int)
    parser.add_argument("--implicit-sdf-residual-pixels", type=float)
    parser.add_argument("--coarse-sdf-surface-weight", type=float)
    parser.add_argument("--sdf-residual-l1-weight", type=float)
    parser.add_argument("--boundary-fuzz-weight", type=float)
    parser.add_argument("--boundary-halo-weight", type=float)
    parser.add_argument("--boundary-renderer-plateau-samples", type=int)
    parser.add_argument("--boundary-renderer-plateau-max-multiplier", type=float)
    parser.add_argument("--boundary-renderer-plateau-stability-scale", type=float)
    parser.add_argument("--augmentation-strength", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--randomise-seed", action="store_true")
    parser.add_argument("--advanced-overrides", default="")
    parser.add_argument("--performance-profile", choices=("optimized", "fast", "balanced", "compatibility"), default="optimized")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--allocate-only", action="store_true")
    parser.add_argument("--stop-after-phase", choices=("sdf-bootstrap","sdf-proof","gate-proof","boundary-hardening","physical-finetune"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    base_config_path = args.base_config if args.base_config.is_absolute() else repo_root / args.base_config
    base_config_path = base_config_path.resolve()
    base = V9Config.load(base_config_path)
    dataset_manifest = repo_root / base.dataset_manifest
    if not dataset_manifest.is_file():
        raise RuntimeError(
            f"fixed Raven preview dataset is missing: {dataset_manifest}. "
            "Run scripts\\build\\nsamdr.bat index raven first."
        )
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    asset = dataset.get("asset", {})
    asset_name = str(asset.get("displayName") or DEFAULT_TUNING_ASSET_NAME)
    asset_query = str(asset.get("query") or DEFAULT_TUNING_ASSET_QUERY)
    selection_key = str(asset.get("selectionKey") or "")

    requested = str(args.experiment).strip().upper()
    if requested in {"", "NEW"}:
        if args.control == "resume":
            raise RuntimeError("cannot resume a new experiment; choose auto or an existing EXP_####")
        overrides = _build_overrides(args, base)
        if args.training_mode == "quick":
            overrides.update({
                "identity_epochs": 1,
                "residual_epochs": 5,
                "boundary_epochs": 2,
                "detail_epochs": 1,
                "physical_finetune_epochs": 2,
                "tiles_per_epoch": int(args.tiles_per_epoch or 96),
                "validation_tiles": int(args.validation_tiles or 16),
            })
        else:
            # Full proof preserves the complete 24-epoch phase schedule but uses
            # a smaller Raven work budget than all-assets production.
            overrides.update({
                "identity_epochs": 1,
                "residual_epochs": 5,
                "boundary_epochs": 8,
                "detail_epochs": 5,
                "physical_finetune_epochs": 5,
                "tiles_per_epoch": int(args.tiles_per_epoch or 128),
                "validation_tiles": int(args.validation_tiles or 32),
            })
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
        restart = False
        print(f"[experiment] Allocated {experiment_id}: {directory}", flush=True)
        print("[experiment] New experiment uses immutable resolved_config.json", flush=True)
    else:
        experiment_id = requested
        directory = experiment_dir(repo_root, experiment_id)
        config = load_resolved_config(repo_root, experiment_id)
        manifest = load_experiment_manifest(repo_root, experiment_id)
        manifest_mode = str(manifest.get("trainingMode") or "full").lower()
        if manifest_mode != args.training_mode:
            raise RuntimeError(
                f"{experiment_id} was created as trainingMode={manifest_mode!r}; "
                f"it cannot be resumed as {args.training_mode!r}. Allocate a new experiment."
            )
        if manifest.get("status") == "completed":
            raise RuntimeError(
                f"{experiment_id} is a completed immutable experiment and cannot be retrained. "
                "Select experiment=new to allocate a new EXP_####."
            )
        state_path = directory / config.training_state_name
        if args.control == "resume" and not state_path.is_file():
            raise RuntimeError(f"resume requested but experiment state is missing: {state_path}")
        resume = args.control == "resume" or (args.control == "auto" and state_path.is_file())
        restart = args.control == "restart"
        print(f"[experiment] Existing immutable config: {directory / 'resolved_config.json'}", flush=True)
        print(f"[experiment] Control: {'resume' if resume else 'restart' if restart else 'fresh'}", flush=True)

    manifest = load_experiment_manifest(repo_root, experiment_id)
    manifest.update({
        "status": "running",
        "trainingMode": args.training_mode,
        "promotionEligible": args.training_mode == "full" and bool(config.appearance_enabled),
        "lastStartedUtc": _utc_now(),
    })
    write_experiment_manifest(directory / "experiment.json", manifest)

    if args.allocate_only:
        manifest = load_experiment_manifest(repo_root, experiment_id)
        manifest.update({
            "status": "awaiting-renderer-preflight",
            "lastStoppedUtc": _utc_now(),
        })
        write_experiment_manifest(directory / "experiment.json", manifest)
        if args.result_file is not None:
            result_path = args.result_file if args.result_file.is_absolute() else repo_root / args.result_file
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({
                    "experiment": experiment_id,
                    "directory": str(directory),
                    "trainingMode": args.training_mode,
                    "allocatedOnly": True,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(f"[experiment] Allocation-only complete: {experiment_id}", flush=True)
        print(f"[experiment] Awaiting Stage-A oracle renderer preflight.", flush=True)
        return 0

    _apply_runtime(config, args)
    print("=" * 68, flush=True)
    print(f"NSAMDR V9 {args.training_mode.upper()} TUNING EXPERIMENT", flush=True)
    print(f"Experiment               : {experiment_id}", flush=True)
    print(f"Model scope              : TUNING - {asset_name}", flush=True)
    print(f"V9.8.3 sign-gauge metric-SDF geometry proof : {not bool(config.appearance_enabled)}", flush=True)
    print(f"Exact analytic geometry  : {config.synthetic_geometry_probability * 100.0:.0f}% of training tiles", flush=True)
    schedule_note = "fast development schedule" if args.training_mode == "quick" else "complete promotion-proof phase schedule"
    print(f"Training mode            : {args.training_mode}", flush=True)
    print(f"Epoch schedule           : {config.total_epochs} ({schedule_note})", flush=True)
    print(f"Tiles per epoch          : {config.tiles_per_epoch} fixed-set samples", flush=True)
    print(f"Held-out validation      : {config.validation_tiles} samples from fixed Raven regions", flush=True)
    print(f"Resolved config          : {directory / 'resolved_config.json'}", flush=True)
    print("=" * 68, flush=True)

    try:
        train_metadata = train_v9(
            config,
            repo_root,
            args.device,
            resume=resume,
            restart=restart,
            early_stop_patience=(args.early_stop_patience if args.training_mode == "full" else None),
            early_stop_min_delta=args.early_stop_min_delta,
            stop_after_phase=args.stop_after_phase,
        )
    except BaseException:
        manifest = load_experiment_manifest(repo_root, experiment_id)
        manifest.update({"status": "interrupted-or-failed", "lastStoppedUtc": _utc_now()})
        write_experiment_manifest(directory / "experiment.json", manifest)
        raise

    if bool(train_metadata.get("stagedStopReached")):
        manifest = load_experiment_manifest(repo_root, experiment_id)
        manifest.update({
            "status": "stage-paused",
            "stagedStopPhase": args.stop_after_phase,
            "epochsCompleted": int(train_metadata.get("epochsCompleted", 0)),
            "lastStoppedUtc": _utc_now(),
        })
        write_experiment_manifest(directory / "experiment.json", manifest)
        if args.result_file is not None:
            result_path = args.result_file if args.result_file.is_absolute() else repo_root / args.result_file
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({
                    "experiment": experiment_id,
                    "directory": str(directory),
                    "trainingMode": args.training_mode,
                    "partialTraining": True,
                    "stagedStopPhase": args.stop_after_phase,
                    "epochsCompleted": int(train_metadata.get("epochsCompleted", 0)),
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print("=" * 68, flush=True)
        print("NSAMDR V9.8.3 STAGED TRAINING PAUSED", flush=True)
        print(f"Experiment               : {experiment_id}", flush=True)
        print(f"Stopped after            : {args.stop_after_phase}", flush=True)
        print(f"Checkpoint               : {directory / config.checkpoint_name}", flush=True)
        print("Next                     : run Stage-B SDF proof before gate training", flush=True)
        print("=" * 68, flush=True)
        return 0

    completed = finalise_experiment(
        repo_root,
        experiment_id,
        asset_name=asset_name,
        asset_query=asset_query,
        selection_key=selection_key,
        training_mode=args.training_mode,
    )
    completed["trainingMode"] = args.training_mode
    completed["promotionEligible"] = args.training_mode == "full" and bool(config.appearance_enabled)
    write_experiment_manifest(directory / "experiment.json", completed)
    if args.result_file is not None:
        result_path = args.result_file if args.result_file.is_absolute() else repo_root / args.result_file
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({
                "experiment": experiment_id,
                "directory": str(directory),
                "trainingMode": args.training_mode,
                "promotionEligible": args.training_mode == "full" and bool(config.appearance_enabled),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("=" * 68, flush=True)
    print("NSAMDR V9 TUNING EXPERIMENT COMPLETE", flush=True)
    print(f"Experiment               : {experiment_id}", flush=True)
    print(f"Preview checkpoint       : {directory / 'checkpoint_best.pt'}", flush=True)
    print(f"Metrics                  : {directory / 'metrics.json'}", flush=True)
    print(f"Training log             : {directory / 'training_log.csv'}", flush=True)
    print(f"Best epoch               : {completed.get('bestEpoch')}", flush=True)
    if not bool(config.appearance_enabled):
        print("V9.8.3 sign-gauge metric-SDF geometry proof: preview/compare enabled; promotion remains LOCKED.", flush=True)
    elif args.training_mode == "quick":
        print("Quick experiment: preview/compare enabled; promotion remains LOCKED.", flush=True)
    else:
        print("Full proof complete: preview success can unlock promotion.", flush=True)
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
