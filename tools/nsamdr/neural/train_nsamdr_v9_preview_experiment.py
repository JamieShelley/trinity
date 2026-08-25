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
from pathlib import Path
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

QUICK_WORK_BUDGET: dict[str, int] = {
    "identity_epochs": 1,
    "residual_epochs": 1,
    "seam_proof_epochs": 1,
    "seam_authority_epochs": 1,
    "boundary_epochs": 1,
    "detail_epochs": 2,
    "physical_finetune_epochs": 1,
    "tiles_per_epoch": 48,
    "validation_tiles": 8,
    "raven_downstream_tiles_per_epoch": 32,
    "parametric_primitive_train_tiles_per_epoch": 14,
}

FULL_MINIMUM_WORK_BUDGET: dict[str, int] = {
    "identity_epochs": 1,
    "residual_epochs": 6,
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
        _set_values(
            resolved,
            {
                field: max(int(getattr(resolved, field)), minimum)
                for field, minimum in FULL_MINIMUM_WORK_BUDGET.items()
            },
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

    try:
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
    except BaseException:
        failed = load_experiment_manifest(repo_root, experiment_id)
        failed.update({"status": "interrupted-or-failed", "lastStoppedUtc": _utc_now()})
        write_experiment_manifest(directory / "experiment.json", failed)
        raise

    if bool(train_metadata.get("stagedStopReached")):
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
