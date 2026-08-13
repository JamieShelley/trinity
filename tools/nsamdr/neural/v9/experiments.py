"""Experiment registry and immutable configuration promotion for NSAMDR V9."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import torch

from .config import V9Config

LEGACY_EXPERIMENT_SCHEMA = "NSAMDR_V9_TUNING_EXPERIMENT_V1"
EXPERIMENT_SCHEMA = "NSAMDR_V9_TUNING_EXPERIMENT_V2"
CAPABILITY_SCHEMA = "NSAMDR_V9_MODEL_CAPABILITY_V1"
PROMOTION_SCHEMA = "NSAMDR_V9_PROMOTED_CONFIG_V1"
EXPERIMENT_RE = re.compile(r"^EXP_(\d{4,})$")
DEFAULT_TUNING_ASSET_NAME = "Raven Navy Issue"
DEFAULT_TUNING_ASSET_QUERY = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"

# These fields define the data/work scope rather than the learned algorithm.
# Promotion restores them from the full-dataset base config while preserving
# architecture, losses, learning rates, augmentation/degradation, batch size,
# seed and all other tunable semantic settings from the chosen experiment.
PRODUCTION_SCOPE_FIELDS = {
    "dataset_manifest",
    "dataset_root",
    "max_families",
    "crops_per_family",
    "source_crop_size",
    "min_source_dimension",
    "min_auxiliary_dimension",
    "validation_fraction",
    "require_complete_pbr_family",
    "tiles_per_epoch",
    "validation_tiles",
    "output_dir",
    "checkpoint_name",
    "metadata_name",
    "training_state_name",
    "diagnostics_dir_name",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def experiments_root(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "nsamdr" / "experiments"


def promoted_root(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "nsamdr" / "promoted"


def experiment_dir(repo_root: Path, experiment_id: str) -> Path:
    value = str(experiment_id).strip().upper()
    if EXPERIMENT_RE.fullmatch(value) is None:
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    return experiments_root(repo_root) / value


def list_experiments(repo_root: Path, *, completed_only: bool = False) -> list[str]:
    root = experiments_root(repo_root)
    if not root.is_dir():
        return []
    result: list[tuple[int, str]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = EXPERIMENT_RE.fullmatch(path.name.upper())
        if match is None:
            continue
        if completed_only and not (path / "checkpoint_best.pt").is_file():
            continue
        result.append((int(match.group(1)), path.name.upper()))
    return [name for _number, name in sorted(result)]


def allocate_experiment(repo_root: Path) -> tuple[str, Path]:
    root = experiments_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    used = {
        int(match.group(1))
        for path in root.iterdir()
        if path.is_dir() and (match := EXPERIMENT_RE.fullmatch(path.name.upper()))
    }
    number = max(used, default=0) + 1
    while True:
        experiment_id = f"EXP_{number:04d}"
        path = root / experiment_id
        try:
            path.mkdir(parents=False, exist_ok=False)
            return experiment_id, path
        except FileExistsError:
            number += 1


def load_experiment_manifest(repo_root: Path, experiment_id: str) -> dict[str, Any]:
    path = experiment_dir(repo_root, experiment_id) / "experiment.json"
    if not path.is_file():
        raise RuntimeError(f"experiment manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {LEGACY_EXPERIMENT_SCHEMA, EXPERIMENT_SCHEMA}:
        raise RuntimeError(f"unsupported experiment manifest schema: {schema!r}")
    if schema == LEGACY_EXPERIMENT_SCHEMA:
        # Pre-4.1 tuning experiments were always the complete 24-epoch Raven
        # proof, so preserve them as Full experiments rather than invalidating
        # checkpoints created before the Quick mode existed.
        payload["schema"] = EXPERIMENT_SCHEMA
        payload.setdefault("trainingMode", "full")
        payload.setdefault("promotionEligible", True)
    return payload


def write_experiment_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _promotion_eligible(training_mode: str, config: V9Config) -> bool:
    # V9.4 is deliberately geometry-only until A/B proof succeeds.  A Full run
    # is still non-promotable while AppearanceNet is disabled.
    return str(training_mode).lower() == "full" and bool(config.appearance_enabled)


def config_sha256(config: V9Config) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initialise_experiment(
    repo_root: Path,
    base_config_path: Path,
    requested_overrides: dict[str, Any],
    *,
    preset: str,
    asset_name: str,
    asset_query: str,
    selection_key: str,
    training_mode: str = "full",
) -> tuple[str, Path, V9Config]:
    experiment_id, directory = allocate_experiment(repo_root)
    config = V9Config.load(base_config_path)
    for key, value in requested_overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"unknown V9 tuning option: {key}")
        current = getattr(config, key)
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, tuple):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} requires a list/tuple")
            value = tuple(int(item) for item in value)
        setattr(config, key, value)

    # Each experiment is isolated. The same canonical checkpoint filename is
    # retained so the existing inference pipeline can point at the directory.
    config.output_dir = str(directory.relative_to(repo_root)).replace("\\", "/")
    config.checkpoint_name = "nsamdr_v9_fidelity.pt"
    config.metadata_name = "nsamdr_v9_fidelity.json"
    config.training_state_name = "nsamdr_v9_training_state.pt"
    config.validate()

    requested = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment": experiment_id,
        "createdUtc": _utc_now(),
        "preset": preset,
        "trainingMode": training_mode,
        "promotionEligible": _promotion_eligible(training_mode, config),
        "baseConfig": str(base_config_path),
        "requestedOverrides": requested_overrides,
        "asset": {
            "displayName": asset_name,
            "query": asset_query,
            "selectionKey": selection_key,
        },
    }
    (directory / "config.json").write_text(
        json.dumps(requested, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_experiment_manifest(
        directory / "experiment.json",
        {
            **requested,
            "status": "created",
            "resolvedConfigSha256": config_sha256(config),
            "modelScope": "tuning",
            "trainingMode": training_mode,
            "promotionEligible": _promotion_eligible(training_mode, config),
            "fullProduction": False,
        },
    )
    return experiment_id, directory, config


def load_resolved_config(repo_root: Path, experiment_id: str) -> V9Config:
    path = experiment_dir(repo_root, experiment_id) / "resolved_config.json"
    if not path.is_file():
        raise RuntimeError(f"experiment resolved config is missing: {path}")
    return V9Config.load(path)


def capability_payload(
    experiment_id: str,
    *,
    asset_name: str,
    asset_query: str,
    selection_key: str,
    training_mode: str = "full",
    appearance_enabled: bool = False,
) -> dict[str, Any]:
    promotion_eligible = str(training_mode).lower() == "full" and bool(appearance_enabled)
    return {
        "schema": CAPABILITY_SCHEMA,
        "modelScope": "tuning",
        "fullProduction": False,
        "experiment": experiment_id,
        "trainingMode": training_mode,
        "promotionEligible": promotion_eligible,
        "supportedAssets": [
            {
                "displayName": asset_name,
                "query": asset_query,
                "selectionKey": selection_key,
            }
        ],
        "lockedCapabilities": [
            "all-ships-preview",
            "full-dataset-inference",
            "production-release",
        ],
    }


def _flatten_metrics(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_metrics(f"{prefix}_{key}" if prefix else str(key), child, row)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        row[prefix] = value


def finalise_experiment(
    repo_root: Path,
    experiment_id: str,
    *,
    asset_name: str,
    asset_query: str,
    selection_key: str,
    training_mode: str = "full",
) -> dict[str, Any]:
    directory = experiment_dir(repo_root, experiment_id)
    config = load_resolved_config(repo_root, experiment_id)
    checkpoint_path = directory / config.checkpoint_name
    state_path = directory / config.training_state_name
    metadata_path = directory / config.metadata_name
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"experiment training did not produce a completed checkpoint under {directory}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = list(checkpoint.get("history", [])) if isinstance(checkpoint, dict) else []

    best_copy = directory / "checkpoint_best.pt"
    shutil.copy2(checkpoint_path, best_copy)

    if state_path.is_file():
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        latest_payload = {
            "schema": checkpoint.get("schema"),
            "state_dict": state.get("state_dict"),
            "config": config.to_dict(),
            "parameter_count": checkpoint.get("parameter_count"),
            "model_sha256": checkpoint.get("model_sha256"),
            "best_epoch": state.get("completed_epoch"),
            "best_validation_total": (
                history[-1].get("validation", {}).get("total") if history else None
            ),
            "dataset_fingerprint": state.get("dataset_fingerprint"),
            "history": history,
        }
        torch.save(latest_payload, directory / "checkpoint_latest.pt")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics_payload = {
        **metadata,
        "experiment": experiment_id,
        "modelScope": "tuning",
        "supportedAsset": asset_name,
        "resolvedConfigSha256": config_sha256(config),
        "epochCount": len(history),
    }
    (directory / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if history:
        flattened: list[dict[str, Any]] = []
        for record in history:
            row: dict[str, Any] = {}
            _flatten_metrics("", record, row)
            flattened.append(row)
        columns = sorted({key for row in flattened for key in row})
        with (directory / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flattened)

    previews = directory / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    capability = capability_payload(
        experiment_id,
        asset_name=asset_name,
        asset_query=asset_query,
        selection_key=selection_key,
        training_mode=training_mode,
        appearance_enabled=config.appearance_enabled,
    )
    (directory / "capabilities.json").write_text(
        json.dumps(capability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = load_experiment_manifest(repo_root, experiment_id)
    manifest.update(
        {
            "status": "completed",
            "trainingMode": training_mode,
            "promotionEligible": _promotion_eligible(training_mode, config),
            "completedUtc": _utc_now(),
            "checkpointBest": str(best_copy),
            "checkpointCanonical": str(checkpoint_path),
            "metrics": str(directory / "metrics.json"),
            "trainingLog": str(directory / "training_log.csv"),
            "capabilities": str(directory / "capabilities.json"),
            "bestEpoch": metadata.get("bestEpoch"),
            "bestValidationTotal": metadata.get("bestValidationTotal"),
            "trainingSafetyPass": metadata.get("trainingSafetyPass", metadata.get("acceptancePass")),
            "acceptancePass": metadata.get("acceptancePass"),
            "reconstructionAcceptancePass": metadata.get("reconstructionAcceptancePass", False),
            "acceptanceRegressionFraction": metadata.get("acceptanceRegressionFraction"),
        }
    )
    write_experiment_manifest(directory / "experiment.json", manifest)
    return manifest


def latest_validation_contact_sheet(repo_root: Path, experiment_id: str) -> Path:
    samples = experiment_dir(repo_root, experiment_id) / "samples"
    candidates = sorted(samples.glob("epoch_*/*validation_contact_sheet.png"))
    if not candidates:
        candidates = sorted(samples.glob("epoch_*/validation_contact_sheet.png"))
    if not candidates:
        raise RuntimeError(f"experiment has no validation contact sheet: {experiment_id}")
    return candidates[-1]


def selected_promotion(repo_root: Path) -> dict[str, Any] | None:
    path = promoted_root(repo_root) / "selected_experiment.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("schema") != PROMOTION_SCHEMA:
        return None
    return payload


def promote_experiment(
    repo_root: Path,
    experiment_id: str,
    *,
    full_base_config_path: Path,
) -> tuple[Path, dict[str, Any]]:
    experiment_id = experiment_id.upper()
    directory = experiment_dir(repo_root, experiment_id)
    if not (directory / "checkpoint_best.pt").is_file():
        raise RuntimeError(f"experiment is not complete and cannot be promoted: {experiment_id}")
    manifest = load_experiment_manifest(repo_root, experiment_id)
    resolved_for_promotion = load_resolved_config(repo_root, experiment_id)
    if not bool(resolved_for_promotion.appearance_enabled):
        raise RuntimeError(
            f"{experiment_id} is a V9.4 geometry-only proof and cannot be promoted. "
            "First prove GeometryNet A/B quality, then enable/train the frozen-geometry appearance stage."
        )
    if str(manifest.get("trainingMode") or "").lower() != "full" or not bool(manifest.get("promotionEligible")):
        raise RuntimeError(
            f"{experiment_id} is not an eligible Full promotion-proof experiment. "
            "Run Stage 1 in Full / promotion proof mode using an appearance-enabled proven configuration first."
        )
    if not bool(manifest.get("trainingSafetyPass", manifest.get("acceptancePass"))):
        raise RuntimeError(
            f"{experiment_id} failed the training safety gate and cannot be promoted. "
            f"regressionFraction={manifest.get('acceptanceRegressionFraction')}"
        )
    if not bool(manifest.get("reconstructionAcceptancePass")):
        raise RuntimeError(
            f"{experiment_id} has not passed the staged reconstruction acceptance gate and cannot be promoted. "
            f"geometryVerdict={manifest.get('combinedGeometryAuditVerdict')}"
        )
    preview_manifest = directory / "previews" / "preview_manifest.json"
    if not preview_manifest.is_file():
        raise RuntimeError(
            f"experiment has not been Raven-previewed and cannot be promoted: {experiment_id}. "
            "Run Stage 1 Full / promotion proof and complete the renderer preview first."
        )
    preview_payload = json.loads(preview_manifest.read_text(encoding="utf-8"))
    quality_gate = preview_payload.get("qualityGate", {})
    if (
        preview_payload.get("status") != "completed"
        or not bool(quality_gate.get("trainingSafetyPass", quality_gate.get("acceptancePass")))
        or not bool(quality_gate.get("reconstructionAcceptancePass"))
    ):
        raise RuntimeError(
            f"{experiment_id} renderer preview exists but its training/reconstruction quality gates did not pass; promotion is locked."
        )

    tuned = load_resolved_config(repo_root, experiment_id)
    full_base = V9Config.load(full_base_config_path)
    promoted = V9Config.load(directory / "resolved_config.json")
    restored: dict[str, Any] = {}
    for field in sorted(PRODUCTION_SCOPE_FIELDS):
        value = getattr(full_base, field)
        setattr(promoted, field, value)
        restored[field] = value
    promoted.validate()

    root = promoted_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / f"v9_full_from_{experiment_id}.json"
    promoted_payload = promoted.to_dict()
    # Traceability metadata is intentionally ignored by V9Config.load but
    # travels with the promoted JSON so a production config is self-identifying.
    promoted_payload.update(
        {
            "source_experiment": experiment_id,
            "promotion_schema": PROMOTION_SCHEMA,
            "source_resolved_config_sha256": config_sha256(tuned),
        }
    )
    config_path.write_text(
        json.dumps(promoted_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    semantic_tuned = tuned.to_dict()
    semantic_promoted = promoted.to_dict()
    for field in PRODUCTION_SCOPE_FIELDS:
        semantic_tuned.pop(field, None)
        semantic_promoted.pop(field, None)
    if semantic_tuned != semantic_promoted:
        raise RuntimeError("promotion altered one or more tuned semantic hyperparameters")

    record = {
        "schema": PROMOTION_SCHEMA,
        "promotedUtc": _utc_now(),
        "sourceExperiment": experiment_id,
        "sourceResolvedConfig": str(directory / "resolved_config.json"),
        "sourceResolvedConfigSha256": config_sha256(tuned),
        "promotedConfig": str(config_path),
        "promotedConfigSha256": config_sha256(promoted),
        "fullBaseConfig": str(full_base_config_path),
        "restoredProductionScopeFields": restored,
        "semanticHyperparametersPreservedExactly": True,
    }
    pointer = root / "selected_experiment.json"
    pointer.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = load_experiment_manifest(repo_root, experiment_id)
    manifest["promotedUtc"] = record["promotedUtc"]
    manifest["promotedConfig"] = str(config_path)
    write_experiment_manifest(directory / "experiment.json", manifest)
    return config_path, record
