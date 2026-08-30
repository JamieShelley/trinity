"""Canonical NSAMDR experiment registry and immutable-final helpers."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
from typing import Any

import torch

from .config import V9Config

EXPERIMENT_SCHEMA = "NSAMDR_EXPERIMENT_V3"
FINAL_MANIFEST_SCHEMA = "NSAMDR_PRODUCTION_FINAL_V1"
EXPERIMENT_RE = re.compile(r"^EXP_(\d{4,})$")
DEFAULT_TUNING_ASSET_NAME = "Raven Navy Issue"
DEFAULT_TUNING_ASSET_QUERY = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"

class ExperimentRepository:
    # Purpose: Implement utc now for ExperimentRepository.
    # Called by: finalise_experiment, freeze_final_checkpoint, initialise_experiment, qualify_final_manifest
    # Calls: No same-class helper methods.
    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Purpose: Implement experiments root for ExperimentRepository.
    # Called by: allocate_experiment, experiment_dir, list_experiments
    # Calls: No same-class helper methods.
    def experiments_root(self, repo_root: Path) -> Path:
        return repo_root / "artifacts" / "nsamdr" / "experiments"

    # Purpose: Implement experiment dir for ExperimentRepository.
    # Called by: finalise_experiment, freeze_final_checkpoint, latest_validation_contact_sheet, load_experiment_manifest, load_final_manifest, load_resolved_config, qualify_final_manifest
    # Calls: experiments_root
    def experiment_dir(self, repo_root: Path, experiment_id: str) -> Path:
        value = str(experiment_id).strip().upper()
        if EXPERIMENT_RE.fullmatch(value) is None:
            raise ValueError(f"invalid experiment id: {experiment_id!r}")
        return self.experiments_root(repo_root) / value

    # Purpose: Implement ensure experiment layout for ExperimentRepository.
    # Called by: allocate_experiment, finalise_experiment, freeze_final_checkpoint, initialise_experiment
    # Calls: No same-class helper methods.
    def ensure_experiment_layout(self, directory: Path) -> None:
        """Create the canonical directories shared by Quick and Full experiments."""
        for name in ("metrics", "checkpoints", "evidence", "previews"):
            (directory / name).mkdir(parents=True, exist_ok=True)

    # Purpose: Implement sha256 file for ExperimentRepository.
    # Called by: freeze_final_checkpoint, load_final_manifest
    # Calls: No same-class helper methods.
    def sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    # Purpose: Implement is read only for ExperimentRepository.
    # Called by: freeze_final_checkpoint, load_final_manifest
    # Calls: No same-class helper methods.
    def _is_read_only(self, path: Path) -> bool:
        return not bool(path.stat().st_mode & stat.S_IWRITE)

    # Purpose: Implement make read only for ExperimentRepository.
    # Called by: freeze_final_checkpoint
    # Calls: No same-class helper methods.
    def _make_read_only(self, path: Path) -> None:
        path.chmod(path.stat().st_mode & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)

    # Purpose: Implement resolved within for ExperimentRepository.
    # Called by: freeze_final_checkpoint, load_final_manifest
    # Calls: No same-class helper methods.
    def _resolved_within(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    # Purpose: Implement is production selection kind for ExperimentRepository.
    # Called by: freeze_final_checkpoint, load_final_manifest
    # Calls: No same-class helper methods.
    def is_production_selection_kind(self, value: object) -> bool:
        """Accept only the trainer's canonical final production authority."""
        return str(value or "").strip().lower() == "production-final"

    # Purpose: Implement load final manifest for ExperimentRepository.
    # Called by: list_experiments, qualify_final_manifest
    # Calls: _is_read_only, _resolved_within, experiment_dir, is_production_selection_kind, sha256_file
    def load_final_manifest(
        self,
        repo_root: Path,
        experiment_id: str,
        *,
        require_qualified: bool = True,
    ) -> dict[str, Any]:
        """Load and verify the exact immutable production checkpoint binding."""
        directory = self.experiment_dir(repo_root, experiment_id)
        path = directory / "final_manifest.json"
        if not path.is_file():
            raise RuntimeError(f"experiment has no final manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != FINAL_MANIFEST_SCHEMA:
            raise RuntimeError(f"unsupported final manifest schema: {payload.get('schema')!r}")
        if str(payload.get("experiment") or "").upper() != experiment_id.upper():
            raise RuntimeError(f"final manifest experiment mismatch: {path}")
        if require_qualified and (
            payload.get("status") != "completed" or payload.get("qualified") is not True
        ):
            raise RuntimeError(
                f"experiment final is not qualified: {experiment_id} "
                f"(status={payload.get('status')!r})"
            )
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"final manifest has no checkpoint binding: {path}")
        checkpoint_path = Path(str(checkpoint.get("path") or ""))
        if not checkpoint_path.is_absolute():
            checkpoint_path = directory / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        final_root = (directory / "checkpoints" / "final").resolve()
        if not self._resolved_within(checkpoint_path, final_root) or not checkpoint_path.is_file():
            raise RuntimeError(f"final checkpoint is missing or outside canonical final directory: {checkpoint_path}")
        expected_sha = str(checkpoint.get("sha256") or "").strip().lower()
        actual_sha = self.sha256_file(checkpoint_path)
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(
                f"final checkpoint SHA-256 mismatch: expected={expected_sha or '<missing>'} "
                f"actual={actual_sha} path={checkpoint_path}"
            )
        if not self._is_read_only(checkpoint_path):
            raise RuntimeError(f"final checkpoint is not immutable/read-only: {checkpoint_path}")
        selection = checkpoint.get("sourceSelectionKind") or checkpoint.get("selectionKind")
        if require_qualified and not self.is_production_selection_kind(selection):
            raise RuntimeError(f"final checkpoint is not selector-qualified: {selection!r}")
        payload["_checkpointPath"] = str(checkpoint_path)
        return payload

    # Purpose: Implement list experiments for ExperimentRepository.
    # Called by: External callers and the owning workflow.
    # Calls: experiments_root, load_final_manifest
    def list_experiments(self, repo_root: Path, *, completed_only: bool = False) -> list[str]:
        root = self.experiments_root(repo_root)
        if not root.is_dir():
            return []
        result: list[tuple[int, str]] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = EXPERIMENT_RE.fullmatch(path.name.upper())
            if match is None:
                continue
            if completed_only:
                try:
                    self.load_final_manifest(repo_root, path.name, require_qualified=True)
                except (OSError, RuntimeError, ValueError):
                    continue
            result.append((int(match.group(1)), path.name.upper()))
        return [name for _number, name in sorted(result)]

    # Purpose: Implement allocate experiment for ExperimentRepository.
    # Called by: initialise_experiment
    # Calls: ensure_experiment_layout, experiments_root
    def allocate_experiment(self, repo_root: Path) -> tuple[str, Path]:
        root = self.experiments_root(repo_root)
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
                self.ensure_experiment_layout(path)
                return experiment_id, path
            except FileExistsError:
                number += 1

    # Purpose: Implement load experiment manifest for ExperimentRepository.
    # Called by: finalise_experiment, freeze_final_checkpoint, qualify_final_manifest
    # Calls: experiment_dir
    def load_experiment_manifest(self, repo_root: Path, experiment_id: str) -> dict[str, Any]:
        path = self.experiment_dir(repo_root, experiment_id) / "experiment.json"
        if not path.is_file():
            raise RuntimeError(f"experiment manifest is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema != EXPERIMENT_SCHEMA:
            raise RuntimeError(f"unsupported experiment manifest schema: {schema!r}")
        return payload

    # Purpose: Implement write experiment manifest for ExperimentRepository.
    # Called by: finalise_experiment, initialise_experiment, qualify_final_manifest
    # Calls: No same-class helper methods.
    def write_experiment_manifest(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Purpose: Implement write final manifest for ExperimentRepository.
    # Called by: freeze_final_checkpoint, qualify_final_manifest
    # Calls: No same-class helper methods.
    def write_final_manifest(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Purpose: Implement load resolved config for ExperimentRepository.
    # Called by: finalise_experiment, freeze_final_checkpoint
    # Calls: experiment_dir
    def load_resolved_config(self, repo_root: Path, experiment_id: str) -> V9Config:
        path = self.experiment_dir(repo_root, experiment_id) / "resolved_config.json"
        if not path.is_file():
            raise RuntimeError(f"experiment resolved config is missing: {path}")
        return V9Config.load(path)

    # Purpose: Implement freeze final checkpoint for ExperimentRepository.
    # Called by: External callers and the owning workflow.
    # Calls: _is_read_only, _make_read_only, _resolved_within, _utc_now, ensure_experiment_layout, experiment_dir, is_production_selection_kind, load_experiment_manifest, load_resolved_config, sha256_file, write_final_manifest
    def freeze_final_checkpoint(
        self,
        repo_root: Path,
        experiment_id: str,
        *,
        source_checkpoint: Path,
        source_metadata: Path,
        preflight_path: Path,
    ) -> dict[str, Any]:
        """Freeze the trained full-state checkpoint and write its exact SHA binding.

        The manifest is deliberately pending until postflight and an uncached
        production forward qualify the frozen bytes. A later call may validate and
        reuse the same immutable copy, but never overwrite it.
        """
        directory = self.experiment_dir(repo_root, experiment_id)
        self.ensure_experiment_layout(directory)
        source_checkpoint = source_checkpoint.resolve()
        source_metadata = source_metadata.resolve()
        if not self._resolved_within(source_checkpoint, directory) or not source_checkpoint.is_file():
            raise RuntimeError(f"trained checkpoint is missing or outside experiment: {source_checkpoint}")
        if not self._resolved_within(source_metadata, directory) or not source_metadata.is_file():
            raise RuntimeError(f"trained checkpoint metadata is missing or outside experiment: {source_metadata}")

        try:
            checkpoint_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint_payload = torch.load(source_checkpoint, map_location="cpu")
        if not isinstance(checkpoint_payload, dict) or not isinstance(checkpoint_payload.get("state_dict"), dict):
            raise RuntimeError(f"trained checkpoint has no complete production state_dict: {source_checkpoint}")
        selection_kind = str(checkpoint_payload.get("selection_kind") or "").strip()
        if not self.is_production_selection_kind(selection_kind):
            raise RuntimeError(
                "training did not produce the production-final checkpoint; "
                f"selection_kind={selection_kind or '<missing>'!r}. Renderer remains blocked."
            )
        trainer_qualification = checkpoint_payload.get("final_qualification")
        if not isinstance(trainer_qualification, dict) or trainer_qualification.get("passed") is not True:
            raise RuntimeError(
                "training did not embed passing strict-load, uncached direct-forward final qualification"
            )
        cache_equivalence = checkpoint_payload.get("cache_equivalence")
        if not isinstance(cache_equivalence, dict) or cache_equivalence.get("passed") is not True:
            raise RuntimeError("training did not embed passing cached-versus-uncached equivalence evidence")

        source_sha = self.sha256_file(source_checkpoint)
        metadata_payload = json.loads(source_metadata.read_text(encoding="utf-8"))
        metadata_sha = str(metadata_payload.get("checkpointSha256") or "").strip().lower()
        if metadata_sha != source_sha:
            raise RuntimeError(
                f"checkpoint metadata SHA-256 mismatch: expected={source_sha} actual={metadata_sha or '<missing>'}"
            )
        if metadata_payload.get("schema") != checkpoint_payload.get("schema"):
            raise RuntimeError("checkpoint metadata schema differs from the checkpoint payload schema")
        final_dir = directory / "checkpoints" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_checkpoint = final_dir / "nsamdr_v9_fidelity.pt"
        final_metadata = final_dir / "nsamdr_v9_fidelity.json"
        for source, target in ((source_checkpoint, final_checkpoint), (source_metadata, final_metadata)):
            if target.exists():
                if not target.is_file() or self.sha256_file(target) != self.sha256_file(source):
                    raise RuntimeError(f"refusing to replace an existing immutable final artifact: {target}")
            else:
                shutil.copy2(source, target)

        copied_sha = self.sha256_file(final_checkpoint)
        copied_metadata_sha = self.sha256_file(final_metadata)
        if copied_sha != source_sha or copied_metadata_sha != self.sha256_file(source_metadata):
            raise RuntimeError(
                f"immutable checkpoint copy mismatch: source={source_sha} copy={copied_sha}"
            )
        self._make_read_only(final_checkpoint)
        self._make_read_only(final_metadata)
        frozen_sha = self.sha256_file(final_checkpoint)
        if frozen_sha != source_sha or not self._is_read_only(final_checkpoint):
            raise RuntimeError(f"immutable checkpoint freeze verification failed: {final_checkpoint}")

        config_path = directory / "resolved_config.json"
        config_sha = self.sha256_file(config_path)
        config = self.load_resolved_config(repo_root, experiment_id)
        dataset_manifest = Path(config.dataset_manifest)
        if not dataset_manifest.is_absolute():
            dataset_manifest = repo_root / dataset_manifest
        dataset_payload = (
            json.loads(dataset_manifest.read_text(encoding="utf-8"))
            if dataset_manifest.is_file() else {}
        )
        experiment = self.load_experiment_manifest(repo_root, experiment_id)
        payload: dict[str, Any] = {
            "schema": FINAL_MANIFEST_SCHEMA,
            "experiment": experiment_id,
            "status": "frozen-pending-qualification",
            "qualified": False,
            "createdUtc": self._utc_now(),
            "trainingMode": str(experiment.get("trainingMode") or ""),
            "modelSchema": checkpoint_payload.get("schema"),
            "selectionKind": "production-final",
            "checkpoint": {
                "path": str(final_checkpoint.relative_to(directory)).replace("\\", "/"),
                "metadataPath": str(final_metadata.relative_to(directory)).replace("\\", "/"),
                "metadataSha256": copied_metadata_sha,
                "sha256": frozen_sha,
                "sizeBytes": final_checkpoint.stat().st_size,
                "readOnly": True,
                "immutable": True,
                "schema": checkpoint_payload.get("schema"),
                "selectionKind": "production-final",
                "sourceSelectionKind": selection_kind,
                "strictStateDictRequired": True,
            },
            "config": {
                "path": "resolved_config.json",
                "sha256": config_sha,
            },
            "dataset": {
                "manifest": str(dataset_manifest),
                "fingerprint": dataset_payload.get("fingerprint"),
                "sourceFingerprint": dataset_payload.get("sourceFingerprint"),
            },
            "architecture": {
                "preflight": str(preflight_path.relative_to(directory)).replace("\\", "/"),
                "participation": "architecture_participation.json",
                "postflightPass": False,
                "uncachedProductionForwardPass": False,
            },
            "candidate": None,
            "renderer": {"status": "not-launched"},
        }
        self.write_final_manifest(directory / "final_manifest.json", payload)
        return payload

    # Purpose: Implement qualify final manifest for ExperimentRepository.
    # Called by: External callers and the owning workflow.
    # Calls: _utc_now, experiment_dir, load_experiment_manifest, load_final_manifest, write_experiment_manifest, write_final_manifest
    def qualify_final_manifest(self, repo_root: Path, experiment_id: str, participation_path: Path) -> dict[str, Any]:
        """Mark frozen bytes qualified only after the architecture postflight passes."""
        directory = self.experiment_dir(repo_root, experiment_id)
        manifest_path = directory / "final_manifest.json"
        participation_path = participation_path.resolve()
        expected_participation = (directory / "architecture_participation.json").resolve()
        if participation_path != expected_participation:
            raise RuntimeError(
                f"architecture participation report must use the canonical path: {expected_participation}"
            )
        if not participation_path.is_file():
            raise RuntimeError(f"architecture participation report is missing: {participation_path}")
        participation = json.loads(participation_path.read_text(encoding="utf-8"))
        payload = self.load_final_manifest(repo_root, experiment_id, require_qualified=False)
        checkpoint_path = Path(str(payload["_checkpointPath"])).resolve()
        checkpoint_sha = str(payload["checkpoint"]["sha256"]).strip().lower()
        reported_checkpoint = Path(str(participation.get("checkpoint") or "")).resolve()
        trainer_qualification = participation.get("trainerFinalQualification")
        cache_equivalence = participation.get("cacheEquivalence")
        qualification_checks = {
            "postflight pass": participation.get("pass") is True,
            "checkpoint path": reported_checkpoint == checkpoint_path,
            "checkpoint SHA-256": str(participation.get("checkpointSha256") or "").lower() == checkpoint_sha,
            "manifest SHA-256": str(participation.get("manifestCheckpointSha256") or "").lower() == checkpoint_sha,
            "model schema": str(participation.get("schema") or "") == str(payload.get("modelSchema") or ""),
            "checkpoint schema": str(participation.get("checkpointSchema") or "") == str(payload.get("modelSchema") or ""),
            "manifest schema": str(participation.get("manifestSchema") or "") == str(payload.get("modelSchema") or ""),
            "production selection": participation.get("selectionKind") == "production-final",
            "strict state load": participation.get("strictStateDictLoad") is True,
            "trainer final qualification": isinstance(trainer_qualification, dict)
            and trainer_qualification.get("passed") is True,
            "cache equivalence": isinstance(cache_equivalence, dict)
            and cache_equivalence.get("passed") is True,
        }
        failed_checks = [label for label, passed in qualification_checks.items() if not passed]
        if failed_checks:
            raise RuntimeError(
                "architecture postflight does not bind and qualify the frozen final checkpoint: "
                + ", ".join(failed_checks)
            )
        payload.pop("_checkpointPath", None)
        payload["status"] = "completed"
        payload["qualified"] = True
        payload["qualifiedUtc"] = self._utc_now()
        architecture = payload.setdefault("architecture", {})
        architecture["participation"] = str(participation_path.relative_to(directory)).replace("\\", "/")
        architecture["postflightPass"] = True
        architecture["uncachedProductionForwardPass"] = bool(
            participation.get("uncachedProductionForwardPass", participation.get("pass"))
        )
        self.write_final_manifest(manifest_path, payload)

        experiment = self.load_experiment_manifest(repo_root, experiment_id)
        experiment.update(
            {
                "status": "completed",
                "completedUtc": self._utc_now(),
                "qualified": True,
                "finalManifest": str(manifest_path),
                "finalCheckpoint": str(payload["checkpoint"]["path"]),
                "finalCheckpointSha256": str(payload["checkpoint"]["sha256"]),
            }
        )
        self.write_experiment_manifest(directory / "experiment.json", experiment)
        return payload

    # Purpose: Implement config sha256 for ExperimentRepository.
    # Called by: finalise_experiment, initialise_experiment
    # Calls: No same-class helper methods.
    def config_sha256(self, config: V9Config) -> str:
        encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    # Purpose: Implement initialise experiment for ExperimentRepository.
    # Called by: External callers and the owning workflow.
    # Calls: _utc_now, allocate_experiment, config_sha256, ensure_experiment_layout, write_experiment_manifest
    def initialise_experiment(
        self,
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
        experiment_id, directory = self.allocate_experiment(repo_root)
        self.ensure_experiment_layout(directory)
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
            "createdUtc": self._utc_now(),
            "preset": preset,
            "trainingMode": training_mode,
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
        self.write_experiment_manifest(
            directory / "experiment.json",
            {
                **requested,
                "status": "created",
                "resolvedConfigSha256": self.config_sha256(config),
                "modelScope": "production",
                "trainingMode": training_mode,
                "fullProduction": True,
            },
        )
        return experiment_id, directory, config

    # Purpose: Implement flatten metrics for ExperimentRepository.
    # Called by: _flatten_metrics, finalise_experiment
    # Calls: _flatten_metrics
    def _flatten_metrics(self, prefix: str, value: Any, row: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                self._flatten_metrics(f"{prefix}_{key}" if prefix else str(key), child, row)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            row[prefix] = value

    # Purpose: Implement finalise experiment for ExperimentRepository.
    # Called by: External callers and the owning workflow.
    # Calls: _flatten_metrics, _utc_now, config_sha256, ensure_experiment_layout, experiment_dir, load_experiment_manifest, load_resolved_config, write_experiment_manifest
    def finalise_experiment(
        self,
        repo_root: Path,
        experiment_id: str,
        *,
        asset_name: str,
        asset_query: str,
        selection_key: str,
        training_mode: str = "full",
    ) -> dict[str, Any]:
        directory = self.experiment_dir(repo_root, experiment_id)
        self.ensure_experiment_layout(directory)
        config = self.load_resolved_config(repo_root, experiment_id)
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

        training_best = directory / "checkpoints" / "training_best.pt"
        shutil.copy2(checkpoint_path, training_best)

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
            torch.save(latest_payload, directory / "checkpoints" / "training_latest.pt")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics_payload = {
            **metadata,
            "experiment": experiment_id,
            "modelScope": "production",
            "supportedAsset": asset_name,
            "resolvedConfigSha256": self.config_sha256(config),
            "epochCount": len(history),
        }
        metrics_copy = directory / "metrics" / "summary.json"
        metrics_copy.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if history:
            flattened: list[dict[str, Any]] = []
            for record in history:
                row: dict[str, Any] = {}
                self._flatten_metrics("", record, row)
                flattened.append(row)
            columns = sorted({key for row in flattened for key in row})
            with (directory / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(flattened)

        manifest = self.load_experiment_manifest(repo_root, experiment_id)
        manifest.update(
            {
                "status": "trained-pending-qualification",
                "trainingMode": training_mode,
                "completedUtc": self._utc_now(),
                "checkpointBest": str(training_best),
                "checkpointCanonical": str(checkpoint_path),
                "metrics": str(metrics_copy),
                "trainingLog": str(directory / "training_log.csv"),
                "bestEpoch": metadata.get("bestEpoch"),
                "bestValidationTotal": metadata.get("bestValidationTotal"),
                "trainingSafetyPass": metadata.get("trainingSafetyPass", metadata.get("acceptancePass")),
                "acceptancePass": metadata.get("acceptancePass"),
                "reconstructionAcceptancePass": metadata.get("reconstructionAcceptancePass", False),
                "acceptanceRegressionFraction": metadata.get("acceptanceRegressionFraction"),
            }
        )
        self.write_experiment_manifest(directory / "experiment.json", manifest)
        return manifest

    # Purpose: Implement latest validation contact sheet for ExperimentRepository.
    # Called by: External callers and the owning workflow.
    # Calls: experiment_dir
    def latest_validation_contact_sheet(self, repo_root: Path, experiment_id: str) -> Path:
        samples = self.experiment_dir(repo_root, experiment_id) / "samples"
        candidates = sorted(samples.glob("epoch_*/*validation_contact_sheet.png"))
        if not candidates:
            candidates = sorted(samples.glob("epoch_*/validation_contact_sheet.png"))
        if not candidates:
            raise RuntimeError(f"experiment has no validation contact sheet: {experiment_id}")
        return candidates[-1]

_experiment_repository = ExperimentRepository()
_utc_now = _experiment_repository._utc_now
experiments_root = _experiment_repository.experiments_root
experiment_dir = _experiment_repository.experiment_dir
ensure_experiment_layout = _experiment_repository.ensure_experiment_layout
sha256_file = _experiment_repository.sha256_file
_is_read_only = _experiment_repository._is_read_only
_make_read_only = _experiment_repository._make_read_only
_resolved_within = _experiment_repository._resolved_within
is_production_selection_kind = _experiment_repository.is_production_selection_kind
load_final_manifest = _experiment_repository.load_final_manifest
list_experiments = _experiment_repository.list_experiments
allocate_experiment = _experiment_repository.allocate_experiment
load_experiment_manifest = _experiment_repository.load_experiment_manifest
write_experiment_manifest = _experiment_repository.write_experiment_manifest
write_final_manifest = _experiment_repository.write_final_manifest
load_resolved_config = _experiment_repository.load_resolved_config
freeze_final_checkpoint = _experiment_repository.freeze_final_checkpoint
qualify_final_manifest = _experiment_repository.qualify_final_manifest
config_sha256 = _experiment_repository.config_sha256
initialise_experiment = _experiment_repository.initialise_experiment
_flatten_metrics = _experiment_repository._flatten_metrics
finalise_experiment = _experiment_repository.finalise_experiment
latest_validation_contact_sheet = _experiment_repository.latest_validation_contact_sheet
