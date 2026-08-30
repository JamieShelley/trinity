"""Experiment allocation, manifest lifecycle, and RAII failure ownership."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import traceback
from types import TracebackType
from typing import Any

from ..config import V9Config
from ..experiments import (
    DEFAULT_TUNING_ASSET_NAME,
    DEFAULT_TUNING_ASSET_QUERY,
    experiment_dir,
    finalise_experiment,
    initialise_experiment,
    load_experiment_manifest,
    load_resolved_config,
    write_experiment_manifest,
)
from .clock import UtcClock
from .configuration import ConfigResolver
from .domain import ExperimentContext, TrainingOptions
from .results import ResultWriter


class ExperimentService:
    """Own experiment identity, immutable resolved config, and manifest transitions."""

    def __init__(
        self,
        repo_root: Path,
        options: TrainingOptions,
        resolver: ConfigResolver,
        results: ResultWriter,
        clock: UtcClock,
    ) -> None:
        """Compose experiment lifecycle dependencies for one CLI invocation.

        Purpose:
            Centralise manifest/path/result ownership previously embedded in main().
        Called by:
            TrainingApplication.run().
        Calls:
            No project functions.
        """
        self.repo_root = Path(repo_root).resolve()
        self.options = options
        self.resolver = resolver
        self.results = results
        self.clock = clock

    def _asset_fields(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Resolve asset display/query/selection fields with production defaults.

        Purpose:
            Keep asset metadata extraction identical for new and resumed experiments.
        Called by:
            ExperimentService._allocate_new(), ExperimentService._resume_existing().
        Calls:
            No project functions.
        """
        asset = payload.get("asset", {}) if isinstance(payload, dict) else {}
        asset_name = str(asset.get("displayName") or DEFAULT_TUNING_ASSET_NAME)
        asset_query = str(asset.get("query") or DEFAULT_TUNING_ASSET_QUERY)
        selection_key = str(asset.get("selectionKey") or "")
        return asset_name, asset_query, selection_key

    def _dataset_manifest_path(self, config: V9Config) -> Path:
        """Resolve the configured dataset manifest against the repository root.

        Purpose:
            Provide one canonical dataset-path rule for allocation and resume.
        Called by:
            ExperimentService._allocate_new(), ExperimentService._resume_existing().
        Calls:
            No project functions.
        """
        path = Path(config.dataset_manifest)
        return path if path.is_absolute() else self.repo_root / path

    def _allocate_new(
        self,
        base_config_path: Path,
        base: V9Config,
        dataset_config: V9Config,
    ) -> ExperimentContext:
        """Allocate a new experiment with one immutable resolved config.

        Purpose:
            Reproduce the original NEW experiment allocation path.
        Called by:
            ExperimentService.allocate_or_resume().
        Calls:
            ConfigResolver.resolve_overrides(), initialise_experiment(), ExperimentService._asset_fields().
        """
        overrides = self.resolver.resolve_overrides(self.options, base, dataset_config)
        resolved_probe = copy.deepcopy(base)
        for key, value in overrides.items():
            setattr(resolved_probe, key, value)
        resolved_probe.validate()

        dataset_manifest = self._dataset_manifest_path(resolved_probe)
        if not dataset_manifest.is_file() and not self.options.allocate_only:
            raise RuntimeError(f"prepared dataset manifest is missing: {dataset_manifest}")
        dataset = (
            json.loads(dataset_manifest.read_text(encoding="utf-8"))
            if dataset_manifest.is_file()
            else {}
        )
        asset_name, asset_query, selection_key = self._asset_fields(dataset)
        experiment_id, directory, config = initialise_experiment(
            self.repo_root,
            base_config_path,
            overrides,
            preset=self.options.preset,
            asset_name=asset_name,
            asset_query=asset_query,
            selection_key=selection_key,
            training_mode=self.options.training_mode,
        )
        print(f"[experiment] Allocated {experiment_id}: {directory}", flush=True)
        return ExperimentContext(
            experiment_id=experiment_id,
            directory=directory,
            config=config,
            resume=False,
            asset_name=asset_name,
            asset_query=asset_query,
            selection_key=selection_key,
        )

    def _resume_existing(self, experiment_id: str) -> ExperimentContext:
        """Open an in-progress experiment without changing its immutable resolved config.

        Purpose:
            Reproduce the original EXP_#### resume validation path.
        Called by:
            ExperimentService.allocate_or_resume().
        Calls:
            experiment_dir(), load_resolved_config(), load_experiment_manifest(),
            ExperimentService._dataset_manifest_path(), ExperimentService._asset_fields().
        """
        directory = experiment_dir(self.repo_root, experiment_id)
        config = load_resolved_config(self.repo_root, experiment_id)
        manifest = load_experiment_manifest(self.repo_root, experiment_id)

        stored_mode = str(manifest.get("trainingMode") or "").lower()
        if stored_mode != self.options.training_mode:
            raise RuntimeError(
                f"{experiment_id} is {stored_mode!r}, not requested {self.options.training_mode!r}"
            )
        status = str(manifest.get("status") or "").lower()
        if status in {"trained-pending-qualification", "completed"}:
            raise RuntimeError(
                f"{experiment_id} already produced a final training result and is immutable; "
                "resume the orchestrator qualification or allocate a new experiment"
            )

        dataset_manifest = self._dataset_manifest_path(config)
        if not dataset_manifest.is_file():
            raise RuntimeError(f"experiment dataset manifest is missing: {dataset_manifest}")

        asset_name, asset_query, selection_key = self._asset_fields(manifest)
        state_path = directory / config.training_state_name
        if self.options.control == "resume" and not state_path.is_file():
            raise RuntimeError(f"resume requested but experiment state is missing: {state_path}")
        resume = state_path.is_file()
        print(
            f"[experiment] Reusing immutable resolved config: {directory / 'resolved_config.json'}",
            flush=True,
        )
        return ExperimentContext(
            experiment_id=experiment_id,
            directory=directory,
            config=config,
            resume=resume,
            asset_name=asset_name,
            asset_query=asset_query,
            selection_key=selection_key,
        )

    def allocate_or_resume(
        self,
        base_config_path: Path,
        base: V9Config,
        dataset_config: V9Config,
    ) -> ExperimentContext:
        """Choose NEW allocation or validated existing-experiment resume.

        Purpose:
            Present TrainingApplication with one resolved experiment context.
        Called by:
            TrainingApplication.run().
        Calls:
            ExperimentService._allocate_new(), ExperimentService._resume_existing().
        """
        requested = str(self.options.experiment).strip().upper()
        if requested in {"", "NEW"}:
            return self._allocate_new(base_config_path, base, dataset_config)
        return self._resume_existing(requested)

    def mark_allocated_only(self, context: ExperimentContext) -> None:
        """Persist a no-training allocation result.

        Purpose:
            Support the hidden allocate-only orchestration probe without starting training.
        Called by:
            TrainingApplication.run().
        Calls:
            load_experiment_manifest(), write_experiment_manifest(), ResultWriter.write().
        """
        manifest = load_experiment_manifest(self.repo_root, context.experiment_id)
        manifest.update({"status": "allocated", "lastStoppedUtc": self.clock.now()})
        write_experiment_manifest(context.directory / "experiment.json", manifest)
        self.results.write(
            {
                "experiment": context.experiment_id,
                "directory": str(context.directory),
                "trainingMode": self.options.training_mode,
                "allocatedOnly": True,
            }
        )
        print(f"[experiment] Allocation complete: {context.experiment_id}", flush=True)

    def mark_running(self, context: ExperimentContext) -> None:
        """Set the experiment manifest to running.

        Purpose:
            Acquire lifecycle ownership immediately before discovery/training begins.
        Called by:
            ExperimentRunSession.__enter__().
        Calls:
            load_experiment_manifest(), write_experiment_manifest().
        """
        manifest = load_experiment_manifest(self.repo_root, context.experiment_id)
        manifest.update(
            {
                "status": "running",
                "trainingMode": self.options.training_mode,
                "lastStartedUtc": self.clock.now(),
            }
        )
        write_experiment_manifest(context.directory / "experiment.json", manifest)

    def mark_failed(
        self,
        context: ExperimentContext,
        *,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Persist an exception-terminated run and its traceback as experiment evidence.

        Purpose:
            Keep exported diagnostics self-contained instead of reducing a runtime
            failure to the generic ``interrupted-or-failed`` manifest status.
        Called by:
            ExperimentRunSession.__exit__().
        Calls:
            load_experiment_manifest(), traceback.format_exception(),
            write_experiment_manifest().
        """
        failure_type = (
            exc_type.__name__
            if exc_type is not None
            else type(exc).__name__ if exc is not None else "UnknownException"
        )
        failure_message = str(exc) if exc is not None else ""
        formatted_traceback = (
            "".join(traceback.format_exception(exc_type, exc, tb))
            if exc_type is not None and exc is not None
            else ""
        )
        evidence = context.directory / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        failure_path = evidence / "runtime_failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema": "NSAMDR_RUNTIME_FAILURE_V1",
                    "experiment": context.experiment_id,
                    "capturedUtc": self.clock.now(),
                    "exceptionType": failure_type,
                    "message": failure_message,
                    "traceback": formatted_traceback,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        failed = load_experiment_manifest(self.repo_root, context.experiment_id)
        failed.update(
            {
                "status": "interrupted-or-failed",
                "lastStoppedUtc": self.clock.now(),
                "failedExceptionType": failure_type,
                "failedExceptionMessage": failure_message,
                "failureEvidence": "evidence/runtime_failure.json",
            }
        )
        write_experiment_manifest(context.directory / "experiment.json", failed)

    def reject(
        self,
        context: ExperimentContext,
        *,
        phase: str,
        gate_label: str,
        metadata: dict[str, Any],
    ) -> int:
        """Persist a fail-closed stage rejection and caller-facing result.

        Purpose:
            Stop downstream execution while preserving exact failure stage/gate evidence.
        Called by:
            PassDrivenPipeline and TrainingApplication discovery rejection.
        Calls:
            load_experiment_manifest(), write_experiment_manifest(), ResultWriter.write().
        """
        manifest = load_experiment_manifest(self.repo_root, context.experiment_id)
        manifest.update(
            {
                "status": "training-rejected",
                "failedStage": phase,
                "failedGate": gate_label,
                "epochsCompleted": int(metadata.get("epochsCompleted", 0)),
                "lastStoppedUtc": self.clock.now(),
            }
        )
        write_experiment_manifest(context.directory / "experiment.json", manifest)
        self.results.write(
            {
                "experiment": context.experiment_id,
                "directory": str(context.directory),
                "trainingMode": self.options.training_mode,
                "status": "training-rejected",
                "failedStage": phase,
                "failedGate": gate_label,
                "epochsCompleted": int(metadata.get("epochsCompleted", 0)),
            }
        )
        return 2

    def pause_stage(
        self,
        context: ExperimentContext,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a hidden/manual staged-stop result.

        Purpose:
            Preserve diagnostic partial-training behaviour without finalising the experiment.
        Called by:
            TrainingApplication._handle_stage_pause().
        Calls:
            load_experiment_manifest(), write_experiment_manifest(), ResultWriter.write().
        """
        paused = load_experiment_manifest(self.repo_root, context.experiment_id)
        paused.update(
            {
                "status": "stage-paused",
                "stagedStopPhase": self.options.stop_after_phase,
                "epochsCompleted": int(metadata.get("epochsCompleted", 0)),
                "lastStoppedUtc": self.clock.now(),
            }
        )
        write_experiment_manifest(context.directory / "experiment.json", paused)
        self.results.write(
            {
                "experiment": context.experiment_id,
                "directory": str(context.directory),
                "trainingMode": self.options.training_mode,
                "partialTraining": True,
                "stagedStopPhase": self.options.stop_after_phase,
            }
        )

    def finalise(self, context: ExperimentContext) -> None:
        """Finalise successful training while leaving qualification to the outer workflow.

        Purpose:
            Preserve the trained-pending-qualification terminal state.
        Called by:
            TrainingApplication.run().
        Calls:
            finalise_experiment(), write_experiment_manifest(), ResultWriter.write().
        """
        completed = finalise_experiment(
            self.repo_root,
            context.experiment_id,
            asset_name=context.asset_name,
            asset_query=context.asset_query,
            selection_key=context.selection_key,
            training_mode=self.options.training_mode,
        )
        write_experiment_manifest(context.directory / "experiment.json", completed)
        self.results.write(
            {
                "experiment": context.experiment_id,
                "directory": str(context.directory),
                "trainingMode": self.options.training_mode,
                "status": "trained-pending-qualification",
            }
        )
        print(
            f"[experiment] Training complete; qualification pending: {context.experiment_id}",
            flush=True,
        )


class ExperimentRunSession:
    """RAII-style manifest owner for discovery/training exception safety."""

    def __init__(self, service: ExperimentService, context: ExperimentContext) -> None:
        """Bind lifecycle service/context without changing manifest state.

        Purpose:
            Prepare lexical ownership of running/failed state.
        Called by:
            TrainingApplication.run() through a with-statement.
        Calls:
            No project functions.
        """
        self.service = service
        self.context = context

    def __enter__(self) -> "ExperimentRunSession":
        """Acquire running lifecycle state.

        Purpose:
            Mark the experiment running exactly when expensive orchestration begins.
        Called by:
            Python context-manager protocol.
        Calls:
            ExperimentService.mark_running().
        """
        self.service.mark_running(self.context)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Release lifecycle ownership and preserve exception evidence on failure.

        Purpose:
            Guarantee failure-state cleanup and retain the traceback required to
            diagnose a failed Quick/Full run from its diagnostics ZIP.
        Called by:
            Python context-manager protocol.
        Calls:
            ExperimentService.mark_failed().
        """
        if exc_type is not None:
            self.service.mark_failed(
                self.context,
                exc_type=exc_type,
                exc=exc,
                tb=tb,
            )
        return False
