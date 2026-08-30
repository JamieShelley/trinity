"""Top-level canonical NSAMDR training application composed from small services."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import V9Config
from ..evolution import EvolutionaryRecoveryController
from .backend import TrainingBackend
from .clock import UtcClock
from .configuration import ConfigResolver
from .domain import ExperimentContext, TrainingOptions
from .experiment import ExperimentRunSession, ExperimentService
from .gates import QualificationGates, StagePlan
from .pipeline import PassDrivenPipeline
from .results import ResultWriter
from .training_state import TrainingStateService


class TrainingApplication:
    """Compose allocation, evolution, staged training, and final lifecycle transitions."""

    def __init__(self, options: TrainingOptions) -> None:
        """Build lightweight application-wide services that do not allocate models.

        Purpose:
            Establish the composition root for one CLI process.
        Called by:
            application.main().
        Calls:
            ConfigResolver(), ResultWriter(), UtcClock(), ExperimentService().
        """
        self.options = options
        self.resolver = ConfigResolver()
        self.results = ResultWriter(options.repo_root, options.result_file)
        self.clock = UtcClock()
        self.experiments = ExperimentService(
            options.repo_root,
            options,
            self.resolver,
            self.results,
            self.clock,
        )

    def _load_configs(self) -> tuple[Path, V9Config, V9Config]:
        """Load production semantics and dataset-scope configs.

        Purpose:
            Resolve CLI config paths before experiment allocation.
        Called by:
            TrainingApplication.run().
        Calls:
            ConfigResolver.resolve_path(), V9Config.load().
        """
        base_path = self.resolver.resolve_path(
            self.options.repo_root,
            self.options.base_config,
        )
        dataset_requested = self.options.dataset_config or self.options.base_config
        dataset_path = self.resolver.resolve_path(
            self.options.repo_root,
            dataset_requested,
        )
        return base_path, V9Config.load(base_path), V9Config.load(dataset_path)

    def _print_banner(self, context: ExperimentContext) -> None:
        """Print the resolved immutable training work budget before expensive work.

        Purpose:
            Preserve the existing CLI/GUI startup summary in one presentation method.
        Called by:
            TrainingApplication.run().
        Calls:
            print().
        """
        print("=" * 72, flush=True)
        print(
            f"NSAMDR COMPLETE PRODUCTION MODEL - "
            f"{self.options.training_mode.upper()} WORK BUDGET",
            flush=True,
        )
        print(f"Experiment               : {context.experiment_id}", flush=True)
        print(
            f"Dataset manifest         : "
            f"{self.options.repo_root / context.config.dataset_manifest}",
            flush=True,
        )
        print(f"Epochs                   : {context.config.total_epochs}", flush=True)
        print(
            f"Tiles / validation       : "
            f"{context.config.tiles_per_epoch} / {context.config.validation_tiles}",
            flush=True,
        )
        print(
            f"Resolved config          : {context.directory / 'resolved_config.json'}",
            flush=True,
        )
        print(
            "Semantic model config    : production (identical for Quick and Full)",
            flush=True,
        )
        print("=" * 72, flush=True)

    def _build_evolution(self, context: ExperimentContext) -> EvolutionaryRecoveryController:
        """Construct the training-only evolutionary controller for this experiment.

        Purpose:
            Keep Quick/Full population/microstep budget selection out of pipeline internals.
        Called by:
            TrainingApplication.run().
        Calls:
            EvolutionaryRecoveryController().
        """
        return EvolutionaryRecoveryController(
            repo_root=self.options.repo_root,
            experiment_dir=context.directory,
            config=context.config,
            device=self.options.device,
            population=4 if self.options.training_mode == "quick" else 6,
            micro_steps=3 if self.options.training_mode == "quick" else 5,
            max_recoveries=2,
        )

    def _run_discovery(
        self,
        context: ExperimentContext,
        evolution: EvolutionaryRecoveryController,
    ) -> int:
        """Run bounded pre-training capacity discovery and persist rejection on failure.

        Purpose:
            Prevent expensive training when no local-boundary genome passes the real Raven microproof.
        Called by:
            TrainingApplication.run().
        Calls:
            EvolutionaryRecoveryController.discover_before_training(), ExperimentService.reject().
        """
        discovery = evolution.discover_before_training()
        if not discovery.passed:
            code = self.experiments.reject(
                context,
                phase="evolution-capacity-microproof",
                gate_label="real Raven local-boundary capacity",
                metadata={},
            )
            self.results.write(
                {
                    "experiment": context.experiment_id,
                    "directory": str(context.directory),
                    "trainingMode": self.options.training_mode,
                    "status": "training-rejected",
                    "failedStage": "evolution-capacity-microproof",
                    "evolutionReport": str(context.directory / "evolution"),
                }
            )
            print(
                "[evolution] no candidate passed after two bounded generations; "
                "expensive training was not started.",
                flush=True,
            )
            return code

        print(
            f"[evolution] capacity microproof PASS; candidate genome="
            f"{discovery.winner.fingerprint()[:12]} "
            "(not production-locked until B1/B2 passes)",
            flush=True,
        )
        return 0

    def _build_pipeline(
        self,
        evolution: EvolutionaryRecoveryController,
    ) -> PassDrivenPipeline:
        """Compose the pass-driven production pipeline and its explicit collaborators.

        Purpose:
            Keep application construction separate from pipeline execution.
        Called by:
            TrainingApplication._run_training().
        Calls:
            TrainingBackend(), QualificationGates(), StagePlan(),
            TrainingStateService(), PassDrivenPipeline().
        """
        backend = TrainingBackend()
        gates = QualificationGates()
        stages = StagePlan(gates)
        state = TrainingStateService(gates)
        return PassDrivenPipeline(
            backend=backend,
            gates=gates,
            stages=stages,
            state=state,
            experiments=self.experiments,
            evolution=evolution,
            options=self.options,
        )

    def _run_diagnostic_stage(self, context: ExperimentContext) -> dict[str, Any]:
        """Run the hidden/manual single-stage trainer invocation.

        Purpose:
            Preserve diagnostic stop-after-phase behaviour outside the pass-driven pipeline.
        Called by:
            TrainingApplication._run_training().
        Calls:
            TrainingBackend.run().
        """
        backend = TrainingBackend()
        return backend.run(
            context.config,
            self.options.repo_root,
            self.options.device,
            resume=context.resume,
            early_stop_patience=self.options.early_stop_patience,
            early_stop_min_delta=self.options.early_stop_min_delta,
            stop_after_phase=self.options.stop_after_phase,
        )

    def _run_training(
        self,
        context: ExperimentContext,
        evolution: EvolutionaryRecoveryController,
    ) -> tuple[dict[str, Any], int]:
        """Choose diagnostic single-stage or complete pass-driven production training.

        Purpose:
            Give run() one small training operation independent of stage mechanics.
        Called by:
            TrainingApplication.run().
        Calls:
            TrainingApplication._run_diagnostic_stage(), TrainingApplication._build_pipeline(),
            PassDrivenPipeline.run().
        """
        if self.options.stop_after_phase is not None:
            return self._run_diagnostic_stage(context), 0

        pipeline = self._build_pipeline(evolution)
        metadata, code = pipeline.run(context)
        if metadata is None and code == 0:
            raise RuntimeError("pass-driven trainer returned no final metadata")
        return metadata or {}, code

    def _handle_stage_pause(
        self,
        context: ExperimentContext,
        metadata: dict[str, Any],
    ) -> bool:
        """Persist and report a reached hidden diagnostic stage stop.

        Purpose:
            Stop normal finalisation when the user explicitly requested a staged pause.
        Called by:
            TrainingApplication.run().
        Calls:
            ExperimentService.pause_stage().
        """
        if (
            self.options.stop_after_phase is None
            or not bool(metadata.get("stagedStopReached"))
        ):
            return False
        self.experiments.pause_stage(context, metadata)
        return True

    def _finalise(self, context: ExperimentContext) -> int:
        """Finalise a successful training application run.

        Purpose:
            Leave the experiment in trained-pending-qualification exactly once.
        Called by:
            TrainingApplication.run().
        Calls:
            ExperimentService.finalise().
        """
        self.experiments.finalise(context)
        return 0

    def run(self) -> int:
        """Execute the complete canonical training application lifecycle.

        Purpose:
            Provide a readable top-level flow: config -> experiment -> discovery -> training -> finalise.
        Called by:
            application.main().
        Calls:
            TrainingApplication._load_configs(), _print_banner(), _build_evolution(),
            _run_discovery(), _run_training(), _handle_stage_pause(), _finalise(),
            ExperimentService.allocate_or_resume(), ExperimentService.mark_allocated_only(),
            ExperimentRunSession.
        """
        base_path, base, dataset_config = self._load_configs()
        context = self.experiments.allocate_or_resume(
            base_path,
            base,
            dataset_config,
        )
        if self.options.allocate_only:
            self.experiments.mark_allocated_only(context)
            return 0

        self._print_banner(context)
        evolution = self._build_evolution(context)

        with ExperimentRunSession(self.experiments, context):
            discovery_code = self._run_discovery(context, evolution)
            if discovery_code != 0:
                return discovery_code

            metadata, pipeline_code = self._run_training(context, evolution)
            if pipeline_code != 0:
                return pipeline_code

            if self._handle_stage_pause(context, metadata):
                return 0

            return self._finalise(context)
