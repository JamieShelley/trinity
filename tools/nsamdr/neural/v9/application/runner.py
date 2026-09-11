"""Top-level SR-first Quick training application."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import V9Config
from .backend import TrainingBackend
from .clock import UtcClock
from .configuration import (
    ConfigResolver,
    assert_quick_stop_phase,
    assert_sr_first_quick_config,
)
from .domain import ExperimentContext, TrainingOptions
from .experiment import ExperimentRunSession, ExperimentService
from .pipeline import PassDrivenPipeline
from .results import ResultWriter


class TrainingApplication:
    """Compose allocation, SR-first training, qualification, and final lifecycle transitions."""

    def __init__(self, options: TrainingOptions) -> None:
        """Build lightweight application-wide services.

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
        """Print the resolved immutable SR-first work budget.

        Purpose:
            Preserve the CLI/GUI startup summary without retired stage terminology.
        Called by:
            TrainingApplication.run().
        Calls:
            print().
        """
        print("=" * 72, flush=True)
        print("NSAMDR V13 SR-FIRST QUICK WORK BUDGET", flush=True)
        print(f"Experiment               : {context.experiment_id}", flush=True)
        print(
            f"Dataset manifest         : "
            f"{self.options.repo_root / context.config.dataset_manifest}",
            flush=True,
        )
        print(f"SR epochs                : {context.config.detail_epochs}", flush=True)
        print(f"Selector epochs          : {context.config.physical_finetune_epochs}", flush=True)
        print(
            f"Tiles / validation       : "
            f"{context.config.tiles_per_epoch} / {context.config.validation_tiles}",
            flush=True,
        )
        print(
            f"Resolved config          : {context.directory / 'resolved_config.json'}",
            flush=True,
        )
        print("Authority                : B -> SR candidate C -> BenefitSelector F", flush=True)
        print("=" * 72, flush=True)

    def _build_pipeline(self) -> PassDrivenPipeline:
        """Compose the current two-stage SR-first production pipeline.

        Purpose:
            Keep application construction separate from pipeline execution.
        Called by:
            TrainingApplication._run_training().
        Calls:
            TrainingBackend(), PassDrivenPipeline().
        """
        return PassDrivenPipeline(
            backend=TrainingBackend(),
            experiments=self.experiments,
            options=self.options,
        )

    def _run_diagnostic_stage(self, context: ExperimentContext) -> dict[str, Any]:
        """Run the only supported hidden stage stop: SR detail reconstruction.

        Purpose:
            Preserve useful diagnostic stopping without exposing retired curricula.
        Called by:
            TrainingApplication._run_training().
        Calls:
            assert_quick_stop_phase(), TrainingBackend.run().
        """
        assert_quick_stop_phase(self.options.stop_after_phase)
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

    def _run_training(self, context: ExperimentContext) -> tuple[dict[str, Any], int]:
        """Choose diagnostic detail-only execution or complete SR-first training.

        Purpose:
            Give run() one small training operation independent of trainer mechanics.
        Called by:
            TrainingApplication.run().
        Calls:
            TrainingApplication._run_diagnostic_stage(), _build_pipeline(), PassDrivenPipeline.run().
        """
        if self.options.stop_after_phase is not None:
            return self._run_diagnostic_stage(context), 0

        metadata, code = self._build_pipeline().run(context)
        if metadata is None and code == 0:
            raise RuntimeError("SR-first trainer returned no final metadata")
        return metadata or {}, code

    def _handle_stage_pause(
        self,
        context: ExperimentContext,
        metadata: dict[str, Any],
    ) -> bool:
        """Persist and report a reached detail-reconstruction diagnostic stop.

        Purpose:
            Stop normal finalisation when detail-only execution was explicitly requested.
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
        """Finalise a successful SR-first training application run.

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
        """Execute the canonical V13 SR-first Quick application lifecycle.

        Purpose:
            Provide one readable flow: config -> experiment -> SR C -> selector F -> finalise.
        Called by:
            application.main().
        Calls:
            TrainingApplication helpers, ExperimentService.allocate_or_resume(), ExperimentRunSession.
        """
        if str(self.options.training_mode).lower() != "quick":
            raise RuntimeError(
                "Full Training is disabled until it is converted to the V13 SR-first authority."
            )

        base_path, base, dataset_config = self._load_configs()
        context = self.experiments.allocate_or_resume(
            base_path,
            base,
            dataset_config,
        )
        assert_sr_first_quick_config(context.config)

        if self.options.allocate_only:
            self.experiments.mark_allocated_only(context)
            return 0

        self._print_banner(context)
        with ExperimentRunSession(self.experiments, context):
            metadata, pipeline_code = self._run_training(context)
            if pipeline_code != 0:
                return pipeline_code
            if self._handle_stage_pause(context, metadata):
                return 0
            return self._finalise(context)
