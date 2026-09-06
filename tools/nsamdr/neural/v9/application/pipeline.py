"""Pass-driven production training pipeline composed from explicit services."""
from __future__ import annotations

from typing import Any

from ..evolution import EvolutionaryRecoveryController, FailureKind
from .backend import TrainingBackend
from .baseline_relative_smoke import BaselineRelativeSmokeService
from .domain import ExperimentContext, StageDefinition, TrainingOptions
from .experiment import ExperimentService
from .gates import QualificationGates, StagePlan
from .training_state import TrainingStateService


class PassDrivenPipeline:
    """Coordinate bounded stage execution without owning trainer/evolution internals."""

    def __init__(
        self,
        *,
        backend: TrainingBackend,
        gates: QualificationGates,
        stages: StagePlan,
        state: TrainingStateService,
        experiments: ExperimentService,
        evolution: EvolutionaryRecoveryController,
        options: TrainingOptions,
    ) -> None:
        """Compose all services required by the pass-driven stage machine.

        Purpose:
            Keep high-level stage control independent from metric math, persistence, and trainer code.
        Called by:
            TrainingApplication._build_pipeline().
        Calls:
            No project functions.
        """
        self.backend = backend
        self.gates = gates
        self.stages = stages
        self.state = state
        self.experiments = experiments
        self.evolution = evolution
        self.options = options
        self.baseline_smoke = BaselineRelativeSmokeService()

    def _invoke(
        self,
        context: ExperimentContext,
        *,
        resume: bool,
        stop_after_phase: str | None,
    ) -> dict[str, Any]:
        """Invoke the canonical trainer for one bounded stage boundary.

        Purpose:
            Remove repeated train_v9 argument wiring from stage control flow.
        Called by:
            PassDrivenPipeline._run_stage(), PassDrivenPipeline._run_final().
        Calls:
            TrainingBackend.run().
        """
        return self.backend.run(
            context.config,
            self.options.repo_root,
            self.options.device,
            resume=resume,
            early_stop_patience=self.options.early_stop_patience,
            early_stop_min_delta=self.options.early_stop_min_delta,
            stop_after_phase=stop_after_phase,
        )

    def _recover_structural_failure(
        self,
        context: ExperimentContext,
        metadata: dict[str, Any],
    ) -> bool:
        """Attempt one bounded representation recovery and reset structural state on success.

        Purpose:
            Self-adjust only representation failures while preserving fail-closed software/numerical behaviour.
        Called by:
            PassDrivenPipeline._run_stage().
        Calls:
            QualificationGates.local_geometry_metrics(), EvolutionaryRecoveryController.can_recover(),
            EvolutionaryRecoveryController.recover_after_structural_failure(),
            TrainingStateService.archive_structural_attempt().
        """
        metrics = self.gates.local_geometry_metrics(metadata)
        failure_kind = self.evolution.failure_detector.classify(metrics=metrics)
        if (
            failure_kind != FailureKind.REPRESENTATION
            or not self.evolution.can_recover(metrics)
        ):
            return False

        print(
            "[evolution] structural gate failed; breeding a bounded "
            "production-supernet generation instead of continuing downstream.",
            flush=True,
        )
        recovery = self.evolution.recover_after_structural_failure(metrics)
        if not recovery.passed:
            print(
                "[evolution] recovery generation produced no viable structural "
                "candidate; fail closed.",
                flush=True,
            )
            return False

        archive = self.state.archive_structural_attempt(
            context.directory,
            context.config,
            attempt=self.evolution.recovery_count,
        )
        print(
            f"[evolution] recovery generation passed microproof; archived "
            f"failed structural state at {archive} and restarting B1/B2 "
            "from the deterministic seed with the evolved genome.",
            flush=True,
        )
        return True

    def _complete_structural_stage(
        self,
        context: ExperimentContext,
        metadata: dict[str, Any],
    ) -> None:
        """Promote real B1/B2 success and lock the winning production genome.

        Purpose:
            Bridge qualified local geometry into downstream resume/checkpoint semantics.
        Called by:
            PassDrivenPipeline._run_stage().
        Calls:
            TrainingStateService.promote_local_geometry(),
            EvolutionaryRecoveryController.lock_production_genome(),
            QualificationGates.local_geometry_metrics().
        """
        self.state.promote_local_geometry(context.directory, context.config, metadata)
        metadata["topologyBootstrapped"] = True
        metadata["geometryQualified"] = True
        metadata["renderQualified"] = True
        locked = self.evolution.lock_production_genome(
            experiment_id=context.experiment_id,
            metrics=self.gates.local_geometry_metrics(metadata),
        )
        print(f"[evolution] production genome locked: {locked}", flush=True)

    def _run_quick_b1a_smoke(
        self,
        context: ExperimentContext,
        *,
        resume_now: bool,
    ) -> tuple[dict[str, Any], bool, int]:
        """Run/reuse one Quick B1a epoch and require identity-safe real-Raven output.

        Purpose:
            Keep topology-only B1a fail-closed without demanding improvement before
            continuous geometry and residual authority become trainable in B1b.
        Called by:
            PassDrivenPipeline._run_stage().
        Calls:
            BaselineRelativeSmokeService, PassDrivenPipeline._invoke(),
            TrainingStateService.latest_phase_validation(), TrainingStateService.snapshot(),
            ExperimentService.reject().
        """
        validation = self.state.latest_phase_validation(
            context.directory,
            context.config,
            phase="sdf-bootstrap",
        )
        latest: dict[str, Any] = {}
        current_resume = bool(resume_now)
        if not validation:
            latest = self._invoke(
                context,
                resume=current_resume,
                stop_after_phase="sdf-bootstrap",
            )
            current_resume = True
            validation = self.state.latest_phase_validation(
                context.directory,
                context.config,
                phase="sdf-bootstrap",
            )

        snapshot = self.state.snapshot(context.directory, context.config)
        metrics = self.baseline_smoke.metrics(validation)
        topology_safe = bool(snapshot.get("topologyBootstrapped", False))
        baseline_safe = self.baseline_smoke.safe_to_refine(validation, context.config)
        print(
            "[quick-smoke] real Raven B1a identity safety: "
            f"B={metrics['baselineMae']:.6f} C={metrics['candidateMae']:.6f} "
            f"gain={metrics['relativeGain']:+.2%} "
            f"wins={metrics['improvementFraction']:.1%} "
            f"regress={metrics['regressionFraction']:.1%}/"
            f"{float(context.config.maximum_validation_regression_fraction):.1%} "
            f"topology={'PASS' if topology_safe else 'FAIL'}",
            flush=True,
        )
        if topology_safe and baseline_safe:
            print(
                "[quick-smoke] PASS B1a: C preserved deterministic baseline B within "
                "the real-Raven safety budget; B1b may now earn positive authority.",
                flush=True,
            )
            return latest, current_resume, 0

        rejected = dict(latest or snapshot)
        rejected["baselineRelativeSmokeValidation"] = validation
        rejected["baselineRelativeSmokeMetrics"] = metrics
        rejected["topologyBootstrapped"] = topology_safe
        print(
            "[quick-smoke] REJECTED before B1b: topology-only B1a did not preserve "
            "baseline safety or topology did not bootstrap.", flush=True,
        )
        code = self.experiments.reject(
            context,
            phase="sdf-bootstrap-baseline-relative-safety",
            gate_label="real Raven B1a identity safety + topology bootstrap",
            metadata=rejected,
        )
        return rejected, current_resume, code


    def _run_quick_b1b_smoke(
        self, context: ExperimentContext, latest: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Require bounded Quick B1b refinement to produce strict real C > B.

        Purpose:
            Move strict baseline improvement to the first phase with continuous
            spline geometry and residual-gain training authority.
        Called by:
            PassDrivenPipeline._run_stage().
        Calls:
            BaselineRelativeSmokeService.metrics(), BaselineRelativeSmokeService.passed(),
            TrainingStateService.latest_phase_validation(), TrainingStateService.snapshot(),
            ExperimentService.reject().
        """
        validation = self.state.latest_phase_validation(
            context.directory, context.config, phase="sdf-proof"
        )
        snapshot = self.state.snapshot(context.directory, context.config)
        metrics = self.baseline_smoke.metrics(validation)
        passed = self.baseline_smoke.passed(validation, context.config)
        print(
            "[quick-smoke] real Raven B1b strict improvement: "
            f"B={metrics['baselineMae']:.6f} C={metrics['candidateMae']:.6f} "
            f"gain={metrics['relativeGain']:+.2%} "
            f"wins={metrics['improvementFraction']:.1%} "
            f"regress={metrics['regressionFraction']:.1%}/"
            f"{float(context.config.maximum_validation_regression_fraction):.1%}",
            flush=True,
        )
        if passed:
            print(
                "[quick-smoke] PASS B1b: C now beats deterministic baseline B on "
                "held-out Raven; normal structural qualification remains fail-closed.",
                flush=True,
            )
            return latest, 0
        rejected = dict(latest or snapshot)
        rejected["baselineRelativeB1bSmokeValidation"] = validation
        rejected["baselineRelativeB1bSmokeMetrics"] = metrics
        print(
            "[quick-smoke] REJECTED after B1b: refinement did not earn positive "
            "real-Raven improvement over B within the safety budget.", flush=True,
        )
        code = self.experiments.reject(
            context,
            phase="sdf-proof-baseline-relative-smoke",
            gate_label="real Raven B1b smoke: C must beat B",
            metadata=rejected,
        )
        return rejected, code

    def _run_stage(
        self,
        context: ExperimentContext,
        definition: StageDefinition,
        *,
        resume_now: bool,
    ) -> tuple[dict[str, Any], bool, int]:
        """Run one stage until its gate passes, recovers structurally, or rejects.

        Purpose:
            Isolate the only loop that may retry a structural representation stage.
        Called by:
            PassDrivenPipeline.run().
        Calls:
            PassDrivenPipeline._invoke(), PassDrivenPipeline._recover_structural_failure(),
            PassDrivenPipeline._complete_structural_stage(), ExperimentService.reject().
        """
        print("=" * 72, flush=True)
        print(f"PASS-DRIVEN STAGE       : {definition.phase}", flush=True)
        print(f"Promotion gate          : {definition.gate_label}", flush=True)
        print("Failure policy          : stop here; do not run downstream", flush=True)
        print("=" * 72, flush=True)

        current_resume = bool(resume_now)
        while True:
            if (
                definition.phase == "sdf-bootstrap"
                and self.options.training_mode == "quick"
            ):
                smoke_latest, current_resume, smoke_code = self._run_quick_b1a_smoke(
                    context,
                    resume_now=current_resume,
                )
                if smoke_code != 0:
                    return smoke_latest, current_resume, smoke_code
                latest = self._invoke(
                    context,
                    resume=True,
                    stop_after_phase="sdf-proof",
                )
                latest, b1b_smoke_code = self._run_quick_b1b_smoke(context, latest)
                if b1b_smoke_code != 0:
                    return latest, current_resume, b1b_smoke_code
            else:
                trainer_stop_phase = (
                    "sdf-proof" if definition.phase == "sdf-bootstrap" else definition.phase
                )
                latest = self._invoke(
                    context,
                    resume=current_resume,
                    stop_after_phase=trainer_stop_phase,
                )
            current_resume = True

            if definition.gate(latest, context.config):
                if definition.phase == "sdf-bootstrap":
                    self._complete_structural_stage(context, latest)
                print(
                    f"[pipeline] PASS {definition.phase}: {definition.gate_label}",
                    flush=True,
                )
                return latest, current_resume, 0

            if definition.phase == "sdf-bootstrap" and self._recover_structural_failure(
                context,
                latest,
            ):
                current_resume = False
                continue

            print(
                f"[pipeline] REJECTED at {definition.phase}: {definition.gate_label} "
                "did not qualify within its bounded production budget. "
                "Downstream stages were not run.",
                flush=True,
            )
            code = self.experiments.reject(
                context,
                phase=definition.phase,
                gate_label=definition.gate_label,
                metadata=latest,
            )
            return latest, current_resume, code

    def _run_final(self, context: ExperimentContext) -> tuple[dict[str, Any], int]:
        """Run physical fine-tuning and require the real production-final selection.

        Purpose:
            Complete the pass-driven curriculum with unchanged final safety requirements.
        Called by:
            PassDrivenPipeline.run().
        Calls:
            PassDrivenPipeline._invoke(), ExperimentService.reject().
        """
        print("=" * 72, flush=True)
        print("PASS-DRIVEN FINAL STAGE : physical-finetune / BenefitSelector", flush=True)
        print("Promotion gate          : production-final + full final qualification", flush=True)
        print("=" * 72, flush=True)

        latest = self._invoke(context, resume=True, stop_after_phase=None)
        final_pass = bool(latest.get("trainingSafetyPass", False)) and str(
            latest.get("selectionKind") or ""
        ) == "production-final"
        if not final_pass:
            code = self.experiments.reject(
                context,
                phase="physical-finetune",
                gate_label="production final selector + strict training safety",
                metadata=latest,
            )
            return latest, code

        print("[pipeline] PASS physical-finetune: production-final selected.", flush=True)
        return latest, 0

    def run(self, context: ExperimentContext) -> tuple[dict[str, Any] | None, int]:
        """Run the complete gated canonical pipeline from persisted state to final selector.

        Purpose:
            Present TrainingApplication with one readable production-stage workflow.
        Called by:
            TrainingApplication._run_training().
        Calls:
            TrainingStateService.snapshot(), StagePlan.already_qualified(),
            PassDrivenPipeline._run_stage(), PassDrivenPipeline._run_final().
        """
        resume_now = bool(context.resume)
        snapshot = self.state.snapshot(context.directory, context.config)
        latest: dict[str, Any] | None = None

        for definition in self.stages.definitions:
            if self.stages.already_qualified(
                definition.phase,
                snapshot,
                context.config,
            ):
                print(
                    f"[pipeline] {definition.phase}: already qualified in persisted state; "
                    "skipping replay.",
                    flush=True,
                )
                continue

            latest, resume_now, code = self._run_stage(
                context,
                definition,
                resume_now=resume_now,
            )
            if code != 0:
                return latest, code
            snapshot.update(latest)

        return self._run_final(context)
