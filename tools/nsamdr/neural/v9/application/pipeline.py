"""SR-first Quick production training pipeline."""
from __future__ import annotations

import json
from typing import Any

from .backend import TrainingBackend
from .domain import ExperimentContext, TrainingOptions
from .experiment import ExperimentService


class PassDrivenPipeline:
    """Run only the active SR candidate and final BenefitSelector stages."""

    def __init__(
        self,
        *,
        backend: TrainingBackend,
        experiments: ExperimentService,
        options: TrainingOptions,
    ) -> None:
        """Compose the active production-stage collaborators.

        Purpose:
            Keep stage control independent from trainer internals and experiment persistence.
        Called by:
            TrainingApplication._build_pipeline().
        Calls:
            No project functions.
        """
        self.backend = backend
        self.experiments = experiments
        self.options = options

    def _invoke(
        self,
        context: ExperimentContext,
        *,
        resume: bool,
        stop_after_phase: str | None,
    ) -> dict[str, Any]:
        """Invoke the canonical trainer for one active SR-first stage boundary.

        Purpose:
            Centralise train_v9 argument wiring for detail and final-selector stages.
        Called by:
            PassDrivenPipeline._run_detail(), PassDrivenPipeline._run_final().
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

    def _persisted_metadata(self, context: ExperimentContext) -> dict[str, Any]:
        """Read current trainer metadata when resuming an already-qualified SR stage.

        Purpose:
            Avoid replaying detail-reconstruction after an interrupted run reached its gate.
        Called by:
            PassDrivenPipeline.run().
        Calls:
            json.loads().
        """
        path = context.directory / context.config.metadata_name
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _run_detail(
        self,
        context: ExperimentContext,
        *,
        resume_now: bool,
    ) -> tuple[dict[str, Any], int]:
        """Train the V13 multi-map SR candidate and require detail qualification.

        Purpose:
            Make C the only learned candidate stage before final BenefitSelector training.
        Called by:
            PassDrivenPipeline.run().
        Calls:
            PassDrivenPipeline._invoke(), ExperimentService.reject().
        """
        print("=" * 72, flush=True)
        print("PASS-DRIVEN STAGE       : detail-reconstruction / SR candidate C", flush=True)
        print("Promotion gate          : detailQualified", flush=True)
        print("=" * 72, flush=True)

        latest = self._invoke(
            context,
            resume=resume_now,
            stop_after_phase="detail-reconstruction",
        )
        if bool(latest.get("detailQualified", False)):
            print("[pipeline] PASS detail-reconstruction: SR candidate qualified.", flush=True)
            return latest, 0

        code = self.experiments.reject(
            context,
            phase="detail-reconstruction",
            gate_label="V13 SR candidate qualification",
            metadata=latest,
        )
        return latest, code

    def _run_final(self, context: ExperimentContext) -> tuple[dict[str, Any], int]:
        """Train BenefitSelector and require representative production-final promotion.

        Purpose:
            Complete the active B -> C -> F curriculum and fail closed on final safety.
        Called by:
            PassDrivenPipeline.run().
        Calls:
            PassDrivenPipeline._invoke(), ExperimentService.reject().
        """
        print("=" * 72, flush=True)
        print("PASS-DRIVEN FINAL STAGE : physical-finetune / BenefitSelector F", flush=True)
        print("Promotion gate          : production-final + representative Raven qualification", flush=True)
        print("=" * 72, flush=True)

        latest = self._invoke(context, resume=True, stop_after_phase=None)
        final_pass = bool(latest.get("trainingSafetyPass", False)) and str(
            latest.get("selectionKind") or ""
        ) == "production-final"
        if final_pass:
            print("[pipeline] PASS BenefitSelector: production-final selected.", flush=True)
            return latest, 0

        code = self.experiments.reject(
            context,
            phase="physical-finetune",
            gate_label="V13 BenefitSelector + representative Raven qualification",
            metadata=latest,
        )
        return latest, code

    def run(self, context: ExperimentContext) -> tuple[dict[str, Any] | None, int]:
        """Run the complete current SR-first Quick pipeline.

        Purpose:
            Execute only detail-reconstruction followed by final selector qualification.
        Called by:
            TrainingApplication._run_training().
        Calls:
            PassDrivenPipeline._persisted_metadata(), _run_detail(), _run_final().
        """
        persisted = self._persisted_metadata(context)
        if bool(persisted.get("detailQualified", False)):
            print(
                "[pipeline] detail-reconstruction already qualified; skipping replay.",
                flush=True,
            )
        else:
            latest, code = self._run_detail(
                context,
                resume_now=bool(context.resume),
            )
            if code != 0:
                return latest, code

        return self._run_final(context)
