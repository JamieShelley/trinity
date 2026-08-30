"""Adapter around the canonical v9.training backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import V9Config


class TrainingBackend:
    """Own installation and invocation of the current production trainer contract."""

    def _synchronize_training_service_contract(self, training: Any) -> None:
        """Copy patched compatibility callbacks onto the object that owns train_v9.

        Purpose:
            Ensure the OOP TrainingService executes the installed V11.4 validator,
            structural microproof, and production-component map rather than stale
            pre-refactor methods reached through self.* calls.
        Called by:
            TrainingBackend.__init__().
        Calls:
            getattr().
        """
        service = getattr(training, "_training_service", None)
        if service is None:
            raise RuntimeError("NSAMDR training module has no TrainingService singleton")
        service._validate_v992_architecture_contract = training._validate_v992_architecture_contract
        service._explicit_primitive_structure_microproof = training._explicit_primitive_structure_microproof
        service._production_component_modules = training._production_component_modules

    def __init__(self) -> None:
        """Install the local-boundary contract once and retain canonical train_v9.

        Purpose:
            Isolate import-time trainer patching from CLI/application orchestration.
        Called by:
            TrainingApplication._build_pipeline() and diagnostic stage execution.
        Calls:
            install_local_boundary_training_contract(),
            TrainingBackend._synchronize_training_service_contract().
        """
        import v9.training as training
        from ..local_boundary_production_contract import install_local_boundary_training_contract

        install_local_boundary_training_contract(training)
        self._synchronize_training_service_contract(training)
        self._trainer = training.train_v9

    def run(
        self,
        config: V9Config,
        repo_root: Path,
        device: str,
        *,
        resume: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
        stop_after_phase: str | None,
    ) -> dict[str, Any]:
        """Invoke the unchanged canonical trainer with one explicit stage boundary.

        Purpose:
            Give application orchestration one narrow dependency on the training implementation.
        Called by:
            PassDrivenPipeline and TrainingApplication diagnostic mode.
        Calls:
            v9.training.train_v9().
        """
        return self._trainer(
            config,
            repo_root,
            device,
            resume=resume,
            restart=False,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            stop_after_phase=stop_after_phase,
        )
