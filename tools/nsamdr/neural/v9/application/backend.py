"""Adapter around the canonical v9.training backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import V9Config


class TrainingBackend:
    """Own installation and invocation of the current production trainer contract."""

    def __init__(self) -> None:
        """Install the local-boundary contract once and retain canonical train_v9.

        Purpose:
            Isolate import-time trainer patching from CLI/application orchestration.
        Called by:
            TrainingApplication._build_pipeline() and diagnostic stage execution.
        Calls:
            install_local_boundary_training_contract().
        """
        import v9.training as training
        from ..local_boundary_production_contract import install_local_boundary_training_contract

        install_local_boundary_training_contract(training)
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
