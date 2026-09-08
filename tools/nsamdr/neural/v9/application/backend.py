"""Adapter around the canonical v9.training backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import V9Config


def _run_final_qualification_with_runtime_reset(
    model: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run canonical final qualification from fresh production runtime state.

    This wrapper is deliberately module-level. Windows DataLoader workers pickle
    the bound TrainingService worker initializer, which serializes the service
    instance. A nested/local function stored on that instance makes spawn fail
    before the first training batch.
    """
    TrainingBackend._prepare_production_runtime(model)
    import v9.training as training

    service = training._training_service
    canonical = type(service)._run_final_qualification
    return canonical(service, model, *args, **kwargs)


class TrainingBackend:
    """Own installation and invocation of the current production trainer contract."""

    @staticmethod
    def _prepare_production_runtime(model: Any) -> None:
        """Clear training-only forward state before production qualification.

        B1a intentionally sets ``_topology_bootstrap_only`` so its forward proves
        topology without invoking the explicit continuous geometry refiner. That
        flag is runtime-only and is absent from the state dict; a freshly loaded
        production model therefore starts with it disabled. Final qualification
        reuses the training model object, so it must reproduce that fresh-load
        runtime state before auditing production-component participation.
        """
        geometry = getattr(model, "geometry_net", None)
        structure = getattr(geometry, "production_structure", None)
        if structure is None:
            return
        if hasattr(structure, "_topology_bootstrap_only"):
            structure._topology_bootstrap_only = False

    def _synchronize_training_service_contract(self, training: Any) -> None:
        """Copy patched compatibility callbacks onto the object that owns train_v9.

        Purpose:
            Ensure the OOP TrainingService executes the installed V11.4 validator,
            structural microproof, and production-component map rather than stale
            pre-refactor methods reached through self.* calls. Final production
            qualification must also clear training-only runtime phase state without
            installing an unpicklable local closure on the service singleton.
        Called by:
            TrainingBackend.__init__().
        Calls:
            getattr(), _run_final_qualification_with_runtime_reset().
        """
        service = getattr(training, "_training_service", None)
        if service is None:
            raise RuntimeError("NSAMDR training module has no TrainingService singleton")
        service._validate_v992_architecture_contract = training._validate_v992_architecture_contract
        service._explicit_primitive_structure_microproof = training._explicit_primitive_structure_microproof
        service._production_component_modules = training._production_component_modules

        if service._run_final_qualification is not _run_final_qualification_with_runtime_reset:
            service._run_final_qualification = _run_final_qualification_with_runtime_reset

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
