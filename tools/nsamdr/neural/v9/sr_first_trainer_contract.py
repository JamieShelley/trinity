from __future__ import annotations

"""V13.3 trainer ownership for the SR-only Quick workflow.

Historical geometry/profile/seam modules remain in the checkpoint state for strict
load compatibility. They are not trainable, are not included in active participation
hooks, and their historical startup microproofs are bypassed. The only trainable
phases are DetailNet SR reconstruction and BenefitSelector authority.
"""

from typing import Any, Mapping

import torch

from .model import FidelityResidualNetV9, MODEL_SCHEMA


SR_TRAINER_REVISION = "V13.3"
_INSTALLED = False
_PREVIOUS_BACKEND_SYNC: Any = None
_PREVIOUS_SET_PHASE: Any = None

_ACTIVE_COMPONENT_PATHS: dict[str, str] = {
    "conditioned detail": "detail_net",
    "albedo physical head": "detail_net.albedo_head",
    "normal physical head": "detail_net.normal_head",
    "material physical head": "detail_net.material_head",
    "confidence": "detail_net.confidence_head",
    "regret": "detail_net.regret_head",
    "BenefitSelector": "benefit_selector",
}

_RETIRED_PHASES = {
    "sdf-bootstrap",
    "sdf-proof",
    "seam-proof",
    "seam-authority",
    "gate-proof",
}


def _module_at(model: Any, path: str) -> Any:
    value = model
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _active_component_modules(
    model: FidelityResidualNetV9,
) -> dict[str, tuple[str, torch.nn.Module]]:
    """Return only modules that can execute in the V13.3 production graph."""
    result: dict[str, tuple[str, torch.nn.Module]] = {}
    for label, path in _ACTIVE_COMPONENT_PATHS.items():
        module = _module_at(model, path)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError(f"V13.3 active component missing: {label} -> {path}")
        result[label] = (path, module)
    return result


def _validate_v133_architecture_contract(contract: Mapping[str, object]) -> None:
    """Reject any trainer model that does not expose the SR-only authority contract."""
    active = tuple(contract.get("productionComponentsActive", ()))
    retired = tuple(contract.get("productionComponentsRetired", ()))
    required_active = {
        "detail_net",
        "detail_net.albedo_head",
        "detail_net.normal_head",
        "detail_net.material_head",
        "detail_net.confidence_head",
        "detail_net.regret_head",
        "benefit_selector",
    }
    required_retired = {
        "geometry_net",
        "boundary_renderer",
        "boundary_specialist",
        "seam_restorer",
    }
    failures: list[str] = []
    if contract.get("schema") != MODEL_SCHEMA:
        failures.append("model schema")
    if contract.get("srFirstRevision") != SR_TRAINER_REVISION:
        failures.append("srFirstRevision")
    if contract.get("productionForward") != "direct B -> DetailNet C -> BenefitSelector F":
        failures.append("productionForward")
    if not required_active.issubset(set(active)):
        failures.append("active component inventory")
    if not required_retired.issubset(set(retired)):
        failures.append("retired component inventory")
    if contract.get("retiredComponentsExecuted") is not False:
        failures.append("retiredComponentsExecuted")
    if contract.get("geometryPixelAuthority") is not False:
        failures.append("geometryPixelAuthority")
    if contract.get("profilePixelAuthority") is not False:
        failures.append("profilePixelAuthority")
    if contract.get("seamPixelAuthority") is not False:
        failures.append("seamPixelAuthority")
    if abs(float(contract.get("srDetailBodyLearningRate", 0.0)) - 1.0e-3) > 1.0e-12:
        failures.append("detail body learning rate")
    if abs(float(contract.get("srDetailAlbedoHeadLearningRate", 0.0)) - 3.0e-3) > 1.0e-12:
        failures.append("albedo head learning rate")
    if failures:
        raise RuntimeError(
            "V13.3 SR-only architecture contract failed: " + ", ".join(failures)
        )


def _set_phase_v133(self: FidelityResidualNetV9, phase: str) -> None:
    """Keep checkpoint-only modules frozen in every supported V13.3 Quick phase."""
    if phase in _RETIRED_PHASES:
        raise RuntimeError(f"V13.3 retired training phase requested: {phase}")
    if phase not in {"detail-reconstruction", "physical-finetune", "boundary-hardening"}:
        raise RuntimeError(f"V13.3 unsupported training phase: {phase}")

    if _PREVIOUS_SET_PHASE is not None:
        _PREVIOUS_SET_PHASE(self, phase)

    for parameter in self.parameters():
        parameter.requires_grad_(False)
    if phase == "detail-reconstruction":
        for parameter in self.detail_net.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in self.benefit_selector.parameters():
            parameter.requires_grad_(True)


def _retired_structure_microproof(_device: Any, _config: Any) -> tuple[float, float, float, float]:
    return 0.0, 0.0, 0.0, 0.0


def _retired_profile_microproof(_device: Any) -> tuple[float, float]:
    return 0.0, 0.0


def _retired_seam_microproof(_device: Any, _config: Any) -> tuple[float, float, float]:
    return 0.0, 0.0, 1.0


def _synchronize_v133_trainer(self: Any, training: Any) -> None:
    if _PREVIOUS_BACKEND_SYNC is None:
        raise RuntimeError("V13.3 trainer contract missing backend synchronizer")
    _PREVIOUS_BACKEND_SYNC(self, training)
    service = getattr(training, "_training_service", None)
    if service is None:
        raise RuntimeError("NSAMDR training module has no TrainingService singleton")

    # These are module-level functions stored directly on the service, matching the
    # existing Windows-spawn-safe compatibility adapters.
    training._validate_v992_architecture_contract = _validate_v133_architecture_contract
    training._production_component_modules = _active_component_modules
    training._explicit_primitive_structure_microproof = _retired_structure_microproof

    service._validate_v992_architecture_contract = _validate_v133_architecture_contract
    service._production_component_modules = _active_component_modules
    service._explicit_primitive_structure_microproof = _retired_structure_microproof
    service._profile_specialist_microproof = _retired_profile_microproof
    service._phase_seam_sr_microproof = _retired_seam_microproof

    training._nsamdr_sr_trainer_revision = SR_TRAINER_REVISION


def install_sr_first_trainer_contract() -> None:
    global _INSTALLED, _PREVIOUS_BACKEND_SYNC, _PREVIOUS_SET_PHASE
    if _INSTALLED:
        return

    from .application.backend import TrainingBackend

    _PREVIOUS_SET_PHASE = FidelityResidualNetV9.set_phase
    FidelityResidualNetV9.set_phase = _set_phase_v133

    _PREVIOUS_BACKEND_SYNC = TrainingBackend._synchronize_training_service_contract
    TrainingBackend._synchronize_training_service_contract = _synchronize_v133_trainer
    _INSTALLED = True
