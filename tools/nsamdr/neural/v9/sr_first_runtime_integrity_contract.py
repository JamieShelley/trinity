from __future__ import annotations

"""V13.3 runtime-integrity audit for the active SR production graph.

The legacy trainer's final check required every historical geometry/profile/seam
component to execute. V13.3 deliberately retires those modules from production, so
runtime integrity instead requires the active B -> DetailNet C -> BenefitSelector F
graph and explicitly requires retired modules to remain bypassed.
"""

import hashlib
from typing import Any, Mapping

import torch

from .model import FidelityResidualNetV9, model_hash


SR_RUNTIME_INTEGRITY_REVISION = "V13.3"
_INSTALLED = False
_PREVIOUS_BACKEND_SYNC: Any = None

_ACTIVE_PATHS: dict[str, str] = {
    "conditioned detail": "detail_net",
    "albedo physical head": "detail_net.albedo_head",
    "normal physical head": "detail_net.normal_head",
    "material physical head": "detail_net.material_head",
    "confidence": "detail_net.confidence_head",
    "regret": "detail_net.regret_head",
    "BenefitSelector": "benefit_selector",
}

_RETIRED_PATHS: dict[str, str] = {
    "geometry": "geometry_net",
    "structural representation": "geometry_net.production_structure",
    "explicit geometry refiner": "geometry_net.production_structure.geometry_refiner",
    "boundary renderer": "boundary_renderer",
    "boundary/profile": "boundary_specialist",
    "PhaseAwareSeamSR": "seam_restorer.phase_sr",
    "seam authority": "seam_restorer.authority",
}

_REQUIRED_OUTPUTS = (
    "albedo",
    "normal_xy",
    "material",
    "detail_candidate_albedo",
    "detail_candidate_normal",
    "detail_candidate_material",
    "detail_confidence",
    "detail_regret",
    "benefit_selector_probability",
    "final_selector_gate",
)


def _module_at(model: Any, path: str) -> Any:
    value = model
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _component_map(model: FidelityResidualNetV9) -> dict[str, tuple[str, torch.nn.Module]]:
    result: dict[str, tuple[str, torch.nn.Module]] = {}
    for label, path in {**_RETIRED_PATHS, **_ACTIVE_PATHS}.items():
        module = _module_at(model, path)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError(f"V13.3 runtime-integrity component is missing: {label} -> {path}")
        result[label] = (path, module)
    return result


def _run_v133_runtime_integrity(
    self: Any,
    model: FidelityResidualNetV9,
    loader: Any,
    config: Any,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    strict_missing_keys: list[str],
    strict_unexpected_keys: list[str],
) -> dict[str, Any]:
    """Strict-reload audit of only the V13.3 deployed SR graph."""
    model.eval()
    raw_batch = next(iter(loader))
    batch = self._move_batch(
        raw_batch,
        device,
        channels_last=bool(config.channels_last and device.type == "cuda"),
    )

    components = _component_map(model)
    forward_counts, handles = self._register_component_forward_hooks(components)
    try:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            outputs = model(batch["input"])
    finally:
        for handle in handles:
            handle.remove()

    if not isinstance(outputs, Mapping):
        raise RuntimeError("V13.3 production runtime returned a non-mapping output")

    missing_or_nonfinite = [
        key
        for key in _REQUIRED_OUTPUTS
        if key not in outputs
        or not torch.is_tensor(outputs[key])
        or not bool(torch.isfinite(outputs[key]).all().item())
    ]
    missed_active = [
        label for label in _ACTIVE_PATHS if int(forward_counts.get(label, 0)) <= 0
    ]
    executed_retired = [
        label for label in _RETIRED_PATHS if int(forward_counts.get(label, 0)) > 0
    ]
    if (
        missing_or_nonfinite
        or missed_active
        or executed_retired
        or strict_missing_keys
        or strict_unexpected_keys
    ):
        raise RuntimeError(
            "V13.3 production runtime integrity failed: "
            f"nonfiniteOrMissingOutputs={missing_or_nonfinite}, "
            f"missedActiveComponents={missed_active}, "
            f"executedRetiredComponents={executed_retired}, "
            f"missingState={strict_missing_keys}, unexpectedState={strict_unexpected_keys}"
        )

    output_digest = hashlib.sha256()
    output_metrics: dict[str, dict[str, float]] = {}
    for key in _REQUIRED_OUTPUTS:
        value = outputs[key].detach().float().cpu().contiguous()
        output_digest.update(key.encode("utf-8"))
        output_digest.update(value.numpy().tobytes())
        output_metrics[key] = {
            "mean": float(value.mean().item()),
            "min": float(value.amin().item()),
            "max": float(value.amax().item()),
        }

    return {
        "passed": True,
        "kind": "production-runtime-integrity",
        "revision": SR_RUNTIME_INTEGRITY_REVISION,
        "strictReload": True,
        "strictMissingKeys": list(strict_missing_keys),
        "strictUnexpectedKeys": list(strict_unexpected_keys),
        "modelClass": type(model).__name__,
        "productionForward": "B -> DetailNet C -> BenefitSelector F",
        "modelEval": True,
        "cacheUsed": False,
        "overridesUsed": False,
        "inputShape": list(batch["input"].shape),
        "activeComponents": list(_ACTIVE_PATHS),
        "retiredComponents": list(_RETIRED_PATHS),
        "retiredComponentsBypassed": True,
        "componentForwardCalls": {
            key: int(value) for key, value in forward_counts.items()
        },
        "requiredOutputs": list(_REQUIRED_OUTPUTS),
        "outputMetrics": output_metrics,
        "outputSha256": output_digest.hexdigest(),
        "candidateModelSha256": model_hash(model),
    }


def _synchronize_runtime_integrity(self: Any, training: Any) -> None:
    if _PREVIOUS_BACKEND_SYNC is None:
        raise RuntimeError("V13.3 runtime-integrity contract missing backend synchronizer")
    _PREVIOUS_BACKEND_SYNC(self, training)
    service = getattr(training, "_training_service", None)
    if service is None:
        raise RuntimeError("NSAMDR training module has no TrainingService singleton")
    service._run_final_qualification = _run_v133_runtime_integrity.__get__(
        service, type(service)
    )
    training._nsamdr_runtime_integrity_revision = SR_RUNTIME_INTEGRITY_REVISION


def install_sr_first_runtime_integrity_contract() -> None:
    global _INSTALLED, _PREVIOUS_BACKEND_SYNC
    if _INSTALLED:
        return

    from .application.backend import TrainingBackend

    _PREVIOUS_BACKEND_SYNC = TrainingBackend._synchronize_training_service_contract
    TrainingBackend._synchronize_training_service_contract = _synchronize_runtime_integrity
    _INSTALLED = True
