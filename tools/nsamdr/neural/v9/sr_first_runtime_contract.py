from __future__ import annotations

"""V13.3 active SR runtime.

This is the final owner of the Stage-4 production path:

    B = deterministic 4x reconstruction
    C = B + bounded multi-map SR residual
    F = B + BenefitSelector * (C - B)

The historical geometry/profile/seam stack remains checkpoint-loadable but is retired
from the active forward, loss and learning-rate paths.  V13.3 also removes the
stacked V12 preservation wrappers and owns one explicit protected-B policy.
"""

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9
from . import parallel_detail_contract as direct_detail
from . import baseline_relative_specialist_contract as preservation
from . import sr_first_contract as sr


SR_RUNTIME_REVISION = "V13.3"
SR_DETAIL_BODY_LR = 1.0e-3
SR_DETAIL_ALBEDO_HEAD_LR = 3.0e-3
SR_PROTECTED_DETAIL_WEIGHT = 24.0
SR_PROTECTED_SELECTOR_GATE_WEIGHT = 64.0
SR_PROTECTED_SELECTOR_WEIGHT_BOOST = 8.0

_INSTALLED = False
_PREVIOUS_FORWARD_IMPL: Any = None
_PREVIOUS_ARCHITECTURE_CONTRACT: Any = None
_PREVIOUS_BACKEND_LOSS: Any = None
_PREVIOUS_BACKEND_SYNC: Any = None
_PREVIOUS_BACKEND_RUN: Any = None
_PREVIOUS_PHASE_LR: Any = None


def _zero_maps(
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros1 = torch.zeros(
        (baseline.shape[0], 1, baseline.shape[-2], baseline.shape[-1]),
        device=baseline.device,
        dtype=baseline.dtype,
    )
    zeros2 = torch.zeros(
        (baseline.shape[0], 2, baseline.shape[-2], baseline.shape[-1]),
        device=baseline.device,
        dtype=baseline.dtype,
    )
    return zeros1, zeros2


def _active_sr_forward(
    self: FidelityResidualNetV9,
    inputs: torch.Tensor,
    *,
    sdf_override: torch.Tensor | None = None,
    gate_override: torch.Tensor | None = None,
    hardness_override: torch.Tensor | None = None,
    seam_authority_override: torch.Tensor | None = None,
    seam_tangent_override: torch.Tensor | None = None,
    phase_only_seam_teacher: bool = False,
) -> dict[str, torch.Tensor]:
    """Execute only B -> C -> BenefitSelector F.

    Legacy override arguments are rejected because the retired geometry/seam graph
    no longer has production authority in V13.3.
    """
    if any(
        value is not None
        for value in (
            sdf_override,
            gate_override,
            hardness_override,
            seam_authority_override,
            seam_tangent_override,
        )
    ) or phase_only_seam_teacher:
        raise RuntimeError(
            "V13.3 SR runtime does not accept retired geometry/seam production overrides"
        )

    baseline_albedo, baseline_normal, baseline_material = direct_detail._deterministic_baseline(
        inputs
    )
    condition = sr._observable_sr_condition(inputs, baseline_albedo.shape[-2:])

    # GeometryConditionedDetailNet.forward is already the V13 direct-detail adapter.
    # Calling it directly avoids GeometryNet, BoundaryRenderer, profile and seam work.
    detail = self._run_training_component(
        self.detail_net,
        inputs,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        condition,
    )

    albedo_cap = float(getattr(self.detail_net, "_parallel_albedo_cap", 0.20))
    normal_cap = float(getattr(self.detail_net, "_parallel_normal_cap", 0.15))
    material_cap = float(getattr(self.detail_net, "_parallel_material_cap", 0.18))

    candidate_albedo = (
        baseline_albedo.float() + detail["albedo_raw"].float() * albedo_cap
    ).clamp(0.0, 1.0)
    candidate_normal = FidelityResidualNetV9._normalize_xy(
        (
            baseline_normal.float() + detail["normal_raw"].float() * normal_cap
        ).to(baseline_normal.dtype)
    )
    candidate_material = (
        baseline_material.float() + detail["material_raw"].float() * material_cap
    ).clamp(0.0, 1.0)

    confidence_logits = detail["confidence_logits"].float()
    regret_logits = detail["regret_logits"].float()
    detail_confidence = torch.sigmoid(confidence_logits)
    detail_regret = torch.sigmoid(regret_logits)
    detail_support = (detail_confidence * (1.0 - detail_regret)).clamp(0.0, 1.0)

    observed_support = self._source_edge_support(
        inputs, int(self.config.geometry_edge_support_radius)
    )
    observed_support = F.interpolate(
        observed_support,
        size=candidate_albedo.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    zeros1, zeros2 = _zero_maps(candidate_albedo)

    selector_features = self._selector_features(
        baseline_albedo,
        candidate_albedo,
        zeros1,
        zeros2,
        zeros1,
        zeros1,
        observed_support,
        zeros1,
        detail_confidence,
        detail_regret,
    )
    selector_logits = self._run_training_component(
        self.benefit_selector,
        selector_features,
    )
    selector_probability = torch.sigmoid(selector_logits.float())
    final_gate = selector_probability

    albedo = (
        baseline_albedo.float() * (1.0 - final_gate)
        + candidate_albedo.float() * final_gate
    ).clamp(0.0, 1.0).to(baseline_albedo.dtype)
    normal = FidelityResidualNetV9._normalize_xy(
        (
            baseline_normal.float() * (1.0 - final_gate)
            + candidate_normal.float() * final_gate
        ).to(baseline_normal.dtype)
    )
    material = (
        baseline_material.float() * (1.0 - final_gate)
        + candidate_material.float() * final_gate
    ).clamp(0.0, 1.0).to(baseline_material.dtype)

    class_centres = torch.linspace(
        0.0,
        1.0,
        self.config.material_classes,
        device=inputs.device,
        dtype=material.dtype,
    )
    material_logits = -(
        (material[:, 0:1] - class_centres.view(1, -1, 1, 1)) ** 2
    ) * 40.0
    confidence = selector_probability.clamp(1.0e-5, 1.0 - 1.0e-5)

    # Retired structural/seam compatibility outputs are explicit identity/B values,
    # not results from executing the old neural redraw stack.
    edge_probability = observed_support.clamp(1.0e-5, 1.0 - 1.0e-5)
    edge_logits = torch.logit(edge_probability)

    return {
        "baseline_albedo": baseline_albedo,
        "baseline_normal": baseline_normal,
        "baseline_material": baseline_material,
        "boundary_pre_seam_albedo": baseline_albedo,
        "boundary_pre_seam_normal": baseline_normal,
        "boundary_pre_seam_material": baseline_material,
        "boundary_reconstructed_albedo": baseline_albedo,
        "boundary_reconstructed_normal": baseline_normal,
        "boundary_reconstructed_material": baseline_material,
        "detail_candidate_albedo": candidate_albedo.to(baseline_albedo.dtype),
        "detail_candidate_normal": candidate_normal,
        "detail_candidate_material": candidate_material.to(baseline_material.dtype),
        "sr_candidate_albedo": candidate_albedo.to(baseline_albedo.dtype),
        "sr_candidate_normal": candidate_normal,
        "sr_candidate_material": candidate_material.to(baseline_material.dtype),
        "fused_candidate_albedo": candidate_albedo.to(baseline_albedo.dtype),
        "fused_candidate_normal": candidate_normal,
        "fused_candidate_material": candidate_material.to(baseline_material.dtype),
        "detail_confidence_logits": confidence_logits.to(baseline_albedo.dtype),
        "detail_regret_logits": regret_logits.to(baseline_albedo.dtype),
        "detail_confidence": detail_confidence.to(baseline_albedo.dtype),
        "detail_regret": detail_regret.to(baseline_albedo.dtype),
        "sr_observable_condition": condition.to(baseline_albedo.dtype),
        "observed_source_edge_support": observed_support.to(baseline_albedo.dtype),
        "parallel_structure_fusion_weight": zeros1,
        "parallel_seam_fusion_weight": zeros1,
        "parallel_detail_fusion_weight": torch.ones_like(zeros1),
        "parallel_fusion_support": detail_support.to(baseline_albedo.dtype),
        "parallel_fusion_conflict": zeros1,
        "albedo": albedo,
        "normal_xy": normal,
        "material": material,
        "emissive": material[:, 1:2],
        "roughness": material[:, 2:3],
        "material_logits": material_logits,
        "confidence": confidence.to(albedo.dtype),
        "confidence_logits": torch.logit(confidence).to(albedo.dtype),
        "benefit_selector_logits": selector_logits.to(albedo.dtype),
        "benefit_selector_probability": selector_probability.to(albedo.dtype),
        "final_selector_gate": final_gate.to(albedo.dtype),
        "boundary_gate_logits": zeros1,
        "boundary_gate_prediction": zeros1,
        "boundary_gate_probability": zeros1,
        "boundary_gate": zeros1,
        "sdf": zeros1,
        "coarse_sdf": zeros1,
        "orientation": zeros2,
        "boundary_normal": zeros2,
        "hardness": zeros1,
        "edge_logits": edge_logits.to(albedo.dtype),
        "contour_normal_offset_pixels": zeros1,
        "sdf_residual_pixels": zeros1,
    }


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if w.shape[1] == 1 and value.shape[1] != 1:
        w = w.expand(-1, value.shape[1], -1, -1)
    return (value.float() * w.float()).sum() / w.float().sum().clamp_min(1.0)


def _gradient_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from .contours import sobel_tensor

    pgx, pgy = sobel_tensor(prediction.float().mean(dim=1, keepdim=True))
    tgx, tgy = sobel_tensor(target.float().mean(dim=1, keepdim=True))
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), weight)


def _detail_support_calibration(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["sr_candidate_albedo"].detach().float()
    baseline_error = (baseline - target.detach().float()).abs().mean(dim=1, keepdim=True)
    candidate_error = (candidate - target.detach().float()).abs().mean(dim=1, keepdim=True)
    improvement = (baseline_error - candidate_error).detach()
    scale = max(float(getattr(config, "gate_error_scale", 0.08)), 1.0e-4)
    confidence_target = (0.5 + improvement / scale).clamp(0.0, 1.0)
    regret_target = (candidate_error > baseline_error + 0.001).float().detach()
    confidence_loss = F.binary_cross_entropy_with_logits(
        outputs["detail_confidence_logits"].float(), confidence_target
    )
    regret_loss = F.binary_cross_entropy_with_logits(
        outputs["detail_regret_logits"].float(), regret_target
    )
    total = (
        confidence_loss * float(getattr(config, "detail_confidence_weight", 0.20))
        + regret_loss * float(getattr(config, "detail_regret_classifier_weight", 0.40))
    )
    return total, confidence_loss, regret_loss


def _detail_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    from .contours import sobel_tensor

    target_albedo = batch["target_albedo"].float()
    target_normal = batch["target_normal"].float()
    target_material = sr._target_material(batch, config)
    baseline_albedo = outputs["baseline_albedo"].detach().float()
    candidate_albedo = outputs["sr_candidate_albedo"].float()
    candidate_normal = outputs["sr_candidate_normal"].float()
    candidate_material = outputs["sr_candidate_material"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    edge_weight = (0.20 + edge * 3.80).detach()

    albedo_error = (candidate_albedo - target_albedo).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline_albedo - target_albedo).abs().mean(dim=1, keepdim=True)
    albedo_global = albedo_error.mean()
    albedo_edge = _weighted_mean(albedo_error, edge_weight)
    regret = _weighted_mean(F.relu(albedo_error - baseline_error), edge_weight)

    candidate_gray = candidate_albedo.mean(dim=1, keepdim=True)
    target_gray = target_albedo.mean(dim=1, keepdim=True)
    cgx, cgy = sobel_tensor(candidate_gray)
    tgx, tgy = sobel_tensor(target_gray)
    gradient = _weighted_mean((cgx - tgx).abs() + (cgy - tgy).abs(), edge_weight)
    laplacian = _weighted_mean(
        (sr._laplacian(candidate_gray) - sr._laplacian(target_gray)).abs(),
        edge_weight,
    )
    pyramid = sr._pyramid_l1(candidate_albedo, target_albedo)
    target_scale = max(1, int(getattr(config, "target_scale", 4)))
    grid_excess = sr._grid_excess(candidate_albedo, target_albedo, target_scale)

    cap = float(getattr(config, "detail_albedo_max_delta", 0.20))
    desired_residual = (target_albedo.detach() - baseline_albedo).clamp(-cap, cap)
    actual_residual = candidate_albedo - baseline_albedo
    residual_supervision = (actual_residual - desired_residual).abs().mean()

    normal_error = (candidate_normal - target_normal).abs().mean(dim=1, keepdim=True)
    material_error = (candidate_material - target_material).abs().mean(dim=1, keepdim=True)
    normal_global = normal_error.mean()
    normal_edge = _weighted_mean(normal_error, edge_weight)
    material_global = material_error.mean()
    material_edge = _weighted_mean(material_error, edge_weight)

    support, confidence_loss, regret_loss = _detail_support_calibration(
        outputs, target_albedo, config
    )
    protected_terms = preservation.protected_preservation_terms(
        candidate_albedo, baseline_albedo, target_albedo
    )
    protected = protected_terms["protected_excess_drift_loss"].float()

    total = (
        albedo_global * 4.0
        + albedo_edge * 12.0
        + gradient * 5.0
        + laplacian * float(sr.SR_LAPLACIAN_WEIGHT)
        + pyramid * float(sr.SR_PYRAMID_WEIGHT)
        + grid_excess * float(sr.SR_GRID_EXCESS_WEIGHT)
        + regret * 16.0
        + residual_supervision * 8.0
        + normal_global * float(sr.SR_NORMAL_GLOBAL_WEIGHT)
        + normal_edge * float(sr.SR_NORMAL_EDGE_WEIGHT)
        + material_global * float(sr.SR_MATERIAL_GLOBAL_WEIGHT)
        + material_edge * float(sr.SR_MATERIAL_EDGE_WEIGHT)
        + support
        + protected * float(SR_PROTECTED_DETAIL_WEIGHT)
    ).float()

    baseline_edge = _weighted_mean(baseline_error, edge_weight)
    return {
        "total": total,
        "sr_albedo_global": albedo_global,
        "sr_albedo_edge": albedo_edge,
        "sr_gradient": gradient,
        "sr_laplacian": laplacian,
        "sr_pyramid": pyramid,
        "sr_grid_excess": grid_excess,
        "sr_regret": regret,
        "sr_residual_supervision": residual_supervision,
        "sr_normal_global": normal_global,
        "sr_normal_edge": normal_edge,
        "sr_material_global": material_global,
        "sr_material_edge": material_edge,
        "sr_edge_recovery": ((baseline_edge - albedo_edge) / baseline_edge.clamp_min(1.0e-6)).detach(),
        "sr_global_recovery": ((baseline_error.mean() - albedo_error.mean()) / baseline_error.mean().clamp_min(1.0e-6)).detach(),
        "detail_confidence": confidence_loss,
        "detail_regret_classifier": regret_loss,
        "detail_support_calibration": support,
        "detail_protected_safety_penalty": protected,
        "detail_protected_safety_rate": protected_terms["protected_preservation_rate"],
        "protected_preservation_fraction": protected_terms["protected_fraction"],
        "protected_preservation_rate": protected_terms["protected_preservation_rate"],
        "protected_preservation_mean_drift": protected_terms["protected_mean_drift"],
        "protected_preservation_max_drift": protected_terms["protected_max_drift"],
        "protected_preservation_penalty": protected,
    }


def _safe_selector_oracle(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .application import backend

    oracle, motion = backend._optimal_residual_gate(baseline, candidate, target)
    protected = preservation.protected_mask(baseline, target)
    delta = (candidate.float() - baseline.float()).abs().amax(dim=1, keepdim=True)
    tolerance = float(preservation.PROTECTED_DRIFT_TOLERANCE)
    safe_max = torch.where(
        delta <= tolerance,
        torch.ones_like(delta),
        (tolerance / delta.clamp_min(1.0e-8)).clamp(0.0, 1.0),
    ).detach()
    safe_oracle = torch.where(
        protected,
        torch.minimum(oracle.float(), safe_max),
        oracle.float(),
    ).detach()
    return safe_oracle, safe_max, protected, motion


def _selector_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["sr_candidate_albedo"].detach().float()
    final = outputs["albedo"].float()
    target = batch["target_albedo"].float()
    probability = outputs["benefit_selector_probability"].float().clamp(1.0e-5, 1.0 - 1.0e-5)
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    edge_weight = (0.20 + edge * 3.80).detach()

    safe_oracle, safe_max, protected, motion = _safe_selector_oracle(
        baseline, candidate, target
    )
    selector_weight = (
        0.10
        + edge * 2.90
        + (motion / 0.02).clamp(0.0, 1.0)
        + protected.float() * float(SR_PROTECTED_SELECTOR_WEIGHT_BOOST)
    ).detach()
    selector_loss = _weighted_mean(
        F.binary_cross_entropy(probability, safe_oracle, reduction="none"),
        selector_weight,
    )

    final_error = (final - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    final_reconstruction = _weighted_mean(final_error, edge_weight)
    final_regret = _weighted_mean(F.relu(final_error - baseline_error), edge_weight)
    final_gradient = _gradient_error(final, target, edge_weight)

    protected_weight = protected.float()
    protected_count = protected_weight.sum().clamp_min(1.0)
    gate_excess = (
        F.relu(probability - safe_max) * protected_weight
    ).sum() / protected_count

    total = (
        selector_loss * float(getattr(config, "benefit_selector_weight", 8.0))
        + final_reconstruction * float(getattr(config, "albedo_weight", 1.0)) * 2.0
        + final_gradient * float(getattr(config, "albedo_gradient_weight", 1.0))
        + final_regret * float(getattr(config, "regret_weight", 1.0)) * 3.0
        + gate_excess * float(SR_PROTECTED_SELECTOR_GATE_WEIGHT)
    ).float()

    return {
        "total": total,
        "selector_optimal_gate": selector_loss,
        "boundary_gate": selector_loss,
        "gate_target": safe_oracle.mean().detach(),
        "selector_optimal_gate_mean": safe_oracle.mean().detach(),
        "selector_edge_reconstruction": final_reconstruction,
        "selector_edge_regret": final_regret,
        "selector_edge_gradient": final_gradient,
        "selector_protected_gate_excess": gate_excess,
        "selector_protected_safe_gate_mean": safe_max[protected].mean().detach() if bool(protected.any().item()) else safe_max.new_tensor(1.0),
    }


def _active_sr_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if phase == "detail-reconstruction":
        return _detail_loss(outputs, batch, config)
    if phase in {"physical-finetune", "boundary-hardening"}:
        return _selector_loss(outputs, batch, config)
    if _PREVIOUS_BACKEND_LOSS is None:
        raise RuntimeError(f"V13.3 has no fallback loss for retired phase {phase!r}")
    return _PREVIOUS_BACKEND_LOSS(outputs, batch, config, phase)


def _phase_lr_sr_first(
    phase: str,
    config: Any,
    epoch: int | None = None,
) -> float:
    if phase == "detail-reconstruction":
        return float(SR_DETAIL_BODY_LR)
    if _PREVIOUS_PHASE_LR is None:
        return float(getattr(config, "learning_rate", 1.0e-4))
    return float(_PREVIOUS_PHASE_LR(phase, config, epoch))


def _architecture_contract_v133(self: FidelityResidualNetV9) -> dict[str, object]:
    if _PREVIOUS_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V13.3 installed without previous architecture contract")
    contract = dict(_PREVIOUS_ARCHITECTURE_CONTRACT(self))
    contract.update(
        {
            "srFirstRevision": SR_RUNTIME_REVISION,
            "productionReconstructionAuthority": "deterministic B + bounded multi-map SR residual C",
            "productionForward": "direct B -> DetailNet C -> BenefitSelector F",
            "productionComponentsActive": (
                "detail_net",
                "detail_net.albedo_head",
                "detail_net.normal_head",
                "detail_net.material_head",
                "detail_net.confidence_head",
                "detail_net.regret_head",
                "benefit_selector",
            ),
            "productionComponentsRetired": (
                "geometry_net",
                "boundary_renderer",
                "boundary_specialist",
                "seam_restorer",
            ),
            "retiredComponentsExecuted": False,
            "geometryPixelAuthority": False,
            "profilePixelAuthority": False,
            "seamPixelAuthority": False,
            "srDetailBodyLearningRate": SR_DETAIL_BODY_LR,
            "srDetailAlbedoHeadLearningRate": SR_DETAIL_ALBEDO_HEAD_LR,
            "srProtectedDetailWeight": SR_PROTECTED_DETAIL_WEIGHT,
            "srProtectionPolicy": "single V13 protected-B excess-drift term; no V12 outer preservation wrapper",
        }
    )
    return contract


def _synchronize_sr_first_runtime(self: Any, training: Any) -> None:
    if _PREVIOUS_BACKEND_SYNC is None:
        raise RuntimeError("V13.3 installed without backend synchronizer")
    _PREVIOUS_BACKEND_SYNC(self, training)
    service = getattr(training, "_training_service", None)
    if service is None:
        raise RuntimeError("NSAMDR training module has no TrainingService singleton")
    service._phase_lr = _phase_lr_sr_first
    training._nsamdr_sr_runtime_revision = SR_RUNTIME_REVISION
    training._nsamdr_sr_detail_body_lr = SR_DETAIL_BODY_LR
    training._nsamdr_sr_detail_albedo_head_lr = SR_DETAIL_ALBEDO_HEAD_LR


def _rename_runtime_integrity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for old in ("finalQualification", "trainerFinalQualification"):
        if old in result and "productionRuntimeIntegrity" not in result:
            result["productionRuntimeIntegrity"] = result.pop(old)
    return result


def _rewrite_metadata_runtime_integrity(config: Any) -> None:
    output_dir = Path(str(getattr(config, "output_dir", "")))
    metadata_name = str(getattr(config, "metadata_name", "nsamdr_v9_fidelity.json"))
    path = output_dir / metadata_name
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    changed = False
    for old in ("finalQualification", "trainerFinalQualification"):
        if old in payload and "productionRuntimeIntegrity" not in payload:
            payload["productionRuntimeIntegrity"] = payload.pop(old)
            changed = True
    if changed:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_with_runtime_terminology(self: Any, config: Any, *args: Any, **kwargs: Any):
    if _PREVIOUS_BACKEND_RUN is None:
        raise RuntimeError("V13.3 installed without backend run")
    result = _PREVIOUS_BACKEND_RUN(self, config, *args, **kwargs)
    _rewrite_metadata_runtime_integrity(config)
    return _rename_runtime_integrity(result)


def install_sr_first_runtime_contract() -> None:
    global _INSTALLED
    global _PREVIOUS_FORWARD_IMPL, _PREVIOUS_ARCHITECTURE_CONTRACT
    global _PREVIOUS_BACKEND_LOSS, _PREVIOUS_BACKEND_SYNC, _PREVIOUS_BACKEND_RUN
    global _PREVIOUS_PHASE_LR
    if _INSTALLED:
        return

    from .application import backend
    from .application.backend import TrainingBackend

    _PREVIOUS_FORWARD_IMPL = FidelityResidualNetV9._forward_impl
    FidelityResidualNetV9._forward_impl = _active_sr_forward

    _PREVIOUS_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.architecture_contract = _architecture_contract_v133

    _PREVIOUS_BACKEND_LOSS = backend._compute_losses_with_production_head_supervision
    backend._compute_losses_with_production_head_supervision = _active_sr_loss

    _PREVIOUS_PHASE_LR = backend._phase_lr_with_head_capacity

    _PREVIOUS_BACKEND_SYNC = TrainingBackend._synchronize_training_service_contract
    TrainingBackend._synchronize_training_service_contract = _synchronize_sr_first_runtime

    _PREVIOUS_BACKEND_RUN = TrainingBackend.run
    TrainingBackend.run = _run_with_runtime_terminology

    _INSTALLED = True
