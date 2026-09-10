from __future__ import annotations

"""V12.6 support arbitration for independent baseline-relative specialists.

The first full G/S/D Micro run exposed two composition problems that the isolated
V12.5 detail proof could not see:

* the direct-detail confidence/regret heads were telemetry-only because the V12.3
  exact detail objective replaced their training terms;
* linear support averaging let weak specialists dilute a stronger candidate.

V12.6 keeps the same B-relative candidates and BenefitSelector topology. Detail
support is supervised explicitly against deterministic baseline B, then local
specialist support is sharpened before convex fusion. Weak support therefore becomes
negligible instead of averaging down a well-supported specialist. No parameters,
checkpoint keys or inference inputs change.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9
from . import parallel_specialist_fusion_contract as v125


PARALLEL_SPECIALIST_ARBITRATION_REVISION = "V12.6"
SUPPORT_POWER = 4.0
DETAIL_SUPPORT_FLOOR = 0.05

_INSTALLED = False
_ORIGINAL_FORWARD_IMPL: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None
_PREVIOUS_PRODUCTION_HEAD_LOSS: Any = None


# Purpose: Convert a soft specialist authority into a winner-preserving convex weight.
# Called by: _forward_with_specialist_arbitration and tests.
# Calls: PyTorch pointwise operations only.
def _sharpen_support(support: torch.Tensor) -> torch.Tensor:
    return support.float().clamp(0.0, 1.0).pow(SUPPORT_POWER)


# Purpose: Derive continuous detail support from the confidence/regret heads.
# Called by: _forward_with_specialist_arbitration and tests.
# Calls: PyTorch pointwise operations only.
def _detail_support(
    confidence: torch.Tensor,
    regret: torch.Tensor,
) -> torch.Tensor:
    evidence = confidence.float().clamp(0.0, 1.0) * (
        1.0 - regret.float().clamp(0.0, 1.0)
    )
    return evidence.clamp_min(DETAIL_SUPPORT_FLOOR).clamp_max(1.0)


# Purpose: Recompute V12.5 U/F using calibrated, sharpened specialist authority.
# Called by: FidelityResidualNetV9._forward_impl after the V12.5 forward.
# Calls: V12.5 fusion helpers and the existing BenefitSelector.
def _forward_with_specialist_arbitration(
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
    if _ORIGINAL_FORWARD_IMPL is None:
        raise RuntimeError("V12.6 arbitration installed without V12.5 forward")

    outputs = _ORIGINAL_FORWARD_IMPL(
        self,
        inputs,
        sdf_override=sdf_override,
        gate_override=gate_override,
        hardness_override=hardness_override,
        seam_authority_override=seam_authority_override,
        seam_tangent_override=seam_tangent_override,
        phase_only_seam_teacher=phase_only_seam_teacher,
    )

    baseline_albedo = outputs["baseline_albedo"]
    baseline_normal = outputs["baseline_normal"]
    baseline_material = outputs["baseline_material"]
    structure_albedo = outputs["structure_candidate_albedo"]
    structure_normal = outputs["structure_candidate_normal"]
    structure_material = outputs["structure_candidate_material"]
    seam_albedo = outputs["seam_candidate_albedo"]
    seam_normal = outputs["seam_candidate_normal"]
    seam_material = outputs["seam_candidate_material"]
    detail_albedo = outputs["detail_candidate_albedo"]
    detail_normal = outputs["detail_candidate_normal"]
    detail_material = outputs["detail_candidate_material"]

    structure_support = outputs["parallel_structure_support"].float().clamp(0.0, 1.0)
    seam_support = outputs["parallel_seam_support"].float().clamp(0.0, 1.0)
    detail_support = _detail_support(
        outputs["detail_confidence"],
        outputs["detail_regret"],
    )

    structure_weight = _sharpen_support(structure_support)
    seam_weight = _sharpen_support(seam_support)
    detail_weight = _sharpen_support(detail_support)
    weights = (structure_weight, seam_weight, detail_weight)

    fused_albedo = v125._weighted_delta_fusion(
        baseline_albedo,
        (structure_albedo, seam_albedo, detail_albedo),
        weights,
    ).clamp(0.0, 1.0)
    fused_normal = FidelityResidualNetV9._normalize_xy(
        v125._weighted_delta_fusion(
            baseline_normal,
            (structure_normal, seam_normal, detail_normal),
            weights,
        )
    )
    fused_material = v125._weighted_delta_fusion(
        baseline_material,
        (structure_material, seam_material, detail_material),
        weights,
    ).clamp(0.0, 1.0)

    observed_support = outputs["observed_source_edge_support"].float()
    if observed_support.shape[-2:] != fused_albedo.shape[-2:]:
        observed_support = F.interpolate(
            observed_support,
            size=fused_albedo.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

    detail_evidence = (
        outputs["detail_confidence"].float().clamp(0.0, 1.0)
        * (1.0 - outputs["detail_regret"].float().clamp(0.0, 1.0))
    )
    fusion_support = torch.maximum(
        structure_support,
        torch.maximum(seam_support, detail_evidence),
    )
    conflict = v125._fusion_conflict(
        baseline_albedo,
        (structure_albedo, seam_albedo, detail_albedo),
        (structure_support, seam_support, detail_support),
    )

    selector_features = self._selector_features(
        baseline_albedo,
        fused_albedo,
        outputs["sdf_pixels"],
        outputs["boundary_normal"],
        outputs["boundary_refined_coverage"],
        outputs["boundary_specialist_confidence"],
        observed_support,
        torch.sigmoid(outputs["edge_logits"].float()),
        fusion_support,
        conflict,
    )
    selector_logits = self._run_training_component(
        self.benefit_selector,
        selector_features,
    )
    selector_probability = torch.sigmoid(selector_logits.float())
    final_gate = selector_probability

    albedo = (
        baseline_albedo.float() * (1.0 - final_gate)
        + fused_albedo.float() * final_gate
    ).clamp(0.0, 1.0).to(baseline_albedo.dtype)
    normal = FidelityResidualNetV9._normalize_xy(
        (
            baseline_normal.float() * (1.0 - final_gate)
            + fused_normal.float() * final_gate
        ).to(baseline_normal.dtype)
    )
    material = (
        baseline_material.float() * (1.0 - final_gate)
        + fused_material.float() * final_gate
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

    outputs.update(
        {
            "albedo": albedo,
            "normal_xy": normal,
            "material": material,
            "emissive": material[:, 1:2],
            "roughness": material[:, 2:3],
            "material_logits": material_logits,
            "confidence": confidence.to(albedo.dtype),
            "confidence_logits": torch.logit(confidence).to(albedo.dtype),
            "boundary_gate_logits": selector_logits.to(albedo.dtype),
            "boundary_gate_prediction": selector_probability.to(albedo.dtype),
            "boundary_gate_probability": selector_probability.to(albedo.dtype),
            "benefit_selector_logits": selector_logits.to(albedo.dtype),
            "benefit_selector_probability": selector_probability.to(albedo.dtype),
            "final_selector_gate": final_gate.to(albedo.dtype),
            "fused_candidate_albedo": fused_albedo.to(albedo.dtype),
            "fused_candidate_normal": fused_normal.to(normal.dtype),
            "fused_candidate_material": fused_material.to(material.dtype),
            "parallel_detail_support": detail_evidence.to(albedo.dtype),
            "parallel_structure_fusion_weight": structure_weight.to(albedo.dtype),
            "parallel_seam_fusion_weight": seam_weight.to(albedo.dtype),
            "parallel_detail_fusion_weight": detail_weight.to(albedo.dtype),
            "parallel_fusion_support": fusion_support.to(albedo.dtype),
            "parallel_fusion_conflict": conflict.to(albedo.dtype),
        }
    )
    return outputs


# Purpose: Train detail support heads against D's improvement/regret relative to B.
# Called by: V12.4 preservation wrapper during detail-reconstruction.
# Calls: the V12.5 aligned production-head loss plus BCE support supervision.
def _loss_with_detail_support_calibration(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _PREVIOUS_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError("V12.6 arbitration installed without V12.5 training loss")

    losses = _PREVIOUS_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase != "detail-reconstruction":
        return losses

    required = (
        "baseline_albedo",
        "detail_candidate_albedo",
        "detail_confidence_logits",
        "detail_regret_logits",
    )
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(f"V12.6 detail support calibration missing outputs: {missing}")

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["detail_candidate_albedo"].detach().float()
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    improvement = (baseline_error - candidate_error).detach()
    confidence_target = (
        0.5 + improvement / max(float(getattr(config, "gate_error_scale", 0.08)), 1.0e-4)
    ).clamp(0.0, 1.0)
    regret_target = (
        candidate_error > baseline_error + 0.001
    ).float().detach()

    confidence_loss = F.binary_cross_entropy_with_logits(
        outputs["detail_confidence_logits"].float(),
        confidence_target.float(),
    )
    regret_loss = F.binary_cross_entropy_with_logits(
        outputs["detail_regret_logits"].float(),
        regret_target.float(),
    )
    confidence_weight = float(getattr(config, "detail_confidence_weight", 0.2))
    regret_weight = float(getattr(config, "detail_regret_classifier_weight", 0.4))
    support_loss = (
        confidence_loss.float() * confidence_weight
        + regret_loss.float() * regret_weight
    )
    losses["detail_support_confidence"] = confidence_loss
    losses["detail_support_regret"] = regret_loss
    losses["detail_support_confidence_target"] = confidence_target.mean().detach()
    losses["detail_support_regret_target"] = regret_target.mean().detach()
    losses["detail_support_calibration"] = support_loss
    losses["total"] = losses["total"].float() + support_loss
    return losses


# Purpose: Record V12.6 arbitration semantics in architecture evidence.
# Called by: FidelityResidualNetV9.architecture_contract.
# Calls: the previously installed architecture contract.
def _architecture_contract_with_specialist_arbitration(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V12.6 arbitration installed without architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelSpecialistArbitrationRevision"] = (
        PARALLEL_SPECIALIST_ARBITRATION_REVISION
    )
    contract["specialistFusionArbitration"] = {
        "supportPower": SUPPORT_POWER,
        "detailSupportFloor": DETAIL_SUPPORT_FLOOR,
        "detailSupport": "detail confidence * (1 - detail regret)",
        "fusion": "support-sharpened convex B-relative delta fusion",
        "reason": "weak specialists cannot linearly dilute a stronger supported candidate",
        "finalAuthority": "BenefitSelector remains sole B-versus-U authority",
    }
    contract["detailSupportTraining"] = (
        "detail confidence/regret are supervised from D improvement/regret relative to deterministic B"
    )
    return contract


# Purpose: Install V12.6 after V12.5 fusion/training while preserving V12.4 as outer loss wrapper.
# Called by: v9 package import.
# Calls: model forward/architecture replacement plus preservation loss chaining.
def install_parallel_specialist_arbitration_contract() -> None:
    global _INSTALLED, _ORIGINAL_FORWARD_IMPL, _ORIGINAL_ARCHITECTURE_CONTRACT
    global _PREVIOUS_PRODUCTION_HEAD_LOSS
    if _INSTALLED:
        return

    from . import baseline_relative_specialist_contract as preservation
    from .application import backend

    _ORIGINAL_FORWARD_IMPL = FidelityResidualNetV9._forward_impl
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9._forward_impl = _forward_with_specialist_arbitration
    FidelityResidualNetV9.architecture_contract = (
        _architecture_contract_with_specialist_arbitration
    )

    current_inner = preservation._ORIGINAL_PRODUCTION_HEAD_LOSS
    if current_inner is None:
        raise RuntimeError("V12.6 arbitration requires V12.4/V12.5 loss chain")
    _PREVIOUS_PRODUCTION_HEAD_LOSS = current_inner
    preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = _loss_with_detail_support_calibration
    backend._compute_losses_with_production_head_supervision = (
        preservation._loss_with_protected_preservation
    )
    _INSTALLED = True
