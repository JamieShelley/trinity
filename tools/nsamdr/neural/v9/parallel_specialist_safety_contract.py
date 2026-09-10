from __future__ import annotations

"""V12.8 detail-anchored specialist safety and protected selector authority.

The V12.7 staged Raven proof established that D can reproduce its isolated capacity
inside the complete curriculum. The remaining failures were compositional:

* averaging weaker G/S candidates with a strong D candidate reduced D recovery;
* the unconstrained B-versus-U least-squares selector oracle could legitimately
  improve authored-target error while moving a baseline-correct pixel by more than
  the V12.4 one-code-value preservation allowance.

V12.8 therefore makes the independently qualified detail candidate the complete
candidate anchor. Structure and seam act only as complementary residual corrections
where D support is weak and candidate conflict is low. The final BenefitSelector
still owns the only B-versus-U deployment decision.

Training additionally caps the selector teacher on V12.4 protected pixels by the
maximum gate that can keep F within one 8-bit code value of B. This target-dependent
cap exists only in supervision; production inference still consumes LR evidence only.
No trainable parameters, checkpoint keys, or inference inputs are added.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9
from . import parallel_specialist_arbitration_contract as v126
from . import parallel_specialist_fusion_contract as v125


PARALLEL_SPECIALIST_SAFETY_REVISION = "V12.8"
AUXILIARY_TOTAL_WEIGHT_MAX = 0.50
DETAIL_PROTECTED_EXTRA_WEIGHT = 96.0
PROTECTED_SELECTOR_WEIGHT_BOOST = 8.0
PROTECTED_GATE_EXCESS_WEIGHT = 64.0

_INSTALLED = False
_ORIGINAL_FORWARD_IMPL: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None
_PREVIOUS_PRODUCTION_HEAD_LOSS: Any = None


# Purpose: Allow G/S only where direct detail is uncertain and specialists agree.
# Called by: _forward_with_detail_anchor and tests.
# Calls: pointwise tensor operations only.
def _complementary_auxiliary_weight(
    raw_weight: torch.Tensor,
    detail_evidence: torch.Tensor,
    conflict: torch.Tensor,
) -> torch.Tensor:
    room = (1.0 - detail_evidence.float().clamp(0.0, 1.0)).square()
    agreement = (1.0 - conflict.float().clamp(0.0, 1.0))
    return raw_weight.float().clamp(0.0, 1.0) * room * agreement


# Purpose: Bound the combined G/S correction authority without diluting D itself.
# Called by: _forward_with_detail_anchor and tests.
# Calls: pointwise tensor operations only.
def _cap_auxiliary_pair(
    structure_weight: torch.Tensor,
    seam_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = structure_weight.float() + seam_weight.float()
    scale = torch.clamp(
        float(AUXILIARY_TOTAL_WEIGHT_MAX) / total.clamp_min(1.0e-6),
        max=1.0,
    )
    return structure_weight.float() * scale, seam_weight.float() * scale


# Purpose: Compose U from full D plus bounded complementary B-relative G/S corrections.
# Called by: _forward_with_detail_anchor and tests.
# Calls: pointwise tensor operations only.
def _detail_anchored_fusion(
    baseline: torch.Tensor,
    detail: torch.Tensor,
    structure: torch.Tensor,
    seam: torch.Tensor,
    structure_weight: torch.Tensor,
    seam_weight: torch.Tensor,
) -> torch.Tensor:
    correction = (
        (structure.float() - baseline.float()) * structure_weight.float()
        + (seam.float() - baseline.float()) * seam_weight.float()
    )
    return detail.float() + correction


# Purpose: Recompose V12.8 U/F while preserving the already-trained specialist values.
# Called by: FidelityResidualNetV9._forward_impl.
# Calls: V12.6 support sharpening, V12.5 conflict features, BenefitSelector.
def _forward_with_detail_anchor(
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
        raise RuntimeError("V12.8 safety installed without V12.6/V12.7 forward")

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
    detail_evidence = (
        outputs["detail_confidence"].float().clamp(0.0, 1.0)
        * (1.0 - outputs["detail_regret"].float().clamp(0.0, 1.0))
    )

    raw_structure_weight = v126._sharpen_support(structure_support)
    raw_seam_weight = v126._sharpen_support(seam_support)
    raw_conflict = v125._fusion_conflict(
        baseline_albedo,
        (structure_albedo, seam_albedo, detail_albedo),
        (structure_support, seam_support, detail_evidence),
    )
    structure_weight = _complementary_auxiliary_weight(
        raw_structure_weight, detail_evidence, raw_conflict
    )
    seam_weight = _complementary_auxiliary_weight(
        raw_seam_weight, detail_evidence, raw_conflict
    )
    structure_weight, seam_weight = _cap_auxiliary_pair(
        structure_weight, seam_weight
    )

    fused_albedo = _detail_anchored_fusion(
        baseline_albedo,
        detail_albedo,
        structure_albedo,
        seam_albedo,
        structure_weight,
        seam_weight,
    ).clamp(0.0, 1.0)
    fused_normal = FidelityResidualNetV9._normalize_xy(
        _detail_anchored_fusion(
            baseline_normal,
            detail_normal,
            structure_normal,
            seam_normal,
            structure_weight,
            seam_weight,
        ).to(baseline_normal.dtype)
    )
    fused_material = _detail_anchored_fusion(
        baseline_material,
        detail_material,
        structure_material,
        seam_material,
        structure_weight,
        seam_weight,
    ).clamp(0.0, 1.0)

    conflict = v125._fusion_conflict(
        baseline_albedo,
        (structure_albedo, seam_albedo, detail_albedo),
        (structure_weight, seam_weight, detail_evidence),
    )
    fusion_support = torch.maximum(
        detail_evidence,
        torch.maximum(structure_weight, seam_weight),
    ).clamp(0.0, 1.0)
    observed_support = outputs["observed_source_edge_support"].float()
    if observed_support.shape[-2:] != fused_albedo.shape[-2:]:
        observed_support = F.interpolate(
            observed_support,
            size=fused_albedo.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

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
            "parallel_structure_fusion_weight": structure_weight.to(albedo.dtype),
            "parallel_seam_fusion_weight": seam_weight.to(albedo.dtype),
            # D is the independently qualified anchor; confidence/regret remains
            # evidence for complementary-room and final-selector decisions.
            "parallel_detail_fusion_weight": torch.ones_like(detail_evidence).to(albedo.dtype),
            "parallel_auxiliary_room": (1.0 - detail_evidence).square().to(albedo.dtype),
            "parallel_fusion_support": fusion_support.to(albedo.dtype),
            "parallel_fusion_conflict": conflict.to(albedo.dtype),
        }
    )
    return outputs


# Purpose: Build a selector teacher that cannot violate V12.4 protected-B drift.
# Called by: _loss_with_protected_selector_safety and tests.
# Calls: backend optimal-gate helper and V12.4 protected mask.
def _safe_selector_oracle(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .application import backend
    from . import baseline_relative_specialist_contract as preservation

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


# Purpose: Preserve D on already-correct B pixels and train F against a preservation-safe oracle.
# Called by: V12.4 protected-preservation outer wrapper.
# Calls: prior V12.7 loss chain, V12.4 preservation helpers, _safe_selector_oracle.
def _loss_with_protected_selector_safety(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _PREVIOUS_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError("V12.8 safety installed without previous production-head loss")

    losses = _PREVIOUS_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase not in {"detail-reconstruction", "physical-finetune"}:
        return losses

    from .application import backend
    from . import baseline_relative_specialist_contract as preservation

    baseline = outputs["baseline_albedo"].detach().float()
    target = batch["target_albedo"].float()

    if phase == "detail-reconstruction":
        detail = outputs["detail_candidate_albedo"].float()
        terms = preservation.protected_preservation_terms(detail, baseline, target)
        penalty = terms["protected_excess_drift_loss"].float()
        losses["detail_protected_safety_penalty"] = penalty
        losses["detail_protected_safety_rate"] = terms[
            "protected_preservation_rate"
        ]
        losses["total"] = (
            losses["total"].float()
            + penalty * float(DETAIL_PROTECTED_EXTRA_WEIGHT)
        )
        return losses

    candidate = outputs["fused_candidate_albedo"].detach().float()
    probability = outputs["benefit_selector_probability"].float().clamp(
        1.0e-5, 1.0 - 1.0e-5
    )
    safe_oracle, safe_max, protected, motion = _safe_selector_oracle(
        baseline, candidate, target
    )
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    selector_weight = (
        0.10
        + edge * 2.90
        + (motion / 0.02).clamp(0.0, 1.0)
        + protected.float() * float(PROTECTED_SELECTOR_WEIGHT_BOOST)
    ).detach()
    safe_loss = backend._weighted_mean(
        F.binary_cross_entropy(probability, safe_oracle, reduction="none"),
        selector_weight,
    )

    protected_weight = protected.float()
    protected_count = protected_weight.sum().clamp_min(1.0)
    gate_excess = (
        F.relu(probability - safe_max) * protected_weight
    ).sum() / protected_count

    previous_selector = losses.get("selector_optimal_gate")
    selector_scale = float(getattr(config, "benefit_selector_weight", 8.0))
    if isinstance(previous_selector, torch.Tensor):
        losses["total"] = (
            losses["total"].float()
            + (safe_loss - previous_selector.float()) * selector_scale
        )
    else:
        losses["total"] = losses["total"].float() + safe_loss * selector_scale
    losses["total"] = (
        losses["total"].float()
        + gate_excess * float(PROTECTED_GATE_EXCESS_WEIGHT)
    )

    losses["selector_optimal_gate"] = safe_loss
    losses["boundary_gate"] = safe_loss
    losses["gate_target"] = safe_oracle.mean().detach()
    losses["selector_optimal_gate_mean"] = safe_oracle.mean().detach()
    losses["selector_protected_safe_gate_mean"] = (
        (safe_max * protected_weight).sum() / protected_count
    ).detach()
    losses["selector_protected_gate_excess"] = gate_excess
    losses["selector_protected_fraction"] = protected_weight.mean().detach()
    return losses


# Purpose: Publish V12.8 composition and selector-safety semantics in architecture evidence.
# Called by: FidelityResidualNetV9.architecture_contract.
# Calls: previous architecture contract.
def _architecture_contract_with_specialist_safety(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V12.8 safety installed without architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelSpecialistSafetyRevision"] = PARALLEL_SPECIALIST_SAFETY_REVISION
    composition = dict(contract.get("specialistComposition") or {})
    composition["fusion"] = (
        "detail-anchored complementary residual fusion: D is retained in full; "
        "G/S add bounded corrections only where D support is weak and candidate conflict is low"
    )
    composition["finalAuthority"] = (
        "BenefitSelector remains the sole B-versus-U deployment authority"
    )
    contract["specialistComposition"] = composition
    contract["detailAnchorFusion"] = True
    contract["auxiliarySpecialistTotalWeightMax"] = AUXILIARY_TOTAL_WEIGHT_MAX
    contract["protectedSelectorTraining"] = (
        "training-only authored target caps the selector oracle on baseline-correct pixels "
        "to the maximum gate consistent with the V12.4 one-code-value drift limit"
    )
    contract["protectedSelectorInferenceTargetUse"] = False
    return contract


# Purpose: Install V12.8 after V12.7 while retaining V12.4 as the outer loss owner.
# Called by: v9 package import.
# Calls: model forward/architecture replacement and preservation loss chaining.
def install_parallel_specialist_safety_contract() -> None:
    global _INSTALLED, _ORIGINAL_FORWARD_IMPL, _ORIGINAL_ARCHITECTURE_CONTRACT
    global _PREVIOUS_PRODUCTION_HEAD_LOSS
    if _INSTALLED:
        return

    from . import baseline_relative_specialist_contract as preservation
    from .application import backend

    _ORIGINAL_FORWARD_IMPL = FidelityResidualNetV9._forward_impl
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9._forward_impl = _forward_with_detail_anchor
    FidelityResidualNetV9.architecture_contract = (
        _architecture_contract_with_specialist_safety
    )

    current_inner = preservation._ORIGINAL_PRODUCTION_HEAD_LOSS
    if current_inner is None:
        raise RuntimeError("V12.8 safety requires the V12.4/V12.7 loss chain")
    _PREVIOUS_PRODUCTION_HEAD_LOSS = current_inner
    preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = (
        _loss_with_protected_selector_safety
    )
    backend._compute_losses_with_production_head_supervision = (
        preservation._loss_with_protected_preservation
    )
    _INSTALLED = True
