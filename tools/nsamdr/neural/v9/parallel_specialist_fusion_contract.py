from __future__ import annotations

"""V12.5 independent baseline-relative specialist fusion.

V12.3 proved the detail specialist only when its candidate was anchored directly to
deterministic B. V12.5 applies the same composition rule to the remaining production
specialists without adding parameters or checkpoint keys:

* structure remains the existing bounded B-relative structural candidate;
* seam runs from B and LR-observable seam evidence instead of the structural result;
* detail remains the proven direct B-relative residual;
* the three already-bounded deltas are fused by local specialist support;
* BenefitSelector is the sole final B-versus-fused authority.

Training-only forced seam/tangent authorities remain available to existing proof code,
but learned geometry tensors are never a prerequisite for the production seam candidate.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9, UPSCALE_FACTOR
from .seam_restoration import (
    DirectionalSeamRestorer,
    multi_map_structure_tensor,
)


PARALLEL_SPECIALIST_FUSION_REVISION = "V12.5"

_INSTALLED = False
_ORIGINAL_FORWARD_IMPL: Any = None
_ORIGINAL_SEAM_FORWARD: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None


# Purpose: Reconstruct deterministic B directly from the native LR physical maps.
# Called by: _seam_forward_from_baseline.
# Calls: PyTorch interpolation only.
def _deterministic_baseline_from_sources(
    source_albedo: torch.Tensor,
    source_normal: torch.Tensor,
    source_material: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    albedo = F.interpolate(
        source_albedo,
        scale_factor=UPSCALE_FACTOR,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    normal = FidelityResidualNetV9._normalize_xy(
        F.interpolate(
            source_normal,
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        )
    )
    material = F.interpolate(
        source_material,
        scale_factor=UPSCALE_FACTOR,
        mode="nearest",
    )
    return albedo, normal, material


# Purpose: Build the seam candidate from B and source-observable LR seam evidence.
# Called by: FidelityResidualNetV9._forward_impl through DirectionalSeamRestorer.
# Calls: original DirectionalSeamRestorer.forward and multi_map_structure_tensor.
def _seam_forward_from_baseline(
    self: DirectionalSeamRestorer,
    albedo: torch.Tensor,
    normal_xy: torch.Tensor,
    material: torch.Tensor,
    *,
    sdf_pixels: torch.Tensor,
    coverage: torch.Tensor,
    profile_confidence: torch.Tensor,
    edge_probability: torch.Tensor,
    geometry_normal: torch.Tensor | None = None,
    source_albedo: torch.Tensor | None = None,
    source_normal: torch.Tensor | None = None,
    source_material: torch.Tensor | None = None,
    authority_override: torch.Tensor | None = None,
    tangent_override: torch.Tensor | None = None,
    phase_only: bool = False,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    if _ORIGINAL_SEAM_FORWARD is None:
        raise RuntimeError("parallel specialist fusion installed without original seam forward")
    if source_albedo is None or source_normal is None or source_material is None:
        # Keep direct component callers/tests compatible. The production model always
        # supplies native LR maps, so its seam path still takes the independent branch.
        return _ORIGINAL_SEAM_FORWARD(
            self,
            albedo,
            normal_xy,
            material,
            sdf_pixels=sdf_pixels,
            coverage=coverage,
            profile_confidence=profile_confidence,
            edge_probability=edge_probability,
            geometry_normal=geometry_normal,
            source_albedo=source_albedo,
            source_normal=source_normal,
            source_material=source_material,
            authority_override=authority_override,
            tangent_override=tangent_override,
            phase_only=phase_only,
            enabled=enabled,
        )

    baseline_albedo, baseline_normal, baseline_material = (
        _deterministic_baseline_from_sources(
            source_albedo,
            source_normal,
            source_material,
        )
    )

    observed = multi_map_structure_tensor(
        source_albedo,
        source_normal,
        source_material,
        radius=self.tensor_radius,
        strength_gain=self.strength_gain,
    )
    observed_edge = F.interpolate(
        observed["strength"].float(),
        size=baseline_albedo.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    # The seam specialist already derives normal/tangent/ridge evidence from native LR.
    # A far constant SDF explicitly removes learned geometry support without changing
    # PhaseAwareSeamSR's checkpoint shape. Coverage/profile are neutral for the same
    # reason; source evidence remains active through observed_edge and the internal
    # structure tensor/ridge path.
    neutral_sdf = torch.full_like(observed_edge, 1.0e4)
    neutral_coverage = torch.zeros_like(observed_edge)
    neutral_profile = torch.zeros_like(observed_edge)

    return _ORIGINAL_SEAM_FORWARD(
        self,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        sdf_pixels=neutral_sdf,
        coverage=neutral_coverage,
        profile_confidence=neutral_profile,
        edge_probability=observed_edge,
        geometry_normal=None,
        source_albedo=source_albedo,
        source_normal=source_normal,
        source_material=source_material,
        authority_override=authority_override,
        tangent_override=tangent_override,
        phase_only=phase_only,
        enabled=enabled,
    )


# Purpose: Fuse already-bounded B-relative specialist candidates without serial ownership.
# Called by: _forward_with_parallel_specialist_fusion and unit tests.
# Calls: PyTorch reductions only.
def _weighted_delta_fusion(
    baseline: torch.Tensor,
    candidates: tuple[torch.Tensor, ...],
    supports: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if not candidates or len(candidates) != len(supports):
        raise ValueError("parallel fusion requires one support per candidate")

    numerator = torch.zeros_like(baseline, dtype=torch.float32)
    denominator = torch.zeros_like(baseline[:, :1], dtype=torch.float32)
    for candidate, support in zip(candidates, supports):
        if candidate.shape != baseline.shape:
            raise ValueError("parallel fusion candidate shape mismatch")
        weight = support.float().clamp(0.0, 1.0)
        if weight.shape[-2:] != baseline.shape[-2:]:
            weight = F.interpolate(
                weight,
                size=baseline.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if weight.shape[1] != 1:
            weight = weight.mean(dim=1, keepdim=True)
        numerator = numerator + (candidate.float() - baseline.float()) * weight
        denominator = denominator + weight

    fused_delta = torch.where(
        denominator > 1.0e-6,
        numerator / denominator.clamp_min(1.0e-6),
        torch.zeros_like(numerator),
    )
    return (baseline.float() + fused_delta).to(baseline.dtype)


# Purpose: Build unchanged-width selector features for the complete fused candidate.
# Called by: original and V12.5 forward composition.
# Calls: FidelityResidualNetV9 gradient helpers.
def _fusion_selector_features(
    self: FidelityResidualNetV9,
    baseline_albedo: torch.Tensor,
    candidate_albedo: torch.Tensor,
    sdf_pixels: torch.Tensor,
    normal: torch.Tensor,
    coverage: torch.Tensor,
    profile_confidence: torch.Tensor,
    observed_support: torch.Tensor,
    edge_probability: torch.Tensor,
    detail_confidence: torch.Tensor | None = None,
    detail_regret: torch.Tensor | None = None,
) -> torch.Tensor:
    del sdf_pixels, normal, coverage, profile_confidence, edge_probability
    baseline_gray = baseline_albedo.float().mean(dim=1, keepdim=True)
    candidate_gray = candidate_albedo.float().mean(dim=1, keepdim=True)
    difference = (candidate_albedo.float() - baseline_albedo.float()).abs().mean(
        dim=1, keepdim=True
    )
    signed_delta = candidate_gray - baseline_gray
    bgx, bgy = self._gradient_xy_scalar(baseline_gray)
    cgx, cgy = self._gradient_xy_scalar(candidate_gray)
    gradient_delta = torch.sqrt(
        (cgx - bgx).square() + (cgy - bgy).square() + 1.0e-8
    )
    fusion_support = (
        torch.zeros_like(observed_support)
        if detail_confidence is None
        else detail_confidence.float()
    )
    fusion_conflict = (
        torch.zeros_like(observed_support)
        if detail_regret is None
        else detail_regret.float()
    )
    return torch.cat(
        (
            baseline_gray,
            candidate_gray,
            difference,
            signed_delta,
            (bgx * 4.0).clamp(-1.0, 1.0),
            (bgy * 4.0).clamp(-1.0, 1.0),
            (cgx * 4.0).clamp(-1.0, 1.0),
            (cgy * 4.0).clamp(-1.0, 1.0),
            (gradient_delta * 4.0).clamp(0.0, 1.0),
            observed_support.float().clamp(0.0, 1.0),
            fusion_support.clamp(0.0, 1.0),
            fusion_conflict.clamp(0.0, 1.0),
        ),
        dim=1,
    )


# Purpose: Measure disagreement between independently proposed albedo deltas.
# Called by: _forward_with_parallel_specialist_fusion.
# Calls: PyTorch pointwise operations only.
def _fusion_conflict(
    baseline: torch.Tensor,
    candidates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    supports: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    deltas = [
        (candidate.float() - baseline.float()).mean(dim=1, keepdim=True)
        for candidate in candidates
    ]
    g, s, d = [support.float().clamp(0.0, 1.0) for support in supports]
    pair_weight = torch.minimum(g, s) + torch.minimum(g, d) + torch.minimum(s, d)
    disagreement = (
        (deltas[0] - deltas[1]).abs() * torch.minimum(g, s)
        + (deltas[0] - deltas[2]).abs() * torch.minimum(g, d)
        + (deltas[1] - deltas[2]).abs() * torch.minimum(s, d)
    )
    return (disagreement / pair_weight.clamp_min(1.0e-6) * 4.0).clamp(0.0, 1.0)


# Purpose: Replace serial production composition with independent B-relative fusion.
# Called by: FidelityResidualNetV9.forward via the existing _forward_impl dispatch.
# Calls: original _forward_impl, _weighted_delta_fusion, BenefitSelector.
def _forward_with_parallel_specialist_fusion(
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
        raise RuntimeError("parallel specialist fusion installed without original forward")
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

    structure_albedo = outputs["boundary_pre_seam_albedo"]
    structure_normal = outputs["boundary_pre_seam_normal"]
    structure_material = outputs["boundary_pre_seam_material"]

    # V12.5 seam is now generated from B by _seam_forward_from_baseline.
    seam_albedo = outputs["boundary_reconstructed_albedo"]
    seam_normal = outputs["boundary_reconstructed_normal"]
    seam_material = outputs["boundary_reconstructed_material"]

    # V12.3 keeps these exact direct-detail candidates independent of G/S.
    detail_albedo = outputs["detail_candidate_albedo"]
    detail_normal = outputs["detail_candidate_normal"]
    detail_material = outputs["detail_candidate_material"]

    structure_support = outputs["boundary_gate"].float().clamp(0.0, 1.0)
    seam_support = outputs["seam_authority"].float().clamp(0.0, 1.0)
    detail_evidence = (
        ((outputs["detail_confidence"].float() - 0.50) / 0.35).clamp(0.0, 1.0)
        * ((0.50 - outputs["detail_regret"].float()) / 0.35).clamp(0.0, 1.0)
    )
    # Direct detail is the already-qualified V12.3 path. Give it unit fusion weight
    # so G/S identity reproduces D exactly; confidence/regret remains selector evidence
    # rather than silently erasing the demonstrated candidate before selection.
    detail_support = torch.ones_like(structure_support)

    supports = (structure_support, seam_support, detail_support)
    fused_albedo = _weighted_delta_fusion(
        baseline_albedo,
        (structure_albedo, seam_albedo, detail_albedo),
        supports,
    ).clamp(0.0, 1.0)
    fused_normal = FidelityResidualNetV9._normalize_xy(
        _weighted_delta_fusion(
            baseline_normal,
            (structure_normal, seam_normal, detail_normal),
            supports,
        )
    )
    fused_material = _weighted_delta_fusion(
        baseline_material,
        (structure_material, seam_material, detail_material),
        supports,
    ).clamp(0.0, 1.0)

    observed_support = outputs["observed_source_edge_support"].float()
    if observed_support.shape[-2:] != fused_albedo.shape[-2:]:
        observed_support = F.interpolate(
            observed_support,
            size=fused_albedo.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

    fusion_support = torch.maximum(
        structure_support,
        torch.maximum(seam_support, detail_evidence),
    )
    conflict = _fusion_conflict(
        baseline_albedo,
        (structure_albedo, seam_albedo, detail_albedo),
        supports,
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
            "structure_candidate_albedo": structure_albedo,
            "structure_candidate_normal": structure_normal,
            "structure_candidate_material": structure_material,
            "seam_candidate_albedo": seam_albedo,
            "seam_candidate_normal": seam_normal,
            "seam_candidate_material": seam_material,
            "fused_candidate_albedo": fused_albedo.to(albedo.dtype),
            "fused_candidate_normal": fused_normal.to(normal.dtype),
            "fused_candidate_material": fused_material.to(material.dtype),
            "parallel_structure_support": structure_support.to(albedo.dtype),
            "parallel_seam_support": seam_support.to(albedo.dtype),
            "parallel_detail_support": detail_evidence.to(albedo.dtype),
            "parallel_detail_fusion_weight": detail_support.to(albedo.dtype),
            "parallel_fusion_support": fusion_support.to(albedo.dtype),
            "parallel_fusion_conflict": conflict.to(albedo.dtype),
        }
    )
    return outputs


# Purpose: Publish the executable V12.5 composition in architecture evidence.
# Called by: FidelityResidualNetV9.architecture_contract.
# Calls: the previously installed architecture contract.
def _architecture_contract_parallel_specialists(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError(
            "parallel specialist fusion installed without architecture contract"
        )
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelSpecialistFusionRevision"] = PARALLEL_SPECIALIST_FUSION_REVISION
    contract["specialistComposition"] = {
        "structure": "existing bounded B-relative structural candidate",
        "seamProfile": (
            "bounded seam candidate generated from deterministic B and native-LR "
            "structure/ridge evidence; no learned-geometry prerequisite"
        ),
        "detail": "V12.3 bounded direct residual over deterministic B",
        "fusion": (
            "support-weighted convex fusion of already-bounded B-relative deltas; "
            "a sole supported specialist is retained exactly"
        ),
        "finalAuthority": (
            "BenefitSelector is the sole final blend between deterministic B and "
            "the complete fused candidate"
        ),
    }
    contract["seamBase"] = "deterministic B"
    contract["seamLearnedGeometryPrerequisite"] = False
    contract["selectorRequires"] = (
        "B + fused candidate + observable LR support + fusion support/conflict"
    )
    contract["candidateAuthority"] = (
        "BenefitSelector probability alone blends B with the fused candidate; "
        "detail confidence/regret are specialist evidence, not hard final vetoes"
    )
    return contract


# Purpose: Install V12.5 after V12.3 detail isolation and V12.4 preservation.
# Called by: v9 package import.
# Calls: model/seam method replacement only; no parameter creation.
def install_parallel_specialist_fusion_contract() -> None:
    global _INSTALLED
    global _ORIGINAL_FORWARD_IMPL, _ORIGINAL_SEAM_FORWARD
    global _ORIGINAL_ARCHITECTURE_CONTRACT

    if _INSTALLED:
        return

    _ORIGINAL_FORWARD_IMPL = FidelityResidualNetV9._forward_impl
    _ORIGINAL_SEAM_FORWARD = DirectionalSeamRestorer.forward
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract

    DirectionalSeamRestorer.forward = _seam_forward_from_baseline
    FidelityResidualNetV9._selector_features = _fusion_selector_features
    FidelityResidualNetV9._forward_impl = _forward_with_parallel_specialist_fusion
    FidelityResidualNetV9.architecture_contract = (
        _architecture_contract_parallel_specialists
    )
    _INSTALLED = True
