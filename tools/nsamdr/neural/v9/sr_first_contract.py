from __future__ import annotations

"""V13 SR-first production authority for EVE texture reconstruction.

The V12 qualification work proved that the existing 4x detail decoder can recover
substantially more Raven detail than the geometry/profile/seam redraw stack. V13
therefore makes learned multi-map super-resolution the primary candidate:

    C = B + bounded SR residual
    F = B + BenefitSelector * (C - B)

Geometry is retained as observable LR-derived conditioning and validation evidence,
not as pixel authority. Learned G/profile/seam candidates remain available in the
forward dictionary for diagnostics but cannot alter C or F.

No trainable parameters, checkpoint keys or inference inputs are added. Existing
GeometryConditionedDetailNet weights remain load-compatible.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9, GeometryConditionedDetailNet
from . import parallel_detail_contract as v123


SR_FIRST_REVISION = "V13.0"
SR_LAPLACIAN_WEIGHT = 3.0
SR_NORMAL_GLOBAL_WEIGHT = 3.0
SR_NORMAL_EDGE_WEIGHT = 4.0
SR_MATERIAL_GLOBAL_WEIGHT = 3.0
SR_MATERIAL_EDGE_WEIGHT = 4.0

_INSTALLED = False
_PREVIOUS_DETAIL_FORWARD: Any = None
_PREVIOUS_FORWARD_IMPL: Any = None
_PREVIOUS_ARCHITECTURE_CONTRACT: Any = None
_PREVIOUS_PRODUCTION_HEAD_LOSS: Any = None


def _observable_sr_condition(
    inputs: torch.Tensor,
    hr_size: tuple[int, int],
) -> torch.Tensor:
    """Build six LR-observable structure channels with no learned G/P/S authority."""
    if inputs.ndim != 4 or inputs.shape[1] < 17:
        raise ValueError(f"V13 expects Bx17xHxW model input, got {tuple(inputs.shape)}")
    condition_lr = torch.cat(
        (
            inputs[:, 16:17].float().clamp(-1.0, 1.0),
            inputs[:, 9:10].float().clamp(-1.0, 1.0),
            inputs[:, 10:11].float().clamp(-1.0, 1.0),
            inputs[:, 11:12].float().abs().clamp(0.0, 1.0),
            inputs[:, 12:13].float().abs().clamp(0.0, 1.0),
            inputs[:, 13:14].float().clamp(-1.0, 1.0),
        ),
        dim=1,
    )
    return F.interpolate(
        condition_lr,
        size=hr_size,
        mode="bilinear",
        align_corners=False,
    )


def _sr_detail_forward(
    self: GeometryConditionedDetailNet,
    inputs: torch.Tensor,
    base_albedo_hr: torch.Tensor,
    base_normal_hr: torch.Tensor,
    base_material_hr: torch.Tensor,
    geometry_condition_hr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Predict the full SR residual from deterministic B + observable LR evidence."""
    del geometry_condition_hr
    direct_forward = v123._ORIGINAL_DETAIL_FORWARD
    if direct_forward is None:
        raise RuntimeError("V13 requires the captured V12.3/V12.7 detail decoder")

    baseline_albedo, baseline_normal, baseline_material = v123._deterministic_baseline(inputs)
    condition = _observable_sr_condition(inputs, baseline_albedo.shape[-2:])
    direct = direct_forward(
        self,
        inputs,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        condition,
    )

    albedo_cap = float(getattr(self, "_parallel_albedo_cap", 0.20))
    normal_cap = float(getattr(self, "_parallel_normal_cap", 0.15))
    material_cap = float(getattr(self, "_parallel_material_cap", 0.18))

    desired_albedo = (
        baseline_albedo.float() + direct["albedo_raw"].float() * albedo_cap
    ).clamp(0.0, 1.0)
    desired_normal_pre = baseline_normal.float() + direct["normal_raw"].float() * normal_cap
    desired_material = (
        baseline_material.float() + direct["material_raw"].float() * material_cap
    ).clamp(0.0, 1.0)

    result = dict(direct)
    result["albedo_raw"] = (
        (desired_albedo - base_albedo_hr.detach().float()) / max(albedo_cap, 1.0e-6)
    ).to(direct["albedo_raw"].dtype)
    result["normal_raw"] = (
        (desired_normal_pre - base_normal_hr.detach().float()) / max(normal_cap, 1.0e-6)
    ).to(direct["normal_raw"].dtype)
    result["material_raw"] = (
        (desired_material - base_material_hr.detach().float()) / max(material_cap, 1.0e-6)
    ).to(direct["material_raw"].dtype)
    return result


def _selector_observed_support(
    self: FidelityResidualNetV9,
    inputs: torch.Tensor,
    hr_size: tuple[int, int],
) -> torch.Tensor:
    support = self._source_edge_support(
        inputs, int(self.config.geometry_edge_support_radius)
    )
    return F.interpolate(
        support,
        size=hr_size,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def _forward_sr_first(
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
    """Run legacy specialists for evidence, but deploy only B -> SR C -> selector F."""
    if _PREVIOUS_FORWARD_IMPL is None:
        raise RuntimeError("V13 installed without previous production forward")

    outputs = _PREVIOUS_FORWARD_IMPL(
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
    candidate_albedo = outputs["detail_candidate_albedo"]
    candidate_normal = outputs["detail_candidate_normal"]
    candidate_material = outputs["detail_candidate_material"]

    observed_support = _selector_observed_support(
        self, inputs, candidate_albedo.shape[-2:]
    )
    zeros = torch.zeros_like(observed_support)
    zeros2 = torch.zeros(
        (
            inputs.shape[0],
            2,
            candidate_albedo.shape[-2],
            candidate_albedo.shape[-1],
        ),
        device=inputs.device,
        dtype=candidate_albedo.dtype,
    )
    detail_confidence = outputs["detail_confidence"].float().clamp(0.0, 1.0)
    detail_regret = outputs["detail_regret"].float().clamp(0.0, 1.0)

    selector_features = self._selector_features(
        baseline_albedo,
        candidate_albedo,
        zeros,
        zeros2,
        zeros,
        zeros,
        observed_support,
        zeros,
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
    detail_support = (
        detail_confidence * (1.0 - detail_regret)
    ).clamp(0.0, 1.0)

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
            "benefit_selector_logits": selector_logits.to(albedo.dtype),
            "benefit_selector_probability": selector_probability.to(albedo.dtype),
            "final_selector_gate": final_gate.to(albedo.dtype),
            "boundary_gate_logits": selector_logits.to(albedo.dtype),
            "boundary_gate_prediction": selector_probability.to(albedo.dtype),
            "boundary_gate_probability": selector_probability.to(albedo.dtype),
            "fused_candidate_albedo": candidate_albedo,
            "fused_candidate_normal": candidate_normal,
            "fused_candidate_material": candidate_material,
            "sr_candidate_albedo": candidate_albedo,
            "sr_candidate_normal": candidate_normal,
            "sr_candidate_material": candidate_material,
            "sr_observable_condition": _observable_sr_condition(
                inputs, candidate_albedo.shape[-2:]
            ).to(albedo.dtype),
            "parallel_structure_fusion_weight": torch.zeros_like(detail_support).to(albedo.dtype),
            "parallel_seam_fusion_weight": torch.zeros_like(detail_support).to(albedo.dtype),
            "parallel_detail_fusion_weight": torch.ones_like(detail_support).to(albedo.dtype),
            "parallel_fusion_support": detail_support.to(albedo.dtype),
            "parallel_fusion_conflict": torch.zeros_like(detail_support).to(albedo.dtype),
        }
    )
    return outputs


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if w.shape[1] == 1 and value.shape[1] != 1:
        w = w.expand(-1, value.shape[1], -1, -1)
    return (value.float() * w.float()).sum() / w.float().sum().clamp_min(1.0)


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    channels = int(value.shape[1])
    kernel = value.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    return F.conv2d(
        value.float(),
        kernel.repeat(channels, 1, 1, 1),
        padding=1,
        groups=channels,
    )


def _target_material(
    batch: dict[str, torch.Tensor],
    config: Any,
) -> torch.Tensor:
    material_class = batch["target_material_class"].float()
    if material_class.ndim == 3:
        material_class = material_class.unsqueeze(1)
    emissive = batch["target_emissive"].float()
    if emissive.ndim == 3:
        emissive = emissive.unsqueeze(1)
    roughness = batch["target_roughness"].float()
    if roughness.ndim == 3:
        roughness = roughness.unsqueeze(1)
    scalar = (material_class + 0.5) / float(max(int(config.material_classes), 1))
    return torch.cat((scalar, emissive, roughness), dim=1)


def _loss_with_sr_first_training(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    """Replace D reconstruction with a direct visual multi-map SR objective."""
    if _PREVIOUS_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError("V13 installed without previous protected loss chain")
    losses = _PREVIOUS_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase != "detail-reconstruction":
        return losses

    from .contours import sobel_tensor

    target_albedo = batch["target_albedo"].float()
    target_normal = batch["target_normal"].float()
    target_material = _target_material(batch, config)
    baseline_albedo = outputs["baseline_albedo"].detach().float()
    candidate_albedo = outputs["detail_candidate_albedo"].float()
    candidate_normal = outputs["detail_candidate_normal"].float()
    candidate_material = outputs["detail_candidate_material"].float()
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
    gradient = _weighted_mean(
        (cgx - tgx).abs() + (cgy - tgy).abs(),
        edge_weight,
    )
    laplacian = _weighted_mean(
        (_laplacian(candidate_gray) - _laplacian(target_gray)).abs(),
        edge_weight,
    )

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

    support = losses.get("detail_support_calibration")
    if not isinstance(support, torch.Tensor):
        support = albedo_global.new_zeros(())
    protected = losses.get("detail_protected_safety_penalty")
    if not isinstance(protected, torch.Tensor):
        protected = albedo_global.new_zeros(())

    losses["sr_albedo_global"] = albedo_global
    losses["sr_albedo_edge"] = albedo_edge
    losses["sr_gradient"] = gradient
    losses["sr_laplacian"] = laplacian
    losses["sr_regret"] = regret
    losses["sr_residual_supervision"] = residual_supervision
    losses["sr_normal_global"] = normal_global
    losses["sr_normal_edge"] = normal_edge
    losses["sr_material_global"] = material_global
    losses["sr_material_edge"] = material_edge
    losses["sr_edge_recovery"] = (
        (_weighted_mean(baseline_error, edge_weight) - albedo_edge)
        / _weighted_mean(baseline_error, edge_weight).clamp_min(1.0e-6)
    ).detach()
    losses["sr_global_recovery"] = (
        (baseline_error.mean() - albedo_error.mean())
        / baseline_error.mean().clamp_min(1.0e-6)
    ).detach()
    losses["total"] = (
        albedo_global * 4.0
        + albedo_edge * 12.0
        + gradient * 5.0
        + laplacian * float(SR_LAPLACIAN_WEIGHT)
        + regret * 16.0
        + residual_supervision * 8.0
        + normal_global * float(SR_NORMAL_GLOBAL_WEIGHT)
        + normal_edge * float(SR_NORMAL_EDGE_WEIGHT)
        + material_global * float(SR_MATERIAL_GLOBAL_WEIGHT)
        + material_edge * float(SR_MATERIAL_EDGE_WEIGHT)
        + support
        + protected * 96.0
    ).float()
    return losses


def _architecture_contract_sr_first(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _PREVIOUS_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V13 installed without architecture contract")
    contract = dict(_PREVIOUS_ARCHITECTURE_CONTRACT(self))
    contract["srFirstRevision"] = SR_FIRST_REVISION
    contract["productionReconstructionAuthority"] = (
        "deterministic B + bounded multi-map GeometryConditionedDetailNet SR residual"
    )
    contract["srConditioning"] = (
        "LR-observable contour SDF, luma gradients, normal/material edges and curvature only; "
        "learned G/profile/seam cannot alter the SR candidate"
    )
    contract["specialistComposition"] = {
        "candidate": "C = B + bounded SR residual",
        "finalAuthority": "F = B + BenefitSelector * (C - B)",
        "legacyStructureSeam": "diagnostic/regularisation evidence only; zero production pixel authority",
    }
    contract["geometryPixelAuthority"] = False
    contract["profilePixelAuthority"] = False
    contract["seamPixelAuthority"] = False
    contract["detailGeometryConditioningAuthority"] = "observable-LR-only"
    contract["finalCandidate"] = "sr_candidate"
    return contract


def install_sr_first_contract() -> None:
    """Install V13 after V12.10 without changing parameter/checkpoint topology."""
    global _INSTALLED
    global _PREVIOUS_DETAIL_FORWARD, _PREVIOUS_FORWARD_IMPL
    global _PREVIOUS_ARCHITECTURE_CONTRACT, _PREVIOUS_PRODUCTION_HEAD_LOSS
    if _INSTALLED:
        return

    from . import baseline_relative_specialist_contract as preservation
    from .application import backend

    _PREVIOUS_DETAIL_FORWARD = GeometryConditionedDetailNet.forward
    GeometryConditionedDetailNet.forward = _sr_detail_forward

    _PREVIOUS_FORWARD_IMPL = FidelityResidualNetV9._forward_impl
    FidelityResidualNetV9._forward_impl = _forward_sr_first

    _PREVIOUS_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.architecture_contract = _architecture_contract_sr_first

    current_inner = preservation._ORIGINAL_PRODUCTION_HEAD_LOSS
    if current_inner is None:
        raise RuntimeError("V13 requires the V12.4 protected loss chain")
    _PREVIOUS_PRODUCTION_HEAD_LOSS = current_inner
    preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = _loss_with_sr_first_training
    backend._compute_losses_with_production_head_supervision = (
        preservation._loss_with_protected_preservation
    )

    _INSTALLED = True
