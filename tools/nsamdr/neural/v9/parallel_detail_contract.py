from __future__ import annotations

"""V12.3 parallel direct-detail production contract.

The direct Raven capacity proof demonstrated that GeometryConditionedDetailNet can
recover the hard Raven patch when it is anchored directly to deterministic baseline
B, while the staged production graph collapses after feeding the detail branch bad
geometry/seam prerequisites. This contract preserves the existing production module
and checkpoint topology but changes its composition semantics:

* detail_net always predicts a bounded residual over deterministic baseline B;
* upstream geometry/seam results cannot alter the detail candidate it produces;
* BenefitSelector features depend on B, the direct-detail candidate, observable LR
  evidence and the detail confidence/regret heads, not learned geometry/seam state;
* geometry and seam candidates remain available for diagnostics/qualification, but
  they do not poison the proven direct-detail candidate;
* detail-reconstruction uses the same baseline-relative objective and approximately
  the same learning rate that actually passed the direct capacity proof.

No new trainable parameters are introduced, so existing state_dict compatibility is
preserved. The model-side contract is installed package-wide for training, preflight
and inference. All installed functions are module-level for Windows spawn safety.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9, GeometryConditionedDetailNet, UPSCALE_FACTOR


PARALLEL_DETAIL_REVISION = "V12.3"
_MODEL_INSTALLED = False
_BACKEND_INSTALLED = False
_ORIGINAL_DETAIL_FORWARD: Any = None
_ORIGINAL_SELECTOR_FEATURES: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None
_ORIGINAL_MODEL_INIT: Any = None
_ORIGINAL_PRODUCTION_HEAD_LOSS: Any = None
_ORIGINAL_PHASE_LR: Any = None


def _normalize_xy(value: torch.Tensor) -> torch.Tensor:
    length = torch.sqrt(value.float().square().sum(dim=1, keepdim=True) + 1.0e-6)
    limiter = torch.maximum(torch.ones_like(length), length / 0.999)
    return (value.float() / limiter).to(value.dtype)


def _deterministic_baseline(
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_albedo = inputs[:, 0:3].clamp(0.0, 1.0)
    source_normal = inputs[:, 3:5].clamp(-1.0, 1.0)
    source_material = inputs[:, 5:8].clamp(0.0, 1.0)
    albedo = F.interpolate(
        source_albedo,
        scale_factor=UPSCALE_FACTOR,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    normal = _normalize_xy(
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


def _model_init_with_parallel_caps(
    self: FidelityResidualNetV9,
    *args: Any,
    **kwargs: Any,
) -> None:
    if _ORIGINAL_MODEL_INIT is None:
        raise RuntimeError("parallel detail contract installed without original model init")
    _ORIGINAL_MODEL_INIT(self, *args, **kwargs)
    self.detail_net._parallel_albedo_cap = float(
        getattr(self.config, "detail_albedo_max_delta", 0.20)
    )
    self.detail_net._parallel_normal_cap = float(
        getattr(self.config, "detail_normal_max_delta", 0.15)
    )
    self.detail_net._parallel_material_cap = float(
        getattr(self.config, "detail_material_max_delta", 0.18)
    )


def _detail_forward_from_baseline(
    self: GeometryConditionedDetailNet,
    inputs: torch.Tensor,
    base_albedo_hr: torch.Tensor,
    base_normal_hr: torch.Tensor,
    base_material_hr: torch.Tensor,
    geometry_condition_hr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return residuals that make the caller's candidate equal B + direct residual.

    The historical parent forward adds ``detail_raw * cap`` to the map supplied to
    detail_net. Rather than duplicating the decoder or changing checkpoint topology,
    this adapter runs the exact direct branch over deterministic B and analytically
    compensates that supplied upstream base. The parent therefore reconstructs the
    same independent B+residual candidate that passed the capacity proof.
    """
    del geometry_condition_hr
    if _ORIGINAL_DETAIL_FORWARD is None:
        raise RuntimeError("parallel detail contract installed without original detail forward")

    baseline_albedo, baseline_normal, baseline_material = _deterministic_baseline(inputs)
    neutral_geometry = torch.zeros(
        (
            inputs.shape[0],
            self.GEOMETRY_CHANNELS,
            baseline_albedo.shape[-2],
            baseline_albedo.shape[-1],
        ),
        device=inputs.device,
        dtype=baseline_albedo.dtype,
    )
    direct = _ORIGINAL_DETAIL_FORWARD(
        self,
        inputs,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        neutral_geometry,
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


def _parallel_selector_features(
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
    """Build selector evidence without any learned geometry/seam prerequisite."""
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
    if detail_confidence is None:
        detail_confidence = torch.zeros_like(observed_support)
    if detail_regret is None:
        detail_regret = torch.zeros_like(observed_support)
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
            detail_confidence.float().clamp(0.0, 1.0),
            detail_regret.float().clamp(0.0, 1.0),
        ),
        dim=1,
    )


def _architecture_contract_parallel_detail(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("parallel detail contract installed without architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelDetailRevision"] = PARALLEL_DETAIL_REVISION
    contract["detailAuthority"] = (
        "GeometryConditionedDetailNet predicts a bounded direct residual over deterministic B; "
        "geometry/seam outputs cannot alter the detail candidate"
    )
    contract["selectorRequires"] = (
        "deterministic B + direct-detail candidate + observable LR edge support + detail "
        "confidence/regret; no learned geometry or seam prerequisite"
    )
    contract["candidateAuthority"] = (
        "one BenefitSelector blend between deterministic B and the independently reconstructed "
        "direct-detail candidate"
    )
    contract["specialistComposition"] = (
        "geometry and seam remain independently auditable specialists; the proven direct-detail "
        "candidate is isolated from them until each specialist independently qualifies"
    )
    contract["detailGeometryConditioningAuthority"] = False
    contract["detailBase"] = "deterministic bicubic/bilinear/nearest B"
    return contract


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _gradient_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from .contours import sobel_tensor

    prediction_gray = prediction.float().mean(dim=1, keepdim=True)
    target_gray = target.float().mean(dim=1, keepdim=True)
    pgx, pgy = sobel_tensor(prediction_gray)
    tgx, tgy = sobel_tensor(target_gray)
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), weight)


def _compute_losses_parallel_detail(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    """Make production detail training match the direct proof that actually passed."""
    if _ORIGINAL_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError("parallel detail contract installed without production-head loss")
    losses = _ORIGINAL_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase != "detail-reconstruction":
        return losses

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["detail_candidate_albedo"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    edge_weight = (0.20 + edge * 3.80).detach()

    error = (candidate - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    global_reconstruction = error.mean()
    edge_reconstruction = _weighted_mean(error, edge_weight)
    regret = _weighted_mean(F.relu(error - baseline_error), edge_weight)
    gradient = _gradient_error(candidate, target, edge_weight)

    cap = float(getattr(config, "detail_albedo_max_delta", 0.20))
    desired_residual = (target.detach() - baseline).clamp(-cap, cap)
    actual_residual = candidate - baseline
    residual_supervision = (actual_residual - desired_residual).abs().mean()

    losses["detail_parallel_global_reconstruction"] = global_reconstruction
    losses["detail_parallel_edge_reconstruction"] = edge_reconstruction
    losses["detail_parallel_regret"] = regret
    losses["detail_parallel_gradient"] = gradient
    losses["detail_parallel_residual_supervision"] = residual_supervision
    losses["detail_parallel_global_recovery"] = (
        (baseline_error.mean() - error.mean()) / baseline_error.mean().clamp_min(1.0e-6)
    ).detach()
    losses["detail_parallel_edge_recovery"] = (
        (_weighted_mean(baseline_error, edge_weight) - edge_reconstruction)
        / _weighted_mean(baseline_error, edge_weight).clamp_min(1.0e-6)
    ).detach()

    # Exact objective from the successful direct-capacity proof. Inherited serial
    # losses remain telemetry only and cannot pull this branch toward bad upstream
    # geometry/seam candidates.
    losses["total"] = (
        global_reconstruction * 4.0
        + edge_reconstruction * 12.0
        + gradient * 5.0
        + regret * 16.0
        + residual_supervision * 8.0
    ).float()
    return losses


def _phase_lr_parallel_detail(
    phase: str,
    config: Any,
    epoch: int | None = None,
) -> float:
    """Use the learning-rate regime that proved direct-detail capacity."""
    if _ORIGINAL_PHASE_LR is None:
        raise RuntimeError("parallel detail contract installed without production phase LR")
    value = float(_ORIGINAL_PHASE_LR(phase, config, epoch))
    if phase != "detail-reconstruction":
        return value
    # Existing adapter gives detail 3x 6e-5 ~= 1.8e-4. The isolated proof passed
    # at 1e-3. Bring production detail close to that demonstrated regime without
    # changing any other phase.
    return max(value, float(getattr(config, "detail_learning_rate", 6.0e-5)) * 16.0)


def install_parallel_detail_model_contract() -> None:
    global _MODEL_INSTALLED
    global _ORIGINAL_DETAIL_FORWARD, _ORIGINAL_SELECTOR_FEATURES
    global _ORIGINAL_ARCHITECTURE_CONTRACT, _ORIGINAL_MODEL_INIT
    if _MODEL_INSTALLED:
        return
    _ORIGINAL_DETAIL_FORWARD = GeometryConditionedDetailNet.forward
    _ORIGINAL_SELECTOR_FEATURES = FidelityResidualNetV9._selector_features
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    _ORIGINAL_MODEL_INIT = FidelityResidualNetV9.__init__
    GeometryConditionedDetailNet.forward = _detail_forward_from_baseline
    FidelityResidualNetV9._selector_features = _parallel_selector_features
    FidelityResidualNetV9.architecture_contract = _architecture_contract_parallel_detail
    FidelityResidualNetV9.__init__ = _model_init_with_parallel_caps
    _MODEL_INSTALLED = True


def install_parallel_detail_backend_contract() -> None:
    """Patch TrainingBackend globals before it installs callbacks on TrainingService."""
    global _BACKEND_INSTALLED, _ORIGINAL_PRODUCTION_HEAD_LOSS, _ORIGINAL_PHASE_LR
    if _BACKEND_INSTALLED:
        return
    from .application import backend

    current_loss = backend._compute_losses_with_production_head_supervision
    if current_loss is not _compute_losses_parallel_detail:
        _ORIGINAL_PRODUCTION_HEAD_LOSS = current_loss
        backend._compute_losses_with_production_head_supervision = _compute_losses_parallel_detail

    current_lr = backend._phase_lr_with_head_capacity
    if current_lr is not _phase_lr_parallel_detail:
        _ORIGINAL_PHASE_LR = current_lr
        backend._phase_lr_with_head_capacity = _phase_lr_parallel_detail
    _BACKEND_INSTALLED = True


def install_parallel_detail_contract() -> None:
    install_parallel_detail_model_contract()
    install_parallel_detail_backend_contract()
