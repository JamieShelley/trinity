from __future__ import annotations

"""V12.2.4 B1b objective alignment.

B1 topology is established before sdf-proof. During B1b the geometry head is
allowed to improve only contours that are actually observable from the LR
source topology. Fine panel lines and other HR-only appearance detail belong to
the later seam/detail heads, not to the connected-spline geometry stage.

The legacy spline/proxy objective remains telemetry-only. B1b SGD is driven by
the exact rendered candidate, but only inside source/target structural support;
outside that support the candidate is regularised back to deterministic B.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .contours import sobel_tensor


B1_PRODUCTION_OBJECTIVE_REVISION = "V12.2.4"
_INSTALLED = False
_ORIGINAL_B1B_LOSS: Any = None


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _edge_gradient_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    prediction_gray = prediction.float().mean(dim=1, keepdim=True)
    target_gray = target.float().mean(dim=1, keepdim=True)
    pgx, pgy = sobel_tensor(prediction_gray)
    tgx, tgy = sobel_tensor(target_gray)
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), weight)


def _observable_structural_support(
    outputs: Any,
    batch: Any,
    config: Any,
) -> torch.Tensor:
    """Return the target-contour support that B1 can physically actuate.

    B1b locks topology. A target edge that has no nearby LR/source contour cannot
    be created by moving the existing connected spline graph; asking B1 to match
    those HR-only lines forces geometry to corrupt otherwise-correct baseline
    pixels. Those residual details are intentionally left for seam/detail stages.
    """
    target_sdf = batch.get("target_sdf")
    source_pixels = outputs.get("source_sdf_prior_pixels")
    if not isinstance(target_sdf, torch.Tensor) or not isinstance(source_pixels, torch.Tensor):
        raise RuntimeError(
            "sdf-proof structural support requires target_sdf and source_sdf_prior_pixels"
        )

    max_distance = float(getattr(config, "contour_sdf_max_distance_pixels", 24.0))
    band = max(float(getattr(config, "sdf_metric_band_pixels", 6.0)), 1.0)
    target_pixels = target_sdf.detach().float() * max_distance
    source_pixels = source_pixels.detach().float()
    if source_pixels.shape[-2:] != target_pixels.shape[-2:]:
        source_pixels = F.interpolate(
            source_pixels,
            size=target_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # Source topology defines what B1 can move. Give a generous source band so a
    # contour may shift several pixels, then concentrate authority toward the
    # authored target zero-set inside that observable neighbourhood.
    source_observable = (source_pixels.abs() <= band * 1.5).float()
    target_proximity = torch.exp(-target_pixels.abs() / max(band * 0.55, 1.0e-3))
    support = (source_observable * target_proximity).detach().clamp(0.0, 1.0)
    if float(support.sum().detach().cpu()) < 1.0:
        raise RuntimeError("sdf-proof patch contains no observable structural contour support")
    return support


def _live_production_objective(
    outputs: Any,
    batch: Any,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the live, deployment-aligned B1 objective without detaching its graph."""
    required = ("boundary_initial_candidate_albedo", "baseline_albedo")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(f"sdf-proof production supervision missing outputs: {missing}")

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["boundary_initial_candidate_albedo"].float()
    support = _observable_structural_support(outputs, batch, config)

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    candidate_structural = _weighted_mean(candidate_error, support)
    baseline_structural = _weighted_mean(baseline_error, support)
    regret = _weighted_mean(F.relu(candidate_error - baseline_error), support)
    gradient = _edge_gradient_error(candidate, target, support)

    # Away from observable source topology, B1 has no business repainting the
    # raster. Preserve B exactly and leave HR-only texture/panel detail to D.
    off_support = (1.0 - support).detach()
    identity = _weighted_mean(
        (candidate - baseline).abs().mean(dim=1, keepdim=True),
        off_support,
    )

    render_weight = float(getattr(config, "spline_graph_render_weight", 96.0))
    gradient_weight = float(getattr(config, "spline_graph_render_gradient_weight", 48.0))
    production = (
        candidate_structural * render_weight
        + regret * render_weight
        + gradient * gradient_weight
        + identity * render_weight * 0.50
    ).float()
    telemetry = {
        "b1b_structural_support_mean": support.mean().detach(),
        "b1b_structural_reconstruction": candidate_structural.detach(),
        "b1b_structural_baseline": baseline_structural.detach(),
        "b1b_structural_recovery": (
            (baseline_structural - candidate_structural)
            / baseline_structural.clamp_min(1.0e-6)
        ).detach(),
        "b1b_structural_regret": regret.detach(),
        "b1b_structural_gradient": gradient.detach(),
        "b1b_off_support_identity": identity.detach(),
    }
    return production, telemetry


def _production_only_b1b_loss(
    outputs: Any,
    batch: Any,
    config: Any,
    phase: str,
) -> dict[str, Any]:
    """Use observable rendered geometry as 100% of B1b SGD authority."""
    if _ORIGINAL_B1B_LOSS is None:
        raise RuntimeError("B1 production objective installed without original B1b loss")

    losses = _ORIGINAL_B1B_LOSS(outputs, batch, config, phase)
    if phase != "sdf-proof":
        return losses

    production, telemetry = _live_production_objective(outputs, batch, config)
    if not production.requires_grad or production.grad_fn is None:
        raise RuntimeError(
            "sdf-proof production objective is detached from the B1 geometry graph"
        )

    combined = losses["total"].float()
    losses.update(telemetry)
    # Keep the broad all-pixel objective visible only as historical telemetry;
    # using it for SGD was proven to punish B1 for HR-only appearance detail.
    old_renderer = losses.get("b1b_renderer_supervision")
    if isinstance(old_renderer, torch.Tensor):
        losses["b1b_renderer_supervision"] = old_renderer.detach()
    losses["b1b_combined_objective_before_alignment"] = combined.detach()
    losses["b1b_nonproduction_objective_ignored"] = (
        combined.detach() - production.detach()
    )
    losses["b1b_production_objective_requires_grad"] = production.new_tensor(1.0).detach()
    losses["total"] = production
    return losses


def install_b1_production_objective_contract() -> None:
    """Patch TrainingBackend's B1b adapter before any backend instance installs it."""
    global _INSTALLED, _ORIGINAL_B1B_LOSS
    if _INSTALLED:
        return

    from .application import backend

    current = backend._compute_losses_with_b1b_renderer_supervision
    if current is not _production_only_b1b_loss:
        _ORIGINAL_B1B_LOSS = current
        backend._compute_losses_with_b1b_renderer_supervision = _production_only_b1b_loss
    _INSTALLED = True
