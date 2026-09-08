from __future__ import annotations

"""V12.2.3 B1b objective alignment.

B1 topology is established before sdf-proof. During B1b the only question is
whether the refined production geometry renders a better Raven result than B.
The legacy spline/proxy objective is therefore retained as telemetry only: it
must not be able to trade improved proxy point/tangent scores for a worse
rendered candidate.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .contours import sobel_tensor


B1_PRODUCTION_OBJECTIVE_REVISION = "V12.2.3"
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
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    prediction_gray = prediction.float().mean(dim=1, keepdim=True)
    target_gray = target.float().mean(dim=1, keepdim=True)
    pgx, pgy = sobel_tensor(prediction_gray)
    tgx, tgy = sobel_tensor(target_gray)
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), edge_weight)


def _live_production_objective(
    outputs: Any,
    batch: Any,
    config: Any,
) -> torch.Tensor:
    """Rebuild the exact B1 production objective without detaching its graph."""
    required = ("boundary_initial_candidate_albedo", "baseline_albedo")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(f"sdf-proof production supervision missing outputs: {missing}")

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["boundary_initial_candidate_albedo"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    edge_weight = (0.20 + edge * 3.80).detach()

    global_reconstruction = candidate_error.mean()
    edge_reconstruction = _weighted_mean(candidate_error, edge_weight)
    regret = _weighted_mean(F.relu(candidate_error - baseline_error), edge_weight)
    gradient = _edge_gradient_error(candidate, target, edge_weight)

    render_weight = float(getattr(config, "spline_graph_render_weight", 96.0))
    gradient_weight = float(getattr(config, "spline_graph_render_gradient_weight", 48.0))
    return (
        global_reconstruction * render_weight * 0.35
        + edge_reconstruction * render_weight * 0.65
        + regret * render_weight
        + gradient * gradient_weight
    ).float()


def _production_only_b1b_loss(
    outputs: Any,
    batch: Any,
    config: Any,
    phase: str,
) -> dict[str, Any]:
    """Use the exact rendered B1 candidate objective as 100% of B1b SGD authority."""
    if _ORIGINAL_B1B_LOSS is None:
        raise RuntimeError("B1 production objective installed without original B1b loss")

    losses = _ORIGINAL_B1B_LOSS(outputs, batch, config, phase)
    if phase != "sdf-proof":
        return losses

    # backend.b1b_renderer_supervision is intentionally detached telemetry. Do not
    # reuse it for backward. Rebuild the same scalar from the live candidate graph.
    production = _live_production_objective(outputs, batch, config)
    if not production.requires_grad or production.grad_fn is None:
        raise RuntimeError(
            "sdf-proof production objective is detached from the B1 geometry graph"
        )

    combined = losses["total"].float()
    telemetry = losses.get("b1b_renderer_supervision")
    if isinstance(telemetry, torch.Tensor):
        losses["b1b_renderer_supervision"] = telemetry.detach()
    else:
        losses["b1b_renderer_supervision"] = production.detach()

    # Keep visibility into how strongly the retired proxy objective disagrees,
    # but give it zero gradient authority. B1b now succeeds or fails solely on
    # the exact rendered candidate that Micro/Quick qualification scores.
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
