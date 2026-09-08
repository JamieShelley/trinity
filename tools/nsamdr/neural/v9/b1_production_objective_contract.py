from __future__ import annotations

"""V12.2.2 B1b objective alignment.

B1 topology is established before sdf-proof. During B1b the only question is
whether the refined production geometry renders a better Raven result than B.
The legacy spline/proxy objective is therefore retained as telemetry only: it
must not be able to trade improved proxy point/tangent scores for a worse
rendered candidate.
"""

from typing import Any

import torch


B1_PRODUCTION_OBJECTIVE_REVISION = "V12.2.2"
_INSTALLED = False
_ORIGINAL_B1B_LOSS: Any = None


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

    production = losses.get("b1b_renderer_supervision")
    if not isinstance(production, torch.Tensor):
        raise RuntimeError("sdf-proof missing b1b_renderer_supervision production objective")

    combined = losses["total"].float()
    production = production.float()
    # Keep visibility into how strongly the retired proxy objective disagrees,
    # but give it zero gradient authority. B1b now succeeds or fails solely on
    # the exact rendered candidate that Micro/Quick qualification scores.
    losses["b1b_combined_objective_before_alignment"] = combined.detach()
    losses["b1b_nonproduction_objective_ignored"] = (combined - production).detach()
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
