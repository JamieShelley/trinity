from __future__ import annotations

"""V13.3 compatibility for the canonical trainer's legacy progress telemetry.

The SR-only loss deliberately removed historical geometry/SDF loss terms, while the
unchanged canonical trainer still formats a small set of legacy metric names after
an epoch.  Those names are display/telemetry only; training authority remains the
V13.3 SR loss.  This adapter supplies truthful SR equivalents (or explicit zeros for
retired structural metrics) so progress reporting cannot abort an otherwise valid
SR epoch.
"""

from typing import Any

import torch


SR_TRAINING_TELEMETRY_REVISION = "V13.3"
_INSTALLED = False
_PREVIOUS_LOSS: Any = None


def _add_v133_trainer_metrics(
    losses: dict[str, torch.Tensor],
    phase: str,
) -> dict[str, torch.Tensor]:
    result = dict(losses)
    total = result.get("total")
    if not isinstance(total, torch.Tensor):
        raise RuntimeError("V13.3 SR loss returned no tensor total")
    zero = total.detach() * 0.0

    if phase == "detail-reconstruction":
        albedo = result.get("sr_albedo_global", zero)
        regret = result.get("sr_regret", zero)
    elif phase in {"physical-finetune", "boundary-hardening"}:
        albedo = result.get("selector_edge_reconstruction", zero)
        regret = result.get("selector_edge_regret", zero)
    else:
        albedo = zero
        regret = zero

    # These five names are directly indexed by the historical trainer's progress
    # formatter.  SDF/tangent/curvature are retired in V13.3 and therefore report
    # explicit zero rather than resurrecting any training objective or authority.
    result.setdefault("sdf", zero)
    result.setdefault("albedo", albedo)
    result.setdefault("regret", regret)
    result.setdefault("tangent_coherence", zero)
    result.setdefault("curvature_coherence", zero)
    return result


def _loss_with_v133_trainer_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _PREVIOUS_LOSS is None:
        raise RuntimeError("V13.3 trainer telemetry installed without active SR loss")
    return _add_v133_trainer_metrics(
        _PREVIOUS_LOSS(outputs, batch, config, phase),
        phase,
    )


def install_sr_first_training_telemetry_contract() -> None:
    global _INSTALLED, _PREVIOUS_LOSS
    if _INSTALLED:
        return

    from .application import backend

    _PREVIOUS_LOSS = backend._compute_losses_with_production_head_supervision
    backend._compute_losses_with_production_head_supervision = _loss_with_v133_trainer_metrics
    _INSTALLED = True
