from __future__ import annotations

"""V13.3 compatibility for the canonical trainer's legacy progress telemetry.

The SR-only loss deliberately removed historical geometry/SDF loss terms, while the
canonical trainer still consumes a small compatibility metric surface for progress,
checkpoint ranking and stage gates.  This adapter supplies those names from the
active B -> C -> F reconstruction itself.  Retired structural metrics are explicit
zeros; no retired loss or pixel authority is restored.
"""

from typing import Any

import torch

from .contours import sobel_tensor


SR_TRAINING_TELEMETRY_REVISION = "V13.3"
_INSTALLED = False
_PREVIOUS_LOSS: Any = None


def _add_v133_trainer_metrics(
    losses: dict[str, torch.Tensor],
    phase: str,
) -> dict[str, torch.Tensor]:
    """Supply only formatter compatibility keys that do not require batch outputs."""
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

    # These names are directly indexed by the historical trainer's progress
    # formatter. SDF/tangent/curvature are retired in V13.3 and therefore report
    # explicit zero rather than resurrecting any training objective or authority.
    result.setdefault("sdf", zero)
    result.setdefault("albedo", albedo)
    result.setdefault("regret", regret)
    result.setdefault("tangent_coherence", zero)
    result.setdefault("curvature_coherence", zero)
    return result


def _sample_l1(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one global reconstruction error per validation sample."""
    return (value.float() - target.float()).abs().flatten(1).mean(dim=1)


def _sample_gradient_l1(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one Sobel-domain reconstruction error per validation sample."""
    value_gray = value.float().mean(dim=1, keepdim=True)
    target_gray = target.float().mean(dim=1, keepdim=True)
    vx, vy = sobel_tensor(value_gray)
    tx, ty = sobel_tensor(target_gray)
    return ((vx - tx).abs() + (vy - ty).abs()).flatten(1).mean(dim=1)


def _reconstruction_outcome_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    phase: str,
) -> dict[str, torch.Tensor]:
    """Measure the active candidate/final against B for legacy stage-gate names."""
    baseline = outputs["baseline_albedo"].detach().float()
    target = batch["target_albedo"].detach().float()
    if phase == "detail-reconstruction":
        value = outputs["sr_candidate_albedo"].detach().float()
    elif phase in {"physical-finetune", "boundary-hardening"}:
        value = outputs["albedo"].detach().float()
    else:
        zero = baseline.new_tensor(0.0)
        return {
            "recovery": zero,
            "gradient_recovery": zero,
            "win_fraction": zero,
            "regression_fraction": zero,
        }

    before = _sample_l1(baseline, target)
    after = _sample_l1(value, target)
    before_gradient = _sample_gradient_l1(baseline, target)
    after_gradient = _sample_gradient_l1(value, target)
    recovery = (
        (before.mean() - after.mean()) / before.mean().clamp_min(1.0e-8)
    ).detach()
    gradient_recovery = (
        (before_gradient.mean() - after_gradient.mean())
        / before_gradient.mean().clamp_min(1.0e-8)
    ).detach()

    # Batch size is normally one in Quick. Averaging these detached indicators in
    # the canonical metric accumulator therefore gives the held-out patch fraction.
    tolerance = 1.0e-8
    win_fraction = (after < before - tolerance).float().mean().detach()
    regression_fraction = (after > before + tolerance).float().mean().detach()
    return {
        "recovery": recovery,
        "gradient_recovery": gradient_recovery,
        "win_fraction": win_fraction,
        "regression_fraction": regression_fraction,
    }


def _loss_with_v133_trainer_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _PREVIOUS_LOSS is None:
        raise RuntimeError("V13.3 trainer telemetry installed without active SR loss")
    result = _add_v133_trainer_metrics(
        _PREVIOUS_LOSS(outputs, batch, config, phase),
        phase,
    )
    outcome = _reconstruction_outcome_metrics(outputs, batch, phase)

    # The two generic names are still directly indexed by the canonical progress
    # logger and final regression safety accounting. They now mean exactly the
    # active reconstruction's per-patch outcome relative to deterministic B.
    result["improvement_fraction"] = outcome["win_fraction"]
    result["regression_fraction"] = outcome["regression_fraction"]

    if phase == "detail-reconstruction":
        result["detail_recovery"] = outcome["recovery"]
        result["detail_gradient_recovery"] = outcome["gradient_recovery"]
        result["detail_win_fraction"] = outcome["win_fraction"]
        result["detail_regression_fraction"] = outcome["regression_fraction"]
    elif phase in {"physical-finetune", "boundary-hardening"}:
        result["final_recovery"] = outcome["recovery"]
        result["final_win_fraction"] = outcome["win_fraction"]
        result["final_regression_fraction"] = outcome["regression_fraction"]
    return result


def install_sr_first_training_telemetry_contract() -> None:
    global _INSTALLED, _PREVIOUS_LOSS
    if _INSTALLED:
        return

    from .application import backend

    _PREVIOUS_LOSS = backend._compute_losses_with_production_head_supervision
    backend._compute_losses_with_production_head_supervision = _loss_with_v133_trainer_metrics
    _INSTALLED = True
