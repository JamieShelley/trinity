"""Small tensor primitives used by the evolutionary structural microproof."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def central_difference(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate central-difference X/Y gradients with replicated borders.

    Purpose:
        Provide the structural fitness objective with metric-gradient evidence.
    Called by:
        StructuralObjective.evaluate(), StructuralFitness.measure().
    Calls:
        torch.nn.functional.pad().
    """
    padded_x = F.pad(value.float(), (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(value.float(), (0, 0, 1, 1), mode="replicate")
    gx = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2])
    gy = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :])
    return gx, gy


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Calculate a numerically safe weighted tensor mean.

    Purpose:
        Centralise weighting semantics shared by microproof loss and fitness.
    Called by:
        align_polarity(), StructuralObjective.evaluate(), StructuralFitness.measure().
    Calls:
        No project functions.
    """
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def align_polarity(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Choose the physically equivalent global SDF polarity with lower error.

    Purpose:
        Avoid penalising an unobservable global SDF sign gauge.
    Called by:
        StructuralObjective.evaluate(), StructuralFitness.measure().
    Calls:
        weighted_mean().
    """
    positive = weighted_mean((predicted - target).abs(), weight)
    negative = weighted_mean((predicted + target).abs(), weight)
    return predicted if float(positive.item()) <= float(negative.item()) else -predicted


def batch_to_device(sample: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move one dataset sample to the requested device and add a batch axis.

    Purpose:
        Normalise the single-sample microproof input format.
    Called by:
        CandidateEvaluator.evaluate().
    Calls:
        No project functions.
    """
    result: dict[str, object] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            tensor = value.unsqueeze(0) if value.ndim in {0, 1, 2, 3} else value
            result[key] = tensor.to(device)
        else:
            result[key] = value
    return result
