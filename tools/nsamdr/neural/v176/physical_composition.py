"""V17.6 Stage 10A: phase-neutral, bounded *albedo-only* detail composition.

Never takes authored HR pixels. Composes against the original *raw* V17.2
residual before tanh, preserving physical caps and normal/material outputs.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .stability import STABILITY_POLICY, NumericalStabilityError, require_tensor


def gaussian_highpass(
    correction: torch.Tensor,
    *,
    sigma: float = 1.25,
    radius: int = 4,
) -> torch.Tensor:
    """Remove coarse components without 4x phase grids or downsampling.

    Uses a translation-equivariant spatial Gaussian in the interior. Reflect
    padding affects only the image boundary, not 4x subpixel phases.
    """
    if correction.ndim != 4 or correction.shape[1] != 1:
        raise ValueError("Stage 10A expects N x 1 x HR x HR grayscale detail")
    if not 0.0 < float(sigma) or int(radius) < 1:
        raise ValueError("Gaussian sigma and radius must be positive")
    if min(correction.shape[-2:]) < 2:
        raise ValueError("detail plane is too small")
    require_tensor("fine_detail_raw", correction, STABILITY_POLICY["rawMaxAbs"])
    # .float() alone does not stop an enclosing autocast from reducing the
    # convolution precision and leaking a large DC component into the detail.
    with torch.autocast(device_type=correction.device.type, enabled=False), torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=False,
    ):
        plane = correction.float()
        offsets = torch.arange(
            -int(radius), int(radius) + 1,
            device=plane.device, dtype=plane.dtype,
        )
        one_d = torch.exp(-0.5 * (offsets / float(sigma)).square())
        one_d = one_d / one_d.sum()
        kernel = torch.outer(one_d, one_d)[None, None]
        border = "reflect" if min(plane.shape[-2:]) > int(radius) else "replicate"
        padded = F.pad(plane, (radius, radius, radius, radius), mode=border)
        lowpass = F.conv2d(padded, kernel)
        result = plane - lowpass
    try:
        require_tensor("fine_detail_highpass", result, STABILITY_POLICY["highpassMaxAbs"])
    except NumericalStabilityError as exc:
        exc.tensors["fine_detail_raw"] = plane.detach().cpu().contiguous()
        raise
    return result


def compose_albedo_detail(
    original: dict[str, torch.Tensor],
    detail_highpass: torch.Tensor,
    *,
    albedo_residual_cap: float,
) -> dict[str, torch.Tensor]:
    """Return revised C1. Delta=0 yields exact C0, including projected maps.

    Stage 10A uses one independent scalar BC4 detail field, broadcast across
    the three existing albedo channels. Normal XY and material are untouched.
    """
    raw = original["candidate_raw_residual_albedo"].float()
    baseline = original["baseline_albedo"].float()
    if raw.ndim != 4 or raw.shape[1] != 3 or baseline.shape != raw.shape:
        raise ValueError("expected aligned V17.2 N x 3 x HR x HR albedo tensors")
    if detail_highpass.ndim != 4 or tuple(detail_highpass.shape) != (
        raw.shape[0], 1, raw.shape[2], raw.shape[3]
    ):
        raise ValueError("detail must be one scalar field aligned with V17.2 HR")
    if not 0.0 < float(albedo_residual_cap) <= 1.0:
        raise ValueError("albedo residual cap must be within (0,1]")
    require_tensor("backbone_raw_albedo", raw)
    require_tensor("baseline_albedo", baseline, 1.0)
    require_tensor("fine_detail_highpass", detail_highpass, STABILITY_POLICY["highpassMaxAbs"])
    updated_raw = raw + detail_highpass.float()
    bounded = torch.tanh(updated_raw) * float(albedo_residual_cap)
    candidate = (baseline + bounded).clamp(0.0, 1.0)

    result = dict(original)
    result.update({
        "base_candidate_albedo": original["candidate_albedo"],
        "candidate_raw_residual_albedo": updated_raw,
        "predicted_residual_albedo": bounded,
        "candidate_albedo": candidate,
        "candidate_residual_albedo": candidate - baseline,
        "fine_detail_highpass": detail_highpass.float(),
    })
    return result
