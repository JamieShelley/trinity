"""V17.6 Stage 10A training-only, undecimated Haar high-band supervision.

The undecimated transform performs *no spatial decimation*. Level 2 operates
on the level-1 lowpass with dilated filters; all details retain HR pixel
resolution. Nothing from authored HR ground truth is passed to inference.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

BAND_NAMES = ("LH", "HL", "HH")
LEVELS = (1, 2)
LOSS_WEIGHTS = {
    "wavelet":1.0,
    "edge":0.25,
    "pixel":0.15,
    "protected":0.10,
    "detail_magnitude":0.02,
}


def stationary_haar_details(plane: torch.Tensor) -> dict[str, torch.Tensor]:
    """2-level depthwise stationary Haar; preserve full HR spatial shape."""
    if plane.ndim != 4 or min(plane.shape[-2:]) < 5:
        raise ValueError("stationary Haar expects N x C x H x W with H,W >= 5")
    value = plane.float()
    channels = value.shape[1]
    norm = 1.0 / math.sqrt(2.0)
    low = value.new_tensor([norm,norm])
    high = value.new_tensor([norm,-norm])
    taps = {
        "LL":torch.outer(low,low),
        "LH":torch.outer(low,high),
        "HL":torch.outer(high,low),
        "HH":torch.outer(high,high),
    }
    output = {}
    # Keep target and candidate bands in the same fp32 arithmetic even when
    # the learned convolutions run under an enclosing mixed-precision policy.
    with torch.autocast(device_type=value.device.type, enabled=False), torch.backends.cudnn.flags(
        enabled=torch.backends.cudnn.enabled,
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=False,
    ):
        for level in LEVELS:
            dilation = 2 ** (level - 1)
            padded = F.pad(value,(0,dilation,0,dilation),mode="reflect")
            bands = {}
            for name,kernel in taps.items():
                weights = kernel[None,None].repeat(channels,1,1,1)
                bands[name] = F.conv2d(
                    padded,weights,groups=channels,dilation=dilation
                )
                if name in BAND_NAMES:
                    output[f"L{level}_{name}"] = bands[name]
            value = bands["LL"]
    return output


def _signed_gradient(plane: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    gx = F.pad(plane[:,:,:,1:] - plane[:,:,:,:-1],(0,1,0,0))
    gy = F.pad(plane[:,:,1:,:] - plane[:,:,:-1,:],(0,0,0,1))
    return gx,gy


def detail_supervision_loss(
    outputs: dict[str,torch.Tensor],
    batch: dict[str,torch.Tensor],
    *,
    denominator_floor: float = 0.001,
) -> dict[str,torch.Tensor]:
    """Train specialist only; explicit original-baseline normalization.

    The true HR target is used exclusively in this *loss*. It is not a
    network input, routing oracle, conditioning feature or inference map.
    """
    candidate = outputs["candidate_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    target = batch["target_albedo"].float()
    base_candidate = outputs["base_candidate_albedo"].detach().float()
    correction = outputs["fine_detail_highpass"].float()
    if (candidate.shape != baseline.shape or candidate.shape != target.shape or
        candidate.shape != base_candidate.shape or candidate.shape[1] != 3 or
        correction.shape != (candidate.shape[0],1,*candidate.shape[-2:])):
        raise ValueError("Stage 10A loss expects aligned 3-channel authored HR maps")
    if not math.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError("baseline denominator floor must be positive")
    value = candidate.mean(dim=1,keepdim=True)
    b = baseline.mean(dim=1,keepdim=True)
    authored = target.mean(dim=1,keepdim=True)
    # Linearity of SWT: W(C1-B) compared to W(A-B).
    reference = stationary_haar_details(authored - b)
    predicted = stationary_haar_details(value - b)
    band_errors = []
    for name in sorted(reference):
        scale = reference[name].detach().abs().mean().clamp_min(denominator_floor)
        band_errors.append((predicted[name] - reference[name]).abs().mean()/scale)
    wavelet = torch.stack(band_errors).mean()
    dx,dy = _signed_gradient(value)
    tx,ty = _signed_gradient(authored)
    edge = (dx-tx).abs().mean() + (dy-ty).abs().mean()
    pixel = (candidate-target).abs().mean()
    protected = ((baseline-target).abs().amax(dim=1,keepdim=True)
                 <= (2.0/255.0)).float()
    protected_change = (
        (candidate-base_candidate).abs().mean(dim=1,keepdim=True)*protected
    ).sum()/protected.sum().clamp_min(1.0)
    magnitude = correction.abs().mean()
    terms = {
        "wavelet":wavelet,
        "edge":edge,
        "pixel":pixel,
        "protected":protected_change,
        "detail_magnitude":magnitude,
    }
    total = sum(LOSS_WEIGHTS[name]*value for name,value in terms.items())
    return {"total":total, **terms}
