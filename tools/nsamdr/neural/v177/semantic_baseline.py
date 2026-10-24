"""V17.7 conservative semantic-aware 4x reference baseline.

Research-only replacement for the preview's B pane, NOT the frozen V17.6
Baseline4x object. C0/C1 still run their checkpoint-authenticated V17.6 path.
A classification of semantic material channels must be supplied by the caller.
Categorical V5 selectors retain their exact slot identity: one scalar packed
selector cannot represent arbitrary mixing between four materials.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

POLICIES = ("legacy_mask", "v5_selector", "unknown")


def normalize_xy(xy: torch.Tensor) -> torch.Tensor:
    if xy.ndim != 4 or xy.shape[1] != 2:
        raise ValueError("normal XY must be NCHW with 2 channels")
    xy = xy.float()
    length = xy.square().sum(1, keepdim=True).clamp_min(1e-8).sqrt()
    return xy / torch.maximum(length / 0.999, torch.ones_like(length))


def _resample(value: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(value.float(), size=size, mode="nearest")
    return F.interpolate(value.float(), size=size, mode=mode, align_corners=False)


class SemanticBaseline4x(nn.Module):
    """Type-preserving baseline for preview comparison; not a trained enhancer."""

    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        if scale != 4:
            raise ValueError("V17.7 proof preview supports only exact 4x")
        self.scale = scale

    def forward(
        self,
        albedo: torch.Tensor,
        normal_xy: torch.Tensor,
        material: torch.Tensor,
        *,
        material_policy: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if material_policy not in POLICIES:
            raise ValueError("unknown material selector policy")
        if (albedo.ndim != 4 or albedo.shape[1] != 3 or
            normal_xy.ndim != 4 or normal_xy.shape[1] != 2 or
            material.ndim != 4 or material.shape[1] != 3 or
            normal_xy.shape[-2:] != albedo.shape[-2:] or
            material.shape[-2:] != albedo.shape[-2:]):
            raise ValueError("baseline requires aligned native albedo/normal/material")
        size = tuple(d * self.scale for d in albedo.shape[-2:])

        # RGB channel remains continuous, but anti-aliased edges must eventually
        # be learned against better-than-native references.
        out_a = _resample(albedo, size, "bicubic").clamp(0, 1)
        # Signed XY vectors need a joint length constraint after interpolation.
        out_n = normalize_xy(_resample(normal_xy, size, "bilinear"))

        selector = material[:, :1]
        if material_policy == "legacy_mask":
            # Legacy PGS mask is continuous coverage: sub-mask and mask are
            # separately packed in the original material texture. This
            # reference treats the selected source channel as coverage.
            chosen = _resample(selector, size, "bilinear")
        else:
            # Four-tone V5 selector values are *identities*; a fraction between
            # slot centres is not a valid arbitrary mixture. Until a separate
            # four-channel coverage representation and shader are implemented,
            # preserve categorical identity exactly. Unknown => fail-safe.
            chosen = _resample(selector, size, "nearest")
        continuous = _resample(material[:, 1:], size, "bilinear")
        out_m = torch.cat((chosen, continuous), dim=1).clamp(0, 1)
        return out_a, out_n, out_m
