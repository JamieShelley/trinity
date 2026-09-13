from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalize_xy(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    length = torch.sqrt((value * value).sum(dim=1, keepdim=True).clamp_min(1.0e-8))
    scale = torch.maximum(torch.ones_like(length), length / 0.999)
    return value / scale


class Baseline4x(nn.Module):
    """Single source of truth for deterministic 4x baseline B."""

    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self.scale = int(scale)

    def forward(
        self,
        lr_albedo: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_material: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = (lr_albedo.shape[-2] * self.scale, lr_albedo.shape[-1] * self.scale)
        albedo = F.interpolate(lr_albedo.float(), size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        normal = normalize_xy(F.interpolate(lr_normal.float(), size=size, mode="bilinear", align_corners=False))
        material = F.interpolate(lr_material.float(), size=size, mode="nearest").clamp(0.0, 1.0)
        return albedo, normal, material
