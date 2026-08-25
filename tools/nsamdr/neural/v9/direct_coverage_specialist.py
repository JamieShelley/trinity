"""Local boundary-profile and benefit specialists for NSAMDR V9.9.3.

The structural network owns geometry.  The profile specialist is a compact
sliding-window network that can only reshape the shared sub-pixel coverage
inside a narrow contour band.  It predicts a correction in coverage-logit
space so it can move an initially soft 0/1 transition all the way to the
Panel-2 teacher profile without gaining arbitrary RGB authority.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        dilation = max(1, int(dilation))
        self.dw = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.pw2(F.gelu(self.pw1(self.dw(x))))
        return x + r * self.scale


class BoundaryProfileSpecialist(nn.Module):
    """Direct local coverage-profile refiner with no RGB authority.

    The output head is identity-initialised: a zero logit correction returns the
    input coverage exactly.  Unlike a bounded additive coverage delta, the logit
    correction can sharpen or move a transition across almost the full [0, 1]
    range while remaining a single shared scalar coverage field for albedo,
    normals and material semantics.
    """

    def __init__(
        self,
        in_channels: int = 18,
        channels: int = 48,
        *,
        max_logit_delta: float = 16.0,
    ) -> None:
        super().__init__()
        self.max_logit_delta = float(max_logit_delta)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 5, padding=2),
            nn.GELU(),
            LocalResidualBlock(channels, dilation=1),
            LocalResidualBlock(channels, dilation=2),
            LocalResidualBlock(channels, dilation=1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        # channel 0: coverage-logit correction; channel 1: confidence
        self.head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        features: torch.Tensor,
        initial_coverage: torch.Tensor,
        band_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.head(self.body(features))
        confidence = torch.sigmoid(raw[:, 1:2])

        initial = initial_coverage.float().clamp(1.0e-4, 1.0 - 1.0e-4)
        initial_logits = torch.logit(initial)
        logit_delta = torch.tanh(raw[:, 0:1].float()) * self.max_logit_delta
        direct_coverage = torch.sigmoid(initial_logits + logit_delta)

        # Full authority near the predicted contour, smooth fade outside.  The
        # 1.5 gain makes approximately the inner 0.4*band_pixels a direct local
        # prediction rather than a weak blend with the blurry source profile.
        authority = (band_weight.float().clamp(0.0, 1.0) * 1.5).clamp(0.0, 1.0)
        refined = initial + (direct_coverage - initial) * authority
        refined = refined.clamp(0.0, 1.0)

        return {
            "coverage": refined,
            "coverage_delta": refined - initial_coverage.float(),
            "coverage_logit_delta": logit_delta,
            "direct_coverage": direct_coverage,
            "authority": authority,
            "confidence": confidence,
        }


class BenefitSelector(nn.Module):
    """Small local selector trained only after candidate reconstruction freezes."""

    def __init__(self, in_channels: int = 10, channels: int = 24) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 5, padding=2),
            nn.GELU(),
            LocalResidualBlock(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, -2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)
