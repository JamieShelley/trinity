"""V17.7 native-to-enhanced 4x residual model.

The model consumes all aligned native *pixels* of albedo RGB, normal XY and
packed semantic channels. Cropping is only an execution/training memory policy;
no global downsampling is introduced. The trained residual is composed around
the conservative semantic B baseline, never the V17.2 neural C0.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_baseline import SemanticBaseline4x, normalize_xy

SCHEMA = "NSAMDR_V177_NATIVE_ENHANCER_V2_CONTOUR"
FEATURE_WIDTH = 32
SCALE = 4


class _Residual(nn.Module):
    def __init__(self, channels: int = FEATURE_WIDTH):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + .25 * self.conv2(F.silu(self.conv1(x)))


class NativeEnhancer4x(nn.Module):
    """Bounded RGB/normal/material enhancement with zero-delta exact fallback."""

    def __init__(self):
        super().__init__()
        self.baseline = SemanticBaseline4x()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, FEATURE_WIDTH, 3, padding=1),
            nn.SiLU(),
            _Residual(),
            _Residual(),
            _Residual(),
        )
        # Resize-convolution uses the SAME spatial kernel at every enhanced
        # pixel phase. PixelShuffle learns 16 separate subpixel phase filters
        # and can create a periodic four-pixel checkerboard on real textures.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=SCALE, mode="bilinear", align_corners=False),
            nn.Conv2d(FEATURE_WIDTH, FEATURE_WIDTH, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(FEATURE_WIDTH, FEATURE_WIDTH, 3, padding=1),
            nn.SiLU(),
        )
        # 3 RGB + 2 XY + 1 legacy selector coverage + 2 continuous + confidence.
        self.delta = nn.Conv2d(FEATURE_WIDTH, 9, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        albedo: torch.Tensor,
        normal: torch.Tensor,
        material: torch.Tensor,
        *,
        material_policy: str,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        base_a, base_n, base_m = self.baseline(
            albedo, normal, material, material_policy=material_policy,
        )
        features = self.encoder(torch.cat((albedo, normal, material), dim=1))
        raw = self.delta(self.up(features))
        gate = torch.sigmoid(raw[:, 8:9])
        correction = torch.tanh(raw[:, :8]) * gate
        enhanced_a = (base_a + .20 * correction[:, :3]).clamp(0, 1)
        enhanced_n = normalize_xy(base_n + .24 * correction[:, 3:5])
        selector = base_m[:, :1]
        if material_policy == "legacy_mask":
            selector = (selector + .20 * correction[:, 5:6]).clamp(0, 1)
        # V5 and unknown selectors must remain EXACTLY the B identity field.
        continuous = (base_m[:, 1:] + .20 * correction[:, 6:8]).clamp(0, 1)
        enhanced_m = torch.cat((selector, continuous), dim=1)
        result = {
            "baseline_albedo": base_a,
            "baseline_normal": base_n,
            "baseline_material": base_m,
            "candidate_albedo": enhanced_a,
            "candidate_normal": enhanced_n,
            "candidate_material": enhanced_m,
        }
        if return_diagnostics:
            result["enhancement_confidence"] = gate
            result["raw_correction"] = raw[:, :8]
        return result
