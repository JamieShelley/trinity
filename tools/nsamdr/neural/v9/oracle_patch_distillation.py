"""Oracle-distilled overlapping local SDF/coverage reconstruction.

Each LR lattice location predicts a small HR signed-distance/coverage patch.
Neighbouring patch predictions overlap and are combined with a fixed centre-
weighted window.  The network therefore learns the same local geometry used by
the GT-SDF renderer without assigning hard ownership of an HR region to one LR
cell.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _LocalBlock(nn.Module):
    # Purpose: Implement init for _LocalBlock.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(max(1, min(8, channels // 8)), channels)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    # Purpose: Implement forward for _LocalBlock.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.norm(x))
        y = self.pointwise(F.gelu(y))
        return x + y * self.scale


class OraclePatchDistillationService:
    # Purpose: Implement blend window for OraclePatchDistillationService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _blend_window(self, patch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
        # Positive raised-cosine window.  A non-zero floor keeps image-border
        # normalization well-conditioned while the central region has most weight.
        n = int(patch_size)
        coord = torch.arange(n, device=device, dtype=torch.float32)
        phase = (coord + 0.5) / float(n)
        w = 0.10 + 0.90 * torch.sin(math.pi * phase).square()
        return (w[:, None] * w[None, :]).reshape(1, n * n, 1)

    # Purpose: Implement extract target patches for OraclePatchDistillationService.
    # Called by: extract_target_patch_validity
    # Calls: No same-class helper methods.
    def extract_target_patches(
        self,
        target_pixels: torch.Tensor,
        *,
        patch_size: int,
        upscale: int = 4,
        footprint_lr: int = 3,
    ) -> torch.Tensor:
        """Extract HR teacher patches in the exact layout used by the predictor."""
        padding_hr = (int(footprint_lr) // 2) * int(upscale)
        return F.unfold(
            target_pixels.float(), kernel_size=int(patch_size),
            stride=int(upscale), padding=padding_hr,
        )

    # Purpose: Implement extract target patch validity for OraclePatchDistillationService.
    # Called by: External callers and the owning workflow.
    # Calls: extract_target_patches
    def extract_target_patch_validity(
        self,
        target_pixels: torch.Tensor,
        *,
        patch_size: int,
        upscale: int = 4,
        footprint_lr: int = 3,
    ) -> torch.Tensor:
        """Return 1 for real HR samples and 0 for unfold padding.

        Edge LR cells predict patches that extend beyond the image. Those padded
        coefficients never contribute to the folded output and must not receive a
        fictitious zero-SDF/zero-coverage teacher target.
        """
        ones = torch.ones_like(target_pixels[:, :1], dtype=torch.float32)
        return self.extract_target_patches(
            ones, patch_size=patch_size, upscale=upscale, footprint_lr=footprint_lr
        )

_oracle_patch_distillation_service = OraclePatchDistillationService()
_blend_window = _oracle_patch_distillation_service._blend_window
extract_target_patches = _oracle_patch_distillation_service.extract_target_patches
extract_target_patch_validity = _oracle_patch_distillation_service.extract_target_patch_validity


class OraclePatchSDFPredictor(nn.Module):
    """Predict overlapping HR SDF and coverage patches from LR evidence.

    A 3x3-LR-footprint / 12x12-HR patch is predicted by default for each LR
    lattice location.  Patches are folded with stride four and overlap by four
    HR pixels on every side.  The output is a continuous aggregate field rather
    than independent per-phase or per-cell geometry.
    """

    # Purpose: Implement init for OraclePatchSDFPredictor.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        *,
        upscale: int = 4,
        footprint_lr: int = 3,
        max_distance_pixels: float = 24.0,
        max_delta_pixels: float = 8.0,
        max_coverage_logit_delta: float = 10.0,
    ) -> None:
        super().__init__()
        self.upscale = int(upscale)
        self.footprint_lr = int(footprint_lr)
        if self.footprint_lr < 1 or self.footprint_lr % 2 == 0:
            raise ValueError("footprint_lr must be a positive odd integer")
        self.patch_size = self.footprint_lr * self.upscale
        self.patch_area = self.patch_size * self.patch_size
        self.padding_hr = (self.footprint_lr // 2) * self.upscale
        self.max_distance_pixels = float(max_distance_pixels)
        self.max_delta_pixels = float(max_delta_pixels)
        self.max_coverage_logit_delta = float(max_coverage_logit_delta)

        hidden = max(32, int(hidden_channels))
        self.body = nn.Sequential(
            nn.Conv2d(feature_channels + 1, hidden, 3, padding=1),
            nn.GELU(),
            _LocalBlock(hidden, 1),
            _LocalBlock(hidden, 2),
            _LocalBlock(hidden, 1),
            _LocalBlock(hidden, 3),
        )
        # First half predicts SDF delta relative to a bilinear LR-SDF patch;
        # second half predicts a coverage-logit residual relative to the
        # coverage induced by that same source patch.
        self.head = nn.Conv2d(hidden, self.patch_area * 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.confidence_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, -1.0)
        self.register_buffer("blend_window", _blend_window(self.patch_size), persistent=False)

    # Purpose: Implement source hr for OraclePatchSDFPredictor.
    # Called by: forward
    # Calls: No same-class helper methods.
    def _source_hr(self, source_sdf_prior_lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            source_sdf_prior_lr.float(), scale_factor=self.upscale,
            mode="bilinear", align_corners=False,
        ) * self.max_distance_pixels

    # Purpose: Implement unfold hr for OraclePatchSDFPredictor.
    # Called by: forward
    # Calls: No same-class helper methods.
    def _unfold_hr(self, value: torch.Tensor) -> torch.Tensor:
        return F.unfold(
            value.float(), kernel_size=self.patch_size,
            stride=self.upscale, padding=self.padding_hr,
        )

    # Purpose: Implement fold for OraclePatchSDFPredictor.
    # Called by: forward
    # Calls: No same-class helper methods.
    def _fold(self, patches: torch.Tensor, height_lr: int, width_lr: int) -> torch.Tensor:
        h_hr, w_hr = height_lr * self.upscale, width_lr * self.upscale
        weight = self.blend_window.to(device=patches.device, dtype=patches.dtype)
        weighted = patches * weight
        numerator = F.fold(
            weighted, output_size=(h_hr, w_hr), kernel_size=self.patch_size,
            stride=self.upscale, padding=self.padding_hr,
        )
        denominator = F.fold(
            weight.expand(patches.shape[0], -1, patches.shape[-1]),
            output_size=(h_hr, w_hr), kernel_size=self.patch_size,
            stride=self.upscale, padding=self.padding_hr,
        )
        return numerator / denominator.clamp_min(1.0e-6)

    # Purpose: Implement forward for OraclePatchSDFPredictor.
    # Called by: External callers and the owning workflow.
    # Calls: _fold, _source_hr, _unfold_hr
    def forward(
        self,
        features_lr: torch.Tensor,
        source_sdf_prior_lr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features_lr.shape[-2:] != source_sdf_prior_lr.shape[-2:]:
            raise ValueError("feature/prior LR sizes must match")
        b, _c, h_lr, w_lr = features_lr.shape
        x = self.body(torch.cat((features_lr, source_sdf_prior_lr.to(features_lr.dtype)), dim=1))
        raw = self.head(x).float().flatten(2)
        raw_sdf, raw_cov = raw.chunk(2, dim=1)

        source_hr_pixels = self._source_hr(source_sdf_prior_lr)
        source_patches = self._unfold_hr(source_hr_pixels)
        source_cov = torch.sigmoid(-source_patches / 0.45)
        source_cov_logit = torch.logit(source_cov.clamp(1.0e-5, 1.0 - 1.0e-5))

        sdf_delta_patches = torch.tanh(raw_sdf) * self.max_delta_pixels
        sdf_patches = (source_patches + sdf_delta_patches).clamp(
            -self.max_distance_pixels, self.max_distance_pixels
        )
        coverage_logit_delta = torch.tanh(raw_cov) * self.max_coverage_logit_delta
        coverage_logit_patches = source_cov_logit + coverage_logit_delta
        coverage_patches = torch.sigmoid(coverage_logit_patches)

        phi_pixels = self._fold(sdf_patches, h_lr, w_lr)
        coverage = self._fold(coverage_patches, h_lr, w_lr).clamp(0.0, 1.0)
        confidence_lr = torch.sigmoid(self.confidence_head(x).float())
        confidence = F.interpolate(
            confidence_lr, scale_factor=self.upscale, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)

        return {
            "phi_pixels": phi_pixels,
            "coverage": coverage,
            "sdf_patches_pixels": sdf_patches,
            "sdf_delta_patches_pixels": sdf_delta_patches,
            "coverage_patches": coverage_patches,
            "coverage_logit_patches": coverage_logit_patches,
            "coverage_logit_delta_patches": coverage_logit_delta,
            "source_sdf_patches_pixels": source_patches,
            "source_sdf_prior_pixels": source_hr_pixels,
            "confidence": confidence,
            "confidence_lr": confidence_lr,
            "patch_size": phi_pixels.new_tensor(float(self.patch_size)),
        }

    # Purpose: Implement query aggregate for OraclePatchSDFPredictor.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def query_aggregate(
        self,
        phi_pixels: torch.Tensor,
        query_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly query the overlap-aggregated HR field at arbitrary coords."""
        return F.grid_sample(
            phi_pixels.float(), query_grid.float(), mode="bilinear",
            padding_mode="border", align_corners=False,
        )
