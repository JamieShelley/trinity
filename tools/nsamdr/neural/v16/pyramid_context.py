"""Bounded wider-LR-context ablation for the V16 candidate.

The current candidate sees only local LR evidence before phase-neutral HR fusion.
This wrapper reuses the trained LR encoder at pooled scales so its effective
receptive field spans most of the 128-LR tile, then learns a zero-initialized
fusion residual. At initialization its output is exactly the existing
StructureConditionedV16Candidate output.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from ..v14.baseline import normalize_xy
    from ..v14.config import V16Config
except ImportError:  # pragma: no cover
    from v14.baseline import normalize_xy
    from v14.config import V16Config

from .conditioning import StructureConditionedV16Candidate


class PyramidContextV16Candidate(StructureConditionedV16Candidate):
    """Existing V16 candidate plus a cheap global LR context pyramid."""

    PYRAMID_FACTORS = (2, 4, 8)

    def __init__(
        self,
        config: V16Config | None = None,
        *,
        structure_channels: int = 48,
        structure_blocks: int = 4,
    ) -> None:
        super().__init__(
            config,
            structure_channels=structure_channels,
            structure_blocks=structure_blocks,
        )
        channels = int(self.config.lr_context_channels)
        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # Exact no-op at construction so an existing checkpoint is a valid control.
        nn.init.zeros_(self.pyramid_fusion[-1].weight)
        nn.init.zeros_(self.pyramid_fusion[-1].bias)

    def _pyramid_context(
        self,
        lr_maps: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        base_size = tuple(int(value) for value in lr_maps.shape[-2:])
        features: list[torch.Tensor] = []
        for factor in self.PYRAMID_FACTORS:
            pooled = F.avg_pool2d(
                lr_maps.float(),
                kernel_size=int(factor),
                stride=int(factor),
            )
            encoded = self.base.context_encoder(pooled)
            features.append(
                F.interpolate(
                    encoded.float(),
                    size=base_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        merged = torch.stack(features, dim=0).mean(dim=0)
        merged_hr = F.interpolate(
            merged,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.base.context_adapter(merged_hr)

    def forward(
        self,
        lr_albedo: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_material: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b_albedo, b_normal, b_material = self.base.baseline(
            lr_albedo,
            lr_normal,
            lr_material,
        )
        lr_maps = torch.cat(
            (lr_albedo.float(), lr_normal.float(), lr_material.float()),
            dim=1,
        )
        baseline_maps = torch.cat((b_albedo, b_normal, b_material), dim=1)
        target_size = tuple(int(value) for value in b_albedo.shape[-2:])

        appearance_context = self.base._phase_neutral_context(lr_maps, target_size)
        structure_context = self.structure(lr_maps, target_size=target_size)
        local_context = appearance_context + self.fusion(
            torch.cat((appearance_context, structure_context), dim=1)
        )
        pyramid_context = self._pyramid_context(lr_maps, target_size)
        context_hr = local_context + self.pyramid_fusion(
            torch.cat((local_context, pyramid_context), dim=1)
        )

        raw = self.base.hr_refiner(baseline_maps, context_hr)
        predicted_albedo = self.base._bounded_residual(
            raw["albedo"], self.config.albedo_residual_cap
        )
        predicted_normal = self.base._bounded_residual(
            raw["normal"], self.config.normal_residual_cap
        )
        predicted_material = self.base._bounded_residual(
            raw["material"], self.config.material_residual_cap
        )

        c_albedo = (b_albedo + predicted_albedo).clamp(0.0, 1.0)
        c_normal = normalize_xy(b_normal + predicted_normal)
        c_material = (b_material + predicted_material).clamp(0.0, 1.0)
        return {
            "baseline_albedo": b_albedo,
            "baseline_normal": b_normal,
            "baseline_material": b_material,
            "candidate_raw_residual_albedo": raw["albedo"],
            "candidate_raw_residual_normal": raw["normal"],
            "candidate_raw_residual_material": raw["material"],
            "predicted_residual_albedo": predicted_albedo,
            "predicted_residual_normal": predicted_normal,
            "predicted_residual_material": predicted_material,
            "candidate_albedo": c_albedo,
            "candidate_normal": c_normal,
            "candidate_material": c_material,
            "structure_context": structure_context,
            "pyramid_context": pyramid_context,
        }

    def set_pyramid_probe_training(self) -> None:
        """Freeze the existing candidate; learn only the wider-context fusion."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.pyramid_fusion.parameters():
            parameter.requires_grad_(True)

    def pyramid_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.pyramid_fusion.parameters()
            if parameter.requires_grad
        ]
