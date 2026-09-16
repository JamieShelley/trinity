"""Structure-conditioned V16 candidate used by the broad-authority proof.

The existing V16 Swin reconstruction body remains unchanged. This wrapper adds
one LR structure-conditioning branch and fuses it into the existing phase-neutral
HR context. The branch predicts features, not pixels, profiles or an external
geometry field.
"""
from __future__ import annotations

import torch
from torch import nn

# Support both execution modes used by NSAMDR:
# - package mode in unit tests: tools.nsamdr.neural.v16.conditioning
# - script mode in diagnostic runners after tools/nsamdr/neural is added to sys.path:
#   v16.conditioning
try:
    from ..v14.baseline import normalize_xy
    from ..v14.config import V16Config
    from ..v14.model import NSAMDRV16
except ImportError:  # pragma: no cover - exercised by script-mode diagnostics
    from v14.baseline import normalize_xy
    from v14.config import V16Config
    from v14.model import NSAMDRV16

from .structure import StructureConditioningEncoder


class StructureConditionedV16Candidate(nn.Module):
    """Candidate-only V16 proof with additive learned structure conditioning."""

    def __init__(
        self,
        config: V16Config | None = None,
        *,
        structure_channels: int = 48,
        structure_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.config = config or V16Config()
        self.config.validate()
        self.base = NSAMDRV16(self.config)
        self.structure = StructureConditioningEncoder(
            scale=self.config.scale,
            channels=int(structure_channels),
            output_channels=self.config.lr_context_channels,
            blocks=int(structure_blocks),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.config.lr_context_channels * 2,
                self.config.lr_context_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.config.lr_context_channels,
                self.config.lr_context_channels,
                3,
                padding=1,
            ),
        )
        # The ablation must begin as the exact control graph. The final fusion
        # projection is therefore zero-initialized; structure can only influence
        # reconstruction after training demonstrates a useful learned residual.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

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
        context_hr = appearance_context + self.fusion(
            torch.cat((appearance_context, structure_context), dim=1)
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
        }

    def set_candidate_training(self) -> None:
        """Train V16 candidate modules plus structure conditioning; selector stays frozen."""

        self.base.set_candidate_training()
        for parameter in self.structure.parameters():
            parameter.requires_grad_(True)
        for parameter in self.fusion.parameters():
            parameter.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
