from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .baseline import Baseline4x, normalize_xy
from .config import MODEL_SCHEMA, V14Config
from .refinement import MultiScaleHRRefinementTrunk


class FeatureResidualBlock(nn.Module):
    """Small residual block for LR context and selector features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(value)))
        return value + residual * 0.20


class LRContextEncoder(nn.Module):
    """Encode LR physical evidence without assigning 4x output phase ownership."""

    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(8, channels, 3, padding=1)
        self.body = nn.Sequential(
            *(FeatureResidualBlock(channels) for _ in range(blocks))
        )

    def forward(self, lr_maps: torch.Tensor) -> torch.Tensor:
        return self.body(F.gelu(self.stem(lr_maps.float())))


class HRContextAdapter(nn.Module):
    """Adapt phase-neutral resized LR context after it enters HR coordinates."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, context_hr: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(context_hr.float()))


def _gray_gradient(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


class BenefitSelector(nn.Module):
    """Select candidate C only where aligned physical-map evidence supports it."""

    FEATURE_CHANNELS = 35

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(self.FEATURE_CHANNELS, channels, 3, padding=1),
            nn.GELU(),
            FeatureResidualBlock(channels),
            FeatureResidualBlock(channels),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.2)

    def forward(
        self,
        baseline_maps: torch.Tensor,
        candidate_maps: torch.Tensor,
        lr_maps: torch.Tensor,
    ) -> torch.Tensor:
        target_size = baseline_maps.shape[-2:]
        delta = (candidate_maps.float() - baseline_maps.float()).abs()
        lr_hr = F.interpolate(
            lr_maps.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        bgrad = _gray_gradient(baseline_maps[:, :3])
        cgrad = _gray_gradient(candidate_maps[:, :3])
        lr_edge = F.interpolate(
            _gray_gradient(lr_maps[:, :3]),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat(
            (
                baseline_maps.float(),
                candidate_maps.float(),
                delta,
                lr_hr,
                bgrad,
                cgrad,
                lr_edge,
            ),
            dim=1,
        )
        return self.net(features)


class NSAMDRV14(nn.Module):
    """V14.2 production graph: B -> deep phase-neutral multi-scale C -> selector F."""

    BASELINE_CHANNELS = 8

    def __init__(self, config: V14Config | None = None) -> None:
        super().__init__()
        self.config = config or V14Config()
        self.config.validate()

        self.baseline = Baseline4x(self.config.scale)
        self.context_encoder = LRContextEncoder(
            self.config.lr_context_channels,
            self.config.lr_blocks,
        )
        self.context_adapter = HRContextAdapter(self.config.lr_context_channels)
        self.hr_refiner = MultiScaleHRRefinementTrunk(
            baseline_channels=self.BASELINE_CHANNELS,
            context_channels=self.config.lr_context_channels,
            hr_channels=self.config.hr_channels,
            half_channels=self.config.half_channels,
            quarter_channels=self.config.quarter_channels,
            hr_encoder_blocks=self.config.hr_encoder_blocks,
            half_encoder_blocks=self.config.half_encoder_blocks,
            quarter_encoder_blocks=self.config.quarter_encoder_blocks,
            bottleneck_blocks=self.config.bottleneck_blocks,
            half_decoder_blocks=self.config.half_decoder_blocks,
            hr_decoder_blocks=self.config.hr_decoder_blocks,
            map_tail_blocks=self.config.map_tail_blocks,
            attention_reduction=self.config.attention_reduction,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )
        self.selector = BenefitSelector(self.config.selector_channels)

    def _phase_neutral_context(
        self,
        lr_maps: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        context_lr = self.context_encoder(lr_maps)
        context_hr = F.interpolate(
            context_lr.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.context_adapter(context_hr)

    def forward(
        self,
        lr_albedo: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_material: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b_albedo, b_normal, b_material = self.baseline(
            lr_albedo,
            lr_normal,
            lr_material,
        )
        lr_maps = torch.cat(
            (lr_albedo.float(), lr_normal.float(), lr_material.float()),
            dim=1,
        )
        baseline_maps = torch.cat((b_albedo, b_normal, b_material), dim=1)
        context_hr = self._phase_neutral_context(
            lr_maps,
            b_albedo.shape[-2:],
        )
        residual = self.hr_refiner(baseline_maps, context_hr)

        c_albedo = (
            b_albedo + residual["albedo"] * self.config.albedo_residual_cap
        ).clamp(0.0, 1.0)
        c_normal = normalize_xy(
            b_normal + residual["normal"] * self.config.normal_residual_cap
        )
        c_material = (
            b_material + residual["material"] * self.config.material_residual_cap
        ).clamp(0.0, 1.0)
        candidate_maps = torch.cat((c_albedo, c_normal, c_material), dim=1)

        selector_logits = self.selector(baseline_maps, candidate_maps, lr_maps)
        gate = torch.sigmoid(selector_logits.float())
        f_albedo = (
            b_albedo * (1.0 - gate) + c_albedo * gate
        ).clamp(0.0, 1.0)
        f_normal = normalize_xy(
            b_normal * (1.0 - gate) + c_normal * gate
        )
        f_material = (
            b_material * (1.0 - gate) + c_material * gate
        ).clamp(0.0, 1.0)

        return {
            "baseline_albedo": b_albedo,
            "baseline_normal": b_normal,
            "baseline_material": b_material,
            "candidate_albedo": c_albedo,
            "candidate_normal": c_normal,
            "candidate_material": c_material,
            "candidate_residual_albedo": c_albedo - b_albedo,
            "candidate_residual_normal": c_normal - b_normal,
            "candidate_residual_material": c_material - b_material,
            "selector_logits": selector_logits,
            "selector_probability": gate,
            "albedo": f_albedo,
            "normal": f_normal,
            "material": f_material,
        }

    def set_candidate_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.context_encoder,
            self.context_adapter,
            self.hr_refiner,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def set_selector_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.selector.parameters():
            parameter.requires_grad_(True)

    def architecture_contract(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "revision": "V14.2",
            "scale": self.config.scale,
            "productionForward": (
                "LR -> deterministic B + phase-neutral LR context -> "
                "multi-scale RCAN-style HR refinement C -> "
                "physical-map-aware BenefitSelector F"
            ),
            "contextUpsampling": "bilinear-phase-neutral + HR 3x3 adapter",
            "decoderUpsampling": "bilinear-phase-neutral + HR convolution",
            "candidateIdentityAtInitialization": "C == B",
            "activeComponents": (
                "baseline",
                "context_encoder",
                "context_adapter",
                "multiscale_hr_refiner",
                "albedo_tail",
                "normal_tail",
                "material_tail",
                "selector",
            ),
            "retiredComponents": (),
            "geometryPixelAuthority": False,
            "seamPixelAuthority": False,
            "profilePixelAuthority": False,
            "lrPhaseGridPixelAuthority": False,
            "pixelShuffleUsed": False,
            "transposedConvolutionUsed": False,
            "selectorUsesPhysicalMaps": True,
            "gradientCheckpointing": bool(
                self.config.use_gradient_checkpointing
            ),
        }
