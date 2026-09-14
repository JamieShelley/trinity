from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .baseline import Baseline4x, normalize_xy
from .config import MODEL_SCHEMA, V15Config
from .refinement import SingleScaleHRRefinementTrunk


class FeatureResidualBlock(nn.Module):
    """Small residual block for LR context and selector features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=False)

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
        return self.body(F.relu(self.stem(lr_maps.float()), inplace=False))


class HRContextAdapter(nn.Module):
    """Adapt phase-neutral resized LR context after it enters HR coordinates."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, context_hr: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(context_hr.float()), inplace=False)


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
            nn.ReLU(inplace=False),
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


class NSAMDRV15(nn.Module):
    """V15.0 production graph with a single-resolution EDSR-style HR backbone."""

    BASELINE_CHANNELS = 8

    def __init__(self, config: V15Config | None = None) -> None:
        super().__init__()
        self.config = config or V15Config()
        self.config.validate()

        self.baseline = Baseline4x(self.config.scale)
        self.context_encoder = LRContextEncoder(
            self.config.lr_context_channels,
            self.config.lr_blocks,
        )
        self.context_adapter = HRContextAdapter(self.config.lr_context_channels)
        self.hr_refiner = SingleScaleHRRefinementTrunk(
            baseline_channels=self.BASELINE_CHANNELS,
            context_channels=self.config.lr_context_channels,
            channels=self.config.hr_channels,
            blocks=self.config.hr_blocks,
            map_tail_blocks=self.config.map_tail_blocks,
            residual_scale=self.config.residual_scale,
            checkpoint_segment_blocks=self.config.checkpoint_segment_blocks,
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

    @staticmethod
    def _bounded_residual(raw: torch.Tensor, cap: float) -> torch.Tensor:
        # V14.1 used this bounded output path and completed Capacity without numerical
        # collapse. V15 restores it while increasing capacity only in the HR body.
        return torch.tanh(raw.float()) * float(cap)

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

        raw = self.hr_refiner(baseline_maps, context_hr)
        predicted_albedo = self._bounded_residual(
            raw["albedo"], self.config.albedo_residual_cap
        )
        predicted_normal = self._bounded_residual(
            raw["normal"], self.config.normal_residual_cap
        )
        predicted_material = self._bounded_residual(
            raw["material"], self.config.material_residual_cap
        )

        c_albedo = (b_albedo + predicted_albedo).clamp(0.0, 1.0)
        c_normal = normalize_xy(b_normal + predicted_normal)
        c_material = (b_material + predicted_material).clamp(0.0, 1.0)
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
            "candidate_raw_residual_albedo": raw["albedo"],
            "candidate_raw_residual_normal": raw["normal"],
            "candidate_raw_residual_material": raw["material"],
            "predicted_residual_albedo": predicted_albedo,
            "predicted_residual_normal": predicted_normal,
            "predicted_residual_material": predicted_material,
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
            "revision": "V15.0",
            "scale": self.config.scale,
            "productionForward": (
                "LR -> deterministic B + phase-neutral LR context -> "
                "single-resolution EDSR-style HR residual body -> "
                "tanh bounded physical-map residual -> C -> "
                "physical-map-aware BenefitSelector F"
            ),
            "backbone": "EDSR-style-single-resolution-HR",
            "contextUpsampling": "bilinear-phase-neutral + HR 3x3 adapter",
            "decoderUpsampling": "none",
            "residualBounding": "tanh",
            "residualSupervision": "bounded-pre-physical-projection",
            "residualScale": float(self.config.residual_scale),
            "candidateIdentityAtInitialization": "C == B",
            "activeComponents": (
                "baseline",
                "context_encoder",
                "context_adapter",
                "single_scale_hr_refiner",
                "albedo_tail",
                "normal_tail",
                "material_tail",
                "selector",
            ),
            "retiredComponents": (
                "multiscale_hr_refiner",
                "downsample_features",
                "upsample_and_fuse",
                "channel_attention",
                "straight_through_residual_limiter",
            ),
            "geometryPixelAuthority": False,
            "seamPixelAuthority": False,
            "profilePixelAuthority": False,
            "lrPhaseGridPixelAuthority": False,
            "multiscaleFeatureHierarchy": False,
            "batchNormalizationUsed": False,
            "channelAttentionUsed": False,
            "pixelShuffleUsed": False,
            "transposedConvolutionUsed": False,
            "selectorUsesPhysicalMaps": True,
            "gradientCheckpointing": bool(
                self.config.use_gradient_checkpointing
            ),
        }


# Compatibility alias for existing imports in the v14 package path.
NSAMDRV14 = NSAMDRV15
