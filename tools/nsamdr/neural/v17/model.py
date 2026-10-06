"""NSAMDR V17.2 multi-scale physical residual reconstruction.

V17.0 failed because repeating subpixel phase coordinates recreated a 4x
lattice. V17.1 removed that phase ownership with a local-ensemble implicit
decoder, but the bilinear blend remained too smooth: it learned broad structure
while failing to reproduce narrow 1-2 px manufactured detail.

V17.2 keeps the proven LR physical encoder and replaces the implicit decoder
with an explicit learned residual pyramid:

    128 LR encoded physical evidence
        -> 256 mid-band residual stage
        -> 512 high-detail residual stage
        -> sum(mid-upsample, detail)
        -> bounded physical residual
        -> deterministic B projection
        -> C

Upsampling is resize-convolution only. There is no PixelShuffle, transposed
convolution, fixed 4x phase tensor, absolute UV input or fixed-HR Swin body.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from ..v14.baseline import Baseline4x, normalize_xy
    from ..v14.config import V16Config
except ImportError:  # pragma: no cover - script-mode diagnostics
    from v14.baseline import Baseline4x, normalize_xy
    from v14.config import V16Config


def _group_gradient(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    dx = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt((dx * dx + dy * dy).mean(dim=1, keepdim=True) + 1.0e-8)


class DilatedResidualBlock(nn.Module):
    """LR residual block with explicit receptive-field growth."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = int(dilation)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=int(dilation),
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=int(dilation),
        )
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(value)))
        return value + residual * 0.20


class ResidualConvBlock(nn.Module):
    """Phase-neutral resize-convolution refinement block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(value)))
        return value + residual * 0.20


class PhysicalMapEncoder(nn.Module):
    """Encode aligned LR physical maps before any HR residual is created."""

    INPUT_CHANNELS = 8
    ANALYTIC_CHANNELS = 3
    DEFAULT_DILATIONS = (1, 2, 4, 8, 4, 2, 1, 1)

    def __init__(
        self,
        *,
        channels: int = 96,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("encoder channels must be >= 8")
        if not dilations or min(int(value) for value in dilations) < 1:
            raise ValueError("encoder dilations must be positive")
        self.channels = int(channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.stem = nn.Conv2d(
            self.INPUT_CHANNELS + self.ANALYTIC_CHANNELS,
            self.channels,
            3,
            padding=1,
        )
        self.blocks = nn.Sequential(
            *(DilatedResidualBlock(self.channels, value) for value in self.dilations)
        )
        self.tail = nn.Conv2d(self.channels, self.channels, 3, padding=1)

    @staticmethod
    def analytic_edges(lr_maps: torch.Tensor) -> torch.Tensor:
        if lr_maps.ndim != 4 or lr_maps.shape[1] != 8:
            raise ValueError(
                "PhysicalMapEncoder expects N x 8 x H x W LR physical maps, "
                f"got {tuple(lr_maps.shape)}"
            )
        return torch.cat(
            (
                _group_gradient(lr_maps[:, 0:3]),
                _group_gradient(lr_maps[:, 3:5]),
                _group_gradient(lr_maps[:, 5:8]),
            ),
            dim=1,
        )

    def forward(self, lr_maps: torch.Tensor) -> torch.Tensor:
        edges = self.analytic_edges(lr_maps)
        value = self.stem(torch.cat((lr_maps.float(), edges), dim=1))
        value = F.gelu(value)
        return self.tail(self.blocks(value))


class MultiScaleResidualDecoder(nn.Module):
    """Learn mid-band and high-detail residuals at 2x and 4x explicitly.

    The 256 stage establishes larger manufactured contours. The 512 stage sees
    the refined 256 features plus direct LR latent/evidence paths and is free to
    generate narrow high-frequency residuals. This avoids forcing 1px details
    through a smooth four-anchor blend.
    """

    OUTPUT_CHANNELS = 8
    SCALES = (2, 4)

    def __init__(
        self,
        *,
        feature_channels: int,
        mid_channels: int = 64,
        detail_channels: int = 48,
        blocks_per_stage: int = 4,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if int(scale) != 4:
            raise ValueError("V17.2 decoder currently requires exactly 4x")
        if min(feature_channels, mid_channels, detail_channels, blocks_per_stage) < 1:
            raise ValueError("decoder dimensions must be positive")
        self.feature_channels = int(feature_channels)
        self.mid_channels = int(mid_channels)
        self.detail_channels = int(detail_channels)
        self.blocks_per_stage = int(blocks_per_stage)
        self.scale = int(scale)

        self.mid_in = nn.Conv2d(
            self.feature_channels + 8,
            self.mid_channels,
            3,
            padding=1,
        )
        self.mid_blocks = nn.Sequential(
            *(ResidualConvBlock(self.mid_channels) for _ in range(self.blocks_per_stage))
        )
        self.mid_head = nn.Conv2d(self.mid_channels, self.OUTPUT_CHANNELS, 3, padding=1)

        self.detail_in = nn.Conv2d(
            self.mid_channels + self.feature_channels + 8,
            self.detail_channels,
            3,
            padding=1,
        )
        self.detail_blocks = nn.Sequential(
            *(
                ResidualConvBlock(self.detail_channels)
                for _ in range(self.blocks_per_stage)
            )
        )
        self.detail_head = nn.Conv2d(
            self.detail_channels,
            self.OUTPUT_CHANNELS,
            3,
            padding=1,
        )

        # Identity-safe start: both residual bands are exactly zero.
        for head in (self.mid_head, self.detail_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            value.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        features_lr: torch.Tensor,
        lr_maps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features_lr.shape[-2:] != lr_maps.shape[-2:]:
            raise ValueError("feature and LR-map sizes must match")
        h, w = (int(value) for value in features_lr.shape[-2:])
        mid_size = (h * 2, w * 2)
        hr_size = (h * 4, w * 4)

        features_mid = self._resize(features_lr, mid_size)
        evidence_mid = self._resize(lr_maps, mid_size)
        mid = F.gelu(self.mid_in(torch.cat((features_mid, evidence_mid), dim=1)))
        mid = self.mid_blocks(mid)
        mid_residual = self.mid_head(mid)

        # The detail stage receives both refined mid features and a direct LR
        # latent/evidence path so narrow detail is not bottlenecked by mid-band
        # smoothing.
        mid_hr = self._resize(mid, hr_size)
        features_hr = self._resize(features_lr, hr_size)
        evidence_hr = self._resize(lr_maps, hr_size)
        detail = F.gelu(
            self.detail_in(torch.cat((mid_hr, features_hr, evidence_hr), dim=1))
        )
        detail = self.detail_blocks(detail)
        detail_residual = self.detail_head(detail)

        mid_residual_hr = self._resize(mid_residual, hr_size)
        raw = mid_residual_hr + detail_residual
        return {
            "albedo": raw[:, 0:3],
            "normal": raw[:, 3:5],
            "material": raw[:, 5:8],
            "mid_residual": mid_residual,
            "mid_residual_hr": mid_residual_hr,
            "detail_residual": detail_residual,
        }


class NSAMDRV17(nn.Module):
    """Active V17.2 candidate: LR encoder + learned multi-scale residual decoder."""

    def __init__(
        self,
        contract: V16Config | None = None,
        *,
        encoder_channels: int = 96,
        mid_channels: int = 64,
        detail_channels: int = 48,
        decoder_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.config = contract or V16Config()
        self.config.validate()
        if int(self.config.scale) != 4:
            raise ValueError("V17 proof currently requires exactly 4x reconstruction")

        self.baseline = Baseline4x(self.config.scale)
        self.encoder = PhysicalMapEncoder(channels=int(encoder_channels))
        self.decoder = MultiScaleResidualDecoder(
            feature_channels=int(encoder_channels),
            mid_channels=int(mid_channels),
            detail_channels=int(detail_channels),
            blocks_per_stage=int(decoder_blocks),
            scale=int(self.config.scale),
        )

    @staticmethod
    def _bounded_residual(raw: torch.Tensor, cap: float) -> torch.Tensor:
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
        features_lr = self.encoder(lr_maps)
        raw = self.decoder(features_lr, lr_maps)

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
            "decoder_mid_residual": raw["mid_residual"],
            "decoder_mid_residual_hr": raw["mid_residual_hr"],
            "decoder_detail_residual": raw["detail_residual"],
        }

    def set_candidate_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def architecture_contract(self) -> dict[str, object]:
        return {
            "revision": "V17.2-proof",
            "scale": int(self.config.scale),
            "productionForward": (
                "LR aligned physical maps -> LR physical encoder -> "
                "2x mid-band resize-convolution residual stage -> "
                "4x high-detail resize-convolution residual stage -> "
                "sum residual bands -> bounded physical residual -> "
                "deterministic B projection -> C"
            ),
            "absoluteUvCoordinatesUsed": False,
            "relativeSubpixelCoordinatesUsed": False,
            "periodicPhaseEncodingUsed": False,
            "localEnsembleUsed": False,
            "multiScaleResidualDecoderUsed": True,
            "decoderScales": list(MultiScaleResidualDecoder.SCALES),
            "resizeConvolutionUsed": True,
            "learnedHrDetailStageUsed": True,
            "learnedHrReconstructionDecoder": True,
            "pixelShuffleUsed": False,
            "transposedConvolutionUsed": False,
            "fixedHrSwinRefinerUsed": False,
            "deterministicBaselinePreserved": True,
            "alignedPhysicalMapsUsed": True,
            "candidateIdentityAtInitialization": True,
            "encoderDilations": list(self.encoder.dilations),
            "encoderChannels": int(self.encoder.channels),
            "decoderMidChannels": int(self.decoder.mid_channels),
            "decoderDetailChannels": int(self.decoder.detail_channels),
            "decoderBlocksPerStage": int(self.decoder.blocks_per_stage),
        }
