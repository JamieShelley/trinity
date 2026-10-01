"""NSAMDR V17 coordinate-conditioned physical-map reconstruction.

V16 built an HR grid first and then refined it. V17 instead learns the
reconstruction mapping in LR space and queries an HR residual at relative
subpixel coordinates. Absolute UV coordinates are never provided.

The deterministic 4x baseline remains the projection anchor:
    C = project(B + predicted_residual)
"""
from __future__ import annotations

import math

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


class PhysicalMapEncoder(nn.Module):
    """Encode aligned LR physical maps before any HR raster is created."""

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


def _relative_coordinate_channels(
    *,
    height: int,
    width: int,
    scale: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return relative HR-pixel coordinates inside each LR texel.

    Coordinates are local only. No absolute UV or authority identity appears.
    For 4x the raw dx/dy values are -0.75, -0.25, +0.25, +0.75.
    """

    if scale < 1:
        raise ValueError("scale must be positive")
    y_phase = (
        ((torch.arange(height, device=device, dtype=dtype) % scale) + 0.5)
        / float(scale)
        * 2.0
        - 1.0
    )
    x_phase = (
        ((torch.arange(width, device=device, dtype=dtype) % scale) + 0.5)
        / float(scale)
        * 2.0
        - 1.0
    )
    dy = y_phase.view(1, 1, height, 1).expand(1, 1, height, width)
    dx = x_phase.view(1, 1, 1, width).expand(1, 1, height, width)
    pi = float(math.pi)
    return torch.cat(
        (
            dx,
            dy,
            dx * dx,
            dy * dy,
            dx * dy,
            torch.sin(pi * dx),
            torch.cos(pi * dx),
            torch.sin(pi * dy),
            torch.cos(pi * dy),
        ),
        dim=1,
    )


class RelativeQueryDecoder(nn.Module):
    """Decode HR residuals from LR features plus local subpixel coordinates.

    A 3x3 LR neighbourhood is encoded before continuous HR querying. The query
    representation is shared across all texels and receives only relative
    subpixel position, preventing direct absolute-layout memorisation.
    """

    COORD_CHANNELS = 9
    OUTPUT_CHANNELS = 8

    def __init__(
        self,
        *,
        feature_channels: int,
        neighbourhood_channels: int = 128,
        hidden_channels: int = 192,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if min(feature_channels, neighbourhood_channels, hidden_channels, scale) < 1:
            raise ValueError("decoder dimensions must be positive")
        self.feature_channels = int(feature_channels)
        self.neighbourhood_channels = int(neighbourhood_channels)
        self.hidden_channels = int(hidden_channels)
        self.scale = int(scale)

        self.neighbourhood_reduce = nn.Sequential(
            nn.Conv2d(
                self.feature_channels * 9,
                self.neighbourhood_channels,
                1,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.neighbourhood_channels,
                self.neighbourhood_channels,
                1,
            ),
            nn.GELU(),
        )
        decoder_inputs = (
            self.feature_channels
            + self.neighbourhood_channels
            + 8
            + self.COORD_CHANNELS
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(decoder_inputs, self.hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 1),
            nn.GELU(),
        )
        self.albedo_head = nn.Conv2d(self.hidden_channels, 3, 1)
        self.normal_head = nn.Conv2d(self.hidden_channels, 2, 1)
        self.material_head = nn.Conv2d(self.hidden_channels, 3, 1)

        # Identity-safe start: before training C == B exactly.
        for head in (self.albedo_head, self.normal_head, self.material_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _neighbourhood(self, features: torch.Tensor) -> torch.Tensor:
        n, c, h, w = features.shape
        unfolded = F.unfold(features, kernel_size=3, padding=1)
        return unfolded.view(n, c * 9, h, w)

    def forward(
        self,
        features_lr: torch.Tensor,
        lr_maps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features_lr.shape[-2:] != lr_maps.shape[-2:]:
            raise ValueError("feature and LR-map sizes must match")
        h, w = (int(value) for value in features_lr.shape[-2:])
        target_size = (h * self.scale, w * self.scale)

        neighbourhood_lr = self.neighbourhood_reduce(
            self._neighbourhood(features_lr)
        )
        # Continuous latent/evidence interpolation supplies cross-cell continuity.
        feature_query = F.interpolate(
            features_lr.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        neighbourhood_query = F.interpolate(
            neighbourhood_lr.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        evidence_query = F.interpolate(
            lr_maps.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        coords = _relative_coordinate_channels(
            height=target_size[0],
            width=target_size[1],
            scale=self.scale,
            device=features_lr.device,
            dtype=features_lr.float().dtype,
        ).expand(features_lr.shape[0], -1, -1, -1)

        hidden = self.mlp(
            torch.cat(
                (
                    feature_query,
                    neighbourhood_query,
                    evidence_query,
                    coords,
                ),
                dim=1,
            )
        )
        return {
            "albedo": self.albedo_head(hidden),
            "normal": self.normal_head(hidden),
            "material": self.material_head(hidden),
        }


class NSAMDRV17(nn.Module):
    """Active V17 candidate: LR physical encoder + relative query decoder."""

    def __init__(
        self,
        contract: V16Config | None = None,
        *,
        encoder_channels: int = 96,
        neighbourhood_channels: int = 128,
        decoder_hidden_channels: int = 192,
    ) -> None:
        super().__init__()
        self.config = contract or V16Config()
        self.config.validate()
        if int(self.config.scale) != 4:
            raise ValueError("V17 proof currently requires exactly 4x reconstruction")

        self.baseline = Baseline4x(self.config.scale)
        self.encoder = PhysicalMapEncoder(channels=int(encoder_channels))
        self.decoder = RelativeQueryDecoder(
            feature_channels=int(encoder_channels),
            neighbourhood_channels=int(neighbourhood_channels),
            hidden_channels=int(decoder_hidden_channels),
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
        }

    def set_candidate_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def architecture_contract(self) -> dict[str, object]:
        return {
            "revision": "V17.0-proof",
            "scale": int(self.config.scale),
            "productionForward": (
                "LR aligned physical maps -> LR physical encoder -> "
                "continuous local relative-coordinate query decoder -> "
                "bounded physical residual -> deterministic B projection -> C"
            ),
            "absoluteUvCoordinatesUsed": False,
            "relativeSubpixelCoordinatesUsed": True,
            "learnedHrReconstructionDecoder": True,
            "pixelShuffleUsed": False,
            "transposedConvolutionUsed": False,
            "fixedHrSwinRefinerUsed": False,
            "deterministicBaselinePreserved": True,
            "alignedPhysicalMapsUsed": True,
            "candidateIdentityAtInitialization": True,
            "encoderDilations": list(self.encoder.dilations),
            "encoderChannels": int(self.encoder.channels),
            "decoderNeighbourhoodChannels": int(self.decoder.neighbourhood_channels),
            "decoderHiddenChannels": int(self.decoder.hidden_channels),
        }
