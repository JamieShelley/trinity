"""NSAMDR V17.1 local-ensemble physical-map reconstruction.

V16 built an HR grid first and refined it. V17.0 introduced explicit repeating
subpixel phase coordinates, but that recreated a 4x lattice. V17.1 keeps the LR
physical encoder and replaces that decoder with a continuity-safe local ensemble.

Each HR query is evaluated against the four surrounding LR feature anchors using
only its relative dx/dy to each anchor. The four shared-decoder predictions are
bilinearly blended, so there is no hard coordinate reset at LR texel boundaries
and no absolute UV input.

The deterministic 4x baseline remains the projection anchor:
    C = project(B + predicted_residual)
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
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :].abs(), (0, 0, 0, 1))
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


def _query_axis_geometry(
    *,
    source_size: int,
    scale: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return continuous LR query coordinate, lower/upper anchors and blend fraction."""

    if source_size < 1 or scale < 1:
        raise ValueError("source size and scale must be positive")
    target_size = int(source_size) * int(scale)
    position = (
        (torch.arange(target_size, device=device, dtype=dtype) + 0.5)
        / float(scale)
        - 0.5
    ).clamp(0.0, float(source_size - 1))
    lower = torch.floor(position).to(torch.long)
    upper = torch.clamp(lower + 1, max=source_size - 1)
    fraction = position - lower.to(dtype)
    fraction = torch.where(upper == lower, torch.zeros_like(fraction), fraction)
    return position, lower, upper, fraction


def _gather_anchor(
    value: torch.Tensor,
    y_index: torch.Tensor,
    x_index: torch.Tensor,
) -> torch.Tensor:
    """Gather one LR anchor value for every HR query position."""

    n, c, h, w = value.shape
    yy = y_index.view(-1, 1).expand(-1, x_index.numel())
    xx = x_index.view(1, -1).expand(y_index.numel(), -1)
    linear = (yy * w + xx).reshape(1, 1, -1).expand(n, c, -1)
    gathered = torch.gather(value.reshape(n, c, h * w), 2, linear)
    return gathered.reshape(n, c, y_index.numel(), x_index.numel())


class RelativeQueryDecoder(nn.Module):
    """Continuity-safe local-ensemble implicit residual decoder.

    For every HR query, the same decoder is evaluated relative to each of the
    four surrounding LR anchors. The outputs are bilinearly blended. Coordinate
    input is only (dx, dy) relative to the sampled anchor; no periodic phase
    encoding and no absolute UV are present.
    """

    COORD_CHANNELS = 2
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

    def _decode_anchor(
        self,
        *,
        features_lr: torch.Tensor,
        neighbourhood_lr: torch.Tensor,
        lr_maps: torch.Tensor,
        y_index: torch.Tensor,
        x_index: torch.Tensor,
        query_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature = _gather_anchor(features_lr, y_index, x_index)
        neighbourhood = _gather_anchor(neighbourhood_lr, y_index, x_index)
        evidence = _gather_anchor(lr_maps, y_index, x_index)

        rel_y = (
            query_y.view(-1, 1)
            - y_index.to(query_y.dtype).view(-1, 1)
        )
        rel_x = (
            query_x.view(1, -1)
            - x_index.to(query_x.dtype).view(1, -1)
        )
        h_hr = query_y.numel()
        w_hr = query_x.numel()
        coords = torch.cat(
            (
                rel_x.view(1, 1, 1, w_hr).expand(
                    features_lr.shape[0], 1, h_hr, w_hr
                ),
                rel_y.view(1, 1, h_hr, 1).expand(
                    features_lr.shape[0], 1, h_hr, w_hr
                ),
            ),
            dim=1,
        )
        hidden = self.mlp(
            torch.cat((feature, neighbourhood, evidence, coords), dim=1)
        )
        return {
            "albedo": self.albedo_head(hidden),
            "normal": self.normal_head(hidden),
            "material": self.material_head(hidden),
        }

    def forward(
        self,
        features_lr: torch.Tensor,
        lr_maps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if features_lr.shape[-2:] != lr_maps.shape[-2:]:
            raise ValueError("feature and LR-map sizes must match")
        h, w = (int(value) for value in features_lr.shape[-2:])
        dtype = features_lr.float().dtype
        query_y, y0, y1, fy = _query_axis_geometry(
            source_size=h,
            scale=self.scale,
            device=features_lr.device,
            dtype=dtype,
        )
        query_x, x0, x1, fx = _query_axis_geometry(
            source_size=w,
            scale=self.scale,
            device=features_lr.device,
            dtype=dtype,
        )
        neighbourhood_lr = self.neighbourhood_reduce(
            self._neighbourhood(features_lr)
        )

        weights = (
            (1.0 - fy).view(-1, 1) * (1.0 - fx).view(1, -1),
            (1.0 - fy).view(-1, 1) * fx.view(1, -1),
            fy.view(-1, 1) * (1.0 - fx).view(1, -1),
            fy.view(-1, 1) * fx.view(1, -1),
        )
        anchors = (
            (y0, x0),
            (y0, x1),
            (y1, x0),
            (y1, x1),
        )

        result: dict[str, torch.Tensor] | None = None
        for weight, (yi, xi) in zip(weights, anchors):
            prediction = self._decode_anchor(
                features_lr=features_lr,
                neighbourhood_lr=neighbourhood_lr,
                lr_maps=lr_maps,
                y_index=yi,
                x_index=xi,
                query_y=query_y,
                query_x=query_x,
            )
            weight_hr = weight.view(1, 1, query_y.numel(), query_x.numel())
            if result is None:
                result = {
                    name: value * weight_hr
                    for name, value in prediction.items()
                }
            else:
                for name, value in prediction.items():
                    result[name] = result[name] + value * weight_hr

        if result is None:  # pragma: no cover - positive dimensions guarantee anchors
            raise RuntimeError("local ensemble produced no predictions")
        return result


class NSAMDRV17(nn.Module):
    """Active V17.1 candidate: LR encoder + local-ensemble implicit decoder."""

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
            "revision": "V17.1-proof",
            "scale": int(self.config.scale),
            "productionForward": (
                "LR aligned physical maps -> LR physical encoder -> "
                "four-anchor local implicit queries using relative dx/dy -> "
                "bilinear local-ensemble blend -> bounded physical residual -> "
                "deterministic B projection -> C"
            ),
            "absoluteUvCoordinatesUsed": False,
            "relativeSubpixelCoordinatesUsed": True,
            "periodicPhaseEncodingUsed": False,
            "localEnsembleUsed": True,
            "localEnsembleAnchors": 4,
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
