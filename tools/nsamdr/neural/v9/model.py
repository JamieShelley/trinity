"""NSAMDR V9.8.3 sign-gauge-invariant metric-SDF geometry-convergence network.

GeometryNet predicts a continuous contour field rather than an RGB correction
or displacement actuator. BoundaryRenderer samples the deterministic texture on
both sides of that contour and rebuilds a controlled sub-pixel transition. The
same boundary field is applied to albedo, normals and material semantics so
geometry and physical texture boundaries stay aligned.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import V9Config
from .contours import build_guidance_numpy

MODEL_SCHEMA = "NSAMDR_SIGN_GAUGE_METRIC_SDF_RENDERER_4X_V9_8_3"
UPSCALE_FACTOR = 4
INPUT_CHANNELS = 16


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        x = value.float()
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.epsilon)
        return (x * self.weight.float() + self.bias.float()).to(dtype)


class ResidualBlock(nn.Module):
    """Memory-safe convolutional residual block with identity initialisation."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.norm = LayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=padding, dilation=dilation, groups=channels
        )
        self.expand = nn.Conv2d(channels, channels * 3, 1)
        self.project = nn.Conv2d(channels * 3, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(self.norm(value))
        residual = self.project(F.gelu(self.expand(residual)))
        return value + residual * self.scale


class WindowAttention2d(nn.Module):
    """Local bottleneck attention; never operates at reconstructed resolution."""

    def __init__(self, channels: int, heads: int, window: int) -> None:
        super().__init__()
        self.channels = channels
        self.window = int(window)
        self.norm = LayerNorm2d(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        b, c, h, w = value.shape
        ws = min(self.window, h, w)
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = self.norm(value)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = x.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        x = x[..., :h, :w]
        return value + x * self.scale


class Downsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(input_channels),
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class ResizeDecoderStage(nn.Module):
    """Bilinear resize + convolution; no PixelShuffle or transpose convolution."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, blocks: int) -> None:
        super().__init__()
        self.pre = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.fuse = nn.Conv2d(output_channels + skip_channels, output_channels, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(output_channels) for _ in range(blocks)])

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = self.pre(value)
        return self.blocks(self.fuse(torch.cat((value, skip), dim=1)))


class ZeroHead(nn.Module):
    """Compact output head whose final layer starts at exact zero."""

    def __init__(self, channels: int, outputs: int, *, bias: float = 0.0, weight_std: float = 0.0) -> None:
        super().__init__()
        hidden = max(16, min(channels, 64))
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, outputs, 1),
        )
        if float(weight_std) > 0.0:
            # V9.8.3: the metric SDF must not start at an exactly flat critical
            # point. The perturbation is deliberately tiny and does not affect
            # the externally visible identity path because the boundary renderer
            # remains disabled until a geometry phase explicitly enables it.
            nn.init.normal_(self.body[-1].weight, mean=0.0, std=float(weight_std))
        else:
            nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class ImplicitSDFResidualHead(nn.Module):
    """Continuous bounded SDF residual decoder.

    V9.7 exposed a repeating modulo-4 coordinate phase to the decoder.  That
    made a periodic four-pixel sawtooth an easy local solution.  V9.8 removes
    that phase completely: the refiner receives only globally continuous
    coordinates plus the upsampled coarse SDF and predicts a small bounded
    correction in *output pixels*.
    """

    def __init__(self, feature_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + 3, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _global_coordinates(value: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = value.shape
        yy = (torch.arange(h, device=value.device, dtype=torch.float32) + 0.5) / max(h, 1)
        xx = (torch.arange(w, device=value.device, dtype=torch.float32) + 0.5) / max(w, 1)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack((gx * 2.0 - 1.0, gy * 2.0 - 1.0), dim=0).unsqueeze(0)
        return coords.expand(b, -1, -1, -1).to(value.dtype)

    def forward(self, features: torch.Tensor, coarse_sdf: torch.Tensor) -> torch.Tensor:
        if coarse_sdf.shape[-2:] != features.shape[-2:]:
            coarse_sdf = F.interpolate(
                coarse_sdf, size=features.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.net(torch.cat((features, coarse_sdf, self._global_coordinates(features)), dim=1))


class GeometryNet(nn.Module):
    """Predict a continuous boundary field; never predicts RGB/material values."""

    def __init__(self, config: V9Config) -> None:
        super().__init__()
        widths = config.widths
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, widths[0], 5, padding=2),
            nn.GELU(),
            ResidualBlock(widths[0]),
            ResidualBlock(widths[0]),
        )
        self.encoders = nn.ModuleList()
        for level, (channels, blocks) in enumerate(zip(widths, config.blocks_per_level)):
            layers: list[nn.Module] = []
            for index in range(blocks):
                dilation = 2 if level >= 2 and index % 3 == 2 else 1
                layers.append(ResidualBlock(channels, dilation=dilation))
            if level == len(widths) - 1:
                layers.append(WindowAttention2d(
                    channels, config.attention_heads, config.attention_window
                ))
            self.encoders.append(nn.Sequential(*layers))
        self.downsamples = nn.ModuleList([
            Downsample(widths[index], widths[index + 1])
            for index in range(len(widths) - 1)
        ])

        decoders: list[nn.Module] = []
        current = widths[-1]
        for decoder_index, skip_index in enumerate(range(len(widths) - 2, -1, -1)):
            output_channels = widths[skip_index]
            decoders.append(ResizeDecoderStage(
                current,
                widths[skip_index],
                output_channels,
                config.decoder_blocks[decoder_index],
            ))
            current = output_channels
        self.decoders = nn.ModuleList(decoders)

        # Geometry heads operate at HR through one narrow auxiliary projection.
        # None can directly paint colour, normals or material values.
        aux_channels = max(16, min(32, widths[0] // 3))
        self.aux_project = nn.Conv2d(widths[0], aux_channels, 1)
        # Directly inject observed multi-map edge guidance into the HR contour
        # branch. This anchors the learned zero-set to actual source boundaries
        # instead of requiring the decoder to rediscover edge location from scratch.
        self.prior_project = nn.Sequential(
            nn.Conv2d(5, aux_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(aux_channels, aux_channels, 1),
        )
        self.aux_refine = nn.Sequential(
            nn.Conv2d(aux_channels, aux_channels, 3, padding=1, groups=aux_channels),
            nn.GELU(),
            nn.Conv2d(aux_channels, aux_channels, 1),
            nn.GELU(),
            ResidualBlock(aux_channels),
        )
        # Coarse topology field is predicted at source resolution and then
        # continuously upsampled.  The HR implicit head may only make a bounded
        # correction, so it cannot invent an unrelated sawtooth zero-set.
        self.coarse_sdf_head = ZeroHead(
            widths[0], 1, weight_std=float(config.sdf_coarse_init_std)
        )
        self.sdf_residual_head = ImplicitSDFResidualHead(
            aux_channels, config.implicit_sdf_hidden_channels
        )
        self.orientation_head = ZeroHead(aux_channels, 2)
        self.edge_head = ZeroHead(aux_channels, 1, bias=-2.0)
        self.hardness_head = ZeroHead(aux_channels, 1, bias=0.0)
        self.boundary_gate_head = ZeroHead(
            aux_channels, 1, bias=config.boundary_gate_initial_bias
        )
        self._residual_limit_pixels = float(config.implicit_sdf_residual_pixels)

    def set_sdf_residual_limit(self, pixels: float) -> None:
        self._residual_limit_pixels = float(
            min(max(float(pixels), 0.0), float(self.config.implicit_sdf_residual_pixels))
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.stem(inputs)
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            skips.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            value = decoder(value, skip)

        aux = self.aux_project(value)
        aux = F.interpolate(
            aux, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False
        )
        # Guidance layout: severity, luma gx/gy, normal edge, material edge,
        # curvature, UV stretch, chart mask. Use the five geometric channels.
        prior = self.prior_project(inputs[:, 9:14].float())
        prior = F.interpolate(
            prior, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False
        )
        aux = self.aux_refine(aux + prior.to(aux.dtype))

        coarse_source = torch.tanh(
            self.coarse_sdf_head(value) * float(self.config.implicit_sdf_coordinate_scale)
        )
        coarse_hr = F.interpolate(
            coarse_source, size=aux.shape[-2:], mode="bilinear", align_corners=False
        )
        residual_unit = torch.tanh(self.sdf_residual_head(aux, coarse_hr))
        residual_pixels = residual_unit * float(self._residual_limit_pixels)
        max_distance = float(self.config.contour_sdf_max_distance_pixels)
        coarse_pixels = coarse_hr.float() * max_distance
        final_pixels = coarse_pixels + residual_pixels.float()
        sdf = (final_pixels / max(max_distance, 1.0e-6)).clamp(-1.0, 1.0)
        return {
            "sdf": sdf.to(aux.dtype),
            "sdf_raw": (final_pixels / max(max_distance, 1.0e-6)).to(aux.dtype),
            "coarse_sdf": coarse_hr.to(aux.dtype),
            "coarse_sdf_pixels": coarse_pixels.to(aux.dtype),
            "sdf_residual_pixels": residual_pixels.to(aux.dtype),
            "orientation_raw": self.orientation_head(aux),
            "edge_logits": self.edge_head(aux),
            "hardness_logits": self.hardness_head(aux),
            "boundary_gate_logits": self.boundary_gate_head(aux),
        }


class BoundaryRenderer(nn.Module):
    """Differentiable implicit-contour renderer shared by every physical map.

    The renderer samples the deterministic reconstruction on the two sides of
    the predicted SDF and reconstructs a sub-pixel transition analytically.
    It therefore changes *where and how a boundary is represented* without
    allowing GeometryNet to invent texture values.
    """

    def __init__(self, config: V9Config) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _smooth01(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _normal_from_sdf(sdf_pixels: torch.Tensor) -> torch.Tensor:
        padded = F.pad(sdf_pixels.float(), (1, 1, 1, 1), mode="replicate")
        gx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) * 0.5
        gy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) * 0.5
        length = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        return torch.cat((gx / length, gy / length), dim=1)

    @staticmethod
    def _sample_offset(
        value: torch.Tensor,
        normal: torch.Tensor,
        offset_pixels: float,
    ) -> torch.Tensor:
        b, _c, h, w = value.shape
        yy = (torch.arange(h, device=value.device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
        xx = (torch.arange(w, device=value.device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        base = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        dx = normal[:, 0].float() * float(offset_pixels) * (2.0 / max(w, 1))
        dy = normal[:, 1].float() * float(offset_pixels) * (2.0 / max(h, 1))
        grid = base + torch.stack((dx, dy), dim=-1)
        return F.grid_sample(
            value,
            grid.to(value.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def _adaptive_plateau_sample(
        self,
        value: torch.Tensor,
        sdf_pixels: torch.Tensor,
        normal: torch.Tensor,
        sign: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find a stable plateau without crossing another SDF boundary.

        V9.8 EXP_0002 exposed a concrete failure on thin diagonal lines: the old
        search could continue through the opposite side of a narrow feature and
        average background samples back into the feature, producing a double edge.
        The SDF already tells us which material side each candidate belongs to, so
        candidate samples are now accepted only while they remain on the requested
        signed side. Once the normal ray crosses another zero-set, all farther
        samples on that ray are rejected.
        """
        count = max(3, int(self.config.boundary_renderer_plateau_samples))
        base = float(self.config.boundary_renderer_sample_pixels)
        maximum = float(self.config.boundary_renderer_plateau_max_multiplier)
        multipliers = torch.linspace(0.45, maximum, count, device=value.device, dtype=torch.float32)

        samples = []
        side_fields = []
        for multiplier in multipliers:
            offset = sign * base * float(multiplier.item())
            samples.append(self._sample_offset(value, normal, offset).float())
            side_fields.append(self._sample_offset(sdf_pixels.float(), normal, offset).float())
        stack = torch.stack(samples, dim=1)  # B,S,C,H,W
        sampled_sdf = torch.stack(side_fields, dim=1)  # B,S,1,H,W

        # A small margin avoids treating interpolated zero-crossing samples as a
        # trustworthy plateau. Prefix validity prevents re-entering a thin feature
        # after the normal ray has already crossed its opposite boundary.
        signed_distance = sampled_sdf * float(sign)
        side_valid = (signed_distance > 0.15).to(torch.float32)
        prefix_valid = torch.cumprod(side_valid, dim=1)

        diffs = []
        for i in range(count):
            j = min(i + 1, count - 1)
            k = max(i - 1, 0)
            neighbour = 0.5 * (stack[:, j] + stack[:, k])
            diffs.append((stack[:, i] - neighbour).abs().mean(dim=1, keepdim=True))
        stability_error = torch.stack(diffs, dim=1)  # B,S,1,H,W

        # Prefer the most interior stable sample on the connected SDF side.
        # For a half-plane this naturally moves outward to the far plateau; for a
        # thin stripe/ring the signed-distance magnitude peaks near the centre and
        # falls again before the opposite zero-set, avoiding both boundary halos.
        scale = float(self.config.boundary_renderer_plateau_stability_scale)
        stability = torch.exp(-stability_error * scale).clamp_min(1.0e-8)
        interior_depth = signed_distance.clamp_min(0.0)
        depth_scale = max(float(self.config.boundary_renderer_sample_pixels), 1.0)
        score = torch.log(stability) + interior_depth / depth_scale * 0.85
        score = score.masked_fill(prefix_valid <= 0.0, -1.0e4)
        weights = torch.softmax(score, dim=1)

        # Complex junctions can have no candidate beyond the margin for a few
        # pixels. Fall back to the nearest candidate instead of emitting NaNs or
        # allowing the softmax to distribute over invalid samples.
        any_valid = prefix_valid.sum(dim=1, keepdim=True) > 0.0
        nearest = torch.zeros_like(weights)
        nearest[:, 0] = 1.0
        weights = torch.where(any_valid, weights, nearest)

        plateau = (stack * weights).sum(dim=1)
        selected_stability = stability
        confidence = (selected_stability * weights * prefix_valid).sum(dim=1).clamp(0.0, 1.0)
        return plateau.to(value.dtype), confidence.to(value.dtype)

    @staticmethod
    def _box_sum(value: torch.Tensor, kernel: int) -> torch.Tensor:
        radius = kernel // 2
        padded = F.pad(value, (radius, radius, radius, radius), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1) * float(kernel * kernel)

    def _geometry_solved_plateaus(
        self,
        source_value_lr: torch.Tensor,
        sdf_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover side plateaus from LR samples using metric-SDF coverage.

        The LR raster is an area-integrated observation.  A hard line can be
        narrower than one LR texel and an input may also contain a normalised
        blur kernel, so pointwise sampling cannot reliably recover its authored
        plateau.  V9.8.2 instead uses a local conservation equation:

            sum(observed) = A * sum(coverage) + B * sum(1-coverage)

        A robust positive-side estimate supplies B; A is then recovered from
        conserved energy.  This remains valid for a normalised blur because blur
        redistributes, rather than creates, local signal energy.  A symmetric
        least-squares fallback handles windows without enough pure-side support.
        """
        src = source_value_lr.float()
        _batch, channels, h_lr, w_lr = src.shape
        h_hr, w_hr = sdf_pixels.shape[-2:]
        if h_hr % h_lr != 0 or w_hr % w_lr != 0:
            raise ValueError("SDF/source dimensions are not integral for plateau solve")
        sy = h_hr // h_lr
        sx = w_hr // w_lr

        # The analytic training contract defines one-pixel coverage directly
        # from metric distance.  Avoid an extra smoothstep here: EXP_0003 showed
        # that biased occupancy becomes an amplified plateau error after inverse
        # mixing, especially on almost-horizontal/vertical lines.
        coverage_hr = (0.5 - sdf_pixels.float()).clamp(0.0, 1.0)
        coverage = F.avg_pool2d(
            coverage_hr, kernel_size=(sy, sx), stride=(sy, sx)
        ).clamp(0.0, 1.0)
        one_minus = 1.0 - coverage

        # A moderately long local window captures blur energy along the tangent
        # while remaining small enough not to become a global material average.
        kernel = 17
        window_area = float(kernel * kernel)

        # Robust estimate of the positive-SDF side.  The iterative weight rejects
        # halo/ringing samples that are geometrically on the correct side but are
        # still photometrically contaminated by the transition.
        pure_b_weight = ((one_minus - 0.80) / 0.20).clamp(0.0, 1.0).pow(4.0)
        support_b = self._box_sum(pure_b_weight, kernel)
        plateau_b = (
            self._box_sum(src * pure_b_weight, kernel)
            / support_b.clamp_min(1.0e-4)
        )
        for _ in range(2):
            deviation = (src - plateau_b).abs().mean(dim=1, keepdim=True)
            robust_weight = pure_b_weight * torch.exp(-deviation * 24.0)
            robust_support = self._box_sum(robust_weight, kernel)
            plateau_b = (
                self._box_sum(src * robust_weight, kernel)
                / robust_support.clamp_min(1.0e-4)
            )

        sum_coverage = self._box_sum(coverage, kernel)
        sum_outside = self._box_sum(one_minus, kernel)
        sum_value = self._box_sum(src, kernel)

        # Blur-invariant conservation solve for the negative-SDF side.
        plateau_a_energy = (
            sum_value - plateau_b * sum_outside
        ) / sum_coverage.clamp_min(1.0e-4)

        # Symmetric two-basis least-squares fallback for ambiguous windows.
        aa = self._box_sum(coverage.square(), kernel)
        bb = self._box_sum(one_minus.square(), kernel)
        ab = self._box_sum(coverage * one_minus, kernel)
        ra = self._box_sum(src * coverage, kernel)
        rb = self._box_sum(src * one_minus, kernel)
        determinant = (aa * bb - ab.square()).clamp_min(1.0e-4)
        plateau_a_ls = (ra * bb - rb * ab) / determinant
        plateau_b_ls = (rb * aa - ra * ab) / determinant

        reliable_b = support_b >= 1.0
        plateau_a = torch.where(reliable_b, plateau_a_energy, plateau_a_ls)
        plateau_b = torch.where(reliable_b, plateau_b, plateau_b_ls)

        # Physical map values remain bounded. Normal XY may be signed.
        lower = -1.0 if float(src.detach().amin().cpu()) < -1.0e-5 else 0.0
        plateau_a = plateau_a.clamp(lower, 1.0)
        plateau_b = plateau_b.clamp(lower, 1.0)

        plateau_a_hr = F.interpolate(
            plateau_a, size=(h_hr, w_hr), mode="bilinear", align_corners=False
        )
        plateau_b_hr = F.interpolate(
            plateau_b, size=(h_hr, w_hr), mode="bilinear", align_corners=False
        )

        # Confidence reflects geometric coverage of both sides plus robust
        # positive-side support. It is telemetry/gating evidence, not authority
        # over the Stage-A forced proof.
        support_a = self._box_sum(coverage, kernel) / window_area
        support_b_fraction = support_b / window_area
        support_outside = self._box_sum(one_minus, kernel) / window_area
        confidence_lr = (
            torch.minimum(support_a, support_outside)
            * 4.0
            * (0.5 + 0.5 * support_b_fraction.clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)
        confidence_hr = F.interpolate(
            confidence_lr, size=(h_hr, w_hr), mode="bilinear", align_corners=False
        )

        # Positive SDF is side B; negative SDF is side A.
        return (
            plateau_b_hr.to(source_value_lr.dtype),
            plateau_a_hr.to(source_value_lr.dtype),
            confidence_hr.to(source_value_lr.dtype),
        )

    def forward(
        self,
        value: torch.Tensor,
        sdf: torch.Tensor,
        edge_logits: torch.Tensor,
        hardness_logits: torch.Tensor,
        boundary_gate_logits: torch.Tensor,
        observed_support: torch.Tensor,
        *,
        enabled: bool,
        plateau_evidence: torch.Tensor | None = None,
        source_value_lr: torch.Tensor | None = None,
        forced_gate: torch.Tensor | None = None,
        forced_hardness: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sdf_pixels = sdf.float() * float(self.config.contour_sdf_max_distance_pixels)
        normal = self._normal_from_sdf(sdf_pixels)
        evidence = value if plateau_evidence is None else plateau_evidence.to(
            device=value.device, dtype=value.dtype, non_blocking=True
        )
        if evidence.shape[-2:] != value.shape[-2:]:
            evidence = F.interpolate(
                evidence, size=value.shape[-2:], mode="bilinear", align_corners=False
            )

        if source_value_lr is not None:
            positive_side, negative_side, plateau_confidence = self._geometry_solved_plateaus(
                source_value_lr.to(device=value.device, dtype=value.dtype, non_blocking=True),
                sdf_pixels,
            )
            plateau_confidence = plateau_confidence.float()
        else:
            positive_side, positive_plateau_confidence = self._adaptive_plateau_sample(
                evidence, sdf_pixels, normal, +1.0
            )
            negative_side, negative_plateau_confidence = self._adaptive_plateau_sample(
                evidence, sdf_pixels, normal, -1.0
            )
            plateau_confidence = torch.minimum(
                positive_plateau_confidence.float(), negative_plateau_confidence.float()
            )

        hardness = torch.sigmoid(hardness_logits.float())
        if forced_hardness is not None:
            hardness = forced_hardness.to(
                device=value.device, dtype=torch.float32, non_blocking=True
            ).clamp(0.0, 1.0)
            if hardness.shape[-2:] != sdf.shape[-2:]:
                hardness = F.interpolate(
                    hardness, size=sdf.shape[-2:], mode="bilinear", align_corners=False
                )
        hard_width = float(self.config.boundary_renderer_hard_width_pixels)
        soft_width = float(self.config.boundary_renderer_soft_width_pixels)
        transition_width = soft_width + (hard_width - soft_width) * hardness
        transition_width = transition_width.clamp_min(0.25)

        # Negative SDF selects the sample from the negative side of the contour.
        coverage_negative = self._smooth01(
            0.5 - sdf_pixels / transition_width
        )
        reconstructed = (
            negative_side.float() * coverage_negative
            + positive_side.float() * (1.0 - coverage_negative)
        ).to(value.dtype)

        edge_probability = torch.sigmoid(edge_logits.float())
        radius = max(1, int(round(float(self.config.boundary_renderer_band_pixels))))
        kernel = radius * 2 + 1
        edge_support = F.max_pool2d(
            edge_probability, kernel_size=kernel, stride=1, padding=radius
        )
        threshold = float(self.config.boundary_renderer_edge_threshold)
        edge_support = self._smooth01(
            (edge_support - threshold) / max(1.0e-4, 1.0 - threshold)
        )
        confidence_floor = float(self.config.boundary_renderer_confidence_floor)
        edge_support = (
            confidence_floor + (1.0 - confidence_floor) * edge_support
        ).clamp(0.0, 1.0)

        band = max(0.5, float(self.config.boundary_renderer_band_pixels))
        # The explicit gate head is the primary authority. Edge and SDF terms are
        # soft anchors, not hard multiplicative kill-switches; this avoids the
        # V9.5 failure where a mediocre SDF made the entire renderer inactive.
        distance_support = (
            0.35 + 0.65 * torch.exp(-sdf_pixels.abs() / band)
        ).clamp(0.0, 1.0)
        learned_gate = torch.sigmoid(boundary_gate_logits.float())
        support = observed_support.float().clamp(0.0, 1.0)
        predicted_gate = (
            learned_gate
            * (0.25 + 0.75 * edge_support)
            * distance_support
            * (0.20 + 0.80 * support)
            * (0.65 + 0.35 * plateau_confidence)
            * float(self.config.boundary_renderer_gate_gain)
        ).clamp(0.0, 1.0)
        if forced_gate is not None:
            teacher_gate = forced_gate.to(
                device=value.device, dtype=torch.float32, non_blocking=True
            ).clamp(0.0, 1.0)
            if teacher_gate.shape[-2:] != predicted_gate.shape[-2:]:
                teacher_gate = F.interpolate(
                    teacher_gate,
                    size=predicted_gate.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            applied_gate = teacher_gate if enabled else torch.zeros_like(teacher_gate)
        else:
            applied_gate = predicted_gate if enabled else torch.zeros_like(predicted_gate)

        rendered = (
            value.float() * (1.0 - applied_gate)
            + reconstructed.float() * applied_gate
        ).to(value.dtype)
        return rendered, {
            "sdf_pixels": sdf_pixels.to(value.dtype),
            "boundary_normal": normal.to(value.dtype),
            "hardness": hardness.to(value.dtype),
            "transition_width": transition_width.to(value.dtype),
            "boundary_gate": applied_gate.to(value.dtype),
            "boundary_gate_prediction": predicted_gate.to(value.dtype),
            "boundary_gate_probability": learned_gate.to(value.dtype),
            "edge_probability": edge_probability.to(value.dtype),
            "plateau_confidence": plateau_confidence.to(value.dtype),
            "forced_gate_used": value.new_tensor(1.0 if forced_gate is not None else 0.0),
        }


class AppearanceNet(nn.Module):
    """Independent low-authority appearance residual model.

    V9.8.3 sign-gauge metric-SDF proof configs leave this disabled. If enabled later, geometry
    freezes before AppearanceNet is allowed to modify the already reconstructed
    physical maps.
    """

    def __init__(self, config: V9Config) -> None:
        super().__init__()
        channels = max(40, min(64, config.widths[0]))
        self.body = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 5, padding=2),
            nn.GELU(),
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        self.albedo_head = ZeroHead(channels, 3)
        self.normal_head = ZeroHead(channels, 2)
        self.material_head = ZeroHead(channels, 3)
        self.gate_head = ZeroHead(channels, 1, bias=config.initial_gate_bias)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.body(inputs)
        return {
            "albedo": self.albedo_head(value),
            "normal": self.normal_head(value),
            "material": self.material_head(value),
            "gate_logits": self.gate_head(value),
        }


class FidelityResidualNetV9(nn.Module):
    """V9.8 coarse-to-fine geometry-convergence reconstruction parent model."""

    def __init__(self, config: V9Config | None = None, **overrides: object) -> None:
        super().__init__()
        config = V9Config() if config is None else config
        for name, value in overrides.items():
            if hasattr(config, name):
                setattr(config, name, value)
        config.validate()
        self.config = config
        self.geometry_net = GeometryNet(config)
        self.boundary_renderer = BoundaryRenderer(config)
        self.appearance_net = AppearanceNet(config)
        self._boundary_enabled = False
        self._appearance_enabled = bool(config.appearance_enabled)

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(enabled)

    @staticmethod
    def _normalize_xy(value: torch.Tensor) -> torch.Tensor:
        length = torch.sqrt(value.float().square().sum(dim=1, keepdim=True) + 1e-6)
        limiter = torch.maximum(torch.ones_like(length), length / 0.999)
        return (value.float() / limiter).to(value.dtype)

    def _safe_direction(self, value: torch.Tensor) -> torch.Tensor:
        epsilon = float(self.config.orientation_normalization_epsilon)
        fp32 = value.float()
        denominator = torch.sqrt(fp32.square().sum(dim=1, keepdim=True) + epsilon * epsilon)
        return (fp32 / denominator).to(value.dtype)

    @staticmethod
    def _source_edge_support(inputs: torch.Tensor, radius: int) -> torch.Tensor:
        """Broad source-grid support around observed physical boundaries."""
        guidance = inputs[:, 8:16].float()
        luma_edge = torch.sqrt(
            guidance[:, 1:2].square() + guidance[:, 2:3].square() + 1e-8
        )
        normal_edge = guidance[:, 3:4].abs()
        material_edge = guidance[:, 4:5].abs()
        curvature = guidance[:, 5:6].abs() * 0.35
        support = torch.maximum(
            torch.maximum(luma_edge * 4.0, normal_edge * 2.5),
            material_edge * 2.5,
        )
        support = torch.maximum(support, curvature).clamp(0.0, 1.0)
        radius = max(0, int(radius))
        if radius > 0:
            kernel = radius * 2 + 1
            support = F.max_pool2d(
                support, kernel_size=kernel, stride=1, padding=radius
            )
            support = F.avg_pool2d(
                F.pad(support, (1, 1, 1, 1), mode="replicate"), 3, 1
            )
        return support.clamp(0.0, 1.0)

    def set_phase(self, phase: str) -> None:
        contour_only = phase in {"sdf-bootstrap"}
        appearance_phase = bool(
            self.config.appearance_enabled
            and phase in {"appearance-reconstruction", "physical-finetune"}
        )
        if self.config.appearance_enabled:
            geometry_trainable = not appearance_phase
            appearance_trainable = appearance_phase
        else:
            geometry_trainable = True
            appearance_trainable = False
        self._set_trainable(self.geometry_net, geometry_trainable)
        if phase == "sdf-bootstrap":
            self.geometry_net.set_sdf_residual_limit(self.config.sdf_bootstrap_residual_pixels)
            self._set_trainable(self.geometry_net.sdf_residual_head, False)
        elif phase == "sdf-proof":
            self.geometry_net.set_sdf_residual_limit(self.config.sdf_proof_residual_pixels)
            self._set_trainable(self.geometry_net.sdf_residual_head, geometry_trainable)
        else:
            self.geometry_net.set_sdf_residual_limit(self.config.implicit_sdf_residual_pixels)
            self._set_trainable(self.geometry_net.sdf_residual_head, geometry_trainable)
        # Gate authority is frozen until the SDF has completed both bootstrap
        # and forced-gate proof phases.  A bad field can therefore not teach the
        # gate that the safest answer is "always off".
        gate_trainable = geometry_trainable and phase not in {"sdf-bootstrap", "sdf-proof"}
        self._set_trainable(self.geometry_net.boundary_gate_head, gate_trainable)
        self._set_trainable(self.appearance_net, appearance_trainable)
        self._boundary_enabled = not contour_only
        self._appearance_enabled = appearance_phase

    def set_inference_mode(self) -> None:
        """Enable the trained renderer after checkpoint loading."""
        self.geometry_net.set_sdf_residual_limit(self.config.implicit_sdf_residual_pixels)
        self._boundary_enabled = True
        self._appearance_enabled = bool(self.config.appearance_enabled)

    def architecture_contract(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "parent": type(self).__name__,
            "geometryModel": type(self.geometry_net).__name__,
            "renderer": type(self.boundary_renderer).__name__,
            "appearanceModel": type(self.appearance_net).__name__,
            "geometryCanPaintRgb": False,
            "appearanceEnabled": bool(self.config.appearance_enabled),
            "geometryOutputs": ("sdf", "edge", "orientation", "hardness", "boundary_gate"),
            "reconstructionPrimitive": "sign-gauge-metric-coarse-sdf-bounded-residual-monotonic-evidence-plateau-renderer",
            "sharedAcrossPhysicalMaps": True,
            "stagedProofs": ("oracle-renderer", "predicted-sdf-forced-gate", "full-predicted"),
            "moduloCoordinatePhase": False,
            "boundedSdfResidualPixels": float(self.config.implicit_sdf_residual_pixels),
            "sdfBootstrapResidualPixels": float(self.config.sdf_bootstrap_residual_pixels),
            "sdfProofResidualPixels": float(self.config.sdf_proof_residual_pixels),
            "plateauEvidence": "monotonic-bilinear-plus-local-coverage-deconvolution",
            "sdfSignGaugeInvariant": bool(self.config.sdf_sign_gauge_invariant),
            "sdfMetricBandPixels": float(self.config.sdf_metric_band_pixels),
            "sdfCoarseInitStd": float(self.config.sdf_coarse_init_std),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        sdf_override: torch.Tensor | None = None,
        gate_override: torch.Tensor | None = None,
        hardness_override: torch.Tensor | None = None,
        renderer_enabled_override: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"V9 input must be Nx{INPUT_CHANNELS}xHxW, got {tuple(inputs.shape)}"
            )

        source_albedo = inputs[:, 0:3].clamp(0.0, 1.0)
        source_normal = inputs[:, 3:5].clamp(-1.0, 1.0)
        source_material = inputs[:, 5:8].clamp(0.0, 1.0)

        baseline_albedo = F.interpolate(
            source_albedo,
            scale_factor=UPSCALE_FACTOR,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        plateau_evidence_albedo = F.interpolate(
            source_albedo,
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        baseline_normal = self._normalize_xy(F.interpolate(
            source_normal,
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        ))
        baseline_material = F.interpolate(
            source_material,
            scale_factor=UPSCALE_FACTOR,
            mode="nearest",
        )

        geometry = self.geometry_net(inputs)
        render_sdf = geometry["sdf"] if sdf_override is None else sdf_override.to(
            device=inputs.device, dtype=geometry["sdf"].dtype, non_blocking=True
        )
        if render_sdf.shape[-2:] != geometry["sdf"].shape[-2:]:
            render_sdf = F.interpolate(
                render_sdf,
                size=geometry["sdf"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        boundary_enabled = (
            self._boundary_enabled
            if renderer_enabled_override is None
            else bool(renderer_enabled_override)
        )
        observed_source_support = self._source_edge_support(
            inputs, self.config.geometry_edge_support_radius
        )
        observed_support_hr = F.interpolate(
            observed_source_support,
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        boundary_albedo, boundary = self.boundary_renderer(
            baseline_albedo,
            render_sdf,
            geometry["edge_logits"],
            geometry["hardness_logits"],
            geometry["boundary_gate_logits"],
            observed_support_hr,
            enabled=boundary_enabled,
            plateau_evidence=plateau_evidence_albedo,
            source_value_lr=source_albedo,
            forced_gate=gate_override,
            forced_hardness=hardness_override,
        )
        boundary_normal, _ = self.boundary_renderer(
            baseline_normal,
            render_sdf,
            geometry["edge_logits"],
            geometry["hardness_logits"],
            geometry["boundary_gate_logits"],
            observed_support_hr,
            enabled=boundary_enabled,
            plateau_evidence=baseline_normal,
            source_value_lr=source_normal,
            forced_gate=gate_override,
            forced_hardness=hardness_override,
        )
        boundary_normal = self._normalize_xy(boundary_normal)
        boundary_material, _ = self.boundary_renderer(
            baseline_material,
            render_sdf,
            geometry["edge_logits"],
            geometry["hardness_logits"],
            geometry["boundary_gate_logits"],
            observed_support_hr,
            enabled=boundary_enabled,
            plateau_evidence=baseline_material,
            source_value_lr=source_material,
            forced_gate=gate_override,
            forced_hardness=hardness_override,
        )

        appearance = self.appearance_net(inputs)
        appearance_gate_source = torch.sigmoid(appearance["gate_logits"])
        appearance_gate = F.interpolate(
            appearance_gate_source,
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        )
        appearance_gate = appearance_gate * (
            1.0
            - float(self.config.appearance_edge_suppression)
            * boundary["boundary_gate"].float()
        ).clamp(0.0, 1.0)
        appearance_scale = 1.0 if self._appearance_enabled else 0.0

        albedo_raw = F.interpolate(
            appearance["albedo"],
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        )
        normal_raw = F.interpolate(
            appearance["normal"],
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        )
        material_raw = F.interpolate(
            appearance["material"],
            scale_factor=UPSCALE_FACTOR,
            mode="bilinear",
            align_corners=False,
        )

        albedo_delta = (
            torch.tanh(albedo_raw)
            * float(self.config.albedo_medium_delta)
            * appearance_gate
            * appearance_scale
        )
        normal_delta = (
            torch.tanh(normal_raw)
            * float(self.config.normal_medium_delta)
            * appearance_gate
            * appearance_scale
        )
        material_delta_rgb = (
            torch.tanh(material_raw) * appearance_gate * appearance_scale
        )
        material_delta_rgb = material_delta_rgb * material_delta_rgb.new_tensor([
            float(self.config.material_delta),
            float(self.config.auxiliary_delta),
            float(self.config.auxiliary_delta),
        ]).view(1, 3, 1, 1)

        albedo = (boundary_albedo + albedo_delta).clamp(0.0, 1.0)
        normal_xy = self._normalize_xy(boundary_normal + normal_delta)
        material = (boundary_material + material_delta_rgb).clamp(0.0, 1.0)
        emissive = material[:, 1:2]
        roughness = material[:, 2:3]

        class_centres = torch.linspace(
            0.0,
            1.0,
            self.config.material_classes,
            device=inputs.device,
            dtype=material.dtype,
        )
        material_logits = -(
            (material[:, 0:1] - class_centres.view(1, -1, 1, 1)) ** 2
        ) * 40.0

        orientation = self._safe_direction(geometry["orientation_raw"])
        confidence = boundary["boundary_gate_prediction"].clamp(1e-5, 1.0 - 1e-5)
        confidence_logits = torch.logit(confidence)

        zero_hr2 = torch.zeros(
            (inputs.shape[0], 2, baseline_albedo.shape[-2], baseline_albedo.shape[-1]),
            device=inputs.device,
            dtype=baseline_albedo.dtype,
        )
        zero_source2 = torch.zeros(
            (inputs.shape[0], 2, inputs.shape[-2], inputs.shape[-1]),
            device=inputs.device,
            dtype=baseline_albedo.dtype,
        )
        zero_albedo = torch.zeros_like(albedo_delta)
        zero_normal = torch.zeros_like(normal_delta)

        return {
            "albedo": albedo,
            "normal_xy": normal_xy,
            "roughness": roughness,
            "emissive": emissive,
            "material": material,
            "material_logits": material_logits,
            "sdf": geometry["sdf"],
            "sdf_raw": geometry["sdf_raw"],
            "predicted_sdf_pixels": geometry["sdf"] * float(self.config.contour_sdf_max_distance_pixels),
            "coarse_sdf": geometry["coarse_sdf"],
            "coarse_sdf_pixels": geometry["coarse_sdf_pixels"],
            "sdf_residual_pixels": geometry["sdf_residual_pixels"],
            "render_sdf": render_sdf,
            "sdf_pixels": boundary["sdf_pixels"],
            "orientation": orientation,
            "edge_logits": geometry["edge_logits"],
            "hardness_logits": geometry["hardness_logits"],
            "boundary_gate_logits": geometry["boundary_gate_logits"],
            "hardness": boundary["hardness"],
            "transition_width": boundary["transition_width"],
            "boundary_normal": boundary["boundary_normal"],
            "boundary_gate": boundary["boundary_gate"],
            "boundary_gate_prediction": boundary["boundary_gate_prediction"],
            "boundary_gate_probability": boundary["boundary_gate_probability"],
            "forced_gate_used": boundary["forced_gate_used"],
            "plateau_confidence": boundary["plateau_confidence"],
            "confidence": confidence,
            "confidence_logits": confidence_logits,
            "baseline_albedo": baseline_albedo,
            "plateau_evidence_albedo": plateau_evidence_albedo,
            "baseline_normal": baseline_normal,
            "baseline_material": baseline_material,
            "boundary_reconstructed_albedo": boundary_albedo,
            "boundary_reconstructed_normal": boundary_normal,
            "boundary_reconstructed_material": boundary_material,
            # Compatibility aliases used by older diagnostics/loss names.
            "warped_baseline_albedo": boundary_albedo,
            "warped_baseline_normal": boundary_normal,
            "warped_baseline_material": boundary_material,
            "displacement": zero_hr2,
            "source_displacement": zero_source2,
            "raw_source_displacement": zero_source2,
            "displacement_gate": boundary["boundary_gate"],
            "source_displacement_gate": observed_source_support,
            "learned_source_displacement_gate": observed_source_support,
            "source_edge_support": observed_source_support,
            "observed_source_edge_support": observed_source_support,
            "appearance_gate": appearance_gate,
            "albedo_gate_medium": appearance_gate,
            "albedo_gate_fine": torch.zeros_like(appearance_gate),
            "normal_gate_medium": appearance_gate,
            "normal_gate_fine": torch.zeros_like(appearance_gate),
            "material_gate": appearance_gate,
            "albedo_delta_medium": albedo_delta,
            "albedo_delta_fine": zero_albedo,
            "normal_delta_medium": normal_delta,
            "normal_delta_fine": zero_normal,
            "material_delta": material_delta_rgb[:, 0:1],
            "appearance_enabled": albedo.new_tensor(
                1.0 if self._appearance_enabled else 0.0
            ),
        }


MaterialPhysicalNet = FidelityResidualNetV9


def build_model_input(
    albedo_rgb: np.ndarray,
    normal_xy: np.ndarray | None = None,
    material_rgb: np.ndarray | None = None,
    degradation_level: float = 1.0,
    uv_stretch: np.ndarray | None = None,
    chart_mask: np.ndarray | None = None,
) -> np.ndarray:
    albedo = np.asarray(albedo_rgb, dtype=np.float32)
    if albedo.max(initial=0.0) > 1.5:
        albedo = albedo / 255.0
    albedo = np.clip(albedo[..., :3], 0.0, 1.0)
    height, width = albedo.shape[:2]
    if normal_xy is None:
        luma = (
            albedo[..., 0] * 0.2126
            + albedo[..., 1] * 0.7152
            + albedo[..., 2] * 0.0722
        )
        gy, gx = np.gradient(luma)
        normal_xy = np.stack((gx, gy), axis=-1)
    normal = np.asarray(normal_xy, dtype=np.float32)[..., :2]
    length = np.sqrt(np.maximum((normal * normal).sum(axis=-1, keepdims=True), 1e-8))
    normal = normal / np.maximum(1.0, length / 0.999)
    if material_rgb is None:
        material = np.zeros((height, width, 3), dtype=np.float32)
    else:
        material = np.asarray(material_rgb, dtype=np.float32)
        if material.max(initial=0.0) > 1.5:
            material = material / 255.0
        material = np.clip(material[..., :3], 0.0, 1.0)
    guidance = build_guidance_numpy(
        albedo,
        normal,
        material,
        degradation_level,
        uv_stretch,
        chart_mask,
    )
    return np.ascontiguousarray(
        np.concatenate((albedo, normal, material, guidance), axis=-1)
        .transpose(2, 0, 1)
    )


def architecture_summary(model: FidelityResidualNetV9) -> Mapping[str, object]:
    config = model.config
    return {
        "schema": MODEL_SCHEMA,
        "inputChannels": INPUT_CHANNELS,
        "inputTile": [config.tile_size, config.tile_size],
        "outputTile": [
            config.tile_size * UPSCALE_FACTOR,
            config.tile_size * UPSCALE_FACTOR,
        ],
        "upscaleFactor": UPSCALE_FACTOR,
        "widths": list(config.widths),
        "blocksPerLevel": list(config.blocks_per_level),
        "decoderBlocks": list(config.decoder_blocks),
        "attention": (
            f"local {config.attention_window}x{config.attention_window} "
            "bottleneck attention"
        ),
        "parameterCount": parameter_count(model),
        "upsampling": (
            "coarse SDF + bounded continuous HR residual + adaptive-plateau boundary renderer; "
            "no PixelShuffle/transposed convolution"
        ),
        "proposalPolicy": (
            "deterministic baseline -> shared implicit-boundary reconstruction "
            "-> optional frozen-geometry appearance residual"
        ),
        "geometryPath": (
            "GeometryNet predicts a coarse topology SDF plus bounded continuous residual, edge probability, tangent/orientation, "
            "boundary hardness and an explicit benefit gate; no RGB authority"
        ),
        "boundaryRenderer": {
            "bandPixels": config.boundary_renderer_band_pixels,
            "samplePixels": config.boundary_renderer_sample_pixels,
            "hardWidthPixels": config.boundary_renderer_hard_width_pixels,
            "softWidthPixels": config.boundary_renderer_soft_width_pixels,
            "farSampleMultiplier": config.boundary_renderer_far_sample_multiplier,
            "farSampleWeight": config.boundary_renderer_far_sample_weight,
            "gateGain": config.boundary_renderer_gate_gain,
            "sharedAcrossPhysicalMaps": True,
            "topologySafeSideSampling": True,
            "rendererRevision": "V9.8.2-robust-conservation-deconvolution",
        },
        "appearancePath": (
            "independent AppearanceNet; disabled for V9.8.3 sign-gauge metric-SDF proof"
        ),
        "appearanceEnabled": config.appearance_enabled,
        "identityInitialization": True,
        "outputs": [
            "albedoRGB",
            "normalXY",
            "roughness",
            "emissive",
            "material",
            "contourSDF",
            "edgeOrientationXY",
            "boundaryHardness",
            "boundaryGate",
            "confidence",
        ],
        "presentationMode": "not included; physical reconstruction only",
    }
