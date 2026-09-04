"""NSAMDR V10.7.9 explicit-parametric deterministic geometry redraw.

The synthetic structural proof no longer predicts a dense SDF or dense medial
field. B1b classifies one manufactured primitive family and regresses a compact
continuous parameter vector; exact analytic geometry then generates the HR SDF.
Panel 3 and Panel 2 continue to share the exact same BoundaryRenderer.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import V9Config
from .redistance import redistance_zero_contour, sdf_gradient_components
from .contours import build_guidance_numpy, contour_targets, lr_contour_prior
from .parametric_boundary import make_query_grid, supersample_coverage
from .parametric_primitives import ParametricPrimitiveField, PRIMITIVE_COUNT, PARAM_DIM
from .direct_coverage_specialist import BoundaryProfileSpecialist, BenefitSelector
from .seam_restoration import DirectionalSeamRestorer

MODEL_SCHEMA = "NSAMDR_RAVEN_PRODUCTION_PARAMETRIC_4X_V11_0_0"
UPSCALE_FACTOR = 4
INPUT_CHANNELS = 17


class ModelService:
    # Purpose: Implement parameter count for ModelService.
    # Called by: External callers and the owning workflow.
    # Calls: ModelService helpers where required.
    def parameter_count(self, model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    # Purpose: Implement model hash for ModelService.
    # Called by: External callers and the owning workflow.
    # Calls: ModelService helpers where required.
    def model_hash(self, model: nn.Module) -> str:
        digest = hashlib.sha256()
        with torch.no_grad():
            for name, value in sorted(model.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    # Purpose: Implement build model input for ModelService.
    # Called by: External callers and the owning workflow.
    # Calls: ModelService helpers where required.
    def build_model_input(
        self,
        albedo_rgb: np.ndarray,
        normal_xy: np.ndarray | None = None,
        material_rgb: np.ndarray | None = None,
        degradation_level: float = 1.0,
        uv_stretch: np.ndarray | None = None,
        chart_mask: np.ndarray | None = None,
        *,
        source_sdf_prior: np.ndarray | None = None,
        contour_sdf_max_distance_pixels: float = 24.0,
        target_scale: int = UPSCALE_FACTOR,
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
        if source_sdf_prior is None:
            # contour_targets measures distance in LR pixels. Normalising by
            # maxDistance/scale means multiplying this channel by the HR maxDistance
            # inside GeometryNet reconstructs distance directly in HR pixels.
            lr_max_distance = max(
                float(contour_sdf_max_distance_pixels) / max(int(target_scale), 1), 1.0
            )
            source_sdf_prior, _source_orientation, _source_edge, _source_confidence = lr_contour_prior(
                albedo, normal, material, 1.0 if material_rgb is not None else 0.0,
                max_distance=lr_max_distance,
            )
        source_sdf_prior = np.asarray(source_sdf_prior, dtype=np.float32)
        if source_sdf_prior.ndim == 2:
            source_sdf_prior = source_sdf_prior[..., None]
        if source_sdf_prior.shape[:2] != (height, width):
            raise ValueError(
                f"source_sdf_prior must match LR input size {(height, width)}, "
                f"got {source_sdf_prior.shape[:2]}"
            )
        source_sdf_prior = np.clip(source_sdf_prior[..., :1], -1.0, 1.0)
        return np.ascontiguousarray(
            np.concatenate((albedo, normal, material, guidance, source_sdf_prior), axis=-1)
            .transpose(2, 0, 1)
        )

    # Purpose: Implement architecture summary for ModelService.
    # Called by: External callers and the owning workflow.
    # Calls: ModelService helpers where required.
    def architecture_summary(self, model: FidelityResidualNetV9) -> Mapping[str, object]:
        config = model.config
        return {
            "schema": MODEL_SCHEMA,
            "inputChannels": INPUT_CHANNELS,
            "inputTile": [config.tile_size, config.tile_size],
            "outputTile": [config.tile_size * UPSCALE_FACTOR, config.tile_size * UPSCALE_FACTOR],
            "upscaleFactor": UPSCALE_FACTOR,
            "widths": list(config.widths),
            "blocksPerLevel": list(config.blocks_per_level),
            "decoderBlocks": list(config.decoder_blocks),
            "attention": f"local {config.attention_window}x{config.attention_window} bottleneck attention",
            "parameterCount": self.parameter_count(model),
            "upsampling": (
                "explicit compact primitive geometry -> analytic metric SDF + "
                "9-sample SDF-derived subpixel coverage; no independent structural coverage, "
                "PixelShuffle or transposed convolution"
            ),
            "proposalPolicy": (
                "B1 synthetic structural proof -> learned compact primitive hypotheses -> exact analytic SDF -> BoundaryRenderer; later stages train real-Raven seam/profile/detail/selector components"
            ),
            "geometryPath": (
                "GeometryNet consumes native LR physical maps, predicts a compact primitive class/parameter vector, and rasterizes its exact analytic primitive SDF before deterministic rendering."
            ),
            "boundaryRenderer": {
                "bandPixels": config.boundary_renderer_band_pixels,
                "samplePixels": config.boundary_renderer_sample_pixels,
                "hardWidthPixels": config.boundary_renderer_hard_width_pixels,
                "softWidthPixels": config.boundary_renderer_soft_width_pixels,
                "sharedAcrossPhysicalMaps": True,
                "topologySafeSideSampling": True,
                "rendererRevision": "V11-production-parametric-forward",
            },
            "profileSpecialist": {
                "kind": "small sliding-window direct coverage-profile network",
                "rgbAuthority": False,
                "bandPixels": config.boundary_specialist_band_pixels,
                "maxCoverageLogitDelta": config.boundary_specialist_logit_delta_max,
            },
            "benefitSelector": {
                "kind": "frozen-candidate local benefit selector",
                "rgbAuthority": False,
            },
            "detailPath": "GeometryConditionedDetailNet with explicit 2x/4x decoder; frozen during B1/B2 then trained on real Raven crops in V10.8",
            "seamPath": "DirectionalSeamRestorer: shared multi-map structure tensor + tangent smoothing + normal sharpening",
            "detailEnabled": bool(getattr(config, "detail_reconstruction_enabled", True)),
            "identityInitialization": True,
            "outputs": [
                "albedoRGB", "normalXY", "roughness", "emissive", "material",
                "topologyAnchoredContourSDF", "edgeOrientationXY", "boundaryHardness",
                "boundaryGate", "confidence",
            ],
            "presentationMode": "not included; physical reconstruction only",
        }

_model_service = ModelService()
parameter_count = _model_service.parameter_count
model_hash = _model_service.model_hash
build_model_input = _model_service.build_model_input
architecture_summary = _model_service.architecture_summary


class LayerNorm2d(nn.Module):
    # Purpose: Implement init for LayerNorm2d.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    # Purpose: Implement forward for LayerNorm2d.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        x = value.float()
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.epsilon)
        return (x * self.weight.float() + self.bias.float()).to(dtype)


class ResidualBlock(nn.Module):
    """Memory-safe convolutional residual block with identity initialisation."""

    # Purpose: Implement init for ResidualBlock.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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

    # Purpose: Implement forward for ResidualBlock.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(self.norm(value))
        residual = self.project(F.gelu(self.expand(residual)))
        return value + residual * self.scale


class WindowAttention2d(nn.Module):
    """Local bottleneck attention; never operates at reconstructed resolution."""

    # Purpose: Implement init for WindowAttention2d.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, channels: int, heads: int, window: int) -> None:
        super().__init__()
        self.channels = channels
        self.window = int(window)
        self.norm = LayerNorm2d(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    # Purpose: Implement forward for WindowAttention2d.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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
    # Purpose: Implement init for Downsample.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(input_channels),
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
        )

    # Purpose: Implement forward for Downsample.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class ResizeDecoderStage(nn.Module):
    """Bilinear resize + convolution; no PixelShuffle or transpose convolution."""

    # Purpose: Implement init for ResizeDecoderStage.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, blocks: int) -> None:
        super().__init__()
        self.pre = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.fuse = nn.Conv2d(output_channels + skip_channels, output_channels, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(output_channels) for _ in range(blocks)])

    # Purpose: Implement forward for ResizeDecoderStage.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = self.pre(value)
        return self.blocks(self.fuse(torch.cat((value, skip), dim=1)))


class ZeroHead(nn.Module):
    """Compact output head whose final layer starts at exact zero."""

    # Purpose: Implement init for ZeroHead.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, channels: int, outputs: int, *, bias: float = 0.0, weight_std: float = 0.0) -> None:
        super().__init__()
        hidden = max(16, min(channels, 64))
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, outputs, 1),
        )
        if float(weight_std) > 0.0:
            # V9.9.3: the raw level set must not start at an exactly flat critical
            # point. The perturbation is deliberately tiny and does not affect
            # the externally visible identity path because the boundary renderer
            # remains disabled until a geometry phase explicitly enables it.
            nn.init.normal_(self.body[-1].weight, mean=0.0, std=float(weight_std))
        else:
            nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, bias)

    # Purpose: Implement forward for ZeroHead.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class GeometryNet(nn.Module):
    """Canonical learned parametric geometry and shared boundary conditioning.

    The compact primitive classifier/regressor is the only production SDF
    authority. The encoder/decoder predicts the shared edge, orientation and
    hardness fields consumed by the renderer and downstream safety modules.
    """

    # Purpose: Implement init for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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

        self.parametric_primitive_field = ParametricPrimitiveField(
            INPUT_CHANNELS,
            int(getattr(config, "parametric_primitive_hidden_channels", 96)),
            upscale=UPSCALE_FACTOR,
            max_distance_pixels=float(config.contour_sdf_max_distance_pixels),
        )
        aux_channels = max(16, min(32, widths[0] // 3))
        self.aux_project = nn.Conv2d(widths[0], aux_channels, 1)
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
        # Axial orientation loss has a flat derivative at an exact (0, 0)
        # vector; a tiny deterministic-scale initialization keeps the head
        # trainable while downstream normalization bounds its authority.
        self.orientation_head = ZeroHead(aux_channels, 2, weight_std=1.0e-3)
        self.edge_head = ZeroHead(aux_channels, 1, bias=-2.0)
        self.hardness_head = ZeroHead(aux_channels, 1, bias=0.0)

    # Purpose: Implement set sdf residual limit for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def set_sdf_residual_limit(self, pixels: float) -> None:
        _ = pixels

    # Purpose: Implement encode for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def encode(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.stem(inputs)
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            skips.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            value = decoder(value, skip)

        source_prior_lr = inputs[:, 16:17].float().clamp(-1.0, 1.0)
        guidance = inputs[:, 9:14].float()
        # Give the topology field a short local path to the raw LR physical-map
        # evidence. Subpixel phase is observable in antialiased albedo/normal/
        # material transitions and must not depend on a deep encoder recovering
        # that information after several nonlinear stages.
        parametric_field = self.parametric_primitive_field(inputs, source_prior_lr)

        aux = self.aux_project(value)
        aux = F.interpolate(aux, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False)
        prior = self.prior_project(guidance)
        prior = F.interpolate(prior, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False)
        aux = self.aux_refine(aux + prior.to(aux.dtype))
        return {
            "source_sdf_prior_lr": source_prior_lr,
            "parametric_field": parametric_field,
            "aux": aux,
        }

    # Purpose: Implement forward for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode(inputs)
        aux = context["aux"]
        parametric_field = context["parametric_field"]
        active_field = parametric_field
        final_pixels = active_field["phi_pixels"].float()
        max_distance = float(self.config.contour_sdf_max_distance_pixels)
        source_prior_pixels = active_field["source_sdf_prior_pixels"].float()
        sdf_raw = final_pixels / max(max_distance, 1.0e-6)
        sdf = sdf_raw.clamp(-1.0, 1.0)
        source_prior_hr = source_prior_pixels / max(max_distance, 1.0e-6)

        gx, gy = sdf_gradient_components(final_pixels)
        gnorm = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        primitive_normal = torch.cat((gx / gnorm, gy / gnorm), dim=1)
        curvature = torch.zeros_like(final_pixels)
        zeros = torch.zeros_like(final_pixels)
        zero_source = torch.zeros_like(context["source_sdf_prior_lr"])
        zero_control2 = torch.zeros(
            (inputs.shape[0], 2, inputs.shape[-2] * 2, inputs.shape[-1] * 2),
            device=inputs.device, dtype=aux.dtype,
        )
        zero_hr2 = torch.zeros(
            (inputs.shape[0], 2, final_pixels.shape[-2], final_pixels.shape[-1]),
            device=inputs.device, dtype=aux.dtype,
        )
        zero_gate = torch.full_like(sdf, -8.0)
        return {
            "sdf": sdf.to(aux.dtype),
            "sdf_raw": sdf_raw.to(aux.dtype),
            "source_sdf_prior": source_prior_hr.to(aux.dtype),
            "source_sdf_prior_pixels": source_prior_pixels.to(aux.dtype),
            "implicit_feature_grid": aux,
            "implicit_source_sdf_prior_lr": context["source_sdf_prior_lr"],
            "primitive_normal": primitive_normal.to(aux.dtype),
            "primitive_curvature_hr": curvature.to(aux.dtype),
            "primitive_phi_pixels": final_pixels.to(aux.dtype),
            "contour_transport_control_pixels": zero_control2,
            "contour_transport_pixels": zero_hr2,
            "contour_dilation_control_pixels": zero_control2[:, 0:1],
            "contour_dilation_pixels": zeros.to(aux.dtype),
            "implicit_residual_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "implicit_direct_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "contour_normal_offset_source_pixels": zero_source.to(aux.dtype),
            "contour_normal_offset_coarse_pixels": zeros.to(aux.dtype),
            "contour_phase_offset_pixels": zeros.to(aux.dtype),
            "contour_normal_offset_pixels": zeros.to(aux.dtype),
            "coarse_sdf": sdf.to(aux.dtype),
            "coarse_sdf_pixels": final_pixels.to(aux.dtype),
            "coarse_sdf_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "sdf_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "sdf_residual_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "parametric_primitive_active": final_pixels.new_tensor(1.0).to(aux.dtype),
            "primitive_class_logits": (
                parametric_field["class_logits"].to(aux.dtype)
            ),
            "primitive_class_index": (
                parametric_field["class_index"]
            ),
            "primitive_params": (
                parametric_field["params"].to(aux.dtype)
            ),
            "primitive_params_by_class": (
                parametric_field["params_by_class"].to(aux.dtype)
            ),
            "primitive_confidence": (
                parametric_field["confidence"].to(aux.dtype)
            ),
            "orientation_raw": self.orientation_head(aux),
            "edge_logits": self.edge_head(aux),
            "hardness_logits": self.hardness_head(aux),
            "boundary_gate_logits": zero_gate.to(aux.dtype),
        }

    # Purpose: Implement query from outputs for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def query_from_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        phi = F.grid_sample(
            outputs["primitive_phi_pixels"].float(), query_grid.float(),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        return {
            "phi_pixels": phi,
            "primitive_phi_pixels": phi,
            "primitive_normal": F.grid_sample(
                outputs["primitive_normal"].float(), query_grid.float(), mode="bilinear",
                padding_mode="border", align_corners=False,
            ),
            "primitive_curvature": torch.zeros_like(phi),
            "transport_pixels": torch.zeros(
                (phi.shape[0], 2, phi.shape[-2], phi.shape[-1]),
                device=phi.device, dtype=phi.dtype,
            ),
            "dilation_pixels": torch.zeros_like(phi),
            "residual_pixels": phi,
            "direct_delta_pixels": phi,
        }


class BoundaryRenderer(nn.Module):
    """Differentiable implicit-contour renderer shared by every physical map.

    The renderer samples the deterministic reconstruction on the two sides of
    the predicted SDF and reconstructs a sub-pixel transition analytically.
    It therefore changes *where and how a boundary is represented* without
    allowing GeometryNet to invent texture values.
    """

    # Purpose: Implement init for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, config: V9Config) -> None:
        super().__init__()
        self.config = config

    # Purpose: Implement smooth01 for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _smooth01(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    # Purpose: Implement sdf gradient components for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _sdf_gradient_components(sdf_pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sdf_gradient_components(sdf_pixels)

    # Purpose: Implement normal from sdf for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @classmethod
    def _normal_from_sdf(cls, sdf_pixels: torch.Tensor) -> torch.Tensor:
        gx, gy = cls._sdf_gradient_components(sdf_pixels)
        length = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        return torch.cat((gx / length, gy / length), dim=1)

    # Purpose: Implement metricize sdf pixels for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _metricize_sdf_pixels(
        self,
        sdf_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy diagnostic helper: rebuild metric distance from an explicit zero contour.

        The legacy V9.8.4 ``phi / local_gradient`` calibration is intentionally
        gone.  Physical distance depends only on interpolated zero-crossing
        position and predicted sign, so uniformly scaling the learned level-set
        cannot widen or narrow the rendered boundary.
        """
        raw = sdf_pixels.float()
        gx, gy = sdf_gradient_components(raw)
        grad = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
        metric = redistance_zero_contour(
            raw, float(self.config.contour_sdf_max_distance_pixels)
        )
        # Compatibility telemetry slot: there is no normalization denominator in
        # this diagnostic path. Ones make the removal explicit without destabilising readers
        # that still consume the tensor.
        denominator = torch.ones_like(metric)
        return metric, grad, denominator

    # Purpose: Implement sample offset for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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

    # Purpose: Implement adaptive plateau sample for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _adaptive_plateau_sample(
        self,
        value: torch.Tensor,
        sdf_pixels: torch.Tensor,
        normal: torch.Tensor,
        sign: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Purpose: Implement box sum for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _box_sum(value: torch.Tensor, kernel: int) -> torch.Tensor:
        radius = kernel // 2
        padded = F.pad(value, (radius, radius, radius, radius), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1) * float(kernel * kernel)

    # Purpose: Implement geometry solved plateaus for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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

    # Purpose: Implement forward for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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
        metricize_sdf: bool = True,
        precomputed_metric_sdf_pixels: torch.Tensor | None = None,
        coverage_negative_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_sdf_pixels = sdf.float() * float(self.config.contour_sdf_max_distance_pixels)
        if precomputed_metric_sdf_pixels is not None:
            sdf_pixels = precomputed_metric_sdf_pixels.to(
                device=sdf.device, dtype=torch.float32, non_blocking=True
            )
            gx, gy = self._sdf_gradient_components(raw_sdf_pixels)
            raw_grad_norm = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
            metric_scale_denominator = torch.ones_like(sdf_pixels)
        elif metricize_sdf:
            sdf_pixels, raw_grad_norm, metric_scale_denominator = self._metricize_sdf_pixels(
                raw_sdf_pixels
            )
        else:
            sdf_pixels = raw_sdf_pixels
            gx, gy = self._sdf_gradient_components(raw_sdf_pixels)
            raw_grad_norm = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
            metric_scale_denominator = torch.ones_like(raw_grad_norm)
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
        # V10 can provide oracle-distilled overlap-aggregated coverage directly.
        # No raster redistance/pixel-centre occupancy is allowed to quantise the
        # renderer-facing contour in the active predicted path.
        if coverage_negative_override is not None:
            coverage_negative = coverage_negative_override.to(
                device=value.device, dtype=torch.float32, non_blocking=True
            ).clamp(0.0, 1.0)
            if coverage_negative.shape[-2:] != sdf_pixels.shape[-2:]:
                coverage_negative = F.interpolate(
                    coverage_negative, size=sdf_pixels.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).clamp(0.0, 1.0)
        else:
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
        metric_gx, metric_gy = self._sdf_gradient_components(sdf_pixels)
        metric_grad_norm = torch.sqrt(metric_gx.square() + metric_gy.square() + 1.0e-8)
        return rendered, {
            "sdf_pixels": sdf_pixels.to(value.dtype),
            "sdf_pixels_raw": raw_sdf_pixels.to(value.dtype),
            "sdf_pixels_metric": sdf_pixels.to(value.dtype),
            "sdf_raw_grad_norm": raw_grad_norm.to(value.dtype),
            "sdf_metric_grad_norm": metric_grad_norm.to(value.dtype),
            "sdf_metric_scale_denominator": metric_scale_denominator.to(value.dtype),
            "sdf_metricization_used": value.new_tensor(1.0 if (metricize_sdf or precomputed_metric_sdf_pixels is not None) else 0.0),
            "boundary_normal": normal.to(value.dtype),
            "hardness": hardness.to(value.dtype),
            "transition_width": transition_width.to(value.dtype),
            "boundary_gate": applied_gate.to(value.dtype),
            "boundary_gate_prediction": predicted_gate.to(value.dtype),
            "boundary_gate_probability": learned_gate.to(value.dtype),
            # Ungated physical reconstruction candidate. V10 uses this to
            # supervise gate authority from realised LR->HR benefit rather than
            # from edge presence alone.
            "reconstructed_candidate": reconstructed.to(value.dtype),
            "edge_probability": edge_probability.to(value.dtype),
            "plateau_confidence": plateau_confidence.to(value.dtype),
            "positive_side": positive_side.to(value.dtype),
            "negative_side": negative_side.to(value.dtype),
            "coverage_negative": coverage_negative.to(value.dtype),
            "forced_gate_used": value.new_tensor(1.0 if forced_gate is not None else 0.0),
        }


class GeometryConditionedDetailNet(nn.Module):
    """Full-resolution residual decoder conditioned by the accepted geometry.

    Unlike the legacy LR AppearanceNet, this branch has explicit 2x and 4x
    decoder stages, so it can reconstruct authored detail above the LR Nyquist
    limit. It never moves the parametric contour: geometry arrives as detached
    conditioning and the branch can only add bounded residuals to the already
    reconstructed physical maps.
    """

    GEOMETRY_CHANNELS = 6
    PHYSICAL_CHANNELS = 8  # albedo RGB + normal XY + material RGB

    # Purpose: Implement init for GeometryConditionedDetailNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def __init__(self, config: V9Config) -> None:
        super().__init__()
        base_channels = int(getattr(config, "detail_feature_channels", 48))
        mid_channels = int(getattr(config, "detail_mid_channels", 40))
        hr_channels = int(getattr(config, "detail_hr_channels", 32))
        input_channels = INPUT_CHANNELS + self.GEOMETRY_CHANNELS
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 5, padding=2),
            nn.GELU(),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels, dilation=2),
            ResidualBlock(base_channels),
        )
        self.up2_pre = nn.Conv2d(base_channels, mid_channels, 3, padding=1)
        self.up2_fuse = nn.Conv2d(
            mid_channels + self.PHYSICAL_CHANNELS + self.GEOMETRY_CHANNELS,
            mid_channels, 1,
        )
        self.up2_body = nn.Sequential(
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels, dilation=2),
        )
        self.up4_pre = nn.Conv2d(mid_channels, hr_channels, 3, padding=1)
        self.up4_fuse = nn.Conv2d(
            hr_channels + self.PHYSICAL_CHANNELS + self.GEOMETRY_CHANNELS,
            hr_channels, 1,
        )
        self.up4_body = nn.Sequential(
            ResidualBlock(hr_channels),
            ResidualBlock(hr_channels),
        )
        self.albedo_head = ZeroHead(hr_channels, 3)
        self.normal_head = ZeroHead(hr_channels, 2)
        self.material_head = ZeroHead(hr_channels, 3)
        self.confidence_head = ZeroHead(
            hr_channels, 1, bias=float(getattr(config, "detail_confidence_initial_bias", -0.5))
        )
        self.regret_head = ZeroHead(
            hr_channels, 1, bias=float(getattr(config, "detail_regret_initial_bias", 0.0))
        )

    # Purpose: Implement resize for GeometryConditionedDetailNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    # Purpose: Implement forward for GeometryConditionedDetailNet.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(
        self,
        inputs: torch.Tensor,
        base_albedo_hr: torch.Tensor,
        base_normal_hr: torch.Tensor,
        base_material_hr: torch.Tensor,
        geometry_condition_hr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lr_size = inputs.shape[-2:]
        geometry_lr = self._resize(geometry_condition_hr.float(), lr_size)
        x = self.encoder(torch.cat((inputs.float(), geometry_lr), dim=1))

        physical_hr = torch.cat((
            base_albedo_hr.float(), base_normal_hr.float(), base_material_hr.float()
        ), dim=1)
        size2 = (lr_size[0] * 2, lr_size[1] * 2)
        x = self._resize(x, size2)
        x = self.up2_pre(x)
        x = self.up2_fuse(torch.cat((
            x,
            self._resize(physical_hr, size2),
            self._resize(geometry_condition_hr.float(), size2),
        ), dim=1))
        x = self.up2_body(x)

        hr_size = base_albedo_hr.shape[-2:]
        x = self._resize(x, hr_size)
        x = self.up4_pre(x)
        x = self.up4_fuse(torch.cat((
            x, physical_hr, geometry_condition_hr.float()
        ), dim=1))
        x = self.up4_body(x)
        return {
            "albedo_raw": torch.tanh(self.albedo_head(x)),
            "normal_raw": torch.tanh(self.normal_head(x)),
            "material_raw": torch.tanh(self.material_head(x)),
            "confidence_logits": self.confidence_head(x),
            "regret_logits": self.regret_head(x),
        }


class FidelityResidualNetV9(nn.Module):
    """V10 oracle-distilled local SDF/coverage model with deterministic rendering."""

    # Purpose: Implement init for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
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
        self.boundary_specialist = BoundaryProfileSpecialist(
            in_channels=18,
            channels=int(getattr(config, "boundary_specialist_channels", 48)),
            max_logit_delta=float(getattr(config, "boundary_specialist_logit_delta_max", 16.0)),
        )
        self.benefit_selector = BenefitSelector(
            in_channels=12,
            channels=int(getattr(config, "benefit_selector_channels", 24)),
        )
        self.detail_net = GeometryConditionedDetailNet(config)
        self.seam_restorer = DirectionalSeamRestorer(config)

    # Purpose: Implement set trainable for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(enabled)

    # Purpose: Implement normalize xy for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _normalize_xy(value: torch.Tensor) -> torch.Tensor:
        length = torch.sqrt(value.float().square().sum(dim=1, keepdim=True) + 1e-6)
        limiter = torch.maximum(torch.ones_like(length), length / 0.999)
        return (value.float() / limiter).to(value.dtype)

    # Purpose: Implement safe direction for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _safe_direction(self, value: torch.Tensor) -> torch.Tensor:
        epsilon = float(self.config.orientation_normalization_epsilon)
        fp32 = value.float()
        denominator = torch.sqrt(fp32.square().sum(dim=1, keepdim=True) + epsilon * epsilon)
        return (fp32 / denominator).to(value.dtype)

    # Purpose: Implement source edge support for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _source_edge_support(inputs: torch.Tensor, radius: int) -> torch.Tensor:
        guidance = inputs[:, 8:16].float()
        luma_edge = torch.sqrt(guidance[:, 1:2].square() + guidance[:, 2:3].square() + 1e-8)
        normal_edge = guidance[:, 3:4].abs()
        material_edge = guidance[:, 4:5].abs()
        curvature = guidance[:, 5:6].abs() * 0.35
        support = torch.maximum(torch.maximum(luma_edge * 4.0, normal_edge * 2.5), material_edge * 2.5)
        support = torch.maximum(support, curvature).clamp(0.0, 1.0)
        radius = max(0, int(radius))
        if radius > 0:
            kernel = radius * 2 + 1
            support = F.max_pool2d(support, kernel_size=kernel, stride=1, padding=radius)
            support = F.avg_pool2d(F.pad(support, (1, 1, 1, 1), mode="replicate"), 3, 1)
        return support.clamp(0.0, 1.0)

    # Purpose: Implement laplacian scalar for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _laplacian_scalar(value: torch.Tensor) -> torch.Tensor:
        kernel = value.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        return F.conv2d(value.float(), kernel.view(1, 1, 3, 3), padding=1)

    # Purpose: Implement gradient xy scalar for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    @staticmethod
    def _gradient_xy_scalar(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = value.float()
        px = F.pad(x, (1, 1, 0, 0), mode="replicate")
        py = F.pad(x, (0, 0, 1, 1), mode="replicate")
        gx = 0.5 * (px[:, :, :, 2:] - px[:, :, :, :-2])
        gy = 0.5 * (py[:, :, 2:, :] - py[:, :, :-2, :])
        return gx, gy

    # Purpose: Implement set phase for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def set_phase(self, phase: str) -> None:
        """Freeze parameters by curriculum stage without changing the forward graph."""
        self._set_trainable(self.geometry_net, False)
        self._set_trainable(self.seam_restorer, False)
        self._set_trainable(self.boundary_specialist, False)
        self._set_trainable(self.benefit_selector, False)
        self._set_trainable(self.detail_net, False)

        if phase == "sdf-bootstrap":
            # The production geometry graph is trained directly: analytic
            # primitive SDF plus renderer-facing edge/orientation/hardness.
            self._set_trainable(self.geometry_net.stem, True)
            self._set_trainable(self.geometry_net.encoders, True)
            self._set_trainable(self.geometry_net.downsamples, True)
            self._set_trainable(self.geometry_net.decoders, True)
            self._set_trainable(self.geometry_net.aux_project, True)
            self._set_trainable(self.geometry_net.prior_project, True)
            self._set_trainable(self.geometry_net.aux_refine, True)
            self._set_trainable(self.geometry_net.orientation_head, True)
            self._set_trainable(self.geometry_net.edge_head, True)
            self._set_trainable(self.geometry_net.hardness_head, True)
            self._set_trainable(self.geometry_net.parametric_primitive_field, True)
        elif phase == "sdf-proof":
            self._set_trainable(self.geometry_net.parametric_primitive_field, True)
        elif phase == "seam-proof":
            self._set_trainable(self.seam_restorer.phase_sr, True)
        elif phase == "seam-authority":
            authority = getattr(self.seam_restorer, "authority", None)
            if authority is not None:
                self._set_trainable(authority, True)
            else:
                self._set_trainable(self.seam_restorer, True)
        elif phase == "gate-proof":
            self._set_trainable(self.boundary_specialist, True)
        elif phase == "detail-reconstruction":
            self._set_trainable(self.detail_net, True)
        elif phase in {"boundary-hardening", "physical-finetune"}:
            self._set_trainable(self.benefit_selector, True)

    # Purpose: Implement set parametric substage for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def set_parametric_substage(self, substage: str) -> None:
        """Select the classifier/regressor subset trained in the B1b curriculum."""
        if substage not in {"classifier", "parameters", "integration"}:
            raise ValueError(f"unsupported parametric substage: {substage}")
        field = self.geometry_net.parametric_primitive_field
        self._set_trainable(field, False)
        if substage in {"classifier", "integration"}:
            self._set_trainable(field.class_encoder, True)
            self._set_trainable(field.class_trunk, True)
            self._set_trainable(field.class_head, True)
        if substage in {"parameters", "integration"}:
            self._set_trainable(field.param_encoder, True)
            self._set_trainable(field.param_trunk, True)
            self._set_trainable(field.param_head, True)

    # Purpose: Implement architecture contract for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def architecture_contract(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "parent": type(self).__name__,
            "geometryModel": type(self.geometry_net).__name__,
            "renderer": type(self.boundary_renderer).__name__,
            "profileSpecialist": type(self.boundary_specialist).__name__,
            "benefitSelector": type(self.benefit_selector).__name__,
            "detailReconstructor": type(self.detail_net).__name__,
            "seamRestorer": type(self.seam_restorer).__name__,
            "productionComponents": {
                "geometry": "geometry_net",
                "structural representation": "geometry_net.parametric_primitive_field",
                "boundary renderer": "boundary_renderer",
                "boundary/profile": "boundary_specialist",
                "PhaseAwareSeamSR": "seam_restorer.phase_sr",
                "seam authority": "seam_restorer.authority",
                "conditioned detail": "detail_net",
                "albedo physical head": "detail_net.albedo_head",
                "normal physical head": "detail_net.normal_head",
                "material physical head": "detail_net.material_head",
                "confidence": "detail_net.confidence_head",
                "regret": "detail_net.regret_head",
                "BenefitSelector": "benefit_selector",
            },
            "directionalSeamEnabled": True,
            "geometryCanPaintRgb": False,
            "profileSpecialistCanPaintRgb": False,
            "profileSpecialistAuthority": "always-active bounded shared-coverage correction",
            "detailReconstructionEnabled": True,
            "geometryOutputs": ("source_sdf_prior", "parametric_primitive_geometry", "edge", "orientation", "hardness"),
            "geometryPrior": "native-LR physical maps plus observable LR SDF",
            "geometryPrediction": "learned primitive class and continuous parameters rendered as one analytic metric SDF",
            "b1bObjective": "supervised primitive classification, continuous parameters and analytic render consistency",
            "topologyGeometryFeatureSplit": True,
            "finiteWidthStrokeRepresentation": "explicit line/ellipse-oval/rounded-box/corner/parallel/ring/junction parameters; circles are zero-eccentricity ellipses and smoothness is guaranteed by analytic construction",
            "parametricPrimitiveClassCount": PRIMITIVE_COUNT,
            "parametricPrimitiveParamDim": PARAM_DIM,
            "parametricPrimitiveClassAccuracyRequired": float(getattr(self.config, "parametric_primitive_class_accuracy_required", 0.95)),
            "parametricPrimitiveParamMaeRequired": float(getattr(self.config, "parametric_primitive_param_mae_required", 0.040)),
            "parametricPrimitiveTrainTilesPerEpoch": int(getattr(self.config, "parametric_primitive_train_tiles_per_epoch", 448)),
            "parametricPrimitiveBatchSize": int(getattr(self.config, "parametric_primitive_batch_size", 14)),
            "parametricPrimitiveLrMultiplier": float(getattr(self.config, "parametric_primitive_lr_multiplier", 10.0)),
            "parametricPrimitiveClassifierEpochs": int(getattr(self.config, "parametric_primitive_classifier_epochs", 10)),
            "parametricPrimitiveParameterEpochs": int(getattr(self.config, "parametric_primitive_parameter_epochs", 16)),
            "parametricPrimitiveIntegrationEpochs": int(getattr(self.config, "parametric_primitive_integration_epochs", 6)),
            "parametricPrimitiveTraining": "checkpointed learned classifier/regressor is the sole structural authority",
            "parametricPrimitiveSpatialEncoding": "normalized LR SDF/guidance evidence + 8x8 spatial lattices + measured centroid/principal-axis seeds + bounded residual heads",
            "reconstructionPrimitive": "compact primitive geometry -> exact analytic metric SDF -> deterministic BoundaryRenderer; structural pixels are analytic redraw, not neural seam painting",
            "seamPrimitiveClasses": ("straight", "curve", "irregular"),
            "seamAuthority": "always-active learned PhaseAwareSeamSR reconstruction and authority",
            "seamDirectionalAngleBins": int(getattr(self.config, "seam_directional_angle_bins", 12)),
            "seamDirectionalKernelSize": int(getattr(self.config, "seam_directional_kernel_size", 7)),
            "seamPhaseAware4x": True,
            "seamPhaseOnlyReconstruction": False,
            "seamPhaseSrMaxDelta": float(getattr(self.config, "seam_phase_sr_max_delta", 0.40)),
            "ddsAwareDegradation": bool(getattr(self.config, "dds_codec_degradation_enabled", True)),
            "sharedAcrossPhysicalMaps": True,
            "stagedProofs": (
                "geometry-conditioning", "B1b-parametric-primitive", "B2-same-deterministic-redraw",
                "phase-aware-seam", "boundary-profile", "physical-detail", "benefit-selector",
            ),
            "moduloCoordinatePhase": False,
            "pointwiseFourierSdfAuthority": False,
            "rendererZeroContourRedistance": False,
            "rendererLocalSdfMetricization": False,
            "productionForward": "FidelityResidualNetV9.forward(inputs) with no override authority",
            "candidateAuthority": "one final selector path; no raw-candidate deployment mode",
            "detailAuthority": "bounded 4x albedo/normal/material residual conditioned on detached structural SDF, boundary normal, shared coverage, hardness and profile confidence",
            "detailUpsampling": "explicit LR -> 2x -> 4x decoder with physical-map and geometry fusion at both scales",
            "detailMovesContour": False,
            "selectorRequires": "BenefitSelector probability multiplied by learned confidence and regret suppression",
            "topologyFrozenDuringProof": True,
            "panel3StructuralTarget": "predicted explicit primitive geometry through the exact same deterministic BoundaryRenderer as GT geometry Panel 2",
            "structuralPixelAuthority": "deterministic renderer only; learned appearance modules cannot move geometry",
            "structuralCoverageAuthority": "derived only from the canonical parametric SDF then refined by the bounded shared profile specialist",
            "subpixelSamples": int(getattr(self.config, "implicit_boundary_supersample_grid", 3)) ** 2,
            "boundarySpecialistPatch": 17,
            "teacherRendererTarget": "training-only GT-SDF forced-gate forced-hardness Panel-2 evidence",
        }

    # Purpose: Detect whether a nested component input carries an autograd dependency.
    # Called by: _run_training_component
    # Calls: _contains_grad_tensor
    @staticmethod
    def _contains_grad_tensor(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.requires_grad)
        if isinstance(value, dict):
            return any(FidelityResidualNetV9._contains_grad_tensor(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(FidelityResidualNetV9._contains_grad_tensor(item) for item in value)
        return False

    # Purpose: Execute one production component with activation recomputation during training.
    # Called by: _forward_impl, _training_render_sdf_teacher
    # Calls: _contains_grad_tensor
    def _run_training_component(
        self,
        module: nn.Module,
        *args: object,
        **kwargs: object,
    ):
        checkpoint_enabled = bool(
            getattr(self.config, "training_activation_checkpointing", True)
        )
        if not checkpoint_enabled or not self.training or not torch.is_grad_enabled():
            return module(*args, **kwargs)

        parameter_grad = any(parameter.requires_grad for parameter in module.parameters())
        input_grad = any(self._contains_grad_tensor(value) for value in args) or any(
            self._contains_grad_tensor(value) for value in kwargs.values()
        )
        if not parameter_grad and not input_grad:
            return module(*args, **kwargs)

        # Non-reentrant checkpointing preserves the exact forward topology and
        # output structure while discarding internal saved activations until
        # backward recomputes them. Parameters/state_dict are unchanged.
        return checkpoint(
            module,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
            **kwargs,
        )

    # Purpose: Implement training render sdf teacher for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: _run_training_component and same-class helpers where required.
    def _training_render_sdf_teacher(
        self,
        inputs: torch.Tensor,
        sdf_override: torch.Tensor,
        gate_override: torch.Tensor,
        hardness_override: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Exact deterministic Panel-2 teacher."""
        source_albedo = inputs[:, 0:3].clamp(0.0, 1.0)
        source_normal = inputs[:, 3:5].clamp(-1.0, 1.0)
        source_material = inputs[:, 5:8].clamp(0.0, 1.0)
        baseline_albedo = F.interpolate(source_albedo, scale_factor=UPSCALE_FACTOR, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
        evidence_albedo = F.interpolate(source_albedo, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        baseline_normal = self._normalize_xy(F.interpolate(source_normal, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False))
        baseline_material = F.interpolate(source_material, scale_factor=UPSCALE_FACTOR, mode="nearest")
        if sdf_override.shape[-2:] != baseline_albedo.shape[-2:]:
            sdf_override = F.interpolate(sdf_override, size=baseline_albedo.shape[-2:], mode="bilinear", align_corners=False)
        zeros = torch.zeros_like(sdf_override)
        ones = torch.ones_like(sdf_override)
        teacher_albedo, boundary = self._run_training_component(
            self.boundary_renderer,
            baseline_albedo, sdf_override, zeros, zeros, zeros, ones,
            enabled=True, plateau_evidence=evidence_albedo,
            source_value_lr=source_albedo, forced_gate=gate_override,
            forced_hardness=hardness_override, metricize_sdf=False,
        )
        metric_pixels = boundary["sdf_pixels_metric"]
        teacher_normal, _ = self._run_training_component(
            self.boundary_renderer,
            baseline_normal, sdf_override, zeros, zeros, zeros, ones,
            enabled=True, plateau_evidence=baseline_normal,
            source_value_lr=source_normal, forced_gate=gate_override,
            forced_hardness=hardness_override, metricize_sdf=False,
            precomputed_metric_sdf_pixels=metric_pixels,
        )
        teacher_material, _ = self._run_training_component(
            self.boundary_renderer,
            baseline_material, sdf_override, zeros, zeros, zeros, ones,
            enabled=True, plateau_evidence=baseline_material,
            source_value_lr=source_material, forced_gate=gate_override,
            forced_hardness=hardness_override, metricize_sdf=False,
            precomputed_metric_sdf_pixels=metric_pixels,
        )
        return {
            "boundary_reconstructed_albedo": teacher_albedo,
            "boundary_reconstructed_normal": self._normalize_xy(teacher_normal),
            "boundary_reconstructed_material": teacher_material,
            # Exact Panel-2 shared coverage is the local specialist's teacher.
            # This is preferable to recreating an approximate profile from the
            # target SDF at pixel centres.
            "coverage_negative": boundary["coverage_negative"].float(),
            "sdf_pixels_metric": metric_pixels,
        }

    # Purpose: Implement continuous coverage for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _continuous_coverage(
        self,
        geometry: dict[str, torch.Tensor],
        hardness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _c, h, w = geometry["sdf"].shape
        hard_width = float(self.config.boundary_renderer_hard_width_pixels)
        soft_width = float(self.config.boundary_renderer_soft_width_pixels)
        transition_width = (soft_width + (hard_width - soft_width) * hardness.float()).clamp_min(0.20)
        grid_n = max(1, int(getattr(self.config, "implicit_boundary_supersample_grid", 3)))
        if grid_n <= 1:
            offsets = ((0.0, 0.0),)
        else:
            # Sample strictly inside each output pixel. For n=3: -1/3,0,+1/3.
            coords = [((i + 0.5) / grid_n - 0.5) for i in range(grid_n)]
            offsets = tuple((x, y) for y in coords for x in coords)

        def query_fn(grid: torch.Tensor) -> torch.Tensor:
            return self.geometry_net.query_from_outputs(geometry, grid)["phi_pixels"]

        sdf_coverage, center_phi, samples = supersample_coverage(
            query_fn,
            batch=b, height=h, width=w, device=geometry["sdf"].device,
            offsets=offsets, transition_width=transition_width,
            return_samples=True,
        )
        # V10.4 has no independent structural coverage head. Coverage is derived
        # only from the shared parametric SDF, so the rendered profile cannot
        # diverge from the geometry used for plateau sampling.
        return sdf_coverage, center_phi, samples

    # Purpose: Implement specialist features for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _specialist_features(
        self,
        baseline_albedo: torch.Tensor,
        candidate_albedo: torch.Tensor,
        sdf_pixels: torch.Tensor,
        normal: torch.Tensor,
        coverage: torch.Tensor,
        plateau_confidence: torch.Tensor,
        observed_support: torch.Tensor,
        edge_probability: torch.Tensor,
    ) -> torch.Tensor:
        baseline_gray = baseline_albedo.float().mean(dim=1, keepdim=True)
        candidate_gray = candidate_albedo.float().mean(dim=1, keepdim=True)
        baseline_gx, baseline_gy = self._gradient_xy_scalar(baseline_gray)
        candidate_gx, candidate_gy = self._gradient_xy_scalar(candidate_gray)
        curvature = self._laplacian_scalar(sdf_pixels).clamp(-4.0, 4.0) / 4.0
        return torch.cat((
            baseline_albedo.float(),
            candidate_albedo.float(),
            (sdf_pixels.float() / max(float(self.config.contour_sdf_max_distance_pixels), 1.0)).clamp(-1.0, 1.0),
            normal.float(),
            curvature,
            coverage.float(),
            plateau_confidence.float(),
            observed_support.float(),
            edge_probability.float(),
            (baseline_gx * 4.0).clamp(-1.0, 1.0),
            (baseline_gy * 4.0).clamp(-1.0, 1.0),
            (candidate_gx * 4.0).clamp(-1.0, 1.0),
            (candidate_gy * 4.0).clamp(-1.0, 1.0),
        ), dim=1)

    # Purpose: Implement selector features for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _selector_features(
        self,
        baseline_albedo: torch.Tensor,
        candidate_albedo: torch.Tensor,
        sdf_pixels: torch.Tensor,
        normal: torch.Tensor,
        coverage: torch.Tensor,
        profile_confidence: torch.Tensor,
        observed_support: torch.Tensor,
        edge_probability: torch.Tensor,
        detail_confidence: torch.Tensor | None = None,
        detail_regret: torch.Tensor | None = None,
    ) -> torch.Tensor:
        baseline_gray = baseline_albedo.float().mean(dim=1, keepdim=True)
        candidate_gray = candidate_albedo.float().mean(dim=1, keepdim=True)
        difference = (candidate_albedo.float() - baseline_albedo.float()).abs().mean(dim=1, keepdim=True)
        if detail_confidence is None:
            detail_confidence = torch.zeros_like(coverage)
        if detail_regret is None:
            detail_regret = torch.zeros_like(coverage)
        return torch.cat((
            baseline_gray,
            candidate_gray,
            difference,
            (sdf_pixels.float() / max(float(self.config.contour_sdf_max_distance_pixels), 1.0)).clamp(-1.0, 1.0),
            normal.float(),
            coverage.float(),
            profile_confidence.float(),
            observed_support.float(),
            edge_probability.float(),
            detail_confidence.float(),
            detail_regret.float(),
        ), dim=1)

    # Purpose: Compute signed baseline-relative structural residual authority.
    # Called by: _forward_impl
    # Calls: No same-class helper methods.
    @staticmethod
    def _structural_residual_weight(
        candidate_locality: torch.Tensor,
        structural_residual_gain: torch.Tensor | None,
        gate_override: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the local structural residual weight with exact-zero default authority.

        Production uses a learned signed gain in [-1, 1]. A teacher/oracle gate
        remains an explicit positive override so proof rendering can still request
        the complete analytic candidate without changing production semantics.
        """
        if gate_override is not None:
            weight = gate_override.to(
                device=candidate_locality.device, dtype=torch.float32, non_blocking=True
            ).clamp(0.0, 1.0)
            if weight.shape[-2:] != candidate_locality.shape[-2:]:
                weight = F.interpolate(
                    weight, size=candidate_locality.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            return weight * candidate_locality

        if structural_residual_gain is None:
            return torch.zeros_like(candidate_locality, dtype=torch.float32)
        gain = structural_residual_gain.to(
            device=candidate_locality.device, dtype=torch.float32, non_blocking=True
        )
        if gain.shape[-2:] != candidate_locality.shape[-2:]:
            gain = F.interpolate(
                gain, size=candidate_locality.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return gain.clamp(-1.0, 1.0) * candidate_locality

    # Purpose: Implement forward impl for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _forward_impl(
        self,
        inputs: torch.Tensor,
        *,
        sdf_override: torch.Tensor | None = None,
        gate_override: torch.Tensor | None = None,
        hardness_override: torch.Tensor | None = None,
        seam_authority_override: torch.Tensor | None = None,
        seam_tangent_override: torch.Tensor | None = None,
        phase_only_seam_teacher: bool = False,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"V9 input must be Nx{INPUT_CHANNELS}xHxW, got {tuple(inputs.shape)}")

        source_albedo = inputs[:, 0:3].clamp(0.0, 1.0)
        source_normal = inputs[:, 3:5].clamp(-1.0, 1.0)
        source_material = inputs[:, 5:8].clamp(0.0, 1.0)
        baseline_albedo = F.interpolate(source_albedo, scale_factor=UPSCALE_FACTOR, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
        plateau_evidence_albedo = F.interpolate(source_albedo, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        baseline_normal = self._normalize_xy(F.interpolate(source_normal, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False))
        baseline_material = F.interpolate(source_material, scale_factor=UPSCALE_FACTOR, mode="nearest")

        geometry = self._run_training_component(self.geometry_net, inputs)
        hardness = torch.sigmoid(geometry["hardness_logits"].float())
        if hardness_override is not None:
            hardness = hardness_override.to(device=inputs.device, dtype=torch.float32, non_blocking=True).clamp(0.0, 1.0)
            if hardness.shape[-2:] != geometry["sdf"].shape[-2:]:
                hardness = F.interpolate(hardness, size=geometry["sdf"].shape[-2:], mode="bilinear", align_corners=False)

        coverage_override = None
        if sdf_override is None:
            coverage_override, center_phi, implicit_phi_samples = self._continuous_coverage(geometry, hardness)
            render_sdf = (center_phi / float(self.config.contour_sdf_max_distance_pixels)).to(geometry["sdf"].dtype)
            metricize_render_sdf = False
        else:
            render_sdf = sdf_override.to(device=inputs.device, dtype=geometry["sdf"].dtype, non_blocking=True)
            if render_sdf.shape[-2:] != geometry["sdf"].shape[-2:]:
                render_sdf = F.interpolate(render_sdf, size=geometry["sdf"].shape[-2:], mode="bilinear", align_corners=False)
            center_phi = render_sdf.float() * float(self.config.contour_sdf_max_distance_pixels)
            implicit_phi_samples = center_phi.repeat(1, 9, 1, 1)
            metricize_render_sdf = False

        observed_source_support = self._source_edge_support(inputs, self.config.geometry_edge_support_radius)
        observed_support_hr = F.interpolate(observed_source_support, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        ones_gate = torch.ones_like(render_sdf)

        # First build the ungated physical candidate and expose its solved side
        # plateaus. V10 applies specialist coverage and selector authority only
        # after this deterministic reconstruction exists.
        _initial_albedo, boundary = self._run_training_component(
            self.boundary_renderer,
            baseline_albedo, render_sdf, geometry["edge_logits"], geometry["hardness_logits"],
            geometry["boundary_gate_logits"], observed_support_hr,
            enabled=True, plateau_evidence=plateau_evidence_albedo,
            source_value_lr=source_albedo, forced_gate=ones_gate,
            forced_hardness=hardness, metricize_sdf=metricize_render_sdf,
            coverage_negative_override=coverage_override,
        )
        metric_pixels = center_phi.float()
        initial_coverage = boundary["coverage_negative"].float()
        initial_candidate_albedo = (
            boundary["negative_side"].float() * initial_coverage
            + boundary["positive_side"].float() * (1.0 - initial_coverage)
        ).to(baseline_albedo.dtype)

        edge_probability = torch.sigmoid(geometry["edge_logits"].float())
        band_pixels = max(float(getattr(self.config, "boundary_specialist_band_pixels", 3.5)), 0.5)
        band_weight = torch.exp(-metric_pixels.abs() / band_pixels).clamp(0.0, 1.0)
        # Plateau reconstruction is a boundary-local operator. It must never
        # replace authored texture over the whole image simply because a proof
        # stage forces the candidate. Use a compact C1 band so pixels outside
        # the physical reconstruction neighbourhood remain *exactly* baseline.
        locality_pixels = max(float(getattr(self.config, "boundary_renderer_band_pixels", 3.5)), 0.5)
        locality_u = (1.0 - metric_pixels.abs() / locality_pixels).clamp(0.0, 1.0)
        candidate_locality = locality_u * locality_u * (3.0 - 2.0 * locality_u)
        specialist_features = self._specialist_features(
            baseline_albedo, initial_candidate_albedo, metric_pixels,
            boundary["boundary_normal"], initial_coverage, boundary["plateau_confidence"],
            observed_support_hr, edge_probability,
        )
        specialist = self._run_training_component(
            self.boundary_specialist,
            specialist_features, initial_coverage, band_weight
        )
        refined_coverage = specialist["coverage"].float()
        candidate_albedo = (
            boundary["negative_side"].float() * refined_coverage
            + boundary["positive_side"].float() * (1.0 - refined_coverage)
        ).to(baseline_albedo.dtype)

        # Solve normal/material plateaus once, then reuse the exact same refined
        # coverage so all physical maps share one geometry.
        _normal_initial, normal_boundary = self._run_training_component(
            self.boundary_renderer,
            baseline_normal, render_sdf, geometry["edge_logits"], geometry["hardness_logits"],
            geometry["boundary_gate_logits"], observed_support_hr,
            enabled=True, plateau_evidence=baseline_normal, source_value_lr=source_normal,
            forced_gate=ones_gate, forced_hardness=hardness, metricize_sdf=False,
            precomputed_metric_sdf_pixels=metric_pixels, coverage_negative_override=initial_coverage,
        )
        candidate_normal = self._normalize_xy((
            normal_boundary["negative_side"].float() * refined_coverage
            + normal_boundary["positive_side"].float() * (1.0 - refined_coverage)
        ).to(baseline_normal.dtype))
        _material_initial, material_boundary = self._run_training_component(
            self.boundary_renderer,
            baseline_material, render_sdf, geometry["edge_logits"], geometry["hardness_logits"],
            geometry["boundary_gate_logits"], observed_support_hr,
            enabled=True, plateau_evidence=baseline_material, source_value_lr=source_material,
            forced_gate=ones_gate, forced_hardness=hardness, metricize_sdf=False,
            precomputed_metric_sdf_pixels=metric_pixels, coverage_negative_override=initial_coverage,
        )
        candidate_material = (
            material_boundary["negative_side"].float() * refined_coverage
            + material_boundary["positive_side"].float() * (1.0 - refined_coverage)
        ).to(baseline_material.dtype)

        # V11.8 structural reconstruction is residual relative to B, not an
        # unconditional replacement inside every contour band. The gain head starts
        # at exactly zero, so a fresh model produces C == B until authored-Raven
        # evidence earns local correction authority.
        structural_residual_weight = self._structural_residual_weight(
            candidate_locality, geometry.get("structural_residual_gain"), gate_override
        )
        structural_gate = structural_residual_weight.abs().clamp(0.0, 1.0)

        boundary_albedo = (
            baseline_albedo.float()
            + structural_residual_weight
            * (candidate_albedo.float() - baseline_albedo.float())
        ).clamp(0.0, 1.0).to(baseline_albedo.dtype)
        boundary_normal_out = self._normalize_xy((
            baseline_normal.float()
            + structural_residual_weight
            * (candidate_normal.float() - baseline_normal.float())
        ).to(baseline_normal.dtype))
        boundary_material = (
            baseline_material.float()
            + structural_residual_weight
            * (candidate_material.float() - baseline_material.float())
        ).clamp(0.0, 1.0).to(baseline_material.dtype)

        # V10.7.1 seam path: manufactured panel seams are local vector features,
        # not solely material-boundary zero sets.  Refine the already reconstructed
        # physical maps with one shared multi-map orientation field.  This branch
        # cannot move the parametric contour; Stage-B still has to pass SDF topology,
        # jitter and roughness gates independently.
        profile_confidence = specialist["confidence"].float()
        seam = self._run_training_component(
            self.seam_restorer,
            boundary_albedo, boundary_normal_out, boundary_material,
            sdf_pixels=metric_pixels, coverage=refined_coverage,
            profile_confidence=profile_confidence, edge_probability=edge_probability,
            geometry_normal=boundary["boundary_normal"].float(),
            source_albedo=source_albedo, source_normal=source_normal, source_material=source_material,
            authority_override=seam_authority_override, tangent_override=seam_tangent_override,
            phase_only=bool(phase_only_seam_teacher),
            enabled=True,
        )
        pre_seam_albedo = boundary_albedo
        pre_seam_normal = boundary_normal_out
        pre_seam_material = boundary_material
        boundary_albedo = seam["albedo"]
        boundary_normal_out = seam["normal_xy"]
        boundary_material = seam["material"]

        sdf_condition = (
            metric_pixels.float() / max(float(self.config.contour_sdf_max_distance_pixels), 1.0)
        ).clamp(-1.0, 1.0)
        geometry_condition = torch.cat((
            sdf_condition,
            boundary["boundary_normal"].float(),
            refined_coverage.float(),
            hardness.float(),
            profile_confidence,
        ), dim=1).detach()
        detail = self._run_training_component(
            self.detail_net,
            inputs,
            boundary_albedo.detach(),
            boundary_normal_out.detach(),
            boundary_material.detach(),
            geometry_condition,
        )
        detail_confidence = torch.sigmoid(detail["confidence_logits"].float())
        detail_regret = torch.sigmoid(detail["regret_logits"].float())
        albedo_delta = (
            detail["albedo_raw"].float()
            * float(getattr(self.config, "detail_albedo_max_delta", 0.20))
        )
        normal_delta = (
            detail["normal_raw"].float()
            * float(getattr(self.config, "detail_normal_max_delta", 0.15))
        )
        material_delta_rgb = (
            detail["material_raw"].float()
            * float(getattr(self.config, "detail_material_max_delta", 0.18))
        )
        full_candidate_albedo = (boundary_albedo.float() + albedo_delta).clamp(0.0, 1.0)
        full_candidate_normal = self._normalize_xy(boundary_normal_out.float() + normal_delta)
        full_candidate_material = (boundary_material.float() + material_delta_rgb).clamp(0.0, 1.0)

        selector_features = self._selector_features(
            baseline_albedo, full_candidate_albedo, metric_pixels, boundary["boundary_normal"],
            refined_coverage, profile_confidence, observed_support_hr, edge_probability,
            detail_confidence, detail_regret,
        )
        selector_logits = self._run_training_component(
            self.benefit_selector, selector_features
        )
        selector_probability = torch.sigmoid(selector_logits.float())
        # Selector, confidence and regret always jointly own final authority.
        confidence_support = ((detail_confidence - 0.50) / 0.35).clamp(0.0, 1.0)
        regret_suppression = ((0.50 - detail_regret) / 0.35).clamp(0.0, 1.0)
        final_gate = selector_probability * confidence_support * regret_suppression

        albedo = (
            baseline_albedo.float() * (1.0 - final_gate)
            + full_candidate_albedo.float() * final_gate
        ).clamp(0.0, 1.0).to(baseline_albedo.dtype)
        normal_xy = self._normalize_xy((
            baseline_normal.float() * (1.0 - final_gate)
            + full_candidate_normal.float() * final_gate
        ).to(baseline_normal.dtype))
        material = (
            baseline_material.float() * (1.0 - final_gate)
            + full_candidate_material.float() * final_gate
        ).clamp(0.0, 1.0).to(baseline_material.dtype)
        emissive = material[:, 1:2]
        roughness = material[:, 2:3]
        class_centres = torch.linspace(0.0, 1.0, self.config.material_classes, device=inputs.device, dtype=material.dtype)
        material_logits = -((material[:, 0:1] - class_centres.view(1, -1, 1, 1)) ** 2) * 40.0
        orientation = self._safe_direction(geometry["orientation_raw"])
        confidence = selector_probability.clamp(1e-5, 1.0 - 1e-5)
        confidence_logits = torch.logit(confidence)
        zero_hr2 = torch.zeros((inputs.shape[0], 2, baseline_albedo.shape[-2], baseline_albedo.shape[-1]), device=inputs.device, dtype=baseline_albedo.dtype)
        zero_source2 = torch.zeros((inputs.shape[0], 2, inputs.shape[-2], inputs.shape[-1]), device=inputs.device, dtype=baseline_albedo.dtype)
        zero_albedo = torch.zeros_like(albedo_delta)
        zero_normal = torch.zeros_like(normal_delta)
        raw_grad_x, raw_grad_y = sdf_gradient_components(metric_pixels)
        raw_grad = torch.sqrt(raw_grad_x.square() + raw_grad_y.square() + 1.0e-8)

        return {
            "albedo": albedo,
            "normal_xy": normal_xy,
            "roughness": roughness,
            "emissive": emissive,
            "material": material,
            "material_logits": material_logits,
            "sdf": (metric_pixels / float(self.config.contour_sdf_max_distance_pixels)).to(albedo.dtype),
            "sdf_raw": geometry["sdf_raw"],
            "predicted_sdf_pixels": metric_pixels.to(albedo.dtype),
            "predicted_sdf_raw_pixels": metric_pixels.to(albedo.dtype),
            "predicted_sdf_redistanced_pixels": metric_pixels.to(albedo.dtype),
            "predicted_sdf_metric_pixels": metric_pixels.to(albedo.dtype),
            "source_sdf_prior": geometry["source_sdf_prior"],
            "source_sdf_prior_pixels": geometry["source_sdf_prior_pixels"],
            "implicit_feature_grid": geometry["implicit_feature_grid"],
            "implicit_source_sdf_prior_lr": geometry["implicit_source_sdf_prior_lr"],
            # V11.5 connected-spline supervision must survive the canonical
            # FidelityResidualNetV9.forward() boundary.  GeometryNet owns these
            # tensors; dropping them here silently zeroed every graph-specific B1
            # loss even though the spline graph was the renderer authority.
            "spline_graph_control_phi_pixels": geometry["spline_graph_control_phi_pixels"],
            "spline_graph_source_control_phi_pixels": geometry["spline_graph_source_control_phi_pixels"],
            "spline_control_point_h_lr": geometry["spline_control_point_h_lr"],
            "spline_control_point_v_lr": geometry["spline_control_point_v_lr"],
            "spline_source_control_point_h_lr": geometry["spline_source_control_point_h_lr"],
            "spline_source_control_point_v_lr": geometry["spline_source_control_point_v_lr"],
            "spline_control_tangent_h": geometry["spline_control_tangent_h"],
            "spline_control_tangent_v": geometry["spline_control_tangent_v"],
            "spline_control_displacement_h_lr": geometry["spline_control_displacement_h_lr"],
            "spline_control_displacement_v_lr": geometry["spline_control_displacement_v_lr"],
            "spline_graph_mask_h": geometry["spline_graph_mask_h"],
            "spline_graph_mask_v": geometry["spline_graph_mask_v"],
            # V11.4 sdf-proof supervises the final analytic control-lattice anchor.
            # GeometryNet already produces this field; expose it through the single
            # production output dictionary so the proof loss sees the live graph.
            "parametric_anchor_distance_pixels": geometry["parametric_anchor_distance_pixels"],
            "parametric_primitive_active": geometry["parametric_primitive_active"],
            "primitive_class_logits": geometry["primitive_class_logits"],
            "primitive_class_index": geometry["primitive_class_index"],
            "primitive_params": geometry["primitive_params"],
            "primitive_params_by_class": geometry["primitive_params_by_class"],
            "primitive_confidence": geometry["primitive_confidence"],
            "primitive_normal": geometry["primitive_normal"],
            "primitive_curvature_hr": geometry["primitive_curvature_hr"],
            "primitive_phi_pixels": geometry["primitive_phi_pixels"],
            "implicit_residual_pixels": geometry["implicit_residual_pixels"],
            "contour_transport_control_pixels": geometry["contour_transport_control_pixels"],
            "contour_transport_pixels": geometry["contour_transport_pixels"],
            "contour_dilation_control_pixels": geometry["contour_dilation_control_pixels"],
            "contour_dilation_pixels": geometry["contour_dilation_pixels"],
            "contour_normal_offset_source_pixels": geometry["contour_normal_offset_source_pixels"],
            "contour_normal_offset_coarse_pixels": geometry["contour_normal_offset_coarse_pixels"],
            "contour_phase_offset_pixels": geometry["contour_phase_offset_pixels"],
            "contour_normal_offset_pixels": geometry["contour_normal_offset_pixels"],
            "coarse_sdf": geometry["coarse_sdf"],
            "coarse_sdf_pixels": geometry["coarse_sdf_pixels"],
            "coarse_sdf_delta_pixels": geometry["coarse_sdf_delta_pixels"],
            "sdf_delta_pixels": geometry["sdf_delta_pixels"],
            "sdf_residual_pixels": geometry["sdf_residual_pixels"],
            "render_sdf": render_sdf,
            "sdf_pixels": metric_pixels.to(albedo.dtype),
            "sdf_pixels_raw": metric_pixels.to(albedo.dtype),
            "sdf_pixels_metric": metric_pixels.to(albedo.dtype),
            "implicit_phi_samples_pixels": implicit_phi_samples.float(),
            "sdf_raw_grad_norm": raw_grad.to(albedo.dtype),
            "sdf_redistanced_grad_norm": raw_grad.to(albedo.dtype),
            "sdf_metric_grad_norm": raw_grad.to(albedo.dtype),
            "sdf_metric_scale_denominator": torch.ones_like(metric_pixels).to(albedo.dtype),
            "sdf_metricization_used": albedo.new_tensor(0.0),
            "orientation": orientation,
            "edge_logits": geometry["edge_logits"],
            "hardness_logits": geometry["hardness_logits"],
            "boundary_gate_logits": selector_logits.to(albedo.dtype),
            "hardness": hardness.to(albedo.dtype),
            "transition_width": (float(self.config.boundary_renderer_soft_width_pixels) + (float(self.config.boundary_renderer_hard_width_pixels) - float(self.config.boundary_renderer_soft_width_pixels)) * hardness).to(albedo.dtype),
            "boundary_normal": boundary["boundary_normal"],
            "boundary_gate": structural_gate.to(albedo.dtype),
            "boundary_structural_residual_weight": structural_residual_weight.to(albedo.dtype),
            "structural_residual_gain": geometry.get(
                "structural_residual_gain", torch.zeros_like(structural_residual_weight)
            ).to(albedo.dtype),
            "boundary_candidate_locality": candidate_locality.to(albedo.dtype),
            "boundary_gate_prediction": selector_probability.to(albedo.dtype),
            "boundary_gate_probability": selector_probability.to(albedo.dtype),
            "forced_gate_used": albedo.new_tensor(1.0 if gate_override is not None else 0.0),
            "plateau_confidence": boundary["plateau_confidence"],
            "confidence": confidence.to(albedo.dtype),
            "confidence_logits": confidence_logits.to(albedo.dtype),
            "baseline_albedo": baseline_albedo,
            "plateau_evidence_albedo": plateau_evidence_albedo,
            "baseline_normal": baseline_normal,
            "baseline_material": baseline_material,
            "boundary_reconstructed_albedo": boundary_albedo,
            "boundary_pre_seam_albedo": pre_seam_albedo,
            "boundary_candidate_albedo": candidate_albedo,
            "boundary_initial_candidate_albedo": initial_candidate_albedo,
            "boundary_reconstructed_normal": boundary_normal_out,
            "boundary_pre_seam_normal": pre_seam_normal,
            "boundary_candidate_normal": candidate_normal,
            "boundary_reconstructed_material": boundary_material,
            "boundary_pre_seam_material": pre_seam_material,
            "boundary_candidate_material": candidate_material,
            "boundary_initial_coverage": initial_coverage.to(albedo.dtype),
            "boundary_refined_coverage": refined_coverage.to(albedo.dtype),
            "boundary_specialist_coverage_delta": specialist["coverage_delta"].to(albedo.dtype),
            "boundary_specialist_coverage_logit_delta": specialist["coverage_logit_delta"].to(albedo.dtype),
            "boundary_specialist_direct_coverage": specialist["direct_coverage"].to(albedo.dtype),
            "boundary_specialist_authority": specialist["authority"].to(albedo.dtype),
            "boundary_specialist_confidence": specialist["confidence"].to(albedo.dtype),
            "benefit_selector_logits": selector_logits.to(albedo.dtype),
            "benefit_selector_probability": selector_probability.to(albedo.dtype),
            "warped_baseline_albedo": boundary_albedo,
            "warped_baseline_normal": boundary_normal_out,
            "warped_baseline_material": boundary_material,
            "displacement": zero_hr2,
            "source_displacement": zero_source2,
            "raw_source_displacement": zero_source2,
            "displacement_gate": structural_gate.to(albedo.dtype),
            "source_displacement_gate": observed_source_support,
            "learned_source_displacement_gate": observed_source_support,
            "source_edge_support": observed_source_support,
            "observed_source_edge_support": observed_source_support,
            "seam_authority": seam["authority"].to(albedo.dtype),
            "seam_learned_authority": seam["learned_authority"].to(albedo.dtype),
            "seam_normal": seam["normal"].to(albedo.dtype),
            "seam_tangent": seam["tangent"].to(albedo.dtype),
            "seam_strength": seam["strength"].to(albedo.dtype),
            "seam_coherence": seam["coherence"].to(albedo.dtype),
            "seam_ridge": seam["ridge"].to(albedo.dtype),
            "seam_authority_forced": seam["authority_forced"].to(albedo.dtype),
            "seam_curvature": seam["curvature"].to(albedo.dtype),
            "seam_primitive_class": seam["primitive_class"].to(albedo.dtype),
            "seam_sharpen": seam["sharpen"].to(albedo.dtype),
            "seam_phase_delta": seam["phase_delta"].to(albedo.dtype),
            "seam_phase_mix": seam["phase_mix"].to(albedo.dtype),
            "seam_phase_only": seam["phase_only"].to(albedo.dtype),
            "detail_confidence_logits": detail["confidence_logits"].to(albedo.dtype),
            "detail_regret_logits": detail["regret_logits"].to(albedo.dtype),
            "detail_confidence": detail_confidence.to(albedo.dtype),
            "detail_regret": detail_regret.to(albedo.dtype),
            "detail_geometry_condition": geometry_condition.to(albedo.dtype),
            "detail_candidate_albedo": full_candidate_albedo.to(albedo.dtype),
            "detail_candidate_normal": full_candidate_normal.to(albedo.dtype),
            "detail_candidate_material": full_candidate_material.to(albedo.dtype),
            "final_selector_gate": final_gate.to(albedo.dtype),
            # Compatibility aliases retained for existing diagnostics.
            "appearance_gate": detail_confidence.to(albedo.dtype),
            "albedo_gate_medium": detail_confidence.to(albedo.dtype),
            "albedo_gate_fine": torch.zeros_like(detail_confidence).to(albedo.dtype),
            "normal_gate_medium": detail_confidence.to(albedo.dtype),
            "normal_gate_fine": torch.zeros_like(detail_confidence).to(albedo.dtype),
            "material_gate": detail_confidence.to(albedo.dtype),
            "albedo_delta_medium": albedo_delta.to(albedo.dtype),
            "albedo_delta_fine": zero_albedo,
            "normal_delta_medium": normal_delta.to(albedo.dtype),
            "normal_delta_fine": zero_normal,
            "material_delta": material_delta_rgb[:, 0:1].to(albedo.dtype),
            "appearance_enabled": albedo.new_tensor(1.0),
        }

    # Purpose: Implement forward for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the only deployable Raven graph.

        Production callers cannot replace geometry, renderer authority, gates,
        hardness, seam authority or cached intermediate state.
        """
        return self._forward_impl(inputs)

    # Purpose: Implement forward training for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: Same-class helpers where required.
    def _forward_training(
        self,
        inputs: torch.Tensor,
        *,
        teacher_sdf: torch.Tensor | None = None,
        teacher_gate: torch.Tensor | None = None,
        teacher_hardness: torch.Tensor | None = None,
        teacher_seam_authority: torch.Tensor | None = None,
        teacher_seam_tangent: torch.Tensor | None = None,
        phase_only_seam_teacher: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Training-only teacher/cache entry point; never used by inference."""
        return self._forward_impl(
            inputs,
            sdf_override=teacher_sdf,
            gate_override=teacher_gate,
            hardness_override=teacher_hardness,
            seam_authority_override=teacher_seam_authority,
            seam_tangent_override=teacher_seam_tangent,
            phase_only_seam_teacher=phase_only_seam_teacher,
        )
