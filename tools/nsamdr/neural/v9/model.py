"""NSAMDR V10.1 topology-anchored continuous SDF reconstruction.

GeometryNet predicts distance magnitudes on one shared LR control lattice while
source-derived control signs remain fixed. This moves existing zero crossings
without giving neighbouring LR cells independent contour ownership. The same
continuous field is queried for every subpixel renderer sample.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import V9Config
from .redistance import redistance_zero_contour, sdf_gradient_components
from .contours import build_guidance_numpy, contour_targets, lr_contour_prior
from .parametric_boundary import LocalParametricBoundaryDecoder, make_query_grid, supersample_coverage
from .direct_coverage_specialist import BoundaryProfileSpecialist, BenefitSelector

MODEL_SCHEMA = "NSAMDR_LOCAL_PARAMETRIC_BOUNDARY_FIELD_4X_V10_1_1"
UPSCALE_FACTOR = 4
INPUT_CHANNELS = 17


class ModelService:
    # Purpose: Implement parameter count for ModelService.
    # Called by: architecture_summary
    # Calls: No same-class helper methods.
    def parameter_count(self, model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters())

    # Purpose: Implement model hash for ModelService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def model_hash(self, model: nn.Module) -> str:
        digest = hashlib.sha256()
        with torch.no_grad():
            for name, value in sorted(model.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    # Purpose: Implement build model input for ModelService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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
    # Calls: parameter_count
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
                "topology-anchored shared zero-crossing SDF + deterministic two-sided plateau solve + "
                "9-sample SDF-derived subpixel coverage + direct local coverage-profile specialist; no independent structural coverage, "
                "PixelShuffle or transposed convolution"
            ),
            "proposalPolicy": (
                "deterministic baseline -> topology-anchored shared zero-crossing SDF -> optional local profile specialist -> frozen-candidate benefit selector"
            ),
            "geometryPath": (
                "GeometryNet encodes LR physical maps and the observable LR SDF; immutable source control signs preserve the contour graph while learned positive magnitudes place shared zero crossings before deterministic rendering."
            ),
            "boundaryRenderer": {
                "bandPixels": config.boundary_renderer_band_pixels,
                "samplePixels": config.boundary_renderer_sample_pixels,
                "hardWidthPixels": config.boundary_renderer_hard_width_pixels,
                "softWidthPixels": config.boundary_renderer_soft_width_pixels,
                "sharedAcrossPhysicalMaps": True,
                "topologySafeSideSampling": True,
                "rendererRevision": "V10.1.0-topology-anchored-zero-crossing-sdf",
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
            "appearancePath": "independent AppearanceNet; disabled during V10 structural proof",
            "appearanceEnabled": config.appearance_enabled,
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
    # Calls: No same-class helper methods.
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    # Purpose: Implement forward for LayerNorm2d.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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
    # Calls: No same-class helper methods.
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
    # Calls: No same-class helper methods.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(self.norm(value))
        residual = self.project(F.gelu(self.expand(residual)))
        return value + residual * self.scale


class WindowAttention2d(nn.Module):
    """Local bottleneck attention; never operates at reconstructed resolution."""

    # Purpose: Implement init for WindowAttention2d.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, channels: int, heads: int, window: int) -> None:
        super().__init__()
        self.channels = channels
        self.window = int(window)
        self.norm = LayerNorm2d(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    # Purpose: Implement forward for WindowAttention2d.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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
    # Calls: No same-class helper methods.
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(input_channels),
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
        )

    # Purpose: Implement forward for Downsample.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class ResizeDecoderStage(nn.Module):
    """Bilinear resize + convolution; no PixelShuffle or transpose convolution."""

    # Purpose: Implement init for ResizeDecoderStage.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, blocks: int) -> None:
        super().__init__()
        self.pre = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.fuse = nn.Conv2d(output_channels + skip_channels, output_channels, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(output_channels) for _ in range(blocks)])

    # Purpose: Implement forward for ResizeDecoderStage.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = self.pre(value)
        return self.blocks(self.fuse(torch.cat((value, skip), dim=1)))


class ZeroHead(nn.Module):
    """Compact output head whose final layer starts at exact zero."""

    # Purpose: Implement init for ZeroHead.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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
    # Calls: No same-class helper methods.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class GeometryNet(nn.Module):
    """Encode LR evidence and reconstruct a topology-anchored continuous SDF.

    The observable LR segmentation supplies the control-lattice sign graph. The
    network predicts only positive metric magnitudes, moving shared edge zero
    crossings without gaining authority to create new sign islands.
    """

    # Purpose: Implement init for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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

        feature_channels = int(getattr(config, "topology_field_feature_channels", 64))
        self.field_feature_project = nn.Sequential(
            nn.Conv2d(widths[0] + 16, feature_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(feature_channels),
            ResidualBlock(feature_channels, dilation=2),
        )
        self.local_boundary_decoder = LocalParametricBoundaryDecoder(
            feature_channels,
            int(getattr(config, "topology_field_hidden_channels", 96)),
            max_distance_pixels=float(config.contour_sdf_max_distance_pixels),
            max_offset_pixels=float(getattr(config, "parametric_boundary_max_offset_pixels", 6.0)),
            max_normal_correction=float(getattr(config, "parametric_boundary_max_normal_correction", 1.5)),
            max_curvature_per_pixel=float(getattr(config, "parametric_boundary_max_curvature_per_pixel", 0.35)),
            max_ribbon_half_width_pixels=float(getattr(config, "parametric_boundary_max_ribbon_half_width_pixels", 6.0)),
            control_scale=int(getattr(config, "parametric_boundary_control_scale", 1)),
            output_scale=int(getattr(config, "target_scale", UPSCALE_FACTOR)),
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
        self.orientation_head = ZeroHead(aux_channels, 2)
        self.edge_head = ZeroHead(aux_channels, 1, bias=-2.0)
        self.hardness_head = ZeroHead(aux_channels, 1, bias=0.0)

    # Purpose: Implement set sdf residual limit for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def set_sdf_residual_limit(self, pixels: float) -> None:
        _ = pixels

    # Purpose: Implement encode for GeometryNet.
    # Called by: forward
    # Calls: No same-class helper methods.
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
        direct_evidence = inputs[:, 0:16].to(value.dtype)
        field_features = self.field_feature_project(
            torch.cat((value, direct_evidence), dim=1)
        )
        field_context = self.local_boundary_decoder.build_context(field_features, source_prior_lr)

        aux = self.aux_project(value)
        aux = F.interpolate(aux, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False)
        prior = self.prior_project(guidance)
        prior = F.interpolate(prior, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False)
        aux = self.aux_refine(aux + prior.to(aux.dtype))
        return {
            "feature_grid": field_features,
            "source_sdf_prior_lr": source_prior_lr,
            "field_context": field_context,
            "aux": aux,
        }

    # Purpose: Implement forward for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: encode
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode(inputs)
        aux = context["aux"]
        field_context = context["field_context"]
        hr_height = inputs.shape[-2] * UPSCALE_FACTOR
        hr_width = inputs.shape[-1] * UPSCALE_FACTOR
        query_grid = make_query_grid(
            inputs.shape[0], hr_height, hr_width, device=inputs.device, dtype=torch.float32
        )
        field = self.local_boundary_decoder.query(field_context, query_grid)
        final_pixels = field["phi_pixels"].float()
        max_distance = float(self.config.contour_sdf_max_distance_pixels)
        source_prior_pixels = field["warped_source_pixels"].float()
        sdf_raw = final_pixels / max(max_distance, 1.0e-6)
        sdf = sdf_raw.clamp(-1.0, 1.0)
        source_prior_hr = source_prior_pixels / max(max_distance, 1.0e-6)

        control_anchor = field_context["anchor_distance_pixels"].float()
        source_control = F.interpolate(
            (context["source_sdf_prior_lr"].float() * max_distance),
            size=control_anchor.shape[-2:], mode="bilinear", align_corners=False,
        )
        source_control_sign = torch.where(
            source_control >= 0.0, torch.ones_like(source_control), -torch.ones_like(source_control)
        )
        magnitude_pixels = control_anchor.abs()
        log_magnitude_delta = torch.log(
            magnitude_pixels.clamp_min(1.0e-4) / source_control.abs().clamp_min(1.0e-4)
        )

        primitive_normal = field["primitive_normal"].float()
        curvature = field["primitive_curvature"].float()
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
            "implicit_feature_grid": context["feature_grid"],
            "implicit_source_sdf_prior_lr": context["source_sdf_prior_lr"],
            "topology_control_phi_pixels": control_anchor.to(aux.dtype),
            "topology_source_control_phi_pixels": source_control.to(aux.dtype),
            "topology_source_control_sign": source_control_sign.to(aux.dtype),
            "topology_magnitude_pixels": magnitude_pixels.to(aux.dtype),
            "topology_log_magnitude_delta": log_magnitude_delta.to(aux.dtype),
            "topology_field_confidence": field_context["confidence"].to(aux.dtype),
            "topology_edit_authority": field["implicit_authority"].to(aux.dtype),
            "topology_saddle_projection_fraction": zeros.to(aux.dtype),
            "primitive_normal": primitive_normal.to(aux.dtype),
            "primitive_curvature_hr": curvature.to(aux.dtype),
            "primitive_phi_pixels": field["primitive_phi_pixels"].to(aux.dtype),
            "parametric_anchor_distance_pixels": field_context["anchor_distance_pixels"].to(aux.dtype),
            "parametric_distance_delta_pixels": field_context["distance_delta_pixels"].to(aux.dtype),
            "branch_anchor_distance_pixels": field_context["branch_anchor_distance_pixels"].to(aux.dtype),
            "branch_normal_x": field_context["branch_normal_x"].to(aux.dtype),
            "branch_normal_y": field_context["branch_normal_y"].to(aux.dtype),
            "branch_curvature_per_pixel": field_context["branch_curvature_per_pixel"].to(aux.dtype),
            "branch_half_width_pixels": field_context["branch_half_width_pixels"].to(aux.dtype),
            "branch_ribbon_mode": field_context["branch_ribbon_mode"].to(aux.dtype),
            "branch_activation": field_context["branch_activation"].to(aux.dtype),
            "csg_logits": field_context["csg_logits"].to(aux.dtype),
            "parametric_confidence": field_context["confidence"].to(aux.dtype),
            "contour_transport_control_pixels": zero_control2,
            "contour_transport_pixels": zero_hr2,
            "contour_dilation_control_pixels": zero_control2[:, 0:1],
            "contour_dilation_pixels": zeros.to(aux.dtype),
            "implicit_residual_pixels": field["residual_pixels"].to(aux.dtype),
            "implicit_direct_delta_pixels": field["direct_delta_pixels"].to(aux.dtype),
            "contour_normal_offset_source_pixels": zero_source.to(aux.dtype),
            "contour_normal_offset_coarse_pixels": zeros.to(aux.dtype),
            "contour_phase_offset_pixels": zeros.to(aux.dtype),
            "contour_normal_offset_pixels": zeros.to(aux.dtype),
            "coarse_sdf": sdf.to(aux.dtype),
            "coarse_sdf_pixels": final_pixels.to(aux.dtype),
            "coarse_sdf_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "sdf_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "sdf_residual_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "orientation_raw": self.orientation_head(aux),
            "edge_logits": self.edge_head(aux),
            "hardness_logits": self.hardness_head(aux),
            "boundary_gate_logits": zero_gate.to(aux.dtype),
        }

    # Purpose: Implement query from outputs for GeometryNet.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def query_from_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = {
            "source_sdf_prior_lr": outputs["implicit_source_sdf_prior_lr"],
            "branch_anchor_distance_pixels": outputs["branch_anchor_distance_pixels"],
            "branch_normal_x": outputs["branch_normal_x"],
            "branch_normal_y": outputs["branch_normal_y"],
            "branch_curvature_per_pixel": outputs["branch_curvature_per_pixel"],
            "branch_half_width_pixels": outputs["branch_half_width_pixels"],
            "branch_ribbon_mode": outputs["branch_ribbon_mode"],
            "branch_activation": outputs["branch_activation"],
            "csg_logits": outputs["csg_logits"],
            "confidence": outputs["parametric_confidence"],
            "anchor_distance_pixels": outputs["parametric_anchor_distance_pixels"],
            "normal_x": outputs["branch_normal_x"][:, 0:1],
            "normal_y": outputs["branch_normal_y"][:, 0:1],
            "curvature_per_pixel": outputs["branch_curvature_per_pixel"][:, 0:1],
            "ribbon_half_width_pixels": outputs["branch_half_width_pixels"][:, 0:1],
            "ribbon_mode": outputs["branch_ribbon_mode"][:, 0:1],
            "distance_delta_pixels": outputs["parametric_distance_delta_pixels"],
            "junction_hint": outputs["branch_activation"][:, 1:].amax(dim=1, keepdim=True),
        }
        return self.local_boundary_decoder.query(context, query_grid)


class BoundaryRenderer(nn.Module):
    """Differentiable implicit-contour renderer shared by every physical map.

    The renderer samples the deterministic reconstruction on the two sides of
    the predicted SDF and reconstructs a sub-pixel transition analytically.
    It therefore changes *where and how a boundary is represented* without
    allowing GeometryNet to invent texture values.
    """

    # Purpose: Implement init for BoundaryRenderer.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, config: V9Config) -> None:
        super().__init__()
        self.config = config

    # Purpose: Implement smooth01 for BoundaryRenderer.
    # Called by: forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _smooth01(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    # Purpose: Implement sdf gradient components for BoundaryRenderer.
    # Called by: _normal_from_sdf, forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _sdf_gradient_components(sdf_pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sdf_gradient_components(sdf_pixels)

    # Purpose: Implement normal from sdf for BoundaryRenderer.
    # Called by: forward
    # Calls: _sdf_gradient_components
    @classmethod
    def _normal_from_sdf(cls, sdf_pixels: torch.Tensor) -> torch.Tensor:
        gx, gy = cls._sdf_gradient_components(sdf_pixels)
        length = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        return torch.cat((gx / length, gy / length), dim=1)

    # Purpose: Implement metricize sdf pixels for BoundaryRenderer.
    # Called by: forward
    # Calls: No same-class helper methods.
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
    # Called by: _adaptive_plateau_sample
    # Calls: No same-class helper methods.
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
    # Called by: forward
    # Calls: _sample_offset
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
    # Called by: _geometry_solved_plateaus
    # Calls: No same-class helper methods.
    @staticmethod
    def _box_sum(value: torch.Tensor, kernel: int) -> torch.Tensor:
        radius = kernel // 2
        padded = F.pad(value, (radius, radius, radius, radius), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1) * float(kernel * kernel)

    # Purpose: Implement geometry solved plateaus for BoundaryRenderer.
    # Called by: forward
    # Calls: _box_sum
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
    # Calls: _adaptive_plateau_sample, _geometry_solved_plateaus, _metricize_sdf_pixels, _normal_from_sdf, _sdf_gradient_components, _smooth01
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


class AppearanceNet(nn.Module):
    """Independent low-authority appearance residual model.

    V10.1 topology-field proof configs leave this disabled. If enabled later, geometry
    freezes before AppearanceNet is allowed to modify the already reconstructed
    physical maps.
    """

    # Purpose: Implement init for AppearanceNet.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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

    # Purpose: Implement forward for AppearanceNet.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.body(inputs)
        return {
            "albedo": self.albedo_head(value),
            "normal": self.normal_head(value),
            "material": self.material_head(value),
            "gate_logits": self.gate_head(value),
        }


class FidelityResidualNetV9(nn.Module):
    """V10 oracle-distilled local SDF/coverage model with deterministic rendering."""

    # Purpose: Implement init for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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
            in_channels=10,
            channels=int(getattr(config, "benefit_selector_channels", 24)),
        )
        self.appearance_net = AppearanceNet(config)
        self._boundary_enabled = False
        self._specialist_enabled = False
        self._selector_enabled = False
        self._appearance_enabled = bool(config.appearance_enabled)

    # Purpose: Implement set trainable for FidelityResidualNetV9.
    # Called by: set_phase
    # Calls: No same-class helper methods.
    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(enabled)

    # Purpose: Implement normalize xy for FidelityResidualNetV9.
    # Called by: forward, render_sdf_teacher
    # Calls: No same-class helper methods.
    @staticmethod
    def _normalize_xy(value: torch.Tensor) -> torch.Tensor:
        length = torch.sqrt(value.float().square().sum(dim=1, keepdim=True) + 1e-6)
        limiter = torch.maximum(torch.ones_like(length), length / 0.999)
        return (value.float() / limiter).to(value.dtype)

    # Purpose: Implement safe direction for FidelityResidualNetV9.
    # Called by: forward
    # Calls: No same-class helper methods.
    def _safe_direction(self, value: torch.Tensor) -> torch.Tensor:
        epsilon = float(self.config.orientation_normalization_epsilon)
        fp32 = value.float()
        denominator = torch.sqrt(fp32.square().sum(dim=1, keepdim=True) + epsilon * epsilon)
        return (fp32 / denominator).to(value.dtype)

    # Purpose: Implement source edge support for FidelityResidualNetV9.
    # Called by: forward
    # Calls: No same-class helper methods.
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
    # Called by: _specialist_features
    # Calls: No same-class helper methods.
    @staticmethod
    def _laplacian_scalar(value: torch.Tensor) -> torch.Tensor:
        kernel = value.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        return F.conv2d(value.float(), kernel.view(1, 1, 3, 3), padding=1)

    # Purpose: Implement gradient xy scalar for FidelityResidualNetV9.
    # Called by: _specialist_features
    # Calls: No same-class helper methods.
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
    # Calls: _set_trainable
    def set_phase(self, phase: str) -> None:
        """Strict specialist isolation.

        Historical phase identifiers are retained in the trainer CLI, but V9.9
        assigns them new authority:
          sdf-bootstrap/sdf-proof -> structural implicit geometry only
          gate-proof              -> boundary profile specialist only
          boundary-hardening      -> benefit selector only
          physical-finetune       -> selector calibration only (no joint polish)
        """
        structure = phase in {"sdf-bootstrap", "sdf-proof"}
        specialist = phase == "gate-proof"
        selector = phase in {"boundary-hardening", "physical-finetune"}
        self._set_trainable(self.geometry_net, structure)
        self._set_trainable(self.boundary_specialist, specialist)
        self._set_trainable(self.benefit_selector, selector)
        self._set_trainable(self.appearance_net, False)
        self._boundary_enabled = phase != "sdf-bootstrap"
        self._specialist_enabled = phase in {"gate-proof", "boundary-hardening", "physical-finetune"}
        self._selector_enabled = selector
        self._appearance_enabled = False

    # Purpose: Implement set inference mode for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def set_inference_mode(self) -> None:
        self._boundary_enabled = True
        self._specialist_enabled = True
        self._selector_enabled = True
        self._appearance_enabled = bool(self.config.appearance_enabled)

    # Purpose: Implement architecture contract for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def architecture_contract(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "parent": type(self).__name__,
            "geometryModel": type(self.geometry_net).__name__,
            "renderer": type(self.boundary_renderer).__name__,
            "profileSpecialist": type(self.boundary_specialist).__name__,
            "benefitSelector": type(self.benefit_selector).__name__,
            "geometryCanPaintRgb": False,
            "profileSpecialistCanPaintRgb": False,
            "profileSpecialistAuthority": "optional residual shared coverage correction after structure qualification",
            "appearanceEnabled": bool(self.config.appearance_enabled),
            "geometryOutputs": ("source_sdf_prior", "topology_control_sdf", "edge", "orientation", "hardness"),
            "geometryPrior": "observable LR multi-map segmentation SDF supplies immutable sign topology on a shared LR control lattice",
            "geometryPrediction": "network changes only positive metric distance magnitudes on the shared sign-fixed control lattice",
            "reconstructionPrimitive": "single topology-anchored bilinear zero-crossing field queried at nine subpixel samples then rendered by the deterministic Panel-2 renderer",
            "sharedAcrossPhysicalMaps": True,
            "stagedProofs": ("oracle-renderer", "topology-anchored-structure", "boundary-profile-specialist", "benefit-selector"),
            "moduloCoordinatePhase": False,
            "pointwiseFourierSdfAuthority": False,
            "rendererZeroContourRedistance": False,
            "rendererLocalSdfMetricization": False,
            "candidateAuthority": "compact boundary-local band only; exact baseline outside",
            "topologyFieldControlScale": int(getattr(self.config, "topology_field_control_scale", 1)),
            "topologyFieldMaxLogMagnitudeDelta": float(getattr(self.config, "topology_field_max_log_magnitude_delta", 8.0)),
            "topologyFieldEditBandPixels": float(getattr(self.config, "topology_field_edit_band_pixels", 12.0)),
            "topologyFieldSignAuthority": "source-derived immutable control signs; learned magnitudes only",
            "topologySaddleConnectivity": "source asymptotic decider preserved by hard shared-vertex projection",
            "structuralCoverageAuthority": "derived only from the shared continuous SDF; no independent patch coverage head",
            "subpixelSamples": int(getattr(self.config, "implicit_boundary_supersample_grid", 3)) ** 2,
            "boundarySpecialistPatch": 17,
            "teacherRendererTarget": "GT-SDF forced-gate forced-hardness Panel-2 teacher SDF/coverage/render",
        }

    # Purpose: Implement render sdf teacher for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: _normalize_xy
    def render_sdf_teacher(
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
        teacher_albedo, boundary = self.boundary_renderer(
            baseline_albedo, sdf_override, zeros, zeros, zeros, ones,
            enabled=True, plateau_evidence=evidence_albedo,
            source_value_lr=source_albedo, forced_gate=gate_override,
            forced_hardness=hardness_override, metricize_sdf=False,
        )
        metric_pixels = boundary["sdf_pixels_metric"]
        teacher_normal, _ = self.boundary_renderer(
            baseline_normal, sdf_override, zeros, zeros, zeros, ones,
            enabled=True, plateau_evidence=baseline_normal,
            source_value_lr=source_normal, forced_gate=gate_override,
            forced_hardness=hardness_override, metricize_sdf=False,
            precomputed_metric_sdf_pixels=metric_pixels,
        )
        teacher_material, _ = self.boundary_renderer(
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
    # Called by: forward
    # Calls: No same-class helper methods.
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
        # V10.1 deliberately has no independent structural coverage head.
        # Coverage is derived only from the same topology-anchored continuous
        # SDF queried by geometry, so a second patch field cannot punch periodic
        # holes into an otherwise coherent zero contour.
        return sdf_coverage, center_phi, samples

    # Purpose: Implement specialist features for FidelityResidualNetV9.
    # Called by: forward
    # Calls: _gradient_xy_scalar, _laplacian_scalar
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
    # Called by: forward
    # Calls: No same-class helper methods.
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
    ) -> torch.Tensor:
        baseline_gray = baseline_albedo.float().mean(dim=1, keepdim=True)
        candidate_gray = candidate_albedo.float().mean(dim=1, keepdim=True)
        difference = (candidate_albedo.float() - baseline_albedo.float()).abs().mean(dim=1, keepdim=True)
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
        ), dim=1)

    # Purpose: Implement forward for FidelityResidualNetV9.
    # Called by: External callers and the owning workflow.
    # Calls: _continuous_coverage, _normalize_xy, _safe_direction, _selector_features, _source_edge_support, _specialist_features
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
            raise ValueError(f"V9 input must be Nx{INPUT_CHANNELS}xHxW, got {tuple(inputs.shape)}")

        source_albedo = inputs[:, 0:3].clamp(0.0, 1.0)
        source_normal = inputs[:, 3:5].clamp(-1.0, 1.0)
        source_material = inputs[:, 5:8].clamp(0.0, 1.0)
        baseline_albedo = F.interpolate(source_albedo, scale_factor=UPSCALE_FACTOR, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
        plateau_evidence_albedo = F.interpolate(source_albedo, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        baseline_normal = self._normalize_xy(F.interpolate(source_normal, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False))
        baseline_material = F.interpolate(source_material, scale_factor=UPSCALE_FACTOR, mode="nearest")

        geometry = self.geometry_net(inputs)
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

        boundary_enabled = self._boundary_enabled if renderer_enabled_override is None else bool(renderer_enabled_override)
        observed_source_support = self._source_edge_support(inputs, self.config.geometry_edge_support_radius)
        observed_support_hr = F.interpolate(observed_source_support, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        ones_gate = torch.ones_like(render_sdf)

        # First build the ungated physical candidate and expose its solved side
        # plateaus. V10 applies specialist coverage and selector authority only
        # after this deterministic reconstruction exists.
        _initial_albedo, boundary = self.boundary_renderer(
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
        specialist = self.boundary_specialist(
            specialist_features, initial_coverage, band_weight
        )
        if self._specialist_enabled:
            refined_coverage = specialist["coverage"].float()
        else:
            refined_coverage = initial_coverage
        candidate_albedo = (
            boundary["negative_side"].float() * refined_coverage
            + boundary["positive_side"].float() * (1.0 - refined_coverage)
        ).to(baseline_albedo.dtype)

        # Solve normal/material plateaus once, then reuse the exact same refined
        # coverage so all physical maps share one geometry.
        _normal_initial, normal_boundary = self.boundary_renderer(
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
        _material_initial, material_boundary = self.boundary_renderer(
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

        selector_features = self._selector_features(
            baseline_albedo, candidate_albedo, metric_pixels, boundary["boundary_normal"],
            refined_coverage, specialist["confidence"], observed_support_hr, edge_probability,
        )
        selector_logits = self.benefit_selector(selector_features)
        selector_probability = torch.sigmoid(selector_logits.float())
        predicted_gate = (selector_probability * candidate_locality).clamp(0.0, 1.0)
        if gate_override is not None:
            applied_gate = gate_override.to(device=inputs.device, dtype=torch.float32, non_blocking=True).clamp(0.0, 1.0)
            if applied_gate.shape[-2:] != predicted_gate.shape[-2:]:
                applied_gate = F.interpolate(applied_gate, size=predicted_gate.shape[-2:], mode="bilinear", align_corners=False)
            applied_gate = applied_gate * candidate_locality
        elif self._selector_enabled and boundary_enabled:
            applied_gate = predicted_gate
        elif boundary_enabled and self._specialist_enabled:
            # Forced Panel-3 candidate authority is full *inside the compact
            # boundary band*, never global over authored texture.
            applied_gate = candidate_locality
        elif boundary_enabled and not self._selector_enabled:
            applied_gate = candidate_locality
        else:
            applied_gate = torch.zeros_like(predicted_gate)

        boundary_albedo = (baseline_albedo.float() * (1.0 - applied_gate) + candidate_albedo.float() * applied_gate).to(baseline_albedo.dtype)
        boundary_normal_out = self._normalize_xy((baseline_normal.float() * (1.0 - applied_gate) + candidate_normal.float() * applied_gate).to(baseline_normal.dtype))
        boundary_material = (baseline_material.float() * (1.0 - applied_gate) + candidate_material.float() * applied_gate).to(baseline_material.dtype)

        appearance = self.appearance_net(inputs)
        appearance_gate = torch.sigmoid(appearance["gate_logits"].float())
        appearance_gate = F.interpolate(appearance_gate, scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False)
        appearance_gate = appearance_gate * (1.0 - applied_gate.detach() * float(self.config.appearance_edge_suppression))
        if not self._appearance_enabled:
            appearance_gate = torch.zeros_like(appearance_gate)
        albedo_delta = F.interpolate(appearance["albedo"], scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False) * float(self.config.albedo_medium_delta) * appearance_gate
        normal_delta = F.interpolate(appearance["normal"], scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False) * float(self.config.normal_medium_delta) * appearance_gate
        material_delta_rgb = F.interpolate(appearance["material"], scale_factor=UPSCALE_FACTOR, mode="bilinear", align_corners=False) * float(self.config.material_delta) * appearance_gate
        albedo = (boundary_albedo + albedo_delta).clamp(0.0, 1.0)
        normal_xy = self._normalize_xy(boundary_normal_out + normal_delta)
        material = (boundary_material + material_delta_rgb).clamp(0.0, 1.0)
        emissive = material[:, 1:2]
        roughness = material[:, 2:3]
        class_centres = torch.linspace(0.0, 1.0, self.config.material_classes, device=inputs.device, dtype=material.dtype)
        material_logits = -((material[:, 0:1] - class_centres.view(1, -1, 1, 1)) ** 2) * 40.0
        orientation = self._safe_direction(geometry["orientation_raw"])
        confidence = predicted_gate.clamp(1e-5, 1.0 - 1e-5)
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
            "topology_control_phi_pixels": geometry["topology_control_phi_pixels"],
            "topology_source_control_phi_pixels": geometry["topology_source_control_phi_pixels"],
            "topology_source_control_sign": geometry["topology_source_control_sign"],
            "topology_magnitude_pixels": geometry["topology_magnitude_pixels"],
            "topology_log_magnitude_delta": geometry["topology_log_magnitude_delta"],
            "topology_field_confidence": geometry["topology_field_confidence"],
            "topology_edit_authority": geometry["topology_edit_authority"],
            "topology_saddle_projection_fraction": geometry["topology_saddle_projection_fraction"],
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
            "boundary_gate": applied_gate.to(albedo.dtype),
            "boundary_candidate_locality": candidate_locality.to(albedo.dtype),
            "boundary_gate_prediction": predicted_gate.to(albedo.dtype),
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
            "boundary_candidate_albedo": candidate_albedo,
            "boundary_initial_candidate_albedo": initial_candidate_albedo,
            "boundary_reconstructed_normal": boundary_normal_out,
            "boundary_candidate_normal": candidate_normal,
            "boundary_reconstructed_material": boundary_material,
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
            "displacement_gate": applied_gate.to(albedo.dtype),
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
            "appearance_enabled": albedo.new_tensor(1.0 if self._appearance_enabled else 0.0),
        }


MaterialPhysicalNet = FidelityResidualNetV9
