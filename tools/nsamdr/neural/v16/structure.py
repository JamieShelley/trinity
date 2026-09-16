"""Shared structural evidence and learned conditioning for NSAMDR V16.2.

The deterministic helpers in this module are diagnostic/training-target utilities.
The active learned path is StructureConditioningEncoder: it does not try to render
an explicit seam profile or claim an external geometry authority.  It extracts
edge/continuity features from the same eight LR physical-map channels already
available to V16, then supplies phase-neutral HR conditioning features to the
existing Swin reconstruction body.

This follows the evidence from the failed boundary-profile oracle: local geometry
plus a 1-D profile is too underconstrained.  Structure therefore conditions a
larger-context learned reconstruction instead of owning output pixels directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StructureTargetConfig:
    """Parameters used to derive deterministic shared structural diagnostics."""

    boundary_threshold: float = 0.28
    normalization_percentile: float = 95.0
    maximum_distance_pixels: float = 24.0
    junction_window: int = 7


def _as_hwc(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected HxWxC array, got shape {array.shape}")
    return array


def _vector_gradient_magnitude(value: np.ndarray) -> np.ndarray:
    """Return channel-aware first-order discontinuity magnitude."""

    array = _as_hwc(value)
    dx = np.zeros(array.shape[:2], dtype=np.float32)
    dy = np.zeros(array.shape[:2], dtype=np.float32)
    if array.shape[1] > 1:
        step = np.linalg.norm(array[:, 1:] - array[:, :-1], axis=2)
        dx[:, :-1] = step
        dx[:, -1] = step[:, -1]
    if array.shape[0] > 1:
        step = np.linalg.norm(array[1:] - array[:-1], axis=2)
        dy[:-1] = step
        dy[-1] = step[-1]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _robust_unit(value: np.ndarray, percentile: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    positive = array[array > 1.0e-8]
    if positive.size == 0:
        return np.zeros_like(array)
    scale = max(float(np.percentile(positive, percentile)), 1.0e-6)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def derive_structure_targets(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    config: StructureTargetConfig | None = None,
) -> dict[str, np.ndarray]:
    """Derive aligned structural diagnostics from authored physical maps.

    These targets are derived from authored A and therefore are never an inference
    authority.  Production conditioning uses LR evidence only.
    """

    cfg = config or StructureTargetConfig()
    albedo = _as_hwc(albedo)
    normal = _as_hwc(normal)
    material = _as_hwc(material)
    if albedo.shape[:2] != normal.shape[:2] or albedo.shape[:2] != material.shape[:2]:
        raise ValueError("albedo, normal and material maps must be spatially aligned")

    components = {
        "albedo": _robust_unit(
            _vector_gradient_magnitude(albedo), cfg.normalization_percentile
        ),
        "normal": _robust_unit(
            _vector_gradient_magnitude(normal), cfg.normalization_percentile
        ),
        "material": _robust_unit(
            _vector_gradient_magnitude(material), cfg.normalization_percentile
        ),
    }
    strength = np.maximum.reduce(tuple(components.values())).astype(np.float32)
    boundary = (strength >= float(cfg.boundary_threshold)).astype(np.uint8)

    inverse = (1 - boundary).astype(np.uint8)
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 5).astype(np.float32)
    distance = np.minimum(distance, float(cfg.maximum_distance_pixels))

    gx = cv2.Sobel(strength, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(strength, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    safe = np.maximum(norm, 1.0e-6)
    tangent_x = (-gy / safe).astype(np.float32)
    tangent_y = (gx / safe).astype(np.float32)
    tangent_x[norm < 1.0e-6] = 0.0
    tangent_y[norm < 1.0e-6] = 0.0

    window = max(3, int(cfg.junction_window) | 1)
    mean_tx = cv2.blur(tangent_x * strength, (window, window))
    mean_ty = cv2.blur(tangent_y * strength, (window, window))
    mean_strength = cv2.blur(strength, (window, window))
    coherence = np.sqrt(mean_tx * mean_tx + mean_ty * mean_ty).astype(np.float32)
    coherence /= np.maximum(mean_strength, 1.0e-6)
    junction = np.clip((1.0 - coherence) * strength, 0.0, 1.0).astype(np.float32)

    return {
        "boundaryStrength": strength,
        "boundaryMask": boundary.astype(bool),
        "distancePixels": distance,
        "tangentX": tangent_x,
        "tangentY": tangent_y,
        "junctionConfidence": junction,
        "componentStrength": components,
    }


def structure_channels(
    targets: Mapping[str, np.ndarray], maximum_distance_pixels: float = 24.0
) -> np.ndarray:
    """Pack deterministic diagnostics as HxWx5 float channels."""

    max_distance = max(float(maximum_distance_pixels), 1.0e-6)
    return np.stack(
        (
            np.asarray(targets["boundaryStrength"], dtype=np.float32),
            np.asarray(targets["distancePixels"], dtype=np.float32) / max_distance,
            np.asarray(targets["tangentX"], dtype=np.float32),
            np.asarray(targets["tangentY"], dtype=np.float32),
            np.asarray(targets["junctionConfidence"], dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


def _group_gradient(value: torch.Tensor) -> torch.Tensor:
    """Return one LR edge magnitude channel for a physical-map group."""

    value = value.float()
    dx = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt((dx * dx + dy * dy).mean(dim=1, keepdim=True) + 1.0e-8)


class StructureConditioningEncoder(nn.Module):
    """Extract learned structure features from the eight LR physical-map channels.

    Inputs are albedo RGB + normal XY + material RGB.  Three explicit gradient
    channels make discontinuity evidence easy to access, while convolutional blocks
    learn wider continuity/orientation context.  Upsampling is bilinear and the
    branch emits features only; it never directly renders physical-map pixels.
    """

    INPUT_CHANNELS = 8
    ANALYTIC_CHANNELS = 3

    def __init__(
        self,
        *,
        scale: int = 4,
        channels: int = 48,
        output_channels: int = 32,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be positive")
        if min(channels, output_channels, blocks) < 1:
            raise ValueError("conditioning dimensions must be positive")
        self.scale = int(scale)
        self.lr_stem = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS + self.ANALYTIC_CHANNELS, channels, 3, padding=1),
            nn.GELU(),
            *(_ResidualBlock(channels) for _ in range(int(blocks))),
        )
        self.hr_adapter = nn.Sequential(
            nn.Conv2d(channels, output_channels, 3, padding=1),
            nn.GELU(),
            _ResidualBlock(output_channels),
        )

    def analytic_edges(self, evidence_lr: torch.Tensor) -> torch.Tensor:
        if evidence_lr.ndim != 4 or evidence_lr.shape[1] != self.INPUT_CHANNELS:
            raise ValueError(
                "StructureConditioningEncoder expects N x 8 x H x W evidence, "
                f"got {tuple(evidence_lr.shape)}"
            )
        return torch.cat(
            (
                _group_gradient(evidence_lr[:, 0:3]),
                _group_gradient(evidence_lr[:, 3:5]),
                _group_gradient(evidence_lr[:, 5:8]),
            ),
            dim=1,
        )

    def forward(
        self,
        evidence_lr: torch.Tensor,
        *,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        edges = self.analytic_edges(evidence_lr)
        features_lr = self.lr_stem(torch.cat((evidence_lr.float(), edges), dim=1))
        size = target_size or (
            int(evidence_lr.shape[-2]) * self.scale,
            int(evidence_lr.shape[-1]) * self.scale,
        )
        features_hr = F.interpolate(
            features_lr,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return self.hr_adapter(features_hr)
