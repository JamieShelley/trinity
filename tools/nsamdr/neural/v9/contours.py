"""Geometric contour, SDF and guidance construction for NSAMDR V9."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


CONTOUR_SCHEMA = "NSAMDR_GEOMETRIC_CONTOUR_V2"


def _sobel_numpy(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    gx = cv2.Sobel(signal.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(signal.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    return gx, gy


def _gaussian(signal: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 1.0e-6:
        return signal.astype(np.float32, copy=False)
    import cv2
    return cv2.GaussianBlur(signal.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)


def _multi_scale_structure_tensor(
    albedo: np.ndarray,
    normal_xy: np.ndarray,
    material: np.ndarray,
    material_valid: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smooth edge strength and a coherent local tangent field.

    A raw thresholded Sobel mask can inherit one-pixel staircase changes.  This
    structure tensor pools gradient evidence over several scales before an edge
    direction is chosen.  The resulting tangent therefore follows the common
    local geometry instead of fitting each damaged raster sample independently.
    """
    import cv2

    luma = (
        albedo[..., 0] * 0.2126
        + albedo[..., 1] * 0.7152
        + albedo[..., 2] * 0.0722
    ).astype(np.float32)
    signals: list[tuple[np.ndarray, float]] = [
        (luma, 1.00),
        (normal_xy[..., 0].astype(np.float32), 0.55),
        (normal_xy[..., 1].astype(np.float32), 0.55),
    ]
    if material_valid > 0.5:
        for channel in range(min(3, material.shape[-1])):
            signals.append((material[..., channel].astype(np.float32), 0.40))

    scales = ((0.60, 0.52), (1.35, 0.31), (2.70, 0.17))
    shape = luma.shape
    jxx = np.zeros(shape, dtype=np.float32)
    jxy = np.zeros(shape, dtype=np.float32)
    jyy = np.zeros(shape, dtype=np.float32)

    for sigma, scale_weight in scales:
        for signal, channel_weight in signals:
            smooth = _gaussian(signal, sigma)
            gx, gy = _sobel_numpy(smooth)
            weight = float(scale_weight * channel_weight)
            jxx += gx * gx * weight
            jxy += gx * gy * weight
            jyy += gy * gy * weight

    # Spatial pooling of the tensor is what rejects alternating one-pixel
    # orientations along diagonals and arcs.
    jxx = cv2.GaussianBlur(jxx, (0, 0), 1.10)
    jxy = cv2.GaussianBlur(jxy, (0, 0), 1.10)
    jyy = cv2.GaussianBlur(jyy, (0, 0), 1.10)

    trace = jxx + jyy
    discriminant = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    lambda_max = np.maximum(0.5 * (trace + discriminant), 0.0)
    strength = np.sqrt(lambda_max + 1.0e-12).astype(np.float32)

    # Principal tensor eigenvector is the local gradient normal.  Tangent is
    # perpendicular to it.  Tangent sign is intentionally axial: +t and -t are
    # the same contour direction and losses account for that symmetry.
    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tangent_x = -np.sin(normal_angle)
    tangent_y = np.cos(normal_angle)
    tangent = np.stack((tangent_x, tangent_y), axis=-1).astype(np.float32)
    return strength, tangent, luma




def _ridge_centreline(strength: np.ndarray) -> np.ndarray:
    """Return a narrow medial ridge from the multi-scale edge response."""
    import cv2

    peak = float(strength.max(initial=0.0))
    if peak <= 1.0e-8:
        return np.zeros(strength.shape, dtype=np.uint8)
    meaningful = strength[strength > peak * 0.003]
    adaptive = float(np.percentile(meaningful, 40.0)) if meaningful.size else peak * 0.05
    threshold = max(adaptive, peak * 0.015, 1.0e-4)
    band = (strength >= threshold).astype(np.uint8)

    # Distance inside the broad response band; local maxima form its medial
    # centreline.  This is fast in OpenCV and avoids explicitly dilating a raw
    # staircase-shaped Sobel mask.
    interior_distance = cv2.distanceTransform(band, cv2.DIST_L2, 5)
    local_max = cv2.dilate(interior_distance, np.ones((3, 3), np.uint8))
    ridge = (band > 0) & (interior_distance >= local_max - 1.0e-5)
    return ridge.astype(np.uint8)


def _signed_distance_from_edge(edge_seed: np.ndarray, luma: np.ndarray, max_distance: float) -> np.ndarray:
    import cv2

    binary = (edge_seed > 0).astype(np.uint8)
    if not binary.any():
        return np.ones((*binary.shape, 1), dtype=np.float32)

    distance_to_edge = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    # The sign is only used to make the SDF locally directional.  Geometry
    # supervision is sign-invariant through tangent/edge terms.
    local_mean = cv2.GaussianBlur(luma.astype(np.float32), (0, 0), 2.0)
    inside_signal = luma < local_mean
    signed = distance_to_edge.astype(np.float32)
    signed[inside_signal] *= -1.0
    signed = np.clip(signed / max(max_distance, 1.0), -1.0, 1.0)
    return signed[..., None].astype(np.float32)


def contour_targets(
    albedo: np.ndarray,
    normal_xy: np.ndarray,
    material: np.ndarray,
    material_valid: float,
    max_distance: float = 24.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return multi-scale SDF, coherent tangent and soft contour target.

    The target intentionally does not dilate a raw 3x3 Sobel mask.  It derives
    orientation from a multi-scale structure tensor and produces a narrow soft
    centreline.  This is substantially less likely to teach a stair-step or
    locally oscillating tangent than the original V9.1 contour target.
    """
    import cv2

    strength, tangent, luma = _multi_scale_structure_tensor(
        albedo, normal_xy, material, material_valid
    )
    seed = _ridge_centreline(strength)

    # Keep the centreline narrow.  A soft distance falloff is preferable to the
    # previous explicit dilation because it does not turn a one-pixel diagonal
    # into a thick staircase-shaped target.
    distance = cv2.distanceTransform(1 - seed, cv2.DIST_L2, 5)
    edge = np.exp(-0.5 * (distance / 0.85) ** 2).astype(np.float32)
    edge[seed > 0] = 1.0

    sdf = _signed_distance_from_edge(seed, luma, max_distance)
    orientation_decay = np.exp(-np.abs(sdf[..., 0]) * 4.0).astype(np.float32)
    orientation = tangent * orientation_decay[..., None]
    return sdf, np.ascontiguousarray(orientation), edge[..., None]


def analytic_contour_targets(
    signed_distance_pixels: np.ndarray,
    *,
    max_distance: float = 24.0,
    edge_sigma: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create exact geometric supervision from an analytic signed distance.

    Used by synthetic line/arc/corner examples.  Because the SDF is defined by
    mathematics rather than a thresholded texture edge, a straight line has a
    constant tangent and a circle has smoothly varying curvature by construction.
    """
    distance = np.asarray(signed_distance_pixels, dtype=np.float32)
    gy, gx = np.gradient(distance)
    length = np.sqrt(gx * gx + gy * gy + 1.0e-8)
    tangent = np.stack((-gy / length, gx / length), axis=-1).astype(np.float32)
    sdf = np.clip(distance / max(max_distance, 1.0), -1.0, 1.0)[..., None].astype(np.float32)
    edge = np.exp(-0.5 * (distance / max(edge_sigma, 0.1)) ** 2)[..., None].astype(np.float32)
    tangent *= np.exp(-np.abs(sdf[..., 0]) * 4.0)[..., None]
    return np.ascontiguousarray(sdf), np.ascontiguousarray(tangent), np.ascontiguousarray(edge)


def build_guidance_numpy(
    albedo: np.ndarray,
    normal_xy: np.ndarray,
    material: np.ndarray,
    degradation_level: float,
    uv_stretch: np.ndarray | None = None,
    chart_mask: np.ndarray | None = None,
) -> np.ndarray:
    luma = albedo[..., 0] * 0.2126 + albedo[..., 1] * 0.7152 + albedo[..., 2] * 0.0722
    lgx, lgy = _sobel_numpy(luma)
    ngx0, ngy0 = _sobel_numpy(normal_xy[..., 0])
    ngx1, ngy1 = _sobel_numpy(normal_xy[..., 1])
    normal_edge = np.sqrt(ngx0 * ngx0 + ngy0 * ngy0 + ngx1 * ngx1 + ngy1 * ngy1)
    material_edge = np.zeros_like(luma)
    for channel in range(min(3, material.shape[-1])):
        gx, gy = _sobel_numpy(material[..., channel])
        material_edge = np.maximum(material_edge, np.sqrt(gx * gx + gy * gy))
    import cv2
    curvature = cv2.Laplacian(luma.astype(np.float32), cv2.CV_32F, ksize=3) * 0.25
    if uv_stretch is None:
        uv_stretch = np.zeros_like(luma, dtype=np.float32)
    if chart_mask is None:
        chart_mask = np.ones_like(luma, dtype=np.float32)
    severity = np.full_like(luma, float(degradation_level), dtype=np.float32)
    guidance = np.stack((
        severity,
        np.clip(lgx, -1.0, 1.0),
        np.clip(lgy, -1.0, 1.0),
        np.clip(normal_edge, 0.0, 1.0),
        np.clip(material_edge, 0.0, 1.0),
        np.clip(curvature, -1.0, 1.0),
        np.clip(uv_stretch, 0.0, 1.0),
        np.clip(chart_mask, 0.0, 1.0),
    ), axis=-1)
    return np.ascontiguousarray(guidance.astype(np.float32))


def sobel_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = value.shape[1]
    kernel_x = value.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    kernel_y = kernel_x.t().contiguous()
    return (
        F.conv2d(value, kernel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3), padding=1, groups=channels),
        F.conv2d(value, kernel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3), padding=1, groups=channels),
    )
