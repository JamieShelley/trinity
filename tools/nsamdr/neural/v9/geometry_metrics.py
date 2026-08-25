"""Shared hard synthetic geometry metrics for NSAMDR V9.9.3.

Training checkpoint selection and the final G0-G5 Stage-B audit import these
same functions.  This prevents photometric proxy selection from disagreeing
with final material-boundary geometry acceptance.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _gray_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.float32)
    return (
        image[..., 0] * 0.2126
        + image[..., 1] * 0.7152
        + image[..., 2] * 0.0722
    ).astype(np.float32)


def synthetic_region_components(image_rgb: np.ndarray, target_rgb: np.ndarray) -> int:
    gray = _gray_rgb(image_rgb)
    target_gray = _gray_rgb(target_rgb)
    threshold = 0.5 * (float(np.min(target_gray)) + float(np.max(target_gray)))
    mask = (gray >= threshold).astype(np.uint8)
    count, _labels = cv2.connectedComponents(mask, connectivity=8)
    return max(0, int(count) - 1)


def topology_mismatch(image_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    return float(
        synthetic_region_components(image_rgb, target_rgb)
        != synthetic_region_components(target_rgb, target_rgb)
    )


def synthetic_region_boundary(image_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Material-midpoint region boundary used by both selection and Stage B."""
    gray = _gray_rgb(image_rgb)
    target_gray = _gray_rgb(target_rgb)
    threshold = 0.5 * (float(np.min(target_gray)) + float(np.max(target_gray)))
    region = (gray >= threshold).astype(np.uint8)
    if not np.any(region):
        return np.zeros_like(region)
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=1)
    return (region != eroded).astype(np.uint8)


def synthetic_region_chamfer(image_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    candidate = synthetic_region_boundary(image_rgb, target_rgb)
    target = synthetic_region_boundary(target_rgb, target_rgb)
    if not np.any(candidate) or not np.any(target):
        return float("inf")
    candidate_distance = cv2.distanceTransform(1 - candidate, cv2.DIST_L2, 3)
    target_distance = cv2.distanceTransform(1 - target, cv2.DIST_L2, 3)
    return 0.5 * (
        float(np.mean(target_distance[candidate.astype(bool)]))
        + float(np.mean(candidate_distance[target.astype(bool)]))
    )


def synthetic_region_chamfer_improvement(
    baseline_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> tuple[float, float, float]:
    before = synthetic_region_chamfer(baseline_rgb, target_rgb)
    after = synthetic_region_chamfer(candidate_rgb, target_rgb)
    if np.isfinite(after) and not np.isfinite(before):
        improvement = 1.0
    elif not np.isfinite(after):
        improvement = -1.0
    else:
        improvement = float((before - after) / max(abs(before), 1.0e-5))
    return float(before), float(after), improvement



def profile_width_rms_pixels(
    image_rgb: np.ndarray,
    target_sdf_pixels: np.ndarray,
    band_pixels: float = 6.0,
) -> float:
    """Gradient-weighted transition width around the analytic target contour."""
    gray = _gray_rgb(image_rgb)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy + 1.0e-10)
    distance = np.abs(np.asarray(target_sdf_pixels, dtype=np.float32))
    if distance.ndim == 3:
        distance = distance[..., 0]
    band = distance <= float(band_pixels)
    weight = grad * band.astype(np.float32)
    total = float(np.sum(weight))
    if total <= 1.0e-8:
        return float("inf")
    return float(np.sqrt(np.sum(weight * distance * distance) / total))


def profile_width_ratio(
    image_rgb: np.ndarray,
    target_rgb: np.ndarray,
    target_sdf_pixels: np.ndarray,
) -> float:
    """Candidate transition width divided by the exact target transition width."""
    candidate = profile_width_rms_pixels(image_rgb, target_sdf_pixels)
    target = profile_width_rms_pixels(target_rgb, target_sdf_pixels)
    if not np.isfinite(candidate) or not np.isfinite(target) or target <= 1.0e-8:
        return float("inf")
    return float(candidate / target)

def zero_crossing_points(field: np.ndarray) -> np.ndarray:
    """Extract sub-pixel zero crossings from a scalar grid by edge interpolation."""
    value = np.asarray(field, dtype=np.float32)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"field must be HxW or HxWx1, got {value.shape}")
    h, w = value.shape
    points: list[np.ndarray] = []

    zeros = np.argwhere(value == 0.0)
    if zeros.size:
        points.append(np.column_stack((zeros[:, 1], zeros[:, 0])).astype(np.float32))

    if w > 1:
        left, right = value[:, :-1], value[:, 1:]
        mask = ((left < 0.0) & (right > 0.0)) | ((left > 0.0) & (right < 0.0))
        ys, xs = np.nonzero(mask)
        if xs.size:
            a, b = np.abs(left[ys, xs]), np.abs(right[ys, xs])
            t = a / np.maximum(a + b, 1.0e-12)
            points.append(np.column_stack((xs.astype(np.float32) + t, ys.astype(np.float32))))

    if h > 1:
        top, bottom = value[:-1, :], value[1:, :]
        mask = ((top < 0.0) & (bottom > 0.0)) | ((top > 0.0) & (bottom < 0.0))
        ys, xs = np.nonzero(mask)
        if xs.size:
            a, b = np.abs(top[ys, xs]), np.abs(bottom[ys, xs])
            t = a / np.maximum(a + b, 1.0e-12)
            points.append(np.column_stack((xs.astype(np.float32), ys.astype(np.float32) + t)))

    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32, copy=False)


def _nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.full((len(query),), np.inf, dtype=np.float32)
    result = np.empty((len(query),), dtype=np.float32)
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        delta = q[:, None, :] - reference[None, :, :]
        distance2 = np.sum(delta * delta, axis=-1)
        result[start:start + len(q)] = np.sqrt(np.min(distance2, axis=1))
    return result



def _binary_topology_signature(mask: np.ndarray, min_component_pixels: int = 8) -> tuple[int, int]:
    """Return (foreground-components, enclosed-holes) after tiny-noise rejection.

    Stage-B is a geometry proof, so topology must be measured on the SDF sign
    regions themselves rather than on a thresholded rendered RGB image.  Very
    small components are ignored because they are below one 2x2 HR footprint
    and otherwise make the diagnostic sensitive to numerical zero-crossing
    speckles instead of material connectivity.
    """
    region = np.asarray(mask, dtype=np.uint8)
    if region.ndim != 2:
        raise ValueError(f"expected a 2-D topology mask, got {region.shape}")
    min_area = max(1, int(min_component_pixels))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )
    components = sum(
        int(stats[index, cv2.CC_STAT_AREA]) >= min_area
        for index in range(1, count)
    )

    inverse = (1 - region).astype(np.uint8)
    bg_count, _bg_labels, bg_stats, _bg_centroids = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    height, width = region.shape
    holes = 0
    for index in range(1, bg_count):
        if int(bg_stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        x = int(bg_stats[index, cv2.CC_STAT_LEFT])
        y = int(bg_stats[index, cv2.CC_STAT_TOP])
        w = int(bg_stats[index, cv2.CC_STAT_WIDTH])
        h = int(bg_stats[index, cv2.CC_STAT_HEIGHT])
        if x > 0 and y > 0 and x + w < width and y + h < height:
            holes += 1
    return int(components), int(holes)


def sdf_topology_signature(
    field: np.ndarray,
    target_field: np.ndarray,
    *,
    min_component_pixels: int = 8,
) -> tuple[int, int]:
    """Gauge-invariant material-region topology of an SDF/level-set field.

    V9 treats global SDF polarity as a gauge choice.  Align the predicted sign
    to the target by whichever global polarity gives greater pixel agreement,
    then compare connected material components and enclosed holes.
    """
    predicted = np.asarray(field, dtype=np.float32)
    target = np.asarray(target_field, dtype=np.float32)
    if predicted.ndim == 3:
        predicted = predicted[..., 0]
    if target.ndim == 3:
        target = target[..., 0]
    if predicted.shape != target.shape:
        raise ValueError(
            f"SDF topology shape mismatch: predicted={predicted.shape}, target={target.shape}"
        )
    predicted_negative = predicted < 0.0
    target_negative = target < 0.0
    direct_agreement = float(np.mean(predicted_negative == target_negative))
    flipped_agreement = float(np.mean((~predicted_negative) == target_negative))
    aligned = predicted_negative if direct_agreement >= flipped_agreement else ~predicted_negative
    return _binary_topology_signature(aligned, min_component_pixels)


def sdf_topology_mismatch(
    field: np.ndarray,
    target_field: np.ndarray,
    *,
    min_component_pixels: int = 8,
) -> float:
    """Return 1 when material-region component/hole topology differs."""
    predicted_signature = sdf_topology_signature(
        field, target_field, min_component_pixels=min_component_pixels
    )
    target = np.asarray(target_field, dtype=np.float32)
    if target.ndim == 3:
        target = target[..., 0]
    target_signature = _binary_topology_signature(
        target < 0.0, min_component_pixels
    )
    return float(predicted_signature != target_signature)

def zero_contour_distance(predicted_field: np.ndarray, target_field: np.ndarray) -> dict[str, float]:
    """Actual symmetric sub-pixel zero-contour Chamfer and RMS in pixels."""
    predicted = zero_crossing_points(predicted_field)
    target = zero_crossing_points(target_field)
    if len(predicted) == 0 or len(target) == 0:
        return {
            "chamferPixels": float("inf"),
            "rmsPixels": float("inf"),
            "predictedCrossings": float(len(predicted)),
            "targetCrossings": float(len(target)),
        }
    p_to_t = _nearest_distances(predicted, target)
    t_to_p = _nearest_distances(target, predicted)
    chamfer = 0.5 * (float(np.mean(p_to_t)) + float(np.mean(t_to_p)))
    rms = math.sqrt(0.5 * (float(np.mean(p_to_t * p_to_t)) + float(np.mean(t_to_p * t_to_p))))
    return {
        "chamferPixels": float(chamfer),
        "rmsPixels": float(rms),
        "predictedCrossings": float(len(predicted)),
        "targetCrossings": float(len(target)),
    }


def _branch_centres_1d(values: np.ndarray, *, minimum_separation: float = 0.75) -> np.ndarray:
    """Return one or two stable contour-branch centres for a 1-D coordinate.

    Thick synthetic lines/rings have two legitimate zero-set branches.  Their
    half-width must not be reported as geometric jitter.  A tiny deterministic
    1-D k-means separates those branches when they are genuinely distinct.
    """
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(data) < 8:
        return np.asarray([float(np.mean(data))], dtype=np.float32)
    c0, c1 = np.percentile(data, [25.0, 75.0]).astype(np.float32)
    for _ in range(8):
        d0 = np.abs(data - c0)
        d1 = np.abs(data - c1)
        left = data[d0 <= d1]
        right = data[d1 < d0]
        if len(left):
            c0 = np.float32(np.mean(left))
        if len(right):
            c1 = np.float32(np.mean(right))
    centres = np.sort(np.asarray([c0, c1], dtype=np.float32))
    if float(centres[1] - centres[0]) < float(minimum_separation):
        return np.asarray([float(np.mean(data))], dtype=np.float32)
    return centres


def _branch_detrended_rms(values: np.ndarray, centres: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    centres = np.asarray(centres, dtype=np.float32).reshape(-1)
    if len(data) == 0 or len(centres) == 0:
        return float("inf")
    labels = np.argmin(np.abs(data[:, None] - centres[None, :]), axis=1)
    residuals: list[np.ndarray] = []
    for index in range(len(centres)):
        branch = data[labels == index] - centres[index]
        if len(branch) < 4:
            continue
        # Translation and global width/radius bias belong to contour/profile
        # metrics.  Jitter is the high-frequency residual around each branch.
        branch = branch - float(np.mean(branch))
        residuals.append(branch.astype(np.float32, copy=False))
    if not residuals:
        return float("inf")
    residual = np.concatenate(residuals)
    return float(np.sqrt(np.mean(residual * residual)))


def line_perpendicular_jitter_pixels(
    predicted_field: np.ndarray,
    target_field: np.ndarray,
) -> float:
    """RMS high-frequency perpendicular residual of each target line branch.

    The target zero set for a rendered line is normally *two* parallel contour
    branches.  Earlier code fitted one centreline and therefore reported roughly
    half the authored line width (~2.5 px) as "jitter" even for the GT SDF
    itself.  Fit the target direction once, separate legitimate parallel
    branches in the normal coordinate, then remove each branch's mean offset.
    """
    predicted = zero_crossing_points(predicted_field)
    target = zero_crossing_points(target_field)
    if len(predicted) < 8 or len(target) < 8:
        return float("inf")
    centre = target.mean(axis=0)
    centered = target - centre[None, :]
    covariance = centered.T @ centered / max(len(target), 1)
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(_eigenvalues))]
    target_signed = centered @ normal
    predicted_signed = (predicted - centre[None, :]) @ normal
    centres = _branch_centres_1d(target_signed)
    return _branch_detrended_rms(predicted_signed, centres)

def circle_radial_roughness_pixels(
    predicted_field: np.ndarray,
    target_field: np.ndarray,
) -> float:
    """High-frequency radial roughness after separating legitimate ring sides.

    A finite-width circular line has inner and outer zero contours.  Measure
    roughness around each target radius independently so authored line width is
    not mistaken for polygonal/faceted geometry.
    """
    predicted = zero_crossing_points(predicted_field)
    target = zero_crossing_points(target_field)
    if len(predicted) < 16 or len(target) < 16:
        return float("inf")
    centre = target.mean(axis=0)
    target_radius = np.sqrt(np.sum((target - centre[None, :]) ** 2, axis=1))
    predicted_radius = np.sqrt(np.sum((predicted - centre[None, :]) ** 2, axis=1))
    centres = _branch_centres_1d(target_radius)
    return _branch_detrended_rms(predicted_radius, centres)


def line_staircase_energy_pixels(
    predicted_field: np.ndarray,
    target_field: np.ndarray,
    *,
    tangent_bin_pixels: float = 1.0,
) -> float:
    """High-frequency zero-contour staircase energy along a target line.

    The target establishes a continuous tangent/normal frame.  Predicted zero
    crossings are assigned to legitimate parallel line branches, binned along
    the tangent, and represented by their mean normal offset.  RMS second
    differences after removal of slow affine drift measure the discrete
    saw-tooth/step energy that makes a shallow upscaled line look pixelated.
    """
    predicted = zero_crossing_points(predicted_field)
    target = zero_crossing_points(target_field)
    if len(predicted) < 12 or len(target) < 12:
        return float("inf")

    centre = target.mean(axis=0)
    centered = target - centre[None, :]
    covariance = centered.T @ centered / max(len(target), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]

    target_n = centered @ normal
    branch_centres = _branch_centres_1d(target_n)
    pred_centered = predicted - centre[None, :]
    pred_t = pred_centered @ tangent
    pred_n = pred_centered @ normal
    labels = np.argmin(np.abs(pred_n[:, None] - branch_centres[None, :]), axis=1)

    energies: list[float] = []
    bin_width = max(float(tangent_bin_pixels), 0.25)
    for branch_index in range(len(branch_centres)):
        mask = labels == branch_index
        if int(np.count_nonzero(mask)) < 12:
            continue
        t = pred_t[mask]
        n = pred_n[mask]
        lo = float(np.min(t))
        hi = float(np.max(t))
        if hi - lo < 4.0 * bin_width:
            continue
        bins = np.floor((t - lo) / bin_width).astype(np.int32)
        unique = np.unique(bins)
        samples_t: list[float] = []
        samples_n: list[float] = []
        for b in unique:
            values = n[bins == b]
            if len(values):
                samples_t.append(lo + (float(b) + 0.5) * bin_width)
                samples_n.append(float(np.mean(values)))
        if len(samples_n) < 6:
            continue
        tt = np.asarray(samples_t, dtype=np.float32)
        nn = np.asarray(samples_n, dtype=np.float32)
        design = np.stack((tt, np.ones_like(tt)), axis=1)
        coeff, *_ = np.linalg.lstsq(design, nn, rcond=None)
        residual = nn - design @ coeff
        second = residual[2:] - 2.0 * residual[1:-1] + residual[:-2]
        if len(second):
            energies.append(float(np.sqrt(np.mean(second * second))))
    if not energies:
        return float("inf")
    return float(np.mean(energies))


def line_staircase_recovery(
    source_field: np.ndarray,
    predicted_field: np.ndarray,
    target_field: np.ndarray,
) -> float:
    """Fraction of LR staircase energy removed relative to the target geometry."""
    source_energy = line_staircase_energy_pixels(source_field, target_field)
    predicted_energy = line_staircase_energy_pixels(predicted_field, target_field)
    target_energy = line_staircase_energy_pixels(target_field, target_field)
    if not np.isfinite(source_energy) or source_energy <= target_energy + 1.0e-8:
        return 0.0
    if not np.isfinite(predicted_energy):
        return -1.0
    denominator = max(source_energy - target_energy, 1.0e-8)
    return float(np.clip((source_energy - predicted_energy) / denominator, -1.0, 1.0))
