"""Boundary-focused paired dataset for NSAMDR V9 geometric fidelity reconstruction."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import V9Config
from .contours import CONTOUR_SCHEMA, analytic_contour_targets, contour_targets
from .model import build_model_input

# Dataset indexing/extraction reuses the neutral authored crop-bundle extractor.
try:
    # Package import used by repository tests and tooling.
    from ..authored_texture_dataset import prepare_dataset as _prepare_crop_bundles
except ImportError:
    # Direct-script import used by the Windows training entry point.
    from authored_texture_dataset import prepare_dataset as _prepare_crop_bundles


SYNTHETIC_GEOMETRY_SCHEMA = "NSAMDR_EXACT_GEOMETRY_V1"


def prepare_dataset(
    repo_root: Path,
    config: V9Config,
    *,
    shared_cache: str | None = None,
    source_root: Path | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    return _prepare_crop_bundles(
        repo_root, config, shared_cache=shared_cache, source_root=source_root, rebuild=rebuild)


def load_dataset_manifest(repo_root: Path, config: V9Config) -> dict[str, Any]:
    path = repo_root / config.dataset_manifest
    if not path.is_file():
        raise RuntimeError(
            f"NSAMDR V9 dataset manifest is missing: {path}. "
            "Run scripts\\build\\nsamdr.bat index eve first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("crops"), list) or not payload["crops"]:
        raise RuntimeError(f"NSAMDR V9 dataset manifest has no crop bundles: {path}")
    return payload


def dataset_fingerprint(manifest: dict[str, Any], config: V9Config) -> str:
    import hashlib
    digest = hashlib.sha256()
    digest.update(b"nsamdr-v9-geometric-placement-v3")
    digest.update(CONTOUR_SCHEMA.encode("utf-8"))
    digest.update(SYNTHETIC_GEOMETRY_SCHEMA.encode("utf-8"))
    payload = {
        "tile": config.tile_size,
        "scale": config.target_scale,
        "families": manifest.get("families", []),
        "cropCount": len(manifest.get("crops", [])),
        "syntheticGeometryProbability": config.synthetic_geometry_probability,
    }
    if bool(manifest.get("fixedPreviewSet")):
        # Fixed tuning experiments must bind to the exact Raven crop selection.
        # Production manifests deliberately retain their historic fingerprint
        # semantics so existing production resume state remains compatible.
        payload["fixedPreviewSelectionFingerprint"] = manifest.get("fingerprint", "")
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _resize_float(array: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    import cv2
    return cv2.resize(array.astype(np.float32), (size, size), interpolation=interpolation)


def _renormalize_normal(normal_xy: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.maximum((normal_xy * normal_xy).sum(axis=-1, keepdims=True), 1.0e-8))
    return normal_xy / np.maximum(1.0, length / 0.999)


def _directional_blur(array: np.ndarray, horizontal: bool, strength: int) -> np.ndarray:
    import cv2
    kernel = (1 + strength * 4, 1) if horizontal else (1, 1 + strength * 4)
    return cv2.GaussianBlur(array.astype(np.float32), kernel, sigmaX=1.0 + strength, sigmaY=1.0 + strength)


def _bc_like(array: np.ndarray, blend: float, levels: float) -> np.ndarray:
    import cv2
    height, width = array.shape[:2]
    small = cv2.resize(array, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
    block = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    quantized = np.round(np.clip(array, 0.0, 1.0) * levels) / levels
    return np.clip(quantized * (1.0 - blend) + block * blend, 0.0, 1.0)



def _renderer_sampling_damage(array: np.ndarray, config: V9Config, rng: random.Random, *, interpolation: int) -> np.ndarray:
    """Approximate oblique UV/LOD sampling with rotated anisotropic minification and phase jitter."""
    import cv2
    h, w = array.shape[:2]
    angle = rng.uniform(0.0, 180.0)
    centre = (w * 0.5, h * 0.5)
    rotate = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(array, rotate, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT_101)
    anisotropy = rng.uniform(1.25, max(1.25, config.renderer_anisotropy_max))
    compressed_w = max(8, round(w / anisotropy))
    squeezed = cv2.resize(rotated, (compressed_w, h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(squeezed, (w, h), interpolation=interpolation)
    inverse = cv2.invertAffineTransform(rotate)
    restored = cv2.warpAffine(restored, inverse, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT_101)
    jitter = float(config.renderer_subpixel_jitter)
    if jitter > 0.0:
        shift = np.float32([[1.0, 0.0, rng.uniform(-jitter, jitter)], [0.0, 1.0, rng.uniform(-jitter, jitter)]])
        restored = cv2.warpAffine(restored, shift, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT_101)
    if array.ndim == 3 and restored.ndim == 2:
        restored = restored[..., None]
    return restored.astype(np.float32)

def degrade_physical_maps(
    albedo_hr: np.ndarray,
    normal_hr_xy: np.ndarray,
    material_hr: np.ndarray,
    lr_size: int,
    config: V9Config,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a hard 4x restoration input with EVE-like mip/compression damage."""
    import cv2

    lod_bias = rng.uniform(config.lod_bias_min, config.lod_bias_max)
    effective = max(16, round(lr_size / (1.0 + 0.70 * lod_bias)))
    albedo = _resize_float(albedo_hr, effective, cv2.INTER_AREA)
    albedo = _resize_float(albedo, lr_size, cv2.INTER_CUBIC)
    normal = _resize_float(normal_hr_xy, effective, cv2.INTER_AREA)
    normal = _resize_float(normal, lr_size, cv2.INTER_LINEAR)
    material = _resize_float(material_hr, effective, cv2.INTER_NEAREST)
    material = _resize_float(material, lr_size, cv2.INTER_LINEAR)

    if rng.random() < config.renderer_sampling_probability:
        albedo = _renderer_sampling_damage(albedo, config, rng, interpolation=cv2.INTER_CUBIC)
        normal = _renderer_sampling_damage(normal, config, rng, interpolation=cv2.INTER_LINEAR)
        material = _renderer_sampling_damage(material, config, rng, interpolation=cv2.INTER_LINEAR)

    if rng.random() < config.anisotropic_blur_probability:
        horizontal = rng.random() < 0.5
        strength = 1 if lod_bias < 1.15 else 2
        albedo = _directional_blur(albedo, horizontal, strength)
        normal = _directional_blur(normal, horizontal, strength)
    if rng.random() < config.bc_block_probability:
        albedo = _bc_like(albedo, rng.uniform(0.14, 0.38), rng.choice((31.0, 63.0)))
        normal_encoded = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
        normal = (_bc_like(normal_encoded, rng.uniform(0.08, 0.24), 63.0) - 0.5) * 2.0
        material = _bc_like(material, rng.uniform(0.08, 0.25), rng.choice((15.0, 31.0, 63.0)))
    if rng.random() < config.chroma_loss_probability:
        luma = albedo[..., 0:1] * 0.2126 + albedo[..., 1:2] * 0.7152 + albedo[..., 2:3] * 0.0722
        albedo = luma + (albedo - luma) * rng.uniform(0.42, 0.80)
    if rng.random() < config.ringing_probability:
        blurred = cv2.GaussianBlur(albedo, (0, 0), 0.8)
        albedo = np.clip(albedo + (albedo - blurred) * rng.uniform(0.25, 0.65), 0.0, 1.0)
    if rng.random() < config.halo_probability:
        luma = albedo[..., 0] * 0.2126 + albedo[..., 1] * 0.7152 + albedo[..., 2] * 0.0722
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip(np.sqrt(gx * gx + gy * gy), 0.0, 1.0)[..., None]
        halo = cv2.GaussianBlur(edge, (0, 0), 1.2)
        if halo.ndim == 2:
            halo = halo[..., None]
        albedo = np.clip(albedo + halo * rng.uniform(0.02, 0.08), 0.0, 1.0)

    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, 0.003 + 0.006 * lod_bias, normal.shape).astype(np.float32)
    normal = _renormalize_normal(normal + noise)
    severity = min(1.0, lod_bias / max(config.lod_bias_max, 1.0e-6))
    return (
        np.clip(albedo, 0.0, 1.0).astype(np.float32),
        normal.astype(np.float32),
        np.clip(material, 0.0, 1.0).astype(np.float32),
        float(severity),
    )


def _augment_maps(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turns = rng.randrange(4)
    if turns:
        albedo = np.rot90(albedo, turns).copy()
        normal = np.rot90(normal, turns).copy()
        material = np.rot90(material, turns).copy()
        for _ in range(turns):
            x, y = normal[..., 0].copy(), normal[..., 1].copy()
            normal[..., 0], normal[..., 1] = -y, x
    if rng.random() < 0.5:
        albedo, normal, material = albedo[:, ::-1].copy(), normal[:, ::-1].copy(), material[:, ::-1].copy()
        normal[..., 0] *= -1.0
    if rng.random() < 0.5:
        albedo, normal, material = albedo[::-1].copy(), normal[::-1].copy(), material[::-1].copy()
        normal[..., 1] *= -1.0
    return albedo, normal, material


def _edge_energy_map(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    valid: float,
) -> np.ndarray:
    """Build one full-crop boundary-energy map for O(1) candidate scoring."""
    import cv2

    luma = albedo[..., 0] * 0.2126 + albedo[..., 1] * 0.7152 + albedo[..., 2] * 0.0722
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    score = np.sqrt(gx * gx + gy * gy)
    for channel in range(2):
        gx = cv2.Sobel(normal[..., channel], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normal[..., channel], cv2.CV_32F, 0, 1, ksize=3)
        score += 0.55 * np.sqrt(gx * gx + gy * gy)
    if valid > 0.5:
        gx = cv2.Sobel(material[..., 0], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(material[..., 0], cv2.CV_32F, 0, 1, ksize=3)
        score += 0.35 * np.sqrt(gx * gx + gy * gy)
    return np.ascontiguousarray(score.astype(np.float32))


def _integral_image(value: np.ndarray) -> np.ndarray:
    integral = np.zeros((value.shape[0] + 1, value.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = value.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    return integral


def _integral_window_mean(integral: np.ndarray, x: int, y: int, size: int) -> float:
    x2, y2 = x + size, y + size
    total = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
    return float(total / max(1, size * size))


def _random_colour(rng: random.Random, low: float = 0.08, high: float = 0.92) -> np.ndarray:
    return np.asarray([rng.uniform(low, high) for _ in range(3)], dtype=np.float32)


def _separated_colour(reference: np.ndarray, rng: random.Random) -> np.ndarray:
    for _ in range(12):
        candidate = _random_colour(rng)
        if float(np.linalg.norm(candidate - reference)) >= 0.32:
            return candidate
    return np.clip(1.0 - reference * 0.65, 0.05, 0.95).astype(np.float32)


def _synthetic_region_colours(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Mix high-contrast analytic colours with dark/low-contrast hull-like pairs."""
    if rng.random() < 0.55:
        base = np.asarray(
            [rng.uniform(0.035, 0.28) for _ in range(3)], dtype=np.float32
        )
        direction = np.asarray(
            [rng.uniform(-1.0, 1.0) for _ in range(3)], dtype=np.float32
        )
        length = float(np.linalg.norm(direction))
        if length < 1.0e-5:
            direction = np.asarray([1.0, -0.4, 0.25], dtype=np.float32)
            length = float(np.linalg.norm(direction))
        direction /= length
        contrast = rng.uniform(0.07, 0.26)
        other = np.clip(base + direction * contrast, 0.015, 0.55).astype(np.float32)
        return base, other
    colour_a = _random_colour(rng)
    return colour_a, _separated_colour(colour_a, rng)


def _analytic_shape_distance(size: int, rng: random.Random) -> tuple[np.ndarray, str]:
    """Return exact/analytic line, arc, stripe and corner geometry in HR pixels."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    kind = rng.choices(
        (
            "line",
            "circle",
            "box",
            "stripe",
            "double_stripe",
            "ellipse",
            "rounded_box",
            "ring",
            "near_double",
        ),
        weights=(0.15, 0.12, 0.11, 0.20, 0.07, 0.09, 0.10, 0.09, 0.07),
        k=1,
    )[0]

    if kind in {"line", "stripe", "double_stripe", "near_double"}:
        # Avoid almost-axis-aligned-only training; arbitrary angles are the main
        # source of visible 4x staircases in the Raven evaluation.
        angle = rng.uniform(0.035, np.pi - 0.035)
        nx, ny = np.cos(angle), np.sin(angle)
        cx = size * rng.uniform(0.28, 0.72)
        cy = size * rng.uniform(0.28, 0.72)
        signed_axis = (xx - cx) * nx + (yy - cy) * ny
        if kind == "line":
            return signed_axis.astype(np.float32), kind
        half_width = rng.uniform(0.55, 3.75)
        if kind == "stripe":
            signed = np.abs(signed_axis) - half_width
        else:
            spacing = rng.uniform(3.5, 8.0) if kind == "near_double" else rng.uniform(8.0, 28.0)
            stripe_a = np.abs(signed_axis - spacing * 0.5) - half_width
            stripe_b = np.abs(signed_axis + spacing * 0.5) - half_width
            signed = np.minimum(stripe_a, stripe_b)
        return signed.astype(np.float32), kind

    if kind == "ring":
        radius = size * rng.uniform(0.12, 0.36)
        width = rng.uniform(1.0, 4.0)
        margin = radius + width + 8.0
        cx = rng.uniform(margin, size - margin)
        cy = rng.uniform(margin, size - margin)
        radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius
        return (np.abs(radial) - width).astype(np.float32), kind

    if kind == "circle":
        radius = size * rng.uniform(0.10, 0.42)
        margin = radius + 8.0
        cx = rng.uniform(margin, size - margin)
        cy = rng.uniform(margin, size - margin)
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius
        return distance.astype(np.float32), kind

    angle = rng.uniform(0.035, np.pi - 0.035)
    ca, sa = np.cos(angle), np.sin(angle)
    cx = size * rng.uniform(0.38, 0.62)
    cy = size * rng.uniform(0.38, 0.62)
    px, py = xx - cx, yy - cy
    lx = ca * px + sa * py
    ly = -sa * px + ca * py

    if kind == "ellipse":
        axis_x = size * rng.uniform(0.16, 0.38)
        axis_y = size * rng.uniform(0.08, 0.30)
        # Smooth analytic implicit ellipse converted to approximately pixel
        # distance. Exact zero-set/orientation is what matters for supervision.
        radial = np.sqrt((lx / axis_x) ** 2 + (ly / axis_y) ** 2)
        return ((radial - 1.0) * min(axis_x, axis_y)).astype(np.float32), kind

    half_w = size * rng.uniform(0.18, 0.38)
    half_h = size * rng.uniform(0.09, 0.30)
    if kind == "rounded_box":
        radius = min(half_w, half_h) * rng.uniform(0.12, 0.48)
        qx = np.abs(lx) - (half_w - radius)
        qy = np.abs(ly) - (half_h - radius)
        outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return (outside + inside - radius).astype(np.float32), kind

    # Standard signed distance to a rotated rectangle: exact straight runs and
    # hard corners stop the network from learning to round authored corners.
    qx = np.abs(lx) - half_w
    qy = np.abs(ly) - half_h
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return (outside + inside).astype(np.float32), kind


def _synthetic_geometry_sample(
    target_size: int,
    config: V9Config,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create exact anti-aliased line/arc/stripe/corner PBR supervision.

    The target geometry exists analytically before rasterisation.  This is the
    key supervision that teaches the network that a staircase can represent a
    single straight line, and that a smooth arc should not become a wavy spline.
    """
    signed_distance, _kind = _analytic_shape_distance(target_size, rng)
    # One-pixel analytic coverage at HR pixel centres. Negative SDF is inside.
    coverage = np.clip(0.5 - signed_distance, 0.0, 1.0)[..., None].astype(np.float32)

    colour_a, colour_b = _synthetic_region_colours(rng)
    albedo = colour_a.reshape(1, 1, 3) * (1.0 - coverage) + colour_b.reshape(1, 1, 3) * coverage

    normal_a = np.asarray([rng.uniform(-0.28, 0.28), rng.uniform(-0.28, 0.28)], dtype=np.float32)
    normal_b = np.asarray([rng.uniform(-0.38, 0.38), rng.uniform(-0.38, 0.38)], dtype=np.float32)
    if float(np.linalg.norm(normal_b - normal_a)) < 0.12:
        normal_b = np.clip(normal_a + np.asarray([0.18, -0.14], dtype=np.float32), -0.55, 0.55)
    normal = normal_a.reshape(1, 1, 2) * (1.0 - coverage) + normal_b.reshape(1, 1, 2) * coverage
    normal = _renormalize_normal(normal.astype(np.float32))

    material_a = np.asarray([
        rng.uniform(0.08, 0.34), rng.uniform(0.0, 0.18), rng.uniform(0.42, 0.88)
    ], dtype=np.float32)
    material_b = np.asarray([
        rng.uniform(0.62, 0.94), rng.uniform(0.0, 0.24), rng.uniform(0.12, 0.58)
    ], dtype=np.float32)
    # Do not let the synthetic branch become dependent on perfect companion
    # maps. Some authored Raven regions have weak/ambiguous normal/material edges.
    if rng.random() < 0.25:
        normal_b = normal_a.copy()
    normal = normal_a.reshape(1, 1, 2) * (1.0 - coverage) + normal_b.reshape(1, 1, 2) * coverage
    normal = _renormalize_normal(normal.astype(np.float32))

    if rng.random() < 0.25:
        material_b = material_a.copy()
    material = material_a.reshape(1, 1, 3) * (1.0 - coverage) + material_b.reshape(1, 1, 3) * coverage

    sdf, orientation, edge = analytic_contour_targets(signed_distance)
    return (
        np.ascontiguousarray(albedo.astype(np.float32)),
        np.ascontiguousarray(normal.astype(np.float32)),
        np.ascontiguousarray(material.astype(np.float32)),
        sdf,
        orientation,
        edge,
    )


def _pack_sample(
    albedo_hr: np.ndarray,
    normal_hr: np.ndarray,
    material_hr: np.ndarray,
    material_valid: float,
    sdf: np.ndarray,
    orientation: np.ndarray,
    edge: np.ndarray,
    config: V9Config,
    rng: random.Random,
    *,
    geometry_exact: float,
) -> dict[str, torch.Tensor]:
    import cv2

    target_size = config.tile_size * config.target_scale
    albedo_lr, normal_lr, material_lr, severity = degrade_physical_maps(
        albedo_hr, normal_hr, material_hr, config.tile_size, config, rng)
    model_input = build_model_input(
        albedo_lr, normal_lr, material_lr, degradation_level=severity)

    roughness = material_hr[..., 2:3]
    emissive = material_hr[..., 1:2]
    material_class = np.minimum(
        config.material_classes - 1,
        np.floor(material_hr[..., 0] * config.material_classes).astype(np.int64),
    )

    baseline = cv2.resize(albedo_lr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    difference = np.mean(np.abs(albedo_hr - baseline), axis=-1, keepdims=True)
    confidence_target = np.clip(difference / 0.16 + edge * 0.35, 0.0, 1.0).astype(np.float32)

    return {
        "input": torch.from_numpy(model_input),
        "target_albedo": torch.from_numpy(albedo_hr.transpose(2, 0, 1).copy()),
        "target_normal": torch.from_numpy(normal_hr.transpose(2, 0, 1).copy()),
        "target_roughness": torch.from_numpy(roughness.transpose(2, 0, 1).copy()),
        "target_emissive": torch.from_numpy(emissive.transpose(2, 0, 1).copy()),
        "target_material_class": torch.from_numpy(material_class.copy()),
        "target_sdf": torch.from_numpy(sdf.transpose(2, 0, 1).copy()),
        "target_orientation": torch.from_numpy(orientation.transpose(2, 0, 1).copy()),
        "target_edge": torch.from_numpy(edge.transpose(2, 0, 1).copy()),
        "target_confidence": torch.from_numpy(confidence_target.transpose(2, 0, 1).copy()),
        "auxiliary_valid": torch.tensor([material_valid], dtype=torch.float32),
        "geometry_exact": torch.tensor([geometry_exact], dtype=torch.float32),
        "severity": torch.tensor([severity], dtype=torch.float32),
    }


class PhysicalTileDatasetV9(Dataset[dict[str, torch.Tensor]]):
    """Read authored crop bundles and mix exact geometric restoration examples."""

    def __init__(
        self,
        manifest: dict[str, Any],
        config: V9Config,
        split: str,
        length: int,
        *,
        seed: int,
    ) -> None:
        self.config = config
        self.split = split
        self.length = max(1, int(length))
        self.seed = int(seed)
        self.records = [record for record in manifest["crops"] if record["split"] == split]
        if not self.records:
            raise RuntimeError(f"NSAMDR V9 dataset contains no {split} crops")

    def __len__(self) -> int:
        return self.length

    def _choose_crop(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        material: np.ndarray,
        valid: float,
        target_size: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        source_h, source_w = albedo.shape[:2]
        max_x = max(0, source_w - target_size)
        max_y = max(0, source_h - target_size)
        if self.split != "train":
            return max_x // 2, max_y // 2
        energy_integral = _integral_image(_edge_energy_map(albedo, normal, material, valid))
        candidates: list[tuple[float, int, int]] = []
        for _ in range(self.config.boundary_candidate_count):
            x = rng.randrange(max_x + 1) if max_x else 0
            y = rng.randrange(max_y + 1) if max_y else 0
            candidates.append((_integral_window_mean(energy_integral, x, y, target_size), x, y))
        if rng.random() < self.config.boundary_sampling_probability:
            _, x, y = max(candidates)
        else:
            _, x, y = rng.choice(candidates)
        return x, y

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index * 1_000_003)
        target_size = self.config.tile_size * self.config.target_scale

        # Only the training split mixes exact analytic geometry. Validation stays
        # on real authored EVE crops so quality/regret metrics remain representative
        # of the actual deployment domain.
        if self.split == "train" and rng.random() < self.config.synthetic_geometry_probability:
            albedo_hr, normal_hr, material_hr, sdf, orientation, edge = _synthetic_geometry_sample(
                target_size, self.config, rng)
            return _pack_sample(
                albedo_hr, normal_hr, material_hr, 1.0,
                sdf, orientation, edge, self.config, rng, geometry_exact=1.0)

        record = self.records[rng.randrange(len(self.records))]
        with np.load(record["path"], allow_pickle=False) as bundle:
            albedo_u8 = np.asarray(bundle["albedo"], dtype=np.uint8)
            normal_u8 = np.asarray(bundle["normal"], dtype=np.uint8)
            material_u8 = np.asarray(bundle["material"], dtype=np.uint8)
            material_valid = float(np.asarray(
                bundle.get("material_valid", np.asarray([0.0], dtype=np.float32))
            ).reshape(-1)[0])

        if min(albedo_u8.shape[:2]) < target_size:
            import cv2
            albedo_u8 = cv2.resize(albedo_u8, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            normal_u8 = cv2.resize(normal_u8, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            material_u8 = cv2.resize(material_u8, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        albedo = albedo_u8.astype(np.float32) / 255.0
        normal = normal_u8[..., :2].astype(np.float32) / 127.5 - 1.0
        normal = _renormalize_normal(normal)
        material = material_u8[..., :3].astype(np.float32) / 255.0
        if self.split == "train":
            albedo, normal, material = _augment_maps(albedo, normal, material, rng)
        x, y = self._choose_crop(albedo, normal, material, material_valid, target_size, rng)
        albedo_hr = np.ascontiguousarray(albedo[y:y + target_size, x:x + target_size])
        normal_hr = np.ascontiguousarray(normal[y:y + target_size, x:x + target_size])
        material_hr = np.ascontiguousarray(material[y:y + target_size, x:x + target_size])

        sdf, orientation, edge = contour_targets(
            albedo_hr, normal_hr, material_hr, material_valid)
        return _pack_sample(
            albedo_hr, normal_hr, material_hr, material_valid,
            sdf, orientation, edge, self.config, rng, geometry_exact=0.0)


class SyntheticGeometryValidationDataset(Dataset):
    """Deterministic exact-geometry validation used only during SDF phases.

    Real Raven validation remains separate. This dataset answers the narrower
    question needed for Stage B checkpoint selection: did GeometryNet actually
    learn a metric contour field before any gate or real-asset tuning is allowed?
    """

    def __init__(self, config: V9Config, length: int, *, seed: int) -> None:
        self.config = config
        self.length = max(1, int(length))
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(index) * 1_000_003)
        target_size = self.config.tile_size * self.config.target_scale
        albedo_hr, normal_hr, material_hr, sdf, orientation, edge = _synthetic_geometry_sample(
            target_size, self.config, rng
        )
        return _pack_sample(
            albedo_hr, normal_hr, material_hr, 1.0,
            sdf, orientation, edge, self.config, rng, geometry_exact=1.0
        )
