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
from .contours import CONTOUR_SCHEMA, analytic_contour_targets, contour_targets, lr_contour_prior
from .model import build_model_input
from .geometry_proof_ladder import PROOF_CASE_COUNT, build_proof_case
from .parametric_primitives import (
    PRIMITIVE_COUNT, PrimitiveTarget, proof_case_primitive_target, random_primitive_target,
    render_parametric_sdf_numpy,
)

# Dataset indexing/extraction reuses the neutral authored crop-bundle extractor.
try:
    # Package import used by repository tests and tooling.
    from ..authored_texture_dataset import prepare_dataset as _prepare_crop_bundles
except ImportError:
    # Direct-script import used by the Windows training entry point.
    from authored_texture_dataset import prepare_dataset as _prepare_crop_bundles


SYNTHETIC_GEOMETRY_SCHEMA = "NSAMDR_LR_HR_OBSERVABLE_DELTA_GEOMETRY_V3"


class NSAMDRDatasetService:
    # Purpose: Implement prepare dataset for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def prepare_dataset(
        self,
        repo_root: Path,
        config: V9Config,
        *,
        shared_cache: str | None = None,
        source_root: Path | None = None,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        return _prepare_crop_bundles(
            repo_root, config, shared_cache=shared_cache, source_root=source_root, rebuild=rebuild)

    # Purpose: Implement load dataset manifest for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def load_dataset_manifest(self, repo_root: Path, config: V9Config) -> dict[str, Any]:
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

    # Purpose: Implement dataset fingerprint for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def dataset_fingerprint(self, manifest: dict[str, Any], config: V9Config) -> str:
        import hashlib
        digest = hashlib.sha256()
        digest.update(b"nsamdr-v9.9-continuous-implicit-boundary-specialists-v1")
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

    # Purpose: Implement resize float for NSAMDRDatasetService.
    # Called by: degrade_physical_maps
    # Calls: No same-class helper methods.
    def _resize_float(self, array: np.ndarray, size: int, interpolation: int) -> np.ndarray:
        import cv2
        return cv2.resize(array.astype(np.float32), (size, size), interpolation=interpolation)

    # Purpose: Implement renormalize normal for NSAMDRDatasetService.
    # Called by: _dds_codec_like, _synthetic_geometry_sample, _synthetic_parametric_geometry_sample, degrade_physical_maps
    # Calls: No same-class helper methods.
    def _renormalize_normal(self, normal_xy: np.ndarray) -> np.ndarray:
        length = np.sqrt(np.maximum((normal_xy * normal_xy).sum(axis=-1, keepdims=True), 1.0e-8))
        return normal_xy / np.maximum(1.0, length / 0.999)

    # Purpose: Implement directional blur for NSAMDRDatasetService.
    # Called by: degrade_physical_maps
    # Calls: No same-class helper methods.
    def _directional_blur(self, array: np.ndarray, horizontal: bool, strength: int) -> np.ndarray:
        import cv2
        kernel = (1 + strength * 4, 1) if horizontal else (1, 1 + strength * 4)
        return cv2.GaussianBlur(array.astype(np.float32), kernel, sigmaX=1.0 + strength, sigmaY=1.0 + strength)

    # Purpose: Implement bc like for NSAMDRDatasetService.
    # Called by: degrade_physical_maps
    # Calls: No same-class helper methods.
    def _bc_like(self, array: np.ndarray, blend: float, levels: float) -> np.ndarray:
        """Legacy generic block proxy retained for old configs/tests."""
        import cv2
        height, width = array.shape[:2]
        small = cv2.resize(array, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
        block = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        quantized = np.round(np.clip(array, 0.0, 1.0) * levels) / levels
        return np.clip(quantized * (1.0 - blend) + block * blend, 0.0, 1.0)

    # Purpose: Implement block view for NSAMDRDatasetService.
    # Called by: _bc1_palette_like, _bc4_palette_like
    # Calls: No same-class helper methods.
    def _block_view(self, array: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Return [BH,BW,16,C] 4x4 blocks and original shape."""
        a = np.asarray(array, dtype=np.float32)
        if a.ndim == 2:
            a = a[..., None]
        h, w, c = a.shape
        ph = (-h) % 4
        pw = (-w) % 4
        if ph or pw:
            a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")
        hp, wp = a.shape[:2]
        blocks = a.reshape(hp // 4, 4, wp // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(hp // 4, wp // 4, 16, c)
        return blocks, (h, w)

    # Purpose: Implement unblock view for NSAMDRDatasetService.
    # Called by: _bc1_palette_like, _bc4_palette_like
    # Calls: No same-class helper methods.
    def _unblock_view(self, blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        bh, bw, _n, c = blocks.shape
        a = blocks.reshape(bh, bw, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, c)
        h, w = shape
        return a[:h, :w]

    # Purpose: Implement bc1 palette like for NSAMDRDatasetService.
    # Called by: _dds_codec_like
    # Calls: _block_view, _unblock_view
    def _bc1_palette_like(self, array: np.ndarray) -> np.ndarray:
        """Vectorised BC1 RGB endpoint/palette approximation on real 4x4 blocks."""
        blocks, shape = self._block_view(np.clip(array, 0.0, 1.0))
        lo = blocks.min(axis=2)
        hi = blocks.max(axis=2)
        levels = np.asarray([31.0, 63.0, 31.0], dtype=np.float32)[:blocks.shape[-1]]
        lo = np.round(lo * levels) / levels
        hi = np.round(hi * levels) / levels
        palette = np.stack((hi, lo, (2.0 * hi + lo) / 3.0, (hi + 2.0 * lo) / 3.0), axis=2)
        dist = ((blocks[:, :, :, None, :] - palette[:, :, None, :, :]) ** 2).sum(axis=-1)
        choice = dist.argmin(axis=3)
        decoded = np.take_along_axis(palette[:, :, None, :, :], choice[:, :, :, None, None], axis=3)[:, :, :, 0, :]
        return np.clip(self._unblock_view(decoded, shape), 0.0, 1.0).astype(np.float32)

    # Purpose: Implement bc4 palette like for NSAMDRDatasetService.
    # Called by: _dds_codec_like
    # Calls: _block_view, _unblock_view
    def _bc4_palette_like(self, channel: np.ndarray) -> np.ndarray:
        """BC4-like scalar endpoint/interpolant approximation."""
        blocks, shape = self._block_view(np.clip(channel, 0.0, 1.0))
        lo = np.round(blocks.min(axis=2) * 255.0) / 255.0
        hi = np.round(blocks.max(axis=2) * 255.0) / 255.0
        alpha = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(1, 1, 8, 1)
        palette = lo[:, :, None, :] * (1.0 - alpha) + hi[:, :, None, :] * alpha
        dist = ((blocks[:, :, :, None, :] - palette[:, :, None, :, :]) ** 2).sum(axis=-1)
        choice = dist.argmin(axis=3)
        decoded = np.take_along_axis(palette[:, :, None, :, :], choice[:, :, :, None, None], axis=3)[:, :, :, 0, :]
        return np.clip(self._unblock_view(decoded, shape), 0.0, 1.0).astype(np.float32)

    # Purpose: Implement codec kind for NSAMDRDatasetService.
    # Called by: _dds_codec_like
    # Calls: No same-class helper methods.
    def _codec_kind(self, format_name: str | None, role: str) -> str:
        text = (format_name or "").upper().replace("-", "_")
        if any(token in text for token in ("BC5", "ATI2", "3DC", "DXGI_83", "DXGI_84")):
            return "bc5"
        if any(token in text for token in ("BC3", "DXT5", "DXGI_77", "DXGI_78")):
            return "bc3"
        if any(token in text for token in ("BC1", "DXT1", "DXGI_71", "DXGI_72")):
            return "bc1"
        return {"albedo": "bc1", "normal": "bc5", "material": "bc3"}.get(role, "bc1")

    # Purpose: Implement dds codec like for NSAMDRDatasetService.
    # Called by: degrade_physical_maps
    # Calls: _bc1_palette_like, _bc4_palette_like, _codec_kind, _renormalize_normal
    def _dds_codec_like(
        self,
        array: np.ndarray, format_name: str | None, role: str, blend: float, rng: random.Random
    ) -> np.ndarray:
        """Crop-phase-aware BC1/BC3/BC5-like degradation using 4x4 palettes.

        A crop can start at any phase inside the source DDS block grid, so pad by a
        deterministic random 0..3 phase before block encoding and crop it back.
        """
        a = np.asarray(array, dtype=np.float32)
        phase_x, phase_y = rng.randrange(4), rng.randrange(4)
        padded = np.pad(a, ((phase_y, 0), (phase_x, 0), (0, 0)), mode="edge")
        kind = self._codec_kind(format_name, role)
        if kind == "bc5":
            encoded = np.clip(padded * 0.5 + 0.5, 0.0, 1.0) if role == "normal" else np.clip(padded, 0.0, 1.0)
            channels = [self._bc4_palette_like(encoded[..., i:i + 1]) for i in range(min(2, encoded.shape[-1]))]
            decoded = np.concatenate(channels + ([encoded[..., 2:]] if encoded.shape[-1] > 2 else []), axis=-1)
            if role == "normal":
                decoded = (decoded - 0.5) * 2.0
                decoded = self._renormalize_normal(decoded[..., :2])
        elif kind == "bc3":
            decoded = self._bc1_palette_like(np.clip(padded, 0.0, 1.0))
        else:
            decoded = self._bc1_palette_like(np.clip(padded, 0.0, 1.0))
        decoded = decoded[phase_y:phase_y + a.shape[0], phase_x:phase_x + a.shape[1]]
        return (a * (1.0 - float(blend)) + decoded * float(blend)).astype(np.float32)

    # Purpose: Implement renderer sampling damage for NSAMDRDatasetService.
    # Called by: degrade_physical_maps
    # Calls: No same-class helper methods.
    def _renderer_sampling_damage(self, array: np.ndarray, config: V9Config, rng: random.Random, *, interpolation: int) -> np.ndarray:
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

    # Purpose: Implement degrade physical maps for NSAMDRDatasetService.
    # Called by: _pack_sample
    # Calls: _bc_like, _dds_codec_like, _directional_blur, _renderer_sampling_damage, _renormalize_normal, _resize_float
    def degrade_physical_maps(
        self,
        albedo_hr: np.ndarray,
        normal_hr_xy: np.ndarray,
        material_hr: np.ndarray,
        lr_size: int,
        config: V9Config,
        rng: random.Random,
        codec_formats: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Generate a hard 4x restoration input with EVE-like mip/compression damage."""
        import cv2

        lod_bias = rng.uniform(config.lod_bias_min, config.lod_bias_max)
        effective = max(16, round(lr_size / (1.0 + 0.70 * lod_bias)))
        albedo = self._resize_float(albedo_hr, effective, cv2.INTER_AREA)
        albedo = self._resize_float(albedo, lr_size, cv2.INTER_CUBIC)
        normal = self._resize_float(normal_hr_xy, effective, cv2.INTER_AREA)
        normal = self._resize_float(normal, lr_size, cv2.INTER_LINEAR)
        material = self._resize_float(material_hr, effective, cv2.INTER_NEAREST)
        material = self._resize_float(material, lr_size, cv2.INTER_LINEAR)

        if rng.random() < config.renderer_sampling_probability:
            albedo = self._renderer_sampling_damage(albedo, config, rng, interpolation=cv2.INTER_CUBIC)
            normal = self._renderer_sampling_damage(normal, config, rng, interpolation=cv2.INTER_LINEAR)
            material = self._renderer_sampling_damage(material, config, rng, interpolation=cv2.INTER_LINEAR)

        if rng.random() < config.anisotropic_blur_probability:
            horizontal = rng.random() < 0.5
            strength = 1 if lod_bias < 1.15 else 2
            albedo = self._directional_blur(albedo, horizontal, strength)
            normal = self._directional_blur(normal, horizontal, strength)
        if bool(getattr(config, "dds_codec_degradation_enabled", True)) and rng.random() < float(getattr(config, "dds_codec_probability", 0.90)):
            formats = codec_formats or {}
            blend = rng.uniform(float(getattr(config, "dds_codec_blend_min", 0.45)), float(getattr(config, "dds_codec_blend_max", 1.0)))
            albedo = self._dds_codec_like(albedo, formats.get("albedo"), "albedo", blend, rng)
            normal = self._dds_codec_like(normal, formats.get("normal"), "normal", min(1.0, blend * 0.85), rng)
            material = self._dds_codec_like(material, formats.get("material"), "material", min(1.0, blend * 0.75), rng)
        elif rng.random() < config.bc_block_probability:
            albedo = self._bc_like(albedo, rng.uniform(0.14, 0.38), rng.choice((31.0, 63.0)))
            normal_encoded = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
            normal = (self._bc_like(normal_encoded, rng.uniform(0.08, 0.24), 63.0) - 0.5) * 2.0
            material = self._bc_like(material, rng.uniform(0.08, 0.25), rng.choice((15.0, 31.0, 63.0)))
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
        normal = self._renormalize_normal(normal + noise)
        severity = min(1.0, lod_bias / max(config.lod_bias_max, 1.0e-6))
        return (
            np.clip(albedo, 0.0, 1.0).astype(np.float32),
            normal.astype(np.float32),
            np.clip(material, 0.0, 1.0).astype(np.float32),
            float(severity),
        )

    # Purpose: Implement augment maps for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _augment_maps(
        self,
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

    # Purpose: Implement edge energy map for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _edge_energy_map(
        self,
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

    # Purpose: Implement integral image for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _integral_image(self, value: np.ndarray) -> np.ndarray:
        integral = np.zeros((value.shape[0] + 1, value.shape[1] + 1), dtype=np.float64)
        integral[1:, 1:] = value.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
        return integral

    # Purpose: Implement integral window mean for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _integral_window_mean(self, integral: np.ndarray, x: int, y: int, size: int) -> float:
        x2, y2 = x + size, y + size
        total = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
        return float(total / max(1, size * size))

    # Purpose: Implement random colour for NSAMDRDatasetService.
    # Called by: _separated_colour, _synthetic_region_colours
    # Calls: No same-class helper methods.
    def _random_colour(self, rng: random.Random, low: float = 0.08, high: float = 0.92) -> np.ndarray:
        return np.asarray([rng.uniform(low, high) for _ in range(3)], dtype=np.float32)

    # Purpose: Implement separated colour for NSAMDRDatasetService.
    # Called by: _synthetic_region_colours
    # Calls: _random_colour
    def _separated_colour(self, reference: np.ndarray, rng: random.Random) -> np.ndarray:
        for _ in range(12):
            candidate = self._random_colour(rng)
            if float(np.linalg.norm(candidate - reference)) >= 0.32:
                return candidate
        return np.clip(1.0 - reference * 0.65, 0.05, 0.95).astype(np.float32)

    # Purpose: Implement synthetic region colours for NSAMDRDatasetService.
    # Called by: _synthetic_geometry_sample, _synthetic_parametric_geometry_sample
    # Calls: _random_colour, _separated_colour
    def _synthetic_region_colours(self, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
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
        colour_a = self._random_colour(rng)
        return colour_a, self._separated_colour(colour_a, rng)

    # Purpose: Implement analytic shape distance for NSAMDRDatasetService.
    # Called by: _synthetic_geometry_sample
    # Calls: No same-class helper methods.
    def _analytic_shape_distance(
        self,
        size: int,
        rng: random.Random,
        *,
        forced_kind: str | None = None,
        forced_angle_deg: float | None = None,
    ) -> tuple[np.ndarray, str]:
        """Return exact analytic contour geometry in HR pixels.

        V9.9.3 includes explicit shallow-angle and branching topology stressors so
        checkpoint selection exercises the same failure modes as the permanent G0-G5
        audit instead of hoping a 12-sample random draw happens to contain them.
        """
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        kind = forced_kind or rng.choices(
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
                "junction",
                "corner",
            ),
            weights=(0.12, 0.10, 0.08, 0.18, 0.06, 0.08, 0.08, 0.08, 0.06, 0.09, 0.07),
            k=1,
        )[0]

        if kind in {"line", "stripe", "double_stripe", "near_double"}:
            # Deliberately oversample shallow/near-axis angles. EXP_0002 showed that
            # uniform-angle training could pass random validation while a 3-degree
            # stripe tore into periodic segments in the permanent audit.
            if forced_angle_deg is not None:
                angle = np.deg2rad(float(forced_angle_deg))
            elif rng.random() < 0.38:
                base = rng.choice((1.0, 3.0, 7.0, 83.0, 87.0, 89.0))
                angle = np.deg2rad(base + rng.uniform(-0.75, 0.75))
            else:
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

        if kind in {"junction", "corner"}:
            cx = size * rng.uniform(0.42, 0.58)
            cy = size * rng.uniform(0.42, 0.58)
            half_width = rng.uniform(1.0, 3.0)
            length = size * rng.uniform(0.24, 0.38)
            rotation = rng.uniform(0.0, 2.0 * np.pi)
            arm_angles = (rotation - 0.72, rotation + 0.72) if kind == "corner" else (
                rotation, rotation + 2.0 * np.pi / 3.0, rotation + 4.0 * np.pi / 3.0
            )
            distances = []
            for arm_angle in arm_angles:
                x1 = cx + length * np.cos(arm_angle)
                y1 = cy + length * np.sin(arm_angle)
                vx = x1 - cx
                vy = y1 - cy
                denom = max(vx * vx + vy * vy, 1.0e-6)
                t = np.clip(((xx - cx) * vx + (yy - cy) * vy) / denom, 0.0, 1.0)
                qx = xx - (cx + t * vx)
                qy = yy - (cy + t * vy)
                distances.append(np.sqrt(qx * qx + qy * qy) - half_width)
            return np.minimum.reduce(distances).astype(np.float32), kind

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

    # Purpose: Implement synthetic parametric geometry sample for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: _renormalize_normal, _synthetic_region_colours
    def _synthetic_parametric_geometry_sample(
        self,
        target_size: int,
        config: V9Config,
        rng: random.Random,
        *,
        forced_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PrimitiveTarget]:
        """V10.7.9 complete-teacher primitive sample.

        Unlike the retired sparse medial teacher, every tile owns one compact exact
        class/parameter target. The same parameters analytically generate the HR SDF.
        """
        target = random_primitive_target(target_size, rng, forced_class=forced_class)
        signed_distance = render_parametric_sdf_numpy(target, target_size)
        coverage = np.clip(0.5 - signed_distance, 0.0, 1.0)[..., None].astype(np.float32)

        colour_a, colour_b = self._synthetic_region_colours(rng)
        albedo = colour_a.reshape(1, 1, 3) * (1.0 - coverage) + colour_b.reshape(1, 1, 3) * coverage
        normal_a = np.asarray([rng.uniform(-0.28, 0.28), rng.uniform(-0.28, 0.28)], dtype=np.float32)
        normal_b = np.asarray([rng.uniform(-0.38, 0.38), rng.uniform(-0.38, 0.38)], dtype=np.float32)
        if float(np.linalg.norm(normal_b - normal_a)) < 0.12:
            normal_b = np.clip(normal_a + np.asarray([0.18, -0.14], dtype=np.float32), -0.55, 0.55)
        if rng.random() < 0.25:
            normal_b = normal_a.copy()
        normal = normal_a.reshape(1, 1, 2) * (1.0 - coverage) + normal_b.reshape(1, 1, 2) * coverage
        normal = self._renormalize_normal(normal.astype(np.float32))

        material_a = np.asarray([rng.uniform(0.08, 0.34), rng.uniform(0.0, 0.18), rng.uniform(0.42, 0.88)], dtype=np.float32)
        material_b = np.asarray([rng.uniform(0.62, 0.94), rng.uniform(0.0, 0.24), rng.uniform(0.12, 0.58)], dtype=np.float32)
        if rng.random() < 0.25:
            material_b = material_a.copy()
        material = material_a.reshape(1, 1, 3) * (1.0 - coverage) + material_b.reshape(1, 1, 3) * coverage
        sdf, orientation, edge = analytic_contour_targets(signed_distance)
        return (
            np.ascontiguousarray(albedo.astype(np.float32)),
            np.ascontiguousarray(normal.astype(np.float32)),
            np.ascontiguousarray(material.astype(np.float32)),
            sdf, orientation, edge, target,
        )

    # Purpose: Implement synthetic geometry sample for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: _analytic_shape_distance, _renormalize_normal, _synthetic_region_colours
    def _synthetic_geometry_sample(
        self,
        target_size: int,
        config: V9Config,
        rng: random.Random,
        *,
        forced_kind: str | None = None,
        forced_angle_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create exact anti-aliased line/arc/stripe/corner PBR supervision.

        The target geometry exists analytically before rasterisation.  This is the
        key supervision that teaches the network that a staircase can represent a
        single straight line, and that a smooth arc should not become a wavy spline.
        """
        signed_distance, _kind = self._analytic_shape_distance(
            target_size, rng, forced_kind=forced_kind, forced_angle_deg=forced_angle_deg
        )
        # One-pixel analytic coverage at HR pixel centres. Negative SDF is inside.
        coverage = np.clip(0.5 - signed_distance, 0.0, 1.0)[..., None].astype(np.float32)

        colour_a, colour_b = self._synthetic_region_colours(rng)
        albedo = colour_a.reshape(1, 1, 3) * (1.0 - coverage) + colour_b.reshape(1, 1, 3) * coverage

        normal_a = np.asarray([rng.uniform(-0.28, 0.28), rng.uniform(-0.28, 0.28)], dtype=np.float32)
        normal_b = np.asarray([rng.uniform(-0.38, 0.38), rng.uniform(-0.38, 0.38)], dtype=np.float32)
        if float(np.linalg.norm(normal_b - normal_a)) < 0.12:
            normal_b = np.clip(normal_a + np.asarray([0.18, -0.14], dtype=np.float32), -0.55, 0.55)
        normal = normal_a.reshape(1, 1, 2) * (1.0 - coverage) + normal_b.reshape(1, 1, 2) * coverage
        normal = self._renormalize_normal(normal.astype(np.float32))

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
        normal = self._renormalize_normal(normal.astype(np.float32))

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

    # Purpose: Implement pack sample for NSAMDRDatasetService.
    # Called by: External callers and the owning workflow.
    # Calls: degrade_physical_maps
    def _pack_sample(
        self,
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
        codec_formats: dict[str, str] | None = None,
        primitive_target: PrimitiveTarget | None = None,
    ) -> dict[str, torch.Tensor]:
        import cv2

        target_size = config.tile_size * config.target_scale
        albedo_lr, normal_lr, material_lr, severity = self.degrade_physical_maps(
            albedo_hr, normal_hr, material_hr, config.tile_size, config, rng, codec_formats=codec_formats)

        # V9.9.3 derives an explicit LR geometric prior from the same degraded
        # physical maps the model receives. This is the source side of the
        # supervised LR->HR geometry correction.
        source_sdf_lr, _source_orientation_lr, source_edge_lr, source_observability = lr_contour_prior(
            albedo_lr, normal_lr, material_lr, material_valid,
            max_distance=float(config.contour_sdf_max_distance_pixels) / float(config.target_scale),
        )
        model_input = build_model_input(
            albedo_lr, normal_lr, material_lr, degradation_level=severity,
            source_sdf_prior=source_sdf_lr,
            contour_sdf_max_distance_pixels=float(config.contour_sdf_max_distance_pixels),
            target_scale=int(config.target_scale),
        )

        roughness = material_hr[..., 2:3]
        emissive = material_hr[..., 1:2]
        material_class = np.minimum(
            config.material_classes - 1,
            np.floor(material_hr[..., 0] * config.material_classes).astype(np.int64),
        )

        baseline = cv2.resize(albedo_lr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        difference = np.mean(np.abs(albedo_hr - baseline), axis=-1, keepdims=True)
        confidence_target = np.clip(difference / 0.16 + edge * 0.35, 0.0, 1.0).astype(np.float32)

        source_sdf_hr = cv2.resize(
            source_sdf_lr[..., 0], (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )[..., None].astype(np.float32)
        source_edge_hr = cv2.resize(
            source_edge_lr[..., 0], (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )[..., None].astype(np.float32)

        # Geometry need is measured from this exact LR/HR pair, not assigned by a
        # hand-authored case class. Sign is a gauge, so SDF need compares unsigned
        # distance-to-contour while image/edge deficits capture width, blur and
        # material-boundary evidence.
        max_distance = float(config.contour_sdf_max_distance_pixels)
        sdf_need = np.clip(
            np.abs(np.abs(sdf * max_distance) - np.abs(source_sdf_hr * max_distance))
            / max(float(config.geometry_need_sdf_scale_pixels), 1.0e-6),
            0.0, 1.0,
        )
        edge_need = np.clip(np.abs(edge - source_edge_hr), 0.0, 1.0)
        image_need = np.clip(difference / 0.12, 0.0, 1.0)
        geometry_need = np.clip(
            0.50 * sdf_need + 0.30 * edge_need + 0.20 * image_need, 0.0, 1.0
        )
        # Synthetic geometry is useful only to the degree that its degraded LR maps
        # still contain observable evidence of the HR boundary.  This is measured
        # from the LR maps themselves; no fixed case-class threshold is used.
        geometry_need *= np.float32(0.25 + 0.75 * float(source_observability))
        geometry_need = geometry_need.astype(np.float32)

        return {
            "input": torch.from_numpy(model_input),
            "target_albedo": torch.from_numpy(albedo_hr.transpose(2, 0, 1).copy()),
            "target_normal": torch.from_numpy(normal_hr.transpose(2, 0, 1).copy()),
            "target_roughness": torch.from_numpy(roughness.transpose(2, 0, 1).copy()),
            "target_emissive": torch.from_numpy(emissive.transpose(2, 0, 1).copy()),
            "target_material_class": torch.from_numpy(material_class.copy()),
            "target_sdf": torch.from_numpy(sdf.transpose(2, 0, 1).copy()),
            "source_sdf": torch.from_numpy(source_sdf_hr.transpose(2, 0, 1).copy()),
            "source_edge": torch.from_numpy(source_edge_hr.transpose(2, 0, 1).copy()),
            "geometry_need": torch.from_numpy(geometry_need.transpose(2, 0, 1).copy()),
            "source_observability": torch.tensor([source_observability], dtype=torch.float32),
            "target_orientation": torch.from_numpy(orientation.transpose(2, 0, 1).copy()),
            "target_edge": torch.from_numpy(edge.transpose(2, 0, 1).copy()),
            "target_confidence": torch.from_numpy(confidence_target.transpose(2, 0, 1).copy()),
            "auxiliary_valid": torch.tensor([material_valid], dtype=torch.float32),
            "geometry_exact": torch.tensor([geometry_exact], dtype=torch.float32),
            "severity": torch.tensor([severity], dtype=torch.float32),
            "primitive_valid": torch.tensor([1.0 if primitive_target is not None else 0.0], dtype=torch.float32),
            "primitive_class": torch.tensor(primitive_target.class_index if primitive_target is not None else -1, dtype=torch.int64),
            "primitive_params": torch.from_numpy((primitive_target.params if primitive_target is not None else np.zeros((12,), dtype=np.float32)).copy()),
            "primitive_param_mask": torch.from_numpy((primitive_target.mask if primitive_target is not None else np.zeros((12,), dtype=np.float32)).copy()),
        }

_n_s_a_m_d_r_dataset_service = NSAMDRDatasetService()
prepare_dataset = _n_s_a_m_d_r_dataset_service.prepare_dataset
load_dataset_manifest = _n_s_a_m_d_r_dataset_service.load_dataset_manifest
dataset_fingerprint = _n_s_a_m_d_r_dataset_service.dataset_fingerprint
_resize_float = _n_s_a_m_d_r_dataset_service._resize_float
_renormalize_normal = _n_s_a_m_d_r_dataset_service._renormalize_normal
_directional_blur = _n_s_a_m_d_r_dataset_service._directional_blur
_bc_like = _n_s_a_m_d_r_dataset_service._bc_like
_block_view = _n_s_a_m_d_r_dataset_service._block_view
_unblock_view = _n_s_a_m_d_r_dataset_service._unblock_view
_bc1_palette_like = _n_s_a_m_d_r_dataset_service._bc1_palette_like
_bc4_palette_like = _n_s_a_m_d_r_dataset_service._bc4_palette_like
_codec_kind = _n_s_a_m_d_r_dataset_service._codec_kind
_dds_codec_like = _n_s_a_m_d_r_dataset_service._dds_codec_like
_renderer_sampling_damage = _n_s_a_m_d_r_dataset_service._renderer_sampling_damage
degrade_physical_maps = _n_s_a_m_d_r_dataset_service.degrade_physical_maps
_augment_maps = _n_s_a_m_d_r_dataset_service._augment_maps
_edge_energy_map = _n_s_a_m_d_r_dataset_service._edge_energy_map
_integral_image = _n_s_a_m_d_r_dataset_service._integral_image
_integral_window_mean = _n_s_a_m_d_r_dataset_service._integral_window_mean
_random_colour = _n_s_a_m_d_r_dataset_service._random_colour
_separated_colour = _n_s_a_m_d_r_dataset_service._separated_colour
_synthetic_region_colours = _n_s_a_m_d_r_dataset_service._synthetic_region_colours
_analytic_shape_distance = _n_s_a_m_d_r_dataset_service._analytic_shape_distance
_synthetic_parametric_geometry_sample = _n_s_a_m_d_r_dataset_service._synthetic_parametric_geometry_sample
_synthetic_geometry_sample = _n_s_a_m_d_r_dataset_service._synthetic_geometry_sample
_pack_sample = _n_s_a_m_d_r_dataset_service._pack_sample


class PhysicalTileDatasetV9(Dataset[dict[str, torch.Tensor]]):
    """Read authored crop bundles and mix exact geometric restoration examples."""

    # Purpose: Implement init for PhysicalTileDatasetV9.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
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

    # Purpose: Implement len for PhysicalTileDatasetV9.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __len__(self) -> int:
        return self.length

    # Purpose: Implement choose crop for PhysicalTileDatasetV9.
    # Called by: __getitem__
    # Calls: No same-class helper methods.
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

    # Purpose: Implement getitem for PhysicalTileDatasetV9.
    # Called by: External callers and the owning workflow.
    # Calls: _choose_crop
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index * 1_000_003)
        target_size = self.config.tile_size * self.config.target_scale

        # Only the training split mixes exact analytic geometry. Validation stays
        # on real authored EVE crops so quality/regret metrics remain representative
        # of the actual deployment domain.
        if self.split == "train" and rng.random() < self.config.synthetic_geometry_probability:
            albedo_hr, normal_hr, material_hr, sdf, orientation, edge, primitive_target = _synthetic_parametric_geometry_sample(
                target_size, self.config, rng)
            return _pack_sample(
                albedo_hr, normal_hr, material_hr, 1.0,
                sdf, orientation, edge, self.config, rng, geometry_exact=1.0,
                primitive_target=primitive_target)

        record = self.records[rng.randrange(len(self.records))]
        with np.load(record["path"], allow_pickle=False) as bundle:
            albedo_u8 = np.asarray(bundle["albedo"], dtype=np.uint8)
            normal_u8 = np.asarray(bundle["normal"], dtype=np.uint8)
            material_u8 = np.asarray(bundle["material"], dtype=np.uint8)
            material_valid = float(np.asarray(
                bundle.get("material_valid", np.asarray([0.0], dtype=np.float32))
            ).reshape(-1)[0])
            codec_formats: dict[str, str] = {}
            if "metadata" in bundle:
                try:
                    metadata = json.loads(str(np.asarray(bundle["metadata"]).reshape(-1)[0]))
                    sources = metadata.get("sources", {}) if isinstance(metadata, dict) else {}
                    for role in ("albedo", "normal", "material"):
                        source = sources.get(role) if isinstance(sources, dict) else None
                        if isinstance(source, dict) and source.get("format"):
                            codec_formats[role] = str(source["format"])
                except (ValueError, TypeError, json.JSONDecodeError):
                    codec_formats = {}

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
            sdf, orientation, edge, self.config, rng, geometry_exact=0.0, codec_formats=codec_formats)


class ParametricPrimitiveTrainingDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed V10.7.9 class-balanced bank with complete primitive supervision.

    B1b deliberately revisits the same analytic cases for several passes. This
    turns the compact classifier/regressor into an ordinary supervised problem
    instead of showing it a one-off geometry once and immediately discarding it.
    The permanent 29-case proof ladder remains a separate held-out dataset.
    """

    # Purpose: Implement init for ParametricPrimitiveTrainingDataset.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, config: V9Config, length: int, *, seed: int) -> None:
        self.config = config
        self.length = max(1, int(length))
        self.seed = int(seed)

    # Purpose: Implement len for ParametricPrimitiveTrainingDataset.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __len__(self) -> int:
        return self.length

    # Purpose: Implement getitem for ParametricPrimitiveTrainingDataset.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(index) * 104729)
        target_size = int(self.config.tile_size) * int(self.config.target_scale)
        albedo, normal, material, sdf, orientation, edge, primitive_target = (
            _synthetic_parametric_geometry_sample(
                target_size, self.config, rng, forced_class=int(index) % PRIMITIVE_COUNT
            )
        )
        return _pack_sample(
            albedo, normal, material, 1.0, sdf, orientation, edge,
            self.config, rng, geometry_exact=1.0, primitive_target=primitive_target,
        )


class SyntheticGeometryValidationDataset(Dataset):
    """Deterministic exact-geometry validation used only during SDF phases.

    Real Raven validation remains separate. This dataset answers the narrower
    question needed for Stage B checkpoint selection: did GeometryNet actually
    learn a metric contour field before any gate or real-asset tuning is allowed?
    """

    # Purpose: Implement init for SyntheticGeometryValidationDataset.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, config: V9Config, length: int, *, seed: int) -> None:
        self.config = config
        self.length = max(1, int(length))
        self.seed = int(seed)

    # Purpose: Implement len for SyntheticGeometryValidationDataset.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __len__(self) -> int:
        return self.length

    # Purpose: Implement getitem for SyntheticGeometryValidationDataset.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the exact same case/degradation consumed by the final audit."""
        import cv2

        target_size = self.config.tile_size * self.config.target_scale
        if target_size != 512:
            raise RuntimeError(
                f"V10.6 canonical geometry proof requires 512 HR pixels, got {target_size}"
            )
        case = build_proof_case(
            int(index) % PROOF_CASE_COUNT,
            size=target_size,
            max_distance=float(self.config.contour_sdf_max_distance_pixels),
        )
        model_input = build_model_input(
            case.low_rgb,
            case.low_normal,
            case.low_material,
            degradation_level=1.0,
            contour_sdf_max_distance_pixels=float(self.config.contour_sdf_max_distance_pixels),
            target_scale=int(self.config.target_scale),
        )
        source_sdf_lr = model_input[-1].astype(np.float32)
        source_sdf_hr = cv2.resize(
            source_sdf_lr,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )[..., None].astype(np.float32)
        # Use the same LR prior extractor for edge telemetry only. Input itself
        # already came from build_model_input above, exactly as in the final audit.
        _sdf_lr, _ori_lr, source_edge_lr, source_observability = lr_contour_prior(
            case.low_rgb,
            case.low_normal,
            case.low_material,
            1.0,
            max_distance=float(self.config.contour_sdf_max_distance_pixels) / float(self.config.target_scale),
        )
        source_edge_hr = cv2.resize(
            source_edge_lr[..., 0],
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )[..., None].astype(np.float32)
        baseline = cv2.resize(case.low_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        difference = np.mean(np.abs(case.target_rgb - baseline), axis=-1, keepdims=True)
        confidence_target = np.clip(difference / 0.16 + case.target_edge * 0.35, 0.0, 1.0).astype(np.float32)
        max_distance = float(self.config.contour_sdf_max_distance_pixels)
        sdf_need = np.clip(
            np.abs(np.abs(case.target_sdf * max_distance) - np.abs(source_sdf_hr * max_distance))
            / max(float(self.config.geometry_need_sdf_scale_pixels), 1.0e-6),
            0.0, 1.0,
        )
        geometry_need = np.clip(
            0.60 * sdf_need + 0.25 * np.abs(case.target_edge - source_edge_hr) + 0.15 * np.clip(difference / 0.12, 0.0, 1.0),
            0.0, 1.0,
        ).astype(np.float32)
        roughness = case.target_material[..., 2:3]
        emissive = case.target_material[..., 1:2]
        material_class = np.minimum(
            self.config.material_classes - 1,
            np.floor(case.target_material[..., 0] * self.config.material_classes).astype(np.int64),
        )
        primitive_target = proof_case_primitive_target(case.name, target_size)
        return {
            "input": torch.from_numpy(model_input),
            "target_albedo": torch.from_numpy(case.target_rgb.transpose(2, 0, 1).copy()),
            "target_normal": torch.from_numpy(case.target_normal.transpose(2, 0, 1).copy()),
            "target_roughness": torch.from_numpy(roughness.transpose(2, 0, 1).copy()),
            "target_emissive": torch.from_numpy(emissive.transpose(2, 0, 1).copy()),
            "target_material_class": torch.from_numpy(material_class.copy()),
            "target_sdf": torch.from_numpy(case.target_sdf.transpose(2, 0, 1).copy()),
            "source_sdf": torch.from_numpy(source_sdf_hr.transpose(2, 0, 1).copy()),
            "source_edge": torch.from_numpy(source_edge_hr.transpose(2, 0, 1).copy()),
            "geometry_need": torch.from_numpy(geometry_need.transpose(2, 0, 1).copy()),
            "source_observability": torch.tensor([source_observability], dtype=torch.float32),
            "target_orientation": torch.from_numpy(case.target_orientation.transpose(2, 0, 1).copy()),
            "target_edge": torch.from_numpy(case.target_edge.transpose(2, 0, 1).copy()),
            "target_confidence": torch.from_numpy(confidence_target.transpose(2, 0, 1).copy()),
            "auxiliary_valid": torch.tensor([1.0], dtype=torch.float32),
            "geometry_exact": torch.tensor([1.0], dtype=torch.float32),
            "severity": torch.tensor([1.0], dtype=torch.float32),
            "synthetic_case_kind": torch.tensor(case.kind, dtype=torch.int64),
            "synthetic_case_stress": torch.tensor(case.stress, dtype=torch.bool),
            "synthetic_case_index": torch.tensor(int(index) % PROOF_CASE_COUNT, dtype=torch.int64),
            "primitive_valid": torch.tensor([1.0], dtype=torch.float32),
            "primitive_class": torch.tensor(primitive_target.class_index, dtype=torch.int64),
            "primitive_params": torch.from_numpy(primitive_target.params.copy()),
            "primitive_param_mask": torch.from_numpy(primitive_target.mask.copy()),
        }
