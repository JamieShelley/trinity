#!/usr/bin/env python3
"""Train the NSAMDR V4 tile-context material reconstruction network.

V4 replaces the per-pixel contour-transport MLP with a fully convolutional
network that sees a large overlapping material neighbourhood. The model is
trained offline and applied to Mode 3 candidate textures before the preview is
launched. The preview therefore compares the untouched source material against
an already reconstructed material using the same renderer, camera and lighting.

The network predicts continuous source transport, bounded RGB residual and a
confidence gate. Dilated residual blocks provide a 125-pixel receptive field at
full resolution, allowing long panel boundaries and repeated staircase patterns
to be treated coherently instead of as unrelated pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit("Missing numpy/Pillow. Install tools/nsamdr/neural/requirements.txt") from exc

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit("Missing PyTorch. Run scripts\\build\\setup_nsamdr_cuda.bat or setup_nsamdr_cpu.bat") from exc


MODEL_SCHEMA = "NSAMDR_TILE_CONTEXT_MATERIAL_V4"
INPUT_CHANNELS = 8  # albedo RGB, normal XY, material selector, paint support, roughness
OUTPUT_CHANNELS = 6  # continuous offset XY, RGB residual, confidence
DILATION_PATTERN = (1, 2, 4, 8, 8, 4, 2, 1)


def receptive_field_pixels(residual_blocks: int) -> int:
    """Return the exact full-resolution receptive field for the configured model."""
    block_count = max(0, min(int(residual_blocks), len(DILATION_PATTERN)))
    # Two 3x3 stem convolutions add four pixels. Each residual block has two
    # 3x3 convolutions at its dilation, adding 4*dilation pixels.
    return 1 + 4 + 4 * sum(DILATION_PATTERN[:block_count])


@dataclass
class TrainingConfig:
    epochs: int = 24
    tiles_per_epoch: int = 2048
    batch_size: int = 8
    learning_rate: float = 0.0002
    base_channels: int = 32
    residual_blocks: int = 8
    tile_size: int = 96
    reconstruction_weight: float = 1.0
    edge_weight: float = 1.5
    identity_weight: float = 0.65
    confidence_weight: float = 0.25
    flow_smoothness_weight: float = 0.05
    max_residual: float = 0.25
    max_offset_pixels: int = 8
    max_source_files: int = 0
    real_source_fraction: float = 0.15
    artifact_fraction: float = 0.80
    seed: int = 1337
    source_root: str = ""
    source_globs: tuple[str, ...] = ("**/*_d.png", "**/*_ar.png", "**/*.dds")
    output_dir: str = "artifacts/nsamdr/neural"
    checkpoint_name: str = "nsamdr_tile_context.pt"
    metadata_name: str = "nsamdr_tile_context.json"
    device: str = "cuda"
    cuda_device_index: int = 0
    matmul_precision: str = "high"
    inference_tile_size: int = 512
    inference_overlap: int = 64

    @classmethod
    def load(cls, path: Path | None) -> "TrainingConfig":
        config = cls()
        if path is None:
            return config
        payload = json.loads(path.read_text(encoding="utf-8"))
        aliases = {
            "samples": "tiles_per_epoch",
            "samplesPerEpoch": "tiles_per_epoch",
            "tilesPerEpoch": "tiles_per_epoch",
            "learningRate": "learning_rate",
            "batchSize": "batch_size",
            "hiddenChannels": "base_channels",
            "baseChannels": "base_channels",
            "residualBlocks": "residual_blocks",
            "tileSize": "tile_size",
            "transportWeight": "reconstruction_weight",
            "reconstructionWeight": "reconstruction_weight",
            "residualWeight": "edge_weight",
            "edgeWeight": "edge_weight",
            "identityWeight": "identity_weight",
            "gateWeight": "confidence_weight",
            "confidenceWeight": "confidence_weight",
            "flowSmoothnessWeight": "flow_smoothness_weight",
            "maxResidual": "max_residual",
            "maxOffsetPixels": "max_offset_pixels",
            "maxSourceFiles": "max_source_files",
            "realSourceFraction": "real_source_fraction",
            "artifactFraction": "artifact_fraction",
            "sourceRoot": "source_root",
            "sourceGlobs": "source_globs",
            "outputDir": "output_dir",
            "checkpointName": "checkpoint_name",
            "metadataName": "metadata_name",
            "cudaDeviceIndex": "cuda_device_index",
            "matmulPrecision": "matmul_precision",
            "inferenceTileSize": "inference_tile_size",
            "inferenceOverlap": "inference_overlap",
        }
        for key, value in payload.items():
            target = aliases.get(key, key)
            if hasattr(config, target):
                if target == "source_globs":
                    value = tuple(str(item) for item in value)
                setattr(config, target, value)
        return config

    def validate(self) -> None:
        self.epochs = max(1, min(int(self.epochs), 300))
        self.tiles_per_epoch = max(128, min(int(self.tiles_per_epoch), 1_000_000))
        self.batch_size = max(1, min(int(self.batch_size), 128))
        self.learning_rate = float(min(max(self.learning_rate, 1.0e-6), 0.01))
        self.base_channels = max(16, min(int(self.base_channels), 96))
        self.residual_blocks = max(4, min(int(self.residual_blocks), len(DILATION_PATTERN)))
        self.tile_size = max(48, min(int(self.tile_size), 256))
        self.tile_size -= self.tile_size % 8
        self.reconstruction_weight = float(min(max(self.reconstruction_weight, 0.0), 8.0))
        self.edge_weight = float(min(max(self.edge_weight, 0.0), 8.0))
        self.identity_weight = float(min(max(self.identity_weight, 0.0), 8.0))
        self.confidence_weight = float(min(max(self.confidence_weight, 0.0), 8.0))
        self.flow_smoothness_weight = float(min(max(self.flow_smoothness_weight, 0.0), 2.0))
        self.max_residual = float(min(max(self.max_residual, 0.01), 0.50))
        self.max_offset_pixels = max(1, min(int(self.max_offset_pixels), 16))
        self.max_source_files = max(0, min(int(self.max_source_files), 20_000))
        self.real_source_fraction = float(min(max(self.real_source_fraction, 0.0), 0.80))
        self.artifact_fraction = float(min(max(self.artifact_fraction, 0.0), 1.0))
        self.device = str(self.device).strip().lower()
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError("device must be one of: cuda, cpu, auto")
        self.cuda_device_index = max(0, int(self.cuda_device_index))
        self.matmul_precision = str(self.matmul_precision).strip().lower()
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmulPrecision must be one of: highest, high, medium")
        self.inference_tile_size = max(128, min(int(self.inference_tile_size), 1024))
        self.inference_tile_size -= self.inference_tile_size % 16
        self.inference_overlap = max(16, min(int(self.inference_overlap), self.inference_tile_size // 3))


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.activation(self.conv1(value)))
        return value + residual * torch.clamp(self.scale, 0.0, 1.0)


class MaterialTileContextNet(nn.Module):
    """Fully convolutional material reconstruction with a broad tile receptive field."""

    def __init__(
        self,
        base_channels: int = 32,
        residual_blocks: int = 8,
        max_offset_pixels: float = 8.0,
        max_residual: float = 0.25,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.residual_blocks = int(residual_blocks)
        self.max_offset_pixels = float(max_offset_pixels)
        self.max_residual = float(max_residual)
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, self.base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.GELU(),
        )
        self.body = nn.ModuleList([
            DilatedResidualBlock(self.base_channels, DILATION_PATTERN[index])
            for index in range(self.residual_blocks)
        ])
        self.head = nn.Conv2d(self.base_channels, OUTPUT_CHANNELS, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[5] = -3.0  # start close to identity

    @staticmethod
    def _sampling_grid(flow_pixels: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = flow_pixels.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=flow_pixels.device, dtype=flow_pixels.dtype),
            torch.arange(width, device=flow_pixels.device, dtype=flow_pixels.dtype),
            indexing="ij",
        )
        base_x = (x + 0.5) * (2.0 / max(width, 1)) - 1.0
        base_y = (y + 0.5) * (2.0 / max(height, 1)) - 1.0
        base = torch.stack((base_x, base_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        normalized = torch.empty_like(flow_pixels)
        normalized[:, 0] = flow_pixels[:, 0] * (2.0 / max(width, 1))
        normalized[:, 1] = flow_pixels[:, 1] * (2.0 / max(height, 1))
        return base + normalized.permute(0, 2, 3, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.stem(inputs)
        for block in self.body:
            features = block(features)
        raw = self.head(features)
        flow = torch.tanh(raw[:, 0:2]) * self.max_offset_pixels
        residual = torch.tanh(raw[:, 2:5]) * self.max_residual
        confidence = torch.sigmoid(raw[:, 5:6])
        grid = self._sampling_grid(flow)
        transported = F.grid_sample(
            inputs[:, 0:3], grid, mode="bilinear", padding_mode="border", align_corners=False)
        proposal = torch.clamp(transported + residual, 0.0, 1.0)
        corrected = inputs[:, 0:3] + (proposal - inputs[:, 0:3]) * confidence
        return {
            "corrected": torch.clamp(corrected, 0.0, 1.0),
            "proposal": proposal,
            "flow": flow,
            "residual": residual,
            "confidence": confidence,
        }


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _shift_clamped(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = array.shape[:2]
    x = np.clip(np.arange(width) + dx, 0, width - 1)
    y = np.clip(np.arange(height) + dy, 0, height - 1)
    return array[y[:, None], x[None, :]]


def _random_colour(rng: random.Random, low: int = 18, high: int = 235) -> tuple[int, int, int]:
    return tuple(rng.randint(low, high) for _ in range(3))


def _synthetic_clean(size: int, rng: random.Random) -> Image.Image:
    scale = 4
    large = size * scale
    image = Image.new("RGB", (large, large), _random_colour(rng, 18, 115))
    draw = ImageDraw.Draw(image, "RGBA")

    # Long hard-surface panel regions and recesses.
    for _ in range(rng.randint(8, 18)):
        x0 = rng.randint(-large // 6, large - 1)
        y0 = rng.randint(-large // 6, large - 1)
        x1 = x0 + rng.randint(large // 8, large // 2)
        y1 = y0 + rng.randint(large // 8, large // 2)
        radius = rng.randint(0, max(scale, large // 30))
        colour = _random_colour(rng, 8, 205) + (rng.randint(205, 255),)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=colour)
        if rng.random() < 0.72:
            width = rng.randint(scale * 2, scale * 7)
            trim = tuple(min(255, channel + rng.randint(18, 90)) for channel in colour[:3]) + (rng.randint(205, 255),)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, outline=trim, width=width)

    # Long contour paths, emissive strips and authored markings.
    for _ in range(rng.randint(18, 42)):
        point_count = rng.randint(2, 8)
        points = [(rng.randint(-large // 8, large + large // 8), rng.randint(-large // 8, large + large // 8))]
        heading = rng.uniform(-math.pi, math.pi)
        for _point in range(1, point_count):
            heading += rng.uniform(-0.50, 0.50)
            distance = rng.randint(large // 18, large // 3)
            points.append((
                int(points[-1][0] + math.cos(heading) * distance),
                int(points[-1][1] + math.sin(heading) * distance),
            ))
        width = rng.randint(max(2, scale), max(4, scale * 7))
        if rng.random() < 0.25:
            value = rng.randint(205, 255)
            colour = (value, min(255, value + rng.randint(-8, 8)), min(255, value + rng.randint(-8, 8)), 255)
        else:
            colour = _random_colour(rng, 8, 245) + (rng.randint(220, 255),)
        draw.line(points, fill=colour, width=width, joint="curve")

    # Low-frequency illumination/material variation. It must not become damage.
    overlay = Image.new("L", (large, large), 0)
    overlay_draw = ImageDraw.Draw(overlay)
    for _ in range(rng.randint(2, 6)):
        cx = rng.randint(0, large)
        cy = rng.randint(0, large)
        radius = rng.randint(large // 4, large)
        overlay_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=rng.randint(35, 150))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=large / 5.0))
    light = np.asarray(overlay, dtype=np.float32)[..., None] / 255.0
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.clip(rgb * (0.70 + light * 0.55), 0.0, 1.0)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), mode="RGB").resize(
        (size, size), Image.Resampling.LANCZOS)


def _inject_staircase_artifacts(image: Image.Image, rng: random.Random, fraction: float) -> Image.Image:
    if rng.random() > fraction:
        return image
    size = image.size[0]
    low_size = max(12, size // rng.choice((3, 4, 5, 6, 8)))
    mask_low = Image.new("L", (low_size, low_size), 0)
    colour_low = Image.new("RGB", (low_size, low_size), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_low)
    colour_draw = ImageDraw.Draw(colour_low)
    for _ in range(rng.randint(3, 12)):
        points = [(rng.randint(-4, low_size + 4), rng.randint(-4, low_size + 4))]
        heading = rng.uniform(-math.pi, math.pi)
        for _segment in range(rng.randint(2, 8)):
            heading += rng.uniform(-0.35, 0.35)
            distance = rng.randint(max(3, low_size // 14), max(5, low_size // 3))
            points.append((
                int(points[-1][0] + math.cos(heading) * distance),
                int(points[-1][1] + math.sin(heading) * distance),
            ))
        width = rng.choice((1, 1, 1, 2))
        value = rng.randint(195, 255) if rng.random() < 0.78 else rng.randint(0, 38)
        tint = tuple(max(0, min(255, value + rng.randint(-10, 10))) for _ in range(3))
        mask_draw.line(points, fill=rng.randint(180, 255), width=width, joint="curve")
        colour_draw.line(points, fill=tint, width=width, joint="curve")
    mask = mask_low.resize(image.size, Image.Resampling.NEAREST)
    colour = colour_low.resize(image.size, Image.Resampling.NEAREST)
    if rng.random() < 0.65:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75)))
    return Image.composite(colour, image, mask)


def _degrade(clean: Image.Image, rng: random.Random, artifact_fraction: float) -> Image.Image:
    width, height = clean.size
    factor = rng.choice((2, 3, 4, 4, 5, 6, 8))
    low = clean.resize((max(8, width // factor), max(8, height // factor)), Image.Resampling.BOX)
    if rng.random() < 0.75:
        low = low.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.85)))
    method = rng.choice((Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC))
    degraded = low.resize(clean.size, method)
    if rng.random() < 0.45:
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.10, 0.80)))
    return _inject_staircase_artifacts(degraded, rng, artifact_fraction)


def _semantic_maps_from_rgb(clean: np.ndarray) -> np.ndarray:
    """Generate training-time material evidence with the same channel semantics used at inference."""
    luma = _luminance(clean)
    gx = 0.5 * (_shift_clamped(luma[..., None], 1, 0)[..., 0] - _shift_clamped(luma[..., None], -1, 0)[..., 0])
    gy = 0.5 * (_shift_clamped(luma[..., None], 0, 1)[..., 0] - _shift_clamped(luma[..., None], 0, -1)[..., 0])
    magnitude = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
    normal_x = np.clip(gx * 5.0, -1.0, 1.0)
    normal_y = np.clip(gy * 5.0, -1.0, 1.0)
    material = np.clip(np.round(luma * 3.0) / 3.0, 0.0, 1.0)
    saturation = np.max(clean, axis=-1) - np.min(clean, axis=-1)
    paint = np.clip((magnitude - 0.012) / 0.18 + saturation * 0.55, 0.0, 1.0)
    roughness = np.clip(0.72 - magnitude * 1.8 + (1.0 - luma) * 0.15, 0.05, 0.95)
    return np.stack((normal_x, normal_y, material, paint, roughness), axis=-1).astype(np.float32)


def build_model_input(rgb: np.ndarray, semantic: np.ndarray | None = None) -> np.ndarray:
    """Return CxHxW float32 input expected by MaterialTileContextNet."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb[..., :3], 0.0, 1.0)
    if semantic is None:
        semantic = _semantic_maps_from_rgb(rgb)
    semantic = np.asarray(semantic, dtype=np.float32)
    if semantic.shape != (*rgb.shape[:2], 5):
        raise ValueError(f"semantic input must be HxWx5, got {semantic.shape}")
    return np.concatenate((rgb, semantic), axis=-1).transpose(2, 0, 1).astype(np.float32)


def _discover_source_images(config: TrainingConfig, repo_root: Path) -> list[Path]:
    if not config.source_root:
        return []
    source_root = Path(config.source_root).expanduser()
    if not source_root.is_absolute():
        source_root = (repo_root / source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"training source root does not exist: {source_root}")
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in config.source_globs:
        for path in sorted(source_root.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
            if config.max_source_files and len(found) >= config.max_source_files:
                return found
    return found


def _load_real_tile(path: Path, size: int, rng: random.Random) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if min(width, height) < 32:
                return None
            crop = min(width, height)
            left = rng.randint(0, width - crop) if width > crop else 0
            top = rng.randint(0, height - crop) if height > crop else 0
            return image.crop((left, top, left + crop, top + crop)).resize((size, size), Image.Resampling.LANCZOS)
    except Exception:
        return None


def _artifact_mask(clean: np.ndarray, degraded: np.ndarray) -> np.ndarray:
    error = np.mean(np.abs(clean - degraded), axis=-1)
    mask = np.clip((error - 0.004) / 0.12, 0.0, 1.0)
    # Slight dilation gives the model room to move a contour instead of only altering the brightest texel.
    tensor = torch.from_numpy(mask[None, None].astype(np.float32))
    tensor = F.max_pool2d(tensor, kernel_size=5, stride=1, padding=2)
    return tensor[0, 0].numpy()


def _training_batch(
    config: TrainingConfig,
    source_images: Sequence[Path],
    rng: random.Random,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    source_index = rng.randrange(len(source_images)) if source_images else 0
    while len(inputs) < batch_size:
        use_real = bool(source_images) and rng.random() < config.real_source_fraction
        if use_real:
            clean_image = _load_real_tile(source_images[source_index % len(source_images)], config.tile_size, rng)
            source_index += 1
            if clean_image is None:
                continue
            degraded_image = clean_image
        else:
            clean_image = _synthetic_clean(config.tile_size, rng)
            degraded_image = _degrade(clean_image, rng, config.artifact_fraction)
        clean = np.asarray(clean_image, dtype=np.float32) / 255.0
        degraded = np.asarray(degraded_image, dtype=np.float32) / 255.0
        semantic = _semantic_maps_from_rgb(clean)
        inputs.append(build_model_input(degraded, semantic))
        targets.append(clean.transpose(2, 0, 1).astype(np.float32))
        masks.append(_artifact_mask(clean, degraded)[None].astype(np.float32))
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(masks)),
    )


def _sobel(value: torch.Tensor) -> torch.Tensor:
    kernel_x = value.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    kernel_y = kernel_x.t()
    channels = value.shape[1]
    weight_x = kernel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    weight_y = kernel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    gx = F.conv2d(value, weight_x, padding=1, groups=channels)
    gy = F.conv2d(value, weight_y, padding=1, groups=channels)
    return torch.sqrt(torch.clamp(gx * gx + gy * gy, min=1.0e-8))


def _tv(value: torch.Tensor) -> torch.Tensor:
    return (value[:, :, 1:, :] - value[:, :, :-1, :]).abs().mean() + \
        (value[:, :, :, 1:] - value[:, :, :, :-1]).abs().mean()


def _cuda_install_hint() -> str:
    return (
        "Run scripts\\build\\setup_nsamdr_cuda.bat to create the dedicated "
        "CUDA 12.8 PyTorch environment for RTX 50-series GPUs."
    )


def resolve_device(config: TrainingConfig, requested_device: str | None = None) -> torch.device:
    selection = (requested_device or config.device).strip().lower()
    if selection == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if selection == "auto":
            print("WARNING: CUDA is unavailable; using CPU.", flush=True)
            return torch.device("cpu")
        raise SystemExit("CUDA training was requested but is unavailable. " + _cuda_install_hint())
    if config.cuda_device_index >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device index {config.cuda_device_index} is invalid")
    torch.cuda.set_device(config.cuda_device_index)
    device = torch.device(f"cuda:{config.cuda_device_index}")
    properties = torch.cuda.get_device_properties(device)
    capability = f"sm_{properties.major}{properties.minor}"
    architectures = set(torch.cuda.get_arch_list())
    if capability not in architectures and f"compute_{properties.major}{properties.minor}" not in architectures:
        raise SystemExit(
            f"Installed PyTorch does not contain {capability} kernels. Available: {' '.join(sorted(architectures))}. "
            + _cuda_install_hint())
    torch.set_float32_matmul_precision(config.matmul_precision)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = config.matmul_precision != "highest"
    return device


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[MaterialTileContextNet, TrainingConfig, dict[str, object]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint root is not a dictionary")
    if str(checkpoint.get("schema", "")) != MODEL_SCHEMA:
        raise RuntimeError(
            f"checkpoint schema {checkpoint.get('schema')!r} does not match required {MODEL_SCHEMA!r}; retraining is required")
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("checkpoint config is missing")
    config = TrainingConfig()
    for key, value in config_payload.items():
        if hasattr(config, key):
            if key == "source_globs":
                value = tuple(str(item) for item in value)
            setattr(config, key, value)
    config.validate()
    model = MaterialTileContextNet(
        config.base_channels,
        config.residual_blocks,
        config.max_offset_pixels,
        config.max_residual,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config, checkpoint


def _blend_window(tile_size: int, overlap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    edge = max(1, overlap)
    one = torch.ones(tile_size, device=device, dtype=dtype)
    ramp = torch.linspace(0.0, 1.0, edge + 2, device=device, dtype=dtype)[1:-1]
    one[:edge] = torch.sin(ramp * math.pi * 0.5) ** 2
    one[-edge:] = torch.flip(one[:edge], dims=(0,))
    return torch.outer(one, one).view(1, 1, tile_size, tile_size).clamp_min(1.0e-4)


@torch.no_grad()
def infer_tiled(
    model: MaterialTileContextNet,
    model_input: np.ndarray | torch.Tensor,
    device: torch.device | str,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Apply the fully convolutional model with overlap blending and no visible tile seams."""
    if isinstance(model_input, np.ndarray):
        tensor = torch.from_numpy(model_input)
    else:
        tensor = model_input.detach().cpu()
    if tensor.ndim != 3 or tensor.shape[0] != INPUT_CHANNELS:
        raise ValueError(f"model input must be {INPUT_CHANNELS}xHxW, got {tuple(tensor.shape)}")
    tensor = tensor.to(device=device, dtype=torch.float32).unsqueeze(0)
    _, _, height, width = tensor.shape
    tile_size = max(64, min(int(tile_size), max(height, width)))
    overlap = max(8, min(int(overlap), tile_size // 3))
    stride = max(16, tile_size - overlap * 2)
    pad_bottom = max(0, tile_size - height)
    pad_right = max(0, tile_size - width)
    if pad_bottom or pad_right:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    padded_height, padded_width = tensor.shape[-2:]
    ys = list(range(0, max(1, padded_height - tile_size + 1), stride))
    xs = list(range(0, max(1, padded_width - tile_size + 1), stride))
    if not ys or ys[-1] != padded_height - tile_size:
        ys.append(max(0, padded_height - tile_size))
    if not xs or xs[-1] != padded_width - tile_size:
        xs.append(max(0, padded_width - tile_size))
    output = torch.zeros((1, 3, padded_height, padded_width), device=device, dtype=torch.float32)
    weights = torch.zeros((1, 1, padded_height, padded_width), device=device, dtype=torch.float32)
    window = _blend_window(tile_size, overlap, torch.device(device), torch.float32)
    use_amp = torch.device(device).type == "cuda"
    for y in ys:
        for x in xs:
            tile = tensor[:, :, y:y + tile_size, x:x + tile_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                corrected = model(tile)["corrected"].float()
            output[:, :, y:y + tile_size, x:x + tile_size] += corrected * window
            weights[:, :, y:y + tile_size, x:x + tile_size] += window
    result = output / weights.clamp_min(1.0e-5)
    return result[0, :, :height, :width].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)


def train(config: TrainingConfig, repo_root: Path, requested_device: str | None = None) -> dict[str, object]:
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = resolve_device(config, requested_device)
    source_images = _discover_source_images(config, repo_root)
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"NSAMDR V4 device: {device}" + (f" - {properties.name}" if properties else ""), flush=True)
    print(f"NSAMDR source textures discovered: {len(source_images)}", flush=True)

    model = MaterialTileContextNet(
        config.base_channels,
        config.residual_blocks,
        config.max_offset_pixels,
        config.max_residual,
    ).to(device)
    print(
        f"Tile-context network: {parameter_count(model):,} parameters | "
        f"tile {config.tile_size} | receptive field {receptive_field_pixels(config.residual_blocks)}px | blocks {config.residual_blocks}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    steps_per_epoch = max(1, math.ceil(config.tiles_per_epoch / config.batch_size))
    rng = random.Random(config.seed)
    final_loss = 0.0
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for _step in range(steps_per_epoch):
            inputs, targets, artifact = _training_batch(config, source_images, rng, config.batch_size)
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            artifact = artifact.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(inputs)
                corrected = prediction["corrected"]
                correction_weight = 1.0 + artifact * 3.0
                reconstruction = ((corrected - targets).abs() * correction_weight).mean()
                edge = (_sobel(corrected) - _sobel(targets)).abs().mean()
                flat = 1.0 - artifact
                identity = ((corrected - inputs[:, 0:3]).abs() * flat).sum() / flat.sum().clamp_min(1.0)
                confidence = F.binary_cross_entropy(prediction["confidence"], artifact)
                smoothness = _tv(prediction["flow"]) + 0.35 * _tv(prediction["residual"])
                loss = (
                    config.reconstruction_weight * reconstruction
                    + config.edge_weight * edge
                    + config.identity_weight * identity
                    + config.confidence_weight * confidence
                    + config.flow_smoothness_weight * smoothness
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
        final_loss = running / steps_per_epoch
        print(f"epoch {epoch + 1}/{config.epochs}: loss={final_loss:.6f}", flush=True)

    model.eval()
    validation_rng = random.Random(config.seed + 1_000_003)
    inputs, targets, artifact = _training_batch(config, source_images[: max(1, len(source_images) // 4)], validation_rng, max(4, config.batch_size))
    inputs = inputs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    artifact = artifact.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(inputs)
        corrected = prediction["corrected"]
        validation_l1 = float((corrected - targets).abs().mean().cpu())
        validation_edge = float((_sobel(corrected) - _sobel(targets)).abs().mean().cpu())
        flat = artifact < 0.05
        flat_delta = float((corrected - inputs[:, 0:3]).abs()[flat.expand_as(corrected)].mean().cpu()) if bool(flat.any()) else 0.0
        confidence_mean = float(prediction["confidence"].mean().cpu())
        flow_mean = float(prediction["flow"].abs().mean().cpu())

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.checkpoint_name
    metadata_path = output_dir / config.metadata_name
    digest = model_hash(model)
    metrics = {
        "trainingLoss": final_loss,
        "validationL1": validation_l1,
        "validationEdge": validation_edge,
        "flatDeltaMean": flat_delta,
        "confidenceMean": confidence_mean,
        "flowMagnitudeMeanPixels": flow_mean,
    }
    checkpoint = {
        "schema": MODEL_SCHEMA,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": asdict(config),
        "model_sha256": digest,
        "parameter_count": parameter_count(model),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "schema": MODEL_SCHEMA,
        "modelSha256": digest,
        "checkpointPath": str(checkpoint_path.resolve()),
        "parameterCount": parameter_count(model),
        "inputChannels": ["albedoR", "albedoG", "albedoB", "normalX", "normalY", "material", "paint", "roughness"],
        "outputs": ["offsetX", "offsetY", "residualR", "residualG", "residualB", "confidence"],
        "dilations": list(DILATION_PATTERN[: config.residual_blocks]),
        "receptiveFieldPixels": receptive_field_pixels(config.residual_blocks),
        "runtime": "offline-overlapping-tile-inference",
        "device": str(device),
        "torchVersion": str(torch.__version__),
        "sourceTextureCount": len(source_images),
        "metrics": metrics,
        "config": asdict(config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"NSAMDR V4 checkpoint: {checkpoint_path}", flush=True)
    print(f"NSAMDR V4 metadata: {metadata_path}", flush=True)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-root", type=str)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"))
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = TrainingConfig.load(args.config)
    if args.source_root is not None:
        config.source_root = args.source_root
    train(config, repo_root, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
