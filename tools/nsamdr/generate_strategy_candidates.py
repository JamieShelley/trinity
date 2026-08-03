#!/usr/bin/env python3
"""Generate the single public NSAMDR reconstruction candidate.

Public viewer modes:
1. untouched source asset;
2. UV/stretch diagnostics rendered from the source asset;
3. trained NSAMDR reconstruction using overlapping tile-context inference
   baked into prepared 4K material textures before live shader sampling.

Historical experimental candidates are no longer generated or exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from strategy_pipeline import CandidateArtifact, StrategyManifest

NEURAL_DIR = Path(__file__).resolve().parent / "neural"
if str(NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(NEURAL_DIR))
import train_nsamdr_kernel as tile_model

REPORT_SCHEMA = "NSAMDR_THREE_MODE_PIPELINE_V2_TILE_CONTEXT"


def _ensure_dependencies(install: bool) -> None:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        if not install:
            raise SystemExit(
                "NumPy and OpenCV are required. Run:\n"
                f'  "{sys.executable}" -m pip install numpy opencv-python-headless'
            )
        print("Installing NSAMDR candidate-generation dependencies...", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "numpy", "opencv-python-headless",
        ])
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for V4 tile-context inference. Run "
            "scripts\\build\\setup_nsamdr_cuda.bat or setup_nsamdr_cpu.bat, "
            "then launch through test_nsamdr_real_eve_asset.bat."
        ) from exc


@dataclass(frozen=True)
class Usage:
    semantic: str
    x_channel: int | None = None  # Manifest channels are RGBA order.
    y_channel: int | None = None


@dataclass
class AlbedoContext:
    normal: Path | None = None
    normal_x_channel: int = 0
    normal_y_channel: int = 1
    material: Path | None = None
    material_channel: int = 0
    paint: Path | None = None
    paint_channel: int = 0
    roughness: Path | None = None
    roughness_channel: int = 0


SEMANTICS = (
    "albedo", "normal", "material", "glow", "dirt", "ao",
    "paint_mask", "roughness_map",
)
CHANNEL_COLUMNS = {
    "normal": ("normal_x_channel", "normal_y_channel"),
    "material": ("material_channel", None),
    "glow": ("glow_channel", None),
    "dirt": ("dirt_channel", None),
    "ao": ("ao_channel", None),
    "paint_mask": ("paint_channel", None),
    "roughness_map": ("roughness_channel", None),
}


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    comments: list[str] = []
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            comments.append(line)
        else:
            data_lines.append(line)
    if not data_lines:
        raise ValueError(f"No material rows in {path}")
    reader = csv.DictReader(data_lines, delimiter="\t")
    rows = [dict(row) for row in reader]
    if reader.fieldnames is None:
        raise ValueError(f"No TSV header in {path}")
    return list(reader.fieldnames), rows, comments


def _resolve_path(value: str, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    return path if path.is_file() else None


def _parse_channel(row: dict[str, str], name: str | None, fallback: int = 0) -> int | None:
    if name is None:
        return None
    try:
        return max(0, min(3, int(float(row.get(name, str(fallback)) or fallback))))
    except ValueError:
        return fallback


def _rgba_to_bgra_channel(channel: int) -> int:
    """Translate manifest RGBA indices to OpenCV BGRA array indices."""
    return (2, 1, 0, 3)[max(0, min(3, channel))]


def _collect_usages(rows: list[dict[str, str]], base: Path) -> dict[Path, list[Usage]]:
    usages: dict[Path, list[Usage]] = {}
    for row in rows:
        for semantic in SEMANTICS:
            source = _resolve_path(row.get(semantic, ""), base)
            if source is None:
                continue
            x_name, y_name = CHANNEL_COLUMNS.get(semantic, (None, None))
            usage = Usage(semantic, _parse_channel(row, x_name), _parse_channel(row, y_name))
            bucket = usages.setdefault(source, [])
            if usage not in bucket:
                bucket.append(usage)
    return usages


def _collect_albedo_contexts(rows: list[dict[str, str]], base: Path) -> dict[Path, AlbedoContext]:
    contexts: dict[Path, AlbedoContext] = {}
    for row in rows:
        albedo = _resolve_path(row.get("albedo", ""), base)
        if albedo is None:
            continue
        context = contexts.setdefault(albedo, AlbedoContext())
        normal = _resolve_path(row.get("normal", ""), base)
        material = _resolve_path(row.get("material", ""), base)
        paint = _resolve_path(row.get("paint_mask", ""), base)
        roughness = _resolve_path(row.get("roughness_map", ""), base)
        if context.normal is None and normal is not None:
            context.normal = normal
            context.normal_x_channel = _parse_channel(row, "normal_x_channel", 0) or 0
            context.normal_y_channel = _parse_channel(row, "normal_y_channel", 1) or 1
        if context.material is None and material is not None:
            context.material = material
            context.material_channel = _parse_channel(row, "material_channel", 0) or 0
        if context.paint is None and paint is not None:
            context.paint = paint
            context.paint_channel = _parse_channel(row, "paint_channel", 0) or 0
        if context.roughness is None:
            selected = roughness or material
            if selected is not None:
                context.roughness = selected
                context.roughness_channel = _parse_channel(row, "roughness_channel", 0) or 0
    return contexts


def _resized_channel(source: Path | None, rgba_channel: int, width: int, height: int, default: float):
    import cv2
    import numpy as np
    if source is None:
        return np.full((height, width), default, dtype=np.float32)
    image = _read_bgra(source)
    channel = image[:, :, _rgba_to_bgra_channel(rgba_channel)].astype(np.float32) / 255.0
    return cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)


def _semantic_context_for_albedo(
    context: AlbedoContext | None,
    rgb: "object",
    width: int,
    height: int,
):
    import cv2
    import numpy as np
    fallback = tile_model._semantic_maps_from_rgb(rgb.astype(np.float32) / 255.0)
    if context is None:
        return fallback
    semantic = fallback.copy()
    if context.normal is not None:
        semantic[:, :, 0] = _resized_channel(context.normal, context.normal_x_channel, width, height, 0.5) * 2.0 - 1.0
        semantic[:, :, 1] = _resized_channel(context.normal, context.normal_y_channel, width, height, 0.5) * 2.0 - 1.0
    if context.material is not None:
        semantic[:, :, 2] = _resized_channel(context.material, context.material_channel, width, height, 0.0)
    if context.paint is not None:
        semantic[:, :, 3] = _resized_channel(context.paint, context.paint_channel, width, height, 0.0)
    if context.roughness is not None:
        semantic[:, :, 4] = _resized_channel(context.roughness, context.roughness_channel, width, height, 0.5)
    return np.ascontiguousarray(semantic.astype(np.float32))


def _apply_tile_context_model(
    high_bgra,
    context: AlbedoContext | None,
    model,
    model_config: tile_model.TrainingConfig,
    device,
):
    import cv2
    import numpy as np
    rgb = cv2.cvtColor(high_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    semantic = _semantic_context_for_albedo(context, rgb, width, height)
    model_input = tile_model.build_model_input(rgb, semantic)
    corrected = tile_model.infer_tiled(
        model,
        model_input,
        device,
        tile_size=model_config.inference_tile_size,
        overlap=model_config.inference_overlap,
    )
    corrected_rgb = np.uint8(np.round(np.clip(corrected, 0.0, 1.0) * 255.0))
    before = rgb.astype(np.float32)
    delta_rms = float(np.sqrt(np.mean((corrected_rgb.astype(np.float32) - before) ** 2)))
    high_bgra[:, :, :3] = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    return delta_rms


def _unique_output_name(source: Path) -> str:
    digest = hashlib.sha1(str(source).lower().encode("utf-8")).hexdigest()[:10]
    return f"{source.stem}_{digest}_4k.png"


def _read_bgra(source: Path):
    import cv2
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read texture: {source}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(f"Unsupported channel count in {source}: {image.shape}")
    return image


def _unsharp(channel, cv2, amount: float, sigma: float):
    blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(channel, 1.0 + amount, blurred, -amount, 0.0)


def _find_realesrgan(tool_dir: Path) -> tuple[Path, str] | None:
    candidates: list[Path] = []
    env_path = os.environ.get("NSAMDR_REALESRGAN_EXE", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    third_party = tool_dir / "third_party" / "realesrgan"
    if third_party.is_dir():
        candidates.extend(third_party.rglob("realesrgan-ncnn-vulkan.exe"))
    for executable in candidates:
        if not executable.is_file():
            continue
        model_roots = [executable.parent / "models", executable.parent]
        available: set[str] = set()
        for root in model_roots:
            if not root.is_dir():
                continue
            for param in root.rglob("*.param"):
                if param.with_suffix(".bin").is_file():
                    available.add(param.stem)
        requested = os.environ.get("NSAMDR_REALESRGAN_MODEL", "").strip()
        preferences = [requested] if requested else []
        preferences += ["realesrnet-x4plus", "realesrgan-x4plus", "realesr-general-x4v3"]
        for model in preferences:
            if model and model in available:
                return executable.resolve(), model
    return None


def _run_realesrgan(source_bgr, executable: Path, model: str, target_size: int):
    import cv2
    with tempfile.TemporaryDirectory(prefix="nsamdr_sr_") as temporary:
        temp = Path(temporary)
        input_path = temp / "input.png"
        output_path = temp / "output.png"
        if not cv2.imwrite(str(input_path), source_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError("Could not write Real-ESRGAN input")
        command = [
            str(executable), "-i", str(input_path), "-o", str(output_path),
            "-n", model, "-s", "4", "-t", "0", "-f", "png",
        ]
        result = subprocess.run(
            command, cwd=executable.parent, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                f"Real-ESRGAN failed with exit code {result.returncode}:\n{result.stdout[-4000:]}"
            )
        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if output is None:
            raise RuntimeError("Real-ESRGAN produced an unreadable image")
        if output.shape[0] != target_size or output.shape[1] != target_size:
            output = cv2.resize(output, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        return output


def _classic_albedo(source_bgr, target_size: int):
    """Fidelity-first deterministic enlargement with anti-ringing."""
    import cv2
    import numpy as np

    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    source_l, source_a, source_b = cv2.split(source_lab)
    # Remove block/fuzz energy from the broad colour field while retaining hard
    # authored edges in a separately bounded residual.
    low = cv2.bilateralFilter(source_l, 9, 16, 5)
    blur = cv2.GaussianBlur(source_l, (0, 0), 1.05)
    residual = source_l.astype(np.float32) - blur.astype(np.float32)
    local_mean = cv2.GaussianBlur(source_l.astype(np.float32), (0, 0), 2.0)
    local_var = cv2.GaussianBlur((source_l.astype(np.float32) - local_mean) ** 2, (0, 0), 2.0)
    sigma = np.sqrt(np.maximum(local_var, 0.0))
    residual = np.clip(residual, -0.82 * sigma - 1.0, 0.82 * sigma + 1.0)

    low_up = cv2.resize(low, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    source_up = cv2.resize(source_l, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    detail_up = cv2.resize(residual, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    restored_l = np.clip(0.60 * low_up + 0.40 * source_up + 1.18 * detail_up, 0.0, 255.0).astype(np.uint8)
    a_up = cv2.resize(source_a, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    b_up = cv2.resize(source_b, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    # Chroma noise is a major source of low-resolution colour fuzz.
    a_up = cv2.bilateralFilter(a_up, 5, 5, 2)
    b_up = cv2.bilateralFilter(b_up, 5, 5, 2)
    return cv2.cvtColor(cv2.merge((restored_l, a_up, b_up)), cv2.COLOR_LAB2BGR)



def _smoothstep_array(edge0: float, edge1: float, value):
    import numpy as np
    t = np.clip((value - edge0) / max(edge1 - edge0, 1.0e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _iterative_back_projection(high, target, cv2, iterations: int = 2, gain: float = 0.62):
    """Restore source-consistent edge energy without unconstrained sharpening."""
    import numpy as np
    source_h, source_w = target.shape[:2]
    result = high.astype(np.float32)
    target_f = target.astype(np.float32)
    for _ in range(max(iterations, 0)):
        down = cv2.resize(result, (source_w, source_h), interpolation=cv2.INTER_AREA)
        residual = target_f - down
        correction = cv2.resize(
            residual,
            (result.shape[1], result.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        result += correction * gain
    return result


def _structure_aware_albedo(source_bgr, target_size: int):
    """Fuzz-suppressing 4K reconstruction that protects coherent authored structure.

    Flat low-confidence high-frequency energy is treated as source fuzz. Coherent
    panel edges and line work are reconstructed through a gated Laplacian pyramid
    and iterative back-projection, then bounded by a local anti-ringing envelope.
    """
    import cv2
    import numpy as np

    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    l8, a8, b8 = cv2.split(source_lab)
    l = l8.astype(np.float32)

    fine_blur = cv2.GaussianBlur(l, (0, 0), 0.62)
    mid_blur = cv2.GaussianBlur(l, (0, 0), 1.35)
    broad_blur = cv2.GaussianBlur(l, (0, 0), 2.8)
    fine = l - fine_blur
    medium = fine_blur - mid_blur
    coarse = mid_blur - broad_blur

    gx = cv2.Scharr(l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(l, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    gradient_reference = max(float(np.percentile(gradient, 94.0)), 1.0)
    edge_strength = np.clip(gradient / gradient_reference, 0.0, 1.0)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.2)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.2)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.2)
    coherence = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0)) / (jxx + jyy + 1.0e-5)

    median = cv2.medianBlur(l8, 3).astype(np.float32)
    impulsive = np.abs(l - median)
    noise_reference = max(float(np.percentile(impulsive, 82.0)), 1.0)
    noise_like = np.clip(impulsive / noise_reference, 0.0, 1.0) * (1.0 - edge_strength)

    detail_confidence = np.clip(
        0.14 + 0.58 * edge_strength + 0.42 * coherence - 0.52 * noise_like,
        0.0,
        1.0,
    )
    fine_gate = _smoothstep_array(0.18, 0.74, detail_confidence)
    medium_gate = _smoothstep_array(0.08, 0.56, detail_confidence)

    # Remove weak, incoherent source fuzz while preserving authored line work.
    reconstructed_source = (
        broad_blur
        + coarse * (0.98 + 0.12 * medium_gate)
        + medium * (0.88 + 0.34 * medium_gate)
        + fine * (0.18 + 0.92 * fine_gate)
    )
    reconstructed_source = np.clip(reconstructed_source, 0.0, 255.0)

    high = cv2.resize(
        reconstructed_source,
        (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4,
    )
    high = _iterative_back_projection(high, reconstructed_source, cv2, iterations=3, gain=0.58)

    edge_high = cv2.resize(edge_strength, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    confidence_high = cv2.resize(detail_confidence, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    local_blur = cv2.GaussianBlur(high, (0, 0), 0.82)
    bounded_detail = high - local_blur
    sharpen_gain = 0.10 + 0.34 * np.clip(edge_high * 0.72 + confidence_high * 0.28, 0.0, 1.0)
    high += bounded_detail * sharpen_gain

    source_min = cv2.erode(reconstructed_source, np.ones((3, 3), np.uint8))
    source_max = cv2.dilate(reconstructed_source, np.ones((3, 3), np.uint8))
    low_envelope = cv2.resize(source_min, (target_size, target_size), interpolation=cv2.INTER_LINEAR) - 2.0
    high_envelope = cv2.resize(source_max, (target_size, target_size), interpolation=cv2.INTER_LINEAR) + 2.0
    high = np.clip(high, low_envelope, high_envelope)
    high = np.clip(high, 0.0, 255.0).astype(np.uint8)

    # Chroma receives stronger fuzz suppression than luminance. This removes the
    # low-resolution colour crawling visible on large dark and metallic panels.
    a_clean = cv2.bilateralFilter(a8, 9, 7, 4)
    b_clean = cv2.bilateralFilter(b8, 9, 7, 4)
    a_high = cv2.resize(a_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    b_high = cv2.resize(b_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.merge((high, a_high, b_high)), cv2.COLOR_LAB2BGR)


def _reconstruct_scalar_plane_v2(source_plane, target_size: int, *, crisp: bool):
    import cv2
    import numpy as np

    source = source_plane.astype(np.float32)
    if crisp:
        cleaned = cv2.bilateralFilter(source, 5, 7.5, 2.2)
        gain = 0.64
        sharpen = 0.22
    else:
        cleaned = cv2.bilateralFilter(source, 7, 5.5, 3.4)
        gain = 0.56
        sharpen = 0.10
    high = cv2.resize(cleaned, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    high = _iterative_back_projection(high, cleaned, cv2, iterations=2, gain=gain)
    blurred = cv2.GaussianBlur(high, (0, 0), 0.78 if crisp else 1.05)
    high += (high - blurred) * sharpen
    minimum = cv2.resize(cv2.erode(source_plane, np.ones((3, 3), np.uint8)), (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    maximum = cv2.resize(cv2.dilate(source_plane, np.ones((3, 3), np.uint8)), (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return np.clip(high, minimum.astype(np.float32) - 1.0, maximum.astype(np.float32) + 1.0)


def _reconstruct_normal_pair_v2(source_x, source_y, target_size: int):
    import cv2
    import numpy as np

    x_source = source_x.astype(np.float32) / 127.5 - 1.0
    y_source = source_y.astype(np.float32) / 127.5 - 1.0
    x_clean = cv2.bilateralFilter(x_source, 5, 0.045, 2.6)
    y_clean = cv2.bilateralFilter(y_source, 5, 0.045, 2.6)
    x_high = cv2.resize(x_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    y_high = cv2.resize(y_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    x_high = _iterative_back_projection(x_high, x_clean, cv2, iterations=2, gain=0.52)
    y_high = _iterative_back_projection(y_high, y_clean, cv2, iterations=2, gain=0.52)

    x_blur = cv2.GaussianBlur(x_high, (0, 0), 0.86)
    y_blur = cv2.GaussianBlur(y_high, (0, 0), 0.86)
    x_high += (x_high - x_blur) * 0.18
    y_high += (y_high - y_blur) * 0.18
    length_sq = x_high * x_high + y_high * y_high
    limiter = np.maximum(1.0, np.sqrt(np.maximum(length_sq, 1.0e-8)) / 0.982)
    x_high /= limiter
    y_high /= limiter
    return x_high, y_high


def _prepare_nsamdr_texture(
    source: Path,
    usages: list[Usage],
    output: Path,
    target_size: int,
    neural: tuple[Path, str] | None,
    albedo_context: AlbedoContext | None,
    tile_runtime: tuple[object, tile_model.TrainingConfig, object] | None,
) -> tuple[int, int, str, dict[str, float]]:
    """Prepare a stable 4K material texture and bake V4 tile-context cleanup into albedo."""
    import cv2
    import numpy as np

    image = _read_bgra(source)
    height, width = image.shape[:2]
    if width == height:
        out_w = out_h = target_size
    else:
        scale = target_size / max(width, height)
        out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))

    high = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    original_high = high.copy()
    semantic_names = {usage.semantic for usage in usages}
    backend = "structure-aware-v2"
    tile_model_delta_rms = 0.0

    if "albedo" in semantic_names:
        classic_v2 = _structure_aware_albedo(image[:, :, :3], target_size)
        albedo = classic_v2
        if neural is not None:
            executable, model = neural
            try:
                learned = _run_realesrgan(image[:, :, :3], executable, model, target_size)
                # The optional learned upsampler proposes edge placement while the
                # deterministic branch remains responsible for texture fidelity.
                albedo = cv2.addWeighted(classic_v2, 0.66, learned, 0.34, 0.0)
                backend = f"structure-aware-v2+realesrgan:{model}"
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: optional neural upsampling failed for {source.name}; using deterministic v2: {exc}", file=sys.stderr)
        if (out_w, out_h) != (target_size, target_size):
            albedo = cv2.resize(albedo, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        high[:, :, :3] = albedo

    processed_channels: set[int] = set((0, 1, 2)) if "albedo" in semantic_names else set()
    normal_pairs: set[tuple[int, int]] = set()
    for usage in usages:
        if usage.semantic == "normal" and usage.x_channel is not None and usage.y_channel is not None:
            normal_pairs.add((_rgba_to_bgra_channel(usage.x_channel), _rgba_to_bgra_channel(usage.y_channel)))
    for x_index, y_index in sorted(normal_pairs):
        x, y = _reconstruct_normal_pair_v2(image[:, :, x_index], image[:, :, y_index], target_size)
        if (out_w, out_h) != (target_size, target_size):
            x = cv2.resize(x, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            y = cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        high[:, :, x_index] = np.clip((x + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        high[:, :, y_index] = np.clip((y + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        processed_channels.update((x_index, y_index))

    crisp_semantics = {"material", "paint_mask", "glow"}
    soft_semantics = {"dirt", "ao", "roughness_map"}
    for usage in usages:
        if usage.x_channel is None or usage.semantic not in crisp_semantics | soft_semantics:
            continue
        channel = _rgba_to_bgra_channel(usage.x_channel)
        if channel in processed_channels:
            continue
        plane = _reconstruct_scalar_plane_v2(
            image[:, :, channel],
            target_size,
            crisp=usage.semantic in crisp_semantics,
        )
        if (out_w, out_h) != (target_size, target_size):
            plane = cv2.resize(plane, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        high[:, :, channel] = np.clip(plane, 0.0, 255.0).astype(np.uint8)
        processed_channels.add(channel)

    for channel in range(4):
        if channel not in processed_channels:
            high[:, :, channel] = original_high[:, :, channel]

    if "albedo" in semantic_names:
        if tile_runtime is not None:
            model, model_config, model_device = tile_runtime
            tile_model_delta_rms = _apply_tile_context_model(
                high, albedo_context, model, model_config, model_device)
            backend += "+tile-context-v4"
        else:
            # Preserve a real Mode 3 comparison before the first V4 training run.
            # This deterministic bootstrap candidate is replaced automatically
            # after nsamdr_tile_context.pt is created.
            backend += "+tile-context-v4-pending"

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), high, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Could not write prepared NSAMDR texture: {output}")

    luma_before = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    down = cv2.resize(high[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
    luma_after = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual_rms = float(np.sqrt(np.mean((luma_after - luma_before) ** 2)))
    return out_w, out_h, backend, {
        "sourceWidth": float(width),
        "sourceHeight": float(height),
        "roundTripLumaRms": residual_rms,
        "tileModelDeltaRms": tile_model_delta_rms,
    }



def _write_manifest(
    output: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    comments: list[str],
    replacements: dict[Path, Path],
    source_dir: Path,
    label: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# NSAMDR_MATERIALS_V5\n")
        handle.write(f"# PHYSICAL_CANDIDATE {label}\n")
        for comment in comments:
            if comment.startswith("#") and "NSAMDR_MATERIALS" not in comment:
                handle.write(comment + "\n")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            adjusted = dict(row)
            for semantic in SEMANTICS:
                source = _resolve_path(row.get(semantic, ""), source_dir)
                if source in replacements:
                    adjusted[semantic] = str(replacements[source].resolve())
            writer.writerow(adjusted)


def _latest_input_mtime(paths: Iterable[Path]) -> float:
    return max((path.stat().st_mtime for path in paths if path.is_file()), default=0.0)


def _report_is_fresh(report_path: Path, inputs: Iterable[Path], target_size: int) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if report.get("schema") != REPORT_SCHEMA or int(report.get("targetSize", 0)) != target_size:
        return False
    required = [Path(str(report.get(key, ""))) for key in (
        "mode3Obj", "mode3Materials", "mode3Analysis", "mode3Validation",
    )]
    if not all(path.is_file() for path in required):
        return False
    return report_path.stat().st_mtime >= _latest_input_mtime(inputs)


def generate_candidates(
    obj: Path,
    materials: Path,
    asset_manifest: Path | None,
    output_root: Path,
    target_size: int,
    force: bool,
    super_resolution_backend: str,
    inference_device: str,
) -> dict[str, object]:
    report_path = output_root / "strategy_candidates.json"
    fields, rows, comments = _read_tsv(materials)
    usages = _collect_usages(rows, materials.parent)
    albedo_contexts = _collect_albedo_contexts(rows, materials.parent)
    tool_dir = Path(__file__).resolve().parent
    neural_sr = _find_realesrgan(tool_dir) if super_resolution_backend in {"auto", "realesrgan"} else None
    if super_resolution_backend == "realesrgan" and neural_sr is None:
        raise RuntimeError(
            "Real-ESRGAN was requested but is not configured. Install the backend or use --super-resolution-backend auto/classic."
        )

    inputs = [
        obj,
        materials,
        Path(__file__).resolve(),
        tool_dir / "strategy_pipeline" / "reconstruction.py",
        *usages.keys(),
    ]
    if neural_sr is not None:
        inputs.append(neural_sr[0])
    repository_root = tool_dir.parents[1]
    neural_checkpoint = repository_root / "artifacts" / "nsamdr" / "neural" / "nsamdr_tile_context.pt"
    neural_metadata = repository_root / "artifacts" / "nsamdr" / "neural" / "nsamdr_tile_context.json"
    checkpoint = {}
    tile_runtime = None
    device = "not-trained"
    if neural_checkpoint.is_file():
        inputs.extend((neural_checkpoint, neural_metadata))
        checkpoint_probe = tile_model.TrainingConfig()
        checkpoint_probe.device = inference_device
        device = tile_model.resolve_device(checkpoint_probe, inference_device)
        model, model_config, checkpoint = tile_model.load_trained_model(neural_checkpoint, device)
        tile_runtime = (model, model_config, device)
    else:
        print(
            "WARNING: V4 tile-context checkpoint is missing. Generating a deterministic Mode 3 bootstrap candidate; "
            "run scripts\\build\\train_nsamdr.bat to replace it with the trained tile-context result. "
            "The previous V3 per-pixel checkpoint is intentionally incompatible.",
            file=sys.stderr,
            flush=True,
        )

    if not force and _report_is_fresh(report_path, inputs, target_size):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"NSAMDR Mode 3 candidate is current: {report_path}", flush=True)
        return report

    mode3_dir = output_root / "mode3_nsamdr_neural"
    mode3_dir.mkdir(parents=True, exist_ok=True)

    replacements: dict[Path, Path] = {}
    dimensions: dict[str, list[int]] = {}
    backends: set[str] = set()
    metrics_by_source: dict[str, dict[str, float]] = {}
    print(
        f"Preparing Mode 3 NSAMDR 4K inputs with V4 tile-context inference "
        f"({len(usages)} unique source images, device={device})...", flush=True)
    for index, (source, source_usages) in enumerate(sorted(usages.items(), key=lambda item: str(item[0]).lower()), 1):
        output = mode3_dir / _unique_output_name(source).replace("_4k.png", "_nsamdr_4k.png")
        width, height, backend, metrics = _prepare_nsamdr_texture(
            source, source_usages, output, target_size, neural_sr,
            albedo_contexts.get(source), tile_runtime,
        )
        replacements[source] = output
        dimensions[str(output.resolve())] = [width, height]
        backends.add(backend)
        metrics_by_source[str(source.resolve())] = metrics
        print(
            f"  [{index}/{len(usages)}] {source.name} -> {output.name} "
            f"({width}x{height}, {backend}, round-trip RMS={metrics['roundTripLumaRms']:.3f}, model delta={metrics['tileModelDeltaRms']:.3f})",
            flush=True,
        )

    mode3_materials = mode3_dir / "ship.materials.mode3.tsv"
    _write_manifest(
        mode3_materials,
        fields,
        rows,
        comments,
        replacements,
        materials.parent,
        "MODE3_NSAMDR_TILE_CONTEXT_V4",
    )
    mode3_obj = mode3_dir / obj.name
    mode3_obj.write_bytes(obj.read_bytes())

    validation_failures = [
        source for source, metrics in metrics_by_source.items()
        if not (0.0 <= float(metrics.get("roundTripLumaRms", 999.0)) <= 18.0)
    ]
    all_outputs_exist = all(path.is_file() for path in replacements.values())

    mode3_analysis = mode3_dir / "mode3.analysis.json"
    mode3_analysis.write_text(json.dumps({
        "schema": "NSAMDR_MODE3_ANALYSIS_V4_TILE_CONTEXT",
        "phase": "prepared-input+offline-overlapping-tile-context-reconstruction",
        "targetSize": target_size,
        "textureCount": len(replacements),
        "textureDimensions": dimensions,
        "preparationBackends": sorted(backends),
        "roundTripMetrics": metrics_by_source,
        "trainedCheckpoint": str(neural_checkpoint.resolve()) if neural_checkpoint.is_file() else "",
        "trainedMetadata": str(neural_metadata.resolve()) if neural_metadata.is_file() else "",
        "trainedMetadataPresent": neural_metadata.is_file(),
        "tileContextApplied": tile_runtime is not None,
        "bootstrapCandidate": tile_runtime is None,
        "modelSha256": str(checkpoint.get("model_sha256", "")),
        "parameterCount": int(checkpoint.get("parameter_count", 0)),
        "runtimeStages": {
            "1_training": "tools/nsamdr/neural/train_nsamdr_kernel.py",
            "2_tileInference": "generate_strategy_candidates.py overlapping 4K tiles",
            "3_bakedMaterialOutput": "mode3_nsamdr_neural/*_nsamdr_4k.png",
            "4_liveShaderSampling": "NSAMDRRenderPipeline.cpp -> NSAMDRPreview.hlsl",
        },
        "semanticMaps": "deterministic-pass-through",
        "authoredUvTopologyPreserved": True,
    }, indent=2) + "\n", encoding="utf-8")

    mode3_validation = mode3_dir / "mode3.validation.json"
    validation_payload = {
        "schema": "NSAMDR_MODE3_VALIDATION_V4_TILE_CONTEXT",
        "passed": all_outputs_exist and not validation_failures,
        "failedTextures": validation_failures,
        "checks": {
            "allPreparedInputsExist": all_outputs_exist,
            "roundTripLumaRmsMaximum": 18.0,
            "tileCheckpointExists": neural_checkpoint.is_file(),
            "trainedMetadataPresent": neural_metadata.is_file(),
            "runtimeComputeKernelRequired": False,
            "overlappingTileInferenceBaked": tile_runtime is not None,
            "bootstrapCandidateAvailable": tile_runtime is None,
            "semanticMapsRemainDeterministic": True,
            "authoredUvTopologyPreserved": True,
        },
    }
    mode3_validation.write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    if not validation_payload["passed"]:
        raise RuntimeError("Mode 3 NSAMDR candidate validation failed")

    manifest = StrategyManifest(REPORT_SCHEMA, target_size, obj, materials)
    manifest.add(CandidateArtifact(
        mode=3,
        label="NSAMDR tile-context cleanup",
        obj=mode3_obj,
        materials=mode3_materials,
        metadata={
            "analysis": str(mode3_analysis.resolve()),
            "validation": str(mode3_validation.resolve()),
            "textureCount": len(replacements),
            "backend": "prepared-4k+trained-tile-context-v4" if tile_runtime is not None else "prepared-4k+deterministic-bootstrap",
            "textureDimensions": dimensions,
            "runtimeNeuralKernel": False,
            "offlineTileContext": tile_runtime is not None,
            "bootstrapCandidate": tile_runtime is None,
            "liveShaderSampling": True,
            "semanticMapsDeterministic": True,
            "authoredUvPreserved": True,
        },
    ))
    manifest.notes.extend([
        "Public Mode 1 is the untouched source asset.",
        "Public Mode 2 is the existing UV/stretch diagnostic render.",
        "Public Mode 3 always exists: before V4 training it uses a deterministic prepared-4K bootstrap; after training it uses overlapping tile-context reconstruction baked offline.",
        "Historical intermediate appearance modes are no longer generated or exported.",
    ])
    report = manifest.to_report()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"NSAMDR three-mode candidate manifest: {report_path}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NSAMDR physical strategy candidates")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--materials", required=True, type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-dependencies", action="store_true")
    parser.add_argument(
        "--super-resolution-backend", choices=("auto", "classic", "realesrgan"), default="auto",
        help="auto uses Real-ESRGAN when installed and otherwise uses the fidelity-first classic backend",
    )
    parser.add_argument(
        "--inference-device", choices=("auto", "cuda", "cpu"), default="auto",
        help="device used to bake the V4 tile-context candidate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _ensure_dependencies(args.install_dependencies)
        generate_candidates(
            args.obj.resolve(), args.materials.resolve(),
            args.asset_manifest.resolve() if args.asset_manifest else None,
            args.output_root.resolve(), args.target_size, args.force,
            args.super_resolution_backend, args.inference_device,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
