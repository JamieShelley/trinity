#!/usr/bin/env python3
"""Render an unqualified V16 training checkpoint on the real EVE ship viewer.

This is the research/live counterpart of the strict production preview path.
It never promotes or qualifies a checkpoint.  It reconstructs deterministic B
and current candidate C from the same LR physical-map evidence, writes temporary
material manifests, and feeds the existing Granny-free DX11 A/B/C renderer.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
for import_root in (HERE, NSAMDR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import eve_asset_test as eve  # type: ignore
from v14.checkpoint import load_checkpoint as load_raven_checkpoint
from v14.config import V16Config
from v16.conditioning import StructureConditionedV16Candidate


FULL_BROAD_CHECKPOINT_SCHEMA = "NSAMDR_V16_FULL_BROAD_CHECKPOINT_V1"
MAIN_PREVIEW_SCHEMA = "NSAMDR_V16_FULL_BROAD_LIVE_PREVIEW_V1"
LIVE_POINTER_SCHEMA = "NSAMDR_LIVE_CANDIDATE_POINTER_V1"
DEFAULT_RAVEN = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"
PATH_COLUMNS = (
    "albedo",
    "normal",
    "material",
    "glow",
    "dirt",
    "ao",
    "paint_mask",
    "roughness_map",
)


@dataclass
class TextureContext:
    normal: Path | None = None
    normal_x_channel: int = 0
    normal_y_channel: int = 1
    material: Path | None = None
    material_channel: int = 0
    glow: Path | None = None
    glow_channel: int = 1
    roughness: Path | None = None
    roughness_channel: int = 2


@dataclass(frozen=True)
class PreviewSource:
    label: str
    ordinal: int
    phase: str
    checkpoint: Path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for render preview but is unavailable")
    return torch.device(name)


def _resolve_raven_source(repo_root: Path, experiment_id: str) -> PreviewSource | None:
    directory = repo_root / "artifacts/nsamdr/experiments" / experiment_id
    if not directory.is_dir():
        return None

    pointer = _read_json(directory / "previews/live/checkpoint_ready.json") or {}
    epoch = int(pointer.get("epoch") or 0)
    phase = str(pointer.get("phase") or "sr-candidate")
    raw_checkpoint = str(pointer.get("checkpoint") or "").strip()
    if raw_checkpoint:
        checkpoint = Path(raw_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = directory / checkpoint
        if checkpoint.is_file():
            return PreviewSource(experiment_id, epoch, phase, checkpoint.resolve())

    checkpoints = sorted(
        (directory / "checkpoints/candidate").glob("epoch_*.pt"),
        key=lambda path: path.name,
    )
    if not checkpoints:
        return None
    checkpoint = checkpoints[-1].resolve()
    try:
        epoch = int(checkpoint.stem.split("_")[-1])
    except ValueError:
        epoch = 0
    return PreviewSource(experiment_id, epoch, phase, checkpoint)


def _resolve_main_source(repo_root: Path) -> PreviewSource | None:
    root = repo_root / "artifacts/nsamdr/diagnostics/v16_full_broad"
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    if root.is_dir():
        for pointer_path in root.glob("probe_*/previews/latest.json"):
            payload = _read_json(pointer_path)
            if payload is None:
                continue
            if payload.get("schema") != MAIN_PREVIEW_SCHEMA:
                continue
            if str(payload.get("augmentationPolicy") or "") != "d4-cyclic":
                continue
            checkpoint_raw = str(payload.get("checkpoint") or "").strip()
            checkpoint = (
                Path(checkpoint_raw)
                if checkpoint_raw
                else pointer_path.parent.parent / "resume_checkpoint.pt"
            )
            if not checkpoint.is_absolute():
                checkpoint = repo_root / checkpoint
            if not checkpoint.is_file():
                continue
            try:
                stamp = pointer_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((stamp, checkpoint.resolve(), payload))

    if candidates:
        _stamp, checkpoint, payload = max(candidates, key=lambda item: item[0])
        return PreviewSource(
            "MAIN_V16_LATEST",
            int(payload.get("step") or 0),
            "main-v16.2-d4",
            checkpoint,
        )

    latest = _read_json(repo_root / "artifacts/nsamdr/main_training/latest.json") or {}
    raw = str(latest.get("resumeCheckpoint") or "").strip()
    if not raw:
        return None
    checkpoint = Path(raw)
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    if not checkpoint.is_file():
        return None
    return PreviewSource(
        "MAIN_V16_LATEST",
        int(latest.get("finalStep") or 0),
        "main-v16.2-d4",
        checkpoint.resolve(),
    )


def _resolve_source(repo_root: Path, subject: str) -> PreviewSource | None:
    value = subject.strip().upper()
    if value == "MAIN_V16_LATEST":
        return _resolve_main_source(repo_root)
    if value.startswith("EXP_"):
        return _resolve_raven_source(repo_root, value)
    raise RuntimeError(f"unknown V16 preview source: {subject}")


def _load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, V16Config, dict[str, Any]]:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid V16 checkpoint payload: {checkpoint}")

    schema = str(payload.get("schema") or "")
    if schema == "NSAMDR_HR_FIRST_MULTI_MAP_SR_4X_V16_0":
        model, loaded = load_raven_checkpoint(checkpoint, device)
        return model, model.config, dict(loaded)

    if schema == FULL_BROAD_CHECKPOINT_SCHEMA:
        raw_config = dict(payload.get("config") or {})
        fields = V16Config.__dataclass_fields__
        config = V16Config(
            **{key: value for key, value in raw_config.items() if key in fields}
        )
        config.validate()
        model = StructureConditionedV16Candidate(
            config,
            structure_channels=48,
            structure_blocks=4,
        ).to(device)
        model.load_state_dict(payload["modelState"], strict=True)
        model.eval()
        return model, config, dict(payload)

    raise RuntimeError(
        f"unsupported V16 render-preview checkpoint schema {schema!r}: {checkpoint}"
    )


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    comments: list[str] = []
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            comments.append(line)
        else:
            data_lines.append(line)
    if not data_lines:
        raise RuntimeError(f"material manifest has no rows: {path}")
    reader = csv.DictReader(data_lines, delimiter="\t")
    if reader.fieldnames is None:
        raise RuntimeError(f"material manifest has no TSV header: {path}")
    return list(reader.fieldnames), [dict(row) for row in reader], comments


def _resolve_path(value: str, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    return path if path.is_file() else None


def _channel(row: Mapping[str, str], name: str, fallback: int) -> int:
    try:
        return max(0, min(3, int(float(row.get(name, str(fallback)) or fallback))))
    except (TypeError, ValueError):
        return fallback


def _collect_contexts(
    rows: list[dict[str, str]],
    base: Path,
) -> dict[Path, TextureContext]:
    contexts: dict[Path, TextureContext] = {}
    for row in rows:
        albedo = _resolve_path(row.get("albedo", ""), base)
        if albedo is None:
            continue
        context = contexts.setdefault(albedo, TextureContext())
        normal = _resolve_path(row.get("normal", ""), base)
        material = _resolve_path(row.get("material", ""), base)
        glow = _resolve_path(row.get("glow", ""), base) or material
        roughness = _resolve_path(row.get("roughness_map", ""), base) or material
        if context.normal is None and normal is not None:
            context.normal = normal
            context.normal_x_channel = _channel(row, "normal_x_channel", 0)
            context.normal_y_channel = _channel(row, "normal_y_channel", 1)
        if context.material is None and material is not None:
            context.material = material
            context.material_channel = _channel(row, "material_channel", 0)
        if context.glow is None and glow is not None:
            context.glow = glow
            context.glow_channel = _channel(row, "glow_channel", 1)
        if context.roughness is None and roughness is not None:
            context.roughness = roughness
            context.roughness_channel = _channel(row, "roughness_channel", 2)
    return contexts


def _read_bgra(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read EVE texture: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(f"unsupported texture channels={image.shape[2]}: {path}")
    return image


def _bgra_channel(channel: int) -> int:
    return (2, 1, 0, 3)[max(0, min(3, int(channel)))]


def _resized_channel(
    path: Path | None,
    channel: int,
    width: int,
    height: int,
    default: float,
) -> np.ndarray:
    if path is None:
        return np.full((height, width), float(default), dtype=np.float32)
    image = _read_bgra(path)
    value = image[:, :, _bgra_channel(channel)].astype(np.float32) / 255.0
    return cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)


def _normalise_xy(value: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.maximum(np.sum(value * value, axis=-1, keepdims=True), 1.0e-8))
    return (value / np.maximum(1.0, length / 0.999)).astype(np.float32)


def _output_size(image: np.ndarray, target_size: int, scale: int) -> tuple[int, int]:
    height, width = image.shape[:2]
    factor = float(target_size) / float(max(width, height))
    out_width = max(scale, int(round(width * factor)))
    out_height = max(scale, int(round(height * factor)))
    # Swin windows and 4x reconstruction are happiest on scale-aligned dimensions.
    quantum = max(int(scale) * 8, int(scale))
    out_width = max(quantum, int(round(out_width / quantum)) * quantum)
    out_height = max(quantum, int(round(out_height / quantum)) * quantum)
    return out_width, out_height


def _model_inputs(
    albedo: Path,
    context: TextureContext,
    *,
    out_width: int,
    out_height: int,
    scale: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lr_width = max(1, int(math.ceil(out_width / scale)))
    lr_height = max(1, int(math.ceil(out_height / scale)))

    source = _read_bgra(albedo)
    rgb = cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2RGB)
    albedo_lr = cv2.resize(
        rgb,
        (lr_width, lr_height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0

    nx = _resized_channel(
        context.normal,
        context.normal_x_channel,
        lr_width,
        lr_height,
        0.5,
    ) * 2.0 - 1.0
    ny = _resized_channel(
        context.normal,
        context.normal_y_channel,
        lr_width,
        lr_height,
        0.5,
    ) * 2.0 - 1.0
    normal_lr = _normalise_xy(np.stack((nx, ny), axis=-1))

    material_lr = np.stack(
        (
            _resized_channel(
                context.material,
                context.material_channel,
                lr_width,
                lr_height,
                0.0,
            ),
            _resized_channel(
                context.glow,
                context.glow_channel,
                lr_width,
                lr_height,
                0.0,
            ),
            _resized_channel(
                context.roughness,
                context.roughness_channel,
                lr_width,
                lr_height,
                0.5,
            ),
        ),
        axis=-1,
    ).astype(np.float32)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.ascontiguousarray(value))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

    return tensor(albedo_lr), tensor(normal_lr), tensor(material_lr)


def _positions(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    values = list(range(0, max(1, size - tile + 1), step))
    last = size - tile
    if values[-1] != last:
        values.append(last)
    return values


def _window(height: int, width: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(height, periodic=False, device=device).clamp_min(0.05)
    wx = torch.hann_window(width, periodic=False, device=device).clamp_min(0.05)
    return (wy[:, None] * wx[None, :]).view(1, 1, height, width)


@torch.no_grad()
def _tiled_maps(
    model: torch.nn.Module,
    config: V16Config,
    lr_albedo: torch.Tensor,
    lr_normal: torch.Tensor,
    lr_material: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    h, w = lr_albedo.shape[-2:]
    tile = int(config.production_tile_lr)
    overlap = int(config.production_overlap_lr)
    ys = _positions(h, tile, overlap)
    xs = _positions(w, tile, overlap)
    scale = int(config.scale)
    out_h, out_w = h * scale, w * scale
    keys = (
        "baseline_albedo",
        "baseline_normal",
        "baseline_material",
        "candidate_albedo",
        "candidate_normal",
        "candidate_material",
    )
    channels = {
        "baseline_albedo": 3,
        "baseline_normal": 2,
        "baseline_material": 3,
        "candidate_albedo": 3,
        "candidate_normal": 2,
        "candidate_material": 3,
    }
    accum = {
        key: torch.zeros(
            (1, channels[key], out_h, out_w),
            device=device,
            dtype=torch.float32,
        )
        for key in keys
    }
    weight = torch.zeros((1, 1, out_h, out_w), device=device, dtype=torch.float32)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    for y in ys:
        for x in xs:
            a = lr_albedo[..., y : y + tile, x : x + tile]
            n = lr_normal[..., y : y + tile, x : x + tile]
            m = lr_material[..., y : y + tile, x : x + tile]
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(a, n, m)
            ph, pw = outputs["candidate_albedo"].shape[-2:]
            win = _window(ph, pw, device)
            oy, ox = y * scale, x * scale
            for key in keys:
                accum[key][..., oy : oy + ph, ox : ox + pw] += (
                    outputs[key].float() * win
                )
            weight[..., oy : oy + ph, ox : ox + pw] += win

    result = {key: value / weight.clamp_min(1.0e-6) for key, value in accum.items()}
    for key in ("baseline_normal", "candidate_normal"):
        value = result[key]
        length = torch.sqrt(value.square().sum(dim=1, keepdim=True).clamp_min(1.0e-8))
        result[key] = value / torch.maximum(
            torch.ones_like(length),
            length / 0.999,
        )
    return result


def _tensor_image(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .float()
        .cpu()[0]
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
    )


def _ensure_canvas(
    canvases: dict[Path, np.ndarray],
    source: Path,
    width: int,
    height: int,
) -> np.ndarray:
    existing = canvases.get(source)
    if existing is not None:
        if existing.shape[:2] != (height, width):
            raise RuntimeError(f"packed texture requested at incompatible sizes: {source}")
        return existing
    original = _read_bgra(source)
    canvas = cv2.resize(original, (width, height), interpolation=cv2.INTER_LANCZOS4)
    canvases[source] = canvas
    return canvas


def _apply_maps(
    *,
    albedo: Path,
    context: TextureContext,
    albedo_map: np.ndarray,
    normal_map: np.ndarray,
    material_map: np.ndarray,
    width: int,
    height: int,
    canvases: dict[Path, np.ndarray],
) -> None:
    albedo_canvas = _ensure_canvas(canvases, albedo, width, height)
    rgb = np.uint8(np.rint(np.clip(albedo_map, 0.0, 1.0) * 255.0))
    albedo_canvas[:, :, :3] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if context.normal is not None:
        canvas = _ensure_canvas(canvases, context.normal, width, height)
        canvas[:, :, _bgra_channel(context.normal_x_channel)] = np.uint8(
            np.rint(np.clip((normal_map[:, :, 0] + 1.0) * 127.5, 0.0, 255.0))
        )
        canvas[:, :, _bgra_channel(context.normal_y_channel)] = np.uint8(
            np.rint(np.clip((normal_map[:, :, 1] + 1.0) * 127.5, 0.0, 255.0))
        )

    for source, channel, plane in (
        (context.material, context.material_channel, material_map[:, :, 0]),
        (context.glow, context.glow_channel, material_map[:, :, 1]),
        (context.roughness, context.roughness_channel, material_map[:, :, 2]),
    ):
        if source is None:
            continue
        canvas = _ensure_canvas(canvases, source, width, height)
        canvas[:, :, _bgra_channel(channel)] = np.uint8(
            np.rint(np.clip(plane, 0.0, 1.0) * 255.0)
        )


def _write_canvases(
    output_root: Path,
    folder: str,
    suffix: str,
    canvases: Mapping[Path, np.ndarray],
) -> dict[Path, Path]:
    root = output_root / folder
    root.mkdir(parents=True, exist_ok=True)
    replacements: dict[Path, Path] = {}
    for source, canvas in sorted(canvases.items(), key=lambda item: str(item[0]).casefold()):
        token = hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:10]
        destination = root / f"{source.stem}_{token}_{suffix}.png"
        if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"could not write render-preview texture: {destination}")
        replacements[source] = destination.resolve()
    return replacements


def _write_material_manifest(
    output: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    comments: list[str],
    replacements: Mapping[Path, Path],
    source_dir: Path,
    label: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# NSAMDR_MATERIALS_V7\n")
        handle.write(f"# PHYSICAL_CANDIDATE {label}\n")
        for comment in comments:
            if comment.startswith("#") and "NSAMDR_MATERIALS" not in comment and "PHYSICAL_CANDIDATE" not in comment:
                handle.write(comment + "\n")
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            adjusted = dict(row)
            for semantic in PATH_COLUMNS:
                source = _resolve_path(row.get(semantic, ""), source_dir)
                if source in replacements:
                    adjusted[semantic] = str(replacements[source].resolve())
            writer.writerow(adjusted)


def _generate_candidate(
    *,
    repo_root: Path,
    source: PreviewSource,
    target_size: int,
    device: torch.device,
    obj_path: Path,
    materials: Path,
) -> dict[str, Any]:
    checkpoint_sha = _sha256(source.checkpoint)
    model, config, payload = _load_model(source.checkpoint, device)
    model.eval()

    fields, rows, comments = _read_tsv(materials)
    contexts = _collect_contexts(rows, materials.parent)
    if not contexts:
        raise RuntimeError("prepared EVE ship has no aligned albedo contexts")

    live_root = repo_root / "artifacts/nsamdr/render_preview" / source.label
    output_root = live_root / "candidates" / f"step_{source.ordinal:06d}_{checkpoint_sha[:12]}"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_canvases: dict[Path, np.ndarray] = {}
    candidate_canvases: dict[Path, np.ndarray] = {}
    for index, (albedo, context) in enumerate(
        sorted(contexts.items(), key=lambda item: str(item[0]).casefold()),
        start=1,
    ):
        original = _read_bgra(albedo)
        width, height = _output_size(original, target_size, int(config.scale))
        lr_a, lr_n, lr_m = _model_inputs(
            albedo,
            context,
            out_width=width,
            out_height=height,
            scale=int(config.scale),
            device=device,
        )
        maps = _tiled_maps(
            model,
            config,
            lr_a,
            lr_n,
            lr_m,
            device=device,
        )
        _apply_maps(
            albedo=albedo,
            context=context,
            albedo_map=_tensor_image(maps["baseline_albedo"]),
            normal_map=_tensor_image(maps["baseline_normal"]),
            material_map=_tensor_image(maps["baseline_material"]),
            width=width,
            height=height,
            canvases=baseline_canvases,
        )
        _apply_maps(
            albedo=albedo,
            context=context,
            albedo_map=_tensor_image(maps["candidate_albedo"]),
            normal_map=_tensor_image(maps["candidate_normal"]),
            material_map=_tensor_image(maps["candidate_material"]),
            width=width,
            height=height,
            canvases=candidate_canvases,
        )
        print(
            f"[render-preview] {source.label} {index}/{len(contexts)} "
            f"{albedo.name}: LR {lr_a.shape[-1]} -> HR {width}x{height}",
            flush=True,
        )
        del maps, lr_a, lr_n, lr_m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_replacements = _write_canvases(
        output_root,
        "baseline_4x",
        "baseline_4x",
        baseline_canvases,
    )
    candidate_replacements = _write_canvases(
        output_root,
        "candidate_c",
        "candidate_c",
        candidate_canvases,
    )

    baseline_materials = output_root / "baseline.materials.tsv"
    candidate_materials = output_root / "candidate.materials.tsv"
    _write_material_manifest(
        baseline_materials,
        fields,
        rows,
        comments,
        baseline_replacements,
        materials.parent,
        "NSAMDR_DETERMINISTIC_4X_BASELINE",
    )
    _write_material_manifest(
        candidate_materials,
        fields,
        rows,
        comments,
        candidate_replacements,
        materials.parent,
        "NSAMDR_TRAINING_INTERMEDIATE_UNQUALIFIED_C",
    )
    candidate_obj = output_root / obj_path.name
    shutil.copy2(obj_path, candidate_obj)

    report = {
        "schema": "NSAMDR_V16_RENDER_PREVIEW_CANDIDATE_V1",
        "authority": "training-intermediate",
        "qualified": False,
        "source": source.label,
        "ordinal": int(source.ordinal),
        "phase": source.phase,
        "checkpoint": str(source.checkpoint.resolve()),
        "checkpointSha256": checkpoint_sha,
        "checkpointSchema": str(payload.get("schema") or ""),
        "baselineObj": str(candidate_obj.resolve()),
        "baselineMaterials": str(baseline_materials.resolve()),
        "candidateObj": str(candidate_obj.resolve()),
        "candidateMaterials": str(candidate_materials.resolve()),
        "targetSize": int(target_size),
    }
    report_path = output_root / "candidate_manifest.json"
    report["reportPath"] = str(report_path.resolve())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _sha256(source.checkpoint) != checkpoint_sha:
        raise RuntimeError("render-preview checkpoint changed during candidate generation")
    return report


def _pointer_text(report: Mapping[str, Any]) -> str:
    token = (
        f"{report['source']}-{int(report['ordinal']):06d}-"
        f"{str(report['checkpointSha256'])[:16]}"
    )
    return "\n".join(
        (
            LIVE_POINTER_SCHEMA,
            f"token={token}",
            f"epoch={report['ordinal']}",
            f"phase={report['phase']}",
            f"stageVariant=C-candidate",
            f"checkpointSha256={report['checkpointSha256']}",
            f"baselineObj={report['baselineObj']}",
            f"baselineMaterials={report['baselineMaterials']}",
            f"candidateObj={report['candidateObj']}",
            f"candidateMaterials={report['candidateMaterials']}",
            f"candidateManifest={report['reportPath']}",
            "authority=training-intermediate",
            "qualified=false",
            "",
        )
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Render a current V16 Raven/Main training checkpoint on the real EVE ship"
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--source", required=True)
    value.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    value.add_argument("--target-size", type=int, default=1024)
    value.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    value.add_argument("--poll-seconds", type=float, default=1.0)
    value.add_argument("--ship-query", default=DEFAULT_RAVEN)
    value.add_argument(
        "--watch",
        action="store_true",
        help="hot-reload the running DX11 viewer when a newer epoch becomes available",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 512 <= int(args.target_size) <= 2048:
        raise RuntimeError("--target-size must be from 512 to 2048 for training render preview")

    repo_root = args.repo_root.resolve()
    device = _device(str(args.device))
    (
        obj_path,
        albedo,
        normal,
        pgs,
        environment,
        environments,
        material_manifest,
        asset_manifest,
        catalog,
        cache_root,
    ) = eve.prepare_asset(repo_root, str(args.shared_cache), str(args.ship_query), "")
    if material_manifest is None or not material_manifest.is_file():
        raise RuntimeError("prepared EVE preview asset has no physical material manifest")

    live_root = repo_root / "artifacts/nsamdr/render_preview" / args.source.strip().upper()
    pointer_path = live_root / "current_candidate.txt"
    launcher = repo_root / "scripts/build/run_nsamdr_obj_preview_dx11.bat"
    renderer_thread: threading.Thread | None = None
    renderer_result: list[int] = []
    last_token = ""

    def launch_renderer() -> None:
        os.environ.update(
            {
                "NSAMDR_PREVIEW_EXPERIMENT": args.source.strip().upper(),
                "NSAMDR_PREVIEW_AUTHORITY": "training-intermediate",
                "NSAMDR_LIVE_CANDIDATE_POINTER": str(pointer_path.resolve()),
                "NSAMDR_PREVIEW_REUSE_EXISTING_VIEWER": "1",
            }
        )
        asset_data = _read_json(asset_manifest) or {}
        selected_query = str(
            (asset_data.get("model") or {}).get("logical") or args.ship_query
        )
        renderer_result.append(
            eve.launch_preview(
                repo_root,
                launcher,
                obj_path,
                albedo,
                normal,
                pgs,
                environment,
                environments,
                material_manifest,
                asset_manifest,
                catalog,
                cache_root,
                selected_query,
                None,
            )
        )

    while True:
        source = _resolve_source(repo_root, str(args.source))
        if source is None:
            if renderer_thread is None:
                raise RuntimeError(
                    f"{args.source} has no completed checkpoint available for render preview"
                )
        else:
            checkpoint_sha = _sha256(source.checkpoint)
            token = f"{source.label}:{source.ordinal}:{checkpoint_sha}"
            if token != last_token:
                report = _generate_candidate(
                    repo_root=repo_root,
                    source=source,
                    target_size=int(args.target_size),
                    device=device,
                    obj_path=obj_path,
                    materials=material_manifest,
                )
                _atomic_text(pointer_path, _pointer_text(report))
                last_token = token
                print(
                    f"[render-preview] PUBLISHED {source.label} "
                    f"{source.ordinal} ({source.phase})",
                    flush=True,
                )
                if renderer_thread is None:
                    renderer_thread = threading.Thread(target=launch_renderer, daemon=True)
                    renderer_thread.start()

        if renderer_thread is not None and not renderer_thread.is_alive():
            return int(renderer_result[-1] if renderer_result else 0)
        if renderer_thread is not None and not args.watch:
            renderer_thread.join()
            return int(renderer_result[-1] if renderer_result else 0)
        time.sleep(max(0.25, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
