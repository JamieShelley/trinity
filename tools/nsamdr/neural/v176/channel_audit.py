"""Research-only native Raven physical-map channel interpolation audit.

No neural inference, no source resampling in production and no learned weights.
Exports small native-coordinate diagnostic crops to compare interpolation
artifacts per semantic channel. A visualization crop is NOT an input crop.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.nsamdr.neural.v176 import raven_renderer as native

SCHEMA = "NSAMDR_V176_RAVEN_CHANNEL_AUDIT_V1"
SEMANTICS = (
    "albedo", "normal_x", "normal_y", "material", "glow",
    "roughness", "paint_mask", "dirt", "ao",
)


def _native_roi(image: np.ndarray, origin: tuple[int, int], crop: int) -> np.ndarray:
    if image.ndim < 2:
        raise ValueError("source texture must have two spatial dimensions")
    x, y = origin
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("ROI origin must be within native source")
    return image[y:min(y + crop, h), x:min(x + crop, w)]


def _pick_roi(source: np.ndarray, crop: int) -> tuple[int, int]:
    """Choose a high-contrast native crop (not an authored target)."""
    h, w = source.shape[:2]
    if h <= crop and w <= crop:
        return 0, 0
    grey = cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2GRAY)
    # Tile the original source without resizing its pixels. Sample at most a
    # few dozen candidate origins to keep audits bounded for 4K sources.
    xs = np.unique(np.linspace(0, max(0, w-crop), min(8, max(1, w//crop+1)), dtype=np.int64))
    ys = np.unique(np.linspace(0, max(0, h-crop), min(8, max(1, h//crop+1)), dtype=np.int64))
    best = (-1.0, 0, 0)
    for y in ys:
        for x in xs:
            patch = grey[int(y):int(y)+crop, int(x):int(x)+crop].astype(np.float32)
            dx = np.abs(np.diff(patch, axis=1)).mean() if patch.shape[1] > 1 else 0
            dy = np.abs(np.diff(patch, axis=0)).mean() if patch.shape[0] > 1 else 0
            score = float(dx+dy)
            if score > best[0]:
                best = (score, int(x), int(y))
    return best[1], best[2]


def _crop_channel(path: Path | None, channel: int, reference_size: tuple[int, int],
                  origin: tuple[int, int], crop: int, default: int) -> np.ndarray:
    """Map ROI coordinates from native albedo to an aligned physical map."""
    target_h, target_w = reference_size
    x, y = origin
    end_x = min(x+crop, target_w)
    end_y = min(y+crop, target_h)
    if path is None:
        return np.full((end_y-y, end_x-x), default, dtype=np.uint8)
    source = native._read_bgra(path)
    source_h, source_w = source.shape[:2]
    # For maps of differing native resolution, identify the same UV rectangle.
    sx0 = int(round(x * source_w / target_w))
    sx1 = int(round(end_x * source_w / target_w))
    sy0 = int(round(y * source_h / target_h))
    sy1 = int(round(end_y * source_h / target_h))
    sx0 = max(0, min(source_w-1, sx0))
    sy0 = max(0, min(source_h-1, sy0))
    sx1 = max(sx0+1, min(source_w, sx1))
    sy1 = max(sy0+1, min(source_h, sy1))
    sampled = source[sy0:sy1, sx0:sx1, native._bgra_channel(channel)]
    desired = (end_x-x, end_y-y)
    if (sampled.shape[1], sampled.shape[0]) == desired:
        return sampled
    method = cv2.INTER_AREA if sampled.shape[0] > desired[1] else cv2.INTER_LINEAR
    return cv2.resize(sampled, desired, interpolation=method)


def _panel(channel: np.ndarray, label: str, interpolation: int, scale: int = 4) -> np.ndarray:
    if channel.ndim != 2:
        raise ValueError("channel view must be grayscale")
    resized = cv2.resize(channel, (channel.shape[1]*scale, channel.shape[0]*scale),
                         interpolation=interpolation)
    out = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1]-1, 28), (15, 15, 15), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255,255,255), 1, cv2.LINE_AA)
    return out


def _write_one(channel: np.ndarray, output: Path, semantic: str) -> dict[str, Any]:
    nearest = cv2.resize(channel, (channel.shape[1]*4, channel.shape[0]*4),
                         interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(channel, (channel.shape[1]*4, channel.shape[0]*4),
                        interpolation=cv2.INTER_LINEAR)
    # A difference map is diagnostic only, not a proposed correction for
    # categorical masks or physically calibrated gloss/normal channels.
    delta = cv2.absdiff(nearest, linear)
    heat = cv2.applyColorMap(delta, cv2.COLORMAP_INFERNO)
    cv2.rectangle(heat, (0,0), (heat.shape[1]-1,28), (15,15,15), -1)
    cv2.putText(heat, "ABS DIFF (nearest vs linear)", (8,20),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
    sheet = np.hstack((
        _panel(channel, f"{semantic}: nearest", cv2.INTER_NEAREST),
        _panel(channel, f"{semantic}: bilinear", cv2.INTER_LINEAR),
        heat,
    ))
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_PNG_COMPRESSION,3]):
        raise RuntimeError(f"could not export diagnostic sheet: {output}")
    return {
        "semantic": semantic,
        "sheet": str(output.resolve()),
        "nativeCropSize": [int(channel.shape[1]), int(channel.shape[0])],
        "differenceMAE_8bit": float(delta.mean()),
        "differenceMaximum_8bit": int(delta.max()),
        "semanticWarning": (
            "Interpolation is visual comparison only. Never smooth categorical "
            "masks without a semantic boundary policy."
        ),
    }


def run(materials: Path, output_root: Path, *, crop_size: int = 128,
        max_textures: int = 6, origin: tuple[int,int] | None = None) -> dict[str, Any]:
    if not (16 <= crop_size <= 512):
        raise ValueError("diagnostic ROI must be 16..512 native pixels")
    if not (1 <= max_textures <= 32):
        raise ValueError("max textures must be 1..32")
    fields, rows, _comments = native._read_tsv(materials)
    contexts = native._collect_contexts(rows, materials.parent)
    if not contexts:
        raise RuntimeError("no native Raven albedo contexts")
    # These three maps pass through the original preview unchanged. Include
    # them so stair-stepping is not misattributed to a neural output channel.
    preserved: dict[Path, dict[str, tuple[Path, int]]] = {}
    for row in rows:
        source = native._resolve_path(row.get("albedo", ""), materials.parent)
        if source is None:
            continue
        entry = preserved.setdefault(source, {})
        for semantic, default in (("paint_mask", 0), ("dirt", 0), ("ao", 0)):
            path = native._resolve_path(row.get(semantic, ""), materials.parent)
            if path is not None:
                entry.setdefault(semantic, (
                    path, native._channel(row, semantic + "_channel", default)
                ))
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for albedo, context in list(sorted(contexts.items(), key=lambda entry: str(entry[0]).lower()))[:max_textures]:
        source = native._read_bgra(albedo)
        point = origin if origin is not None else _pick_roi(source, crop_size)
        h, w = source.shape[:2]
        if point[0] >= w or point[1] >= h or min(point) < 0:
            raise ValueError(f"requested diagnostic ROI {point} outside {w}x{h}")
        hash_id = hashlib.sha1(str(albedo.resolve()).encode("utf-8")).hexdigest()[:10]
        folder = output_root / f"{albedo.stem}_{hash_id}"
        folder.mkdir(exist_ok=True)
        channels = {
            "albedo": cv2.cvtColor(_native_roi(source, point, crop_size)[:, :, :3],
                                  cv2.COLOR_BGR2GRAY),
            "normal_x": _crop_channel(context.normal, context.normal_x_channel,
                                      (h,w), point, crop_size, 128),
            "normal_y": _crop_channel(context.normal, context.normal_y_channel,
                                      (h,w), point, crop_size, 128),
            "material": _crop_channel(context.material, context.material_channel,
                                      (h,w), point, crop_size, 0),
            "glow": _crop_channel(context.glow, context.glow_channel,
                                      (h,w), point, crop_size, 0),
            "roughness": _crop_channel(context.roughness, context.roughness_channel,
                                      (h,w), point, crop_size, 128),
        }
        for semantic, (path, channel) in preserved.get(albedo, {}).items():
            channels[semantic] = _crop_channel(
                path, channel, (h, w), point, crop_size, 0
            )
        produced = []
        for semantic, channel in channels.items():
            if semantic not in ("albedo", "paint_mask", "dirt", "ao") and getattr(context, {
                "normal_x":"normal", "normal_y":"normal",
                "material":"material", "glow":"glow",
                "roughness":"roughness"}[semantic]) is None:
                continue
            produced.append(_write_one(channel, folder / f"{semantic}.png", semantic))
        reports.append({
            "source": str(albedo.resolve()),
            "sourceNativeDimensions": [int(w),int(h)],
            "roiOriginNative": list(point),
            "roiExtentNative": [int(channels["albedo"].shape[1]),int(channels["albedo"].shape[0])],
            "channels": produced,
        })
    report = {
        "schema": SCHEMA,
        "mode": "native-roi-diagnostic-only",
        "modelWeightsUpdated": False,
        "neuralInferencePerformed": False,
        "sourceAltered": False,
        "nativeResolutionPreserved": True,
        "productionPromotionAllowed": False,
        "materialsManifest": str(materials.resolve()),
        "textures": reports,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report
