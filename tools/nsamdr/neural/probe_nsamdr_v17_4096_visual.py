#!/usr/bin/env python3
"""One inference-only 1024->4096 V17.2 A/B/C diagnostic for a reviewed V2 authority.

Uses a native authored 4096 albedo region, V2's selected checkpoint and a
1024 clean physical input. Never opens an untouched V2 final-test authority.
No training, checkpoint modification, qualification or native EVE promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from authored_texture_dataset import AuthoredTextureDatasetRepository, TextureSource
from probe_nsamdr_v16_full_broad import _autocast, _device
from probe_nsamdr_v17_generalization_v2 import (
    CHECKPOINT_SCHEMA, PREVIOUS_VALIDATION_AUTHORITIES, ROOT, SCHEMA,
)
from probe_nsamdr_v17_sibling_architecture import _preview_panel
from v14.config import V16Config
from v14.dataset import _degrade, _normalise_xy
from v17.model import NSAMDRV17

DIAGNOSTIC_SCHEMA = "NSAMDR_V17_4096_VISUAL_DIAGNOSTIC_V1"
HR_SIZE = 4096
LR_SIZE = 1024


def _latest_report(repo: Path) -> tuple[Path, dict[str, Any]]:
    reports: list[tuple[float, Path, dict[str, Any]]] = []
    for path in (repo / ROOT).glob("proof_*/report.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("schema") == SCHEMA and obj.get("selectedCheckpoint"):
                reports.append((path.stat().st_mtime, path.resolve(), obj))
        except (OSError, ValueError, TypeError):
            continue
    if not reports:
        raise RuntimeError("No completed Generalization V2 report with a selected checkpoint")
    _, path, data = max(reports, key=lambda x: x[0])
    return path, data


def _reviewed_source(
    manifest: dict[str, Any],
    report: dict[str, Any],
    authority: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only already-evaluated V1 development authorities; reject sealed families."""
    reviewed_ids = set(report.get("validationCropIds") or ())
    reviewed_families = {item.rsplit("_", 1)[0] for item in reviewed_ids}
    if authority not in PREVIOUS_VALIDATION_AUTHORITIES or authority not in reviewed_families:
        raise RuntimeError(f"authority {authority} is not an already-reviewed development authority")
    if authority in {item.rsplit("_", 1)[0] for item in report.get("finalTestCropIds") or ()}:
        raise RuntimeError("refusing to open a sealed final-test authority")
    family = next(
        (item for item in manifest.get("families", [])
         if str(item.get("familyId") or "") == authority), None,
    )
    crop = next(
        (item for item in manifest.get("crops", [])
         if str(item.get("crop_id") or item.get("cropId") or "") == f"{authority}_001"), None,
    )
    if family is None or crop is None:
        raise RuntimeError(f"no authored family/crop provenance for reviewed authority {authority}")
    source = family.get("albedo") or {}
    if min(int(source.get("width") or 0), int(source.get("height") or 0)) < HR_SIZE:
        raise RuntimeError(f"native authored albedo for {authority} is smaller than {HR_SIZE}")
    normal = family.get("normal") or {}
    if not normal.get("path"):
        raise RuntimeError(f"aligned authored normal missing for {authority}")
    if not Path(str(source.get("path") or "")).is_file() or not Path(str(normal["path"])).is_file():
        raise RuntimeError(f"original authored albedo/normal source files unavailable for {authority}")
    return family, crop


def _choose_reviewed_source(
    manifest: dict[str, Any], report: dict[str, Any], requested: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    choices = [requested] if requested != "auto" else list(PREVIOUS_VALIDATION_AUTHORITIES)
    problems: list[str] = []
    for authority in choices:
        try:
            family, crop = _reviewed_source(manifest, report, authority)
            return authority, family, crop
        except RuntimeError as error:
            problems.append(str(error))
    raise RuntimeError(
        "No reviewed authority has a local genuine >=4096 authored albedo "
        "and aligned normal. The existing 512 crops cannot be passed off as "
        "native 4096 ground truth. " + " | ".join(problems)
    )


def _region_box(crop: dict[str, Any], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Native 4096 region centered around the previously reviewed sibling crop."""
    height, width = shape
    if min(height, width) < HR_SIZE:
        raise RuntimeError("refusing to upscale authored ground truth")
    values = crop.get("source_box") or crop.get("sourceBox")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise RuntimeError("reviewed sibling source box missing")
    normalized = [float(x) / 1_000_000.0 for x in values]
    if not all(0.0 <= x <= 1.0 for x in normalized):
        raise RuntimeError("invalid reviewed source box")
    center_x = int(round((normalized[0] + normalized[2]) * 0.5 * width))
    center_y = int(round((normalized[1] + normalized[3]) * 0.5 * height))
    left = max(0, min(width - HR_SIZE, center_x - HR_SIZE // 2))
    top = max(0, min(height - HR_SIZE, center_y - HR_SIZE // 2))
    return left, top, left + HR_SIZE, top + HR_SIZE


def _prepare_authored(
    repo: Path, family: dict[str, Any], crop: dict[str, Any],
    temp: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode original physical maps, retaining *native* albedo pixels for A."""
    loader = AuthoredTextureDatasetRepository()
    sources: dict[str, Path] = {}
    source_meta: dict[str, Any] = {}
    for role in ("albedo", "normal", "material"):
        value = family.get(role)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid {role} authored source descriptor")
        src = TextureSource(**value)
        if role == "material" and not Path(src.path).is_file():
            continue  # same neutral unsupervised material policy as the V2 corpus
        if not Path(src.path).is_file():
            raise RuntimeError(f"missing authored source: {src.path}")
        sources[role] = loader._decode_source(
            src, temp / f"{family['familyId']}_{role}.png", repo, None,
        )
        source_meta[role] = {"logical": src.logical, "path": src.path,
                             "width": src.width, "height": src.height}
    albedo = loader._load_rgb(sources["albedo"])
    left, top, right, bottom = _region_box(crop, albedo.shape[:2])
    reference = np.ascontiguousarray(albedo[top:bottom, left:right])
    normalized_box = (
        left / albedo.shape[1], top / albedo.shape[0],
        right / albedo.shape[1], bottom / albedo.shape[0],
    )
    normal, normal_encoding = loader.load_normal_training_rgb(sources["normal"])
    normal = loader._crop_aligned(normal, normalized_box, HR_SIZE, nearest=False)
    normal_xy = _normalise_xy(normal[..., :2].astype(np.float32) / 127.5 - 1.0)
    if "material" in sources:
        material = loader._crop_aligned(
            loader._load_rgb(sources["material"]), normalized_box, HR_SIZE, nearest=True,
        ).astype(np.float32) / 255.0
        material_status = "authored"
    else:
        material = np.full((HR_SIZE, HR_SIZE, 3), 128.0 / 255.0, dtype=np.float32)
        material_status = "unsupervised-neutral-placeholder"
    return reference, normal_xy, material, {
        "nativeAlbedoBox": [left, top, right, bottom],
        "sourceDimensions": [int(albedo.shape[1]), int(albedo.shape[0])],
        "sources": source_meta, "normalEncoding": normal_encoding,
        "materialStatus": material_status,
        "nativeAuthoredPixels": True,
    }


def _tile_ranges(length: int, tile: int) -> list[tuple[int, int]]:
    if length < 1 or tile < 1:
        raise ValueError("positive tile dimensions required")
    return [(start, min(length, start + tile)) for start in range(0, length, tile)]


def _tiled_4x(
    model: NSAMDRV17,
    lr_albedo: np.ndarray, lr_normal: np.ndarray, lr_material: np.ndarray,
    *, device: torch.device, precision: str, tile_lr: int = 128, halo_lr: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Halo-crop and stitch native-scale interior pixels, no blending or fake detail."""
    if tile_lr < 8 or halo_lr < 1:
        raise ValueError("tile_lr must be >=8 and halo_lr positive")
    if lr_albedo.shape != lr_material.shape or lr_albedo.shape[-1] != 3:
        raise ValueError("albedo/material arrays must align as HWC three-channel maps")
    if lr_normal.shape[:2] != lr_albedo.shape[:2] or lr_normal.shape[-1] != 2:
        raise ValueError("normal XY must be aligned with albedo")
    height, width = lr_albedo.shape[:2]
    b = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
    c = np.empty_like(b)
    model.eval()

    def tensor(value: np.ndarray) -> torch.Tensor:
        data = np.ascontiguousarray(value.transpose(2, 0, 1))
        return torch.from_numpy(data).unsqueeze(0).to(device=device, dtype=torch.float32)

    count = 0
    for y0, y1 in _tile_ranges(height, tile_lr):
        for x0, x1 in _tile_ranges(width, tile_lr):
            xx0, xx1 = max(0, x0 - halo_lr), min(width, x1 + halo_lr)
            yy0, yy1 = max(0, y0 - halo_lr), min(height, y1 + halo_lr)
            with torch.inference_mode():
                with _autocast(device, precision):
                    outputs = model(
                        tensor(lr_albedo[yy0:yy1, xx0:xx1]),
                        tensor(lr_normal[yy0:yy1, xx0:xx1]),
                        tensor(lr_material[yy0:yy1, xx0:xx1]),
                    )
                sy, sx = 4 * (y0 - yy0), 4 * (x0 - xx0)
                ey, ex = sy + 4 * (y1 - y0), sx + 4 * (x1 - x0)
                for key, dst in (("baseline_albedo", b), ("candidate_albedo", c)):
                    core = outputs[key][0, :, sy:ey, sx:ex]
                    rgb = (
                        core.float().clamp(0, 1).mul(255).round()
                        .to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
                    )
                    dst[4*y0:4*y1, 4*x0:4*x1] = rgb
            count += 1
            if count % 8 == 0 or count == 1:
                print(f"[v17-4096] tiles={count} origin_lr=({x0},{y0})", flush=True)
    return b, c


def _metrics(authored: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Diagnostic pixel/edge errors only. Not a trained/validation acceptance gate."""
    def edge(gray: np.ndarray) -> np.ndarray:
        return cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    a = authored.astype(np.float32) / 255.0
    b = baseline.astype(np.float32) / 255.0
    c = candidate.astype(np.float32) / 255.0
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    be = edge(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
    ce = edge(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY))
    ae = edge(ag)
    return {
        "deterministicMAE": float(np.abs(a-b).mean()),
        "candidateMAE": float(np.abs(a-c).mean()),
        "deterministicLaplacianMAE": float(np.abs(ae-be).mean()),
        "candidateLaplacianMAE": float(np.abs(ae-ce).mean()),
        "candidateChangedFraction1LSB": float(np.mean(np.any(baseline != candidate, axis=2))),
    }


def _save_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"could not save diagnostic PNG: {path}")


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo = args.repo_root.resolve()
    report_path, report = _latest_report(repo)
    checkpoint = Path(report["selectedCheckpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (repo / checkpoint).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"V2 selected checkpoint unavailable: {checkpoint}")
    # Local research-produced checkpoint only; do not load untrusted external .pt files.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA
        or not isinstance(payload.get("cropIds"), dict)
        or payload["cropIds"].get("training") != report.get("trainingCropIds")
        or payload["cropIds"].get("validation") != report.get("validationCropIds")
        or payload["cropIds"].get("final_test") != report.get("finalTestCropIds")
        or int(payload.get("update", -1)) != int(report["bestValidationUpdate"])
    ):
        raise RuntimeError("V2 selected checkpoint/report provenance mismatch")
    manifest_path = Path(str(payload["manifest"])).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"original V2 corpus manifest unavailable: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authority, family, crop = _choose_reviewed_source(manifest, report, args.authority_id)
    config = V16Config(
        dataset_manifest=str(manifest_path),
        train_hr_size=512, train_lr_size=128,
        validation_hr_size=512, validation_lr_size=128,
        minimum_heldout_samples=4, tiles_per_epoch=1,
        validation_tiles=1, clean_epochs=1, robust_epochs=1,
        selector_epochs=1, use_gradient_checkpointing=False,
    )
    contract = payload["architecture"]
    model = NSAMDRV17(
        config, encoder_channels=int(contract["encoderChannels"]),
        mid_channels=int(contract["decoderMidChannels"]),
        detail_channels=int(contract["decoderDetailChannels"]),
        decoder_blocks=int(contract["decoderBlocksPerStage"]),
    )
    if model.architecture_contract() != contract:
        raise RuntimeError("selected checkpoint architecture does not match V17.2")
    model.load_state_dict(payload["modelState"], strict=True)
    device = _device(args.device)
    model = model.to(device).eval()
    clock = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="nsamdr_v17_4096_") as scratch:
        authored, normal, material, provenance = _prepare_authored(
            repo, family, crop, Path(scratch),
        )
    # Reuse the same clean 4x degradation used for 512px V17.2 validation.
    lr_a, lr_n, lr_m = _degrade(
        authored.astype(np.float32)/255.0, normal, material,
        LR_SIZE, "clean", random.Random(17001),
    )
    del normal, material
    print(
        f"[v17-4096] selected={checkpoint} authority={authority} "
        f"input={LR_SIZE}x{LR_SIZE} output={HR_SIZE}x{HR_SIZE} "
        f"tileLR={args.tile_lr} haloLR={args.halo_lr} training=disabled",
        flush=True,
    )
    baseline, candidate = _tiled_4x(
        model, lr_a, lr_n, lr_m, device=device, precision=args.amp_precision,
        tile_lr=args.tile_lr, halo_lr=args.halo_lr,
    )
    if authored.shape != baseline.shape or authored.shape != candidate.shape:
        raise RuntimeError("4096 output dimensions do not match native authored reference")
    root = report_path.parent / "resolution_4096"
    root.mkdir(exist_ok=True)
    result_dir = root / f"visual_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    result_dir.mkdir(exist_ok=False)
    images = {
        "A_authored_4096": authored,
        "B_deterministic_4096": baseline,
        "C_v17_4096": candidate,
    }
    paths: dict[str, str] = {}
    for label, image in images.items():
        path = result_dir / f"{label}.png"
        _save_png(path, image)
        paths[label] = str(path.resolve())
    # Downsampled contact sheet is for navigation only; inspect originals at 100%.
    reduced = [
        _preview_panel(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0, label)
        for label, img in (("A AUTHORED 4096", authored),
                           ("B DETERMINISTIC 4X", baseline),
                           ("C V17.2 4096", candidate))
    ]
    sheet_path = result_dir / "ABC_overview_1024.png"
    if not cv2.imwrite(str(sheet_path), np.concatenate(reduced, axis=1)):
        raise RuntimeError(f"cannot write overview sheet: {sheet_path}")
    outcome = {
        "schema": DIAGNOSTIC_SCHEMA, "researchOnly": True,
        "productionPromotionAllowed": False, "qualificationGatesModified": False,
        "checkpoint": str(checkpoint), "v2Report": str(report_path),
        "checkpointUpdate": int(payload["update"]), "authorityId": authority,
        "reviewedDevelopmentCrop": f"{authority}_001",
        "finalTestCropsOpened": 0, "trainingUpdates": 0,
        "outputSize": [HR_SIZE, HR_SIZE], "inputSize": [LR_SIZE, LR_SIZE],
        "tileLR": args.tile_lr, "haloLR": args.halo_lr,
        "nativeReference": provenance, "images": paths,
        "overview": str(sheet_path.resolve()),
        "diagnosticMetrics": _metrics(authored, baseline, candidate),
        "elapsedSeconds": time.monotonic()-clock,
    }
    output = result_dir / "report.json"
    output.write_text(json.dumps(outcome, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(f"[v17-4096] complete report={output} overview={sheet_path}", flush=True)
    return 0, output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--authority-id", default="auto", help="auto or a previously reviewed V1 development authority")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--tile-lr", type=int, default=128)
    p.add_argument("--halo-lr", type=int, default=64)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
