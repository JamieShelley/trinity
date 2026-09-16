"""Shared data access for V16 Stage 2 CPU diagnostics.

This module owns run discovery, crop loading and deterministic baseline replay so
individual diagnostics stay focused on the question they test.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v14.baseline import Baseline4x
from v14.dataset import _degrade, _normalise_xy


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_completed_run(root: Path) -> Path | None:
    candidates: list[Path] = []
    for run_dir in root.glob("multiregion_*"):
        report_path = run_dir / "report.json"
        if not run_dir.is_dir() or not report_path.is_file():
            continue
        try:
            report = read_json(report_path)
        except (OSError, ValueError, TypeError):
            continue
        if int(report.get("completedSteps") or 0) > 0:
            candidates.append(run_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_run(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        value = requested if requested.is_absolute() else repo_root / requested
        value = value.resolve()
        if not (value / "report.json").is_file():
            raise RuntimeError(f"Stage 2 run has no report.json: {value}")
        return value
    root = repo_root / "artifacts/nsamdr/diagnostics/v16_mini"
    latest = latest_completed_run(root)
    if latest is None:
        raise RuntimeError(f"No completed V16 Stage 2 multiregion run found under {root}")
    return latest.resolve()


def resolve_crop(repo_root: Path, raw: str, split: str) -> Path:
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()
    basename = direct.name
    legacy = repo_root / "artifacts/nsamdr/training_v9_preview_raven/crops" / split / basename
    if legacy.is_file():
        return legacy.resolve()
    matches = list((repo_root / "artifacts/nsamdr").glob(f"**/crops/{split}/{basename}"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise RuntimeError(f"Selected Stage 2 crop is missing: {raw}")


def _normalise_loaded_normal(normal: np.ndarray) -> np.ndarray:
    xy = np.asarray(normal, dtype=np.uint8)[..., :2].astype(np.float32) / 127.5 - 1.0
    return _normalise_xy(xy)


def load_crop(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as bundle:
        albedo = np.asarray(bundle["albedo"], dtype=np.uint8).astype(np.float32) / 255.0
        normal = _normalise_loaded_normal(np.asarray(bundle["normal"], dtype=np.uint8))
        material = np.asarray(bundle["material"], dtype=np.uint8)[..., :3].astype(np.float32) / 255.0
        metadata: dict[str, Any] = {}
        if "metadata" in bundle.files:
            raw = np.asarray(bundle["metadata"]).reshape(-1)
            if raw.size:
                try:
                    metadata = json.loads(str(raw[0]))
                except (ValueError, TypeError):
                    metadata = {}
    return albedo, normal, material, metadata


def _batch(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).unsqueeze(0)


def baseline_maps(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = albedo.shape[:2]
    if height != width or height % 4:
        raise RuntimeError(f"Audit expects a square 4x Stage 2 crop; got {width}x{height}")
    lr_size = height // 4
    lr_a, lr_n, lr_m = _degrade(
        albedo, normal, material, lr_size, "clean", random.Random(14001)
    )
    baseline = Baseline4x(4).cpu().eval()
    with torch.no_grad():
        b_a, b_n, b_m = baseline(_batch(lr_a), _batch(lr_n), _batch(lr_m))
    return _batch(albedo), _batch(normal), _batch(material), b_a, b_n, b_m
