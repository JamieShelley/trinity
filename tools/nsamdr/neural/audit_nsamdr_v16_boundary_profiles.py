#!/usr/bin/env python3
"""CPU-only boundary-profile oracle audit for NSAMDR V16.2.

The structure audit proves that B->A error is boundary-localised and LR physical
maps observe those boundaries. This audit tests the next variable: whether the
baseline cross-boundary physical profile plus correct geometry predicts the
held-out authored A-B profile.

Authored HR geometry is an oracle used only by this diagnostic. Production must
later predict geometry from LR evidence and qualify it independently.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v16.profiles import PHYSICAL_CHANNELS, PROFILE_OFFSETS
from v16.stage2_data import baseline_maps, load_crop, read_json, resolve_crop, resolve_run
from v16.structure import StructureTargetConfig, derive_structure_targets

SCHEMA = "NSAMDR_V16_BOUNDARY_PROFILE_ORACLE_AUDIT_V2"
SAMPLE_STRIDE = 5
MAX_TRAIN_PROFILES = 2500
MAX_HELD_PROFILES = 1000
K_NEIGHBOURS = 4
GEOMETRY_SCALE = 0.35


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _hwc(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    return np.ascontiguousarray(array.transpose(1, 2, 0))


def _physical(albedo: np.ndarray, normal: np.ndarray, material: np.ndarray) -> np.ndarray:
    return np.concatenate((albedo[..., :3], normal[..., :2], material[..., :3]), axis=2).astype(np.float32)


def _sample_profile(image: np.ndarray, y: float, x: float, nx: float, ny: float) -> np.ndarray:
    offsets = np.asarray(PROFILE_OFFSETS, dtype=np.float32)
    xs = (x + nx * offsets).reshape(1, -1).astype(np.float32)
    ys = (y + ny * offsets).reshape(1, -1).astype(np.float32)
    sampled = cv2.remap(
        image.astype(np.float32), xs, ys,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if sampled.ndim == 2:
        sampled = sampled[..., None]
    return np.asarray(sampled[0], dtype=np.float32)


def _curvature(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    components = (
        cv2.Sobel(tx, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(tx, cv2.CV_32F, 0, 1, ksize=3),
        cv2.Sobel(ty, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ty, cv2.CV_32F, 0, 1, ksize=3),
    )
    return np.sqrt(sum(component * component for component in components))


def _record_profiles(path: Path, split: str, cfg: StructureTargetConfig, limit: int) -> dict[str, Any]:
    albedo, normal, material, metadata = load_crop(path)
    _ta, _tn, _tm, ba, bn, bm = baseline_maps(albedo, normal, material)
    target = _physical(albedo, normal, material)
    baseline = _physical(_hwc(ba), _hwc(bn), _hwc(bm))
    structure = derive_structure_targets(albedo, normal, material, config=cfg)

    boundary = np.asarray(structure["boundaryMask"], dtype=bool)
    tx = np.asarray(structure["tangentX"], dtype=np.float32)
    ty = np.asarray(structure["tangentY"], dtype=np.float32)
    junction = np.asarray(structure["junctionConfidence"], dtype=np.float32)
    strength = np.asarray(structure["boundaryStrength"], dtype=np.float32)
    curve = _curvature(tx, ty)

    ys, xs = np.nonzero(boundary)
    if ys.size == 0:
        raise RuntimeError(f"No boundary pixels in {path}")
    sample_indices = np.arange(0, ys.size, SAMPLE_STRIDE, dtype=np.int64)
    if sample_indices.size > limit:
        sample_indices = sample_indices[
            np.linspace(0, sample_indices.size - 1, limit, dtype=np.int64)
        ]

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in sample_indices:
        y, x = int(ys[index]), int(xs[index])
        tangent_x, tangent_y = float(tx[y, x]), float(ty[y, x])
        norm = max((tangent_x * tangent_x + tangent_y * tangent_y) ** 0.5, 1.0e-6)
        tangent_x, tangent_y = tangent_x / norm, tangent_y / norm
        normal_x, normal_y = -tangent_y, tangent_x
        baseline_profile = _sample_profile(baseline, y, x, normal_x, normal_y)
        target_profile = _sample_profile(target, y, x, normal_x, normal_y)
        geometry = np.asarray(
            [
                abs(tangent_x), abs(tangent_y), float(strength[y, x]),
                float(junction[y, x]), min(float(curve[y, x]), 4.0) / 4.0,
            ],
            dtype=np.float32,
        )
        features.append(
            np.concatenate((baseline_profile.reshape(-1), geometry * GEOMETRY_SCALE))
        )
        targets.append((target_profile - baseline_profile).reshape(-1))

    family_id = str(metadata.get("familyId") or path.name.split("_", 1)[0])
    return {
        "path": str(path),
        "split": split,
        "familyId": family_id,
        "family": str(metadata.get("sourceAssetName") or metadata.get("familyName") or family_id),
        "features": np.stack(features).astype(np.float32),
        "targets": np.stack(targets).astype(np.float32),
    }


def _knn_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int) -> np.ndarray:
    train_norm = (train_x * train_x).sum(axis=1)
    output: list[np.ndarray] = []
    for start in range(0, len(query_x), 128):
        query = query_x[start : start + 128]
        distance = (
            (query * query).sum(axis=1, keepdims=True)
            + train_norm[None, :]
            - 2.0 * query @ train_x.T
        )
        count = min(k, train_x.shape[0])
        indices = np.argpartition(distance, kth=count - 1, axis=1)[:, :count]
        selected_distance = np.take_along_axis(distance, indices, axis=1)
        weights = 1.0 / np.maximum(selected_distance, 1.0e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        output.append((train_y[indices] * weights[..., None]).sum(axis=1))
    return np.concatenate(output, axis=0)


def _channel_indices(start: int, count: int) -> np.ndarray:
    indices: list[int] = []
    for sample in range(len(PROFILE_OFFSETS)):
        base = sample * PHYSICAL_CHANNELS + start
        indices.extend(range(base, base + count))
    return np.asarray(indices, dtype=np.int64)


ALBEDO = _channel_indices(0, 3)
NORMAL = _channel_indices(3, 2)
MATERIAL = _channel_indices(5, 3)


def _recovery(truth: np.ndarray, prediction: np.ndarray, indices: np.ndarray) -> float:
    baseline_error = float(np.abs(truth[:, indices]).mean())
    predicted_error = float(np.abs(truth[:, indices] - prediction[:, indices]).mean())
    return 1.0 - predicted_error / max(baseline_error, 1.0e-12)


def _edge_recovery(truth: np.ndarray, prediction: np.ndarray) -> float:
    shape = (truth.shape[0], len(PROFILE_OFFSETS), PHYSICAL_CHANNELS)
    target = truth.reshape(shape)[..., :3].mean(axis=2)
    predicted = prediction.reshape(shape)[..., :3].mean(axis=2)
    baseline_gradient_error = float(np.abs(np.diff(target, axis=1)).mean())
    predicted_gradient_error = float(np.abs(np.diff(target - predicted, axis=1)).mean())
    return 1.0 - predicted_gradient_error / max(baseline_gradient_error, 1.0e-12)


def _evaluate(train: list[dict[str, Any]], held: list[dict[str, Any]]) -> dict[str, float | int]:
    train_x = np.concatenate([item["features"] for item in train], axis=0)
    train_y = np.concatenate([item["targets"] for item in train], axis=0)
    held_x = np.concatenate([item["features"] for item in held], axis=0)
    truth = np.concatenate([item["targets"] for item in held], axis=0)
    prediction = _knn_predict(train_x, train_y, held_x, K_NEIGHBOURS)
    all_channels = np.arange(truth.shape[1], dtype=np.int64)
    return {
        "profileRecovery": _recovery(truth, prediction, all_channels),
        "albedoRecovery": _recovery(truth, prediction, ALBEDO),
        "normalRecovery": _recovery(truth, prediction, NORMAL),
        "materialRecovery": _recovery(truth, prediction, MATERIAL),
        "edgeProfileRecovery": _edge_recovery(truth, prediction),
        "trainProfileCount": int(train_x.shape[0]),
        "heldOutProfileCount": int(held_x.shape[0]),
    }


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    run_dir = resolve_run(repo_root, args.run_dir)
    report = read_json(run_dir / "report.json")
    cfg = StructureTargetConfig()

    records: list[dict[str, Any]] = []
    for split, field, budget in (
        ("train", "trainRecords", MAX_TRAIN_PROFILES),
        ("validation", "validationRecords", MAX_HELD_PROFILES),
    ):
        raw_records = report.get(field) or []
        per_record = max(64, budget // max(1, len(raw_records)))
        for raw in raw_records:
            path = resolve_crop(repo_root, str(raw), split)
            records.append(_record_profiles(path, split, cfg, per_record))

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "validation": []}
    )
    for record in records:
        grouped[record["familyId"]][record["split"]].append(record)

    families: dict[str, Any] = {}
    for family_id, splits in sorted(grouped.items()):
        if not splits["train"] or not splits["validation"]:
            continue
        families[family_id] = {
            "family": (splits["train"] + splits["validation"])[0]["family"],
            **_evaluate(splits["train"], splits["validation"]),
        }

    values = list(families.values())
    median_profile = float(statistics.median(v["profileRecovery"] for v in values)) if values else 0.0
    median_albedo = float(statistics.median(v["albedoRecovery"] for v in values)) if values else 0.0
    median_edge = float(statistics.median(v["edgeProfileRecovery"] for v in values)) if values else 0.0

    if median_profile >= 0.20 or (median_albedo >= 0.20 and median_edge >= 0.25):
        decision = "boundary-profile-formulation-supported-build-profile-model"
    elif median_profile <= 0.05 and median_edge <= 0.10:
        decision = "boundary-profile-prior-insufficient-revisit-structural-formulation"
    else:
        decision = "boundary-profile-signal-marginal-run-small-learned-profile-proof-only"

    payload = {
        "schema": SCHEMA,
        "sourceRun": str(run_dir),
        "cpuOnly": True,
        "datasetMutation": False,
        "trainedModelInference": False,
        "oracleUsesAuthoredHrGeometry": True,
        "profileOffsetsPixels": list(PROFILE_OFFSETS),
        "kNeighbours": K_NEIGHBOURS,
        "families": families,
        "aggregate": {
            "medianProfileRecovery": median_profile,
            "medianAlbedoRecovery": median_albedo,
            "medianEdgeProfileRecovery": median_edge,
        },
        "decision": decision,
    }
    json_path = run_dir / "boundary_profile_audit.json"
    summary_path = run_dir / "boundary_profile_audit_summary.txt"
    _atomic_json(json_path, payload)

    lines = [
        "NSAMDR V16 BOUNDARY PROFILE ORACLE AUDIT", "=" * 100,
        f"Run: {run_dir}",
        "CPU-only: yes; authored HR geometry is used only as an oracle diagnostic", "",
        "Family                     Profile  Albedo  Normal  Material  Edge-profile",
        "-" * 100,
    ]
    for family_id, item in families.items():
        lines.append(
            f"{family_id:<26} {item['profileRecovery']*100:>6.1f}%"
            f" {item['albedoRecovery']*100:>7.1f}% {item['normalRecovery']*100:>7.1f}%"
            f" {item['materialRecovery']*100:>9.1f}% {item['edgeProfileRecovery']*100:>12.1f}%"
        )
    lines += [
        "", "Aggregate", "-" * 100,
        f"Median profile recovery      : {median_profile*100:.1f}%",
        f"Median albedo recovery       : {median_albedo*100:.1f}%",
        f"Median edge-profile recovery : {median_edge*100:.1f}%",
        "", "DECISION", "-" * 100, decision, "",
        "Interpretation:",
        "- Inputs are deterministic B cross-boundary profiles plus oracle geometry.",
        "- Targets are aligned authored A-B profiles across albedo, normal and material.",
        "- This is evidence for/against an explicit BoundaryProfileNet, not an upper bound.",
    ]
    _atomic_text(summary_path, "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0, json_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit V16 boundary-profile recoverability")
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--run-dir", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
