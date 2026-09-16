#!/usr/bin/env python3
"""CPU-only recoverability audit for a completed V16/V16.1 Stage 2 run.

This audit asks a different question from the family-difficulty audit: how much of the
authored residual is plausibly recoverable from the 4x LR evidence at all?

It measures, per authored family and physical map:
- residual spectral energy below / near / above the LR Nyquist limit,
- correlation between LR patch detail and authored residual magnitude,
- cross-validated within-family LR-patch recurrence,
- a non-neural kNN residual probe trained only on Stage 2 train domains and evaluated
  only on the pixel-disjoint held-out domain.

No trained NSAMDR checkpoint is loaded. CUDA is never used. The dataset is not mutated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_nsamdr_v16_stage2_family_difficulty import (
    _latest_completed_run,
    _load_crop,
    _read_json,
    _resolve_crop,
)
from v14.baseline import Baseline4x
from v14.dataset import _degrade


SCHEMA = "NSAMDR_V16_STAGE2_RECOVERABILITY_AUDIT_V1"
SCALE = 4
LR_NYQUIST_HR_CYCLES_PER_PIXEL = 0.5 / SCALE
NEAR_NYQUIST_LOW = LR_NYQUIST_HR_CYCLES_PER_PIXEL * 0.75
NEAR_NYQUIST_HIGH = LR_NYQUIST_HR_CYCLES_PER_PIXEL * 1.25
MAPS = ("albedo", "normal", "material")


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


def _resolve_run(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        run_dir = requested if requested.is_absolute() else repo_root / requested
        run_dir = run_dir.resolve()
        if not (run_dir / "report.json").is_file():
            raise RuntimeError(f"Stage 2 run has no report.json: {run_dir}")
        return run_dir
    root = repo_root / "artifacts/nsamdr/diagnostics/v16_mini"
    latest = _latest_completed_run(root)
    if latest is None:
        raise RuntimeError(f"No completed Stage 2 run found under {root}")
    return latest.resolve()


def _to_chw(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float()


def _exact_evidence(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    height, width = albedo.shape[:2]
    if height != width or height % SCALE:
        raise RuntimeError(f"Recoverability audit expects square 4x crops; got {width}x{height}")
    lr_size = height // SCALE
    lr_a, lr_n, lr_m = _degrade(
        albedo,
        normal,
        material,
        lr_size,
        "clean",
        random.Random(14001),
    )
    lr = {
        "albedo": _to_chw(lr_a),
        "normal": _to_chw(lr_n),
        "material": _to_chw(lr_m),
    }
    target = {
        "albedo": _to_chw(albedo),
        "normal": _to_chw(normal),
        "material": _to_chw(material),
    }
    baseline_model = Baseline4x(SCALE).cpu().eval()
    with torch.no_grad():
        b_a, b_n, b_m = baseline_model(
            lr["albedo"].unsqueeze(0),
            lr["normal"].unsqueeze(0),
            lr["material"].unsqueeze(0),
        )
    baseline = {
        "albedo": b_a[0].cpu(),
        "normal": b_n[0].cpu(),
        "material": b_m[0].cpu(),
    }
    return lr, target, baseline


def _spectral_partition(residual: torch.Tensor) -> dict[str, float]:
    """Partition residual FFT energy using HR cycles/pixel and the 4x LR Nyquist."""
    value = residual.float()
    if value.ndim != 3:
        raise ValueError("residual must be CHW")
    h, w = int(value.shape[-2]), int(value.shape[-1])
    spectrum = torch.fft.fft2(value, norm="ortho")
    energy = (spectrum.real.square() + spectrum.imag.square()).sum(dim=0)
    fy = torch.fft.fftfreq(h, d=1.0)
    fx = torch.fft.fftfreq(w, d=1.0)
    radial = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    below = radial < NEAR_NYQUIST_LOW
    near = (radial >= NEAR_NYQUIST_LOW) & (radial <= NEAR_NYQUIST_HIGH)
    above = radial > NEAR_NYQUIST_HIGH
    total = float(energy.sum().item())
    if total <= 1.0e-20:
        return {
            "totalEnergy": 0.0,
            "belowFraction": 0.0,
            "nearFraction": 0.0,
            "aboveFraction": 0.0,
            "belowOrNearFraction": 0.0,
        }

    def fraction(mask: torch.Tensor) -> float:
        return float(energy[mask].sum().item() / total)

    low = fraction(below)
    boundary = fraction(near)
    high = fraction(above)
    return {
        "totalEnergy": total / float(value.numel()),
        "belowFraction": low,
        "nearFraction": boundary,
        "aboveFraction": high,
        "belowOrNearFraction": low + boundary,
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 1.0e-20 else 0.0


def _lr_detail_score(patch: np.ndarray) -> float:
    # CHW patch. Local first-order LR variation is evidence available to any
    # deterministic SR method.
    dx = np.abs(patch[:, :, 1:] - patch[:, :, :-1]).mean() if patch.shape[2] > 1 else 0.0
    dy = np.abs(patch[:, 1:, :] - patch[:, :-1, :]).mean() if patch.shape[1] > 1 else 0.0
    return float(dx + dy)


def _patch_samples(
    lr: dict[str, torch.Tensor],
    residual: dict[str, torch.Tensor],
    *,
    stride: int,
    patch_radius: int,
) -> dict[str, Any]:
    evidence = torch.cat((lr["albedo"], lr["normal"], lr["material"]), dim=0).numpy()
    h, w = evidence.shape[-2:]
    features: list[np.ndarray] = []
    input_detail: list[float] = []
    residual_vectors: dict[str, list[np.ndarray]] = {name: [] for name in MAPS}
    residual_magnitude: dict[str, list[float]] = {name: [] for name in MAPS}

    for y in range(patch_radius, h - patch_radius, stride):
        for x in range(patch_radius, w - patch_radius, stride):
            patch = evidence[
                :,
                y - patch_radius : y + patch_radius + 1,
                x - patch_radius : x + patch_radius + 1,
            ]
            channel_mean = patch.mean(axis=(1, 2), keepdims=True)
            centered = patch - channel_mean
            feature = np.concatenate((centered.reshape(-1), channel_mean.reshape(-1)))
            features.append(feature.astype(np.float32, copy=False))
            input_detail.append(_lr_detail_score(patch))

            y0, y1 = y * SCALE, (y + 1) * SCALE
            x0, x1 = x * SCALE, (x + 1) * SCALE
            for name in MAPS:
                block = residual[name][:, y0:y1, x0:x1].numpy().astype(np.float32, copy=False)
                vector = block.reshape(-1)
                residual_vectors[name].append(vector)
                residual_magnitude[name].append(float(np.abs(vector).mean()))

    return {
        "features": np.stack(features, axis=0),
        "inputDetail": np.asarray(input_detail, dtype=np.float32),
        "residual": {name: np.stack(residual_vectors[name], axis=0) for name in MAPS},
        "residualMagnitude": {
            name: np.asarray(residual_magnitude[name], dtype=np.float32) for name in MAPS
        },
    }


def _concat_samples(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": np.concatenate([item["features"] for item in items], axis=0),
        "inputDetail": np.concatenate([item["inputDetail"] for item in items], axis=0),
        "residual": {
            name: np.concatenate([item["residual"][name] for item in items], axis=0)
            for name in MAPS
        },
        "residualMagnitude": {
            name: np.concatenate([item["residualMagnitude"][name] for item in items], axis=0)
            for name in MAPS
        },
    }


def _knn_indices(train_features: np.ndarray, query_features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-4)
    train = ((train_features - mean) / std).astype(np.float32)
    query = ((query_features - mean) / std).astype(np.float32)
    scale = math.sqrt(float(train.shape[1]))
    train_sq = np.sum(train * train, axis=1)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    k = max(1, min(int(k), len(train)))
    for start in range(0, len(query), 128):
        q = query[start : start + 128]
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        distances_sq = np.maximum(q_sq + train_sq[None, :] - 2.0 * (q @ train.T), 0.0)
        chosen = np.argpartition(distances_sq, kth=k - 1, axis=1)[:, :k]
        chosen_d2 = np.take_along_axis(distances_sq, chosen, axis=1)
        order = np.argsort(chosen_d2, axis=1)
        chosen = np.take_along_axis(chosen, order, axis=1)
        chosen_d2 = np.take_along_axis(chosen_d2, order, axis=1)
        all_indices.append(chosen)
        all_distances.append(np.sqrt(chosen_d2) / scale)
    return np.concatenate(all_indices, axis=0), np.concatenate(all_distances, axis=0)


def _knn_probe(
    train: dict[str, Any],
    heldout: dict[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    indices, distances = _knn_indices(train["features"], heldout["features"], k)
    weights = 1.0 / np.maximum(distances, 1.0e-4)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
    result: dict[str, Any] = {
        "trainPatchCount": int(len(train["features"])),
        "heldOutPatchCount": int(len(heldout["features"])),
        "k": int(indices.shape[1]),
        "nearestInputDistanceMean": float(distances[:, 0].mean()),
        "nearestInputDistanceMedian": float(np.median(distances[:, 0])),
        "maps": {},
    }
    for name in MAPS:
        neighbor_residuals = train["residual"][name][indices]
        prediction = (neighbor_residuals * weights[..., None]).sum(axis=1)
        actual = heldout["residual"][name]
        baseline_mae = float(np.abs(actual).mean())
        probe_mae = float(np.abs(prediction - actual).mean())
        recovery = 1.0 - probe_mae / max(baseline_mae, 1.0e-8)
        correlation = _pearson(prediction, actual)
        input_energy_corr = _pearson(
            heldout["inputDetail"], heldout["residualMagnitude"][name]
        )
        neighbor_mean = neighbor_residuals.mean(axis=1, keepdims=True)
        neighbor_dispersion = float(np.abs(neighbor_residuals - neighbor_mean).mean())
        target_scale = float(np.abs(neighbor_residuals).mean())
        normalized_dispersion = neighbor_dispersion / max(target_scale, 1.0e-8)
        result["maps"][name] = {
            "baselineMae": baseline_mae,
            "probeMae": probe_mae,
            "recovery": float(recovery),
            "predictionResidualCorrelation": float(correlation),
            "inputDetailResidualMagnitudeCorrelation": float(input_energy_corr),
            "neighborResidualDispersionRelative": float(normalized_dispersion),
        }
    return result


def _family_name_map(report: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    per_family = candidate.get("perFamily") if isinstance(candidate, dict) else {}
    if isinstance(per_family, dict):
        for family_id, metrics in per_family.items():
            if isinstance(metrics, dict):
                output[str(family_id)] = str(metrics.get("sourceAssetName") or family_id)
    return output


def _selected_global_map(report: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    per_family = candidate.get("perFamily") if isinstance(candidate, dict) else {}
    if isinstance(per_family, dict):
        for family_id, metrics in per_family.items():
            if isinstance(metrics, dict) and metrics.get("medianGlobalRecovery") is not None:
                result[str(family_id)] = float(metrics["medianGlobalRecovery"])
    return result


def _load_selected_record(
    repo_root: Path,
    raw_path: str,
    split: str,
    fallback_box: list[int] | None,
    family_names: dict[str, str],
    *,
    stride: int,
    patch_radius: int,
) -> dict[str, Any]:
    path = _resolve_crop(repo_root, raw_path, split)
    albedo, normal, material, metadata = _load_crop(path)
    family_id = str(metadata.get("familyId") or path.name.split("_", 1)[0])
    lr, target, baseline = _exact_evidence(albedo, normal, material)
    residual = {name: target[name] - baseline[name] for name in MAPS}
    spectral = {name: _spectral_partition(residual[name]) for name in MAPS}
    samples = _patch_samples(lr, residual, stride=stride, patch_radius=patch_radius)
    source_box = metadata.get("pixelBox") or fallback_box
    return {
        "path": str(path),
        "split": split,
        "familyId": family_id,
        "family": family_names.get(family_id, family_id),
        "sourceBox": source_box,
        "spectral": spectral,
        "samples": samples,
    }


def _weighted_spectral(records: list[dict[str, Any]], map_name: str) -> dict[str, float]:
    if not records:
        return {}
    energies = np.asarray(
        [float(record["spectral"][map_name]["totalEnergy"]) for record in records],
        dtype=np.float64,
    )
    weights = energies / max(float(energies.sum()), 1.0e-20)
    keys = ("belowFraction", "nearFraction", "aboveFraction", "belowOrNearFraction")
    result = {
        key: float(
            sum(
                float(weight) * float(record["spectral"][map_name][key])
                for weight, record in zip(weights, records)
            )
        )
        for key in keys
    }
    result["meanResidualEnergy"] = float(energies.mean())
    return result


def _family_interpretation(albedo_spectral: dict[str, float], albedo_probe: dict[str, float]) -> str:
    recover = float(albedo_probe.get("recovery") or 0.0)
    corr = float(albedo_probe.get("predictionResidualCorrelation") or 0.0)
    high = float(albedo_spectral.get("aboveFraction") or 0.0)
    if recover >= 0.20 or corr >= 0.35:
        return "recoverable-structure-detected"
    if recover <= 0.05 and corr <= 0.15 and high >= 0.60:
        return "weak-lr-evidence-high-frequency-dominated"
    if recover <= 0.05 and corr <= 0.15:
        return "weak-cross-validated-predictability"
    return "inconclusive-moderate-predictability"


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "NSAMDR V16 STAGE 2 RECOVERABILITY AUDIT",
        "=" * 88,
        f"Run: {payload['runDir']}",
        "CPU-only: yes; trained model inference: no; dataset mutation: no",
        f"LR Nyquist on HR grid: {payload['frequencyBands']['lrNyquistHrCyclesPerPixel']:.5f} cycles/pixel",
        "",
        "Family                     Albedo residual spectrum         kNN probe              V16 held  Interpretation",
        "                           below   near   above      recovery   corr   input-corr     global",
        "-" * 112,
    ]
    for family in payload["families"].values():
        spectral = family["heldOutSpectral"]["albedo"]
        probe = family["patchProbe"]["maps"]["albedo"]
        lines.append(
            f"{family['family'][:26]:26s} "
            f"{spectral['belowFraction']*100:6.1f}% {spectral['nearFraction']*100:6.1f}% {spectral['aboveFraction']*100:6.1f}%   "
            f"{probe['recovery']*100:+8.1f}% {probe['predictionResidualCorrelation']:+6.3f} "
            f"{probe['inputDetailResidualMagnitudeCorrelation']:+9.3f}   "
            f"{family['selectedHeldOutGlobalRecovery']*100:+7.1f}%  {family['interpretation']}"
        )
    lines.extend(
        [
            "",
            "Decision",
            "-" * 88,
            f"{payload['assessment']['suggestedNextExperiment']}",
            "",
            "Method notes:",
            "- Spectral bands are disjoint: below < 0.75x LR Nyquist; near = 0.75x..1.25x; above > 1.25x.",
            "- The kNN probe uses only within-family Stage 2 train-domain LR patches to predict residual 4x4 blocks in the held-out domain.",
            "- Probe recovery is relative to deterministic baseline B on sampled held-out blocks; it is evidence, not a theoretical upper bound.",
            "- High above-Nyquist energy does not prove impossibility: repeated texture priors can still make some high-frequency residual statistically predictable.",
        ]
    )
    return "\n".join(lines) + "\n"


def _assessment(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        families.values(),
        key=lambda family: float(family.get("selectedHeldOutGlobalRecovery") or 0.0),
        reverse=True,
    )
    weak = ordered[1:] if len(ordered) > 1 else ordered
    weak_recovery = statistics.median(
        float(family["patchProbe"]["maps"]["albedo"]["recovery"]) for family in weak
    )
    weak_corr = statistics.median(
        float(family["patchProbe"]["maps"]["albedo"]["predictionResidualCorrelation"])
        for family in weak
    )
    weak_above = statistics.median(
        float(family["heldOutSpectral"]["albedo"]["aboveFraction"]) for family in weak
    )
    recoverable_count = sum(
        family["interpretation"] == "recoverable-structure-detected" for family in weak
    )

    if recoverable_count >= max(1, math.ceil(len(weak) / 2)):
        next_step = "test-gradient-conflict-mitigation-short-run-pcgrad-or-cagrad"
        conclusion = "weak-families-contain-cross-validated-recoverable-residual-structure"
    elif weak_recovery <= 0.05 and weak_corr <= 0.15:
        next_step = "reformulate-input-or-data-evidence-before-more-optimizer-experiments"
        conclusion = "weak-families-show-little-cross-validated-lr-to-residual-predictability"
    else:
        next_step = "run-targeted-short-probe-before-long-training"
        conclusion = "recoverability-evidence-is-mixed"
    return {
        "weakFamilyMedianKnnAlbedoRecovery": float(weak_recovery),
        "weakFamilyMedianKnnAlbedoCorrelation": float(weak_corr),
        "weakFamilyMedianAboveNyquistAlbedoEnergyFraction": float(weak_above),
        "weakFamilyRecoverableCount": int(recoverable_count),
        "conclusion": conclusion,
        "suggestedNextExperiment": next_step,
    }


def run(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    repo_root = args.repo_root.resolve()
    run_dir = _resolve_run(repo_root, args.run_dir)
    report = _read_json(run_dir / "report.json")
    if not isinstance(report, dict):
        raise RuntimeError(f"Invalid Stage 2 report: {run_dir / 'report.json'}")

    family_names = _family_name_map(report)
    selected_global = _selected_global_map(report)
    train_paths = [str(value) for value in report.get("trainRecords") or []]
    held_paths = [str(value) for value in report.get("validationRecords") or []]
    train_boxes = list(report.get("trainSourceBoxes") or [])
    held_boxes = list(report.get("validationSourceBoxes") or [])
    if not train_paths or not held_paths:
        raise RuntimeError("Stage 2 report does not contain selected train/validation records")

    print(f"[recoverability] run: {run_dir}", flush=True)
    print("[recoverability] CPU-only; no checkpoint/model inference", flush=True)

    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for index, raw in enumerate(train_paths):
        print(f"[recoverability] train {index + 1}/{len(train_paths)}: {Path(raw).name}", flush=True)
        records["train"].append(
            _load_selected_record(
                repo_root,
                raw,
                "train",
                train_boxes[index] if index < len(train_boxes) else None,
                family_names,
                stride=args.patch_stride,
                patch_radius=args.patch_radius,
            )
        )
    for index, raw in enumerate(held_paths):
        print(f"[recoverability] held-out {index + 1}/{len(held_paths)}: {Path(raw).name}", flush=True)
        records["validation"].append(
            _load_selected_record(
                repo_root,
                raw,
                "validation",
                held_boxes[index] if index < len(held_boxes) else None,
                family_names,
                stride=args.patch_stride,
                patch_radius=args.patch_radius,
            )
        )

    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "validation": []}
    )
    for split in ("train", "validation"):
        for record in records[split]:
            by_family[record["familyId"]][split].append(record)

    families: dict[str, dict[str, Any]] = {}
    for family_id, grouped in sorted(
        by_family.items(), key=lambda item: family_names.get(item[0], item[0])
    ):
        if not grouped["train"] or not grouped["validation"]:
            continue
        train_samples = _concat_samples([record["samples"] for record in grouped["train"]])
        held_samples = _concat_samples([record["samples"] for record in grouped["validation"]])
        probe = _knn_probe(train_samples, held_samples, k=args.knn_k)
        held_spectral = {
            name: _weighted_spectral(grouped["validation"], name) for name in MAPS
        }
        train_spectral = {name: _weighted_spectral(grouped["train"], name) for name in MAPS}
        interpretation = _family_interpretation(
            held_spectral["albedo"], probe["maps"]["albedo"]
        )
        family_name = family_names.get(family_id, grouped["validation"][0]["family"])
        families[family_id] = {
            "familyId": family_id,
            "family": family_name,
            "selectedHeldOutGlobalRecovery": float(selected_global.get(family_id, 0.0)),
            "trainRecordCount": len(grouped["train"]),
            "heldOutRecordCount": len(grouped["validation"]),
            "trainSpectral": train_spectral,
            "heldOutSpectral": held_spectral,
            "patchProbe": probe,
            "interpretation": interpretation,
        }
        a = held_spectral["albedo"]
        p = probe["maps"]["albedo"]
        print(
            f"  {family_name:24s} spectrum(low/near/high)="
            f"{a['belowFraction']*100:.1f}/{a['nearFraction']*100:.1f}/{a['aboveFraction']*100:.1f}% "
            f"kNN recovery={p['recovery']*100:+.1f}% "
            f"corr={p['predictionResidualCorrelation']:+.3f} => {interpretation}",
            flush=True,
        )

    assessment = _assessment(families)
    payload = {
        "schema": SCHEMA,
        "runDir": str(run_dir),
        "sourceReport": str((run_dir / "report.json").resolve()),
        "datasetFingerprint": report.get("datasetFingerprint"),
        "completedSteps": int(report.get("completedSteps") or 0),
        "selectedStep": int(report.get("selectedStep") or 0),
        "cpuOnly": True,
        "trainedModelInference": False,
        "datasetMutation": False,
        "frequencyBands": {
            "scale": SCALE,
            "lrNyquistHrCyclesPerPixel": LR_NYQUIST_HR_CYCLES_PER_PIXEL,
            "belowUpperExclusive": NEAR_NYQUIST_LOW,
            "nearLowerInclusive": NEAR_NYQUIST_LOW,
            "nearUpperInclusive": NEAR_NYQUIST_HIGH,
            "aboveLowerExclusive": NEAR_NYQUIST_HIGH,
        },
        "patchProbeConfig": {
            "strideLrPixels": int(args.patch_stride),
            "patchRadiusLrPixels": int(args.patch_radius),
            "patchWidthLrPixels": int(args.patch_radius * 2 + 1),
            "residualBlockHrPixels": SCALE,
            "k": int(args.knn_k),
            "scope": "within-family-train-to-heldout-only",
        },
        "families": families,
        "assessment": assessment,
    }
    json_path = run_dir / "recoverability_audit.json"
    text_path = run_dir / "recoverability_audit_summary.txt"
    _atomic_json(json_path, payload)
    _atomic_text(text_path, _summary(payload))
    print(f"[recoverability] JSON   : {json_path}", flush=True)
    print(f"[recoverability] summary: {text_path}", flush=True)
    print(f"[recoverability] decision: {assessment['suggestedNextExperiment']}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPU-only NSAMDR V16 Stage 2 recoverability audit")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--patch-stride", type=int, default=8)
    p.add_argument("--patch-radius", type=int, default=2)
    p.add_argument("--knn-k", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.patch_stride < 1:
        raise SystemExit("--patch-stride must be >= 1")
    if args.patch_radius < 0:
        raise SystemExit("--patch-radius must be >= 0")
    if args.knn_k < 1:
        raise SystemExit("--knn-k must be >= 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
