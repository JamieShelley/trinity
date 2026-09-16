#!/usr/bin/env python3
"""CPU-only Stage 2 recovery evidence sweep.

This diagnostic follows the V16/V16.1 recoverability audit.  It does not train a
model and never uses CUDA.  It asks which evidence change is most likely to make
the README's fixed 4x reconstruction target learnable without relaxing the
qualification gates or permitting unsupported texture invention.

Three controlled sweeps are run against the same pixel-disjoint Stage 2 records:

1. Scale sweep: 2x / ~3x / 4x evidence from the same authored HR crops.
2. Context sweep: 5x5 / 9x9 / 17x17 / 33x33 LR retrieval context at 4x.
3. Prior sweep: same-family selected train records vs all selected families vs the
   wider prepared Stage 2 train-crop corpus currently available on disk.

The non-neural kNN residual probe is diagnostic evidence only.  It is not a
production algorithm and its recovery is not a theoretical upper bound.
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
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_nsamdr_v16_stage2_family_difficulty import (
    _load_crop,
    _read_json,
    _resolve_crop,
)
from audit_nsamdr_v16_stage2_recoverability import (
    MAPS,
    _concat_samples,
    _family_name_map,
    _knn_probe,
    _resolve_run,
    _selected_global_map,
)
from v14.baseline import normalize_xy
from v14.dataset import _degrade


SCHEMA = "NSAMDR_V16_STAGE2_RECOVERY_EVIDENCE_SWEEP_V1"
SCALES = (2.0, 3.0, 4.0)
CONTEXT_WIDTHS = (5, 9, 17, 33)
DESCRIPTOR_GRID = 5
DEFAULT_HR_SAMPLE_STRIDE = 32
DEFAULT_K = 4


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


def _to_chw(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float()


def _evidence_for_scale(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    requested_scale: float,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    float,
]:
    height, width = albedo.shape[:2]
    if height != width:
        raise RuntimeError(f"Recovery sweep expects square Stage 2 crops; got {width}x{height}")
    lr_size = max(8, int(round(float(height) / float(requested_scale))))
    effective_scale = float(height) / float(lr_size)
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
    target_size = (height, width)
    with torch.no_grad():
        b_a = F.interpolate(
            lr["albedo"].unsqueeze(0),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)[0]
        b_n = normalize_xy(
            F.interpolate(
                lr["normal"].unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        )[0]
        b_m = F.interpolate(
            lr["material"].unsqueeze(0),
            size=target_size,
            mode="nearest",
        ).clamp(0.0, 1.0)[0]
    baseline = {"albedo": b_a, "normal": b_n, "material": b_m}
    return lr, target, baseline, effective_scale


def _spectral_partition(residual: torch.Tensor, effective_scale: float) -> dict[str, float]:
    value = residual.float()
    h, w = int(value.shape[-2]), int(value.shape[-1])
    spectrum = torch.fft.fft2(value, norm="ortho")
    energy = (spectrum.real.square() + spectrum.imag.square()).sum(dim=0)
    fy = torch.fft.fftfreq(h, d=1.0)
    fx = torch.fft.fftfreq(w, d=1.0)
    radial = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    nyquist = 0.5 / float(effective_scale)
    low_limit = nyquist * 0.75
    high_limit = nyquist * 1.25
    masks = {
        "below": radial < low_limit,
        "near": (radial >= low_limit) & (radial <= high_limit),
        "above": radial > high_limit,
    }
    total = float(energy.sum().item())
    if total <= 1.0e-20:
        return {
            "lrNyquistHrCyclesPerPixel": nyquist,
            "totalEnergy": 0.0,
            "belowFraction": 0.0,
            "nearFraction": 0.0,
            "aboveFraction": 0.0,
        }

    def fraction(name: str) -> float:
        return float(energy[masks[name]].sum().item() / total)

    return {
        "lrNyquistHrCyclesPerPixel": nyquist,
        "totalEnergy": total / float(value.numel()),
        "belowFraction": fraction("below"),
        "nearFraction": fraction("near"),
        "aboveFraction": fraction("above"),
    }


def _pooled_descriptor(patch: np.ndarray, grid: int = DESCRIPTOR_GRID) -> np.ndarray:
    channels, height, width = patch.shape
    y_edges = np.linspace(0, height, grid + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, grid + 1, dtype=np.int32)
    pooled: list[np.ndarray] = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
            cell = patch[:, y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
            pooled.append(cell.mean(axis=(1, 2)))
    grid_values = np.stack(pooled, axis=1).reshape(channels, grid, grid)
    means = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    centered = grid_values - means[:, None, None]
    return np.concatenate((centered.reshape(-1), means, stds)).astype(np.float32, copy=False)


def _lr_detail_score(patch: np.ndarray) -> float:
    dx = np.abs(patch[:, :, 1:] - patch[:, :, :-1]).mean() if patch.shape[2] > 1 else 0.0
    dy = np.abs(patch[:, 1:, :] - patch[:, :-1, :]).mean() if patch.shape[1] > 1 else 0.0
    return float(dx + dy)


def _fixed_residual_block(
    residual: torch.Tensor,
    x: int,
    y: int,
    lr_width: int,
    lr_height: int,
) -> np.ndarray:
    hr_height, hr_width = int(residual.shape[-2]), int(residual.shape[-1])
    scale_x = float(hr_width) / float(lr_width)
    scale_y = float(hr_height) / float(lr_height)
    block_w = max(1, int(round(scale_x)))
    block_h = max(1, int(round(scale_y)))
    cx = int(round((float(x) + 0.5) * scale_x - 0.5))
    cy = int(round((float(y) + 0.5) * scale_y - 0.5))
    x0 = min(max(0, cx - block_w // 2), max(0, hr_width - block_w))
    y0 = min(max(0, cy - block_h // 2), max(0, hr_height - block_h))
    block = residual[:, y0 : y0 + block_h, x0 : x0 + block_w]
    return block.numpy().astype(np.float32, copy=False).reshape(-1)


def _patch_samples(
    lr: dict[str, torch.Tensor],
    residual: dict[str, torch.Tensor],
    *,
    context_width: int,
    hr_stride: int,
) -> dict[str, Any]:
    if context_width < 1 or context_width % 2 == 0:
        raise ValueError("context_width must be a positive odd integer")
    radius = context_width // 2
    evidence = torch.cat((lr["albedo"], lr["normal"], lr["material"]), dim=0).numpy()
    lr_h, lr_w = evidence.shape[-2:]
    effective_scale = float(residual["albedo"].shape[-1]) / float(lr_w)
    stride = max(1, int(round(float(hr_stride) / effective_scale)))

    features: list[np.ndarray] = []
    detail: list[float] = []
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in MAPS}
    magnitudes: dict[str, list[float]] = {name: [] for name in MAPS}

    for y in range(radius, lr_h - radius, stride):
        for x in range(radius, lr_w - radius, stride):
            patch = evidence[:, y - radius : y + radius + 1, x - radius : x + radius + 1]
            features.append(_pooled_descriptor(patch))
            detail.append(_lr_detail_score(patch))
            for name in MAPS:
                vector = _fixed_residual_block(residual[name], x, y, lr_w, lr_h)
                vectors[name].append(vector)
                magnitudes[name].append(float(np.abs(vector).mean()))

    if not features:
        raise RuntimeError(
            f"No retrieval patches for context {context_width} on LR grid {lr_w}x{lr_h}"
        )
    return {
        "features": np.stack(features, axis=0),
        "inputDetail": np.asarray(detail, dtype=np.float32),
        "residual": {name: np.stack(vectors[name], axis=0) for name in MAPS},
        "residualMagnitude": {
            name: np.asarray(magnitudes[name], dtype=np.float32) for name in MAPS
        },
    }


def _weighted_spectral(records: list[dict[str, Any]], map_name: str) -> dict[str, float]:
    if not records:
        return {}
    energies = np.asarray(
        [float(record["spectral"][map_name]["totalEnergy"]) for record in records],
        dtype=np.float64,
    )
    if float(energies.sum()) <= 1.0e-20:
        weights = np.full(len(records), 1.0 / max(1, len(records)), dtype=np.float64)
    else:
        weights = energies / float(energies.sum())
    result: dict[str, float] = {}
    for key in ("belowFraction", "nearFraction", "aboveFraction"):
        result[key] = float(
            sum(float(weight) * float(record["spectral"][map_name][key]) for weight, record in zip(weights, records))
        )
    result["meanResidualEnergy"] = float(energies.mean())
    result["lrNyquistHrCyclesPerPixel"] = float(
        statistics.mean(float(record["spectral"][map_name]["lrNyquistHrCyclesPerPixel"]) for record in records)
    )
    return result


def _record_at_scale(
    record: dict[str, Any],
    requested_scale: float,
    *,
    context_width: int,
    hr_stride: int,
) -> dict[str, Any]:
    albedo, normal, material = record["data"]
    lr, target, baseline, effective_scale = _evidence_for_scale(
        albedo, normal, material, requested_scale
    )
    residual = {name: target[name] - baseline[name] for name in MAPS}
    return {
        "familyId": record["familyId"],
        "family": record["family"],
        "path": record["path"],
        "effectiveScale": effective_scale,
        "spectral": {name: _spectral_partition(residual[name], effective_scale) for name in MAPS},
        "samples": _patch_samples(
            lr,
            residual,
            context_width=context_width,
            hr_stride=hr_stride,
        ),
        "evidence": (lr, residual),
    }


def _load_report_records(
    repo_root: Path,
    report: dict[str, Any],
    family_names: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, field in (("train", "trainRecords"), ("validation", "validationRecords")):
        for raw in report.get(field) or []:
            path = _resolve_crop(repo_root, str(raw), split)
            albedo, normal, material, metadata = _load_crop(path)
            family_id = str(metadata.get("familyId") or path.name.split("_", 1)[0])
            output[split].append(
                {
                    "path": str(path),
                    "familyId": family_id,
                    "family": family_names.get(family_id, family_id),
                    "data": (albedo, normal, material),
                }
            )
    return output


def _group(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result[str(record["familyId"])].append(record)
    return dict(result)


def _probe_by_family(
    train_records: list[dict[str, Any]],
    held_records: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
    train_by_family = _group(train_records)
    held_by_family = _group(held_records)
    output: dict[str, Any] = {}
    for family_id, held in held_by_family.items():
        train = train_by_family.get(family_id) or []
        if not train:
            continue
        probe = _knn_probe(
            _concat_samples([record["samples"] for record in train]),
            _concat_samples([record["samples"] for record in held]),
            k=k,
        )
        output[family_id] = {
            "family": held[0]["family"],
            "probe": probe,
            "heldOutSpectral": {
                name: _weighted_spectral(held, name) for name in MAPS
            },
        }
    return output


def _median_for_families(
    result: dict[str, Any], family_ids: list[str], path: tuple[str, ...]
) -> float:
    values: list[float] = []
    for family_id in family_ids:
        value: Any = result.get(family_id)
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            values.append(float(value))
    return float(statistics.median(values)) if values else float("nan")


def _selected_weak_families(selected_global: dict[str, float]) -> tuple[str | None, list[str]]:
    if not selected_global:
        return None, []
    strongest = max(selected_global, key=selected_global.get)
    weak = [family_id for family_id in selected_global if family_id != strongest]
    return strongest, weak


def _prepared_corpus_paths(repo_root: Path, selected_train: list[dict[str, Any]], limit: int) -> list[Path]:
    roots: list[Path] = []
    canonical = repo_root / "artifacts/nsamdr/training_v9_preview_raven/crops/train"
    if canonical.is_dir():
        roots.append(canonical)
    for record in selected_train:
        parent = Path(str(record["path"])).parent
        if parent.is_dir() and parent not in roots:
            roots.append(parent)
    paths: dict[str, Path] = {}
    for root in roots:
        for path in sorted(root.glob("*.npz")):
            try:
                key = str(path.resolve()).lower()
            except OSError:
                key = str(path).lower()
            paths[key] = path
    for record in selected_train:
        path = Path(str(record["path"]))
        paths[str(path.resolve()).lower()] = path
    ordered = sorted(paths.values(), key=lambda path: path.name)
    return ordered[: max(1, int(limit))]


def _load_corpus_samples(
    paths: list[Path],
    *,
    context_width: int,
    hr_stride: int,
) -> tuple[dict[str, Any], list[str]]:
    samples: list[dict[str, Any]] = []
    used: list[str] = []
    for index, path in enumerate(paths):
        print(f"[recovery-sweep] corpus {index + 1}/{len(paths)}: {path.name}", flush=True)
        albedo, normal, material, _metadata = _load_crop(path)
        lr, target, baseline, _effective = _evidence_for_scale(albedo, normal, material, 4.0)
        residual = {name: target[name] - baseline[name] for name in MAPS}
        samples.append(
            _patch_samples(
                lr,
                residual,
                context_width=context_width,
                hr_stride=hr_stride,
            )
        )
        used.append(str(path.resolve()))
    return _concat_samples(samples), used


def _prior_probe(
    train_records: list[dict[str, Any]],
    held_records: list[dict[str, Any]],
    *,
    corpus_samples: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    train_by_family = _group(train_records)
    held_by_family = _group(held_records)
    all_selected = _concat_samples([record["samples"] for record in train_records])
    output: dict[str, Any] = {}
    for family_id, held in held_by_family.items():
        held_samples = _concat_samples([record["samples"] for record in held])
        family_train = train_by_family.get(family_id) or []
        if not family_train:
            continue
        output[family_id] = {
            "family": held[0]["family"],
            "withinFamilySelected": _knn_probe(
                _concat_samples([record["samples"] for record in family_train]),
                held_samples,
                k=k,
            ),
            "selectedAllFamilies": _knn_probe(all_selected, held_samples, k=k),
            "preparedCorpus": _knn_probe(corpus_samples, held_samples, k=k),
        }
    return output


def _assessment(
    selected_global: dict[str, float],
    scale_sweep: dict[str, dict[str, Any]],
    context_sweep: dict[str, dict[str, Any]],
    prior_sweep: dict[str, Any],
) -> dict[str, Any]:
    strongest, weak = _selected_weak_families(selected_global)
    if not weak:
        weak = list(selected_global)

    scale_medians: dict[str, float] = {}
    for scale_name, result in scale_sweep.items():
        scale_medians[scale_name] = _median_for_families(
            result, weak, ("probe", "maps", "albedo", "recovery")
        )
    context_medians: dict[str, float] = {}
    for width, result in context_sweep.items():
        context_medians[width] = _median_for_families(
            result, weak, ("probe", "maps", "albedo", "recovery")
        )

    def prior_median(scope: str) -> float:
        values = [
            float(prior_sweep[family_id][scope]["maps"]["albedo"]["recovery"])
            for family_id in weak
            if family_id in prior_sweep
        ]
        return float(statistics.median(values)) if values else float("nan")

    prior_medians = {
        scope: prior_median(scope)
        for scope in ("withinFamilySelected", "selectedAllFamilies", "preparedCorpus")
    }

    scale4 = scale_medians.get("4x", float("nan"))
    scale2 = scale_medians.get("2x", float("nan"))
    scale_gain = scale2 - scale4 if math.isfinite(scale2) and math.isfinite(scale4) else 0.0
    base_context = context_medians.get("5x5", float("nan"))
    best_context_name = max(
        context_medians,
        key=lambda key: context_medians[key] if math.isfinite(context_medians[key]) else -1.0e9,
    )
    best_context = context_medians[best_context_name]
    context_gain = (
        best_context - base_context
        if math.isfinite(best_context) and math.isfinite(base_context)
        else 0.0
    )
    within = prior_medians["withinFamilySelected"]
    best_prior_name = max(
        prior_medians,
        key=lambda key: prior_medians[key] if math.isfinite(prior_medians[key]) else -1.0e9,
    )
    best_prior = prior_medians[best_prior_name]
    prior_gain = best_prior - within if math.isfinite(best_prior) and math.isfinite(within) else 0.0

    signals: list[str] = []
    recommendations: list[str] = []
    if scale_gain >= 0.10 or (math.isfinite(scale2) and scale2 >= 0.15):
        signals.append("lower-scale-evidence-materially-improves-recoverability")
        recommendations.append("test-progressive-2x-to-4x-or-multiscale-evidence-path")
    if context_gain >= 0.10 and math.isfinite(best_context) and best_context >= 0.05:
        signals.append("larger-context-materially-improves-recoverability")
        recommendations.append("increase-effective-spatial-context-before-changing-qualification-gates")
    if prior_gain >= 0.10 and math.isfinite(best_prior) and best_prior >= 0.05:
        signals.append("broader-authored-prior-materially-improves-recoverability")
        recommendations.append("broaden-genuine-authored-same-task-pretraining-corpus")
    if not signals:
        signals.append("current-lr-evidence-remains-underconstrained")
        recommendations.append("revisit-input-evidence-or-reconstruction-formulation-before-more-gpu-training")

    return {
        "strongestCurrentFamilyId": strongest,
        "weakFamilyIds": weak,
        "weakFamilyMedianAlbedoRecoveryByScale": scale_medians,
        "weakFamilyMedianAlbedoRecoveryByContext": context_medians,
        "weakFamilyMedianAlbedoRecoveryByPrior": prior_medians,
        "scale2Vs4Gain": float(scale_gain),
        "bestContext": best_context_name,
        "bestContextGainVs5x5": float(context_gain),
        "bestPrior": best_prior_name,
        "bestPriorGainVsWithinFamily": float(prior_gain),
        "signals": signals,
        "recommendedDevelopmentDirections": recommendations,
        "qualificationPolicy": "keep-existing-v16-gates-unchanged",
        "productionGoal": "retain-readme-4x-physical-map-reconstruction-without-unsupported-invention",
    }


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "NSAMDR V16 STAGE 2 RECOVERY EVIDENCE SWEEP",
        "=" * 104,
        f"Run: {payload['runDir']}",
        "CPU-only: yes; trained model inference: no; dataset mutation: no",
        "Goal contract: preserve README 4x physical-map target and qualification gates.",
        "",
        "SCALE SWEEP - held-out albedo",
        "-" * 104,
        "Family                     2x recovery/high     ~3x recovery/high     4x recovery/high      V16 held",
    ]
    family_ids = list(payload["families"])
    for family_id in family_ids:
        family = payload["families"][family_id]
        pieces: list[str] = []
        for scale_name in ("2x", "3x", "4x"):
            entry = payload["scaleSweep"][scale_name][family_id]
            recovery = float(entry["probe"]["maps"]["albedo"]["recovery"])
            high = float(entry["heldOutSpectral"]["albedo"]["aboveFraction"])
            pieces.append(f"{recovery*100:+6.1f}%/{high*100:5.1f}%")
        lines.append(
            f"{family['family'][:26]:26s} {pieces[0]:>17s} {pieces[1]:>19s} {pieces[2]:>19s} "
            f"{family['selectedHeldOutGlobalRecovery']*100:+8.1f}%"
        )

    lines.extend(
        [
            "",
            "CONTEXT SWEEP - 4x within-family held-out albedo kNN recovery",
            "-" * 104,
            "Family                       5x5       9x9      17x17      33x33",
        ]
    )
    for family_id in family_ids:
        family = payload["families"][family_id]
        values = []
        for width in ("5x5", "9x9", "17x17", "33x33"):
            recovery = float(payload["contextSweep"][width][family_id]["probe"]["maps"]["albedo"]["recovery"])
            values.append(f"{recovery*100:+8.1f}%")
        lines.append(f"{family['family'][:26]:26s} " + " ".join(values))

    lines.extend(
        [
            "",
            f"PRIOR SWEEP - 4x, context {payload['priorSweepContext']}",
            "-" * 104,
            "Family                    within family    selected all    prepared corpus",
        ]
    )
    for family_id in family_ids:
        family = payload["families"][family_id]
        entry = payload["priorSweep"][family_id]
        values = [
            float(entry[scope]["maps"]["albedo"]["recovery"])
            for scope in ("withinFamilySelected", "selectedAllFamilies", "preparedCorpus")
        ]
        lines.append(
            f"{family['family'][:26]:26s} "
            f"{values[0]*100:+12.1f}% {values[1]*100:+14.1f}% {values[2]*100:+17.1f}%"
        )

    assessment = payload["assessment"]
    lines.extend(
        [
            "",
            "DECISION",
            "-" * 104,
            "Signals:",
            *[f"- {signal}" for signal in assessment["signals"]],
            "Recommended development directions:",
            *[f"- {item}" for item in assessment["recommendedDevelopmentDirections"]],
            "- keep-existing-v16-gates-unchanged",
            "",
            "Interpretation notes:",
            "- 2x/~3x/4x are synthesized from the same authored HR crop; ~3x uses the nearest integer LR grid.",
            "- Context descriptors pool the larger LR neighborhood to a fixed 5x5 descriptor grid, so distance cost stays bounded.",
            "- selected-all and prepared-corpus prior tests never use held-out validation crops as training evidence.",
            "- kNN recovery is evidence about statistical predictability, not a production method or a theoretical ceiling.",
            "- The README production target remains 4x; a 2x advantage would motivate progressive/multi-scale evidence, not lowering the goal.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    repo_root = args.repo_root.resolve()
    run_dir = _resolve_run(repo_root, args.run_dir)
    report = _read_json(run_dir / "report.json")
    if not isinstance(report, dict):
        raise RuntimeError(f"Invalid Stage 2 report: {run_dir / 'report.json'}")

    family_names = _family_name_map(report)
    selected_global = _selected_global_map(report)
    source = _load_report_records(repo_root, report, family_names)
    if not source["train"] or not source["validation"]:
        raise RuntimeError("Stage 2 report does not contain selected train/validation records")

    families: dict[str, dict[str, Any]] = {}
    for record in source["validation"]:
        family_id = record["familyId"]
        families[family_id] = {
            "familyId": family_id,
            "family": record["family"],
            "selectedHeldOutGlobalRecovery": float(selected_global.get(family_id, 0.0)),
        }

    print(f"[recovery-sweep] run: {run_dir}", flush=True)
    print("[recovery-sweep] CPU-only; no trained model inference", flush=True)

    scale_sweep: dict[str, dict[str, Any]] = {}
    scale_record_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for requested_scale in SCALES:
        name = f"{int(requested_scale)}x"
        context_width = int(args.scale_context_width)
        print(f"[recovery-sweep] scale {name}", flush=True)
        train = [
            _record_at_scale(
                record,
                requested_scale,
                context_width=context_width,
                hr_stride=args.hr_sample_stride,
            )
            for record in source["train"]
        ]
        held = [
            _record_at_scale(
                record,
                requested_scale,
                context_width=context_width,
                hr_stride=args.hr_sample_stride,
            )
            for record in source["validation"]
        ]
        scale_record_cache[name] = {"train": train, "validation": held}
        scale_sweep[name] = _probe_by_family(train, held, k=args.knn_k)
        for family_id, entry in scale_sweep[name].items():
            p = entry["probe"]["maps"]["albedo"]
            s = entry["heldOutSpectral"]["albedo"]
            print(
                f"  {entry['family']:24s} recovery={p['recovery']*100:+6.1f}% "
                f"corr={p['predictionResidualCorrelation']:+.3f} high={s['aboveFraction']*100:5.1f}%",
                flush=True,
            )

    base_4x = scale_record_cache["4x"]
    context_sweep: dict[str, dict[str, Any]] = {}
    context_record_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for width in CONTEXT_WIDTHS:
        name = f"{width}x{width}"
        print(f"[recovery-sweep] context {name}", flush=True)
        train: list[dict[str, Any]] = []
        held: list[dict[str, Any]] = []
        for split_name, output in (("train", train), ("validation", held)):
            for record in base_4x[split_name]:
                lr, residual = record["evidence"]
                clone = dict(record)
                clone["samples"] = _patch_samples(
                    lr,
                    residual,
                    context_width=width,
                    hr_stride=args.hr_sample_stride,
                )
                output.append(clone)
        context_record_cache[name] = {"train": train, "validation": held}
        context_sweep[name] = _probe_by_family(train, held, k=args.knn_k)

    _strongest, weak = _selected_weak_families(selected_global)
    if not weak:
        weak = list(families)
    context_scores = {
        name: _median_for_families(
            result, weak, ("probe", "maps", "albedo", "recovery")
        )
        for name, result in context_sweep.items()
    }
    best_context_name = max(
        context_scores,
        key=lambda key: context_scores[key] if math.isfinite(context_scores[key]) else -1.0e9,
    )
    prior_context_width = int(best_context_name.split("x", 1)[0])
    prior_records = context_record_cache[best_context_name]

    corpus_paths = _prepared_corpus_paths(repo_root, source["train"], args.max_corpus_crops)
    corpus_samples, corpus_used = _load_corpus_samples(
        corpus_paths,
        context_width=prior_context_width,
        hr_stride=args.hr_sample_stride,
    )
    prior_sweep = _prior_probe(
        prior_records["train"],
        prior_records["validation"],
        corpus_samples=corpus_samples,
        k=args.knn_k,
    )

    assessment = _assessment(selected_global, scale_sweep, context_sweep, prior_sweep)
    payload = {
        "schema": SCHEMA,
        "runDir": str(run_dir),
        "sourceReport": str((run_dir / "report.json").resolve()),
        "datasetFingerprint": report.get("datasetFingerprint"),
        "cpuOnly": True,
        "trainedModelInference": False,
        "datasetMutation": False,
        "readmeGoal": {
            "scale": "4x",
            "physicalMaps": ["albedo", "normal", "material"],
            "preserveAuthoredStructure": True,
            "avoidUnsupportedTextureInvention": True,
            "keepQualificationGates": True,
        },
        "config": {
            "scales": list(SCALES),
            "scaleContextWidthLrPixels": int(args.scale_context_width),
            "contextWidthsLrPixels": list(CONTEXT_WIDTHS),
            "descriptorGrid": DESCRIPTOR_GRID,
            "hrSampleStridePixels": int(args.hr_sample_stride),
            "knnK": int(args.knn_k),
            "maxPreparedCorpusCrops": int(args.max_corpus_crops),
        },
        "families": families,
        "scaleSweep": scale_sweep,
        "contextSweep": context_sweep,
        "priorSweepContext": best_context_name,
        "priorSweep": prior_sweep,
        "preparedCorpus": {
            "cropCount": len(corpus_used),
            "paths": corpus_used,
            "selectedTrainCropCount": len(source["train"]),
            "hasAdditionalPreparedEvidence": len(corpus_used) > len(source["train"]),
        },
        "assessment": assessment,
    }

    json_path = run_dir / "recovery_sweep.json"
    summary_path = run_dir / "recovery_sweep_summary.txt"
    _atomic_json(json_path, payload)
    _atomic_text(summary_path, _summary(payload))
    print(f"[recovery-sweep] JSON   : {json_path}", flush=True)
    print(f"[recovery-sweep] summary: {summary_path}", flush=True)
    print("[recovery-sweep] signals:", flush=True)
    for signal in assessment["signals"]:
        print(f"  - {signal}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPU-only NSAMDR Stage 2 scale/context/prior recovery sweep")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--scale-context-width", type=int, default=5)
    p.add_argument("--hr-sample-stride", type=int, default=DEFAULT_HR_SAMPLE_STRIDE)
    p.add_argument("--knn-k", type=int, default=DEFAULT_K)
    p.add_argument("--max-corpus-crops", type=int, default=64)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.scale_context_width < 1 or args.scale_context_width % 2 == 0:
        raise SystemExit("--scale-context-width must be a positive odd integer")
    if args.hr_sample_stride < 4:
        raise SystemExit("--hr-sample-stride must be >= 4")
    if args.knn_k < 1:
        raise SystemExit("--knn-k must be >= 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
