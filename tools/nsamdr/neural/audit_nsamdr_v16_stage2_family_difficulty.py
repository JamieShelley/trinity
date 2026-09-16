#!/usr/bin/env python3
"""CPU-only difficulty audit for a completed V16 Stage 2 run.

The audit intentionally does not load a trained model or use CUDA. It replays the
exact deterministic baseline B from the selected train/held-out crop bundles and
measures whether the four authored families differ mainly because of intrinsic
baseline/task difficulty, train-vs-held-out content shift, or limited unique spatial
coverage.

Outputs are written into the audited multiregion run directory:
  difficulty_audit.json
  difficulty_audit_summary.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.baseline import Baseline4x
from v14.dataset import _degrade, _normalise_xy


SCHEMA = "NSAMDR_V16_STAGE2_FAMILY_DIFFICULTY_AUDIT_V1"
EDGE_THRESHOLD = 0.02
SHIFT_FEATURES = (
    "baselineAlbedoMae",
    "targetGradientMean",
    "edgeDensity",
    "detailEnergy1px",
    "detailEnergy2px",
    "detailEnergy4px",
    "albedoStd",
    "normalStd",
    "materialStd",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _latest_completed_run(root: Path) -> Path | None:
    candidates: list[Path] = []
    for run_dir in root.glob("multiregion_*"):
        report_path = run_dir / "report.json"
        if not run_dir.is_dir() or not report_path.is_file():
            continue
        try:
            report = _read_json(report_path)
        except (OSError, ValueError, TypeError):
            continue
        if int(report.get("completedSteps") or 0) > 0:
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_run(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        value = requested
        if not value.is_absolute():
            value = repo_root / value
        value = value.resolve()
        if not (value / "report.json").is_file():
            raise RuntimeError(f"Stage 2 run has no report.json: {value}")
        return value
    root = repo_root / "artifacts/nsamdr/diagnostics/v16_mini"
    latest = _latest_completed_run(root)
    if latest is None:
        raise RuntimeError(f"No completed V16 Stage 2 multiregion run found under {root}")
    return latest.resolve()


def _resolve_crop(repo_root: Path, raw: str, split: str) -> Path:
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()
    basename = direct.name
    fallback = (
        repo_root
        / "artifacts/nsamdr/training_v9_preview_raven/crops"
        / split
        / basename
    )
    if fallback.is_file():
        return fallback.resolve()
    matches = list(
        (repo_root / "artifacts/nsamdr").glob(f"**/crops/{split}/{basename}")
    )
    if len(matches) == 1:
        return matches[0].resolve()
    raise RuntimeError(f"Selected Stage 2 crop is missing: {raw}")


def _normalise_loaded_normal(normal: np.ndarray) -> np.ndarray:
    xy = np.asarray(normal, dtype=np.uint8)[..., :2].astype(np.float32) / 127.5 - 1.0
    return _normalise_xy(xy)


def _load_crop(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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


def _baseline_maps(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = albedo.shape[:2]
    if height != width or height % 4:
        raise RuntimeError(f"Audit expects a square 4x Stage 2 crop; got {width}x{height}")
    lr_size = height // 4
    lr_a, lr_n, lr_m = _degrade(
        albedo,
        normal,
        material,
        lr_size,
        "clean",
        random.Random(14001),
    )
    baseline = Baseline4x(4).cpu().eval()
    with torch.no_grad():
        b_a, b_n, b_m = baseline(_batch(lr_a), _batch(lr_n), _batch(lr_m))
    return _batch(albedo), _batch(normal), _batch(material), b_a, b_n, b_m


def _gradient_map(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def _detail_band(value: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = radius * 2 + 1
    smooth = F.avg_pool2d(value.float(), kernel_size=kernel, stride=1, padding=radius)
    return value.float() - smooth


def _distribution(value: torch.Tensor) -> dict[str, float]:
    array = value.detach().float().cpu().numpy().reshape(-1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p01": float(np.quantile(array, 0.01)),
        "p99": float(np.quantile(array, 0.99)),
        "p01ToP99Range": float(np.quantile(array, 0.99) - np.quantile(array, 0.01)),
    }


def _map_error(target: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    delta = baseline.float() - target.float()
    return {
        "mae": float(delta.abs().mean().item()),
        "rmse": float(torch.sqrt((delta * delta).mean()).item()),
        "residualEnergy": float((delta * delta).mean().item()),
    }


def _record_metrics(
    path: Path,
    *,
    split: str,
    fallback_box: list[int] | None,
) -> dict[str, Any]:
    albedo, normal, material, metadata = _load_crop(path)
    t_a, t_n, t_m, b_a, b_n, b_m = _baseline_maps(albedo, normal, material)
    gradient = _gradient_map(t_a)
    albedo_dist = _distribution(t_a)
    normal_dist = _distribution(t_n)
    material_dist = _distribution(t_m)
    albedo_error = _map_error(t_a, b_a)
    normal_error = _map_error(t_n, b_n)
    material_error = _map_error(t_m, b_m)

    detail = {
        radius: float(_detail_band(t_a, radius).abs().mean().item())
        for radius in (1, 2, 4)
    }
    source_box = metadata.get("pixelBox") or fallback_box
    family_id = str(metadata.get("familyId") or path.name.split("_", 1)[0])
    effective_frequency = detail[1] / max(albedo_dist["std"], 1.0e-8)

    return {
        "path": str(path),
        "split": split,
        "familyId": family_id,
        "sourceBox": source_box,
        "cropSize": [int(albedo.shape[1]), int(albedo.shape[0])],
        "baselineError": {
            "albedo": albedo_error,
            "normal": normal_error,
            "material": material_error,
        },
        "targetStatistics": {
            "albedo": albedo_dist,
            "normal": normal_dist,
            "material": material_dist,
            "gradientMean": float(gradient.mean().item()),
            "gradientRms": float(torch.sqrt((gradient * gradient).mean()).item()),
            "edgeDensity": float((gradient > EDGE_THRESHOLD).float().mean().item()),
            "detailEnergy": {
                "1px": detail[1],
                "2px": detail[2],
                "4px": detail[4],
            },
            "effectiveFrequencyIndex": float(effective_frequency),
        },
        "auditVector": {
            "baselineAlbedoMae": albedo_error["mae"],
            "targetGradientMean": float(gradient.mean().item()),
            "edgeDensity": float((gradient > EDGE_THRESHOLD).float().mean().item()),
            "detailEnergy1px": detail[1],
            "detailEnergy2px": detail[2],
            "detailEnergy4px": detail[4],
            "albedoStd": albedo_dist["std"],
            "normalStd": normal_dist["std"],
            "materialStd": material_dist["std"],
        },
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else float("nan")


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"sampleCount": 0}
    result: dict[str, Any] = {
        "sampleCount": len(records),
        "baselineError": {},
        "targetStatistics": {},
        "auditVector": {},
    }
    for map_name in ("albedo", "normal", "material"):
        result["baselineError"][map_name] = {
            key: _mean(float(record["baselineError"][map_name][key]) for record in records)
            for key in ("mae", "rmse", "residualEnergy")
        }
    result["targetStatistics"] = {
        "gradientMean": _mean(float(record["targetStatistics"]["gradientMean"]) for record in records),
        "gradientRms": _mean(float(record["targetStatistics"]["gradientRms"]) for record in records),
        "edgeDensity": _mean(float(record["targetStatistics"]["edgeDensity"]) for record in records),
        "detailEnergy": {
            key: _mean(float(record["targetStatistics"]["detailEnergy"][key]) for record in records)
            for key in ("1px", "2px", "4px")
        },
        "effectiveFrequencyIndex": _mean(
            float(record["targetStatistics"]["effectiveFrequencyIndex"])
            for record in records
        ),
        "albedo": {
            key: _mean(float(record["targetStatistics"]["albedo"][key]) for record in records)
            for key in ("mean", "std", "p01", "p99", "p01ToP99Range")
        },
        "normal": {
            key: _mean(float(record["targetStatistics"]["normal"][key]) for record in records)
            for key in ("mean", "std", "p01", "p99", "p01ToP99Range")
        },
        "material": {
            key: _mean(float(record["targetStatistics"]["material"][key]) for record in records)
            for key in ("mean", "std", "p01", "p99", "p01ToP99Range")
        },
    }
    result["auditVector"] = {
        key: _mean(float(record["auditVector"][key]) for record in records)
        for key in SHIFT_FEATURES
    }
    return result


def _symmetric_relative_difference(a: float, b: float) -> float:
    denominator = max((abs(a) + abs(b)) * 0.5, 1.0e-8)
    return abs(a - b) / denominator


def _shift(train: dict[str, Any], heldout: dict[str, Any]) -> dict[str, Any]:
    if not train.get("sampleCount") or not heldout.get("sampleCount"):
        return {"available": False}
    per_feature: dict[str, float] = {}
    for key in SHIFT_FEATURES:
        a = float(train["auditVector"][key])
        b = float(heldout["auditVector"][key])
        per_feature[key] = _symmetric_relative_difference(a, b)
    values = list(per_feature.values())
    return {
        "available": True,
        "medianSymmetricRelativeShift": float(statistics.median(values)),
        "meanSymmetricRelativeShift": _mean(values),
        "maxSymmetricRelativeShift": max(values),
        "perFeature": per_feature,
    }


def _union_area(boxes: list[list[int]]) -> int:
    rectangles = [tuple(int(v) for v in box[:4]) for box in boxes if box and len(box) >= 4]
    if not rectangles:
        return 0
    xs = sorted({x for rect in rectangles for x in (rect[0], rect[2])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals: list[tuple[int, int]] = []
        for x0, y0, x1, y1 in rectangles:
            if x0 < right and x1 > left and y1 > y0:
                intervals.append((y0, y1))
        if not intervals:
            continue
        intervals.sort()
        merged = 0
        start, end = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= end:
                end = max(end, y1)
            else:
                merged += max(0, end - start)
                start, end = y0, y1
        merged += max(0, end - start)
        area += (right - left) * merged
    return int(area)


def _box_area(box: list[int] | None) -> int:
    if not box or len(box) < 4:
        return 0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _coverage(
    boxes: list[list[int]],
    source_size: list[int] | None,
    domain_box: list[int] | None,
) -> dict[str, Any]:
    union = _union_area(boxes)
    source_area = 0
    if source_size and len(source_size) >= 2:
        source_area = max(0, int(source_size[0])) * max(0, int(source_size[1]))
    domain_area = _box_area(domain_box)
    return {
        "selectedWindows": len(boxes),
        "uniqueSelectedPixels": union,
        "sourcePixelFraction": float(union / source_area) if source_area else None,
        "domainPixelFraction": float(union / domain_area) if domain_area else None,
        "sourceSize": source_size,
        "domainBox": domain_box,
    }


def _safe_ratio(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 1.0e-8]
    if len(finite) < 2:
        return None
    return max(finite) / min(finite)


def _source_names(report: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("candidate", "trainCandidate"):
        payload = report.get(key)
        if not isinstance(payload, dict):
            continue
        per_family = payload.get("perFamily")
        if not isinstance(per_family, dict):
            continue
        for family_id, family in per_family.items():
            if isinstance(family, dict):
                result[str(family_id)] = str(family.get("sourceAssetName") or family_id)
    return result


def _curve_metrics(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    curve_path = run_dir / "generalisation_curve.json"
    if not curve_path.is_file():
        return {}, {}
    curve = _read_json(curve_path)
    if not isinstance(curve, list) or not curve:
        return {}, {}
    selected: dict[str, Any] = {}
    final: dict[str, Any] = curve[-1] if isinstance(curve[-1], dict) else {}
    return selected, final


def _metric_for_family(payload: Any, family_id: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    per_family = payload.get("perFamily")
    if not isinstance(per_family, dict):
        return None
    value = per_family.get(family_id)
    return dict(value) if isinstance(value, dict) else None


def _diagnostic_assessment(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    heldout_mae: list[float] = []
    heldout_frequency: list[float] = []
    shifts: list[float] = []
    weak_shifts: list[float] = []
    train_coverages: list[float] = []

    for family in families.values():
        heldout = family.get("heldOut", {})
        if heldout.get("sampleCount"):
            heldout_mae.append(float(heldout["baselineError"]["albedo"]["mae"]))
            heldout_frequency.append(float(heldout["targetStatistics"]["effectiveFrequencyIndex"]))
        shift = family.get("trainHeldOutShift", {})
        if shift.get("available"):
            value = float(shift["medianSymmetricRelativeShift"])
            shifts.append(value)
            selected_metrics = family.get("selectedStage2HeldOutMetrics") or {}
            global_recovery = float(selected_metrics.get("medianGlobalRecovery", 0.0) or 0.0)
            if global_recovery < 0.45:
                weak_shifts.append(value)
        coverage = family.get("coverage", {}).get("train", {}).get("domainPixelFraction")
        if coverage is not None:
            train_coverages.append(float(coverage))

    difficulty_ratio = _safe_ratio(heldout_mae)
    frequency_ratio = _safe_ratio(heldout_frequency)
    median_shift = float(statistics.median(shifts)) if shifts else None
    weak_median_shift = float(statistics.median(weak_shifts)) if weak_shifts else None
    minimum_train_coverage = min(train_coverages) if train_coverages else None

    signals: list[dict[str, Any]] = []
    if difficulty_ratio is not None and difficulty_ratio >= 1.35:
        signals.append(
            {
                "signal": "family-baseline-difficulty-imbalance",
                "strength": difficulty_ratio,
                "evidence": f"held-out albedo baseline MAE spread is {difficulty_ratio:.2f}x across families",
            }
        )
    if frequency_ratio is not None and frequency_ratio >= 1.50:
        signals.append(
            {
                "signal": "family-frequency-content-imbalance",
                "strength": frequency_ratio,
                "evidence": f"held-out effective detail-frequency spread is {frequency_ratio:.2f}x across families",
            }
        )
    if weak_median_shift is not None and weak_median_shift >= 0.35:
        signals.append(
            {
                "signal": "train-heldout-content-distribution-shift",
                "strength": weak_median_shift,
                "evidence": f"median symmetric feature shift for families below the global gate is {weak_median_shift:.3f}",
            }
        )
    if minimum_train_coverage is not None and minimum_train_coverage < 0.65:
        signals.append(
            {
                "signal": "limited-independent-spatial-coverage",
                "strength": 1.0 - minimum_train_coverage,
                "evidence": f"lowest unique selected train coverage is {minimum_train_coverage*100:.1f}% of its train domain",
            }
        )

    signals.sort(key=lambda item: float(item.get("strength") or 0.0), reverse=True)
    names = {str(item["signal"]) for item in signals}
    if "train-heldout-content-distribution-shift" in names:
        next_experiment = "broaden-authored-spatial-and-family-diversity-before-retraining"
    elif "family-baseline-difficulty-imbalance" in names or "family-frequency-content-imbalance" in names:
        next_experiment = "test-family-and-map-loss-normalisation-before-another-long-run"
    elif "limited-independent-spatial-coverage" in names:
        next_experiment = "increase-independent-spatial-coverage-before-retraining"
    else:
        next_experiment = "no-single-cause-dominates; inspect per-family audit before changing V16"

    return {
        "crossFamilyHeldOutBaselineAlbedoMaeRatio": difficulty_ratio,
        "crossFamilyHeldOutEffectiveFrequencyRatio": frequency_ratio,
        "medianTrainHeldOutFeatureShift": median_shift,
        "weakFamilyMedianTrainHeldOutFeatureShift": weak_median_shift,
        "minimumTrainDomainCoverageFraction": minimum_train_coverage,
        "signals": signals,
        "suggestedNextExperiment": next_experiment,
        "heuristicThresholds": {
            "baselineDifficultyRatio": 1.35,
            "effectiveFrequencyRatio": 1.50,
            "trainHeldOutShift": 0.35,
            "trainDomainCoverageFraction": 0.65,
        },
    }


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "NSAMDR V16 STAGE 2 FAMILY DIFFICULTY AUDIT",
        "=" * 72,
        f"Run: {payload['runDir']}",
        f"Selected step: {payload.get('selectedStep')}  Completed: {payload.get('completedSteps')}",
        "CPU-only: yes; trained model inference: no; dataset mutation: no",
        "",
        "Family                     Base MAE train/held   Shift    Train cov  Held global(selected/final)",
        "-" * 104,
    ]
    for family_id, family in sorted(
        payload["families"].items(), key=lambda item: str(item[1].get("sourceAssetName") or item[0])
    ):
        name = str(family.get("sourceAssetName") or family_id)[:24]
        train = family.get("train", {})
        held = family.get("heldOut", {})
        train_mae = train.get("baselineError", {}).get("albedo", {}).get("mae")
        held_mae = held.get("baselineError", {}).get("albedo", {}).get("mae")
        shift = family.get("trainHeldOutShift", {}).get("medianSymmetricRelativeShift")
        coverage = family.get("coverage", {}).get("train", {}).get("domainPixelFraction")
        selected_global = (family.get("selectedStage2HeldOutMetrics") or {}).get("medianGlobalRecovery")
        final_global = (family.get("finalStage2HeldOutMetrics") or {}).get("medianGlobalRecovery")
        lines.append(
            f"{name:<26} "
            f"{float(train_mae) if train_mae is not None else float('nan'):.4f}/"
            f"{float(held_mae) if held_mae is not None else float('nan'):.4f}    "
            f"{float(shift) if shift is not None else float('nan'):.3f}    "
            f"{_fmt_pct(coverage):>8}   "
            f"{_fmt_pct(selected_global):>7}/{_fmt_pct(final_global):<7}"
        )

    assessment = payload["assessment"]
    lines.extend(("", "Signals", "-" * 72))
    if assessment["signals"]:
        for item in assessment["signals"]:
            lines.append(f"- {item['signal']}: {item['evidence']}")
    else:
        lines.append("- No single audit heuristic crossed its diagnostic threshold.")
    lines.extend(
        (
            "",
            f"Suggested next experiment: {assessment['suggestedNextExperiment']}",
            "",
            "Method notes:",
            f"- Edge density is the fraction of authored albedo pixels with |dx|+|dy| > {EDGE_THRESHOLD:.3f}.",
            "- Detail energy uses the same 1/2/4-pixel average-pool high-pass bands as V16 qualification.",
            "- Shift is a median symmetric relative difference over baseline error, gradient, edge, detail, and map-range features.",
            "- Coverage is the exact union of selected source boxes divided by the hard train/held-out domain area.",
        )
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only V16 Stage 2 family difficulty audit")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Completed multiregion run directory. Defaults to the latest completed run.",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    run_dir = _resolve_run(repo_root, args.run_dir)
    report = _read_json(run_dir / "report.json")
    if not isinstance(report, dict):
        raise RuntimeError(f"Invalid Stage 2 report: {run_dir / 'report.json'}")

    train_paths = [str(value) for value in report.get("trainRecords") or []]
    heldout_paths = [str(value) for value in report.get("validationRecords") or []]
    train_boxes = list(report.get("trainSourceBoxes") or [])
    heldout_boxes = list(report.get("validationSourceBoxes") or [])
    if not train_paths or not heldout_paths:
        raise RuntimeError("Stage 2 report does not contain selected train and held-out record paths")

    records: list[dict[str, Any]] = []
    for split, paths, boxes in (
        ("train", train_paths, train_boxes),
        ("validation", heldout_paths, heldout_boxes),
    ):
        for index, raw in enumerate(paths):
            path = _resolve_crop(repo_root, raw, split)
            fallback_box = boxes[index] if index < len(boxes) else None
            records.append(_record_metrics(path, split=split, fallback_box=fallback_box))

    names = _source_names(report)
    spatial_domains: dict[str, dict[str, Any]] = {}
    split_policy = dict(report.get("datasetSplitPolicy") or {})
    for item in split_policy.get("spatialDomains") or []:
        if isinstance(item, dict) and item.get("familyId"):
            spatial_domains[str(item["familyId"])] = dict(item)

    family_ids = sorted({str(record["familyId"]) for record in records})
    selected_candidate = dict(report.get("candidate") or {})
    selected_train_candidate = dict(report.get("trainCandidate") or {})
    curve = _read_json(run_dir / "generalisation_curve.json") if (run_dir / "generalisation_curve.json").is_file() else []
    final_curve = curve[-1] if isinstance(curve, list) and curve and isinstance(curve[-1], dict) else {}

    families: dict[str, dict[str, Any]] = {}
    for family_id in family_ids:
        family_records = [record for record in records if record["familyId"] == family_id]
        train_records = [record for record in family_records if record["split"] == "train"]
        heldout_records = [record for record in family_records if record["split"] == "validation"]
        train = _aggregate_records(train_records)
        heldout = _aggregate_records(heldout_records)
        domain = spatial_domains.get(family_id, {})
        source_size = domain.get("sourceSize")
        train_domain = domain.get("trainDomain")
        heldout_domain = domain.get("validationDomain")
        train_source_boxes = [record["sourceBox"] for record in train_records if record.get("sourceBox")]
        heldout_source_boxes = [record["sourceBox"] for record in heldout_records if record.get("sourceBox")]
        families[family_id] = {
            "familyId": family_id,
            "sourceAssetName": names.get(family_id, family_id),
            "sourceSize": source_size,
            "train": train,
            "heldOut": heldout,
            "trainHeldOutShift": _shift(train, heldout),
            "coverage": {
                "train": _coverage(train_source_boxes, source_size, train_domain),
                "heldOut": _coverage(heldout_source_boxes, source_size, heldout_domain),
                "trainCandidateWindows": domain.get("trainCandidateWindows"),
                "heldOutCandidateWindows": domain.get("validationCandidateWindows"),
            },
            "selectedStage2TrainMetrics": _metric_for_family(selected_train_candidate, family_id),
            "selectedStage2HeldOutMetrics": _metric_for_family(selected_candidate, family_id),
            "finalStage2TrainMetrics": _metric_for_family(final_curve.get("train"), family_id),
            "finalStage2HeldOutMetrics": _metric_for_family(final_curve.get("validation"), family_id),
            "records": family_records,
        }

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "runDir": str(run_dir),
        "sourceReport": str((run_dir / "report.json").resolve()),
        "datasetFingerprint": report.get("datasetFingerprint"),
        "selectedStep": int(report.get("selectedStep") or 0),
        "completedSteps": int(report.get("completedSteps") or 0),
        "cpuOnly": True,
        "trainedModelInference": False,
        "datasetMutation": False,
        "edgeDensityThreshold": EDGE_THRESHOLD,
        "families": families,
    }
    payload["assessment"] = _diagnostic_assessment(families)

    json_path = run_dir / "difficulty_audit.json"
    summary_path = run_dir / "difficulty_audit_summary.txt"
    _atomic_json(json_path, payload)
    _atomic_text(summary_path, _summary(payload))

    print(_summary(payload), end="")
    print(f"[v16-stage2-audit] JSON    : {json_path}")
    print(f"[v16-stage2-audit] summary : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
