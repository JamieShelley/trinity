#!/usr/bin/env python3
"""CPU-only V16.2 structure-support audit with area-normalized evidence.

The earlier audit reported the fraction of B->A error inside one dilated boundary
band.  That number can be misleading when the band covers a large fraction of the
image.  V3 therefore sweeps boundary thresholds and band widths, reports boundary
area explicitly, and measures error enrichment relative to occupied image area.

Authored HR maps are used only to define diagnostic structure.  Production never
receives authored HR structure or any external SDF/geometry authority.
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

from v16.stage2_data import baseline_maps, load_crop, read_json, resolve_crop, resolve_run
from v16.structure import StructureTargetConfig, derive_structure_targets

SCHEMA = "NSAMDR_V16_STRUCTURE_SUPPORT_AUDIT_V3"
THRESHOLDS = (0.35, 0.50, 0.65, 0.80)
BAND_RADII = (0, 2, 4)
SUPPORT_TOLERANCE = 2
NORMALIZATION_PERCENTILE = 99.0


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


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def _error_fraction(error: np.ndarray, mask: np.ndarray) -> float:
    total = float(np.asarray(error, dtype=np.float64).sum())
    if total <= 1.0e-12:
        return 0.0
    return float(np.asarray(error, dtype=np.float64)[mask].sum() / total)


def _band_metrics(
    boundary: np.ndarray,
    radius: int,
    errors: dict[str, np.ndarray],
) -> dict[str, Any]:
    band = _dilate(boundary, radius)
    area = float(band.mean())
    result: dict[str, Any] = {
        "bandRadiusPixels": int(radius),
        "boundaryAreaFraction": area,
    }
    for name, error in errors.items():
        fraction = _error_fraction(error, band)
        result[f"{name}ErrorFraction"] = fraction
        result[f"{name}ErrorEnrichment"] = (
            fraction / max(area, 1.0e-12) if area > 0.0 else 0.0
        )
    return result


def _support_metrics(
    authored_boundary: np.ndarray,
    evidence_boundary: np.ndarray,
) -> dict[str, float]:
    authored_tolerance = _dilate(authored_boundary, SUPPORT_TOLERANCE)
    evidence_tolerance = _dilate(evidence_boundary, SUPPORT_TOLERANCE)
    precision = (
        float(authored_tolerance[evidence_boundary].mean())
        if evidence_boundary.any()
        else 1.0
    )
    recall = (
        float(evidence_tolerance[authored_boundary].mean())
        if authored_boundary.any()
        else 1.0
    )
    return {"precision": precision, "recall": recall}


def _record(path: Path, split: str) -> dict[str, Any]:
    albedo, normal, material, metadata = load_crop(path)
    _ta, _tn, _tm, ba, bn, bm = baseline_maps(albedo, normal, material)
    baseline = (_hwc(ba), _hwc(bn), _hwc(bm))

    errors = {
        "albedo": np.abs(albedo - baseline[0]).mean(axis=2),
        "normal": np.abs(normal - baseline[1]).mean(axis=2),
        "material": np.abs(material - baseline[2]).mean(axis=2),
    }
    errors["combined"] = (
        errors["albedo"] + 0.25 * errors["normal"] + 0.25 * errors["material"]
    )

    sweep: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        cfg = StructureTargetConfig(
            boundary_threshold=float(threshold),
            normalization_percentile=NORMALIZATION_PERCENTILE,
        )
        authored = derive_structure_targets(albedo, normal, material, config=cfg)
        evidence = derive_structure_targets(*baseline, config=cfg)
        boundary = np.asarray(authored["boundaryMask"], dtype=bool)
        evidence_boundary = np.asarray(evidence["boundaryMask"], dtype=bool)
        support = _support_metrics(boundary, evidence_boundary)

        component_support: dict[str, float] = {}
        for name, strength in evidence["componentStrength"].items():
            visible = _dilate(
                np.asarray(strength, dtype=np.float32) >= float(threshold),
                SUPPORT_TOLERANCE,
            )
            component_support[name] = (
                float(visible[boundary].mean()) if boundary.any() else 1.0
            )

        threshold_key = f"{threshold:.2f}"
        sweep[threshold_key] = {
            "threshold": float(threshold),
            "authoredBoundaryPixelFraction": float(boundary.mean()),
            "evidenceBoundaryPixelFraction": float(evidence_boundary.mean()),
            "support": support,
            "componentRecall": component_support,
            "bands": {
                str(radius): _band_metrics(boundary, radius, errors)
                for radius in BAND_RADII
            },
        }

    family_id = str(metadata.get("familyId") or path.name.split("_", 1)[0])
    return {
        "path": str(path),
        "split": split,
        "familyId": family_id,
        "family": str(
            metadata.get("sourceAssetName")
            or metadata.get("familyName")
            or family_id
        ),
        "sweep": sweep,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        tkey = f"{threshold:.2f}"
        entries = [record["sweep"][tkey] for record in records]
        threshold_result: dict[str, Any] = {
            "threshold": float(threshold),
            "authoredBoundaryPixelFraction": _median(
                [float(item["authoredBoundaryPixelFraction"]) for item in entries]
            ),
            "evidenceBoundaryPixelFraction": _median(
                [float(item["evidenceBoundaryPixelFraction"]) for item in entries]
            ),
            "precision": _median(
                [float(item["support"]["precision"]) for item in entries]
            ),
            "recall": _median(
                [float(item["support"]["recall"]) for item in entries]
            ),
            "bands": {},
        }
        for radius in BAND_RADII:
            rkey = str(radius)
            bands = [item["bands"][rkey] for item in entries]
            threshold_result["bands"][rkey] = {
                key: _median([float(band[key]) for band in bands])
                for key in bands[0]
                if key != "bandRadiusPixels"
            }
            threshold_result["bands"][rkey]["bandRadiusPixels"] = int(radius)
        result[tkey] = threshold_result
    return result


def _aggregate_families(families: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    values = list(families.values())
    for threshold in THRESHOLDS:
        tkey = f"{threshold:.2f}"
        for radius in BAND_RADII:
            rkey = str(radius)
            points.append(
                {
                    "threshold": float(threshold),
                    "bandRadiusPixels": int(radius),
                    "medianBoundaryAreaFraction": _median(
                        [float(item["sweep"][tkey]["bands"][rkey]["boundaryAreaFraction"]) for item in values]
                    ),
                    "medianCombinedErrorFraction": _median(
                        [float(item["sweep"][tkey]["bands"][rkey]["combinedErrorFraction"]) for item in values]
                    ),
                    "medianCombinedErrorEnrichment": _median(
                        [float(item["sweep"][tkey]["bands"][rkey]["combinedErrorEnrichment"]) for item in values]
                    ),
                    "medianAlbedoErrorEnrichment": _median(
                        [float(item["sweep"][tkey]["bands"][rkey]["albedoErrorEnrichment"]) for item in values]
                    ),
                    "medianNormalErrorEnrichment": _median(
                        [float(item["sweep"][tkey]["bands"][rkey]["normalErrorEnrichment"]) for item in values]
                    ),
                    "medianMaterialErrorEnrichment": _median(
                        [float(item["sweep"][tkey]["bands"][rkey]["materialErrorEnrichment"]) for item in values]
                    ),
                    "medianPrecision": _median(
                        [float(item["sweep"][tkey]["precision"]) for item in values]
                    ),
                    "medianRecall": _median(
                        [float(item["sweep"][tkey]["recall"]) for item in values]
                    ),
                }
            )
    return points


def _select_operating_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    compact = [
        point
        for point in points
        if 0.01 <= float(point["medianBoundaryAreaFraction"]) <= 0.35
    ]
    pool = compact or points
    return max(
        pool,
        key=lambda point: (
            float(point["medianCombinedErrorEnrichment"]),
            float(point["medianCombinedErrorFraction"]),
            -float(point["medianBoundaryAreaFraction"]),
        ),
    )


def _decision(point: dict[str, Any]) -> str:
    enrichment = float(point["medianCombinedErrorEnrichment"])
    error_fraction = float(point["medianCombinedErrorFraction"])
    precision = float(point["medianPrecision"])
    recall = float(point["medianRecall"])
    if enrichment >= 1.75 and error_fraction >= 0.50 and precision >= 0.25 and recall >= 0.50:
        return "compact-structure-signal-concentrated-and-observable"
    if enrichment >= 1.25 and error_fraction >= 0.30:
        return "structure-signal-present-use-conditioning-not-explicit-profile-rendering"
    return "compact-structure-evidence-insufficient-as-primary-reconstruction-authority"


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    run_dir = resolve_run(repo_root, args.run_dir)
    report = read_json(run_dir / "report.json")

    held = [
        _record(resolve_crop(repo_root, str(raw), "validation"), "validation")
        for raw in (report.get("validationRecords") or [])
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in held:
        grouped[record["familyId"]].append(record)

    families: dict[str, Any] = {}
    for family_id, records in sorted(grouped.items()):
        families[family_id] = {
            "family": records[0]["family"],
            "sweep": _aggregate_records(records),
        }

    points = _aggregate_families(families)
    selected = _select_operating_point(points)
    decision = _decision(selected)
    payload = {
        "schema": SCHEMA,
        "sourceRun": str(run_dir),
        "cpuOnly": True,
        "datasetMutation": False,
        "trainedModelInference": False,
        "normalizationPercentile": NORMALIZATION_PERCENTILE,
        "thresholds": list(THRESHOLDS),
        "bandRadiiPixels": list(BAND_RADII),
        "supportTolerancePixels": SUPPORT_TOLERANCE,
        "deprecatedV2Metric": (
            "The old single 99.9% boundary-band error figure is not used for decisions "
            "because it did not report occupied image area."
        ),
        "families": families,
        "aggregateSweep": points,
        "selectedOperatingPoint": selected,
        "decision": decision,
    }
    json_path = run_dir / "structure_support_audit.json"
    summary_path = run_dir / "structure_support_audit_summary.txt"
    _atomic_json(json_path, payload)

    lines = [
        "NSAMDR V16 STRUCTURE SUPPORT AUDIT V3",
        "=" * 108,
        f"Run: {run_dir}",
        "CPU-only: yes; authored HR structure is diagnostic authority only",
        "Old 99.9% single-band result: DEPRECATED (band area was not reported)",
        "",
        "Threshold  Radius  Area     Error in band  Enrichment  Precision  Recall",
        "-" * 108,
    ]
    for point in points:
        lines.append(
            f"{point['threshold']:>8.2f} {point['bandRadiusPixels']:>7d} "
            f"{point['medianBoundaryAreaFraction']*100:>6.1f}% "
            f"{point['medianCombinedErrorFraction']*100:>12.1f}% "
            f"{point['medianCombinedErrorEnrichment']:>10.2f}x "
            f"{point['medianPrecision']*100:>9.1f}% "
            f"{point['medianRecall']*100:>6.1f}%"
        )
    lines += [
        "",
        "Selected compact operating point",
        "-" * 108,
        f"Threshold                  : {selected['threshold']:.2f}",
        f"Band radius                : {selected['bandRadiusPixels']} px",
        f"Median occupied area       : {selected['medianBoundaryAreaFraction']*100:.1f}%",
        f"Median B->A error captured : {selected['medianCombinedErrorFraction']*100:.1f}%",
        f"Median error enrichment    : {selected['medianCombinedErrorEnrichment']:.2f}x",
        f"Albedo enrichment          : {selected['medianAlbedoErrorEnrichment']:.2f}x",
        f"Normal enrichment          : {selected['medianNormalErrorEnrichment']:.2f}x",
        f"Material enrichment        : {selected['medianMaterialErrorEnrichment']:.2f}x",
        f"LR boundary precision      : {selected['medianPrecision']*100:.1f}%",
        f"LR boundary recall         : {selected['medianRecall']*100:.1f}%",
        "",
        "DECISION",
        "-" * 108,
        decision,
        "",
        "Interpretation:",
        "- Enrichment >1 means B->A error is denser near compact authored boundaries than elsewhere.",
        "- Structure is conditioning evidence only; the failed explicit BoundaryProfileNet is not revived.",
        "- Threshold and band sweeps replace the old permissive one-threshold/large-band claim.",
    ]
    _atomic_text(summary_path, "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0, json_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Audit compact V16 shared-structure observability with area normalization"
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--run-dir", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
