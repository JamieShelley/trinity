#!/usr/bin/env python3
"""V12.9.1 correction layer for the hard-gated Raven Micro qualification.

V12.9 intentionally made upstream specialists mandatory, but its first Raven run
exposed two diagnostic-semantic mistakes before any production-model conclusion
could be drawn:

* G0 required absolute equality with authored-HR topology.  Production B1a only
  requires that learned topology does not regress the observable LR topology and
  does not create additional missing contours.
* Raven SDF win/regression metrics compared signed fields without resolving the
  model's global SDF-sign gauge.

This compatibility layer keeps the V12.9 ladder/budgets/artifacts intact, aligns
Raven SDF polarity before local improvement statistics, restores the production
B1a no-regression topology gate, and adds rendered-G topology non-regression to
G1.  It changes diagnostics only; no production parameters or inference semantics
are modified.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

import run_nsamdr_v9_raven_micro_diagnostic_v5 as v5
from v9.geometry_metrics import topology_mismatch


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_HARD_QUALIFICATION_V6"
QUALIFICATION_REVISION = "V12.9.1"

_ORIGINAL_GEOMETRY_METRICS = v5._geometry_metrics
_ORIGINAL_RUN_STAGE = v5._run_stage
_INSTALLED = False


def _align_global_sdf_polarity(
    field: np.ndarray,
    target: np.ndarray,
    band: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Resolve the global +/- SDF gauge by minimum target-band absolute error."""
    value = np.asarray(field, dtype=np.float32)
    target_value = np.asarray(target, dtype=np.float32)
    if band is None:
        active = np.ones_like(target_value, dtype=bool)
    else:
        active = np.asarray(band, dtype=bool)
        if not np.any(active):
            active = np.ones_like(target_value, dtype=bool)
    direct = float(np.mean(np.abs(value[active] - target_value[active])))
    flipped = float(np.mean(np.abs(-value[active] - target_value[active])))
    polarity = 1.0 if direct <= flipped else -1.0
    return value * polarity, polarity


def _sdf_arrays(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_distance = float(config.contour_sdf_max_distance_pixels)
    target = v5._tensor_to_numpy(batch["target_sdf"]) * max_distance

    source_value = outputs.get("source_sdf_prior_pixels")
    if not isinstance(source_value, torch.Tensor):
        source_value = outputs["source_sdf_prior"].float() * max_distance
    source = v5._tensor_to_numpy(source_value)

    predicted_value = outputs.get("sdf_pixels")
    if not isinstance(predicted_value, torch.Tensor):
        predicted_value = outputs.get("coarse_sdf_pixels")
    if not isinstance(predicted_value, torch.Tensor):
        predicted_value = outputs["sdf_raw"].float() * max_distance
    predicted = v5._tensor_to_numpy(predicted_value)
    return source, predicted, target


def _rgb_numpy(value: torch.Tensor) -> np.ndarray:
    return (
        value[0]
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )


def _geometry_metrics_v1291(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, float]:
    """Retain V12.9 evidence while correcting gauge and rendered topology checks."""
    result = dict(_ORIGINAL_GEOMETRY_METRICS(outputs, batch, config))
    source, predicted, target = _sdf_arrays(outputs, batch, config)
    band = np.abs(target) <= float(config.sdf_metric_band_pixels)
    if not np.any(band):
        band = np.ones_like(target, dtype=bool)

    gauge_invariant = bool(getattr(config, "sdf_sign_gauge_invariant", True))
    if gauge_invariant:
        source_aligned, source_polarity = _align_global_sdf_polarity(
            source, target, band
        )
        predicted_aligned, predicted_polarity = _align_global_sdf_polarity(
            predicted, target, band
        )
    else:
        source_aligned, source_polarity = source, 1.0
        predicted_aligned, predicted_polarity = predicted, 1.0

    source_error = np.abs(source_aligned - target)
    predicted_error = np.abs(predicted_aligned - target)
    regression_margin = float(
        getattr(config, "sdf_improvement_margin_pixels", 0.05)
    )
    result["winFraction"] = float(
        np.mean(predicted_error[band] < source_error[band])
    )
    result["regressionFraction"] = float(
        np.mean(
            predicted_error[band]
            > source_error[band] + regression_margin
        )
    )
    result["sdfGaugeInvariant"] = float(gauge_invariant)
    result["sourceGaugePolarity"] = float(source_polarity)
    result["predictedGaugePolarity"] = float(predicted_polarity)
    result["regressionMarginPixels"] = regression_margin

    target_rgb = _rgb_numpy(batch["target_albedo"])
    baseline_rgb = _rgb_numpy(outputs["baseline_albedo"])
    structure_value = outputs.get("structure_candidate_albedo")
    if not isinstance(structure_value, torch.Tensor):
        structure_value = outputs.get("boundary_reconstructed_albedo")
    if not isinstance(structure_value, torch.Tensor):
        structure_value = outputs["baseline_albedo"]
    structure_rgb = _rgb_numpy(structure_value)
    source_rendered_topology = float(topology_mismatch(baseline_rgb, target_rgb))
    predicted_rendered_topology = float(
        topology_mismatch(structure_rgb, target_rgb)
    )
    result["sourceRenderedTopologyMismatch"] = source_rendered_topology
    result["predictedRenderedTopologyMismatch"] = predicted_rendered_topology
    result["renderedTopologyRegression"] = float(
        predicted_rendered_topology > source_rendered_topology
    )
    return result


def _run_stage_v1291(**kwargs: Any) -> dict[str, Any]:
    """Correct G0/G1 gates while delegating all optimization/artifacts to V12.9."""
    values = dict(kwargs)
    name = str(values.get("name", ""))
    config = values.get("config")

    if name == "G0_geometry_topology":
        # Production B1a is a topology-safety bootstrap.  Authored HR can contain
        # topology/detail that native LR cannot observe, so absolute HR topology
        # equality is not a valid Raven bootstrap requirement.
        def topology_score(item: dict[str, Any]) -> float:
            geometry = item["geometry"]
            missing_excess = max(
                0.0,
                float(geometry["predictedMissingContourFraction"])
                - float(geometry["sourceMissingContourFraction"]),
            )
            chamfer = float(geometry["predictedChamferPixels"])
            if not np.isfinite(chamfer):
                chamfer = 1000.0
            return (
                -100.0 * float(geometry["topologyRegression"])
                -10.0 * missing_excess
                -min(chamfer, 1000.0)
            )

        def topology_qualified(item: dict[str, Any]) -> bool:
            geometry = item["geometry"]
            tolerance = float(
                getattr(config, "sdf_missing_contour_tolerance", 0.0)
            )
            return bool(
                float(geometry["topologyRegression"]) == 0.0
                and float(geometry["predictedMissingContourFraction"])
                <= float(geometry["sourceMissingContourFraction"]) + tolerance
            )

        values["score"] = topology_score
        values["qualified"] = topology_qualified

    elif name == "G1_geometry_metric_render":
        previous_score = values["score"]
        previous_qualified = values["qualified"]

        def geometry_score(item: dict[str, Any]) -> float:
            return float(previous_score(item)) - 4.0 * float(
                item["geometry"].get("renderedTopologyRegression", 1.0)
            )

        def geometry_qualified(item: dict[str, Any]) -> bool:
            geometry = item["geometry"]
            catastrophic = float(
                getattr(config, "sdf_catastrophic_chamfer_pixels", 48.0)
            )
            return bool(
                previous_qualified(item)
                and float(geometry.get("renderedTopologyRegression", 1.0)) == 0.0
                and float(geometry["predictedChamferPixels"]) <= catastrophic
            )

        values["score"] = geometry_score
        values["qualified"] = geometry_qualified

    return _ORIGINAL_RUN_STAGE(**values)


def install_v1291_diagnostic_semantics() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    v5.REPORT_SCHEMA = REPORT_SCHEMA
    v5.QUALIFICATION_REVISION = QUALIFICATION_REVISION
    v5._geometry_metrics = _geometry_metrics_v1291
    v5._run_stage = _run_stage_v1291
    _INSTALLED = True


def main(argv: list[str] | None = None) -> int:
    install_v1291_diagnostic_semantics()
    print(
        "[micro-v6] V12.9.1: production B1a topology-safety + gauge-corrected G1 semantics active",
        flush=True,
    )
    return v5.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
