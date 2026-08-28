"""Failure classification policy for bounded evolutionary recovery."""
from __future__ import annotations

import math
from typing import Any, Mapping

from .domain import FailureKind


class FailureDetector:
    """Policy object that separates software/numerical/learning/representation failures."""

    def classify(
        self,
        *,
        error: BaseException | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> FailureKind:
        """Classify failure evidence before recovery is allowed.

        Purpose:
            Prevent architecture evolution from hiding software or numerical defects.
        Called by:
            EvolutionaryRecoveryController.can_recover(), compatibility classify_failure().
        Calls:
            No project functions.
        """
        if error is not None:
            text = f"{type(error).__name__}: {error}".lower()
            numerical_markers = (
                "out of memory", "cuda oom", "nan", "nonfinite", "non-finite",
                "overflow", "underflow", "inf loss", "gradient explosion",
            )
            if any(marker in text for marker in numerical_markers):
                return FailureKind.NUMERICAL
            return FailureKind.SOFTWARE

        data = metrics or {}
        try:
            topology_regression = float(data.get("sdf_stageb_topology_regression_fraction", 0.0))
            missing = float(data.get("sdf_predicted_missing_contour_fraction", 0.0))
            source_missing = float(data.get("sdf_source_missing_contour_fraction", 0.0))
            gain = float(data.get("sdf_zero_contour_relative_gain_mean", -1.0))
            chamfer = float(data.get("sdf_zero_contour_chamfer_pixels", math.inf))
        except (TypeError, ValueError):
            return FailureKind.SOFTWARE

        if (
            topology_regression > 0.0
            or missing > source_missing + 1.0e-6
            or gain < 0.0
            or not math.isfinite(chamfer)
        ):
            return FailureKind.REPRESENTATION
        return FailureKind.LEARNING


def classify_failure(
    *,
    error: BaseException | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> FailureKind:
    """Compatibility function for existing callers of evolutionary_recovery.py.

    Purpose:
        Preserve the old public functional API while delegating policy to an object.
    Called by:
        Legacy NSAMDR orchestration and tests.
    Calls:
        FailureDetector.classify().
    """
    return FailureDetector().classify(error=error, metrics=metrics)
