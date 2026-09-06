"""Real-Raven baseline-relative fail-fast evidence for Quick structural training."""
from __future__ import annotations

import math
from typing import Any

from ..config import V9Config


class BaselineRelativeSmokeService:
    """Evaluate held-out real-Raven structural stages against deterministic baseline B."""

    def metrics(self, validation: dict[str, Any]) -> dict[str, float]:
        """Normalize the real-Raven A/B/C smoke metrics with fail-closed defaults.

        Purpose:
            Expose one stable metric bundle for logging, rejection evidence, and gating.
        Called by:
            BaselineRelativeSmokeService.safe_to_refine(), BaselineRelativeSmokeService.passed(),
            PassDrivenPipeline._run_quick_b1a_smoke(), PassDrivenPipeline._run_quick_b1b_smoke().
        Calls:
            No same-class helper methods.
        """
        def value(key: str, default: float) -> float:
            try:
                return float(validation.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        return {
            "baselineMae": value("sdf_stageb_baseline_mae", float("inf")),
            "candidateMae": value("sdf_stageb_renderer_mae", float("inf")),
            "relativeGain": value("sdf_stageb_renderer_improvement", float("-inf")),
            "improvementFraction": value("improvement_fraction", 0.0),
            "regressionFraction": value("regression_fraction", 1.0),
        }

    def safe_to_refine(self, validation: dict[str, Any], config: V9Config) -> bool:
        """Allow B1b only when topology-only B1a preserves baseline safety.

        Purpose:
            Accept intentional C == B identity after B1a while rejecting unsafe
            held-out Raven regressions before continuous refinement.
        Called by:
            PassDrivenPipeline._run_quick_b1a_smoke().
        Calls:
            BaselineRelativeSmokeService.metrics().
        """
        metrics = self.metrics(validation)
        finite = all(math.isfinite(value) for value in metrics.values())
        tolerance = max(1.0e-6, abs(metrics["baselineMae"]) * 1.0e-5)
        return bool(
            finite
            and metrics["candidateMae"] <= metrics["baselineMae"] + tolerance
            and metrics["regressionFraction"]
            <= float(config.maximum_validation_regression_fraction)
        )

    def passed(self, validation: dict[str, Any], config: V9Config) -> bool:
        """Require B1b C to beat B without exceeding the regression budget.

        Purpose:
            Enforce strict positive baseline-relative improvement once B1b can
            train continuous geometry and structural residual authority.
        Called by:
            PassDrivenPipeline._run_quick_b1b_smoke().
        Calls:
            BaselineRelativeSmokeService.metrics().
        """
        metrics = self.metrics(validation)
        finite = all(math.isfinite(value) for value in metrics.values())
        return bool(
            finite
            and metrics["candidateMae"] < metrics["baselineMae"]
            and metrics["relativeGain"] > 0.0
            and metrics["regressionFraction"]
            <= float(config.maximum_validation_regression_fraction)
        )
