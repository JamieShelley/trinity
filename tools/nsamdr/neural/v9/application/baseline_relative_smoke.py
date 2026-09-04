"""Real-Raven baseline-relative fail-fast evidence for Quick structural training."""
from __future__ import annotations

import math
from typing import Any

from ..config import V9Config


class BaselineRelativeSmokeService:
    """Evaluate held-out real-Raven B1a validation against deterministic baseline B."""

    def metrics(self, validation: dict[str, Any]) -> dict[str, float]:
        """Normalize the real-Raven A/B/C smoke metrics with fail-closed defaults.

        Purpose:
            Expose one stable metric bundle for logging, rejection evidence, and gating.
        Called by:
            BaselineRelativeSmokeService.passed(), PassDrivenPipeline._run_quick_b1a_smoke().
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

    def passed(self, validation: dict[str, Any], config: V9Config) -> bool:
        """Require C to beat B on held-out Raven without exceeding safety regressions.

        Purpose:
            Enforce the Quick fail-fast contract before any B1b work can begin.
        Called by:
            PassDrivenPipeline._run_quick_b1a_smoke().
        Calls:
            metrics().
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
