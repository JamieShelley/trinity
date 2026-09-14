from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StabilityDecision:
    diverged: bool
    reason: str | None


class DivergenceMonitor:
    """Detect persistent dead-gradient or residual-saturation collapse.

    This monitor does not change model gradients or qualification thresholds. It only
    stops a Capacity run when continued optimization no longer provides a valid capacity
    test.
    """

    def __init__(
        self,
        *,
        saturation_limit: float = 0.95,
        saturation_patience_reports: int = 3,
        zero_gradient_epsilon: float = 1.0e-12,
        zero_gradient_patience_reports: int = 3,
    ) -> None:
        self.saturation_limit = float(saturation_limit)
        self.saturation_patience_reports = int(saturation_patience_reports)
        self.zero_gradient_epsilon = float(zero_gradient_epsilon)
        self.zero_gradient_patience_reports = int(zero_gradient_patience_reports)

        self._saturation_reports = 0
        self._zero_gradient_reports = 0
        self._saw_nonzero_gradient = False

    def check_scalar_integrity(
        self,
        *,
        loss: float,
        gradient_norm: float | None = None,
    ) -> StabilityDecision:
        if not math.isfinite(float(loss)):
            return StabilityDecision(True, "non-finite-loss")
        if gradient_norm is not None and not math.isfinite(float(gradient_norm)):
            return StabilityDecision(True, "non-finite-gradient-norm")
        return StabilityDecision(False, None)

    def observe_report(
        self,
        *,
        loss: float,
        gradient_norm: float,
        saturation: dict[str, float],
    ) -> StabilityDecision:
        scalar = self.check_scalar_integrity(
            loss=loss,
            gradient_norm=gradient_norm,
        )
        if scalar.diverged:
            return scalar

        if gradient_norm > self.zero_gradient_epsilon:
            self._saw_nonzero_gradient = True
            self._zero_gradient_reports = 0
        elif self._saw_nonzero_gradient:
            self._zero_gradient_reports += 1
        else:
            self._zero_gradient_reports = 0

        if any(
            float(value) >= self.saturation_limit
            for value in saturation.values()
        ):
            self._saturation_reports += 1
        else:
            self._saturation_reports = 0

        if self._zero_gradient_reports >= self.zero_gradient_patience_reports:
            return StabilityDecision(True, "persistent-zero-gradient")

        if self._saturation_reports >= self.saturation_patience_reports:
            saturated = sorted(
                key
                for key, value in saturation.items()
                if float(value) >= self.saturation_limit
            )
            suffix = ",".join(saturated) if saturated else "unknown"
            return StabilityDecision(
                True,
                f"persistent-residual-saturation:{suffix}",
            )

        return StabilityDecision(False, None)

    def state(self) -> dict[str, object]:
        return {
            "saturationLimit": self.saturation_limit,
            "saturationPatienceReports": self.saturation_patience_reports,
            "zeroGradientEpsilon": self.zero_gradient_epsilon,
            "zeroGradientPatienceReports": self.zero_gradient_patience_reports,
            "consecutiveSaturationReports": self._saturation_reports,
            "consecutiveZeroGradientReports": self._zero_gradient_reports,
            "sawNonzeroGradient": self._saw_nonzero_gradient,
        }
