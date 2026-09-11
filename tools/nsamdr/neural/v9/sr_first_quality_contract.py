from __future__ import annotations

"""V13.1 quality contract for SR-first production reconstruction.

The first V13 Raven proof established that the SR-first architecture is visually
useful, but it also showed two quality ceilings:

* the albedo residual saturated the previous 0.20 cap;
* selector qualification over-rewarded preservation beyond the required 99%,
  allowing visible SR recovery to be discarded unnecessarily.

V13.1 raises only the albedo SR headroom and publishes the stricter quality target.
It does not add trainable parameters, checkpoint keys or inference inputs.
"""

from typing import Any

from .model import FidelityResidualNetV9


SR_QUALITY_REVISION = "V13.1"
SR_ALBEDO_RESIDUAL_CAP = 0.40
SR_REQUIRED_EDGE_RECOVERY = 0.70
SR_REQUIRED_GLOBAL_RECOVERY = 0.50
SR_REQUIRED_GRADIENT_RECOVERY = 0.40
SR_REQUIRED_SELECTOR_RETENTION = 0.90
SR_PLATEAU_MIN_STEPS = 1024
SR_PLATEAU_STALE_EVALS = 8
SR_PLATEAU_MIN_SCORE_DELTA = 0.002
SELECTOR_PLATEAU_MIN_STEPS = 256
SELECTOR_PLATEAU_STALE_EVALS = 8
SELECTOR_PLATEAU_MIN_SCORE_DELTA = 0.001

_INSTALLED = False
_PREVIOUS_MODEL_INIT: Any = None
_PREVIOUS_ARCHITECTURE_CONTRACT: Any = None


def _model_init_with_v131_sr_cap(
    self: FidelityResidualNetV9,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Install the V13.1 0.40 albedo headroom on every production model instance."""
    if _PREVIOUS_MODEL_INIT is None:
        raise RuntimeError("V13.1 installed without previous model init")
    _PREVIOUS_MODEL_INIT(self, *args, **kwargs)

    # Keep the public config and the V12.3/V13 detail-adapter cap in exact parity.
    # Normal/material headroom is intentionally unchanged.
    self.config.detail_albedo_max_delta = float(SR_ALBEDO_RESIDUAL_CAP)
    self.detail_net._parallel_albedo_cap = float(SR_ALBEDO_RESIDUAL_CAP)


def _architecture_contract_with_v131_quality(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _PREVIOUS_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V13.1 installed without architecture contract")
    contract = dict(_PREVIOUS_ARCHITECTURE_CONTRACT(self))
    contract["srQualityRevision"] = SR_QUALITY_REVISION
    contract["srAlbedoResidualCap"] = SR_ALBEDO_RESIDUAL_CAP
    contract["srVisualQualification"] = {
        "edgeRecovery": SR_REQUIRED_EDGE_RECOVERY,
        "globalRecovery": SR_REQUIRED_GLOBAL_RECOVERY,
        "gradientRecovery": SR_REQUIRED_GRADIENT_RECOVERY,
        "selectorRetention": SR_REQUIRED_SELECTOR_RETENTION,
        "protectedPreservation": 0.99,
        "maximumSteps": 3072,
        "plateauStop": True,
    }
    return contract


def selector_checkpoint_key(
    *,
    preservation: float,
    edge_recovery: float,
    global_recovery: float,
    edge_retention: float,
    global_retention: float,
    preservation_required: float,
) -> tuple[int, float, float]:
    """Lexicographic selector ranking: safety first, then fidelity, not excess safety."""
    safe = int(float(preservation) >= float(preservation_required))
    if safe:
        fidelity = (
            float(edge_recovery)
            + 0.50 * float(global_recovery)
            + 0.20 * float(edge_retention)
            + 0.10 * float(global_retention)
        )
        return (1, fidelity, 0.0)

    # Before any checkpoint satisfies the hard safety constraint, move toward it.
    # Fidelity is only a tie-breaker in the unsafe regime.
    fidelity = float(edge_recovery) + 0.50 * float(global_recovery)
    return (0, float(preservation), fidelity)


def install_sr_first_quality_contract() -> None:
    """Install V13.1 after V13 SR-first composition."""
    global _INSTALLED, _PREVIOUS_MODEL_INIT, _PREVIOUS_ARCHITECTURE_CONTRACT
    if _INSTALLED:
        return

    _PREVIOUS_MODEL_INIT = FidelityResidualNetV9.__init__
    _PREVIOUS_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.__init__ = _model_init_with_v131_sr_cap
    FidelityResidualNetV9.architecture_contract = _architecture_contract_with_v131_quality
    _INSTALLED = True
