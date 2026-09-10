from __future__ import annotations

"""V12.4 baseline-relative specialist and protected-preservation contract.

NSAMDR is anchored to deterministic reconstruction B. Geometry, seam/profile and
appearance/detail are specialists that may only spend authority where authored
training evidence shows that B is defective. The V12.3 direct-detail contract
already makes detail independent of upstream geometry/seam state; this contract
adds the shared preservation rule used while training that candidate and the
final selector.

A protected pixel is one where deterministic B already matches authored A within
a small RGB tolerance. At least 99% of those pixels must remain within one 8-bit
code value of B. The target is used only to supervise and measure this behaviour
while training/qualifying; production inference never receives authored HR.

No parameters or checkpoint keys are added. Existing candidate generation and
BenefitSelector remain the production authority path.
"""

from typing import Any

import torch

from .model import FidelityResidualNetV9


BASELINE_RELATIVE_SPECIALIST_REVISION = "V12.4"
PROTECTED_PRESERVATION_REQUIRED = 0.990
PROTECTED_BASELINE_ERROR_TOLERANCE = 2.0 / 255.0
PROTECTED_DRIFT_TOLERANCE = 1.0 / 255.0
PROTECTED_DETAIL_LOSS_WEIGHT = 24.0
PROTECTED_SELECTOR_LOSS_WEIGHT = 48.0

_MODEL_INSTALLED = False
_BACKEND_INSTALLED = False
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None
_ORIGINAL_PRODUCTION_HEAD_LOSS: Any = None


# Purpose: Identify pixels where deterministic B already reconstructs authored A.
# Called by: protected_preservation_terms and tests/diagnostics.
# Calls: PyTorch tensor reductions only.
def protected_mask(
    baseline: torch.Tensor,
    target: torch.Tensor,
    *,
    baseline_error_tolerance: float = PROTECTED_BASELINE_ERROR_TOLERANCE,
) -> torch.Tensor:
    if baseline.shape != target.shape:
        raise ValueError(
            f"baseline/target shape mismatch: {tuple(baseline.shape)} != {tuple(target.shape)}"
        )
    error = (baseline.detach().float() - target.detach().float()).abs().amax(
        dim=1, keepdim=True
    )
    return (error <= float(baseline_error_tolerance)).detach()


# Purpose: Compute the 99%-preservation evidence and differentiable excess-drift penalty.
# Called by: _loss_with_protected_preservation and tests/diagnostics.
# Calls: protected_mask.
def protected_preservation_terms(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    *,
    baseline_error_tolerance: float = PROTECTED_BASELINE_ERROR_TOLERANCE,
    drift_tolerance: float = PROTECTED_DRIFT_TOLERANCE,
) -> dict[str, torch.Tensor]:
    if candidate.shape != baseline.shape:
        raise ValueError(
            f"candidate/baseline shape mismatch: {tuple(candidate.shape)} != {tuple(baseline.shape)}"
        )

    protected = protected_mask(
        baseline,
        target,
        baseline_error_tolerance=baseline_error_tolerance,
    )
    protected_weight = protected.float()
    protected_count = protected_weight.sum()
    pixel_count = protected_weight.new_tensor(float(protected_weight.numel()))

    drift = (candidate.float() - baseline.detach().float()).abs().amax(
        dim=1, keepdim=True
    )
    within_tolerance = (drift <= float(drift_tolerance)).float()
    denominator = protected_count.clamp_min(1.0)
    protected_fraction = protected_count / pixel_count.clamp_min(1.0)
    preservation_rate = (
        (within_tolerance * protected_weight).sum() / denominator
        if bool(protected.any().item())
        else drift.new_tensor(1.0)
    )
    mean_drift = (
        (drift * protected_weight).sum() / denominator
        if bool(protected.any().item())
        else drift.new_zeros(())
    )
    excess_drift = torch.relu(drift - float(drift_tolerance))
    excess_drift_loss = (excess_drift * protected_weight).sum() / denominator

    if bool(protected.any().item()):
        masked_drift = torch.where(protected, drift, torch.zeros_like(drift))
        max_drift = masked_drift.amax()
    else:
        max_drift = drift.new_zeros(())

    return {
        "protected_mask": protected,
        "protected_fraction": protected_fraction.detach(),
        "protected_preservation_rate": preservation_rate.detach(),
        "protected_mean_drift": mean_drift.detach(),
        "protected_max_drift": max_drift.detach(),
        "protected_excess_drift_loss": excess_drift_loss,
    }


# Purpose: Attach baseline-relative preservation supervision to detail and final-selector training.
# Called by: TrainingBackend's installed production-head loss callback.
# Calls: the previously installed production-head loss and protected_preservation_terms.
def _loss_with_protected_preservation(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _ORIGINAL_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError(
            "baseline-relative specialist contract installed without production-head loss"
        )

    losses = _ORIGINAL_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase not in {"detail-reconstruction", "physical-finetune"}:
        return losses

    baseline = outputs.get("baseline_albedo")
    target = batch.get("target_albedo")
    if not isinstance(baseline, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise RuntimeError(
            f"{phase} protected preservation requires baseline_albedo and target_albedo"
        )

    candidate_key = (
        "detail_candidate_albedo" if phase == "detail-reconstruction" else "albedo"
    )
    candidate = outputs.get(candidate_key)
    if not isinstance(candidate, torch.Tensor):
        raise RuntimeError(
            f"{phase} protected preservation requires output {candidate_key!r}"
        )

    terms = protected_preservation_terms(candidate, baseline, target)
    penalty = terms["protected_excess_drift_loss"]
    weight = (
        PROTECTED_DETAIL_LOSS_WEIGHT
        if phase == "detail-reconstruction"
        else PROTECTED_SELECTOR_LOSS_WEIGHT
    )

    losses["protected_preservation_fraction"] = terms["protected_fraction"]
    losses["protected_preservation_rate"] = terms["protected_preservation_rate"]
    losses["protected_preservation_mean_drift"] = terms["protected_mean_drift"]
    losses["protected_preservation_max_drift"] = terms["protected_max_drift"]
    losses["protected_preservation_required"] = penalty.new_tensor(
        PROTECTED_PRESERVATION_REQUIRED
    ).detach()
    losses["protected_preservation_drift_tolerance"] = penalty.new_tensor(
        PROTECTED_DRIFT_TOLERANCE
    ).detach()
    losses["protected_preservation_penalty"] = penalty
    losses["total"] = losses["total"].float() + penalty.float() * float(weight)
    return losses


# Purpose: Describe the baseline-centred parallel specialist production contract in architecture evidence.
# Called by: FidelityResidualNetV9.architecture_contract after package installation.
# Calls: the previously installed architecture contract.
def _architecture_contract_baseline_relative(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError(
            "baseline-relative specialist contract installed without architecture contract"
        )
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["baselineRelativeSpecialistRevision"] = BASELINE_RELATIVE_SPECIALIST_REVISION
    contract["baselineAnchor"] = (
        "deterministic B: bicubic albedo + normalized bilinear normal XY + nearest material"
    )
    contract["specialistComposition"] = {
        "structure": "bounded B-relative contour/geometry correction; independently qualified",
        "seamProfile": "bounded B-relative seam/profile correction; independently qualified",
        "detail": "bounded direct residual over B; independent of learned structure/seam state",
        "fusion": "only independently useful specialist evidence may enter the complete candidate",
        "finalAuthority": "BenefitSelector blends deterministic B with the complete useful candidate",
    }
    contract["protectedPreservation"] = {
        "requiredRate": PROTECTED_PRESERVATION_REQUIRED,
        "baselineErrorTolerance": PROTECTED_BASELINE_ERROR_TOLERANCE,
        "candidateDriftTolerance": PROTECTED_DRIFT_TOLERANCE,
        "trainingOnlyTargetUse": True,
        "inferenceTargetUse": False,
    }
    contract["failureIdentity"] = (
        "unsupported, zero-authority or unqualified specialist behaviour reduces to deterministic B"
    )
    return contract


# Purpose: Install model-side architecture evidence without changing parameters or checkpoint topology.
# Called by: v9 package import.
# Calls: FidelityResidualNetV9.architecture_contract replacement.
def install_baseline_relative_model_contract() -> None:
    global _MODEL_INSTALLED, _ORIGINAL_ARCHITECTURE_CONTRACT
    if _MODEL_INSTALLED:
        return
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.architecture_contract = _architecture_contract_baseline_relative
    _MODEL_INSTALLED = True


# Purpose: Install target-supervised preservation loss after V12.3 detail loss alignment.
# Called by: v9 package import.
# Calls: application.backend production-head callback replacement.
def install_baseline_relative_backend_contract() -> None:
    global _BACKEND_INSTALLED, _ORIGINAL_PRODUCTION_HEAD_LOSS
    if _BACKEND_INSTALLED:
        return

    from .application import backend

    current = backend._compute_losses_with_production_head_supervision
    if current is not _loss_with_protected_preservation:
        _ORIGINAL_PRODUCTION_HEAD_LOSS = current
        backend._compute_losses_with_production_head_supervision = (
            _loss_with_protected_preservation
        )
    _BACKEND_INSTALLED = True


# Purpose: Install both model evidence and training supervision for the V12.4 contract.
# Called by: v9 package import.
# Calls: install_baseline_relative_model_contract and install_baseline_relative_backend_contract.
def install_baseline_relative_specialist_contract() -> None:
    install_baseline_relative_model_contract()
    install_baseline_relative_backend_contract()
