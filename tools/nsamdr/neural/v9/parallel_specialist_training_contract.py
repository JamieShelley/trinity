from __future__ import annotations

"""V12.5 training alignment for independent baseline-relative specialist fusion.

V12.5 changes production composition from the historical serial path to:

    B -> independent G / S / D -> support-weighted U -> BenefitSelector -> F

V12.3 already owns the direct B-relative detail objective and V12.4 owns protected
preservation. This contract only realigns the remaining deployed-authority losses:

* seam authority is supervised against the B-relative seam proposal it deploys;
* BenefitSelector is supervised against fused candidate U, not detail candidate D;
* V12.4 remains the outermost protected-preservation wrapper.

No parameters, checkpoint keys or inference inputs are changed.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9


PARALLEL_SPECIALIST_TRAINING_REVISION = "V12.5"

_INSTALLED = False
_PREVIOUS_PRODUCTION_HEAD_LOSS: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None


# Purpose: Retrieve the pre-production-head base objective so stale selector terms cannot leak in.
# Called by: V12.5 selector supervision.
# Calls: the base compute_losses callback captured by application.backend.
def _production_base_total(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> torch.Tensor:
    import v9.training as training

    base = getattr(training, "_nsamdr_production_head_base_compute_losses", None)
    if base is None:
        raise RuntimeError("V12.5 selector alignment missing production-head base loss")
    base_losses = base(outputs, batch, config, phase)
    total = base_losses.get("total")
    if not isinstance(total, torch.Tensor):
        raise RuntimeError("V12.5 selector alignment base loss has no tensor total")
    return total.float()


# Purpose: Train seam/selector authority against the exact V12.5 candidates they deploy.
# Called by: V12.4 protected-preservation wrapper through its captured inner callback.
# Calls: the previously installed V12.3 production-head loss plus backend loss helpers.
def _compute_losses_with_parallel_specialist_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _PREVIOUS_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError("parallel specialist training installed without previous loss")

    losses = _PREVIOUS_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase not in {"seam-authority", "boundary-hardening", "physical-finetune"}:
        return losses

    from .application import backend

    target = batch["target_albedo"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    edge_weight = (0.20 + edge * 3.80).detach()
    baseline = outputs["baseline_albedo"].detach().float()

    if phase == "seam-authority":
        required = (
            "seam_phase_delta",
            "seam_learned_authority",
            "seam_candidate_albedo",
        )
        missing = [key for key in required if key not in outputs]
        if missing:
            raise RuntimeError(
                f"V12.5 seam-authority supervision missing outputs: {missing}"
            )

        phase_delta = outputs["seam_phase_delta"][:, 0:3].detach().float()
        proposal = (baseline + phase_delta).clamp(0.0, 1.0)
        maximum = max(float(getattr(config, "seam_max_authority", 0.90)), 1.0e-4)
        effective_oracle, motion = backend._optimal_residual_gate(
            baseline,
            proposal,
            target,
            maximum=maximum,
        )
        oracle = (effective_oracle / maximum).clamp(0.0, 1.0).detach()
        learned = outputs["seam_learned_authority"].float().clamp(
            1.0e-5, 1.0 - 1.0e-5
        )

        authority_weight = (
            0.10
            + edge * 2.90
            + (motion / 0.02).clamp(0.0, 1.0)
        ).detach()
        authority_bce = backend._weighted_mean(
            F.binary_cross_entropy(learned, oracle, reduction="none"),
            authority_weight,
        )
        authority_l1 = backend._weighted_mean(
            (learned - oracle).abs(), authority_weight
        )

        deployed = outputs["seam_candidate_albedo"].float()
        deployed_error = (deployed - target).abs().mean(dim=1, keepdim=True)
        baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
        deployed_reconstruction = backend._weighted_mean(deployed_error, edge_weight)
        deployed_regret = backend._weighted_mean(
            F.relu(deployed_error - baseline_error), edge_weight
        )

        support_threshold = 0.20
        predicted_support = learned >= support_threshold
        oracle_support = oracle >= support_threshold
        intersection = (predicted_support & oracle_support).float().sum()
        union = (predicted_support | oracle_support).float().sum()
        authority_iou = torch.where(
            union > 0.0,
            intersection / union.clamp_min(1.0),
            torch.ones_like(union),
        ).detach()

        losses["seam_authority_teacher"] = authority_bce
        losses["seam_authority_oracle_l1"] = authority_l1
        losses["seam_authority_iou"] = authority_iou
        losses["seam_teacher_mean"] = oracle.mean().detach()
        losses["seam_authority_oracle_mean"] = oracle.mean().detach()
        losses["seam_deployed_reconstruction"] = deployed_reconstruction
        losses["seam_deployed_regret"] = deployed_regret
        losses["seam_deployed_recovery"] = (
            (backend._weighted_mean(baseline_error, edge_weight) - deployed_reconstruction)
            / backend._weighted_mean(baseline_error, edge_weight).clamp_min(1.0e-6)
        ).detach()

        regularization = losses.get("seam_authority_regularization")
        if not isinstance(regularization, torch.Tensor):
            regularization = deployed_reconstruction.new_zeros(())
        losses["total"] = (
            deployed_reconstruction
            * float(getattr(config, "seam_reconstruction_weight", 28.0))
            + deployed_regret
            * float(getattr(config, "seam_authority_teacher_weight", 18.0))
            + authority_bce
            * float(getattr(config, "seam_authority_teacher_weight", 18.0))
            + authority_l1
            * float(getattr(config, "seam_authority_teacher_weight", 18.0))
            * 0.50
            + regularization
            * float(getattr(config, "seam_authority_regularization_weight", 0.20))
        ).float()
        return losses

    required = (
        "fused_candidate_albedo",
        "benefit_selector_probability",
        "albedo",
    )
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(f"V12.5 selector supervision missing outputs: {missing}")

    candidate = outputs["fused_candidate_albedo"].detach().float()
    final = outputs["albedo"].float()
    probability = outputs["benefit_selector_probability"].float().clamp(
        1.0e-5, 1.0 - 1.0e-5
    )
    oracle, motion = backend._optimal_residual_gate(baseline, candidate, target)
    selector_weight = (
        0.10
        + edge * 2.90
        + (motion / 0.02).clamp(0.0, 1.0)
    ).detach()
    selector_loss = backend._weighted_mean(
        F.binary_cross_entropy(probability, oracle, reduction="none"),
        selector_weight,
    )

    final_error = (final - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    final_reconstruction = backend._weighted_mean(final_error, edge_weight)
    final_regret = backend._weighted_mean(
        F.relu(final_error - baseline_error), edge_weight
    )
    final_gradient = backend._edge_gradient_error(final, target, edge_weight)

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    candidate_edge = backend._weighted_mean(candidate_error, edge_weight)
    baseline_edge = backend._weighted_mean(baseline_error, edge_weight)

    losses["selector_optimal_gate"] = selector_loss
    losses["boundary_gate"] = selector_loss
    losses["gate_target"] = oracle.mean().detach()
    losses["selector_optimal_gate_mean"] = oracle.mean().detach()
    losses["selector_edge_reconstruction"] = final_reconstruction
    losses["selector_edge_regret"] = final_regret
    losses["selector_edge_gradient"] = final_gradient
    losses["selector_fused_candidate_edge_recovery"] = (
        (baseline_edge - candidate_edge) / baseline_edge.clamp_min(1.0e-6)
    ).detach()
    losses["total"] = (
        _production_base_total(outputs, batch, config, phase) * 0.25
        + selector_loss * float(getattr(config, "benefit_selector_weight", 8.0))
        + final_reconstruction * float(getattr(config, "albedo_weight", 1.0)) * 2.0
        + final_gradient * float(getattr(config, "albedo_gradient_weight", 1.0))
        + final_regret * float(getattr(config, "regret_weight", 1.0)) * 3.0
    ).float()
    return losses


# Purpose: Record the installed V12.5 training authority in architecture diagnostics.
# Called by: FidelityResidualNetV9.architecture_contract.
# Calls: the previously installed V12.5 architecture contract.
def _architecture_contract_with_parallel_training(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("parallel specialist training installed without architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelSpecialistTrainingRevision"] = PARALLEL_SPECIALIST_TRAINING_REVISION
    contract["specialistTrainingAuthority"] = {
        "structure": "existing B-relative rendered structural objective",
        "seamProfile": "B-relative seam proposal and deployed seam candidate",
        "detail": "V12.3 direct B-relative detail objective",
        "finalSelector": "oracle B-versus-fused-U authority",
        "protectedPreservation": "V12.4 remains outermost training wrapper",
    }
    return contract


# Purpose: Install V12.5 loss alignment without bypassing V12.4 preservation.
# Called by: v9 package import after parallel specialist fusion installation.
# Calls: baseline_relative_specialist_contract and application.backend patch points.
def install_parallel_specialist_training_contract() -> None:
    global _INSTALLED, _PREVIOUS_PRODUCTION_HEAD_LOSS, _ORIGINAL_ARCHITECTURE_CONTRACT
    if _INSTALLED:
        return

    from . import baseline_relative_specialist_contract as preservation
    from .application import backend

    current_inner = preservation._ORIGINAL_PRODUCTION_HEAD_LOSS
    if current_inner is None:
        raise RuntimeError(
            "V12.5 training alignment requires V12.4 preservation to install first"
        )
    if current_inner is not _compute_losses_with_parallel_specialist_supervision:
        _PREVIOUS_PRODUCTION_HEAD_LOSS = current_inner
        preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = (
            _compute_losses_with_parallel_specialist_supervision
        )

    # TrainingBackend installs this global callback into v9.training when it is
    # instantiated. Keep V12.4 as the outer wrapper so protected-preservation is
    # never bypassed by V12.5 alignment.
    backend._compute_losses_with_production_head_supervision = (
        preservation._loss_with_protected_preservation
    )

    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.architecture_contract = (
        _architecture_contract_with_parallel_training
    )
    _INSTALLED = True
