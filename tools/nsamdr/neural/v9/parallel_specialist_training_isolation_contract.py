from __future__ import annotations

"""V12.7 staged-training isolation for baseline-relative specialists.

The V12.6 full staged Micro run proved that representation and isolated integration
were no longer the blocker: Direct Residual and Parallel Detail passed, while the
same detail decoder stayed effectively at identity when trained inside the complete
curriculum. The run exposed two training-only authority leaks:

* detail confidence/regret heads shared decoder features with the residual heads, so
  their classification losses could reshape the candidate backbone while D was
  trying to reproduce the independently proven direct-residual solution;
* B1b supervised the raw boundary redraw rather than the deployed B-relative
  structure candidate G, leaving structural_residual_gain without an exact deployed
  benefit objective.

V12.7 changes no inference values, parameters, checkpoint keys or input contract.
It detaches the shared detail feature only on the path into confidence/regret heads,
trains those support heads from B-relative D benefit/regret targets, and makes B1b's
rendered objective score the deployed structure candidate G.
"""

from typing import Any

import torch
from torch.nn import functional as F

from .model import FidelityResidualNetV9, GeometryConditionedDetailNet


PARALLEL_SPECIALIST_TRAINING_ISOLATION_REVISION = "V12.7"

_INSTALLED = False
_ORIGINAL_DETAIL_DIRECT_FORWARD: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None
_PREVIOUS_PRODUCTION_HEAD_LOSS: Any = None


# Purpose: Preserve the direct-detail residual feature path while isolating support-head gradients.
# Called by: V12.3 _detail_forward_from_baseline through its captured original callback.
# Calls: the existing GeometryConditionedDetailNet layers only.
def _detail_forward_with_isolated_support_heads(
    self: GeometryConditionedDetailNet,
    inputs: torch.Tensor,
    base_albedo_hr: torch.Tensor,
    base_normal_hr: torch.Tensor,
    base_material_hr: torch.Tensor,
    geometry_condition_hr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    lr_size = inputs.shape[-2:]
    geometry_lr = self._resize(geometry_condition_hr.float(), lr_size)
    x = self.encoder(torch.cat((inputs.float(), geometry_lr), dim=1))

    physical_hr = torch.cat(
        (
            base_albedo_hr.float(),
            base_normal_hr.float(),
            base_material_hr.float(),
        ),
        dim=1,
    )
    size2 = (lr_size[0] * 2, lr_size[1] * 2)
    x = self._resize(x, size2)
    x = self.up2_pre(x)
    x = self.up2_fuse(
        torch.cat(
            (
                x,
                self._resize(physical_hr, size2),
                self._resize(geometry_condition_hr.float(), size2),
            ),
            dim=1,
        )
    )
    x = self.up2_body(x)

    hr_size = base_albedo_hr.shape[-2:]
    x = self._resize(x, hr_size)
    x = self.up4_pre(x)
    x = self.up4_fuse(
        torch.cat((x, physical_hr, geometry_condition_hr.float()), dim=1)
    )
    x = self.up4_body(x)

    # Detach only the support-classification branch. The tensor value is unchanged,
    # so inference is numerically identical; only confidence/regret gradients are
    # prevented from rewriting the residual reconstruction backbone.
    support_x = x.detach()
    return {
        "albedo_raw": torch.tanh(self.albedo_head(x)),
        "normal_raw": torch.tanh(self.normal_head(x)),
        "material_raw": torch.tanh(self.material_head(x)),
        "confidence_logits": self.confidence_head(support_x),
        "regret_logits": self.regret_head(support_x),
    }


# Purpose: Compute the exact B-relative support labels for the independently trained D candidate.
# Called by: _loss_with_b_relative_detail_support.
# Calls: tensor reductions only.
def _detail_support_targets(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["detail_candidate_albedo"].detach().float()
    target = batch["target_albedo"].detach().float()

    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    improvement = (baseline_error - candidate_error).detach()
    scale = max(float(getattr(config, "gate_error_scale", 0.08)), 1.0e-4)
    confidence_target = (0.5 + improvement / scale).clamp(0.0, 1.0)
    regret_target = (
        candidate_error > baseline_error + 0.001
    ).float().detach()
    return confidence_target, regret_target


# Purpose: Replace stale serial support supervision while preserving the V12.6 loss chain.
# Called by: V12.4 protected-preservation outer wrapper.
# Calls: previous V12.6 loss, _detail_support_targets and BCE-with-logits.
def _loss_with_b_relative_detail_support(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    if _PREVIOUS_PRODUCTION_HEAD_LOSS is None:
        raise RuntimeError("V12.7 detail support installed without previous loss")

    losses = _PREVIOUS_PRODUCTION_HEAD_LOSS(outputs, batch, config, phase)
    if phase != "detail-reconstruction":
        return losses

    confidence_logits = outputs.get("detail_confidence_logits")
    regret_logits = outputs.get("detail_regret_logits")
    if not isinstance(confidence_logits, torch.Tensor) or not isinstance(
        regret_logits, torch.Tensor
    ):
        raise RuntimeError("V12.7 detail support requires confidence/regret logits")

    confidence_target, regret_target = _detail_support_targets(outputs, batch, config)
    confidence_loss = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), confidence_target.float()
    )
    regret_loss = F.binary_cross_entropy_with_logits(
        regret_logits.float(), regret_target.float()
    )
    confidence_weight = float(getattr(config, "detail_confidence_weight", 0.20))
    regret_weight = float(getattr(config, "detail_regret_classifier_weight", 0.40))
    corrected_support = (
        confidence_loss * confidence_weight + regret_loss * regret_weight
    ).float()

    # V12.6 added its support term using the legacy serial labels. Remove that exact
    # graph contribution before installing the B-relative replacement.
    previous_support = losses.get("detail_support_calibration")
    if isinstance(previous_support, torch.Tensor):
        losses["total"] = losses["total"].float() - previous_support.float()

    losses["detail_confidence"] = confidence_loss
    losses["detail_regret_classifier"] = regret_loss
    losses["detail_support_confidence"] = confidence_loss
    losses["detail_support_regret"] = regret_loss
    losses["detail_support_confidence_target"] = confidence_target.mean().detach()
    losses["detail_support_regret_target"] = regret_target.mean().detach()
    losses["detail_support_calibration"] = corrected_support
    losses["total"] = losses["total"].float() + corrected_support
    return losses


# Purpose: Score the structure specialist exactly as deployed: deterministic B versus G.
# Called by: b1_production_objective_contract._production_only_b1b_loss.
# Calls: that contract's observable support / weighted metric helpers.
def _live_deployed_structure_objective(
    outputs: Any,
    batch: Any,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from . import b1_production_objective_contract as b1

    required = ("structure_candidate_albedo", "baseline_albedo")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(
            f"V12.7 sdf-proof deployed-structure supervision missing outputs: {missing}"
        )

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["structure_candidate_albedo"].float()
    support = b1._observable_structural_support(outputs, batch, config)

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    candidate_structural = b1._weighted_mean(candidate_error, support)
    baseline_structural = b1._weighted_mean(baseline_error, support)
    regret = b1._weighted_mean(F.relu(candidate_error - baseline_error), support)
    gradient = b1._edge_gradient_error(candidate, target, support)

    off_support = (1.0 - support).detach()
    identity = b1._weighted_mean(
        (candidate - baseline).abs().mean(dim=1, keepdim=True),
        off_support,
    )
    render_weight = float(getattr(config, "spline_graph_render_weight", 96.0))
    gradient_weight = float(
        getattr(config, "spline_graph_render_gradient_weight", 48.0)
    )
    production = (
        candidate_structural * render_weight
        + regret * render_weight
        + gradient * gradient_weight
        + identity * render_weight * 0.50
    ).float()
    if not production.requires_grad or production.grad_fn is None:
        raise RuntimeError("V12.7 deployed G objective is detached from structure training")

    telemetry = {
        "b1b_structural_support_mean": support.mean().detach(),
        "b1b_structural_reconstruction": candidate_structural.detach(),
        "b1b_structural_baseline": baseline_structural.detach(),
        "b1b_structural_recovery": (
            (baseline_structural - candidate_structural)
            / baseline_structural.clamp_min(1.0e-6)
        ).detach(),
        "b1b_structural_regret": regret.detach(),
        "b1b_structural_gradient": gradient.detach(),
        "b1b_off_support_identity": identity.detach(),
        "b1b_deployed_structure_objective": production.detach(),
    }
    return production, telemetry


# Purpose: Publish the corrected training authority in the executable architecture contract.
# Called by: FidelityResidualNetV9.architecture_contract.
# Calls: the previous V12.6 architecture contract.
def _architecture_contract_with_training_isolation(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V12.7 training isolation missing architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelSpecialistTrainingIsolationRevision"] = (
        PARALLEL_SPECIALIST_TRAINING_ISOLATION_REVISION
    )
    contract["structureTrainingAuthority"] = (
        "deployed B-relative structure candidate G, including structural residual gain"
    )
    contract["detailSupportTraining"] = (
        "confidence/regret labels are computed from D-versus-B benefit; support-head gradients do not alter the residual backbone"
    )
    contract["detailSupportBackboneAuthority"] = False
    return contract


# Purpose: Install V12.7 after V12.6 without changing inference or checkpoint topology.
# Called by: v9 package import.
# Calls: V12.3 direct-forward callback, B1 objective and V12.4 preservation chain.
def install_parallel_specialist_training_isolation_contract() -> None:
    global _INSTALLED, _ORIGINAL_DETAIL_DIRECT_FORWARD
    global _ORIGINAL_ARCHITECTURE_CONTRACT, _PREVIOUS_PRODUCTION_HEAD_LOSS
    if _INSTALLED:
        return

    from . import parallel_detail_contract as detail_contract
    from . import b1_production_objective_contract as b1
    from . import baseline_relative_specialist_contract as preservation
    from .application import backend

    current_direct = detail_contract._ORIGINAL_DETAIL_FORWARD
    if current_direct is None:
        raise RuntimeError("V12.7 requires the V12.3 direct-detail adapter")
    _ORIGINAL_DETAIL_DIRECT_FORWARD = current_direct
    detail_contract._ORIGINAL_DETAIL_FORWARD = (
        _detail_forward_with_isolated_support_heads
    )

    b1._live_production_objective = _live_deployed_structure_objective

    current_inner = preservation._ORIGINAL_PRODUCTION_HEAD_LOSS
    if current_inner is None:
        raise RuntimeError("V12.7 requires the V12.4/V12.6 loss chain")
    _PREVIOUS_PRODUCTION_HEAD_LOSS = current_inner
    preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = (
        _loss_with_b_relative_detail_support
    )
    backend._compute_losses_with_production_head_supervision = (
        preservation._loss_with_protected_preservation
    )

    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.architecture_contract = (
        _architecture_contract_with_training_isolation
    )
    _INSTALLED = True
