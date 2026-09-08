"""Adapter around the canonical v9.training backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..config import V9Config
from ..contours import sobel_tensor


PRODUCTION_HEAD_SUPERVISION_REVISION = "V12.2.1"


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _optimal_residual_gate(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
    *,
    maximum: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares per-pixel authority for B + g(C-B), bounded for deployment."""
    delta = candidate.float() - baseline.float()
    desired = target.float() - baseline.float()
    numerator = (delta * desired).sum(dim=1, keepdim=True)
    denominator = delta.square().sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    gate = (numerator / denominator).clamp(0.0, float(maximum)).detach()
    motion = delta.abs().mean(dim=1, keepdim=True).detach()
    return gate, motion


def _edge_gradient_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    prediction_gray = prediction.float().mean(dim=1, keepdim=True)
    target_gray = target.float().mean(dim=1, keepdim=True)
    pgx, pgy = sobel_tensor(prediction_gray)
    tgx, tgy = sobel_tensor(target_gray)
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), edge_weight)


def _run_final_qualification_with_runtime_reset(
    model: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run canonical final qualification from fresh production runtime state.

    This wrapper is deliberately module-level. Windows DataLoader workers pickle
    the bound TrainingService worker initializer, which serializes the service
    instance. A nested/local function stored on that instance makes spawn fail
    before the first training batch.
    """
    TrainingBackend._prepare_production_runtime(model)
    import v9.training as training

    service = training._training_service
    canonical = type(service)._run_final_qualification
    return canonical(service, model, *args, **kwargs)


def _forward_for_phase_with_deployed_seam_authority(
    model: Any,
    batch: dict[str, torch.Tensor],
    phase: str,
    config: V9Config,
) -> dict[str, torch.Tensor]:
    """Train B4 on the exact production seam evidence it will see in B5/D.

    The canonical B4 path still injected the authored HR tangent. That teacher is
    absent as soon as gate-proof starts, changing the authority network's curvature
    and primitive-class features after B4 qualification. B4 must therefore run the
    ordinary production graph; the target orientation remains loss supervision only.
    """
    if phase == "seam-authority":
        return model(batch["input"])

    import v9.training as training

    service = training._training_service
    canonical = type(service)._forward_for_phase
    return canonical(service, model, batch, phase, config)


def _phase_lr_with_head_capacity(
    phase: str,
    config: V9Config,
    epoch: int | None = None,
) -> float:
    """Give isolated randomly-initialised heads enough update authority to prove capacity."""
    import v9.training as training

    service = training._training_service
    canonical = type(service)._phase_lr
    base = float(canonical(service, phase, config, epoch))
    multiplier = {
        # B1b now receives the exact production-render objective. A modest increase
        # is enough; topology was already established by B1a.
        "sdf-proof": 2.0,
        # These stages isolate one small head. The old 8e-5 gate-proof LR was about
        # 100x below the LR used by the specialist microproof that demonstrates
        # capacity, so 64 Micro steps could not move the head measurably.
        "seam-authority": 5.0,
        "gate-proof": 25.0,
        "detail-reconstruction": 3.0,
        "boundary-hardening": 20.0,
        "physical-finetune": 20.0,
    }.get(phase, 1.0)
    return base * multiplier


def _compute_losses_with_b1b_renderer_supervision(
    outputs: Any,
    batch: Any,
    config: V9Config,
    phase: str,
) -> dict[str, Any]:
    """Make B1b optimize the exact raw production geometry candidate Micro scores.

    The old adapter added ``boundary_photometric`` from the teacher-gated same-
    renderer path. Micro V2 instead judges ``boundary_initial_candidate_albedo``.
    On the failing Raven patch those two objectives disagreed badly: the structural
    telemetry was only about -1.3% while the deployable raw candidate was -28.8%
    globally / -8.3% on edges. Train the candidate that actually has to qualify.
    """
    import v9.training as training

    base = getattr(training, "_nsamdr_b1b_base_compute_losses", None)
    if base is None:
        raise RuntimeError("B1b renderer supervision installed without base compute_losses")

    losses = base(outputs, batch, config, phase)
    if phase != "sdf-proof":
        return losses

    required = (
        "boundary_initial_candidate_albedo",
        "baseline_albedo",
    )
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(f"sdf-proof production supervision missing outputs: {missing}")

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["boundary_initial_candidate_albedo"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    edge_weight = (0.20 + edge * 3.80).detach()

    global_reconstruction = candidate_error.mean()
    edge_reconstruction = _weighted_mean(candidate_error, edge_weight)
    regret = _weighted_mean(F.relu(candidate_error - baseline_error), edge_weight)
    gradient = _edge_gradient_error(candidate, target, edge_weight)

    render_weight = float(getattr(config, "spline_graph_render_weight", 96.0))
    gradient_weight = float(getattr(config, "spline_graph_render_gradient_weight", 48.0))
    production_objective = (
        global_reconstruction * render_weight * 0.35
        + edge_reconstruction * render_weight * 0.65
        + regret * render_weight
        + gradient * gradient_weight
    ).float()

    losses["b1b_production_global_reconstruction"] = global_reconstruction
    losses["b1b_production_edge_reconstruction"] = edge_reconstruction
    losses["b1b_production_regret"] = regret
    losses["b1b_production_gradient"] = gradient
    losses["b1b_production_global_recovery"] = (
        (baseline_error.mean() - candidate_error.mean())
        / baseline_error.mean().clamp_min(1.0e-6)
    ).detach()
    losses["b1b_production_edge_recovery"] = (
        (_weighted_mean(baseline_error, edge_weight) - edge_reconstruction)
        / _weighted_mean(baseline_error, edge_weight).clamp_min(1.0e-6)
    ).detach()
    losses["b1b_renderer_supervision"] = production_objective.detach()

    # Preserve topology/metric regularisation, but stop its large raw numerical
    # scale from drowning the production renderer gradient.
    losses["total"] = losses["total"].float() * 0.10 + production_objective
    return losses


def _install_b1b_renderer_supervision(training: Any) -> None:
    """Install the module-level B1b loss adapter once without touching worker state."""
    if bool(getattr(training, "_nsamdr_b1b_renderer_supervision_installed", False)):
        training.compute_losses = _compute_losses_with_b1b_renderer_supervision
        return

    training._nsamdr_b1b_base_compute_losses = training.compute_losses
    training.compute_losses = _compute_losses_with_b1b_renderer_supervision
    training._nsamdr_b1b_renderer_supervision_installed = True


def _compute_losses_with_production_head_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V9Config,
    phase: str,
) -> dict[str, torch.Tensor]:
    """Train B4/detail/selector heads on the exact candidates they deploy."""
    import v9.training as training

    base = getattr(training, "_nsamdr_production_head_base_compute_losses", None)
    if base is None:
        raise RuntimeError("production-head supervision installed without base compute_losses")
    losses = base(outputs, batch, config, phase)

    target = batch["target_albedo"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    edge_weight = (0.20 + edge * 3.80).detach()

    if phase == "seam-authority":
        before = outputs["boundary_pre_seam_albedo"].detach().float()
        phase_delta = outputs["seam_phase_delta"][:, 0:3].detach().float()
        proposal = (before + phase_delta).clamp(0.0, 1.0)
        maximum = max(float(getattr(config, "seam_max_authority", 0.90)), 1.0e-4)
        effective_oracle, motion = _optimal_residual_gate(
            before, proposal, target, maximum=maximum
        )
        oracle = (effective_oracle / maximum).clamp(0.0, 1.0).detach()
        learned = outputs["seam_learned_authority"].float().clamp(1.0e-5, 1.0 - 1.0e-5)

        authority_weight = (
            0.10
            + edge * 2.90
            + (motion / 0.02).clamp(0.0, 1.0)
        ).detach()
        authority_bce = _weighted_mean(
            F.binary_cross_entropy(learned, oracle, reduction="none"),
            authority_weight,
        )
        authority_l1 = _weighted_mean((learned - oracle).abs(), authority_weight)

        deployed = outputs["boundary_reconstructed_albedo"].float()
        deployed_error = (deployed - target).abs().mean(dim=1, keepdim=True)
        before_error = (before - target).abs().mean(dim=1, keepdim=True)
        deployed_reconstruction = _weighted_mean(deployed_error, edge_weight)
        deployed_regret = _weighted_mean(
            F.relu(deployed_error - before_error), edge_weight
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
            (_weighted_mean(before_error, edge_weight) - deployed_reconstruction)
            / _weighted_mean(before_error, edge_weight).clamp_min(1.0e-6)
        ).detach()
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
            + losses["seam_authority_regularization"]
            * float(getattr(config, "seam_authority_regularization_weight", 0.20))
        ).float()
        return losses

    if phase == "detail-reconstruction":
        before = outputs["boundary_reconstructed_albedo"].detach().float()
        candidate = outputs["detail_candidate_albedo"].float()
        candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
        before_error = (before - target).abs().mean(dim=1, keepdim=True)

        reconstruction = _weighted_mean(candidate_error, edge_weight)
        regret = _weighted_mean(
            F.relu(candidate_error - before_error), edge_weight
        )
        global_regret = F.relu(candidate_error - before_error).mean()
        gradient = _edge_gradient_error(candidate, target, edge_weight)

        losses["detail_deployed_reconstruction"] = reconstruction
        losses["detail_deployed_regret"] = regret
        losses["detail_deployed_global_regret"] = global_regret
        losses["detail_deployed_gradient"] = gradient
        losses["detail_deployed_edge_recovery"] = (
            (_weighted_mean(before_error, edge_weight) - reconstruction)
            / _weighted_mean(before_error, edge_weight).clamp_min(1.0e-6)
        ).detach()
        losses["total"] = (
            losses["total"].float()
            + reconstruction * 8.0
            + gradient * 5.0
            + regret * 24.0
            + global_regret * 12.0
        )
        return losses

    if phase not in {"boundary-hardening", "physical-finetune"}:
        return losses

    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["detail_candidate_albedo"].detach().float()
    final = outputs["albedo"].float()
    probability = outputs["benefit_selector_probability"].float().clamp(1.0e-5, 1.0 - 1.0e-5)
    oracle, motion = _optimal_residual_gate(baseline, candidate, target)
    selector_weight = (
        0.10
        + edge * 2.90
        + (motion / 0.02).clamp(0.0, 1.0)
    ).detach()
    selector_loss = _weighted_mean(
        F.binary_cross_entropy(probability, oracle, reduction="none"),
        selector_weight,
    )

    final_error = (final - target).abs().mean(dim=1, keepdim=True)
    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    final_reconstruction = _weighted_mean(final_error, edge_weight)
    final_regret = _weighted_mean(F.relu(final_error - baseline_error), edge_weight)
    final_gradient = _edge_gradient_error(final, target, edge_weight)

    losses["selector_optimal_gate"] = selector_loss
    losses["boundary_gate"] = selector_loss
    losses["gate_target"] = oracle.mean().detach()
    losses["selector_optimal_gate_mean"] = oracle.mean().detach()
    losses["selector_edge_reconstruction"] = final_reconstruction
    losses["selector_edge_regret"] = final_regret
    losses["selector_edge_gradient"] = final_gradient
    losses["total"] = (
        losses["total"].float() * 0.25
        + selector_loss * float(getattr(config, "benefit_selector_weight", 8.0))
        + final_reconstruction * float(getattr(config, "albedo_weight", 1.0)) * 2.0
        + final_gradient * float(getattr(config, "albedo_gradient_weight", 1.0))
        + final_regret * float(getattr(config, "regret_weight", 1.0)) * 3.0
    ).float()
    return losses


def _install_production_head_supervision(training: Any) -> None:
    """Install exact deployed-candidate objectives after V12.2 authority alignment."""
    if bool(getattr(training, "_nsamdr_production_head_supervision_installed", False)):
        training.compute_losses = _compute_losses_with_production_head_supervision
        return

    training._nsamdr_production_head_base_compute_losses = training.compute_losses
    training.compute_losses = _compute_losses_with_production_head_supervision
    training._nsamdr_production_head_supervision_installed = True


class TrainingBackend:
    """Own installation and invocation of the current production trainer contract."""

    @staticmethod
    def _prepare_production_runtime(model: Any) -> None:
        """Clear training-only forward state before production qualification.

        B1a intentionally sets ``_topology_bootstrap_only`` so its forward proves
        topology without invoking the explicit continuous geometry refiner. That
        flag is runtime-only and is absent from the state dict; a freshly loaded
        production model therefore starts with it disabled. Final qualification
        reuses the training model object, so it must reproduce that fresh-load
        runtime state before auditing production-component participation.
        """
        geometry = getattr(model, "geometry_net", None)
        structure = getattr(geometry, "production_structure", None)
        if structure is None:
            return
        if hasattr(structure, "_topology_bootstrap_only"):
            structure._topology_bootstrap_only = False

    def _synchronize_training_service_contract(self, training: Any) -> None:
        """Copy patched compatibility callbacks onto the object that owns train_v9.

        All installed callbacks are module-level so Windows spawn can pickle the
        TrainingService singleton reached through the DataLoader worker initializer.
        """
        service = getattr(training, "_training_service", None)
        if service is None:
            raise RuntimeError("NSAMDR training module has no TrainingService singleton")
        service._validate_v992_architecture_contract = training._validate_v992_architecture_contract
        service._explicit_primitive_structure_microproof = training._explicit_primitive_structure_microproof
        service._production_component_modules = training._production_component_modules

        if service._run_final_qualification is not _run_final_qualification_with_runtime_reset:
            service._run_final_qualification = _run_final_qualification_with_runtime_reset
        if service._forward_for_phase is not _forward_for_phase_with_deployed_seam_authority:
            service._forward_for_phase = _forward_for_phase_with_deployed_seam_authority
        if service._phase_lr is not _phase_lr_with_head_capacity:
            service._phase_lr = _phase_lr_with_head_capacity

    def __init__(self) -> None:
        """Install the current production model/loss/trainer contracts."""
        import v9.training as training
        from ..authority_alignment_contract import install_authority_alignment_contract
        from ..local_boundary_production_contract import install_local_boundary_training_contract

        install_local_boundary_training_contract(training)
        _install_b1b_renderer_supervision(training)
        install_authority_alignment_contract(training)
        _install_production_head_supervision(training)
        self._synchronize_training_service_contract(training)
        self._trainer = training.train_v9

    def run(
        self,
        config: V9Config,
        repo_root: Path,
        device: str,
        *,
        resume: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
        stop_after_phase: str | None,
    ) -> dict[str, Any]:
        """Invoke the unchanged canonical trainer with one explicit stage boundary."""
        return self._trainer(
            config,
            repo_root,
            device,
            resume=resume,
            restart=False,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            stop_after_phase=stop_after_phase,
        )
