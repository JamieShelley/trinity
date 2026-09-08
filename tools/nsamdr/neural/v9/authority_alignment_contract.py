from __future__ import annotations

"""V12.2 authority/optimization alignment for the production NSAMDR graph.

The production network already contains useful structural, seam and detail proposals,
but several later heads were trained on signals that did not match what they actually
controlled at inference. This contract makes training and deployment agree:

* B1b renders the refined connected spline with full positive structural authority;
  the retired signed gain cannot invert a bad geometry proposal.
* The explicit spline refiner is first-order differentiable during outer training, so
  renderer loss on the refined geometry reaches the neural spline initializer.
  Inference retains the exact original parameter-free detached optimizer.
* The proven PhaseAwareSeamSR proposal is also the deployed seam proposal, and one
  learned seam-authority head directly controls it.
* BenefitSelector is the one final residual authority. Detail confidence and regret
  remain useful selector features/auxiliary heads rather than independent vetoes.
* Selector supervision is computed from the exact full candidate it gates and Stage D
  receives differentiable loss on the actual final output.

The model-side contract is installed by :mod:`v9` itself so preflight, inference and
training execute the same graph. Training-loss adapters are installed separately by
TrainingBackend. All adapters are module-level to remain Windows-spawn safe.
"""

import math
from typing import Any

import torch
from torch.nn import functional as F

from .contours import sobel_tensor
from .explicit_spline_refiner import ExplicitSplineGeometryRefiner
from .local_boundary_production_contract import LocalBoundaryProductionStructure
from .model import FidelityResidualNetV9
from .seam_restoration import DirectionalSeamRestorer


AUTHORITY_ALIGNMENT_REVISION = "V12.2"
B1B_STRUCTURAL_LATCH = 0.999
_MODEL_INSTALLED = False

_ORIGINAL_FORWARD_IMPL: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None
_ORIGINAL_SEAM_FORWARD: Any = None
_ORIGINAL_REFINER_FORWARD: Any = None
_ORIGINAL_LOCK_TOPOLOGY_FOR_PROOF: Any = None


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _nonnegative_structural_residual_weight(
    candidate_locality: torch.Tensor,
    structural_residual_gain: torch.Tensor | None,
    gate_override: torch.Tensor | None,
) -> torch.Tensor:
    """Use structural gain only as the B1a/B1b phase latch, never inversion authority."""
    if gate_override is not None:
        weight = gate_override.to(
            device=candidate_locality.device, dtype=torch.float32, non_blocking=True
        ).clamp(0.0, 1.0)
        if weight.shape[-2:] != candidate_locality.shape[-2:]:
            weight = F.interpolate(
                weight, size=candidate_locality.shape[-2:], mode="bilinear", align_corners=False
            )
        return weight * candidate_locality.float()

    if structural_residual_gain is None:
        return torch.zeros_like(candidate_locality, dtype=torch.float32)
    gain = structural_residual_gain.to(
        device=candidate_locality.device, dtype=torch.float32, non_blocking=True
    )
    if gain.shape[-2:] != candidate_locality.shape[-2:]:
        gain = F.interpolate(
            gain, size=candidate_locality.shape[-2:], mode="bilinear", align_corners=False
        )
    # B1a is exactly zero. B1b stores a positive checkpointed latch and therefore
    # must make the raw refined geometry itself beat B. The final selector is the
    # later safety authority; B1 cannot pass by learning a negative inverse blend.
    active = (gain > 0.5).to(dtype=torch.float32)
    return active * candidate_locality.float()


def _lock_topology_for_proof_with_structural_latch(
    self: LocalBoundaryProductionStructure,
) -> None:
    """Enter B1b with full positive raw-geometry authority and freeze the old gain head."""
    if _ORIGINAL_LOCK_TOPOLOGY_FOR_PROOF is None:
        raise RuntimeError("authority alignment installed without original B1b topology lock")
    _ORIGINAL_LOCK_TOPOLOGY_FOR_PROOF(self)
    head = self.structural_residual_gain_head[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(math.atanh(B1B_STRUCTURAL_LATCH))
    for parameter in self.structural_residual_gain_head.parameters():
        parameter.requires_grad_(False)


def _refiner_training_path_active(
    self: ExplicitSplineGeometryRefiner,
    proposal_graph: dict[str, torch.Tensor],
) -> bool:
    if not self.training or not torch.is_grad_enabled():
        return False
    return any(
        isinstance(proposal_graph.get(key), torch.Tensor)
        and bool(proposal_graph[key].requires_grad)
        for key in (
            "spline_control_point_h_lr",
            "spline_control_point_v_lr",
            "spline_control_tangent_h",
            "spline_control_tangent_v",
        )
    )


def _refiner_forward_with_outer_gradient(
    self: ExplicitSplineGeometryRefiner,
    spline_graph: Any,
    proposal_graph: dict[str, torch.Tensor],
    source_sdf_lr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """First-order unroll of the production refiner for B1b outer supervision.

    Inner gradients are detached, avoiding second-order differentiation, while each
    updated continuous state retains its direct proposal dependency. Hence loss on
    the exact refined/rendered graph reaches the neural initializer. Eval/inference
    calls the original detached production optimizer and is numerically unchanged.
    """
    if not _refiner_training_path_active(self, proposal_graph):
        if _ORIGINAL_REFINER_FORWARD is None:
            raise RuntimeError("authority alignment installed without original refiner forward")
        return _ORIGINAL_REFINER_FORWARD(self, spline_graph, proposal_graph, source_sdf_lr)

    proposal_h = proposal_graph["spline_control_point_h_lr"].float()
    proposal_v = proposal_graph["spline_control_point_v_lr"].float()
    tangent_h = proposal_graph["spline_control_tangent_h"].float()
    tangent_v = proposal_graph["spline_control_tangent_v"].float()
    mask_h = proposal_graph["spline_graph_mask_h"][:, 0].detach().float()
    mask_v = proposal_graph["spline_graph_mask_v"][:, 0].detach().float()
    source = source_sdf_lr.detach().float()

    def finish_noop() -> dict[str, torch.Tensor]:
        graph = self._graph_with_geometry(
            proposal_graph,
            proposal_h[..., 0], proposal_v[..., 1], tangent_h, tangent_v,
        )
        zero = source_sdf_lr.new_zeros(())
        graph["spline_refiner_energy_before"] = zero
        graph["spline_refiner_energy_after"] = zero
        graph["spline_refiner_node_shift_rms_pixels"] = zero
        graph["spline_refiner_steps"] = zero
        graph["spline_refiner_node_source_error"] = zero
        graph["spline_refiner_span_source_error"] = zero
        return graph

    if not self.enabled or self.steps <= 0:
        return finish_noop()
    if float(mask_h.sum().detach().cpu()) + float(mask_v.sum().detach().cpu()) <= 0.0:
        return finish_noop()

    h_x = proposal_h[..., 0]
    v_y = proposal_v[..., 1]
    h_reference = proposal_h[..., 0].detach()
    v_reference = proposal_v[..., 1].detach()
    h_edge_start = torch.floor(h_reference)
    v_edge_start = torch.floor(v_reference)
    max_move_lattice = self.max_move_pixels / max(float(spline_graph.spacing_pixels), 1.0e-6)
    epsilon = 1.0e-3

    energy_before, _ = self._energy(
        spline_graph, proposal_graph, source, h_x, v_y, tangent_h, tangent_v
    )
    for _step in range(self.steps):
        energy, _ = self._energy(
            spline_graph, proposal_graph, source, h_x, v_y, tangent_h, tangent_v
        )
        gradients = torch.autograd.grad(
            energy,
            (h_x, v_y, tangent_h, tangent_v),
            create_graph=False,
            retain_graph=False,
        )
        # First-order unrolling: do not differentiate through the inner gradient,
        # but preserve the direct proposal -> refined-state path.
        h_x = h_x - self.position_step * gradients[0].detach()
        v_y = v_y - self.position_step * gradients[1].detach()
        tangent_h = tangent_h - self.tangent_step * gradients[2].detach()
        tangent_v = tangent_v - self.tangent_step * gradients[3].detach()

        h_min = torch.maximum(h_edge_start + epsilon, h_reference - max_move_lattice)
        h_max = torch.minimum(h_edge_start + 1.0 - epsilon, h_reference + max_move_lattice)
        v_min = torch.maximum(v_edge_start + epsilon, v_reference - max_move_lattice)
        v_max = torch.minimum(v_edge_start + 1.0 - epsilon, v_reference + max_move_lattice)
        h_x = torch.maximum(h_min, torch.minimum(h_max, h_x))
        v_y = torch.maximum(v_min, torch.minimum(v_max, v_y))
        tangent_h = F.normalize(tangent_h, dim=-1, eps=1.0e-6)
        tangent_v = F.normalize(tangent_v, dim=-1, eps=1.0e-6)

    energy_after, parts = self._energy(
        spline_graph, proposal_graph, source, h_x, v_y, tangent_h, tangent_v
    )
    refined = self._graph_with_geometry(
        proposal_graph, h_x, v_y, tangent_h, tangent_v
    )
    spacing = float(spline_graph.spacing_pixels)
    shift_numerator = (
        (((refined["spline_control_point_h_lr"][..., 0] - h_reference) * spacing).square() * mask_h).sum()
        + (((refined["spline_control_point_v_lr"][..., 1] - v_reference) * spacing).square() * mask_v).sum()
    )
    shift_denominator = (mask_h.sum() + mask_v.sum()).clamp_min(1.0)
    refined["spline_refiner_energy_before"] = energy_before.detach()
    refined["spline_refiner_energy_after"] = energy_after.detach()
    refined["spline_refiner_node_shift_rms_pixels"] = torch.sqrt(
        shift_numerator / shift_denominator + 1.0e-12
    ).detach()
    refined["spline_refiner_steps"] = source_sdf_lr.new_tensor(float(self.steps))
    refined["spline_refiner_node_source_error"] = parts["nodeSource"]
    refined["spline_refiner_span_source_error"] = parts["spanSource"]
    return refined


def _seam_forward_deployed_authority(
    self: DirectionalSeamRestorer,
    albedo: torch.Tensor,
    normal_xy: torch.Tensor,
    material: torch.Tensor,
    *,
    sdf_pixels: torch.Tensor,
    coverage: torch.Tensor,
    profile_confidence: torch.Tensor,
    edge_probability: torch.Tensor,
    geometry_normal: torch.Tensor | None = None,
    source_albedo: torch.Tensor | None = None,
    source_normal: torch.Tensor | None = None,
    source_material: torch.Tensor | None = None,
    authority_override: torch.Tensor | None = None,
    tangent_override: torch.Tensor | None = None,
    phase_only: bool = False,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Deploy the same PhaseAwareSeamSR proposal proven in B3 with one learned authority."""
    del phase_only  # V12.2 deliberately makes the proven phase-SR proposal canonical.
    if _ORIGINAL_SEAM_FORWARD is None:
        raise RuntimeError("authority alignment installed without original seam forward")
    result = _ORIGINAL_SEAM_FORWARD(
        self,
        albedo,
        normal_xy,
        material,
        sdf_pixels=sdf_pixels,
        coverage=coverage,
        profile_confidence=profile_confidence,
        edge_probability=edge_probability,
        geometry_normal=geometry_normal,
        source_albedo=source_albedo,
        source_normal=source_normal,
        source_material=source_material,
        authority_override=authority_override,
        tangent_override=tangent_override,
        phase_only=True,
        enabled=enabled,
    )

    learned_authority = result["learned_authority"].float().clamp(0.0, 1.0)
    if authority_override is not None:
        authority = authority_override.to(
            device=albedo.device, dtype=torch.float32, non_blocking=True
        ).clamp(0.0, 1.0)
        if authority.shape[-2:] != albedo.shape[-2:]:
            authority = F.interpolate(
                authority, size=albedo.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
        authority_forced = torch.ones_like(authority)
    else:
        # Strength/ridge/geometry support already enters the authority network as
        # evidence. Multiplying the learned answer by those features again created
        # the B3->B4 collapse measured by Micro V2.
        authority = learned_authority * float(self.max_authority)
        authority_forced = torch.zeros_like(authority)
    if not enabled:
        authority = torch.zeros_like(authority)

    phase_delta = result["phase_delta"].float()
    phase_albedo = (albedo.float() + phase_delta[:, 0:3]).clamp(0.0, 1.0)
    phase_normal = self._normalise_xy(normal_xy.float() + phase_delta[:, 3:5])
    phase_material = (material.float() + phase_delta[:, 5:8]).clamp(0.0, 1.0)

    result["albedo"] = (
        albedo.float() * (1.0 - authority) + phase_albedo * authority
    ).to(albedo.dtype)
    result["normal_xy"] = self._normalise_xy(
        normal_xy.float() * (1.0 - authority) + phase_normal * authority
    ).to(normal_xy.dtype)
    result["material"] = (
        material.float() * (1.0 - authority) + phase_material * authority
    ).clamp(0.0, 1.0).to(material.dtype)
    result["authority"] = authority.to(albedo.dtype)
    result["authority_forced"] = authority_forced.to(albedo.dtype)
    result["phase_mix"] = torch.ones_like(authority).to(albedo.dtype)
    result["kernel_mix"] = torch.zeros_like(authority).to(albedo.dtype)
    result["phase_only"] = torch.ones_like(authority).to(albedo.dtype)
    return result


def _forward_impl_single_final_selector(
    self: FidelityResidualNetV9,
    *args: Any,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Make BenefitSelector the sole final residual authority."""
    if _ORIGINAL_FORWARD_IMPL is None:
        raise RuntimeError("authority alignment installed without original model forward")
    outputs = _ORIGINAL_FORWARD_IMPL(self, *args, **kwargs)
    gate = outputs["benefit_selector_probability"].float().clamp(0.0, 1.0)
    baseline_albedo = outputs["baseline_albedo"].float()
    baseline_normal = outputs["baseline_normal"].float()
    baseline_material = outputs["baseline_material"].float()
    candidate_albedo = outputs["detail_candidate_albedo"].float()
    candidate_normal = outputs["detail_candidate_normal"].float()
    candidate_material = outputs["detail_candidate_material"].float()

    outputs["albedo"] = (
        baseline_albedo * (1.0 - gate) + candidate_albedo * gate
    ).clamp(0.0, 1.0).to(outputs["baseline_albedo"].dtype)
    outputs["normal_xy"] = self._normalize_xy(
        baseline_normal * (1.0 - gate) + candidate_normal * gate
    ).to(outputs["baseline_normal"].dtype)
    outputs["material"] = (
        baseline_material * (1.0 - gate) + candidate_material * gate
    ).clamp(0.0, 1.0).to(outputs["baseline_material"].dtype)
    outputs["roughness"] = outputs["material"][:, 2:3]
    outputs["emissive"] = outputs["material"][:, 1:2]
    class_centres = torch.linspace(
        0.0,
        1.0,
        self.config.material_classes,
        device=outputs["material"].device,
        dtype=outputs["material"].dtype,
    )
    outputs["material_logits"] = -(
        outputs["material"][:, 0:1] - class_centres.view(1, -1, 1, 1)
    ).square() * 40.0
    outputs["final_selector_gate"] = gate.to(outputs["baseline_albedo"].dtype)
    outputs["confidence"] = gate.clamp(1.0e-5, 1.0 - 1.0e-5).to(
        outputs["baseline_albedo"].dtype
    )
    outputs["confidence_logits"] = outputs["benefit_selector_logits"].to(
        outputs["baseline_albedo"].dtype
    )
    return outputs


def _architecture_contract_authority_aligned(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("authority alignment installed without original architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["authorityAlignmentRevision"] = AUTHORITY_ALIGNMENT_REVISION
    contract["selectorRequires"] = (
        "one BenefitSelector probability; detail confidence/regret are selector features only"
    )
    contract["candidateAuthority"] = (
        "one final BenefitSelector residual blend over deterministic B"
    )
    contract["seamAuthority"] = (
        "learned seam authority directly gates the proven PhaseAwareSeamSR proposal"
    )
    contract["seamPhaseOnlyReconstruction"] = True
    contract["reconstructionPrimitive"] = (
        "deterministic B + raw refined connected-spline redraw + phase-SR/detail residual; "
        "one final BenefitSelector residual blend"
    )
    contract["b1bObjective"] = (
        "neural connected-spline initializer -> first-order differentiable LR-only explicit "
        "refinement -> same deterministic renderer -> baseline-relative HR training loss"
    )
    contract["structuralResidualAuthority"] = (
        "B1a zero-authority; B1b raw refined geometry; no signed inverse blend"
    )
    contract["explicitGeometryRefinementDifferentiableTraining"] = True
    contract["explicitGeometryRefinementGradient"] = (
        "first-order unrolled refined-render loss -> neural spline initializer"
    )
    return contract


def _optimal_residual_gate(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-pixel least-squares blend coefficient for B + g(C-B)."""
    delta = candidate.float() - baseline.float()
    desired = target.float() - baseline.float()
    numerator = (delta * desired).sum(dim=1, keepdim=True)
    denominator = delta.square().sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    gate = (numerator / denominator).clamp(0.0, 1.0).detach()
    motion = delta.abs().mean(dim=1, keepdim=True).detach()
    return gate, motion


def _compute_losses_authority_aligned(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    """Train authority heads on the exact deployed outputs they control."""
    import v9.training as training

    base = getattr(training, "_nsamdr_authority_base_compute_losses", None)
    if base is None:
        raise RuntimeError("authority alignment installed without base compute_losses")
    losses = base(outputs, batch, config, phase)

    if phase == "seam-authority":
        target = batch["target_albedo"].float()
        before = outputs["boundary_pre_seam_albedo"].detach().float()
        deployed = outputs["boundary_reconstructed_albedo"].float()
        edge = batch["target_edge"].float().clamp(0.0, 1.0)
        error_map = (deployed - target).abs().mean(dim=1, keepdim=True)
        before_error_map = (before - target).abs().mean(dim=1, keepdim=True)
        seam_weight = (0.10 + edge * 1.90).detach()
        deployed_reconstruction = _weighted_mean(error_map, seam_weight)
        deployed_regret = _weighted_mean(
            F.relu(error_map - before_error_map), seam_weight
        )
        losses["seam_deployed_reconstruction"] = deployed_reconstruction
        losses["seam_deployed_regret"] = deployed_regret
        losses["seam_deployed_recovery"] = (
            (before_error_map.mean() - error_map.mean())
            / before_error_map.mean().clamp_min(1.0e-6)
        ).detach()
        losses["total"] = (
            deployed_reconstruction
            * float(getattr(config, "seam_reconstruction_weight", 28.0))
            + deployed_regret
            * float(getattr(config, "seam_authority_teacher_weight", 18.0))
            + losses["seam_authority_teacher"]
            * float(getattr(config, "seam_authority_teacher_weight", 18.0))
            * 0.25
            + losses["seam_authority_regularization"]
            * float(getattr(config, "seam_authority_regularization_weight", 0.20))
        ).float()
        return losses

    if phase not in {"boundary-hardening", "physical-finetune"}:
        return losses

    target_albedo = batch["target_albedo"].float()
    baseline_albedo = outputs["baseline_albedo"].detach().float()
    candidate_albedo = outputs["detail_candidate_albedo"].detach().float()
    final_albedo = outputs["albedo"].float()
    probability = outputs["benefit_selector_probability"].float().clamp(1.0e-5, 1.0 - 1.0e-5)
    optimal_gate, motion = _optimal_residual_gate(
        baseline_albedo, candidate_albedo, target_albedo
    )
    motion_weight = (0.10 + (motion / 0.02).clamp(0.0, 1.0) * 0.90).detach()
    selector_gate_loss = _weighted_mean(
        F.binary_cross_entropy(probability, optimal_gate, reduction="none"),
        motion_weight,
    )

    final_error_map = (final_albedo - target_albedo).abs().mean(dim=1, keepdim=True)
    baseline_error_map = (baseline_albedo - target_albedo).abs().mean(dim=1, keepdim=True)
    final_albedo_loss = torch.sqrt(
        (final_albedo - target_albedo).square() + 1.0e-6
    ).mean()
    final_regret = F.relu(final_error_map - baseline_error_map).mean()

    final_gray = final_albedo.mean(dim=1, keepdim=True)
    target_gray = target_albedo.mean(dim=1, keepdim=True)
    fgx, fgy = sobel_tensor(final_gray)
    tgx, tgy = sobel_tensor(target_gray)
    final_gradient = (fgx - tgx).abs().mean() + (fgy - tgy).abs().mean()

    final_normal = outputs["normal_xy"].float()
    target_normal = batch["target_normal"].float()
    final_normal_loss = (final_normal - target_normal).abs().mean()

    target_material_scalar = (
        batch["target_material_class"].float().unsqueeze(1) + 0.5
    ) / float(max(int(config.material_classes), 1))
    target_material = torch.cat((
        target_material_scalar,
        batch["target_emissive"].float(),
        batch["target_roughness"].float(),
    ), dim=1)
    final_material_loss = (
        outputs["material"].float() - target_material
    ).abs().mean()

    candidate_error_map = (candidate_albedo - target_albedo).abs().mean(dim=1, keepdim=True)
    losses["selector_optimal_gate"] = selector_gate_loss
    losses["selector_optimal_gate_mean"] = optimal_gate.mean().detach()
    losses["selector_candidate_gain"] = (
        baseline_error_map.mean() - candidate_error_map.mean()
    ).detach()
    losses["selector_final_albedo"] = final_albedo_loss
    losses["selector_final_gradient"] = final_gradient
    losses["selector_final_normal"] = final_normal_loss
    losses["selector_final_material"] = final_material_loss
    losses["selector_final_regret"] = final_regret

    # Existing log readers keep the historical names, but they now describe the
    # exact full detail candidate and the exact final selector head.
    losses["boundary_gate"] = selector_gate_loss
    losses["gate_target"] = optimal_gate.mean().detach()
    losses["boundary_candidate_gain"] = losses["selector_candidate_gain"]
    losses["boundary_candidate_win_fraction"] = (
        candidate_error_map <= baseline_error_map + (1.0 / 255.0)
    ).float().mean().detach()

    losses["total"] = (
        final_albedo_loss * float(config.albedo_weight)
        + final_gradient * float(config.albedo_gradient_weight)
        + final_normal_loss * float(config.normal_weight)
        + final_material_loss * float(config.material_weight)
        + final_regret * float(config.regret_weight)
        + selector_gate_loss * float(config.benefit_selector_weight)
    ).float()
    return losses


def install_authority_alignment_model_contract() -> None:
    """Install production graph semantics for every v9 consumer, including inference."""
    global _MODEL_INSTALLED
    global _ORIGINAL_FORWARD_IMPL
    global _ORIGINAL_ARCHITECTURE_CONTRACT
    global _ORIGINAL_SEAM_FORWARD
    global _ORIGINAL_REFINER_FORWARD
    global _ORIGINAL_LOCK_TOPOLOGY_FOR_PROOF

    if _MODEL_INSTALLED:
        return
    _ORIGINAL_FORWARD_IMPL = FidelityResidualNetV9._forward_impl
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    _ORIGINAL_SEAM_FORWARD = DirectionalSeamRestorer.forward
    _ORIGINAL_REFINER_FORWARD = ExplicitSplineGeometryRefiner.forward
    _ORIGINAL_LOCK_TOPOLOGY_FOR_PROOF = LocalBoundaryProductionStructure.lock_topology_for_proof

    FidelityResidualNetV9._forward_impl = _forward_impl_single_final_selector
    FidelityResidualNetV9._structural_residual_weight = staticmethod(
        _nonnegative_structural_residual_weight
    )
    FidelityResidualNetV9.architecture_contract = _architecture_contract_authority_aligned
    DirectionalSeamRestorer.forward = _seam_forward_deployed_authority
    ExplicitSplineGeometryRefiner.forward = _refiner_forward_with_outer_gradient
    LocalBoundaryProductionStructure.lock_topology_for_proof = (
        _lock_topology_for_proof_with_structural_latch
    )
    _MODEL_INSTALLED = True


def install_authority_alignment_training_contract(training: Any) -> None:
    """Install training-only loss alignment after the local-boundary/B1b adapters."""
    install_authority_alignment_model_contract()
    training._nsamdr_authority_alignment_model_installed = True
    if not bool(getattr(training, "_nsamdr_authority_alignment_loss_installed", False)):
        training._nsamdr_authority_base_compute_losses = training.compute_losses
        training._nsamdr_authority_alignment_loss_installed = True
    training.compute_losses = _compute_losses_authority_aligned


def install_authority_alignment_contract(training: Any) -> None:
    """Compatibility entry point used by TrainingBackend."""
    install_authority_alignment_training_contract(training)
