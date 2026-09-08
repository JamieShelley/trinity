from __future__ import annotations

from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import SimpleNamespace
import ast
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
V9 = NEURAL / "v9"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_authority_contract_is_module_level_and_windows_spawn_safe():
    from v9 import training as training_module
    from v9.application.backend import TrainingBackend

    TrainingBackend()
    ForkingPickler.dumps(training_module._data_worker_init)

    source = (V9 / "authority_alignment_contract.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "<locals>" not in source
    assert "install_authority_alignment_contract(training)" in (
        V9 / "application/backend.py"
    ).read_text(encoding="utf-8")


def test_authority_model_contract_is_installed_for_preflight_and_inference():
    package = (V9 / "__init__.py").read_text(encoding="utf-8")
    authority = (V9 / "authority_alignment_contract.py").read_text(encoding="utf-8")
    assert "install_authority_alignment_model_contract()" in package
    assert "def install_authority_alignment_model_contract()" in authority
    assert "def install_authority_alignment_training_contract(training" in authority
    assert 'AUTHORITY_ALIGNMENT_REVISION = "V12.2"' in authority


def test_structural_authority_cannot_invert_refined_geometry():
    from v9.application.backend import TrainingBackend
    from v9.model import FidelityResidualNetV9

    TrainingBackend()
    locality = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    negative = -torch.ones_like(locality)
    positive = torch.ones_like(locality)
    assert torch.equal(
        FidelityResidualNetV9._structural_residual_weight(locality, negative, None),
        torch.zeros_like(locality),
    )
    assert torch.equal(
        FidelityResidualNetV9._structural_residual_weight(locality, positive, None),
        locality,
    )


def test_seam_deploys_phase_sr_with_one_learned_authority():
    source = (V9 / "authority_alignment_contract.py").read_text(encoding="utf-8")
    assert "phase_only=True" in source
    assert "authority = learned_authority * float(self.max_authority)" in source
    assert "Strength/ridge/geometry support already enters" in source
    assert 'result["phase_mix"] = torch.ones_like(authority)' in source
    assert 'result["kernel_mix"] = torch.zeros_like(authority)' in source
    assert 'losses["seam_deployed_reconstruction"]' in source
    assert 'losses["seam_deployed_regret"]' in source


def test_final_selector_targets_and_controls_the_exact_full_candidate():
    source = (V9 / "authority_alignment_contract.py").read_text(encoding="utf-8")
    assert 'candidate_albedo = outputs["detail_candidate_albedo"]' in source
    assert 'gate = outputs["benefit_selector_probability"]' in source
    assert 'outputs["final_selector_gate"] = gate' in source
    assert "confidence_support" not in source
    assert "regret_suppression" not in source
    assert "def _optimal_residual_gate(" in source
    assert 'losses["selector_optimal_gate"]' in source
    assert 'losses["selector_final_albedo"]' in source
    assert 'losses["selector_final_regret"]' in source


def test_architecture_contract_describes_single_final_authority_and_gradient_path():
    source = (V9 / "authority_alignment_contract.py").read_text(encoding="utf-8")
    assert "one BenefitSelector probability" in source
    assert "confidence/regret are selector features only" in source
    assert "no signed inverse blend" in source
    assert 'contract["explicitGeometryRefinementDifferentiableTraining"] = True' in source
    assert "first-order unrolled refined-render loss" in source


def test_production_head_supervision_is_module_level_and_windows_spawn_safe():
    from v9 import training as training_module
    from v9.application.backend import TrainingBackend

    backend = (V9 / "application/backend.py").read_text(encoding="utf-8")
    ast.parse(backend)
    assert 'PRODUCTION_HEAD_SUPERVISION_REVISION = "V12.2.1"' in backend
    assert "def _forward_for_phase_with_deployed_seam_authority(" in backend
    assert "def _phase_lr_with_head_capacity(" in backend
    assert "def _compute_losses_with_production_head_supervision(" in backend
    assert "<locals>" not in backend

    TrainingBackend()
    ForkingPickler.dumps(training_module._data_worker_init)


def test_b1b_optimizes_raw_production_geometry_candidate():
    backend = (V9 / "application/backend.py").read_text(encoding="utf-8")
    assert 'candidate = outputs["boundary_initial_candidate_albedo"].float()' in backend
    assert 'losses["b1b_production_global_reconstruction"]' in backend
    assert 'losses["b1b_production_edge_reconstruction"]' in backend
    assert 'losses["b1b_production_regret"]' in backend
    assert 'losses["b1b_production_gradient"]' in backend
    assert 'renderer_loss = losses.get("boundary_photometric")' not in backend


def test_b1b_proxy_objective_has_zero_sgd_authority():
    package = (V9 / "__init__.py").read_text(encoding="utf-8")
    source = (V9 / "b1_production_objective_contract.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'B1_PRODUCTION_OBJECTIVE_REVISION = "V12.2.4"' in source
    assert "install_b1_production_objective_contract()" in package
    assert "production, telemetry = _live_production_objective(outputs, batch, config)" in source
    assert "production.requires_grad" in source
    assert 'losses["b1b_nonproduction_objective_ignored"]' in source
    assert 'losses["total"] = production' in source
    assert "_compute_losses_with_b1b_renderer_supervision = _production_only_b1b_loss" in source
    assert "<locals>" not in source


def test_b1b_uses_observable_source_target_structural_support():
    source = (V9 / "b1_production_objective_contract.py").read_text(encoding="utf-8")
    assert "def _observable_structural_support(" in source
    assert 'target_sdf = batch.get("target_sdf")' in source
    assert 'source_pixels = outputs.get("source_sdf_prior_pixels")' in source
    assert "source_observable" in source
    assert "target_proximity" in source
    assert 'losses.update(telemetry)' in source
    assert '"b1b_structural_recovery"' in source
    assert '"b1b_off_support_identity"' in source


def test_b1b_production_objective_retains_candidate_gradient():
    from v9.b1_production_objective_contract import _live_production_objective

    candidate = torch.full((1, 3, 8, 8), 0.20, dtype=torch.float32, requires_grad=True)
    baseline = torch.zeros_like(candidate)
    target = torch.full_like(candidate, 0.75)
    target_sdf = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    source_sdf_pixels = torch.zeros_like(target_sdf)
    config = SimpleNamespace(
        spline_graph_render_weight=96.0,
        spline_graph_render_gradient_weight=48.0,
        contour_sdf_max_distance_pixels=24.0,
        sdf_metric_band_pixels=6.0,
    )

    objective, telemetry = _live_production_objective(
        {
            "boundary_initial_candidate_albedo": candidate,
            "baseline_albedo": baseline,
            "source_sdf_prior_pixels": source_sdf_pixels,
        },
        {
            "target_albedo": target,
            "target_sdf": target_sdf,
        },
        config,
    )
    assert objective.requires_grad
    assert objective.grad_fn is not None
    assert float(telemetry["b1b_structural_support_mean"].item()) > 0.0
    objective.backward()
    assert candidate.grad is not None
    assert bool(torch.isfinite(candidate.grad).all().item())
    assert float(candidate.grad.abs().sum().item()) > 0.0


def test_b4_uses_deployed_forward_and_exact_phase_sr_oracle():
    backend = (V9 / "application/backend.py").read_text(encoding="utf-8")
    assert 'if phase == "seam-authority":\n        return model(batch["input"])' in backend
    assert 'phase_delta = outputs["seam_phase_delta"][:, 0:3].detach().float()' in backend
    assert "effective_oracle, motion = _optimal_residual_gate(" in backend
    assert 'learned = outputs["seam_learned_authority"]' in backend
    assert 'losses["seam_authority_iou"] = authority_iou' in backend
    assert 'losses["seam_authority_oracle_mean"] = oracle.mean().detach()' in backend


def test_detail_and_selector_penalize_deployed_edge_regressions():
    backend = (V9 / "application/backend.py").read_text(encoding="utf-8")
    assert 'if phase == "detail-reconstruction":' in backend
    assert 'candidate = outputs["detail_candidate_albedo"].float()' in backend
    assert 'losses["detail_deployed_regret"]' in backend
    assert 'losses["detail_deployed_global_regret"]' in backend
    assert 'losses["detail_deployed_gradient"]' in backend

    assert 'probability = outputs["benefit_selector_probability"]' in backend
    assert "oracle, motion = _optimal_residual_gate(baseline, candidate, target)" in backend
    assert 'losses["selector_edge_reconstruction"]' in backend
    assert 'losses["selector_edge_regret"]' in backend
    assert 'losses["selector_edge_gradient"]' in backend


def test_isolated_head_phases_receive_capacity_learning_rates():
    backend = (V9 / "application/backend.py").read_text(encoding="utf-8")
    for fragment in (
        '"sdf-proof": 2.0',
        '"seam-authority": 5.0',
        '"gate-proof": 25.0',
        '"detail-reconstruction": 3.0',
        '"boundary-hardening": 20.0',
        '"physical-finetune": 20.0',
    ):
        assert fragment in backend

    assert "service._forward_for_phase = _forward_for_phase_with_deployed_seam_authority" in backend
    assert "service._phase_lr = _phase_lr_with_head_capacity" in backend
    assert "_install_production_head_supervision(training)" in backend
