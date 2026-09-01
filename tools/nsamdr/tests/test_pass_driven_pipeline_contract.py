"""Pass-driven orchestration contracts after composition-oriented Stage 2 refactor."""
from __future__ import annotations

import inspect
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


class TestPassDrivenPipelineContract:
    def test_stage_order_is_local_then_full_production_downstream(self) -> None:
        """Verify explicit StagePlan order while B1/B2 executes through sdf-proof.

        Purpose:
            Preserve the current local-boundary curriculum across OOP decomposition.
        Called by:
            pytest.
        Calls:
            QualificationGates(), StagePlan().
        """
        from v9.application.gates import QualificationGates, StagePlan

        phases = [
            definition.phase
            for definition in StagePlan(QualificationGates()).definitions
        ]
        assert phases == [
            "sdf-bootstrap",
            "seam-proof",
            "seam-authority",
            "gate-proof",
            "detail-reconstruction",
        ]
        assert "sdf-proof" not in phases

    def test_each_stage_is_still_canonical_train_v9(self) -> None:
        """Verify stage and final execution both route through TrainingBackend.

        Purpose:
            Ensure the refactor did not introduce alternate trainer implementations.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.pipeline import PassDrivenPipeline

        stage = inspect.getsource(PassDrivenPipeline._run_stage)
        final = inspect.getsource(PassDrivenPipeline._run_final)
        invoke = inspect.getsource(PassDrivenPipeline._invoke)
        assert "self._invoke" in stage
        assert '"sdf-proof" if definition.phase == "sdf-bootstrap"' in stage
        assert "stop_after_phase=trainer_stop_phase" in stage
        assert "self._invoke" in final
        assert "stop_after_phase=None" in final
        assert "self.backend.run" in invoke

    def test_local_geometry_promotion_is_fail_closed(self) -> None:
        """Verify structural success promotes explicitly and failure rejects downstream.

        Purpose:
            Preserve the original B1/B2 promotion/rejection semantics.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.pipeline import PassDrivenPipeline

        source = inspect.getsource(PassDrivenPipeline._run_stage)
        assert "_complete_structural_stage" in source
        assert "self.experiments.reject" in source
        assert "Downstream stages were not run." in source

    def test_final_stage_requires_real_production_final(self) -> None:
        """Verify final stage still requires trainingSafetyPass and production-final.

        Purpose:
            Preserve strict final selection semantics across orchestration refactoring.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.pipeline import PassDrivenPipeline

        source = inspect.getsource(PassDrivenPipeline._run_final)
        assert 'latest.get("trainingSafetyPass", False)' in source
        assert '== "production-final"' in source
    def test_backend_patch_keeps_windows_worker_initializer_pickleable(self) -> None:
        """Verify V11.4 callbacks do not break spawn-based DataLoader workers.

        Purpose:
            Prevent nested trainer callbacks from making the TrainingService singleton
            unpickleable when Windows serializes its bound worker initializer.
        Called by:
            pytest.
        Calls:
            TrainingBackend(), pickle.dumps().
        """
        import v9.training as training
        from v9.application.backend import TrainingBackend

        TrainingBackend()
        callback = training._training_service._explicit_primitive_structure_microproof
        assert "<locals>" not in callback.__qualname__
        pickle.dumps(training._data_worker_init)

    def test_v114_sdf_proof_uses_live_local_production_graph(self) -> None:
        """Prevent the retired B1b loss from replacing V11.4 gradients.

        Purpose:
            Keep sdf-proof on the current production_structure while retaining the
            legacy compact loss only for schemas without that structure.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        assert 'hasattr(model.geometry_net, "production_structure")' in source
        assert 'phase == "sdf-proof" and not local_structure_phase' in source
        assert 'b1b_classifier_qualified = True' in source
        assert 'b1b_parameters_qualified = True' in source

    def test_v114_sdf_proof_uses_production_batch_loader(self) -> None:
        """Keep the full V11.4 graph off the retired batch-8 primitive loader.

        Purpose:
            Ensure local-boundary sdf-proof uses the canonical production loader so
            the full 128-to-512 graph remains batch-size 1 for bounded VRAM/RAM use.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        assert 'local_structure_train_loader' in source
        assert 'if local_structure_phase' in source
        assert 'else parametric_train_loader' in source
        assert 'batch_size=config.batch_size' in source

    def test_b1a_topology_checkpoint_locks_after_first_pass(self) -> None:
        """Prevent later bootstrap epochs from replacing qualified B1a topology.

        Purpose:
            Keep the first topology-safe B1a checkpoint authoritative for sdf-proof.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        block = source.split(
            '# B1a qualifies topology only; smoothness belongs exclusively to B1b.', 1
        )[1].split('elif phase == "sdf-proof":', 1)[0]
        assert 'if not topology_bootstrapped:' in block
        assert block.index('if not topology_bootstrapped:') < block.index('best_b1a_path')
        assert 'topology checkpoint locked for sdf-proof' in block

    def test_evolution_requires_permanent_topology_audit(self) -> None:
        """Prevent one-crop evolutionary fitness from bypassing production topology.

        Purpose:
            Require candidate acceptance to use the permanent 29-case proof family and
            the same connected-component/hole topology mismatch used by B1/B2.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.evolution.candidate import CandidateEvaluator
        from v9.evolution.fitness import StructuralFitness

        topology_source = inspect.getsource(CandidateEvaluator._measure_permanent_topology)
        fitness_source = inspect.getsource(StructuralFitness.measure)
        assert 'SyntheticGeometryValidationDataset' in topology_source
        assert 'max(29' in topology_source
        assert '+ 9_911' in topology_source
        assert 'sdf_topology_mismatch' in topology_source
        assert 'predicted_topology > source_topology' in topology_source
        assert 'topology_regression_fraction <= 0.0' in fitness_source

    def test_v114_sdf_proof_trains_subpixel_quality_terms(self) -> None:
        """Keep B1b authority on the defects judged by the structural gate.

        Purpose:
            Prevent V11.4 sdf-proof from computing anti-staircase, radial-smoothness,
            and same-renderer profile evidence only as detached diagnostics.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.local_boundary_production_contract import LocalBoundaryProductionContract

        source = inspect.getsource(LocalBoundaryProductionContract._local_compute_losses)
        proof = source.split('if phase == "sdf-proof":', 1)[1]
        assert 'losses["implicit_subpixel_surface"]' in proof
        assert 'losses["implicit_subpixel_gradient"]' in proof
        assert 'losses["implicit_subpixel_eikonal"]' in proof
        assert 'losses["sdf_curvature"]' in proof
        assert 'losses["sdf_teacher_gradient"]' in proof
        assert 'losses["sdf_teacher_profile"]' in proof

    def test_v114_reuses_remaining_bootstrap_budget_for_sdf_proof(self) -> None:
        """Do not discard structural epochs after B1a topology qualifies.

        Purpose:
            Keep the fixed Quick/Full structural budget pass-driven so remaining
            nominal bootstrap epochs train V11.4 contour/profile quality.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        assert 'planned_phase == "sdf-bootstrap" and topology_bootstrapped' in source
        assert 'phase = "sdf-proof"' in source
        assert 'reallocating remaining' in source
        assert 'bootstrap epoch to V11.4 sdf-proof quality training' in source

    def test_v114_sdf_proof_locks_b1a_topology_controls(self) -> None:
        """Keep B1b from rewriting branch/ribbon/CSG topology decisions.

        Purpose:
            Require V11.4 sdf-proof to retain the qualified B1a topology producer
            while continuous distance/normal/curvature/width rows remain trainable.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.local_boundary_production_contract import LocalBoundaryProductionContract
        from v9.parametric_boundary import PrimitiveParameterHead

        assert PrimitiveParameterHead.TOPOLOGY_CHANNELS == (5, 11, 17, 18, 19, 20, 21, 22)
        phase_source = inspect.getsource(LocalBoundaryProductionContract._set_phase)
        lock_source = inspect.getsource(LocalBoundaryProductionContract._lock_proof_topology)
        assert 'unlock_topology_for_bootstrap()' in phase_source
        assert '_lock_proof_topology(self)' in phase_source
        assert 'model.geometry_net.stem' in lock_source
        assert 'model.geometry_net.decoders' in lock_source

    def test_v114_topology_lock_is_structurally_independent(self) -> None:
        """Keep B1a topology immutable while B1b retains hidden geometry capacity.

        Purpose:
            Prevent the continuous geometry optimizer from sharing mutable hidden
            features or output parameters with ribbon/branch/CSG topology.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.local_boundary_production_contract import LocalBoundaryProductionStructure
        from v9.parametric_boundary import PrimitiveParameterHead

        init_source = inspect.getsource(PrimitiveParameterHead.__init__)
        lock_source = inspect.getsource(PrimitiveParameterHead.lock_topology)
        structure_lock = inspect.getsource(LocalBoundaryProductionStructure.lock_topology_for_proof)
        assert 'self.geometry_net = self._make_branch' in init_source
        assert 'self.topology_net = self._make_branch' in init_source
        assert 'self.topology_net.parameters()' in lock_source
        assert 'parameter.requires_grad_(False)' in lock_source
        assert 'self.topology_feature_project.parameters()' in structure_lock
        assert 'self.geometry_feature_project.parameters()' in structure_lock
        assert 'head.geometry_net.parameters()' in structure_lock

    def test_v114_sdf_proof_uses_batch_one_analytic_teacher_bank(self) -> None:
        """Keep V11.4 quality proof on analytic teachers without restoring batch inflation.

        Purpose:
            Require the full V11.4 sdf-proof graph to use a class-balanced analytic
            complete-teacher loader at the canonical production batch size.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        assert 'local_structure_train_dataset = ParametricPrimitiveTrainingDataset(' in source
        assert 'batch_size=config.batch_size' in source
        proof = source.split('if phase == "sdf-proof":', 1)[1]
        loader_block = proof.split('elif phase == "seam-proof":', 1)[0]
        assert 'local_structure_train_loader' in loader_block
        assert 'train_loader if local_structure_phase' not in loader_block
        assert '(int(config.tiles_per_epoch) + PRIMITIVE_COUNT - 1)' in source

    def test_v114_sdf_proof_cancels_raster_phase_in_final_geometry(self) -> None:
        """Train the final analytic geometry rather than smoothing source-relative residuals.

        Purpose:
            Prevent shallow lines and curves from preserving amplified LR raster phase
            when B1a topology is locked and sdf-proof refines continuous geometry.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.local_boundary_production_contract import LocalBoundaryProductionContract
        from v9.parametric_boundary import LocalParametricBoundaryDecoder, PrimitiveParameterHead

        head_source = inspect.getsource(PrimitiveParameterHead.forward)
        init_source = inspect.getsource(PrimitiveParameterHead.__init__)
        assert 'geometry_raw = self.geometry_net(x)' in head_source
        assert 'topology_raw = self.topology_net(topology_x)' in head_source
        assert 'self.geometry_net = self._make_branch' in init_source
        assert 'self.topology_net = self._make_branch' in init_source

        context_source = inspect.getsource(LocalParametricBoundaryDecoder.build_context)
        assert 'topology_feature_grid: torch.Tensor | None = None' in context_source
        assert 'self.parameter_head(geometry_head_input, topology_head_input)' in context_source
        assert 'binomial = source_pixels.new_tensor' in context_source
        assert 'kernel_x = binomial.view(1, 1, 1, 5)' in context_source
        assert 'kernel_y = binomial.view(1, 1, 5, 1)' in context_source
        assert '0.40 * source_pixels' not in context_source

        loss_source = inspect.getsource(LocalBoundaryProductionContract._local_compute_losses)
        assert 'anchor_error = anchor.float() - target_control' in loss_source
        assert 'error_dx = anchor_error' in loss_source
        assert 'error_dy = anchor_error' in loss_source
        assert 'losses["parametric_offset_smoothness"] = 0.5 * (' in loss_source

    def test_v114_continuous_geometry_is_coherent_and_zero_crossing_safe(self) -> None:
        """Smooth analytic geometry while preventing B1b from deleting the contour.

        Purpose:
            Require the continuous representation to preserve affine lines, derive
            normal/curvature from one coherent anchor field and keep a source-sign
            envelope outside the free subpixel reconstruction band.
        Called by:
            pytest.
        Calls:
            inspect.getsource(), torch.allclose().
        """
        import torch
        from v9.parametric_boundary import LocalParametricBoundaryDecoder

        smooth_source = inspect.getsource(
            LocalParametricBoundaryDecoder._smooth_geometry_field
        )
        coherent_source = inspect.getsource(
            LocalParametricBoundaryDecoder._coherent_branch_geometry
        )
        build_source = inspect.getsource(LocalParametricBoundaryDecoder.build_context)
        query_source = inspect.getsource(LocalParametricBoundaryDecoder.query)
        assert 'binomial = value.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0))' in smooth_source
        assert 'coherent_distance = self._smooth_geometry_field(distance)' in coherent_source
        assert 'nx, ny = gx / norm, gy / norm' in coherent_source
        assert 'curvature = self._smooth_geometry_field(' in coherent_source
        assert 'self._coherent_branch_geometry(raw_branch_distance)' in build_source
        assert 'zero_crossing_guard = 2.0' in query_source
        assert 'branch_center = self._hermite_sample_scalar(d_field, query_grid)' in query_source
        assert 'centre_surface = d +' not in query_source
        assert 'context["branch_normal_x"]' not in query_source
        assert 'context["branch_curvature_per_pixel"]' not in query_source
        assert 'phi_param.clamp_min(zero_crossing_epsilon)' in query_source
        assert 'phi_param.clamp_max(-zero_crossing_epsilon)' in query_source

        yy, xx = torch.meshgrid(
            torch.arange(20, dtype=torch.float32),
            torch.arange(24, dtype=torch.float32),
            indexing='ij',
        )
        affine = (0.13 * xx + 0.07 * yy - 2.0).unsqueeze(0).unsqueeze(0)
        smoothed = LocalParametricBoundaryDecoder._smooth_geometry_field(affine)
        assert torch.allclose(
            smoothed[:, :, 2:-2, 2:-2], affine[:, :, 2:-2, 2:-2], atol=1.0e-5
        )

    def test_v114_analytic_bank_uses_active_structural_budget_only(self) -> None:
        """Do not inflate the full production proof with the retired compact-bank floor.

        Purpose:
            Keep V11.4 class-balanced analytic proof at the active epoch budget while
            preserving batch-one memory behaviour and the held-out 29-case ladder.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        budget = source.split('local_structure_train_tiles =', 1)[1].split(
            'local_structure_train_dataset =', 1
        )[0]
        assert 'parametric_primitive_train_tiles_per_epoch' not in budget
        assert 'PRIMITIVE_COUNT' in budget
        loader = source.split('local_structure_train_loader = self._build_loader(', 1)[1].split(
            'validation_loader =', 1
        )[0]
        assert 'batch_size=config.batch_size' in loader

    def test_v114_outer_forward_exposes_analytic_anchor_for_sdf_proof(self) -> None:
        """Keep the V11.4 target-relative anchor proof connected to production output.

        Purpose:
            Require FidelityResidualNetV9 to propagate GeometryNet's live analytic
            control-lattice anchor into the dictionary consumed by sdf-proof losses.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.model import FidelityResidualNetV9

        source = inspect.getsource(FidelityResidualNetV9._forward_impl)
        assert (
            '"parametric_anchor_distance_pixels": '
            'geometry["parametric_anchor_distance_pixels"]'
        ) in source

def test_v114_connected_scalar_field_query_contract() -> None:
    """V11.4 reconstructs each branch from one distance-authoritative Hermite field."""
    import inspect
    import math

    import torch

    from v9.parametric_boundary import (
        LocalParametricBoundaryDecoder,
        PrimitiveParameterHead,
        make_query_grid,
    )

    source = inspect.getsource(LocalParametricBoundaryDecoder.query)
    hermite_source = inspect.getsource(LocalParametricBoundaryDecoder._hermite_sample_scalar)
    assert "branch_center = self._hermite_sample_scalar(d_field, query_grid)" in source
    assert "centre_surface = d +" not in source
    assert "_gather_control(value, ix, iy)" in hermite_source
    assert "2.0 * t3 - 3.0 * t2 + 1.0" in hermite_source
    assert 'context["branch_normal_x"]' not in source
    assert 'context["branch_normal_y"]' not in source
    assert 'context["branch_curvature_per_pixel"]' not in source
    assert "zero_crossing_guard = 2.0" in source

    h, w = 14, 18
    scale = 4.0
    p_y = (torch.arange(h, dtype=torch.float32) + 0.5) * scale
    p_x = (torch.arange(w, dtype=torch.float32) + 0.5) * scale
    py, px = torch.meshgrid(p_y, p_x, indexing="ij")
    angle = math.radians(3.0)
    nx, ny = -math.sin(angle), math.cos(angle)
    primary = (
        nx * (px - float(w) * scale * 0.5)
        + ny * (py - float(h) * scale * 0.5)
    ).unsqueeze(0).unsqueeze(0)
    inactive = torch.full_like(primary, 16.0)
    branches = torch.cat((primary, inactive, inactive), dim=1)
    zeros3 = torch.zeros_like(branches)
    activation = torch.cat(
        (torch.ones_like(primary), torch.zeros_like(primary), torch.zeros_like(primary)),
        dim=1,
    )
    csg = torch.cat(
        (
            torch.full_like(primary, 12.0),
            torch.full_like(primary, -12.0),
            torch.full_like(primary, -12.0),
        ),
        dim=1,
    )
    context = {
        "source_sdf_prior_lr": (primary / 16.0).clamp(-1.0, 1.0),
        "branch_anchor_distance_pixels": branches,
        "branch_normal_x": zeros3,
        "branch_normal_y": zeros3,
        "branch_curvature_per_pixel": zeros3,
        "branch_half_width_pixels": zeros3,
        "branch_ribbon_mode": zeros3,
        "branch_activation": activation,
        "csg_logits": csg,
        "confidence": torch.ones_like(primary),
        "distance_delta_pixels": torch.zeros_like(primary),
    }
    decoder = LocalParametricBoundaryDecoder(
        1,
        24,
        max_distance_pixels=16.0,
        control_scale=1,
        output_scale=4,
    )
    grid = make_query_grid(1, h * 4, w * 4, device=primary.device)
    actual = decoder.query(context, grid)["primitive_phi_pixels"]

    field = actual[0, 0]
    crosses = ((field[:-1, :] <= 0.0) != (field[1:, :] <= 0.0)).any(dim=0)
    assert float(crosses.float().mean()) >= 0.95

    # Query-time normal/curvature tensors are telemetry only. Deliberately corrupt
    # them and prove that the rendered zero-set remains controlled by branch distance.
    adversarial = dict(context)
    adversarial["branch_normal_x"] = torch.full_like(zeros3, 99.0)
    adversarial["branch_normal_y"] = torch.full_like(zeros3, -77.0)
    adversarial["branch_curvature_per_pixel"] = torch.full_like(zeros3, 12.0)
    adversarial_actual = decoder.query(adversarial, grid)["primitive_phi_pixels"]
    assert torch.allclose(actual, adversarial_actual, atol=1.0e-6, rtol=0.0)

    head = PrimitiveParameterHead(8, 24)
    before = tuple(head.state_dict().keys())
    assert head.OUTPUTS == 24
    assert head.TOPOLOGY_CHANNELS == (5, 11, 17, 18, 19, 20, 21, 22)
    assert head.GEOMETRY_CHANNELS == (
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 23
    )
    head.lock_topology()
    assert all(not p.requires_grad for p in head.topology_net.parameters())
    assert all(p.requires_grad for p in head.geometry_net.parameters())
    assert tuple(head.state_dict().keys()) == before


def test_v114_c1_hermite_passes_representation_microproof() -> None:
    """The production reconstruction must satisfy the unchanged strict geometry gate."""
    import torch

    from v9.training import TrainingService

    before, after, line_jitter, curve_rough = (
        TrainingService()._parametric_structure_microproof(torch.device("cpu"))
    )
    assert after <= 0.02
    assert after < before * 0.08
    assert line_jitter <= 0.05
    assert curve_rough <= 0.08


def test_v114_connected_circle_and_ellipse_remain_closed() -> None:
    """C1 Hermite branch fields keep closed analytic contours closed."""
    import numpy as np
    import torch

    from v9.geometry_metrics import GeometryMetrics
    from v9.parametric_boundary import LocalParametricBoundaryDecoder, make_query_grid

    metrics = GeometryMetrics()
    h = w = 32
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cx = cy = 15.5
    fields = (
        torch.sqrt((xx - cx).square() + (yy - cy).square() + 1.0e-6) - 9.0,
        torch.sqrt(
            ((xx - cx) / 1.35).square()
            + ((yy - cy) / 0.72).square()
            + 1.0e-6
        ) - 8.0,
    )
    decoder = LocalParametricBoundaryDecoder(
        1,
        24,
        max_distance_pixels=16.0,
        control_scale=1,
        output_scale=4,
    )
    for field in fields:
        primary = field.unsqueeze(0).unsqueeze(0)
        inactive = torch.full_like(primary, 16.0)
        branches = torch.cat((primary, inactive, inactive), dim=1)
        zeros3 = torch.zeros_like(branches)
        activation = torch.cat(
            (torch.ones_like(primary), torch.zeros_like(primary), torch.zeros_like(primary)),
            dim=1,
        )
        csg = torch.cat(
            (
                torch.full_like(primary, 12.0),
                torch.full_like(primary, -12.0),
                torch.full_like(primary, -12.0),
            ),
            dim=1,
        )
        context = {
            "source_sdf_prior_lr": (primary / 16.0).clamp(-1.0, 1.0),
            "branch_anchor_distance_pixels": branches,
            "branch_normal_x": zeros3,
            "branch_normal_y": zeros3,
            "branch_curvature_per_pixel": zeros3,
            "branch_half_width_pixels": zeros3,
            "branch_ribbon_mode": zeros3,
            "branch_activation": activation,
            "csg_logits": csg,
            "confidence": torch.ones_like(primary),
            "distance_delta_pixels": torch.zeros_like(primary),
        }
        grid = make_query_grid(1, h * 4, w * 4, device=primary.device)
        value = decoder.query(context, grid)["primitive_phi_pixels"][0, 0]
        value = value.detach().cpu().numpy()
        assert value[value.shape[0] // 2, value.shape[1] // 2] < 0.0
        assert np.all(value[0, :] > 0.0)
        assert np.all(value[-1, :] > 0.0)
        assert np.all(value[:, 0] > 0.0)
        assert np.all(value[:, -1] > 0.0)
        assert metrics._binary_topology_signature(value < 0.0) == (1, 0)


def test_v114_staircase_recovery_rejects_fragmented_contours() -> None:
    """A missing/broken line cannot score as successful staircase removal."""
    import numpy as np

    from v9.geometry_metrics import GeometryMetrics

    metrics = GeometryMetrics()
    h, w = 96, 128
    yy, xx = np.indices((h, w), dtype=np.float32)
    slope = 0.15
    centre = 38.0 + slope * xx
    target = np.abs(yy - centre) / np.sqrt(1.0 + slope * slope) - 2.0

    fragmented = target.copy()
    fragmented[:, 52:76] = np.maximum(fragmented[:, 52:76], 4.0)
    assert metrics.sdf_topology_mismatch(fragmented, target) == 1.0
    assert metrics.line_staircase_recovery(target, fragmented, target) == 0.0

    missing = np.full_like(target, 4.0)
    assert metrics.line_staircase_recovery(target, missing, target) == 0.0


def test_v114_two_pixel_sign_guard_is_hard_at_runtime() -> None:
    """Stable source signs survive outside the quantised-prior uncertainty band."""
    import torch
    from torch.nn import functional as F

    from v9.parametric_boundary import LocalParametricBoundaryDecoder, make_query_grid

    h = w = 18
    yy, _xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    source_pixels = (yy - 8.25).unsqueeze(0).unsqueeze(0)
    primary = -source_pixels
    inactive = torch.full_like(primary, 16.0)
    branches = torch.cat((primary, inactive, inactive), dim=1)
    zeros3 = torch.zeros_like(branches)
    activation = torch.cat(
        (torch.ones_like(primary), torch.zeros_like(primary), torch.zeros_like(primary)),
        dim=1,
    )
    csg = torch.cat(
        (
            torch.full_like(primary, 12.0),
            torch.full_like(primary, -12.0),
            torch.full_like(primary, -12.0),
        ),
        dim=1,
    )
    source_normalized = (source_pixels / 16.0).clamp(-1.0, 1.0)
    context = {
        "source_sdf_prior_lr": source_normalized,
        "branch_anchor_distance_pixels": branches,
        "branch_normal_x": zeros3,
        "branch_normal_y": zeros3,
        "branch_curvature_per_pixel": zeros3,
        "branch_half_width_pixels": zeros3,
        "branch_ribbon_mode": zeros3,
        "branch_activation": activation,
        "csg_logits": csg,
        "confidence": torch.ones_like(primary),
        "distance_delta_pixels": torch.zeros_like(primary),
    }
    decoder = LocalParametricBoundaryDecoder(
        1,
        24,
        max_distance_pixels=16.0,
        control_scale=1,
        output_scale=4,
    )
    grid = make_query_grid(1, h * 4, w * 4, device=primary.device)
    predicted = decoder.query(context, grid)["primitive_phi_pixels"]
    sampled_source = F.grid_sample(
        source_normalized,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ) * 16.0

    stable_positive = sampled_source >= 2.0
    stable_negative = sampled_source <= -2.0
    assert bool(torch.any(stable_positive))
    assert bool(torch.any(stable_negative))
    assert torch.all(predicted[stable_positive] >= 0.05)
    assert torch.all(predicted[stable_negative] <= -0.05)

def test_v114_c1_hermite_reproduces_affine_field_to_float_precision() -> None:
    """A shallow affine contour crosses cell seams without interpolation phase error."""
    import torch

    from v9.parametric_boundary import LocalParametricBoundaryDecoder, make_query_grid

    h = w = 24
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    lattice = (0.37 * xx + 0.19 * yy - 5.0).unsqueeze(0).unsqueeze(0)
    decoder = LocalParametricBoundaryDecoder(
        1, 24, max_distance_pixels=24.0, control_scale=1, output_scale=4
    )
    grid = make_query_grid(1, 96, 96, device=lattice.device)
    actual = decoder._hermite_sample_scalar(lattice, grid)
    qy, qx = torch.meshgrid(
        torch.arange(96, dtype=torch.float32),
        torch.arange(96, dtype=torch.float32),
        indexing="ij",
    )
    exact = (
        0.37 * ((qx + 0.5) / 4.0 - 0.5)
        + 0.19 * ((qy + 0.5) / 4.0 - 0.5)
        - 5.0
    ).unsqueeze(0).unsqueeze(0)
    assert float((actual - exact).abs().max()) <= 2.0e-5


def test_v114_structural_gate_rejects_rendered_topology_regression() -> None:
    """Raw-SDF topology cannot hide a topology regression introduced by rendering."""
    import inspect

    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'sdf_stageb_rendered_topology_regression_fraction' in source
    assert 'rendered_topology_regression == 0.0' in source
    bootstrap = source.split('topology_ok = (', 1)[1].split('# Once topology qualifies', 1)[0]
    assert 'rendered_topology_regression == 0.0' in bootstrap


def test_v115_connected_spline_graph_is_the_renderer_geometry_authority() -> None:
    """Final geometry comes from shared graph nodes, not the compatibility scalar field."""
    import inspect

    from v9.local_boundary_production_contract import (
        LocalBoundaryProductionContract,
        LocalBoundaryProductionStructure,
    )
    from v9.spline_graph import ConnectedSplineGraph

    structure = inspect.getsource(LocalBoundaryProductionStructure.forward)
    query = inspect.getsource(LocalBoundaryProductionContract._geometry_query_from_outputs)
    cell_spans = inspect.getsource(ConnectedSplineGraph._cell_spans)
    assert 'spline = self.spline_graph(' in structure
    assert 'self.decoder.query(' not in structure
    assert '"spline_graph": spline["graph"]' in structure
    assert 'self.production_structure.spline_graph.query(graph, query_grid)' in query
    assert 'count == 2' in cell_spans
    assert 'count == 4' in cell_spans
    assert 'pair_a' in cell_spans and 'pair_b' in cell_spans


def test_v115_canonical_spline_losses_are_training_authority() -> None:
    """The existing graph/node/tangent losses must be in the actual B1 total."""
    import inspect

    from v9.local_boundary_production_contract import LocalBoundaryProductionContract

    source = inspect.getsource(LocalBoundaryProductionContract._local_compute_losses)
    for name in (
        'spline_graph_topology_control', 'spline_graph_topology_sign',
        'spline_graph_point', 'spline_graph_tangent',
        'spline_graph_span_smoothness', 'spline_graph_span_tangent',
        'spline_graph_span_separation', 'spline_graph_sdf',
        'spline_graph_gradient', 'spline_graph_eikonal',
        'spline_graph_curvature', 'spline_metric_offset',
        'spline_metric_eikonal_near',
    ):
        assert f'losses["{name}"]' in source


def test_v115_spline_graph_representation_removes_shallow_line_and_circle_faceting() -> None:
    """Untrained graph geometry itself is smooth at the exact production 24->96 scale."""
    import math
    from types import SimpleNamespace

    import numpy as np
    import torch
    from torch.nn import functional as F

    from v9.parametric_boundary import make_query_grid
    from v9.spline_graph import ConnectedSplineGraph

    config = SimpleNamespace(
        spline_graph_hidden_channels=24,
        spline_graph_control_scale=2,
        target_scale=4,
        contour_sdf_max_distance_pixels=24.0,
        spline_graph_max_topology_delta_pixels=8.0,
        spline_graph_topology_edit_band_pixels=4.0,
        spline_graph_max_displacement_pixels=4.0,
        spline_graph_max_tangent_residual=0.75,
        spline_graph_edit_band_pixels=12.0,
        spline_graph_neighbour_radius=2,
        spline_graph_samples_per_span=4,
    )
    decoder = ConnectedSplineGraph(4, config)
    feature = torch.zeros((1, 4, 24, 24), dtype=torch.float32)
    y_lr, x_lr = torch.meshgrid(
        torch.arange(24, dtype=torch.float32),
        torch.arange(24, dtype=torch.float32),
        indexing='ij',
    )
    x_phys = 2.0 + 4.0 * x_lr
    y_phys = 2.0 + 4.0 * y_lr
    grid = make_query_grid(1, 96, 96, device=torch.device('cpu'))

    slope = math.tan(math.radians(1.0))
    source_line = (
        (y_phys - (40.0 + slope * x_phys)) / math.sqrt(1.0 + slope * slope)
    ).clamp(-24.0, 24.0).unsqueeze(0).unsqueeze(0) / 24.0
    line = decoder(feature, feature, source_line, grid)["field"]["phi_pixels"][0, 0]
    line_np = line.detach().numpy()
    crossings = []
    for x in range(96):
        column = line_np[:, x]
        indices = np.where((column[:-1] >= 0.0) != (column[1:] >= 0.0))[0]
        assert len(indices) == 1
        y0 = int(indices[0])
        a, b = float(column[y0]), float(column[y0 + 1])
        t = abs(a) / max(abs(a) + abs(b), 1.0e-12)
        crossings.append(y0 + 0.5 + t)
    exact = 40.0 + slope * (np.arange(96, dtype=np.float32) + 0.5)
    line_jitter = float(np.std(np.asarray(crossings) - exact))
    assert line_jitter <= 0.05

    source_circle = (
        torch.sqrt((x_phys - 48.0).square() + (y_phys - 48.0).square() + 1.0e-8)
        - 30.0
    ).clamp(-24.0, 24.0).unsqueeze(0).unsqueeze(0) / 24.0
    circle = decoder(feature, feature, source_circle, grid)["field"]["phi_pixels"]
    angles = torch.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    radii = torch.linspace(26.0, 34.0, 129)
    sample_x = 48.0 + torch.cos(angles)[:, None] * radii[None, :]
    sample_y = 48.0 + torch.sin(angles)[:, None] * radii[None, :]
    radial_grid = torch.stack(
        (2.0 * sample_x / 96.0 - 1.0, 2.0 * sample_y / 96.0 - 1.0), dim=-1
    ).unsqueeze(0)
    radial = F.grid_sample(
        circle.float(), radial_grid, mode='bilinear', padding_mode='border', align_corners=False
    )[0, 0].detach().numpy()
    recovered = []
    radius_np = radii.numpy()
    for row in radial:
        indices = np.where((row[:-1] < 0.0) & (row[1:] >= 0.0))[0]
        assert len(indices) >= 1
        index = int(indices[0])
        a, b = float(row[index]), float(row[index + 1])
        t = abs(a) / max(abs(a) + abs(b), 1.0e-12)
        recovered.append(radius_np[index] + t * (radius_np[index + 1] - radius_np[index]))
    curve_roughness = float(np.std(np.asarray(recovered)))
    assert curve_roughness <= 0.08
