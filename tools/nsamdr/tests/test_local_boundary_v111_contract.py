"""Static contracts for the V11.1 local-boundary production override."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestLocalBoundaryV111Contract:
    # Purpose: Implement test current v11 is patched not replaced for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_current_v11_is_patched_not_replaced(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert "_ORIGINAL_GEOMETRY_INIT(self, config)" in source
        assert "requires the current V11 model.py" in source
        assert "parametric_primitive_field" in source

    # Purpose: Implement test structural authority is local decoder for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_structural_authority_is_local_decoder(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert "LocalParametricBoundaryDecoder" in source
        assert 'production["structural representation"] = "geometry_net.local_boundary_decoder"' in source
        assert '"wholeTilePrimitiveClassifierAuthority": False' in source

    # Purpose: Implement test complete production graph is preserved for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_complete_production_graph_is_preserved(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        for path in (
            "model.boundary_renderer",
            "model.boundary_specialist",
            "model.seam_restorer.phase_sr",
            "model.seam_restorer.authority",
            "model.detail_net",
            "model.detail_net.albedo_head",
            "model.detail_net.normal_head",
            "model.detail_net.material_head",
            "model.detail_net.confidence_head",
            "model.detail_net.regret_head",
            "model.benefit_selector",
        ):
            assert path in source

    # Purpose: Implement test local losses are existing physical geometry losses for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_local_losses_are_existing_physical_geometry_losses(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        for name in (
            'losses["sdf_surface"]',
            'losses["sdf_sign"]',
            'losses["sdf_topology_sign"]',
            'losses["sdf_eikonal"]',
            'losses["sdf_metric_gradient"]',
            'losses["sdf_improvement_regret"]',
            'losses["parametric_anchor"]',
            'losses["parametric_normal"]',
            'losses["parametric_curvature"]',
            'losses["parametric_offset_smoothness"]',
        ):
            assert name in source

    # Purpose: Implement test quick has no whole tile b1b epoch budget for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_quick_has_no_whole_tile_b1b_epoch_budget(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        quick = source[source.index("QUICK_WORK_BUDGET"):source.index("FULL_MINIMUM_WORK_BUDGET")]
        assert '"identity_epochs": 3' in quick
        assert '"residual_epochs": 0' in quick
        assert '"tiles_per_epoch": 64' in quick

    # Purpose: Implement test pass driven plan skips retired sdf proof stage for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_pass_driven_plan_skips_retired_sdf_proof_stage(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        plan = source[source.index("_PASS_DRIVEN_STAGE_PLAN"):source.index("def _stage_already_qualified")]
        assert '"sdf-bootstrap"' in plan
        assert '"sdf-proof"' not in plan
        assert '"seam-proof"' in plan

    # Purpose: Implement test local gate does not consume primitive classifier metrics for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_local_gate_does_not_consume_primitive_classifier_metrics(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        gate = source[source.index("def _local_geometry_gate"):source.index("def _promote_local_geometry_state")]
        assert "primitive_class_accuracy" not in gate
        assert "primitive_teacher_param_mae" not in gate
        assert "sdf_stageb_topology_regression_fraction" in gate
        assert "sdf_zero_contour_relative_gain_mean" in gate
        assert "sdf_oracle_render_band_mae_mean" in gate

    # Purpose: Implement test package init installs one local model contract for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_package_init_installs_one_local_model_contract(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        assert "install_local_boundary_model_contract()" in source
        assert "install_b1a_parametric_bootstrap" not in source
        assert "install_classifier_generalisation_contract" not in source

    # Purpose: Implement test local promotion does not force retired classifier gates for TestLocalBoundaryV111Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_local_promotion_does_not_force_retired_classifier_gates(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        promotion = source[source.index("def _promote_local_geometry_state"):source.index("_PASS_DRIVEN_STAGE_PLAN")]
        assert 'state["b1b_classifier_qualified"] = True' not in promotion
        assert 'state["b1b_parameters_qualified"] = True' not in promotion
