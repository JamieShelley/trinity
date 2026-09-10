"""Static contract checks for the V12.9 hard-gated Raven Micro ladder."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V5 = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic_v5.py"
COMPAT = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"


def _source() -> str:
    return V5.read_text(encoding="utf-8")


class TestRavenMicroHardQualificationV5:
    def test_source_parses(self) -> None:
        ast.parse(_source())

    def test_compatibility_entry_point_routes_to_v5(self) -> None:
        source = COMPAT.read_text(encoding="utf-8")
        assert "run_nsamdr_v9_raven_micro_diagnostic_v5 import main" in source

    def test_default_capacity_budgets_are_realistic(self) -> None:
        source = _source()
        for text in (
            '"geometryTopology": (512, 64)',
            '"geometryMetric": (3072, 128)',
            '"profile": (1536, 64)',
            '"seamCapacity": (1536, 64)',
            '"seamAuthority": (1024, 64)',
            '"detail": (3072, 128)',
            '"selector": (1024, 64)',
        ):
            assert text in source

    def test_all_upstream_specialists_are_hard_pass_requirements(self) -> None:
        source = _source()
        for flag in (
            "geometryTopologyPass",
            "geometryMetricPass",
            "structureRenderPass",
            "profilePass",
            "seamCapacityPass",
            "seamAuthorityPass",
            "detailPass",
            "fusionPass",
            "selectorPass",
            "protectedPreservationPass",
        ):
            assert f'"{flag}": False' in source
        assert "production_pass = all(pass_flags.values())" in source

    def test_ladder_stops_before_downstream_training_on_failure(self) -> None:
        source = _source()
        checkpoints = (
            ('return stop_report("geometryTopology")', 'name="G1_geometry_metric_render"'),
            ('return stop_report("geometryMetricOrRender")', 'name="P_boundary_profile"'),
            ('return stop_report("profile")', 'name="S0_forced_seam_capacity"'),
            ('return stop_report("seamCapacity")', 'name="S1_seam_authority"'),
            ('return stop_report("seamAuthority")', 'name="D_detail"'),
            ('return stop_report("detail")', 'name="U_fusion"'),
            ('return stop_report("fusion")', 'name="F_selector_preservation"'),
        )
        for stop, next_stage in checkpoints:
            assert source.index(stop) < source.index(next_stage)

    def test_geometry_gate_uses_production_relative_criteria(self) -> None:
        source = _source()
        for field in (
            "sdf_relative_gain_required",
            "sdf_relative_win_fraction",
            "sdf_relative_regression_fraction",
            "sdf_missing_contour_tolerance",
            "topologyRegression",
        ):
            assert field in source
        assert 'item["structureEdgeRecovery"] >= 0.0' in source
        assert 'item["structureGlobalRecovery"] >= 0.0' in source

    def test_profile_proves_teacher_and_learned_geometry(self) -> None:
        source = _source()
        assert "_profile_teacher_metrics" in source
        assert "sdf_override=target_sdf" in source
        assert 'item["teacherGeometryEdgeRecovery"] >= profile_required' in source
        assert 'item["learnedGeometryEdgeRecovery"] >= profile_required' in source

    def test_seam_capacity_is_forced_before_authority(self) -> None:
        source = _source()
        assert "seam_authority_override=ones" in source
        assert "seam_forced_recovery_required" in source
        assert "seam_authority_iou_required" in source
        assert source.index('name="S0_forced_seam_capacity"') < source.index(
            'name="S1_seam_authority"'
        )

    def test_v4_gui_flags_remain_accepted(self) -> None:
        source = _source()
        assert '"--required-fusion-retention"' in source
        assert '"--required-final-retention"' in source
        assert 'else int(DEFAULT_STAGE_BUDGETS["detail"][0])' in source
        assert 'int(DEFAULT_STAGE_BUDGETS["selector"][0])' in source

    def test_final_gate_requires_preservation_and_retention(self) -> None:
        source = _source()
        assert "PROTECTED_PRESERVATION_REQUIRED" in source
        assert 'item["edgeRetention"] >= float(args.required_final_retention)' in source
        assert 'item["globalRetention"] >= float(args.required_final_retention)' in source
        assert 'item["protectedPreservationRate"]' in source
