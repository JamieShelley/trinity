"""Static checks for the three V12.6 GUI Raven diagnostics."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
PARALLEL = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic_v4.py"
PARALLEL_ENTRY = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic.py"
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic_v4.py"
MICRO_V3 = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic_v3.py"
MICRO_ENTRY = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"
LAYOUT = ROOT / "tools/nsamdr/neural/v9/diagnostics_layout.py"


class TestV126RavenDiagnosticContract:
    def test_gui_has_exact_three_diagnostic_stages(self) -> None:
        source = GUI.read_text(encoding="utf-8")
        assert '"1",\n    "Direct Residual Capacity"' in source
        assert '"2",\n    "Parallel Detail Integration"' in source
        assert '"3",\n    "Raven Staged Micro"' in source
        assert 'self._value("direct_steps", "3072")' in source
        assert 'self._value("parallel_detail_steps", "3072")' in source
        assert 'self._value("micro_detail_capacity_steps", "0")' in source
        assert 'self._value("micro_selector_capacity_steps", "512")' in source
        assert 'self._value("micro_fusion_retention", "0.95")' in source
        assert 'self._value("micro_final_retention", "0.85")' in source

    def test_parallel_detail_uses_v126_detail_only_fusion(self) -> None:
        source = PARALLEL.read_text(encoding="utf-8")
        entry = PARALLEL_ENTRY.read_text(encoding="utf-8")
        assert 'REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V4"' in source
        assert "gate_override=zero_authority" in source
        assert "seam_authority_override=zero_authority" in source
        assert "confidence * (1.0 - regret)" in source
        assert "run_nsamdr_v9_raven_parallel_detail_diagnostic_v4 import main" in entry

    def test_micro_keeps_full_parallel_specialist_probe(self) -> None:
        v3_source = MICRO_V3.read_text(encoding="utf-8")
        source = MICRO.read_text(encoding="utf-8")
        entry = MICRO_ENTRY.read_text(encoding="utf-8")
        for label in (
            '"G STRUCTURE"',
            '"S SEAM"',
            '"D DETAIL"',
            '"U FUSED"',
            '"F FINAL"',
        ):
            assert label in v3_source
        assert '"fusedCandidate", "fused_candidate_albedo"' in v3_source
        assert "v3._write_probe_sheet_v125" in source
        assert "run_nsamdr_v9_raven_micro_diagnostic_v4 import main" in entry

    def test_micro_uses_production_equivalent_detail_budget_and_retention(self) -> None:
        source = MICRO.read_text(encoding="utf-8")
        assert "int(production_reference.tiles_per_epoch)" in source
        assert "selector_epochs = max(int(config.physical_finetune_epochs), 1)" in source
        assert 'elif phase == "detail-reconstruction":' in source
        assert 'elif phase == "physical-finetune":' in source
        assert "detail_steps_per_epoch" in source
        assert "selector_steps_per_epoch" in source
        assert "fusion_edge_retention = _retention(d_edge, u_edge)" in source
        assert "final_edge_retention = _retention(u_edge, f_edge)" in source
        assert "protected_preservation_terms(" in source
        assert "PROTECTED_PRESERVATION_REQUIRED" in source
        assert "final_edge >=" not in source

    def test_preview_launcher_redirects_to_canonical_path(self) -> None:
        source = LAYOUT.read_text(encoding="utf-8")
        assert "def _canonical_result_launcher(" in source
        assert "return canonical_root / relative" in source
        assert "with _canonical_result_launcher(repo_root, legacy_folder, category):" in source
