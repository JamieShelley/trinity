"""Static checks for the three V12.5 GUI Raven diagnostics."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
PARALLEL = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic_v3.py"
PARALLEL_ENTRY = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic.py"
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic_v3.py"
MICRO_ENTRY = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"
LAYOUT = ROOT / "tools/nsamdr/neural/v9/diagnostics_layout.py"


class TestV125RavenDiagnosticContract:
    # Purpose: Keep the GUI suite to the requested three diagnostic stages.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_gui_has_exact_three_diagnostic_stages(self) -> None:
        source = GUI.read_text(encoding="utf-8")
        assert '"1",\n    "Direct Residual Capacity"' in source
        assert '"2",\n    "Parallel Detail Integration"' in source
        assert '"3",\n    "Raven Staged Micro"' in source
        assert 'self._value("direct_steps", "3072")' in source
        assert 'self._value("parallel_detail_steps", "3072")' in source

    # Purpose: Ensure Test 2 isolates G/S authority so V12.5 U must equal D.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_parallel_detail_uses_v125_detail_only_fusion(self) -> None:
        source = PARALLEL.read_text(encoding="utf-8")
        entry = PARALLEL_ENTRY.read_text(encoding="utf-8")
        assert 'REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V3"' in source
        assert "gate_override=zero_authority" in source
        assert "seam_authority_override=zero_authority" in source
        assert "detail_evidence" in source
        assert "run_nsamdr_v9_raven_parallel_detail_diagnostic_v3 import main" in entry

    # Purpose: Ensure Test 3 exposes the full B/G/S/D/U/F composition instead of old serial panels.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_micro_exposes_parallel_specialist_chain(self) -> None:
        source = MICRO.read_text(encoding="utf-8")
        entry = MICRO_ENTRY.read_text(encoding="utf-8")
        for label in (
            '"G STRUCTURE"',
            '"S SEAM"',
            '"D DETAIL"',
            '"U FUSED"',
            '"F FINAL"',
        ):
            assert label in source
        assert '"fusedCandidate", "fused_candidate_albedo"' in source
        assert "run_nsamdr_v9_raven_micro_diagnostic_v3 import main" in entry

    # Purpose: Prevent asynchronous image viewers from receiving a temporary legacy path.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_preview_launcher_redirects_to_canonical_path(self) -> None:
        source = LAYOUT.read_text(encoding="utf-8")
        assert "def _canonical_result_launcher(" in source
        assert "return canonical_root / relative" in source
        assert "with _canonical_result_launcher(repo_root, legacy_folder, category):" in source
