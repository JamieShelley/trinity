"""Static contract checks for the unified NSAMDR diagnostic artifact hierarchy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAYOUT = ROOT / "tools/nsamdr/neural/v9/diagnostics_layout.py"
DIRECT = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_direct_residual_diagnostic.py"
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"
MICRO_CAPACITY = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_overfit.py"
PARALLEL = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic_v2.py"
README = ROOT / "tools/nsamdr/README.md"


class TestDiagnosticsLayoutContract:
    # Purpose: Require one canonical diagnostic root with explicit category subfolders.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_one_diagnostic_root(self) -> None:
        source = LAYOUT.read_text(encoding="utf-8")
        assert 'DIAGNOSTICS_ROOT = Path("artifacts/nsamdr/diagnostics")' in source
        assert '"micro_diagnostics": "micro"' in source
        assert '"direct_residual_diagnostics": "direct_residual"' in source
        assert '"parallel_detail_diagnostics": "parallel_detail"' in source

    # Purpose: Ensure every active diagnostic entry point consolidates legacy output.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_active_entry_points_use_consolidated_runner(self) -> None:
        for path in (DIRECT, MICRO, MICRO_CAPACITY, PARALLEL):
            source = path.read_text(encoding="utf-8")
            assert "run_consolidated_diagnostic" in source, path.name

    # Purpose: Ensure old local runs are migrated and partial failed-run evidence is retained.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_layout_migrates_before_and_after_runs(self) -> None:
        source = LAYOUT.read_text(encoding="utf-8")
        assert source.count("consolidate_legacy_diagnostics(repo_root)") >= 3
        assert "except BaseException" in source
        assert "preserved" in source

    # Purpose: Keep the user-facing artifact layout documented alongside the runtime rule.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_readme_documents_subfolders(self) -> None:
        source = README.read_text(encoding="utf-8")
        assert "artifacts/nsamdr/diagnostics/" in source
        assert "direct_residual/" in source
        assert "parallel_detail/" in source
