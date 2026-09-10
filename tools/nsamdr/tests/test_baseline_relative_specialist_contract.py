"""Static contract checks for V12.4 baseline-relative specialist preservation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/baseline_relative_specialist_contract.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"
README = ROOT / "tools/nsamdr/README.md"


class TestBaselineRelativeSpecialistContract:
    # Purpose: Require the approved 99%-100% protected-region preservation range.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_protected_preservation_threshold_is_99_percent(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert "PROTECTED_PRESERVATION_REQUIRED = 0.990" in source
        assert "PROTECTED_DRIFT_TOLERANCE = 1.0 / 255.0" in source
        assert "PROTECTED_BASELINE_ERROR_TOLERANCE = 2.0 / 255.0" in source

    # Purpose: Ensure target-derived preservation supervision is training-only and baseline anchored.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_preservation_supervises_detail_and_final_selector(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'phase not in {"detail-reconstruction", "physical-finetune"}' in source
        assert '"detail_candidate_albedo" if phase == "detail-reconstruction" else "albedo"' in source
        assert 'losses["total"] = losses["total"].float() + penalty.float()' in source
        assert '"inferenceTargetUse": False' in source

    # Purpose: Ensure package installation occurs after the proven V12.3 direct-detail contracts.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_install_order_preserves_parallel_detail_contract(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        parallel = source.index("install_parallel_detail_contract")
        optimisation = source.index("install_parallel_detail_optimization_contract")
        baseline = source.index("install_baseline_relative_specialist_contract")
        assert parallel < optimisation < baseline

    # Purpose: Keep documentation and executable threshold semantics synchronized.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_readme_states_baseline_anchor_and_preservation_gate(self) -> None:
        source = README.read_text(encoding="utf-8")
        assert "protectedPreservationRate >= 0.990" in source
        assert "deterministic baseline" in source.lower()
        assert "BenefitSelector" in source
