"""The validator-minimum B1b epoch exists in config but is retired in V11.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"


class TestB1bReservedSubstageBudgetContract:
    # Purpose: Implement test quick and full use legal minimum residual slot for TestB1bReservedSubstageBudgetContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_quick_and_full_use_legal_minimum_residual_slot(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        quick = source[source.index("QUICK_WORK_BUDGET"):source.index("FULL_MINIMUM_WORK_BUDGET")]
        full = source[source.index("FULL_MINIMUM_WORK_BUDGET"):source.index("def _utc_now")]
        assert '"residual_epochs": 1' in quick
        assert '"residual_epochs": 1' in full

    # Purpose: Implement test retired slot is skipped only after local gate promotion for TestB1bReservedSubstageBudgetContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_retired_slot_is_skipped_only_after_local_gate_promotion(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        promotion = source[source.index("def _promote_local_geometry_state"):source.index("_PASS_DRIVEN_STAGE_PLAN")]
        assert 'state["completed_epoch"] = max(learned_epoch, retired_end)' in promotion
        assert "whole-tile primitive classifier removed" in promotion
