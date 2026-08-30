"""Historical whole-tile B1b substages are retired by V11.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestB1bStagedContract:
    # Purpose: Implement test no b1b installer in production entrypoint for TestB1bStagedContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_no_b1b_installer_in_production_entrypoint(self) -> None:
        source = ENTRY.read_text(encoding="utf-8")
        assert "install_b1b_staged_contract" not in source
        assert "install_local_boundary_training_contract" in source

    # Purpose: Implement test no classifier generalisation installer for TestB1bStagedContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_no_classifier_generalisation_installer(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        assert "classifier_generalisation" not in source
