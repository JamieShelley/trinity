"""Historical whole-tile B1b substages are retired by V11.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


def test_no_b1b_installer_in_production_entrypoint() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    assert "install_b1b_staged_contract" not in source
    assert "install_local_boundary_training_contract" in source


def test_no_classifier_generalisation_installer() -> None:
    source = INIT.read_text(encoding="utf-8")
    assert "classifier_generalisation" not in source
