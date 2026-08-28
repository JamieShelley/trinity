"""Historical B1a classifier bootstrap is retired by V11.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"


def test_old_b1a_classifier_bootstrap_is_not_installed() -> None:
    source = INIT.read_text(encoding="utf-8")
    assert "install_b1a_parametric_bootstrap" not in source
    assert "install_local_boundary_model_contract()" in source


def test_local_geometry_has_direct_loss_authority() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    assert 'losses["parametric_anchor"]' in source
    assert 'losses["sdf_topology_sign"]' in source
