"""Historical B1a classifier bootstrap is retired by V11.2."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"


class TestB1aParametricBootstrapContract:
    # Purpose: Implement test old b1a classifier bootstrap is not installed for TestB1aParametricBootstrapContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_old_b1a_classifier_bootstrap_is_not_installed(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        assert "install_b1a_parametric_bootstrap" not in source
        assert "install_local_boundary_model_contract()" in source

    # Purpose: Implement test local geometry has direct loss authority for TestB1aParametricBootstrapContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_local_geometry_has_direct_loss_authority(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'losses["parametric_anchor"]' in source
        assert 'losses["sdf_topology_sign"]' in source
