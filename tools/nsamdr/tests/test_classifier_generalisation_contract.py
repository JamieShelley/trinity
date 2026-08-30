"""Whole-tile classifier tuning is no longer production authority."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"


class TestClassifierGeneralisationContract:
    # Purpose: Implement test whole tile classifier is explicitly retired for TestClassifierGeneralisationContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_whole_tile_classifier_is_explicitly_retired(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert '"wholeTilePrimitiveClassifierAuthority": False' in source
        assert "self.geometry_net.parametric_primitive_field, False" in source

    # Purpose: Implement test structural path is auditable local wrapper for TestClassifierGeneralisationContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_structural_path_is_auditable_local_wrapper(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert '"geometry_net.production_structure"' in source
        assert "LocalParametricBoundaryDecoder" in source
