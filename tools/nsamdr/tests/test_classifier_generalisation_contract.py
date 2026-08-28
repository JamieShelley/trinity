"""Whole-tile classifier tuning is no longer production authority."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"


def test_whole_tile_classifier_is_explicitly_retired() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    assert '"wholeTilePrimitiveClassifierAuthority": False' in source
    assert "self.geometry_net.parametric_primitive_field, False" in source


def test_structural_path_is_auditable_local_wrapper() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    assert '"geometry_net.production_structure"' in source
    assert "LocalParametricBoundaryDecoder" in source
