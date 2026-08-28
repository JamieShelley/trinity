"""Quick local structure must fail fast before expensive downstream work."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"


def test_quick_local_gate_precedes_seams() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    plan = source[source.index("_PASS_DRIVEN_STAGE_PLAN"):source.index("def _stage_already_qualified")]
    assert plan.index('"sdf-bootstrap"') < plan.index('"seam-proof"')


def test_no_primitive_accuracy_or_mae_gate_in_local_promotion() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    gate = source[source.index("def _local_geometry_gate"):source.index("def _promote_local_geometry_state")]
    assert "parametric_primitive_class_accuracy_required" not in gate
    assert "parametric_primitive_param_mae_required" not in gate
