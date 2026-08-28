"""Pass-driven orchestration contracts for V11.3 local-boundary production."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"


def _source() -> str:
    return ENTRY.read_text(encoding="utf-8")


def test_stage_order_is_local_then_full_production_downstream() -> None:
    source = _source()
    plan = source[source.index("_PASS_DRIVEN_STAGE_PLAN"):source.index("def _stage_already_qualified")]
    positions = [
        plan.index('"sdf-bootstrap"'),
        plan.index('"seam-proof"'),
        plan.index('"seam-authority"'),
        plan.index('"gate-proof"'),
        plan.index('"detail-reconstruction"'),
    ]
    assert positions == sorted(positions)
    assert '"sdf-proof"' not in plan


def test_each_stage_is_still_canonical_train_v9() -> None:
    source = _source()
    assert "stop_after_phase=phase" in source
    assert "stop_after_phase=None" in source
    assert "resume=resume_now" in source
    assert "resume=True" in source


def test_local_geometry_promotion_is_fail_closed() -> None:
    source = _source()
    assert "_local_geometry_gate" in source
    assert "_promote_local_geometry_state" in source
    assert '"status": "training-rejected"' in source
    assert "Downstream stages were not run." in source


def test_final_stage_requires_real_production_final() -> None:
    source = _source()
    assert 'latest.get("trainingSafetyPass", False)' in source
    assert '== "production-final"' in source
