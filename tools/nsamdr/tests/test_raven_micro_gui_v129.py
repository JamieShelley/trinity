"""Contract checks for the V12.9 Raven Staged Micro GUI wiring."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"


def test_micro_gui_exposes_v129_hard_stage_budgets() -> None:
    source = GUI.read_text(encoding="utf-8")
    assert "V12.9 hard-gated qualification ladder" in source
    for flag in (
        "--geometry-topology-steps",
        "--geometry-steps",
        "--profile-steps",
        "--seam-capacity-steps",
        "--seam-authority-steps",
        "--detail-capacity-steps",
        "--selector-capacity-steps",
    ):
        assert flag in source
    assert 'self._value("micro_geometry_topology_steps", "512")' in source
    assert 'self._value("micro_geometry_steps", "3072")' in source
    assert 'self._value("micro_profile_steps", "1536")' in source
    assert 'self._value("micro_seam_capacity_steps", "1536")' in source
    assert 'self._value("micro_seam_authority_steps", "1024")' in source
    assert 'self._value("micro_detail_capacity_steps", "3072")' in source
    assert 'self._value("micro_selector_capacity_steps", "1024")' in source


def test_gui_states_hard_stop_semantics() -> None:
    source = GUI.read_text(encoding="utf-8")
    assert "A failed stage stops immediately; downstream specialists cannot hide it" in source
    assert "target-SDF teacher geometry and learned-G profile" in source
