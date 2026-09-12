from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_test3_launcher_routes_to_v131_sr_quality() -> None:
    source = (
        ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"
    ).read_text(encoding="utf-8")
    assert "from run_nsamdr_v13_1_raven_sr_diagnostic import main as _v131_main" in source
    assert "--required-gradient-recovery" in source
    assert '"0.50", "0.70"' in source
    assert '"0.25", "0.50"' in source
    assert '"0.85", "0.90"' in source
    assert "run_nsamdr_v9_raven_micro_diagnostic_v8" not in source


def test_gui_exposes_sr_first_visual_fidelity_controls() -> None:
    source = (
        ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
    ).read_text(encoding="utf-8")
    assert '"V13.1 SR Visual Fidelity"' in source
    assert '"V13.3 SR Raven Quick"' in source
    assert '"SR maximum steps"' in source
    assert '"Required edge recovery"' in source
    assert '"Required global recovery"' in source
    assert '"Required selector retention"' in source
    assert '"Only V13.3 SR-first Quick experiments are listed' in source
    assert '"detail_learning_rate", "detailLearningRate"' in source
    assert '"G topology max steps"' not in source
    assert '"Forced seam max steps"' not in source
