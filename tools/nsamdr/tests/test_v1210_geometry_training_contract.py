"""Source-contract checks for V12.10 direct geometry authority and Raven fail-fast."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V9_INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"
GEOMETRY_CONTRACT = (
    ROOT / "tools/nsamdr/neural/v9/parallel_specialist_geometry_training_contract.py"
)
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic_v7.py"
LAUNCHER = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"


def test_v1210_restores_same_edge_geometry_authority_after_v128() -> None:
    source = GEOMETRY_CONTRACT.read_text(encoding="utf-8")
    assert 'PARALLEL_SPECIALIST_GEOMETRY_TRAINING_REVISION = "V12.10"' in source
    assert '"spline_graph_point"' in source
    assert '"spline_graph_tangent"' in source
    assert '"spline_graph_point_regret"' in source
    assert 'getattr(config, "spline_graph_point_weight", 96.0)' in source
    assert 'getattr(config, "spline_graph_tangent_weight", 48.0)' in source
    assert 'losses["total"] = deployed_before + direct_geometry' in source

    init_source = V9_INIT.read_text(encoding="utf-8")
    safety = init_source.index("install_parallel_specialist_safety_contract")
    geometry = init_source.index("install_parallel_specialist_geometry_training_contract")
    assert safety < geometry


def test_v1210_raven_g1_uses_direct_teacher_and_fail_fast() -> None:
    source = MICRO.read_text(encoding="utf-8")
    for token in (
        "spline_graph_point_gain",
        "spline_graph_point_win_fraction",
        "spline_graph_point_regret",
        "spline_graph_same_edge_teacher_coverage",
        "GEOMETRY_PLATEAU_MIN_STEPS = 512",
        "GEOMETRY_PLATEAU_PATIENCE_EVALS = 4",
        '"status": "failed-plateau"',
        '"stopReason": "geometry-score-plateau"',
    ):
        assert token in source
    # Synthetic-ladder aggregate contour-gain is deliberately not reused as the
    # single Raven patch's mandatory 25% capacity threshold.
    assert "sdf_relative_gain_required" not in source


def test_raven_micro_launcher_routes_to_v1210() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "run_nsamdr_v9_raven_micro_diagnostic_v7 import main" in source
    assert "V12.10 hard-gated staged Micro diagnostic" in source
