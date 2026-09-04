from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_quick_b1a_is_one_epoch_before_baseline_relative_verdict():
    from v9.application.configuration import QUICK_WORK_BUDGET

    assert QUICK_WORK_BUDGET["identity_epochs"] == 1


def test_exp0003_style_real_raven_regression_is_rejected():
    from v9.application.baseline_relative_smoke import BaselineRelativeSmokeService

    validation = {
        "sdf_stageb_baseline_mae": 0.068426,
        "sdf_stageb_renderer_mae": 0.142875,
        "sdf_stageb_renderer_improvement": -1.084767,
        "improvement_fraction": 0.18866,
        "regression_fraction": 0.768524,
    }
    config = SimpleNamespace(maximum_validation_regression_fraction=0.08)

    service = BaselineRelativeSmokeService()
    assert not service.passed(validation, config)


def test_real_raven_candidate_must_beat_baseline_and_respect_regression_limit():
    from v9.application.baseline_relative_smoke import BaselineRelativeSmokeService

    config = SimpleNamespace(maximum_validation_regression_fraction=0.08)
    service = BaselineRelativeSmokeService()
    passing = {
        "sdf_stageb_baseline_mae": 0.070,
        "sdf_stageb_renderer_mae": 0.060,
        "sdf_stageb_renderer_improvement": 0.142857,
        "improvement_fraction": 0.70,
        "regression_fraction": 0.05,
    }
    worse_candidate = dict(passing, sdf_stageb_renderer_mae=0.071)
    too_many_regressions = dict(passing, regression_fraction=0.081)

    assert service.passed(passing, config)
    assert not service.passed(worse_candidate, config)
    assert not service.passed(too_many_regressions, config)


def test_quick_pipeline_rejects_failed_b1a_smoke_before_b1b():
    from v9.application.pipeline import PassDrivenPipeline

    source = inspect.getsource(PassDrivenPipeline._run_stage)
    smoke = inspect.getsource(PassDrivenPipeline._run_quick_b1a_smoke)
    assert 'stop_after_phase="sdf-bootstrap"' in smoke
    assert 'stop_after_phase="sdf-proof"' in source
    assert "if smoke_code != 0" in source
    assert "return smoke_latest, current_resume, smoke_code" in source
    assert "REJECTED before B1b" in smoke
    assert "baselineRelativeSmokeMetrics" in smoke

def test_local_structural_objective_trains_against_baseline_regret():
    from v9.local_boundary_production_contract import LocalBoundaryProductionContract

    source = inspect.getsource(LocalBoundaryProductionContract._local_compute_losses)
    assert 'baseline_relative_supervision = (' in source
    assert 'losses["sdf_improvement_regret"] * float(config.sdf_improvement_regret_weight)' in source
    assert 'losses["geometry_regret"] * float(config.geometry_regret_weight)' in source
    assert 'losses["boundary_pixel_regret"] * float(config.boundary_pixel_regret_weight)' in source
    assert 'total = total + baseline_relative_supervision' in source
    assert source.count('losses["sdf_improvement_regret"] * float(config.sdf_improvement_regret_weight)') == 1
