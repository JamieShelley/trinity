from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_quick_b1b_runs_smoke_then_full_authored_bank_before_verdict():
    from v9.application.configuration import QUICK_WORK_BUDGET

    assert QUICK_WORK_BUDGET["identity_epochs"] == 1
    assert QUICK_WORK_BUDGET["residual_epochs"] == 2
    assert QUICK_WORK_BUDGET["tiles_per_epoch"] == 64

    training = (NEURAL / "v9/training.py").read_text(encoding="utf-8")
    assert "b1b_stage_epoch == 1" in training
    assert "structural_smoke_batch_limit = min(epoch_batch_count, 14)" in training
    assert "structural_smoke_epoch = structural_smoke_batch_limit is not None" in training
    assert "not structural_smoke_epoch" in training

    # Pipeline applies the strict real-Raven C > B verdict only after the trainer
    # has completed the configured sdf-proof budget. With two residual epochs this
    # means 14 smoke tiles first, then the complete 64-tile authored Quick bank.
    pipeline = (NEURAL / "v9/application/pipeline.py").read_text(encoding="utf-8")
    invoke = pipeline.index('stop_after_phase="sdf-proof"')
    verdict = pipeline.index("_run_quick_b1b_smoke(context, latest)", invoke)
    assert invoke < verdict
