from __future__ import annotations

from types import SimpleNamespace

import pytest

from v9.config import V9Config
from v9.application.configuration import (
    QUICK_WORK_BUDGET,
    assert_quick_stop_phase,
    assert_sr_first_quick_config,
)
from v9.application.runner import TrainingApplication
from v9 import sr_first_generalization_contract as v132
from v9 import sr_first_runtime_contract as v133


def _quick_config() -> V9Config:
    config = V9Config()
    for key, value in QUICK_WORK_BUDGET.items():
        setattr(config, key, value)
    config.validate()
    return config


def test_current_quick_schedule_is_native_sr_first() -> None:
    config = _quick_config()
    assert_sr_first_quick_config(config)
    assert v132.is_sr_first_quick_config(config)
    assert v133.SR_RUNTIME_REVISION == "V13.3"
    assert config.identity_epochs == 0
    assert config.residual_epochs == 0
    assert config.seam_proof_epochs == 0
    assert config.seam_authority_epochs == 0
    assert config.boundary_epochs == 0
    assert config.detail_epochs == 8
    assert config.physical_finetune_epochs == 3
    assert config.tile_size == 32
    assert config.batch_size == 1
    assert config.detail_learning_rate == pytest.approx(1.0e-3)


def test_pre_v133_quick_schedule_is_rejected() -> None:
    config = _quick_config()
    config.identity_epochs = 1
    with pytest.raises(RuntimeError, match="Pre-V13.3 Quick schedules are retired"):
        assert_sr_first_quick_config(config)


def test_stale_detail_lr_schedule_is_rejected() -> None:
    config = _quick_config()
    config.detail_learning_rate = 1.0e-3 / 3.0
    with pytest.raises(RuntimeError, match="proven V13.3 SR training regime"):
        assert_sr_first_quick_config(config)


def test_retired_hidden_quick_phases_are_rejected() -> None:
    for phase in (
        "sdf-bootstrap",
        "sdf-proof",
        "seam-proof",
        "seam-authority",
        "gate-proof",
        "boundary-hardening",
    ):
        with pytest.raises(RuntimeError, match="retired by V13.3"):
            assert_quick_stop_phase(phase)
    assert_quick_stop_phase(None)
    assert_quick_stop_phase("detail-reconstruction")


def test_full_training_is_rejected_before_allocation() -> None:
    application = object.__new__(TrainingApplication)
    application.options = SimpleNamespace(training_mode="full")
    with pytest.raises(RuntimeError, match="Full Training is disabled"):
        application.run()
