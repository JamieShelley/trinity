from __future__ import annotations

from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_v124_trainer_contract_declares_explicit_refiner():
    from v9 import FidelityResidualNetV9, V9Config
    from v9 import training as training_module
    from v9.local_boundary_production_contract import (
        install_local_boundary_training_contract,
    )

    # Reproduce the canonical trainer entrypoint ordering. This redirects the
    # legacy V10.7.9 validator/component map to the installed V12 local contract.
    install_local_boundary_training_contract(training_module)

    config = V9Config()
    config.training_activation_checkpointing = False
    model = FidelityResidualNetV9(config)
    contract = model.architecture_contract()

    production = contract["productionComponents"]
    assert production["structural representation"] == "geometry_net.production_structure"
    assert production["explicit geometry refiner"] == (
        "geometry_net.production_structure.geometry_refiner"
    )

    # This is the exact trainer validator that aborted the real Raven run.
    training_module._validate_v992_architecture_contract(contract)

    modules = training_module._production_component_modules(model)
    path, module = modules["explicit geometry refiner"]
    assert path == "geometry_net.production_structure.geometry_refiner"
    assert module is model.geometry_net.production_structure.geometry_refiner


def test_final_qualification_clears_b1a_runtime_refiner_bypass():
    from v9 import FidelityResidualNetV9, V9Config
    from v9 import training as training_module
    from v9.application.backend import (
        TrainingBackend,
        _run_final_qualification_with_runtime_reset,
    )

    backend = TrainingBackend()
    config = V9Config()
    config.training_activation_checkpointing = False
    model = FidelityResidualNetV9(config)

    model.set_phase("sdf-bootstrap")
    structure = model.geometry_net.production_structure
    assert structure._topology_bootstrap_only is True

    # A fresh production load does not persist B1a's runtime-only bypass. Final
    # qualification must reproduce that state before requiring refiner activity.
    backend._prepare_production_runtime(model)
    assert structure._topology_bootstrap_only is False

    wrapped = training_module._training_service._run_final_qualification
    assert wrapped is _run_final_qualification_with_runtime_reset
    assert "<locals>" not in wrapped.__qualname__


def test_b1b_directly_optimizes_real_raven_renderer_smoke_metric():
    from v9 import training as training_module
    from v9.application import backend as backend_module

    backend_module.TrainingBackend()
    renderer_metric = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    original_base = training_module._nsamdr_b1b_base_compute_losses

    def fake_base(_outputs, _batch, _config, _phase):
        return {
            "boundary_photometric": renderer_metric,
            "total": renderer_metric * 0.0,
        }

    try:
        training_module._nsamdr_b1b_base_compute_losses = fake_base
        losses = backend_module._compute_losses_with_b1b_renderer_supervision(
            {},
            {},
            SimpleNamespace(boundary_photometric_weight=2.0),
            "sdf-proof",
        )
        assert torch.equal(
            losses["b1b_renderer_supervision"],
            torch.tensor(4.0, dtype=torch.float32),
        )
        assert torch.equal(losses["total"], torch.tensor(4.0, dtype=torch.float32))
        losses["total"].backward()
        assert torch.equal(
            renderer_metric.grad,
            torch.tensor(2.0, dtype=torch.float32),
        )
    finally:
        training_module._nsamdr_b1b_base_compute_losses = original_base


def test_windows_spawn_can_pickle_training_worker_after_backend_install():
    from v9 import training as training_module
    from v9.application.backend import (
        TrainingBackend,
        _compute_losses_with_b1b_renderer_supervision,
    )

    TrainingBackend()

    assert training_module.compute_losses is _compute_losses_with_b1b_renderer_supervision
    assert "<locals>" not in training_module.compute_losses.__qualname__

    # Windows DataLoader spawn serializes this bound worker initializer. Because
    # it carries the TrainingService singleton, every callback stored on that
    # singleton must itself remain pickle-safe. This reproduces the real failure
    # that previously occurred before Raven's first training batch.
    ForkingPickler.dumps(training_module._data_worker_init)
