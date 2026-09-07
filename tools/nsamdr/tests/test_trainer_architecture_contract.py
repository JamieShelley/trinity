from __future__ import annotations

from pathlib import Path
import sys

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
