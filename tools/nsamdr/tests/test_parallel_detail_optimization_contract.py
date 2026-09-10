from __future__ import annotations

from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import ast
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
V9 = NEURAL / "v9"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_parallel_detail_optimizer_contract_is_package_wide_and_spawn_safe():
    source = (V9 / "parallel_detail_optimization_contract.py").read_text(encoding="utf-8")
    package = (V9 / "__init__.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'PARALLEL_DETAIL_OPTIMIZATION_REVISION = "V12.3.1"' in source
    assert "DETAIL_ALBEDO_HEAD_LR_MULTIPLIER = 3.0" in source
    assert "install_parallel_detail_optimization_contract()" in package
    assert "<locals>" not in source

    from v9.application.backend import TrainingBackend
    import v9.training as training

    TrainingBackend()
    ForkingPickler.dumps(training._training_service._build_optimizer)


def test_production_optimizer_reproduces_direct_detail_body_and_head_regime():
    from v9 import V9Config
    from v9.application.backend import TrainingBackend
    from v9.model import FidelityResidualNetV9
    import v9.training as training

    config = V9Config()
    model = FidelityResidualNetV9(config)
    TrainingBackend()
    optimizer, _mode = training._training_service._build_optimizer(
        model, config, torch.device("cpu")
    )

    parameter_group: dict[int, dict[str, object]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_group[id(parameter)] = group

    head = list(model.detail_net.albedo_head.parameters())
    body = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("detail_net.") and not name.startswith("detail_net.albedo_head.")
    ]
    assert head
    assert body
    assert all(float(parameter_group[id(parameter)]["lr_scale"]) == 3.0 for parameter in head)
    assert all(float(parameter_group[id(parameter)]["weight_decay"]) == 0.0 for parameter in head)
    assert all(float(parameter_group[id(parameter)]["lr_scale"]) == 1.0 for parameter in body)
    assert all(float(parameter_group[id(parameter)]["weight_decay"]) == 0.0 for parameter in body)


def test_detail_phase_lr_combines_with_head_group_to_match_capacity_proof():
    from v9 import V9Config
    from v9.application.backend import TrainingBackend
    import v9.training as training

    config = V9Config()
    TrainingBackend()
    body_lr = float(training._training_service._phase_lr("detail-reconstruction", config, None))
    assert body_lr >= 9.5e-4
    assert body_lr <= 1.05e-3
    assert body_lr * 3.0 >= 2.85e-3
    assert body_lr * 3.0 <= 3.15e-3
