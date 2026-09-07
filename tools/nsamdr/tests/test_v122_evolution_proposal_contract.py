from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def _line_sample(torch, size_lr: int, max_distance: float):
    inputs = torch.zeros((1, 17, size_lr, size_lr), dtype=torch.float32)
    xx_lr = torch.arange(size_lr, dtype=torch.float32).view(1, 1, 1, size_lr)
    source_x_lr = float(size_lr) * 0.5 - 0.35
    source_lr_pixels = (xx_lr - source_x_lr).expand(1, 1, size_lr, size_lr)
    inputs[:, 16:17] = (source_lr_pixels / max(max_distance / 4.0, 1.0)).clamp(-1.0, 1.0)

    size_hr = size_lr * 4
    xx_hr = torch.arange(size_hr, dtype=torch.float32).view(1, 1, 1, size_hr)
    source_x_hr = source_x_lr * 4.0
    target_x_hr = source_x_hr + 0.75
    source = (xx_hr - source_x_hr).expand(1, 1, size_hr, size_hr)
    target = (xx_hr - target_x_hr).expand(1, 1, size_hr, size_hr)
    return inputs, {
        "source_sdf": (source / max_distance).clamp(-1.0, 1.0),
        "target_sdf": (target / max_distance).clamp(-1.0, 1.0),
    }


def test_v12_evolution_trains_neural_proposal_not_detached_refined_sdf():
    torch = pytest.importorskip("torch")
    from v9 import FidelityResidualNetV9, V9Config
    from v9.evolution.fitness import StructuralObjective

    config = V9Config()
    config.training_activation_checkpointing = False
    config.spline_refiner_steps = 1
    model = FidelityResidualNetV9(config).train()
    max_distance = float(config.contour_sdf_max_distance_pixels)
    inputs, sample = _line_sample(torch, 12, max_distance)

    geometry = model.geometry_net(inputs)
    assert geometry["spline_proposal_control_point_h_lr"].requires_grad
    assert geometry["spline_proposal_control_point_v_lr"].requires_grad
    # V12 final geometry is explicit-refiner authority and intentionally detached.
    assert not geometry["spline_control_point_h_lr"].requires_grad
    assert not geometry["spline_control_point_v_lr"].requires_grad

    loss, metrics = StructuralObjective(config).evaluate(geometry, sample, max_distance)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()

    geometry_head = model.geometry_net.production_structure.spline_graph.geometry_head
    gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in geometry_head.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0.0
    assert math.isfinite(metrics["proposalPoint"])
    assert math.isfinite(metrics["proposalRegret"])


def test_v12_evolution_does_not_report_software_failure_as_capacity_failure():
    from v9.evolution.controller import EvolutionaryRecoveryController
    from v9.evolution.domain import CandidateResult, Genome

    failed = CandidateResult(
        generation=0,
        index=0,
        genome=Genome(),
        finite=False,
        train_loss_before=math.inf,
        train_loss_after=math.inf,
        source_band_mae=math.inf,
        predicted_band_mae=math.inf,
        relative_gain=-math.inf,
        sign_regression=math.inf,
        gradient_mae=math.inf,
        correction_rms=math.inf,
        fitness=-math.inf,
        elapsed_seconds=0.1,
        passed_microproof=False,
        topology_regression_fraction=math.inf,
        error="RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn",
    )
    controller = object.__new__(EvolutionaryRecoveryController)
    with pytest.raises(RuntimeError, match="did not execute"):
        controller._select_result(Genome(), 0, [failed])
