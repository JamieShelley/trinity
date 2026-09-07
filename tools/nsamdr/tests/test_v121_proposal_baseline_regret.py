from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_identity_has_zero_regret_and_improvement_has_no_regret():
    from v9.edge_constrained_spline_graph import _baseline_relative_point_objective

    source_h = torch.tensor([[[[0.0, 0.0]]]])
    source_v = torch.tensor([[[[0.0, 0.0]]]])
    target_h = torch.tensor([[[[1.0, 0.0]]]])
    target_v = torch.tensor([[[[0.0, 1.0]]]])
    mask_h = torch.ones((1, 1, 1))
    mask_v = torch.ones((1, 1, 1))

    point, regret, gain, wins = _baseline_relative_point_objective(
        source_h, source_v, source_h, source_v, target_h, target_v, mask_h, mask_v
    )
    assert point.item() == 1.0
    assert regret.item() == 0.0
    assert gain.item() == 0.0
    assert wins.item() == 0.0

    better_h = torch.tensor([[[[0.5, 0.0]]]], requires_grad=True)
    better_v = torch.tensor([[[[0.0, 0.5]]]], requires_grad=True)
    point, regret, gain, wins = _baseline_relative_point_objective(
        better_h, better_v, source_h, source_v, target_h, target_v, mask_h, mask_v
    )
    assert point.item() == 0.5
    assert regret.item() == 0.0
    assert gain.item() == 0.5
    assert wins.item() == 1.0


def test_regressing_proposal_gets_positive_gradient_bearing_regret():
    from v9.edge_constrained_spline_graph import _baseline_relative_point_objective

    source_h = torch.tensor([[[[0.0, 0.0]]]])
    source_v = torch.tensor([[[[0.0, 0.0]]]])
    target_h = torch.tensor([[[[1.0, 0.0]]]])
    target_v = torch.tensor([[[[0.0, 1.0]]]])
    proposal_h = torch.tensor([[[[-0.5, 0.0]]]], requires_grad=True)
    proposal_v = torch.tensor([[[[0.0, -0.5]]]], requires_grad=True)
    mask_h = torch.ones((1, 1, 1))
    mask_v = torch.ones((1, 1, 1))

    _point, regret, gain, wins = _baseline_relative_point_objective(
        proposal_h, proposal_v, source_h, source_v, target_h, target_v, mask_h, mask_v
    )
    assert regret.item() == 0.5
    assert gain.item() == -0.5
    assert wins.item() == 0.0
    regret.backward()
    assert proposal_h.grad is not None and proposal_h.grad.abs().sum().item() > 0.0
    assert proposal_v.grad is not None and proposal_v.grad.abs().sum().item() > 0.0


def test_b1b_total_consumes_proposal_space_regret_under_existing_point_weight():
    local = (NEURAL / "v9/local_boundary_production_contract.py").read_text(encoding="utf-8")
    edge = (NEURAL / "v9/edge_constrained_spline_graph.py").read_text(encoding="utf-8")
    assert 'result["spline_graph_point_regret"]' in edge
    assert 'losses["spline_graph_point"] + losses["spline_graph_point_regret"]' in local
    assert 'float(config.spline_graph_point_weight)' in local
    assert "deterministic same-edge source crossing" in local


def test_model_proposal_objective_reaches_geometry_head():
    from v9 import FidelityResidualNetV9, V9Config
    from v9.edge_constrained_spline_graph import _baseline_relative_point_objective

    config = V9Config()
    config.training_activation_checkpointing = False
    config.spline_refiner_steps = 1
    model = FidelityResidualNetV9(config)
    model.set_phase("sdf-proof")
    model.eval()

    h = w = 16
    inputs = torch.zeros((1, 17, h, w), dtype=torch.float32)
    inputs[:, 0:3] = 0.5
    x = torch.linspace(-1.0, 1.0, w, dtype=torch.float32).view(1, 1, 1, w)
    inputs[:, 16:17] = x.expand(1, 1, h, w)
    outputs = model(inputs)

    proposal_h = outputs["spline_proposal_control_point_h_lr"]
    proposal_v = outputs["spline_proposal_control_point_v_lr"]
    source_h = outputs["spline_source_control_point_h_lr"].detach()
    source_v = outputs["spline_source_control_point_v_lr"].detach()
    mask_h = outputs["spline_graph_mask_h"][:, 0].detach().float()
    mask_v = outputs["spline_graph_mask_v"][:, 0].detach().float()
    assert float(mask_h.sum() + mask_v.sum()) > 0.0

    # Zero-initialized B1b is exact identity relative to the projected same-edge
    # source geometry, including vertex-aligned source crossings.
    identity_error = (
        ((proposal_h.detach() - source_h).abs().sum(dim=-1) * mask_h).sum()
        + ((proposal_v.detach() - source_v).abs().sum(dim=-1) * mask_v).sum()
    )
    assert identity_error.item() == 0.0

    target_h = source_h.clone()
    target_v = source_v.clone()
    h_fraction = source_h[..., 0] - torch.floor(source_h[..., 0])
    v_fraction = source_v[..., 1] - torch.floor(source_v[..., 1])
    h_direction = torch.where(h_fraction > 0.5, -torch.ones_like(h_fraction), torch.ones_like(h_fraction))
    v_direction = torch.where(v_fraction > 0.5, -torch.ones_like(v_fraction), torch.ones_like(v_fraction))
    target_h[..., 0] = target_h[..., 0] + 0.25 * h_direction * mask_h
    target_v[..., 1] = target_v[..., 1] + 0.25 * v_direction * mask_v

    point, regret, _gain, _wins = _baseline_relative_point_objective(
        proposal_h, proposal_v, source_h, source_v, target_h, target_v, mask_h, mask_v
    )
    geometry_head = model.geometry_net.production_structure.spline_graph.geometry_head
    last_bias = geometry_head[-1].bias

    # This synthetic contour intentionally lands on a projected edge boundary.
    # Prove that the hard feasible projection still transmits position gradient.
    boundary_h = (mask_h > 0.5) & ((h_fraction < 0.002) | (h_fraction > 0.998))
    boundary_v = (mask_v > 0.5) & ((v_fraction < 0.002) | (v_fraction > 0.998))
    assert bool(boundary_h.any() or boundary_v.any())
    if bool(boundary_h.any()):
        index = torch.nonzero(boundary_h, as_tuple=False)[0]
        selected = proposal_h[index[0], index[1], index[2], 0]
    else:
        index = torch.nonzero(boundary_v, as_tuple=False)[0]
        selected = proposal_v[index[0], index[1], index[2], 1]
    selected_grad = torch.autograd.grad(
        selected, last_bias, retain_graph=True, allow_unused=True
    )[0]
    assert selected_grad is not None and selected_grad.abs().sum().item() > 0.0

    model.zero_grad(set_to_none=True)
    (point + regret).backward()
    grad_total = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in geometry_head.parameters()
        if parameter.grad is not None
    )
    assert grad_total > 0.0
