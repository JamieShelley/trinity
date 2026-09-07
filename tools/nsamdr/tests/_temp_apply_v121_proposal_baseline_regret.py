from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V9 = ROOT / "tools/nsamdr/neural/v9"
TESTS = ROOT / "tools/nsamdr/tests"
OLD_SCHEMA = "NSAMDR_RAVEN_PRODUCTION_NEURAL_PROPOSAL_EXPLICIT_REFINER_SPLINE_GRAPH_4X_V12_0_0"
NEW_SCHEMA = "NSAMDR_RAVEN_PRODUCTION_BASELINE_SAFE_NEURAL_PROPOSAL_EXPLICIT_REFINER_SPLINE_GRAPH_4X_V12_1_0"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one replacement target, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


edge = V9 / "edge_constrained_spline_graph.py"
replace_once(edge, OLD_SCHEMA, NEW_SCHEMA)
replace_once(
    edge,
    '''\ndef _compute_losses(\n    outputs: dict[str, torch.Tensor],\n''',
    '''\ndef _baseline_relative_point_objective(\n    proposal_h: torch.Tensor,\n    proposal_v: torch.Tensor,\n    source_h: torch.Tensor,\n    source_v: torch.Tensor,\n    target_h: torch.Tensor,\n    target_v: torch.Tensor,\n    mask_h: torch.Tensor,\n    mask_v: torch.Tensor,\n) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:\n    """Score the neural node proposal as a residual correction over baseline B.\n\n    The deterministic source crossings are the zero-correction baseline.  Direct\n    target supervision still supplies the estimator gradient, while the hinge\n    term adds extra authority only where the proposal is farther from the authored\n    same-edge target than that baseline.  Exact identity therefore has zero regret.\n    """\n    proposal_h_error = (proposal_h.float() - target_h).abs().sum(dim=-1)\n    proposal_v_error = (proposal_v.float() - target_v).abs().sum(dim=-1)\n    source_h_error = (source_h.detach().float() - target_h).abs().sum(dim=-1)\n    source_v_error = (source_v.detach().float() - target_v).abs().sum(dim=-1)\n    denom = (mask_h.sum() + mask_v.sum()).clamp_min(1.0)\n\n    point = (\n        (proposal_h_error * mask_h).sum() + (proposal_v_error * mask_v).sum()\n    ) / denom\n    regret = (\n        (F.relu(proposal_h_error - source_h_error) * mask_h).sum()\n        + (F.relu(proposal_v_error - source_v_error) * mask_v).sum()\n    ) / denom\n    gain = (\n        ((source_h_error - proposal_h_error) * mask_h).sum()\n        + ((source_v_error - proposal_v_error) * mask_v).sum()\n    ) / denom\n    wins = (\n        ((proposal_h_error < source_h_error).float() * mask_h).sum()\n        + ((proposal_v_error < source_v_error).float() * mask_v).sum()\n    ) / denom\n    return point, regret, gain.detach(), wins.detach()\n\n\ndef _compute_losses(\n    outputs: dict[str, torch.Tensor],\n''',
)
replace_once(
    edge,
    '''    spline_mask_h = outputs.get("spline_graph_mask_h")\n    spline_mask_v = outputs.get("spline_graph_mask_v")\n    source_prior = outputs.get("source_sdf_prior_pixels")\n    if any(\n        value is None\n        for value in (\n            spline_control, spline_h, spline_v, spline_tan_h, spline_tan_v,\n            spline_mask_h, spline_mask_v, source_prior,\n        )\n    ):\n''',
    '''    spline_mask_h = outputs.get("spline_graph_mask_h")\n    spline_mask_v = outputs.get("spline_graph_mask_v")\n    spline_src_h = outputs.get("spline_source_control_point_h_lr")\n    spline_src_v = outputs.get("spline_source_control_point_v_lr")\n    source_prior = outputs.get("source_sdf_prior_pixels")\n    if any(\n        value is None\n        for value in (\n            spline_control, spline_h, spline_v, spline_tan_h, spline_tan_v,\n            spline_mask_h, spline_mask_v, spline_src_h, spline_src_v, source_prior,\n        )\n    ):\n''',
)
replace_once(
    edge,
    '''    point_h_error = (spline_h.float() - target_h).abs().sum(dim=-1)\n    point_v_error = (spline_v.float() - target_v).abs().sum(dim=-1)\n    result["spline_graph_point"] = (\n        (point_h_error * mh).sum() + (point_v_error * mv).sum()\n    ) / denom\n''',
    '''    (\n        result["spline_graph_point"],\n        result["spline_graph_point_regret"],\n        result["spline_graph_point_gain"],\n        result["spline_graph_point_win_fraction"],\n    ) = _baseline_relative_point_objective(\n        spline_h, spline_v, spline_src_h, spline_src_v,\n        target_h, target_v, mh, mv,\n    )\n''',
)

local = V9 / "local_boundary_production_contract.py"
replace_once(local, OLD_SCHEMA, NEW_SCHEMA)
replace_once(
    local,
    '''            total = (\n                losses["spline_graph_point"] * float(config.spline_graph_point_weight)\n                + losses["spline_graph_tangent"] * float(config.spline_graph_tangent_weight)\n''',
    '''            # Residual/identity-safe B1b authority: direct point supervision\n            # moves the neural initializer toward authored geometry, while point\n            # regret adds extra cost only when that proposal is worse than the\n            # deterministic same-edge source crossing. No new tuning weight is\n            # introduced; both terms have identical point-error units.\n            total = (\n                (losses["spline_graph_point"] + losses["spline_graph_point_regret"])\n                * float(config.spline_graph_point_weight)\n                + losses["spline_graph_tangent"] * float(config.spline_graph_tangent_weight)\n''',
)
replace_once(
    local,
    '''        # A structural candidate is useful only when it improves on the observed\n        # source/baseline. The canonical loss already computes these differentiable\n        # regret terms on both authored Raven and analytic examples; keep them as\n        # training authority in both B1a and B1b instead of telemetry-only values.\n''',
    '''        # Dense baseline-relative terms remain useful qualification evidence.\n        # In V12.1, B1b's guaranteed gradient-bearing baseline safety lives in the\n        # neural proposal space above; the final explicit-refiner geometry is\n        # intentionally detached from outer SGD.\n''',
)

test_v120 = TESTS / "test_v120_explicit_geometry_refiner_contract.py"
replace_once(test_v120, OLD_SCHEMA, NEW_SCHEMA)

baseline_test = TESTS / "test_v117_baseline_relative_contract.py"
replace_once(baseline_test, OLD_SCHEMA, NEW_SCHEMA)

new_test = TESTS / "test_v121_proposal_baseline_regret.py"
new_test.write_text(
    '''from __future__ import annotations\n\nfrom pathlib import Path\nimport sys\n\nimport torch\n\nROOT = Path(__file__).resolve().parents[3]\nNEURAL = ROOT / "tools/nsamdr/neural"\nif str(NEURAL) not in sys.path:\n    sys.path.insert(0, str(NEURAL))\n\n\ndef test_identity_has_zero_regret_and_improvement_has_no_regret():\n    from v9.edge_constrained_spline_graph import _baseline_relative_point_objective\n\n    source_h = torch.tensor([[[[0.0, 0.0]]]])\n    source_v = torch.tensor([[[[0.0, 0.0]]]])\n    target_h = torch.tensor([[[[1.0, 0.0]]]])\n    target_v = torch.tensor([[[[0.0, 1.0]]]])\n    mask_h = torch.ones((1, 1, 1))\n    mask_v = torch.ones((1, 1, 1))\n\n    point, regret, gain, wins = _baseline_relative_point_objective(\n        source_h, source_v, source_h, source_v, target_h, target_v, mask_h, mask_v\n    )\n    assert point.item() == 1.0\n    assert regret.item() == 0.0\n    assert gain.item() == 0.0\n    assert wins.item() == 0.0\n\n    better_h = torch.tensor([[[[0.5, 0.0]]]], requires_grad=True)\n    better_v = torch.tensor([[[[0.0, 0.5]]]], requires_grad=True)\n    point, regret, gain, wins = _baseline_relative_point_objective(\n        better_h, better_v, source_h, source_v, target_h, target_v, mask_h, mask_v\n    )\n    assert point.item() == 0.5\n    assert regret.item() == 0.0\n    assert gain.item() == 0.5\n    assert wins.item() == 1.0\n\n\ndef test_regressing_proposal_gets_positive_gradient_bearing_regret():\n    from v9.edge_constrained_spline_graph import _baseline_relative_point_objective\n\n    source_h = torch.tensor([[[[0.0, 0.0]]]])\n    source_v = torch.tensor([[[[0.0, 0.0]]]])\n    target_h = torch.tensor([[[[1.0, 0.0]]]])\n    target_v = torch.tensor([[[[0.0, 1.0]]]])\n    proposal_h = torch.tensor([[[[-0.5, 0.0]]]], requires_grad=True)\n    proposal_v = torch.tensor([[[[0.0, -0.5]]]], requires_grad=True)\n    mask_h = torch.ones((1, 1, 1))\n    mask_v = torch.ones((1, 1, 1))\n\n    _point, regret, gain, wins = _baseline_relative_point_objective(\n        proposal_h, proposal_v, source_h, source_v, target_h, target_v, mask_h, mask_v\n    )\n    assert regret.item() == 0.5\n    assert gain.item() == -0.5\n    assert wins.item() == 0.0\n    regret.backward()\n    assert proposal_h.grad is not None and proposal_h.grad.abs().sum().item() > 0.0\n    assert proposal_v.grad is not None and proposal_v.grad.abs().sum().item() > 0.0\n\n\ndef test_b1b_total_consumes_proposal_space_regret_under_existing_point_weight():\n    local = (NEURAL / "v9/local_boundary_production_contract.py").read_text(encoding="utf-8")\n    edge = (NEURAL / "v9/edge_constrained_spline_graph.py").read_text(encoding="utf-8")\n    assert 'result["spline_graph_point_regret"]' in edge\n    assert 'losses["spline_graph_point"] + losses["spline_graph_point_regret"]' in local\n    assert 'float(config.spline_graph_point_weight)' in local\n    assert "deterministic same-edge source crossing" in local\n\n\ndef test_model_proposal_objective_reaches_geometry_head():\n    from v9 import FidelityResidualNetV9, V9Config\n    from v9.edge_constrained_spline_graph import _baseline_relative_point_objective\n\n    config = V9Config()\n    config.training_activation_checkpointing = False\n    config.spline_refiner_steps = 1\n    model = FidelityResidualNetV9(config)\n    model.set_phase("sdf-proof")\n    model.eval()\n\n    h = w = 16\n    inputs = torch.zeros((1, 17, h, w), dtype=torch.float32)\n    inputs[:, 0:3] = 0.5\n    x = torch.linspace(-1.0, 1.0, w, dtype=torch.float32).view(1, 1, 1, w)\n    inputs[:, 16:17] = x.expand(1, 1, h, w)\n    outputs = model(inputs)\n\n    proposal_h = outputs["spline_proposal_control_point_h_lr"]\n    proposal_v = outputs["spline_proposal_control_point_v_lr"]\n    source_h = outputs["spline_source_control_point_h_lr"].detach()\n    source_v = outputs["spline_source_control_point_v_lr"].detach()\n    target_h = proposal_h.detach().clone()\n    target_v = proposal_v.detach().clone()\n    target_h[..., 0] = target_h[..., 0] + 0.25\n    target_v[..., 1] = target_v[..., 1] + 0.25\n    mask_h = torch.ones(proposal_h.shape[:-1], dtype=torch.float32)\n    mask_v = torch.ones(proposal_v.shape[:-1], dtype=torch.float32)\n\n    point, regret, _gain, _wins = _baseline_relative_point_objective(\n        proposal_h, proposal_v, source_h, source_v, target_h, target_v, mask_h, mask_v\n    )\n    model.zero_grad(set_to_none=True)\n    (point + regret).backward()\n    geometry_head = model.geometry_net.production_structure.spline_graph.geometry_head\n    grad_total = sum(\n        float(parameter.grad.detach().abs().sum())\n        for parameter in geometry_head.parameters()\n        if parameter.grad is not None\n    )\n    assert grad_total > 0.0\n''',
    encoding="utf-8",
)

print("applied V12.1 proposal-space baseline regret")
