from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V9 = ROOT / "tools/nsamdr/neural/v9"
TESTS = ROOT / "tools/nsamdr/tests"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one replacement target, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


edge = V9 / "edge_constrained_spline_graph.py"
replace_once(
    edge,
    '''    The final fraction is clamped to the interior of the same edge, so geometry\n    refinement cannot jump a node into a neighbouring cell while retaining the\n    old topology mask.\n''',
    '''    The observed source crossing is first projected into the representable\n    interior of the same owning edge. The neural branch then predicts a residual\n    around that projected source identity. Forward projection remains hard, while\n    a straight-through derivative preserves B1b learning at projection boundaries.\n''',
)
replace_once(
    edge,
    '''    h_fraction_pred = (\n        h_fraction[:, 0]\n        + torch.tanh(raw_h[:, 0]) * max_displacement_lattice * scale\n    ).clamp(1.0e-3, 1.0 - 1.0e-3)\n    v_fraction_pred = (\n        v_fraction[:, 0]\n        + torch.tanh(raw_v[:, 3]) * max_displacement_lattice * scale\n    ).clamp(1.0e-3, 1.0 - 1.0e-3)\n\n    source_h = torch.stack(\n        (\n            h_x.expand(batch, 1, height, width - 1)[:, 0] + h_fraction[:, 0],\n            h_y.expand(batch, 1, height, width - 1)[:, 0],\n        ),\n        dim=-1,\n    )\n    source_v = torch.stack(\n        (\n            v_x.expand(batch, 1, height - 1, width)[:, 0],\n            v_y.expand(batch, 1, height - 1, width)[:, 0] + v_fraction[:, 0],\n        ),\n        dim=-1,\n    )\n''',
    '''    # The representation uses an open-edge feasible interval to avoid vertex\n    # degeneracy in connected Hermite cells. Project the observed source crossing\n    # into that same feasible set first, so zero neural correction is exact identity\n    # in the representation used by both B1b supervision and the explicit refiner.\n    epsilon = 1.0e-3\n    source_h_fraction = h_fraction[:, 0].clamp(epsilon, 1.0 - epsilon)\n    source_v_fraction = v_fraction[:, 0].clamp(epsilon, 1.0 - epsilon)\n    unconstrained_h_fraction = (\n        source_h_fraction\n        + torch.tanh(raw_h[:, 0]) * max_displacement_lattice * scale\n    )\n    unconstrained_v_fraction = (\n        source_v_fraction\n        + torch.tanh(raw_v[:, 3]) * max_displacement_lattice * scale\n    )\n    projected_h_fraction = unconstrained_h_fraction.clamp(epsilon, 1.0 - epsilon)\n    projected_v_fraction = unconstrained_v_fraction.clamp(epsilon, 1.0 - epsilon)\n\n    # Hard projection owns the forward geometry. The detached correction makes the\n    # local derivative identity-valued, so a proposal sitting exactly on a feasible\n    # boundary can still learn an inward residual instead of receiving zero gradient.\n    h_fraction_pred = unconstrained_h_fraction + (\n        projected_h_fraction - unconstrained_h_fraction\n    ).detach()\n    v_fraction_pred = unconstrained_v_fraction + (\n        projected_v_fraction - unconstrained_v_fraction\n    ).detach()\n\n    source_h = torch.stack(\n        (\n            h_x.expand(batch, 1, height, width - 1)[:, 0] + source_h_fraction,\n            h_y.expand(batch, 1, height, width - 1)[:, 0],\n        ),\n        dim=-1,\n    )\n    source_v = torch.stack(\n        (\n            v_x.expand(batch, 1, height - 1, width)[:, 0],\n            v_y.expand(batch, 1, height - 1, width)[:, 0] + source_v_fraction,\n        ),\n        dim=-1,\n    )\n''',
)

path = TESTS / "test_v121_proposal_baseline_regret.py"
text = path.read_text(encoding="utf-8")
old = '''    target_h = proposal_h.detach().clone()\n    target_v = proposal_v.detach().clone()\n    target_h[..., 0] = target_h[..., 0] + 0.25\n    target_v[..., 1] = target_v[..., 1] + 0.25\n    mask_h = torch.ones(proposal_h.shape[:-1], dtype=torch.float32)\n    mask_v = torch.ones(proposal_v.shape[:-1], dtype=torch.float32)\n'''
new = '''    mask_h = outputs["spline_graph_mask_h"][:, 0].detach().float()\n    mask_v = outputs["spline_graph_mask_v"][:, 0].detach().float()\n    assert float(mask_h.sum() + mask_v.sum()) > 0.0\n\n    # Zero-initialized B1b is exact identity relative to the projected same-edge\n    # source geometry, including vertex-aligned source crossings.\n    identity_error = (\n        ((proposal_h.detach() - source_h).abs().sum(dim=-1) * mask_h).sum()\n        + ((proposal_v.detach() - source_v).abs().sum(dim=-1) * mask_v).sum()\n    )\n    assert identity_error.item() == 0.0\n\n    target_h = source_h.clone()\n    target_v = source_v.clone()\n    h_fraction = source_h[..., 0] - torch.floor(source_h[..., 0])\n    v_fraction = source_v[..., 1] - torch.floor(source_v[..., 1])\n    h_direction = torch.where(h_fraction > 0.5, -torch.ones_like(h_fraction), torch.ones_like(h_fraction))\n    v_direction = torch.where(v_fraction > 0.5, -torch.ones_like(v_fraction), torch.ones_like(v_fraction))\n    target_h[..., 0] = target_h[..., 0] + 0.25 * h_direction * mask_h\n    target_v[..., 1] = target_v[..., 1] + 0.25 * v_direction * mask_v\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one gradient-probe setup block, found {text.count(old)}")
text = text.replace(old, new, 1)
old_tail = '''    model.zero_grad(set_to_none=True)\n    (point + regret).backward()\n    geometry_head = model.geometry_net.production_structure.spline_graph.geometry_head\n    grad_total = sum(\n        float(parameter.grad.detach().abs().sum())\n        for parameter in geometry_head.parameters()\n        if parameter.grad is not None\n    )\n    assert grad_total > 0.0\n'''
new_tail = '''    geometry_head = model.geometry_net.production_structure.spline_graph.geometry_head\n    last_bias = geometry_head[-1].bias\n\n    # This synthetic contour intentionally lands on a projected edge boundary.\n    # Prove that the hard feasible projection still transmits position gradient.\n    boundary_h = (mask_h > 0.5) & ((h_fraction < 0.002) | (h_fraction > 0.998))\n    boundary_v = (mask_v > 0.5) & ((v_fraction < 0.002) | (v_fraction > 0.998))\n    assert bool(boundary_h.any() or boundary_v.any())\n    if bool(boundary_h.any()):\n        index = torch.nonzero(boundary_h, as_tuple=False)[0]\n        selected = proposal_h[index[0], index[1], index[2], 0]\n    else:\n        index = torch.nonzero(boundary_v, as_tuple=False)[0]\n        selected = proposal_v[index[0], index[1], index[2], 1]\n    selected_grad = torch.autograd.grad(\n        selected, last_bias, retain_graph=True, allow_unused=True\n    )[0]\n    assert selected_grad is not None and selected_grad.abs().sum().item() > 0.0\n\n    model.zero_grad(set_to_none=True)\n    (point + regret).backward()\n    grad_total = sum(\n        float(parameter.grad.detach().abs().sum())\n        for parameter in geometry_head.parameters()\n        if parameter.grad is not None\n    )\n    assert grad_total > 0.0\n'''
if text.count(old_tail) != 1:
    raise RuntimeError(f"expected one gradient-probe tail, found {text.count(old_tail)}")
path.write_text(text.replace(old_tail, new_tail, 1), encoding="utf-8")
print("applied V12.1 projected-source identity and straight-through edge projection")
