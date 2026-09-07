from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().with_name("test_v121_proposal_baseline_regret.py")
text = path.read_text(encoding="utf-8")
old = '''    target_h = proposal_h.detach().clone()\n    target_v = proposal_v.detach().clone()\n    target_h[..., 0] = target_h[..., 0] + 0.25\n    target_v[..., 1] = target_v[..., 1] + 0.25\n    mask_h = torch.ones(proposal_h.shape[:-1], dtype=torch.float32)\n    mask_v = torch.ones(proposal_v.shape[:-1], dtype=torch.float32)\n'''
new = '''    mask_h = outputs["spline_graph_mask_h"][:, 0].detach().float()\n    mask_v = outputs["spline_graph_mask_v"][:, 0].detach().float()\n    assert float(mask_h.sum() + mask_v.sum()) > 0.0\n    target_h = proposal_h.detach().clone()\n    target_v = proposal_v.detach().clone()\n    # Perturb only true source-crossed edges. Non-crossing fractions are\n    # deliberately clamped and are outside the B1b same-edge teacher domain.\n    target_h[..., 0] = target_h[..., 0] + 0.25 * mask_h\n    target_v[..., 1] = target_v[..., 1] + 0.25 * mask_v\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one gradient-probe block, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("adjusted V12.1 gradient probe to active spline edges")
