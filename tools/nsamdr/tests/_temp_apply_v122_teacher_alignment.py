from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FITNESS = ROOT / "tools/nsamdr/neural/v9/evolution/fitness.py"

text = FITNESS.read_text(encoding="utf-8")
old = '''            "spline_graph_control_phi_pixels",\n            "spline_graph_mask_h",\n'''
new = '''            "spline_graph_control_phi_pixels",\n            "source_sdf_prior_pixels",\n            "spline_graph_mask_h",\n'''
if text.count(old) != 1:
    raise RuntimeError(f"required-output insertion count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        target = sample["target_sdf"].float() * float(max_distance)\n        control = geometry["spline_graph_control_phi_pixels"]\n'''
new = '''        raw_target = sample["target_sdf"].float() * float(max_distance)\n        source_prior = geometry["source_sdf_prior_pixels"].detach().float()\n        if bool(getattr(self.config, "sdf_sign_gauge_invariant", True)):\n            from .. import losses as canonical_losses\n            polarity = canonical_losses._losses_service._sdf_global_polarity(\n                source_prior,\n                raw_target,\n                float(getattr(self.config, "sdf_metric_band_pixels", 6.0)),\n            )\n            target = raw_target * polarity\n        else:\n            target = raw_target\n        control = geometry["spline_graph_control_phi_pixels"]\n'''
if text.count(old) != 1:
    raise RuntimeError(f"target-alignment replacement count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        denom = (mask_h.sum() + mask_v.sum()).clamp_min(1.0)\n        dot_h = (tangent_h * target_tan_h).sum(dim=-1).abs().clamp(0.0, 1.0)\n'''
new = '''        active_teacher = mask_h.sum() + mask_v.sum()\n        active_source = (\n            geometry["spline_graph_mask_h"][:, 0].float().sum()\n            + geometry["spline_graph_mask_v"][:, 0].float().sum()\n        ).clamp_min(1.0)\n        denom = active_teacher.clamp_min(1.0)\n        dot_h = (tangent_h * target_tan_h).sum(dim=-1).abs().clamp(0.0, 1.0)\n'''
if text.count(old) != 1:
    raise RuntimeError(f"teacher-coverage insertion count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''            "proposalTangent": float(tangent.detach().item()),\n        }\n'''
new = '''            "proposalTangent": float(tangent.detach().item()),\n            "teacherCoverage": float((active_teacher / active_source).detach().item()),\n        }\n'''
if text.count(old) != 1:
    raise RuntimeError(f"teacher metric insertion count={text.count(old)}")
text = text.replace(old, new, 1)
FITNESS.write_text(text, encoding="utf-8")
print("applied V12.2 canonical teacher alignment")
