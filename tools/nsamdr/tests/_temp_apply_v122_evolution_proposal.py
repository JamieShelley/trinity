from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FITNESS = ROOT / "tools/nsamdr/neural/v9/evolution/fitness.py"
CANDIDATE = ROOT / "tools/nsamdr/neural/v9/evolution/candidate.py"
CONTROLLER = ROOT / "tools/nsamdr/neural/v9/evolution/controller.py"

text = FITNESS.read_text(encoding="utf-8")
text = text.replace("from typing import Mapping", "from typing import Any, Mapping", 1)
start = text.index("class StructuralObjective:")
end = text.index("\n\nclass StructuralFitness:")
objective = '''class StructuralObjective:
    """V12 proposal-space capacity objective for evolutionary recovery."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def evaluate(
        self,
        geometry: Mapping[str, torch.Tensor],
        sample: Mapping[str, torch.Tensor],
        max_distance: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train the neural spline proposal, never the detached explicit-refiner result.

        V12 deliberately makes final refined geometry parameter-free and detached from
        outer SGD.  The evolutionary capacity proof therefore uses the same authored
        same-edge point/tangent teacher as B1b.  Held-out fitness below still measures
        the final refined SDF, so proposal learning cannot masquerade as qualification.
        """
        from ..edge_constrained_spline_graph import (
            _baseline_relative_point_objective,
            _same_edge_targets,
        )

        required = (
            "spline_graph_control_phi_pixels",
            "spline_graph_mask_h",
            "spline_graph_mask_v",
            "spline_source_control_point_h_lr",
            "spline_source_control_point_v_lr",
            "spline_proposal_control_point_h_lr",
            "spline_proposal_control_point_v_lr",
            "spline_proposal_control_tangent_h",
            "spline_proposal_control_tangent_v",
        )
        missing = [name for name in required if name not in geometry]
        if missing:
            raise RuntimeError(
                "V12 evolutionary capacity proof requires neural proposal outputs: "
                + ", ".join(missing)
            )

        target = sample["target_sdf"].float() * float(max_distance)
        control = geometry["spline_graph_control_phi_pixels"]
        control_scale = float(getattr(self.config, "spline_graph_control_scale", 2))
        control_spacing_hr = 4.0 / max(control_scale, 1.0)
        target_h, target_v, target_tan_h, target_tan_v, valid_h, valid_v = (
            _same_edge_targets(
                target,
                tuple(control.shape[-2:]),
                control_spacing_hr=control_spacing_hr,
                control_origin=2.0,
            )
        )

        proposal_h = geometry["spline_proposal_control_point_h_lr"].float()
        proposal_v = geometry["spline_proposal_control_point_v_lr"].float()
        source_h = geometry["spline_source_control_point_h_lr"].detach().float()
        source_v = geometry["spline_source_control_point_v_lr"].detach().float()
        tangent_h = geometry["spline_proposal_control_tangent_h"].float()
        tangent_v = geometry["spline_proposal_control_tangent_v"].float()
        mask_h = geometry["spline_graph_mask_h"][:, 0].float() * valid_h.float()
        mask_v = geometry["spline_graph_mask_v"][:, 0].float() * valid_v.float()

        point, regret, gain, wins = _baseline_relative_point_objective(
            proposal_h,
            proposal_v,
            source_h,
            source_v,
            target_h,
            target_v,
            mask_h,
            mask_v,
        )
        denom = (mask_h.sum() + mask_v.sum()).clamp_min(1.0)
        dot_h = (tangent_h * target_tan_h).sum(dim=-1).abs().clamp(0.0, 1.0)
        dot_v = (tangent_v * target_tan_v).sum(dim=-1).abs().clamp(0.0, 1.0)
        tangent = (
            ((1.0 - dot_h) * mask_h).sum()
            + ((1.0 - dot_v) * mask_v).sum()
        ) / denom

        point_weight = max(
            float(getattr(self.config, "spline_graph_point_weight", 1.0)), 1.0e-6
        )
        tangent_ratio = float(
            getattr(self.config, "spline_graph_tangent_weight", 1.0)
        ) / point_weight
        total = point + regret + tangent * tangent_ratio
        if not total.requires_grad:
            raise RuntimeError(
                "V12 evolutionary proposal objective is detached from outer SGD"
            )
        metrics = {
            "proposalPoint": float(point.detach().item()),
            "proposalRegret": float(regret.detach().item()),
            "proposalGain": float(gain.detach().item()),
            "proposalWins": float(wins.detach().item()),
            "proposalTangent": float(tangent.detach().item()),
        }
        return total, metrics
'''
FITNESS.write_text(text[:start] + objective + text[end:], encoding="utf-8")

text = CANDIDATE.read_text(encoding="utf-8")
old = "self.objective = objective or StructuralObjective()"
new = "self.objective = objective or StructuralObjective(config)"
if text.count(old) != 1:
    raise RuntimeError(f"candidate objective constructor occurrence count={text.count(old)}")
CANDIDATE.write_text(text.replace(old, new, 1), encoding="utf-8")

text = CONTROLLER.read_text(encoding="utf-8")
old = '''        viable = [item for item in results if item.finite]\n        winner_result = max(viable, key=lambda item: item.fitness) if viable else results[0]\n'''
new = '''        viable = [item for item in results if item.finite]\n        if not viable:\n            execution_errors = sorted({\n                str(item.error) for item in results if item.error\n            })\n            if execution_errors:\n                raise RuntimeError(\n                    "evolution capacity proof did not execute; candidate error(s): "\n                    + " | ".join(execution_errors)\n                )\n        winner_result = max(viable, key=lambda item: item.fitness) if viable else results[0]\n'''
if text.count(old) != 1:
    raise RuntimeError(f"controller viable block occurrence count={text.count(old)}")
CONTROLLER.write_text(text.replace(old, new, 1), encoding="utf-8")

print("applied V12.2 evolution proposal-space capacity proof")
