from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


local_path = "tools/nsamdr/neural/v9/local_boundary_production_contract.py"
model_path = "tools/nsamdr/neural/v9/model.py"
edge_path = "tools/nsamdr/neural/v9/edge_constrained_spline_graph.py"
test_path = "tools/nsamdr/tests/test_v117_baseline_relative_contract.py"
design_path = "tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md"

replace_once(
    local_path,
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_CONNECTED_SPLINE_GRAPH_4X_V11_5_0"',
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_BASELINE_RESIDUAL_SPLINE_GRAPH_4X_V11_8_0"',
)

replace_once(
    local_path,
    '''        self.topology_feature_project = self._make_feature_project(\n            int(decoded_channels), feature_channels\n        )\n        initial_genome = torch.tensor(\n''',
    '''        self.topology_feature_project = self._make_feature_project(\n            int(decoded_channels), feature_channels\n        )\n        # V11.8 structural output is a residual correction to deterministic B.\n        # A zero-initialized final layer gives exact C == B at construction while\n        # tanh keeps the learned signed correction bounded and differentiable.\n        self.structural_residual_gain_head = nn.Sequential(\n            nn.Conv2d(int(decoded_channels) + 16, feature_channels, 3, padding=1),\n            nn.GELU(),\n            nn.Conv2d(feature_channels, 1, 1),\n        )\n        nn.init.zeros_(self.structural_residual_gain_head[-1].weight)\n        nn.init.zeros_(self.structural_residual_gain_head[-1].bias)\n        initial_genome = torch.tensor(\n''',
)

replace_once(
    local_path,
    '''        geometry_feature_grid = self.geometry_feature_project(projected_input)\n        topology_feature_grid = self.topology_feature_project(projected_input)\n\n        # Retain the V11.4 local context only for checkpoint/public telemetry. It has\n''',
    '''        geometry_feature_grid = self.geometry_feature_project(projected_input)\n        topology_feature_grid = self.topology_feature_project(projected_input)\n        structural_residual_gain = torch.tanh(\n            self.structural_residual_gain_head(projected_input).float()\n        )\n        structural_residual_gain = F.interpolate(\n            structural_residual_gain, size=query_grid.shape[1:3],\n            mode="bilinear", align_corners=False,\n        )\n\n        # Retain the V11.4 local context only for checkpoint/public telemetry. It has\n''',
)

replace_once(
    local_path,
    '''        return {\n            "feature_grid": geometry_feature_grid,\n            "context": context,\n            "spline_graph": spline["graph"],\n            "field": field,\n            "genome": self.evolution_genome,\n        }\n''',
    '''        return {\n            "feature_grid": geometry_feature_grid,\n            "context": context,\n            "spline_graph": spline["graph"],\n            "field": field,\n            "structural_residual_gain": structural_residual_gain,\n            "genome": self.evolution_genome,\n        }\n''',
)

replace_once(
    local_path,
    '''        spline_graph = structure["spline_graph"]\n        field = structure["field"]\n\n        final_pixels = field["phi_pixels"].float()\n''',
    '''        spline_graph = structure["spline_graph"]\n        field = structure["field"]\n        structural_residual_gain = structure["structural_residual_gain"].float()\n\n        final_pixels = field["phi_pixels"].float()\n''',
)

replace_once(
    local_path,
    '''            "local_parametric_confidence": local_context["confidence"].to(aux.dtype),\n            "local_implicit_authority": field["implicit_authority"].to(aux.dtype),\n            "contour_transport_control_pixels": zero_control2,\n''',
    '''            "local_parametric_confidence": local_context["confidence"].to(aux.dtype),\n            "local_implicit_authority": field["implicit_authority"].to(aux.dtype),\n            "structural_residual_gain": structural_residual_gain.to(aux.dtype),\n            "contour_transport_control_pixels": zero_control2,\n''',
)

replace_once(
    local_path,
    '''            "reconstructionPrimitive": "connected marching-squares cubic-Hermite spline graph",\n            "b1bObjective": "shared graph-node, tangent, span-smoothness, metric-SDF and same-renderer reconstruction",\n            "geometryOutputs": (\n                "source_sdf_prior", "connected_spline_graph", "edge", "orientation", "hardness"\n            ),\n            "topologyGeometryFeatureSplit": True,\n''',
    '''            "reconstructionPrimitive": (\n                "deterministic B + zero-initialized bounded structural residual gain "\n                "* (connected marching-squares cubic-Hermite redraw - B)"\n            ),\n            "b1bObjective": "shared graph-node, tangent, span-smoothness, metric-SDF and same-renderer reconstruction",\n            "geometryOutputs": (\n                "source_sdf_prior", "connected_spline_graph", "structural_residual_gain",\n                "edge", "orientation", "hardness"\n            ),\n            "baselineRelativeStructuralIdentity": True,\n            "topologyGeometryFeatureSplit": True,\n''',
)

replace_once(
    model_path,
    '''    # Purpose: Implement forward impl for FidelityResidualNetV9.\n    # Called by: External callers and the owning workflow.\n    # Calls: Same-class helpers where required.\n    def _forward_impl(\n''',
    '''    # Purpose: Compute signed baseline-relative structural residual authority.\n    # Called by: _forward_impl\n    # Calls: No same-class helper methods.\n    @staticmethod\n    def _structural_residual_weight(\n        candidate_locality: torch.Tensor,\n        structural_residual_gain: torch.Tensor | None,\n        gate_override: torch.Tensor | None,\n    ) -> torch.Tensor:\n        """Return the local structural residual weight with exact-zero default authority.\n\n        Production uses a learned signed gain in [-1, 1]. A teacher/oracle gate\n        remains an explicit positive override so proof rendering can still request\n        the complete analytic candidate without changing production semantics.\n        """\n        if gate_override is not None:\n            weight = gate_override.to(\n                device=candidate_locality.device, dtype=torch.float32, non_blocking=True\n            ).clamp(0.0, 1.0)\n            if weight.shape[-2:] != candidate_locality.shape[-2:]:\n                weight = F.interpolate(\n                    weight, size=candidate_locality.shape[-2:],\n                    mode="bilinear", align_corners=False,\n                )\n            return weight * candidate_locality\n\n        if structural_residual_gain is None:\n            return torch.zeros_like(candidate_locality, dtype=torch.float32)\n        gain = structural_residual_gain.to(\n            device=candidate_locality.device, dtype=torch.float32, non_blocking=True\n        )\n        if gain.shape[-2:] != candidate_locality.shape[-2:]:\n            gain = F.interpolate(\n                gain, size=candidate_locality.shape[-2:],\n                mode="bilinear", align_corners=False,\n            )\n        return gain.clamp(-1.0, 1.0) * candidate_locality\n\n    # Purpose: Implement forward impl for FidelityResidualNetV9.\n    # Called by: External callers and the owning workflow.\n    # Calls: Same-class helpers where required.\n    def _forward_impl(\n''',
)

replace_once(
    model_path,
    '''        # Geometry-only candidate. Structural reconstruction remains compact/local\n        # and therefore preserves the true bicubic baseline exactly elsewhere.\n        if gate_override is not None:\n            structural_gate = gate_override.to(\n                device=inputs.device, dtype=torch.float32, non_blocking=True\n            ).clamp(0.0, 1.0)\n            if structural_gate.shape[-2:] != candidate_locality.shape[-2:]:\n                structural_gate = F.interpolate(\n                    structural_gate, size=candidate_locality.shape[-2:], mode="bilinear", align_corners=False\n                )\n            structural_gate = structural_gate * candidate_locality\n        else:\n            structural_gate = candidate_locality\n\n        boundary_albedo = (\n            baseline_albedo.float() * (1.0 - structural_gate)\n            + candidate_albedo.float() * structural_gate\n        ).to(baseline_albedo.dtype)\n        boundary_normal_out = self._normalize_xy((\n            baseline_normal.float() * (1.0 - structural_gate)\n            + candidate_normal.float() * structural_gate\n        ).to(baseline_normal.dtype))\n        boundary_material = (\n            baseline_material.float() * (1.0 - structural_gate)\n            + candidate_material.float() * structural_gate\n        ).to(baseline_material.dtype)\n''',
    '''        # V11.8 structural reconstruction is residual relative to B, not an\n        # unconditional replacement inside every contour band. The gain head starts\n        # at exactly zero, so a fresh model produces C == B until authored-Raven\n        # evidence earns local correction authority.\n        structural_residual_weight = self._structural_residual_weight(\n            candidate_locality, geometry.get("structural_residual_gain"), gate_override\n        )\n        structural_gate = structural_residual_weight.abs().clamp(0.0, 1.0)\n\n        boundary_albedo = (\n            baseline_albedo.float()\n            + structural_residual_weight\n            * (candidate_albedo.float() - baseline_albedo.float())\n        ).clamp(0.0, 1.0).to(baseline_albedo.dtype)\n        boundary_normal_out = self._normalize_xy((\n            baseline_normal.float()\n            + structural_residual_weight\n            * (candidate_normal.float() - baseline_normal.float())\n        ).to(baseline_normal.dtype))\n        boundary_material = (\n            baseline_material.float()\n            + structural_residual_weight\n            * (candidate_material.float() - baseline_material.float())\n        ).clamp(0.0, 1.0).to(baseline_material.dtype)\n''',
)

replace_once(
    model_path,
    '''            "boundary_gate": structural_gate.to(albedo.dtype),\n            "boundary_candidate_locality": candidate_locality.to(albedo.dtype),\n            "boundary_gate_prediction": selector_probability.to(albedo.dtype),\n''',
    '''            "boundary_gate": structural_gate.to(albedo.dtype),\n            "boundary_structural_residual_weight": structural_residual_weight.to(albedo.dtype),\n            "structural_residual_gain": geometry.get(\n                "structural_residual_gain", torch.zeros_like(structural_residual_weight)\n            ).to(albedo.dtype),\n            "boundary_candidate_locality": candidate_locality.to(albedo.dtype),\n            "boundary_gate_prediction": selector_probability.to(albedo.dtype),\n''',
)

replace_once(
    edge_path,
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_EDGE_CONSTRAINED_SPLINE_GRAPH_4X_V11_6_0"',
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_BASELINE_RESIDUAL_SPLINE_GRAPH_4X_V11_8_0"',
)

replace_once(
    design_path,
    '''Training is useful only when C improves on B while moving toward A. Production-final remains a separate fail-closed authority.\n\n## Literature corrections carried into V11.7\n''',
    '''Training is useful only when C improves on B while moving toward A. Production-final remains a separate fail-closed authority.\n\n## V11.8 structural identity contract\n\nThe connected-spline graph proposes an analytic redraw **R**, but proposal quality alone does not authorize replacement of B. The production structural stage is residual: **C = B + g(R - B)**. The learned local gain **g** is bounded to [-1, 1] and its final prediction layer is initialized to exact zero, therefore a fresh model starts with **C == B**. Baseline-relative geometry/pixel regret then supplies gradient authority for opening the correction only where authored Raven evidence reduces error. Explicit oracle/teacher gates may still force the full analytic proposal for proof construction; they do not change production C semantics.\n\nThis follows the residual-learning principle used by image-restoration/SR systems and the zero-initialized residual-gating principle: preserve the known reconstruction at initialization and learn only the evidence-supported correction.\n\n## Literature corrections carried into V11.7\n''',
)

append_test = '''\n\ndef test_v118_structural_candidate_is_exact_baseline_at_zero_gain():\n    from v9.model import FidelityResidualNetV9, MODEL_SCHEMA\n\n    locality = torch.ones((1, 1, 5, 7), dtype=torch.float32)\n    zero_gain = torch.zeros_like(locality)\n    weight = FidelityResidualNetV9._structural_residual_weight(\n        locality, zero_gain, None\n    )\n    assert torch.equal(weight, torch.zeros_like(weight))\n\n    baseline = torch.linspace(0.05, 0.95, 5 * 7 * 3, dtype=torch.float32).reshape(1, 3, 5, 7)\n    proposal = torch.flip(baseline, dims=(-1,))\n    candidate = baseline + weight * (proposal - baseline)\n    assert torch.equal(candidate, baseline)\n    assert MODEL_SCHEMA == "NSAMDR_RAVEN_PRODUCTION_BASELINE_RESIDUAL_SPLINE_GRAPH_4X_V11_8_0"\n\n\ndef test_v118_structural_residual_gain_is_zero_initialized_and_checkpointed():\n    local = text("tools/nsamdr/neural/v9/local_boundary_production_contract.py")\n    model = text("tools/nsamdr/neural/v9/model.py")\n    assert "self.structural_residual_gain_head = nn.Sequential(" in local\n    assert "nn.init.zeros_(self.structural_residual_gain_head[-1].weight)" in local\n    assert "nn.init.zeros_(self.structural_residual_gain_head[-1].bias)" in local\n    assert '"structural_residual_gain": structural_residual_gain' in local\n    assert 'geometry.get("structural_residual_gain")' in model\n    assert 'boundary_structural_residual_weight' in model\n    assert 'structural_gate = candidate_locality' not in model\n'''

test_file = ROOT / test_path
test_text = test_file.read_text(encoding="utf-8")
if "test_v118_structural_candidate_is_exact_baseline_at_zero_gain" in test_text:
    raise RuntimeError("V11.8 tests already present")
test_file.write_text(test_text.rstrip() + append_test + "\n", encoding="utf-8")

print("Applied V11.8 baseline-relative structural residual contract")
