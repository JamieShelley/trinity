from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
V9 = ROOT / "tools/nsamdr/neural/v9"
SCHEMA = "NSAMDR_RAVEN_PRODUCTION_NEURAL_PROPOSAL_EXPLICIT_REFINER_SPLINE_GRAPH_4X_V12_0_0"


class TestV120ExplicitGeometryRefinerContract(unittest.TestCase):
    def test_refiner_is_parameter_free_lr_only_geometry_solver(self) -> None:
        source = (V9 / "explicit_spline_refiner.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("class ExplicitSplineGeometryRefiner(nn.Module)", source)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn("source_sdf_lr", source)
        self.assertNotIn('batch["target_sdf"]', source)
        self.assertNotIn("target_albedo", source)
        self.assertNotIn("nn.Parameter", source)
        self.assertIn("proposal_graph", source)
        self.assertIn("spline_graph_mask_h", source)
        self.assertIn("spline_graph_mask_v", source)

    def test_production_path_is_proposal_then_refiner_then_query(self) -> None:
        source = (V9 / "local_boundary_production_contract.py").read_text(encoding="utf-8")
        ast.parse(source)
        build = source.index("proposal_graph = self.spline_graph.build_graph(")
        refine = source.index("refined_graph = self.geometry_refiner(")
        query = source.index("self.spline_graph.query(refined_graph, query_grid)")
        self.assertLess(build, refine)
        self.assertLess(refine, query)
        self.assertIn('"neuralGeometryIsInitializerOnly": True', source)
        self.assertIn('"explicitGeometryRefinement": True', source)
        self.assertIn('"explicitGeometryRefinementUsesTargetHR": False', source)
        self.assertIn('"explicitGeometryRefinementCanChangeTopology": False', source)
        self.assertIn('"explicit geometry refiner": (', source)
        self.assertIn("for parameter in self.decoder.parameters():", source)
        self.assertIn("parameter.requires_grad_(False)", source)

    def test_b1b_supervises_neural_proposal_not_refined_solver_state(self) -> None:
        edge = (V9 / "edge_constrained_spline_graph.py").read_text(encoding="utf-8")
        local = (V9 / "local_boundary_production_contract.py").read_text(encoding="utf-8")
        self.assertIn("spline_proposal_control_point_h_lr", edge)
        self.assertIn("spline_proposal_control_tangent_h", edge)
        self.assertIn("Final dense spline/SDF", local)
        self.assertIn('losses["spline_graph_point"]', local)
        self.assertIn('losses["spline_graph_tangent"]', local)

    def test_schema_marks_architecture_break(self) -> None:
        for path in (
            V9 / "local_boundary_production_contract.py",
            V9 / "edge_constrained_spline_graph.py",
            ROOT / "tools/nsamdr/tests/test_v117_baseline_relative_contract.py",
        ):
            self.assertIn(SCHEMA, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
