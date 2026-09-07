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


model = V9 / "model.py"
replace_once(
    model,
    '''            "spline_control_tangent_h": geometry["spline_control_tangent_h"],
            "spline_control_tangent_v": geometry["spline_control_tangent_v"],
            "spline_control_displacement_h_lr": geometry["spline_control_displacement_h_lr"],
''',
    '''            "spline_control_tangent_h": geometry["spline_control_tangent_h"],
            "spline_control_tangent_v": geometry["spline_control_tangent_v"],
            # V12 estimator/refiner split: B1b supervises these gradient-bearing
            # neural initializer tensors. The refined spline state below remains
            # the detached production/qualification geometry.
            "spline_proposal_control_point_h_lr": geometry["spline_proposal_control_point_h_lr"],
            "spline_proposal_control_point_v_lr": geometry["spline_proposal_control_point_v_lr"],
            "spline_proposal_control_tangent_h": geometry["spline_proposal_control_tangent_h"],
            "spline_proposal_control_tangent_v": geometry["spline_proposal_control_tangent_v"],
            "spline_refiner_energy_before": geometry["spline_refiner_energy_before"],
            "spline_refiner_energy_after": geometry["spline_refiner_energy_after"],
            "spline_refiner_node_shift_rms_pixels": geometry["spline_refiner_node_shift_rms_pixels"],
            "spline_refiner_steps": geometry["spline_refiner_steps"],
            "spline_refiner_node_source_error": geometry["spline_refiner_node_source_error"],
            "spline_refiner_span_source_error": geometry["spline_refiner_span_source_error"],
            "spline_control_displacement_h_lr": geometry["spline_control_displacement_h_lr"],
''',
)

local = V9 / "local_boundary_production_contract.py"
replace_once(
    local,
    '''                "spline_control_tangent_h",
                "spline_control_tangent_v",
                "spline_control_displacement_h_lr",
''',
    '''                "spline_control_tangent_h",
                "spline_control_tangent_v",
                "spline_proposal_control_point_h_lr",
                "spline_proposal_control_point_v_lr",
                "spline_proposal_control_tangent_h",
                "spline_proposal_control_tangent_v",
                "spline_control_displacement_h_lr",
''',
)
replace_once(
    local,
    '''                    "V11.5 connected-spline B1 supervision is disconnected from "
                    "the production forward; missing outputs="
''',
    '''                    "V12 connected-spline B1 supervision is disconnected from "
                    "the production forward; refined and neural-proposal outputs are required; missing outputs="
''',
)

edge = V9 / "edge_constrained_spline_graph.py"
replace_once(
    edge,
    '''    spline_control = outputs.get("spline_graph_control_phi_pixels")
    # V12: supervise the neural initializer. Final continuous geometry is
''',
    '''    spline_control = outputs.get("spline_graph_control_phi_pixels")
    if phase in {"sdf-bootstrap", "sdf-proof"}:
        required_proposal_outputs = (
            "spline_proposal_control_point_h_lr",
            "spline_proposal_control_point_v_lr",
            "spline_proposal_control_tangent_h",
            "spline_proposal_control_tangent_v",
        )
        missing_proposal_outputs = [
            key for key in required_proposal_outputs if key not in outputs
        ]
        if missing_proposal_outputs:
            raise RuntimeError(
                "V12 B1 neural-initializer supervision requires proposal tensors from "
                "the canonical production forward; missing outputs="
                f"{missing_proposal_outputs}"
            )
    # V12: supervise the neural initializer. Final continuous geometry is
''',
)

refiner = V9 / "explicit_spline_refiner.py"
replace_once(
    refiner,
    '''        proposal_h = proposal_graph["spline_control_point_h_lr"].float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].float()
        point_h = torch.stack((h_x, proposal_h[..., 1]), dim=-1)
        point_v = torch.stack((proposal_v[..., 0], v_y), dim=-1)
        source_h = proposal_graph["spline_source_control_point_h_lr"].float()
        source_v = proposal_graph["spline_source_control_point_v_lr"].float()
''',
    '''        # The proposal supplies fixed coordinates/topology to the inner solver,
        # never an outer-SGD path into final refined geometry.
        proposal_h = proposal_graph["spline_control_point_h_lr"].detach().float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].detach().float()
        point_h = torch.stack((h_x, proposal_h[..., 1]), dim=-1)
        point_v = torch.stack((proposal_v[..., 0], v_y), dim=-1)
        source_h = proposal_graph["spline_source_control_point_h_lr"].detach().float()
        source_v = proposal_graph["spline_source_control_point_v_lr"].detach().float()
''',
)
replace_once(
    refiner,
    '''        if not self.enabled or self.steps <= 0:
            graph = dict(proposal_graph)
            zero = source_sdf_lr.new_zeros(())
''',
    '''        if not self.enabled or self.steps <= 0:
            # A no-op solver still establishes detached final-geometry authority.
            graph = self._graph_with_geometry(
                proposal_graph,
                proposal_graph["spline_control_point_h_lr"][..., 0].detach().float(),
                proposal_graph["spline_control_point_v_lr"][..., 1].detach().float(),
                proposal_graph["spline_control_tangent_h"].detach().float(),
                proposal_graph["spline_control_tangent_v"].detach().float(),
            )
            zero = source_sdf_lr.new_zeros(())
''',
)
replace_once(
    refiner,
    '''        if float(mask_h.sum().detach().cpu()) + float(mask_v.sum().detach().cpu()) <= 0.0:
            graph = dict(proposal_graph)
            zero = source_sdf_lr.new_zeros(())
''',
    '''        if float(mask_h.sum().detach().cpu()) + float(mask_v.sum().detach().cpu()) <= 0.0:
            # Empty topology has no optimization work, but final geometry remains
            # detached from the neural initializer exactly like the normal path.
            graph = self._graph_with_geometry(
                proposal_graph,
                proposal_h[..., 0],
                proposal_v[..., 1],
                proposal_graph["spline_control_tangent_h"].detach().float(),
                proposal_graph["spline_control_tangent_v"].detach().float(),
            )
            zero = source_sdf_lr.new_zeros(())
''',
)

contract_test = TESTS / "test_v120_explicit_geometry_refiner_contract.py"
replace_once(
    contract_test,
    '''        self.assertIn("spline_proposal_control_point_h_lr", edge)
        self.assertIn("spline_proposal_control_tangent_h", edge)
        self.assertIn("Final dense spline/SDF", local)
''',
    '''        model = (V9 / "model.py").read_text(encoding="utf-8")
        refiner = (V9 / "explicit_spline_refiner.py").read_text(encoding="utf-8")
        self.assertIn("spline_proposal_control_point_h_lr", edge)
        self.assertIn("spline_proposal_control_tangent_h", edge)
        self.assertIn('"spline_proposal_control_point_h_lr": geometry["spline_proposal_control_point_h_lr"]', model)
        self.assertIn('"spline_proposal_control_tangent_h": geometry["spline_proposal_control_tangent_h"]', model)
        self.assertIn("required_proposal_outputs", edge)
        self.assertIn("refined and neural-proposal outputs are required", local)
        self.assertIn("A no-op solver still establishes detached final-geometry authority", refiner)
        self.assertIn("Empty topology has no optimization work", refiner)
        self.assertIn("Final dense spline/SDF", local)
''',
)

print("applied V12 proposal-gradient forward contract")
