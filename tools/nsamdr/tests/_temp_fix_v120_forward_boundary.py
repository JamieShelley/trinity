from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "tools/nsamdr/neural/v9/model.py"
TEST = ROOT / "tools/nsamdr/tests/test_v120_explicit_geometry_refiner_contract.py"


def patch_model() -> None:
    source = MODEL.read_text(encoding="utf-8")
    marker = '            "spline_proposal_control_point_h_lr": geometry["spline_proposal_control_point_h_lr"],'
    if marker in source:
        return
    old = '''            "spline_graph_mask_h": geometry["spline_graph_mask_h"],
            "spline_graph_mask_v": geometry["spline_graph_mask_v"],
            # V11.4 sdf-proof supervises the final analytic control-lattice anchor.
'''
    new = '''            "spline_graph_mask_h": geometry["spline_graph_mask_h"],
            "spline_graph_mask_v": geometry["spline_graph_mask_v"],
            # V12 keeps the neural initializer visible through the canonical
            # production-forward boundary. B1b proposal teachers must backpropagate
            # through these tensors; the refined spline tensors above are detached
            # final solver state and therefore qualification/inference evidence only.
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
            # V11.4 sdf-proof supervises the final analytic control-lattice anchor.
'''
    if source.count(old) != 1:
        raise RuntimeError("canonical spline-output boundary marker changed")
    MODEL.write_text(source.replace(old, new), encoding="utf-8")


def patch_test() -> None:
    source = TEST.read_text(encoding="utf-8")
    method = "test_canonical_forward_preserves_proposal_gradient_boundary"
    if method in source:
        return
    old = '''    def test_schema_marks_architecture_break(self) -> None:
'''
    new = '''    def test_canonical_forward_preserves_proposal_gradient_boundary(self) -> None:
        model = (V9 / "model.py").read_text(encoding="utf-8")
        for key in (
            "spline_proposal_control_point_h_lr",
            "spline_proposal_control_point_v_lr",
            "spline_proposal_control_tangent_h",
            "spline_proposal_control_tangent_v",
            "spline_refiner_energy_before",
            "spline_refiner_energy_after",
            "spline_refiner_node_shift_rms_pixels",
            "spline_refiner_steps",
            "spline_refiner_node_source_error",
            "spline_refiner_span_source_error",
        ):
            self.assertIn(f'"{key}": geometry["{key}"]', model)

    def test_schema_marks_architecture_break(self) -> None:
'''
    if source.count(old) != 1:
        raise RuntimeError("V12 test insertion marker changed")
    TEST.write_text(source.replace(old, new), encoding="utf-8")


def main() -> None:
    patch_model()
    patch_test()


if __name__ == "__main__":
    main()
