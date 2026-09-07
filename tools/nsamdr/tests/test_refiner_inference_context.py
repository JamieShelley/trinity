from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def _vertical_source(torch, size: int):
    x = torch.linspace(-1.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    return x.expand(1, 1, size, size).contiguous()


def test_explicit_refiner_runs_inside_inference_mode_with_nonempty_topology():
    torch = pytest.importorskip("torch")
    from v9 import V9Config
    from v9.explicit_spline_refiner import ExplicitSplineGeometryRefiner
    from v9.spline_graph import ConnectedSplineGraph

    config = V9Config()
    config.spline_refiner_steps = 2
    feature_channels = int(getattr(config, "spline_graph_feature_channels", 64))
    graph_impl = ConnectedSplineGraph(feature_channels, config)
    refiner = ExplicitSplineGeometryRefiner(config)
    features = torch.zeros((1, feature_channels, 12, 12), dtype=torch.float32)
    source = _vertical_source(torch, 12)

    # Reproduce the real failure mode: both proposal and source tensors are
    # inference tensors when a normal production/evaluation caller uses
    # torch.inference_mode(). The inner explicit optimiser must still run.
    with torch.inference_mode():
        proposal = graph_impl.build_graph(
            features, features, source, topology_scale=1.0, displacement_scale=1.0
        )
        active = float(
            proposal["spline_graph_mask_h"].sum()
            + proposal["spline_graph_mask_v"].sum()
        )
        assert active > 0.0
        result = refiner(graph_impl, proposal, source)

    assert float(result["spline_refiner_steps"].detach().cpu()) == 2.0
    assert torch.isfinite(result["spline_refiner_energy_before"]).all()
    assert torch.isfinite(result["spline_refiner_energy_after"]).all()
    assert torch.isfinite(result["spline_refiner_node_source_error"]).all()
    assert torch.isfinite(result["spline_refiner_span_source_error"]).all()
    assert not result["spline_control_point_h_lr"].requires_grad
    assert not result["spline_control_point_v_lr"].requires_grad


def test_evolution_validation_path_can_run_refined_geometry_under_inference_mode(monkeypatch):
    torch = pytest.importorskip("torch")
    from v9 import V9Config
    from v9.evolution.candidate import CandidateEvaluator

    config = V9Config()
    config.training_activation_checkpointing = True
    config.spline_refiner_steps = 1
    evaluator = CandidateEvaluator(
        config=config,
        device="cpu",
        seed=1337,
        micro_steps=1,
    )
    model = evaluator._build_model(torch.device("cpu"))
    model.set_phase("sdf-proof")
    model.set_parametric_substage("integration")

    size = 12
    inputs = torch.zeros((1, 17, size, size), dtype=torch.float32)
    source_lr = _vertical_source(torch, size)
    inputs[:, 16:17] = source_lr
    max_distance = float(config.contour_sdf_max_distance_pixels)
    source_hr = torch.nn.functional.interpolate(
        source_lr, scale_factor=4, mode="bilinear", align_corners=False
    )
    batch = {
        "input": inputs,
        "source_sdf": source_hr.clamp(-1.0, 1.0),
        "target_sdf": source_hr.clamp(-1.0, 1.0),
    }

    monkeypatch.setattr(evaluator, "_measure_permanent_topology", lambda *_args: 0.0)
    evidence = evaluator._measure_candidate(
        model,
        batch,
        max_distance,
        train_loss_before=1.0,
        train_loss_after=0.9,
    )
    assert bool(evidence["finite"])
    assert float(evidence["topology_regression_fraction"]) == 0.0
