from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))

REQUIRED = (
    "spline_refiner_energy_before",
    "spline_refiner_energy_after",
    "spline_refiner_node_shift_rms_pixels",
    "spline_refiner_steps",
    "spline_refiner_node_source_error",
    "spline_refiner_span_source_error",
)

def _assert_zero_telemetry(graph):
    for key in REQUIRED:
        assert key in graph, key
        assert torch.is_tensor(graph[key])
        assert torch.isfinite(graph[key]).all()
        assert float(graph[key].detach().cpu()) == 0.0

def _proposal(config):
    from v9.spline_graph import ConnectedSplineGraph
    feature_channels = int(getattr(config, "spline_graph_feature_channels", 64))
    graph_impl = ConnectedSplineGraph(feature_channels, config)
    source = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    features = torch.zeros((1, feature_channels, 8, 8), dtype=torch.float32)
    proposal = graph_impl.build_graph(
        features, features, source, topology_scale=1.0, displacement_scale=1.0
    )
    return graph_impl, source, proposal

def test_noop_refiner_returns_complete_telemetry():
    from v9 import V9Config
    from v9.explicit_spline_refiner import ExplicitSplineGeometryRefiner
    config = V9Config()
    config.spline_refiner_steps = 0
    graph_impl, source, proposal = _proposal(config)
    result = ExplicitSplineGeometryRefiner(config)(graph_impl, proposal, source)
    _assert_zero_telemetry(result)

def test_empty_topology_refiner_returns_complete_telemetry():
    from v9 import V9Config
    from v9.explicit_spline_refiner import ExplicitSplineGeometryRefiner
    config = V9Config()
    config.spline_refiner_steps = 3
    graph_impl, source, proposal = _proposal(config)
    assert float(proposal["spline_graph_mask_h"].sum() + proposal["spline_graph_mask_v"].sum()) == 0.0
    result = ExplicitSplineGeometryRefiner(config)(graph_impl, proposal, source)
    _assert_zero_telemetry(result)

def test_direct_production_forward_survives_empty_refiner_topology():
    from v9 import FidelityResidualNetV9, V9Config
    config = V9Config()
    config.training_activation_checkpointing = False
    config.spline_refiner_steps = 3
    model = FidelityResidualNetV9(config).eval()
    inputs = torch.zeros((1, 17, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
    for key in ("albedo", "normal_xy", "material", "roughness", "emissive"):
        assert key in outputs
        assert torch.isfinite(outputs[key]).all()
    for key in REQUIRED:
        assert key in outputs
        assert torch.isfinite(outputs[key]).all()
