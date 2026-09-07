from __future__ import annotations

import math

import torch


def _config():
    class Config:
        spline_graph_hidden_channels = 24
        spline_graph_control_scale = 2
        target_scale = 4
        contour_sdf_max_distance_pixels = 24.0
        spline_graph_max_topology_delta_pixels = 8.0
        spline_graph_topology_edit_band_pixels = 4.0
        spline_graph_max_displacement_pixels = 4.0
        spline_graph_max_tangent_residual = 2.0
        spline_graph_edit_band_pixels = 12.0
        spline_graph_neighbour_radius = 2
        spline_graph_samples_per_span = 4
        spline_graph_query_chunk_pixels = 256

    return Config()


def test_v116_crossing_nodes_never_leave_their_owning_edge() -> None:
    from v9.spline_graph import ConnectedSplineGraph

    graph = ConnectedSplineGraph(4, _config())
    control_phi = torch.tensor(
        [[[[3.0, 2.0, 1.0, -1.0, -2.0],
           [2.0, 1.0, -1.0, -2.0, -3.0],
           [1.0, -1.0, -2.0, -3.0, -4.0],
           [-1.0, -2.0, -3.0, -4.0, -5.0],
           [-2.0, -3.0, -4.0, -5.0, -6.0]]]],
        dtype=torch.float32,
    )
    geometry_raw = torch.randn(1, 8, 5, 5) * 8.0
    edge = graph._edge_graph(control_phi, geometry_raw, 1.0)

    point_h = edge["spline_control_point_h_lr"]
    source_h = edge["spline_source_control_point_h_lr"]
    point_v = edge["spline_control_point_v_lr"]
    source_v = edge["spline_source_control_point_v_lr"]
    disp_h = edge["spline_control_displacement_h_lr"]
    disp_v = edge["spline_control_displacement_v_lr"]

    assert torch.equal(point_h[..., 1], source_h[..., 1])
    assert torch.equal(point_v[..., 0], source_v[..., 0])
    assert torch.count_nonzero(disp_h[..., 1]) == 0
    assert torch.count_nonzero(disp_v[..., 0]) == 0

    h_owner = torch.arange(
        control_phi.shape[-1] - 1, dtype=torch.float32
    ).view(1, 1, -1).expand(point_h.shape[0], point_h.shape[1], -1)
    v_owner = torch.arange(
        control_phi.shape[-2] - 1, dtype=torch.float32
    ).view(1, -1, 1).expand(point_v.shape[0], -1, point_v.shape[2])
    assert torch.all(point_h[..., 0] > h_owner)
    assert torch.all(point_h[..., 0] < h_owner + 1.0)
    assert torch.all(point_v[..., 1] > v_owner)
    assert torch.all(point_v[..., 1] < v_owner + 1.0)


def test_v116_teacher_uses_target_zero_crossing_on_same_edge() -> None:
    from v9.edge_constrained_spline_graph import _same_edge_targets

    hr_h = hr_w = 32
    yy, xx = torch.meshgrid(
        torch.arange(hr_h, dtype=torch.float32) + 0.5,
        torch.arange(hr_w, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    slope = math.tan(math.radians(7.0))
    target = (
        (yy - (10.0 + slope * xx)) / math.sqrt(1.0 + slope * slope)
    ).unsqueeze(0).unsqueeze(0)
    target_h, target_v, _tan_h, _tan_v, valid_h, valid_v = _same_edge_targets(
        target,
        (15, 15),
        control_spacing_hr=2.0,
        control_origin=2.0,
    )

    # Horizontal crossing nodes keep their control-row Y exactly; vertical nodes
    # keep their control-column X exactly.
    h_rows = torch.arange(15, dtype=torch.float32).view(1, 15, 1)
    v_cols = torch.arange(15, dtype=torch.float32).view(1, 1, 15)
    assert torch.equal(target_h[..., 1], h_rows.expand_as(target_h[..., 1]))
    assert torch.equal(target_v[..., 0], v_cols.expand_as(target_v[..., 0]))

    # The analytic line evaluated at every valid same-edge teacher is subpixel-close
    # to zero in physical HR coordinates.
    h_x = 2.0 + 2.0 * target_h[..., 0]
    h_y = 2.0 + 2.0 * target_h[..., 1]
    v_x = 2.0 + 2.0 * target_v[..., 0]
    v_y = 2.0 + 2.0 * target_v[..., 1]
    h_error = (h_y - (10.0 + slope * h_x)).abs()
    v_error = (v_y - (10.0 + slope * v_x)).abs()
    assert float(h_error[valid_h].max()) <= 0.20
    assert float(v_error[valid_v].max()) <= 0.20


def test_v116_vectorised_query_preserves_v115_distance_math() -> None:
    from v9.edge_constrained_spline_graph import _ORIGINAL_QUERY
    from v9.parametric_boundary import make_query_grid
    from v9.spline_graph import ConnectedSplineGraph

    torch.manual_seed(17)
    model = ConnectedSplineGraph(8, _config()).eval()
    h = w = 8
    features = torch.randn(1, 8, h, w)
    geometry = torch.randn(1, 8, h, w)
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    source = ((0.65 * xx + 0.31 * yy - 3.6) / 24.0).unsqueeze(0).unsqueeze(0)
    grid = make_query_grid(1, h * 4, w * 4, device=torch.device("cpu"))
    graph = model.build_graph(features, geometry, source)

    with torch.no_grad():
        reference = _ORIGINAL_QUERY(model, graph, grid)["phi_pixels"]
        vectorised = model.query(graph, grid)["phi_pixels"]
    assert torch.isfinite(vectorised).all()
    assert float((reference - vectorised).abs().max()) <= 2.0e-5


def test_v116_schema_and_loss_teacher_are_installed_before_training_import() -> None:
    from v9 import losses
    from v9.edge_constrained_spline_graph import SCHEMA
    from v9.local_boundary_production_contract import SCHEMA as ACTIVE_SCHEMA

    assert ACTIVE_SCHEMA == SCHEMA
    assert losses.compute_losses.__module__.endswith("edge_constrained_spline_graph")


def test_v116_same_edge_teacher_is_captured_by_production_b1b_wrapper() -> None:
    from v9 import training
    from v9 import local_boundary_production_contract as local

    # This is the same installation entry point used by the workflow/trainer.
    local.install_local_boundary_training_contract(training)
    captured = local._ORIGINAL_COMPUTE_LOSSES
    assert captured is not None
    assert captured.__module__.endswith("edge_constrained_spline_graph")
    assert training.compute_losses is local._local_compute_losses
