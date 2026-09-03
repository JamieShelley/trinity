from __future__ import annotations

"""V11.6 edge-constrained connected-spline geometry.

V11.5 correctly made one connected marching-squares graph the structural
zero-set authority, but allowed every edge crossing to move freely in 2-D.
That breaks the representation invariant: a crossing owned by one control edge
must remain on that edge.  It also wastes most B1b time by rebuilding identical
Hermite span samples for every queried pixel.

This module is installed before the V11 local-boundary contract.  It keeps the
same public ConnectedSplineGraph type/checkpoint layout while replacing only:

* edge-node motion: one scalar degree of freedom along the owning edge;
* B1b node teacher: exact GT zero crossing on that same owning edge;
* query execution: precompute Hermite polyline segments once per forward, then
  reuse them in the existing bounded neighbour search.

Topology masks, cubic Hermite spans, signed-distance semantics, renderer and
promotion gates are unchanged.
"""

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .redistance import sdf_gradient_components
from . import spline_graph as _spline_graph
from . import losses as _losses


SCHEMA = "NSAMDR_RAVEN_PRODUCTION_EDGE_CONSTRAINED_SPLINE_GRAPH_4X_V11_6_0"

_INSTALLED = False
_ORIGINAL_EDGE_GRAPH = _spline_graph.ConnectedSplineGraph._edge_graph
_ORIGINAL_QUERY = _spline_graph.ConnectedSplineGraph.query
_ORIGINAL_COMPUTE_LOSSES = _losses.compute_losses


def _edge_graph(
    self: _spline_graph.ConnectedSplineGraph,
    control_phi: torch.Tensor,
    geometry_raw: torch.Tensor,
    displacement_scale: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Build shared crossings with exactly one positional DOF per owning edge.

    A horizontal crossing may move only in X and a vertical crossing only in Y.
    The final fraction is clamped to the interior of the same edge, so geometry
    refinement cannot jump a node into a neighbouring cell while retaining the
    old topology mask.
    """
    batch, _channels, height, width = control_phi.shape
    gx, gy = _spline_graph._central_difference(control_phi)

    h_a, h_b = control_phi[:, :, :, :-1], control_phi[:, :, :, 1:]
    h_fraction = self._crossing_fraction(h_a, h_b)
    mask_h = ((h_a >= 0.0) != (h_b >= 0.0)).float().detach()
    h_y = torch.arange(
        height, device=control_phi.device, dtype=torch.float32
    ).view(1, 1, height, 1)
    h_x = torch.arange(
        width - 1, device=control_phi.device, dtype=torch.float32
    ).view(1, 1, 1, width - 1)

    v_a, v_b = control_phi[:, :, :-1, :], control_phi[:, :, 1:, :]
    v_fraction = self._crossing_fraction(v_a, v_b)
    mask_v = ((v_a >= 0.0) != (v_b >= 0.0)).float().detach()
    v_y = torch.arange(
        height - 1, device=control_phi.device, dtype=torch.float32
    ).view(1, 1, height - 1, 1)
    v_x = torch.arange(
        width, device=control_phi.device, dtype=torch.float32
    ).view(1, 1, 1, width)

    raw_h = 0.5 * (geometry_raw[:, :, :, :-1] + geometry_raw[:, :, :, 1:])
    raw_v = 0.5 * (geometry_raw[:, :, :-1, :] + geometry_raw[:, :, 1:, :])
    max_displacement_lattice = self.max_displacement_pixels / max(
        self.spacing_pixels, 1.0e-6
    )
    scale = torch.as_tensor(
        displacement_scale, device=control_phi.device, dtype=torch.float32
    )

    # Keep the original channel ownership: H.x was channel 0 and V.y was channel
    # 3.  The retired orthogonal position channels remain in the checkpoint but
    # have no structural authority.
    h_fraction_pred = (
        h_fraction[:, 0]
        + torch.tanh(raw_h[:, 0]) * max_displacement_lattice * scale
    ).clamp(1.0e-3, 1.0 - 1.0e-3)
    v_fraction_pred = (
        v_fraction[:, 0]
        + torch.tanh(raw_v[:, 3]) * max_displacement_lattice * scale
    ).clamp(1.0e-3, 1.0 - 1.0e-3)

    source_h = torch.stack(
        (
            h_x.expand(batch, 1, height, width - 1)[:, 0] + h_fraction[:, 0],
            h_y.expand(batch, 1, height, width - 1)[:, 0],
        ),
        dim=-1,
    )
    source_v = torch.stack(
        (
            v_x.expand(batch, 1, height - 1, width)[:, 0],
            v_y.expand(batch, 1, height - 1, width)[:, 0] + v_fraction[:, 0],
        ),
        dim=-1,
    )
    point_h = torch.stack(
        (
            h_x.expand(batch, 1, height, width - 1)[:, 0] + h_fraction_pred,
            h_y.expand(batch, 1, height, width - 1)[:, 0],
        ),
        dim=-1,
    )
    point_v = torch.stack(
        (
            v_x.expand(batch, 1, height - 1, width)[:, 0],
            v_y.expand(batch, 1, height - 1, width)[:, 0] + v_fraction_pred,
        ),
        dim=-1,
    )
    displacement_h = point_h - source_h
    displacement_v = point_v - source_v

    h_gx = (
        (1.0 - h_fraction) * gx[:, :, :, :-1]
        + h_fraction * gx[:, :, :, 1:]
    )
    h_gy = (
        (1.0 - h_fraction) * gy[:, :, :, :-1]
        + h_fraction * gy[:, :, :, 1:]
    )
    v_gx = (
        (1.0 - v_fraction) * gx[:, :, :-1, :]
        + v_fraction * gx[:, :, 1:, :]
    )
    v_gy = (
        (1.0 - v_fraction) * gy[:, :, :-1, :]
        + v_fraction * gy[:, :, 1:, :]
    )
    tangent_h = torch.cat((-h_gy, h_gx), dim=1).permute(0, 2, 3, 1)
    tangent_v = torch.cat((-v_gy, v_gx), dim=1).permute(0, 2, 3, 1)
    tangent_h = F.normalize(
        tangent_h
        + torch.tanh(raw_h[:, 4:6]).permute(0, 2, 3, 1)
        * self.max_tangent_residual * scale,
        dim=-1,
        eps=1.0e-6,
    )
    tangent_v = F.normalize(
        tangent_v
        + torch.tanh(raw_v[:, 6:8]).permute(0, 2, 3, 1)
        * self.max_tangent_residual * scale,
        dim=-1,
        eps=1.0e-6,
    )
    return {
        "spline_control_point_h_lr": point_h,
        "spline_control_point_v_lr": point_v,
        "spline_source_control_point_h_lr": source_h,
        "spline_source_control_point_v_lr": source_v,
        "spline_control_tangent_h": tangent_h,
        "spline_control_tangent_v": tangent_v,
        "spline_control_displacement_h_lr": displacement_h,
        "spline_control_displacement_v_lr": displacement_v,
        "spline_graph_mask_h": mask_h,
        "spline_graph_mask_v": mask_v,
    }


def _precompute_segments(
    self: _spline_graph.ConnectedSplineGraph,
    p0: torch.Tensor,
    p1: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate each fixed Hermite sample once instead of once per query pixel."""
    chord = torch.linalg.vector_norm(p1 - p0, dim=-1, keepdim=True).clamp_min(1.0e-4)
    m0 = t0 * chord
    m1 = t1 * chord
    points = [p0]
    normals = [n0]
    for sample_index in range(1, self.samples_per_span + 1):
        s = float(sample_index) / float(self.samples_per_span)
        s2, s3 = s * s, s * s * s
        current = (
            (2.0 * s3 - 3.0 * s2 + 1.0) * p0
            + (s3 - 2.0 * s2 + s) * m0
            + (-2.0 * s3 + 3.0 * s2) * p1
            + (s3 - s2) * m1
        )
        current_normal = F.normalize(
            (1.0 - s) * n0 + s * n1, dim=-1, eps=1.0e-6
        )
        points.append(current)
        normals.append(current_normal)
    sampled_points = torch.stack(points, dim=-2)
    sampled_normals = torch.stack(normals, dim=-2)
    segment_start = sampled_points[..., :-1, :]
    segment_end = sampled_points[..., 1:, :]
    normal_start = sampled_normals[..., :-1, :]
    normal_end = sampled_normals[..., 1:, :]
    segment_active = active.unsqueeze(-1).expand(
        *active.shape, self.samples_per_span
    )
    return segment_start, segment_end, normal_start, normal_end, segment_active


def _unsigned_distance_precomputed(
    self: _spline_graph.ConnectedSplineGraph,
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    normal_start: torch.Tensor,
    normal_end: torch.Tensor,
    segment_active: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    base_x: torch.Tensor,
    base_y: torch.Tensor,
    control_h: int,
    control_w: int,
) -> torch.Tensor:
    """Same V11.5 polyline distance, vectorised over all samples in one cell."""
    batch, query_h, query_w = query_x.shape
    best_signed = torch.full(
        (batch, query_h, query_w),
        1.0e6,
        device=query_x.device,
        dtype=torch.float32,
    )
    query = torch.stack((query_x, query_y), dim=-1).unsqueeze(-2).unsqueeze(-2)
    for offset_y in range(-self.neighbour_radius, self.neighbour_radius + 1):
        iy = (base_y + offset_y).clamp(0, control_h - 2)
        for offset_x in range(-self.neighbour_radius, self.neighbour_radius + 1):
            ix = (base_x + offset_x).clamp(0, control_w - 2)
            start = self._gather_cell(segment_start, ix, iy)
            end = self._gather_cell(segment_end, ix, iy)
            n_start = self._gather_cell(normal_start, ix, iy)
            n_end = self._gather_cell(normal_end, ix, iy)
            active = self._gather_cell(segment_active, ix, iy)
            segment = end - start
            projection = (
                ((query - start) * segment).sum(dim=-1)
                / segment.square().sum(dim=-1).clamp_min(1.0e-8)
            ).clamp(0.0, 1.0)
            closest = start + projection[..., None] * segment
            closest_normal = F.normalize(
                n_start + projection[..., None] * (n_end - n_start),
                dim=-1,
                eps=1.0e-6,
            )
            distance = torch.linalg.vector_norm(query - closest, dim=-1) * self.spacing_pixels
            side = ((query - closest) * closest_normal).sum(dim=-1)
            signed = torch.where(side >= 0.0, distance, -distance)
            signed = torch.where(active, signed, torch.full_like(signed, 1.0e6))
            flat_signed = signed.flatten(start_dim=-2)
            local_index = flat_signed.abs().argmin(dim=-1, keepdim=True)
            local_best = torch.gather(
                flat_signed, dim=-1, index=local_index
            ).squeeze(-1)
            best_signed = torch.where(
                local_best.abs() < best_signed.abs(), local_best, best_signed
            )
    return best_signed


def _query(
    self: _spline_graph.ConnectedSplineGraph,
    graph: dict[str, torch.Tensor],
    query_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """V11.5 query semantics with invariant Hermite work hoisted out of pixel loops."""
    control_phi = graph["spline_graph_control_phi_pixels"].float()
    source = graph["source_sdf_prior_lr"].float()
    batch, _channels, control_h, control_w = control_phi.shape
    query_h, query_w = query_grid.shape[1:3]
    physical_x = (query_grid[..., 0].float() + 1.0) * (float(query_w) * 0.5)
    physical_y = (query_grid[..., 1].float() + 1.0) * (float(query_h) * 0.5)
    query_x = (physical_x - self.origin_pixels) / self.spacing_pixels
    query_y = (physical_y - self.origin_pixels) / self.spacing_pixels
    base_x = torch.floor(query_x).long()
    base_y = torch.floor(query_y).long()

    p0, p1, t0, t1, n0, n1, active = self._cell_spans(graph)
    segment_start, segment_end, normal_start, normal_end, segment_active = (
        _precompute_segments(self, p0, p1, t0, t1, n0, n1, active)
    )
    chunk_rows = max(
        1,
        min(query_h, self.query_chunk_pixels // max(query_w, 1)),
    )
    signed_chunks: list[torch.Tensor] = []
    checkpoint_distance = (
        self.training
        and torch.is_grad_enabled()
        and any(
            value.requires_grad
            for value in (segment_start, segment_end, normal_start, normal_end)
        )
    )
    for row_start in range(0, query_h, chunk_rows):
        row_end = min(query_h, row_start + chunk_rows)
        qx = query_x[:, row_start:row_end]
        qy = query_y[:, row_start:row_end]
        bx = base_x[:, row_start:row_end]
        by = base_y[:, row_start:row_end]
        if checkpoint_distance:
            signed = checkpoint(
                lambda cs, ce, cns, cne, cqx, cqy, cbx, cby: (
                    _unsigned_distance_precomputed(
                        self, cs, ce, cns, cne, segment_active,
                        cqx, cqy, cbx, cby, control_h, control_w,
                    )
                ),
                segment_start, segment_end, normal_start, normal_end,
                qx, qy, bx, by,
                use_reentrant=False,
            )
        else:
            signed = _unsigned_distance_precomputed(
                self,
                segment_start, segment_end, normal_start, normal_end, segment_active,
                qx, qy, bx, by, control_h, control_w,
            )
        signed_chunks.append(signed)
    graph_phi = torch.cat(signed_chunks, dim=1)

    sampled_source = F.grid_sample(
        source,
        query_grid.float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[:, 0] * self.max_distance_pixels
    has_graph = graph_phi.abs() < 1.0e5
    inner = max(2.0, self.edit_band_pixels * (2.0 / 3.0))
    outer = max(inner + 1.0, self.edit_band_pixels)
    authority = ((outer - sampled_source.abs()) / (outer - inner)).clamp(0.0, 1.0)
    authority = authority * authority * (3.0 - 2.0 * authority)
    authority = authority * has_graph.float()
    phi = sampled_source * (1.0 - authority) + graph_phi * authority
    phi_field = phi.unsqueeze(1)

    gx, gy = _spline_graph._central_difference(phi_field)
    norm = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
    normal = torch.cat((gx / norm, gy / norm), dim=1)
    nx_x, _ = _spline_graph._central_difference(normal[:, 0:1])
    _, ny_y = _spline_graph._central_difference(normal[:, 1:2])
    curvature = nx_x + ny_y
    source_field = sampled_source.unsqueeze(1)
    return {
        "phi_pixels": phi_field,
        "primitive_phi_pixels": phi_field,
        "primitive_normal": normal,
        "primitive_curvature": curvature,
        "primitive_confidence": authority.unsqueeze(1),
        "primitive_junction_hint": torch.zeros_like(phi_field),
        "primitive_distance_delta_pixels": phi_field - source_field,
        "transport_pixels": torch.zeros(
            (batch, 2, query_h, query_w),
            device=phi_field.device,
            dtype=phi_field.dtype,
        ),
        "dilation_pixels": torch.zeros_like(phi_field),
        "residual_pixels": phi_field - source_field,
        "direct_delta_pixels": phi_field - source_field,
        "warped_source_pixels": source_field,
        "implicit_authority": authority.unsqueeze(1),
        "spline_graph_authority": torch.ones(
            (), device=phi_field.device, dtype=phi_field.dtype
        ),
    }


def _same_edge_targets(
    target_sdf_pixels: torch.Tensor,
    control_shape: tuple[int, int],
    *,
    control_spacing_hr: float,
    control_origin: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Return GT crossing/tangent teachers on each graph edge, never off-edge."""
    target = target_sdf_pixels.detach().float()
    batch, _channels, hr_h, hr_w = target.shape
    ch, cw = control_shape
    cy, cx = torch.meshgrid(
        torch.arange(ch, device=target.device, dtype=torch.float32),
        torch.arange(cw, device=target.device, dtype=torch.float32),
        indexing="ij",
    )
    physical_x = control_origin + control_spacing_hr * cx
    physical_y = control_origin + control_spacing_hr * cy
    control_grid = torch.stack(
        (
            2.0 * physical_x / float(hr_w) - 1.0,
            2.0 * physical_y / float(hr_h) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0).expand(batch, -1, -1, -1)
    target_control = F.grid_sample(
        target, control_grid, mode="bilinear", padding_mode="border", align_corners=False
    )[:, 0]

    def fraction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        denom = a - b
        eps = torch.where(
            denom >= 0.0,
            torch.full_like(denom, 1.0e-6),
            torch.full_like(denom, -1.0e-6),
        )
        return (a / (denom + eps)).clamp(1.0e-3, 1.0 - 1.0e-3)

    h_a, h_b = target_control[:, :, :-1], target_control[:, :, 1:]
    v_a, v_b = target_control[:, :-1, :], target_control[:, 1:, :]
    h_valid = (h_a >= 0.0) != (h_b >= 0.0)
    v_valid = (v_a >= 0.0) != (v_b >= 0.0)
    h_fraction = fraction(h_a, h_b)
    v_fraction = fraction(v_a, v_b)

    h_y = torch.arange(ch, device=target.device, dtype=torch.float32).view(1, ch, 1)
    h_x = torch.arange(cw - 1, device=target.device, dtype=torch.float32).view(1, 1, cw - 1)
    v_y = torch.arange(ch - 1, device=target.device, dtype=torch.float32).view(1, ch - 1, 1)
    v_x = torch.arange(cw, device=target.device, dtype=torch.float32).view(1, 1, cw)
    target_h = torch.stack(
        (
            h_x.expand(batch, ch, cw - 1) + h_fraction,
            h_y.expand(batch, ch, cw - 1),
        ),
        dim=-1,
    )
    target_v = torch.stack(
        (
            v_x.expand(batch, ch - 1, cw),
            v_y.expand(batch, ch - 1, cw) + v_fraction,
        ),
        dim=-1,
    )

    tgx, tgy = sdf_gradient_components(target)

    def tangent_at(points: torch.Tensor) -> torch.Tensor:
        px = control_origin + control_spacing_hr * points[..., 0]
        py = control_origin + control_spacing_hr * points[..., 1]
        grid = torch.stack(
            (2.0 * px / float(hr_w) - 1.0, 2.0 * py / float(hr_h) - 1.0),
            dim=-1,
        )
        grad = F.grid_sample(
            torch.cat((tgx, tgy), dim=1),
            grid.float(), mode="bilinear", padding_mode="border", align_corners=False,
        ).permute(0, 2, 3, 1)
        normal = F.normalize(grad, dim=-1, eps=1.0e-6)
        return torch.stack((-normal[..., 1], normal[..., 0]), dim=-1).detach()

    return (
        target_h.detach(), target_v.detach(), tangent_at(target_h), tangent_at(target_v),
        h_valid.detach(), v_valid.detach(),
    )


def _compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    """Replace only V11 spline node/tangent teachers with same-edge GT crossings."""
    result = _ORIGINAL_COMPUTE_LOSSES(outputs, batch, config, phase)
    spline_control = outputs.get("spline_graph_control_phi_pixels")
    spline_h = outputs.get("spline_control_point_h_lr")
    spline_v = outputs.get("spline_control_point_v_lr")
    spline_tan_h = outputs.get("spline_control_tangent_h")
    spline_tan_v = outputs.get("spline_control_tangent_v")
    spline_mask_h = outputs.get("spline_graph_mask_h")
    spline_mask_v = outputs.get("spline_graph_mask_v")
    source_prior = outputs.get("source_sdf_prior_pixels")
    if any(
        value is None
        for value in (
            spline_control, spline_h, spline_v, spline_tan_h, spline_tan_v,
            spline_mask_h, spline_mask_v, source_prior,
        )
    ):
        return result

    max_distance = float(config.contour_sdf_max_distance_pixels)
    raw_target = batch["target_sdf"].float() * max_distance
    if bool(config.sdf_sign_gauge_invariant):
        polarity = _losses._losses_service._sdf_global_polarity(
            source_prior.detach().float(), raw_target,
            float(config.sdf_metric_band_pixels),
        )
    else:
        polarity = torch.ones(
            (raw_target.shape[0], 1, 1, 1),
            device=raw_target.device,
            dtype=raw_target.dtype,
        )
    target = raw_target * polarity
    control_scale = float(getattr(config, "spline_graph_control_scale", 2))
    control_spacing_hr = 4.0 / max(control_scale, 1.0)
    control_origin = 2.0
    target_h, target_v, target_tan_h, target_tan_v, valid_h, valid_v = (
        _same_edge_targets(
            target,
            tuple(spline_control.shape[-2:]),
            control_spacing_hr=control_spacing_hr,
            control_origin=control_origin,
        )
    )
    mh = spline_mask_h[:, 0].float() * valid_h.float()
    mv = spline_mask_v[:, 0].float() * valid_v.float()
    denom = (mh.sum() + mv.sum()).clamp_min(1.0)
    point_h_error = (spline_h.float() - target_h).abs().sum(dim=-1)
    point_v_error = (spline_v.float() - target_v).abs().sum(dim=-1)
    result["spline_graph_point"] = (
        (point_h_error * mh).sum() + (point_v_error * mv).sum()
    ) / denom
    dot_h = (spline_tan_h.float() * target_tan_h).sum(dim=-1).abs().clamp(0.0, 1.0)
    dot_v = (spline_tan_v.float() * target_tan_v).sum(dim=-1).abs().clamp(0.0, 1.0)
    result["spline_graph_tangent"] = (
        ((1.0 - dot_h) * mh).sum() + ((1.0 - dot_v) * mv).sum()
    ) / denom
    result["spline_graph_same_edge_teacher_coverage"] = (
        (mh.sum() + mv.sum())
        / (
            spline_mask_h[:, 0].float().sum()
            + spline_mask_v[:, 0].float().sum()
        ).clamp_min(1.0)
    ).detach()
    return result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _spline_graph.ConnectedSplineGraph._edge_graph = _edge_graph
    _spline_graph.ConnectedSplineGraph.query = _query
    _losses.compute_losses = _compute_losses
    _INSTALLED = True


def install_schema(local_boundary_module: Any) -> None:
    """Apply the V11.6 schema before the local-boundary installer runs."""
    local_boundary_module.SCHEMA = SCHEMA
