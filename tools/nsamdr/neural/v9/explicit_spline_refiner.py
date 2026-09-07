from __future__ import annotations

"""Explicit continuous spline refinement for NSAMDR V12.

The neural structural branch is an initializer, not the final geometry solver.
Topology and an initial connected-spline proposal come from the network.  This
module then optimizes only the continuous node/tangent parameters against the
observed LR structural evidence while keeping topology fixed and remaining close
to the neural proposal.  It has no trainable parameters and never sees authored
HR targets, so the exact same refinement runs in training and production.

This follows the estimator -> explicit refinement decomposition used by Deep
Vectorization of Technical Drawings and the continuous-parameter optimization
role demonstrated by differentiable vector-graphics systems such as DiffVG.
"""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ExplicitSplineGeometryRefiner(nn.Module):
    """Refine a neural spline proposal with bounded LR-consistency optimization."""

    # Purpose: Configure fixed inference-time geometry optimization.
    # Called by: LocalBoundaryProductionStructure.__init__.
    # Calls: No same-class helper methods.
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.enabled = bool(getattr(config, "spline_refiner_enabled", True))
        self.steps = max(0, int(getattr(config, "spline_refiner_steps", 3)))
        self.position_step = float(getattr(config, "spline_refiner_position_step", 0.08))
        self.tangent_step = float(getattr(config, "spline_refiner_tangent_step", 0.08))
        self.source_node_weight = float(getattr(config, "spline_refiner_source_node_weight", 0.35))
        self.source_span_weight = float(getattr(config, "spline_refiner_source_span_weight", 0.20))
        self.proposal_position_weight = float(
            getattr(config, "spline_refiner_proposal_position_weight", 1.00)
        )
        self.source_tangent_weight = float(
            getattr(config, "spline_refiner_source_tangent_weight", 0.15)
        )
        self.proposal_tangent_weight = float(
            getattr(config, "spline_refiner_proposal_tangent_weight", 0.60)
        )
        self.max_move_pixels = float(getattr(config, "spline_refiner_max_move_pixels", 0.75))
        self.huber_beta_pixels = float(getattr(config, "spline_refiner_huber_beta_pixels", 0.35))
        self.output_scale = max(1, int(getattr(config, "target_scale", 4)))
        self.max_distance_pixels = float(config.contour_sdf_max_distance_pixels)

    # Purpose: Compute a stable masked scalar average.
    # Called by: _energy.
    # Calls: No same-class helper methods.
    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(device=value.device, dtype=value.dtype)
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    # Purpose: Compute LR source-field gradients for tangent evidence.
    # Called by: _source_tangent.
    # Calls: No same-class helper methods.
    @staticmethod
    def _central_difference(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = value.float()
        xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
        yp = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return (
            0.5 * (xp[:, :, :, 2:] - xp[:, :, :, :-2]),
            0.5 * (yp[:, :, 2:, :] - yp[:, :, :-2, :]),
        )

    # Purpose: Map control-lattice points into the observed LR image domain.
    # Called by: _sample_field.
    # Calls: No same-class helper methods.
    def _point_grid(
        self,
        points: torch.Tensor,
        source_sdf_lr: torch.Tensor,
        spline_graph: Any,
    ) -> torch.Tensor:
        hr_h = float(source_sdf_lr.shape[-2] * self.output_scale)
        hr_w = float(source_sdf_lr.shape[-1] * self.output_scale)
        physical_x = float(spline_graph.origin_pixels) + float(spline_graph.spacing_pixels) * points[..., 0]
        physical_y = float(spline_graph.origin_pixels) + float(spline_graph.spacing_pixels) * points[..., 1]
        return torch.stack(
            (2.0 * physical_x / max(hr_w, 1.0) - 1.0,
             2.0 * physical_y / max(hr_h, 1.0) - 1.0),
            dim=-1,
        )

    # Purpose: Sample one scalar/vector LR evidence field at arbitrary spline points.
    # Called by: _source_tangent, _energy.
    # Calls: _point_grid.
    def _sample_field(
        self,
        field: torch.Tensor,
        points: torch.Tensor,
        source_sdf_lr: torch.Tensor,
        spline_graph: Any,
    ) -> torch.Tensor:
        grid = self._point_grid(points, source_sdf_lr, spline_graph)
        batch = int(points.shape[0])
        sample_shape = tuple(points.shape[1:-1])
        flat_grid = grid.reshape(batch, -1, 1, 2)
        sampled = F.grid_sample(
            field.float(),
            flat_grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, :, :, 0]
        if sampled.shape[1] == 1:
            return sampled[:, 0].reshape(batch, *sample_shape)
        return sampled.permute(0, 2, 1).reshape(batch, *sample_shape, sampled.shape[1])

    # Purpose: Derive observed contour tangents and reliability from the LR SDF.
    # Called by: _energy.
    # Calls: _central_difference, _sample_field.
    def _source_tangent(
        self,
        source_sdf_lr: torch.Tensor,
        points: torch.Tensor,
        spline_graph: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gx, gy = self._central_difference(source_sdf_lr.float())
        sampled_grad = self._sample_field(
            torch.cat((gx, gy), dim=1), points, source_sdf_lr, spline_graph
        )
        magnitude = torch.linalg.vector_norm(sampled_grad, dim=-1)
        normal = F.normalize(sampled_grad, dim=-1, eps=1.0e-6)
        tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
        reliability = (magnitude / 0.035).clamp(0.0, 1.0).detach()
        return tangent, reliability

    # Purpose: Replace only continuous geometry while preserving proposal topology.
    # Called by: forward, _energy.
    # Calls: No same-class helper methods.
    @staticmethod
    def _graph_with_geometry(
        proposal_graph: dict[str, torch.Tensor],
        h_x: torch.Tensor,
        v_y: torch.Tensor,
        tangent_h: torch.Tensor,
        tangent_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph = dict(proposal_graph)
        # The proposal supplies fixed coordinates/topology to the inner solver,
        # never an outer-SGD path into final refined geometry.
        proposal_h = proposal_graph["spline_control_point_h_lr"].detach().float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].detach().float()
        point_h = torch.stack((h_x, proposal_h[..., 1]), dim=-1)
        point_v = torch.stack((proposal_v[..., 0], v_y), dim=-1)
        source_h = proposal_graph["spline_source_control_point_h_lr"].detach().float()
        source_v = proposal_graph["spline_source_control_point_v_lr"].detach().float()
        graph["spline_control_point_h_lr"] = point_h
        graph["spline_control_point_v_lr"] = point_v
        graph["spline_control_tangent_h"] = F.normalize(tangent_h, dim=-1, eps=1.0e-6)
        graph["spline_control_tangent_v"] = F.normalize(tangent_v, dim=-1, eps=1.0e-6)
        graph["spline_control_displacement_h_lr"] = point_h - source_h
        graph["spline_control_displacement_v_lr"] = point_v - source_v
        return graph

    # Purpose: Score continuous proposal parameters against observed LR evidence.
    # Called by: forward.
    # Calls: _graph_with_geometry, _masked_mean, _sample_field, _source_tangent.
    def _energy(
        self,
        spline_graph: Any,
        proposal_graph: dict[str, torch.Tensor],
        source_sdf_lr: torch.Tensor,
        h_x: torch.Tensor,
        v_y: torch.Tensor,
        tangent_h: torch.Tensor,
        tangent_v: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        graph = self._graph_with_geometry(
            proposal_graph, h_x, v_y, tangent_h, tangent_v
        )
        point_h = graph["spline_control_point_h_lr"]
        point_v = graph["spline_control_point_v_lr"]
        mask_h = proposal_graph["spline_graph_mask_h"][:, 0].float()
        mask_v = proposal_graph["spline_graph_mask_v"][:, 0].float()
        source_pixels = source_sdf_lr.float() * self.max_distance_pixels

        sampled_h = self._sample_field(source_pixels, point_h, source_sdf_lr, spline_graph)
        sampled_v = self._sample_field(source_pixels, point_v, source_sdf_lr, spline_graph)
        node_h = F.smooth_l1_loss(
            sampled_h, torch.zeros_like(sampled_h),
            beta=max(self.huber_beta_pixels, 1.0e-4), reduction="none",
        )
        node_v = F.smooth_l1_loss(
            sampled_v, torch.zeros_like(sampled_v),
            beta=max(self.huber_beta_pixels, 1.0e-4), reduction="none",
        )
        node_source = 0.5 * (
            self._masked_mean(node_h, mask_h) + self._masked_mean(node_v, mask_v)
        )

        proposal_h = proposal_graph["spline_control_point_h_lr"].detach().float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].detach().float()
        spacing = float(spline_graph.spacing_pixels)
        proposal_position = 0.5 * (
            self._masked_mean(
                ((h_x - proposal_h[..., 0]) * spacing).square(), mask_h
            )
            + self._masked_mean(
                ((v_y - proposal_v[..., 1]) * spacing).square(), mask_v
            )
        )

        observed_h, reliability_h = self._source_tangent(
            source_sdf_lr, point_h, spline_graph
        )
        observed_v, reliability_v = self._source_tangent(
            source_sdf_lr, point_v, spline_graph
        )
        normalized_h = F.normalize(tangent_h, dim=-1, eps=1.0e-6)
        normalized_v = F.normalize(tangent_v, dim=-1, eps=1.0e-6)
        observed_tangent = 0.5 * (
            self._masked_mean(
                1.0 - (normalized_h * observed_h).sum(dim=-1).abs().clamp(0.0, 1.0),
                mask_h * reliability_h,
            )
            + self._masked_mean(
                1.0 - (normalized_v * observed_v).sum(dim=-1).abs().clamp(0.0, 1.0),
                mask_v * reliability_v,
            )
        )
        proposal_tangent_h = F.normalize(
            proposal_graph["spline_control_tangent_h"].detach().float(), dim=-1, eps=1.0e-6
        )
        proposal_tangent_v = F.normalize(
            proposal_graph["spline_control_tangent_v"].detach().float(), dim=-1, eps=1.0e-6
        )
        proposal_tangent = 0.5 * (
            self._masked_mean(
                1.0 - (normalized_h * proposal_tangent_h).sum(dim=-1).abs().clamp(0.0, 1.0),
                mask_h,
            )
            + self._masked_mean(
                1.0 - (normalized_v * proposal_tangent_v).sum(dim=-1).abs().clamp(0.0, 1.0),
                mask_v,
            )
        )

        p0, p1, t0, t1, _n0, _n1, active = spline_graph._cell_spans(graph)
        chord = torch.linalg.vector_norm(p1 - p0, dim=-1, keepdim=True).clamp_min(1.0e-4)
        m0 = t0 * chord
        m1 = t1 * chord
        span_source = source_pixels.new_zeros(())
        for fraction in (0.25, 0.50, 0.75):
            s = float(fraction)
            s2, s3 = s * s, s * s * s
            point = (
                (2.0 * s3 - 3.0 * s2 + 1.0) * p0
                + (s3 - 2.0 * s2 + s) * m0
                + (-2.0 * s3 + 3.0 * s2) * p1
                + (s3 - s2) * m1
            )
            sampled = self._sample_field(source_pixels, point, source_sdf_lr, spline_graph)
            span_error = F.smooth_l1_loss(
                sampled, torch.zeros_like(sampled),
                beta=max(self.huber_beta_pixels, 1.0e-4), reduction="none",
            )
            span_source = span_source + self._masked_mean(span_error, active.float()) / 3.0

        total = (
            node_source * self.source_node_weight
            + span_source * self.source_span_weight
            + proposal_position * self.proposal_position_weight
            + observed_tangent * self.source_tangent_weight
            + proposal_tangent * self.proposal_tangent_weight
        )
        return total, {
            "nodeSource": node_source.detach(),
            "spanSource": span_source.detach(),
            "proposalPosition": proposal_position.detach(),
            "sourceTangent": observed_tangent.detach(),
            "proposalTangent": proposal_tangent.detach(),
        }

    # Purpose: Run fixed-step continuous optimization without changing topology.
    # Called by: LocalBoundaryProductionStructure.forward.
    # Calls: _energy, _graph_with_geometry.
    def forward(
        self,
        spline_graph: Any,
        proposal_graph: dict[str, torch.Tensor],
        source_sdf_lr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled or self.steps <= 0:
            # A no-op solver still establishes detached final-geometry authority.
            graph = self._graph_with_geometry(
                proposal_graph,
                proposal_graph["spline_control_point_h_lr"][..., 0].detach().float(),
                proposal_graph["spline_control_point_v_lr"][..., 1].detach().float(),
                proposal_graph["spline_control_tangent_h"].detach().float(),
                proposal_graph["spline_control_tangent_v"].detach().float(),
            )
            zero = source_sdf_lr.new_zeros(())
            graph["spline_refiner_energy_before"] = zero
            graph["spline_refiner_energy_after"] = zero
            graph["spline_refiner_node_shift_rms_pixels"] = zero
            graph["spline_refiner_steps"] = zero
            graph["spline_refiner_node_source_error"] = zero
            graph["spline_refiner_span_source_error"] = zero
            return graph

        proposal_h = proposal_graph["spline_control_point_h_lr"].detach().float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].detach().float()
        mask_h = proposal_graph["spline_graph_mask_h"][:, 0].float()
        mask_v = proposal_graph["spline_graph_mask_v"][:, 0].float()
        if float(mask_h.sum().detach().cpu()) + float(mask_v.sum().detach().cpu()) <= 0.0:
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
            graph["spline_refiner_energy_before"] = zero
            graph["spline_refiner_energy_after"] = zero
            graph["spline_refiner_node_shift_rms_pixels"] = zero
            graph["spline_refiner_steps"] = zero
            graph["spline_refiner_node_source_error"] = zero
            graph["spline_refiner_span_source_error"] = zero
            return graph

        max_move_lattice = self.max_move_pixels / max(float(spline_graph.spacing_pixels), 1.0e-6)
        epsilon = 1.0e-3
        h_edge_start = torch.floor(proposal_h[..., 0]).detach()
        v_edge_start = torch.floor(proposal_v[..., 1]).detach()

        with torch.enable_grad():
            h_x = proposal_h[..., 0].clone().requires_grad_(True)
            v_y = proposal_v[..., 1].clone().requires_grad_(True)
            tangent_h = proposal_graph["spline_control_tangent_h"].detach().float().clone().requires_grad_(True)
            tangent_v = proposal_graph["spline_control_tangent_v"].detach().float().clone().requires_grad_(True)
            energy_before, _ = self._energy(
                spline_graph, proposal_graph, source_sdf_lr,
                h_x, v_y, tangent_h, tangent_v,
            )
            for _step in range(self.steps):
                energy, _ = self._energy(
                    spline_graph, proposal_graph, source_sdf_lr,
                    h_x, v_y, tangent_h, tangent_v,
                )
                gradients = torch.autograd.grad(
                    energy,
                    (h_x, v_y, tangent_h, tangent_v),
                    create_graph=False,
                    retain_graph=False,
                )
                with torch.no_grad():
                    h_x = h_x - self.position_step * gradients[0]
                    v_y = v_y - self.position_step * gradients[1]
                    tangent_h = tangent_h - self.tangent_step * gradients[2]
                    tangent_v = tangent_v - self.tangent_step * gradients[3]

                    h_min = torch.maximum(
                        h_edge_start + epsilon,
                        proposal_h[..., 0] - max_move_lattice,
                    )
                    h_max = torch.minimum(
                        h_edge_start + 1.0 - epsilon,
                        proposal_h[..., 0] + max_move_lattice,
                    )
                    v_min = torch.maximum(
                        v_edge_start + epsilon,
                        proposal_v[..., 1] - max_move_lattice,
                    )
                    v_max = torch.minimum(
                        v_edge_start + 1.0 - epsilon,
                        proposal_v[..., 1] + max_move_lattice,
                    )
                    h_x = torch.maximum(h_min, torch.minimum(h_max, h_x))
                    v_y = torch.maximum(v_min, torch.minimum(v_max, v_y))
                    tangent_h = F.normalize(tangent_h, dim=-1, eps=1.0e-6)
                    tangent_v = F.normalize(tangent_v, dim=-1, eps=1.0e-6)
                h_x = h_x.detach().requires_grad_(True)
                v_y = v_y.detach().requires_grad_(True)
                tangent_h = tangent_h.detach().requires_grad_(True)
                tangent_v = tangent_v.detach().requires_grad_(True)

            energy_after, parts = self._energy(
                spline_graph, proposal_graph, source_sdf_lr,
                h_x, v_y, tangent_h, tangent_v,
            )
            refined = self._graph_with_geometry(
                proposal_graph,
                h_x.detach(), v_y.detach(), tangent_h.detach(), tangent_v.detach(),
            )

        spacing = float(spline_graph.spacing_pixels)
        shift_numerator = (
            (((refined["spline_control_point_h_lr"][..., 0] - proposal_h[..., 0]) * spacing).square() * mask_h).sum()
            + (((refined["spline_control_point_v_lr"][..., 1] - proposal_v[..., 1]) * spacing).square() * mask_v).sum()
        )
        shift_denominator = (mask_h.sum() + mask_v.sum()).clamp_min(1.0)
        refined["spline_refiner_energy_before"] = energy_before.detach()
        refined["spline_refiner_energy_after"] = energy_after.detach()
        refined["spline_refiner_node_shift_rms_pixels"] = torch.sqrt(
            shift_numerator / shift_denominator + 1.0e-12
        ).detach()
        refined["spline_refiner_steps"] = source_sdf_lr.new_tensor(float(self.steps))
        refined["spline_refiner_node_source_error"] = parts["nodeSource"]
        refined["spline_refiner_span_source_error"] = parts["spanSource"]
        return refined
