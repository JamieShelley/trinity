from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor not found in {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1. New deterministic continuous-geometry refiner.  It intentionally has no
# trainable parameters and never receives target HR/authored tensors.
refiner_path = ROOT / "tools/nsamdr/neural/v9/explicit_spline_refiner.py"
refiner_path.write_text(r'''from __future__ import annotations

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
        proposal_h = proposal_graph["spline_control_point_h_lr"].float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].float()
        point_h = torch.stack((h_x, proposal_h[..., 1]), dim=-1)
        point_v = torch.stack((proposal_v[..., 0], v_y), dim=-1)
        source_h = proposal_graph["spline_source_control_point_h_lr"].float()
        source_v = proposal_graph["spline_source_control_point_v_lr"].float()
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
            graph = dict(proposal_graph)
            zero = source_sdf_lr.new_zeros(())
            graph["spline_refiner_energy_before"] = zero
            graph["spline_refiner_energy_after"] = zero
            graph["spline_refiner_node_shift_rms_pixels"] = zero
            graph["spline_refiner_steps"] = zero
            return graph

        proposal_h = proposal_graph["spline_control_point_h_lr"].detach().float()
        proposal_v = proposal_graph["spline_control_point_v_lr"].detach().float()
        mask_h = proposal_graph["spline_graph_mask_h"][:, 0].float()
        mask_v = proposal_graph["spline_graph_mask_v"][:, 0].float()
        if float(mask_h.sum().detach().cpu()) + float(mask_v.sum().detach().cpu()) <= 0.0:
            graph = dict(proposal_graph)
            zero = source_sdf_lr.new_zeros(())
            graph["spline_refiner_energy_before"] = zero
            graph["spline_refiner_energy_after"] = zero
            graph["spline_refiner_node_shift_rms_pixels"] = zero
            graph["spline_refiner_steps"] = zero
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
''', encoding="utf-8")

# 2. Configuration: fixed deterministic refiner budget/weights.
config = ROOT / "tools/nsamdr/neural/v9/config.py"
replace(
    config,
    "    spline_graph_lr_multiplier: float = 4.0\n\n    # V10.7.9 finite-width seam/ridge geometry.",
    "    spline_graph_lr_multiplier: float = 4.0\n\n"
    "    # V12.0 neural-proposal + explicit-geometry-refinement split. The refiner\n"
    "    # has no learned parameters and sees only observed LR structural evidence\n"
    "    # plus the neural proposal. It never consumes authored HR at inference.\n"
    "    spline_refiner_enabled: bool = True\n"
    "    spline_refiner_steps: int = 3\n"
    "    spline_refiner_position_step: float = 0.08\n"
    "    spline_refiner_tangent_step: float = 0.08\n"
    "    spline_refiner_source_node_weight: float = 0.35\n"
    "    spline_refiner_source_span_weight: float = 0.20\n"
    "    spline_refiner_proposal_position_weight: float = 1.00\n"
    "    spline_refiner_source_tangent_weight: float = 0.15\n"
    "    spline_refiner_proposal_tangent_weight: float = 0.60\n"
    "    spline_refiner_max_move_pixels: float = 0.75\n"
    "    spline_refiner_huber_beta_pixels: float = 0.35\n\n"
    "    # V10.7.9 finite-width seam/ridge geometry.",
)

# 3. Edge-constrained graph: schema + direct GT supervision applies to the
# neural proposal initializer, not the detached explicit-refined final geometry.
edge = ROOT / "tools/nsamdr/neural/v9/edge_constrained_spline_graph.py"
replace(
    edge,
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_PRESEAM_RESIDUAL_SPLINE_GRAPH_4X_V11_10_0"',
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_NEURAL_PROPOSAL_EXPLICIT_REFINER_SPLINE_GRAPH_4X_V12_0_0"',
)
replace(
    edge,
    '    spline_h = outputs.get("spline_control_point_h_lr")\n'
    '    spline_v = outputs.get("spline_control_point_v_lr")\n'
    '    spline_tan_h = outputs.get("spline_control_tangent_h")\n'
    '    spline_tan_v = outputs.get("spline_control_tangent_v")',
    '    # V12: supervise the neural initializer. Final continuous geometry is\n'
    '    # produced by the explicit LR-consistency refiner and is intentionally\n'
    '    # detached from the outer optimizer, matching the estimator/refiner split.\n'
    '    spline_h = outputs.get("spline_proposal_control_point_h_lr", outputs.get("spline_control_point_h_lr"))\n'
    '    spline_v = outputs.get("spline_proposal_control_point_v_lr", outputs.get("spline_control_point_v_lr"))\n'
    '    spline_tan_h = outputs.get("spline_proposal_control_tangent_h", outputs.get("spline_control_tangent_h"))\n'
    '    spline_tan_v = outputs.get("spline_proposal_control_tangent_v", outputs.get("spline_control_tangent_v"))',
)

# 4. Production contract: compose proposal -> explicit refiner -> renderer.
local = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"
replace(
    local,
    'from .spline_graph import ConnectedSplineGraph\n',
    'from .spline_graph import ConnectedSplineGraph\nfrom .explicit_spline_refiner import ExplicitSplineGeometryRefiner\n',
)
replace(
    local,
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_PRESEAM_RESIDUAL_SPLINE_GRAPH_4X_V11_10_0"',
    'SCHEMA = "NSAMDR_RAVEN_PRODUCTION_NEURAL_PROPOSAL_EXPLICIT_REFINER_SPLINE_GRAPH_4X_V12_0_0"',
)
replace(
    local,
    '            "spline_graph_mask_h",\n'
    '            "spline_graph_mask_v",\n'
    '        )',
    '            "spline_graph_mask_h",\n'
    '            "spline_graph_mask_v",\n'
    '            "spline_proposal_control_point_h_lr",\n'
    '            "spline_proposal_control_point_v_lr",\n'
    '            "spline_proposal_control_tangent_h",\n'
    '            "spline_proposal_control_tangent_v",\n'
    '            "spline_refiner_energy_before",\n'
    '            "spline_refiner_energy_after",\n'
    '            "spline_refiner_node_shift_rms_pixels",\n'
    '            "spline_refiner_steps",\n'
    '            "spline_refiner_node_source_error",\n'
    '            "spline_refiner_span_source_error",\n'
    '        )',
)
replace(
    local,
    '            "geometryPrediction": (\n'
    '                "bounded 2x topology field -> shared edge-crossing nodes -> "\n'
    '                "connected cubic-Hermite contour graph -> metric SDF"\n'
    '            ),',
    '            "geometryPrediction": (\n'
    '                "neural topology + continuous spline proposal -> explicit "\n'
    '                "LR-consistency geometry refinement -> connected cubic-Hermite "\n'
    '                "contour graph -> metric SDF"\n'
    '            ),',
)
replace(
    local,
    '            "b1bObjective": "shared graph-node, tangent, span-smoothness, metric-SDF and same-renderer reconstruction",',
    '            "b1bObjective": (\n'
    '                "GT-supervised neural graph initializer + deterministic LR-consistency "\n'
    '                "continuous refinement + baseline-relative structural authority"\n'
    '            ),\n'
    '            "neuralGeometryIsInitializerOnly": True,\n'
    '            "explicitGeometryRefinement": True,\n'
    '            "explicitGeometryRefinementUsesTargetHR": False,\n'
    '            "explicitGeometryRefinementCanChangeTopology": False,',
)
replace(
    local,
    '            "structural representation": (\n'
    '                "geometry_net.production_structure",\n'
    '                model.geometry_net.production_structure,\n'
    '            ),\n'
    '            "boundary renderer":',
    '            "structural representation": (\n'
    '                "geometry_net.production_structure",\n'
    '                model.geometry_net.production_structure,\n'
    '            ),\n'
    '            "explicit geometry refiner": (\n'
    '                "geometry_net.production_structure.geometry_refiner",\n'
    '                model.geometry_net.production_structure.geometry_refiner,\n'
    '            ),\n'
    '            "boundary renderer":',
)
replace(
    local,
    '            "structural representation": "geometry_net.production_structure",\n'
    '            "boundary renderer": "boundary_renderer",',
    '            "structural representation": "geometry_net.production_structure",\n'
    '            "explicit geometry refiner": "geometry_net.production_structure.geometry_refiner",\n'
    '            "boundary renderer": "boundary_renderer",',
)
replace(
    local,
    '        self.spline_graph = ConnectedSplineGraph(feature_channels, config)\n'
    '        self._topology_bootstrap_only = False',
    '        self.spline_graph = ConnectedSplineGraph(feature_channels, config)\n'
    '        # V12: the network proposes continuous geometry; this deterministic,\n'
    '        # parameter-free module obtains the final continuous configuration.\n'
    '        self.geometry_refiner = ExplicitSplineGeometryRefiner(config)\n'
    '        self._topology_bootstrap_only = False',
)
replace(
    local,
    '        head = self.decoder.parameter_head\n'
    '        head.lock_topology()\n'
    '        self.spline_graph.lock_topology()\n'
    '        for parameter in self.topology_feature_project.parameters():\n'
    '            parameter.requires_grad_(False)\n'
    '        for parameter in self.geometry_feature_project.parameters():\n'
    '            parameter.requires_grad_(True)\n'
    '        # B1b is the first phase allowed to earn structural authority over B.\n'
    '        for parameter in self.structural_residual_gain_head.parameters():\n'
    '            parameter.requires_grad_(True)\n'
    '        for parameter in head.geometry_net.parameters():\n'
    '            parameter.requires_grad_(True)\n'
    '        for parameter in self.spline_graph.geometry_head.parameters():\n'
    '            parameter.requires_grad_(True)',
    '        head = self.decoder.parameter_head\n'
    '        head.lock_topology()\n'
    '        self.spline_graph.lock_topology()\n'
    '        for parameter in self.topology_feature_project.parameters():\n'
    '            parameter.requires_grad_(False)\n'
    '        for parameter in self.geometry_feature_project.parameters():\n'
    '            parameter.requires_grad_(True)\n'
    '        # V12 B1b: learned geometry is an initializer only. The retired local\n'
    '        # decoder stays telemetry-only; final continuous geometry comes from\n'
    '        # the parameter-free explicit refiner using observed LR evidence.\n'
    '        for parameter in self.structural_residual_gain_head.parameters():\n'
    '            parameter.requires_grad_(True)\n'
    '        for parameter in self.decoder.parameters():\n'
    '            parameter.requires_grad_(False)\n'
    '        for parameter in self.spline_graph.geometry_head.parameters():\n'
    '            parameter.requires_grad_(True)',
)
replace(
    local,
    '        spline = self.spline_graph(\n'
    '            topology_feature_grid,\n'
    '            geometry_feature_grid,\n'
    '            source_prior_lr,\n'
    '            query_grid,\n'
    '            topology_scale=distance_scale,\n'
    '            displacement_scale=geometry_scale,\n'
    '        )\n'
    '        field = self._apply_query_genome(spline["field"])\n'
    '        return {\n'
    '            "feature_grid": geometry_feature_grid,\n'
    '            "context": context,\n'
    '            "spline_graph": spline["graph"],',
    '        proposal_graph = self.spline_graph.build_graph(\n'
    '            topology_feature_grid,\n'
    '            geometry_feature_grid,\n'
    '            source_prior_lr,\n'
    '            topology_scale=distance_scale,\n'
    '            displacement_scale=geometry_scale,\n'
    '        )\n'
    '        if self._topology_bootstrap_only:\n'
    '            refined_graph = dict(proposal_graph)\n'
    '            zero = source_prior_lr.new_zeros(())\n'
    '            for key in (\n'
    '                "spline_refiner_energy_before", "spline_refiner_energy_after",\n'
    '                "spline_refiner_node_shift_rms_pixels", "spline_refiner_steps",\n'
    '                "spline_refiner_node_source_error", "spline_refiner_span_source_error",\n'
    '            ):\n'
    '                refined_graph[key] = zero\n'
    '        else:\n'
    '            refined_graph = self.geometry_refiner(\n'
    '                self.spline_graph, proposal_graph, source_prior_lr\n'
    '            )\n'
    '        refined_graph["spline_proposal_control_point_h_lr"] = proposal_graph["spline_control_point_h_lr"]\n'
    '        refined_graph["spline_proposal_control_point_v_lr"] = proposal_graph["spline_control_point_v_lr"]\n'
    '        refined_graph["spline_proposal_control_tangent_h"] = proposal_graph["spline_control_tangent_h"]\n'
    '        refined_graph["spline_proposal_control_tangent_v"] = proposal_graph["spline_control_tangent_v"]\n'
    '        field = self._apply_query_genome(\n'
    '            self.spline_graph.query(refined_graph, query_grid)\n'
    '        )\n'
    '        return {\n'
    '            "feature_grid": geometry_feature_grid,\n'
    '            "context": context,\n'
    '            "spline_graph": refined_graph,',
)

# B1b outer training now learns the initializer + residual authority. Dense
# final-spline metrics remain diagnostics of the deterministic solver, not fake
# gradients into a detached inner optimization.
old_b1b = '''        else:\n            total = (\n                losses["sdf_surface"] * float(config.sdf_surface_weight)\n                + losses["sdf_sign"] * float(config.sdf_sign_weight)\n                + losses["sdf_topology_sign"] * float(config.sdf_topology_weight)\n                + losses["spline_graph_topology_control"] * float(config.spline_graph_topology_control_weight)\n                + losses["spline_graph_topology_sign"] * float(config.spline_graph_topology_sign_weight)\n                + losses["spline_graph_point"] * float(config.spline_graph_point_weight)\n                + losses["spline_graph_tangent"] * float(config.spline_graph_tangent_weight)\n                + losses["spline_graph_span_smoothness"] * float(config.spline_graph_span_smoothness_weight)\n                + losses["spline_graph_span_tangent"] * float(config.spline_graph_span_tangent_weight)\n                + losses["spline_graph_span_separation"] * float(config.spline_graph_span_separation_weight)\n                + losses["spline_graph_sdf"] * float(config.spline_graph_sdf_weight)\n                + losses["spline_graph_gradient"] * float(config.spline_graph_gradient_weight)\n                + losses["spline_graph_eikonal"] * float(config.spline_graph_eikonal_weight)\n                + losses["spline_graph_curvature"] * float(config.spline_graph_curvature_weight)\n                + losses["spline_metric_offset"] * float(config.spline_metric_offset_weight)\n                + losses["spline_metric_eikonal_near"] * float(config.spline_metric_eikonal_near_weight)\n                + losses["edge"] * float(config.edge_weight)\n                + losses["edge_sdf_consistency"] * float(config.boundary_edge_sdf_consistency_weight)\n                + losses["orientation"] * float(config.orientation_weight)\n                + losses["hardness"] * float(config.boundary_hardness_weight)\n            )\n'''
new_b1b = '''        else:\n            # V12 B1b outer optimization trains the neural initializer, not the\n            # final solver state. Point/tangent teachers therefore target proposal\n            # parameters; the parameter-free refiner obtains final geometry from\n            # that proposal plus observed LR consistency. Final dense spline/SDF\n            # terms remain qualification telemetry and cannot masquerade as an\n            # outer-loop gradient through the detached explicit optimizer.\n            total = (\n                losses["spline_graph_point"] * float(config.spline_graph_point_weight)\n                + losses["spline_graph_tangent"] * float(config.spline_graph_tangent_weight)\n                + losses["edge"] * float(config.edge_weight)\n                + losses["edge_sdf_consistency"] * float(config.boundary_edge_sdf_consistency_weight)\n                + losses["orientation"] * float(config.orientation_weight)\n                + losses["hardness"] * float(config.boundary_hardness_weight)\n            )\n'''
replace(local, old_b1b, new_b1b)

# 5. Tests: schema + architectural split assertions.
test117 = ROOT / "tools/nsamdr/tests/test_v117_baseline_relative_contract.py"
text = test117.read_text(encoding="utf-8")
text = text.replace(
    "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_PRESEAM_RESIDUAL_SPLINE_GRAPH_4X_V11_10_0",
    "NSAMDR_RAVEN_PRODUCTION_NEURAL_PROPOSAL_EXPLICIT_REFINER_SPLINE_GRAPH_4X_V12_0_0",
)
test117.write_text(text, encoding="utf-8")

test120 = ROOT / "tools/nsamdr/tests/test_v120_explicit_geometry_refiner_contract.py"
test120.write_text(r'''from __future__ import annotations

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
''', encoding="utf-8")

# 6. Design/readme: make the estimator/refiner split explicit and auditable.
design = ROOT / "tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md"
design_text = design.read_text(encoding="utf-8")
if "## V12.0 estimator/refiner architecture contract" not in design_text:
    design_text += r'''

## V12.0 estimator/refiner architecture contract

B1 continuous geometry is no longer treated as a one-shot neural prediction.
The neural branch proposes fixed topology plus initial same-edge node positions
and tangents. A separate **parameter-free explicit geometry refiner** then
optimizes only those continuous parameters against the observed LR source-SDF
evidence while remaining bounded around the neural proposal. The topology masks
are immutable during refinement. The explicit optimizer never receives authored
HR targets, so training and production use the same refinement evidence.

This deliberately follows the decomposition used by *Deep Vectorization of
Technical Drawings*: learned estimation supplies an initial primitive
configuration and an iterative geometric optimization obtains the final
configuration. It also uses DiffVG/LIVE only for the narrower lesson that
continuous vector parameters can be optimized against raster evidence; discrete
topology remains outside that optimization.

B1b outer SGD therefore supervises the **neural proposal initializer** with
held-out authored geometry teachers. The explicit refiner is detached from that
outer optimizer and has no parameters to train. Final B1 qualification still
judges the actual refined pre-seam C against B; the residual authority gate may
open only when that final refined geometry produces a real baseline-relative
improvement.
'''
design.write_text(design_text, encoding="utf-8")

readme = ROOT / "tools/nsamdr/README.md"
replace(
    readme,
    "The current production structural implementation uses a learned bounded 2x\n"
    "topology field to form a hard-connected marching-squares graph. Shared\n"
    "edge-crossing nodes and tangents are rendered as connected cubic-Hermite\n"
    "spans and queried as a metric SDF. V11.6 constrains each learned crossing to\n"
    "its owning control edge, while V11.7 gives baseline-relative regret losses\n"
    "direct optimisation authority during structural training. The retired\n"
    "whole-tile primitive classifier/regressor remains compatibility telemetry\n"
    "only and has no production structural authority.",
    "The current production structural implementation separates **neural estimation**\n"
    "from **final geometric refinement**. A learned bounded topology field forms a\n"
    "hard-connected marching-squares graph and predicts an initial set of same-edge\n"
    "crossing nodes/tangents. V12 then runs a parameter-free explicit optimizer over\n"
    "only those continuous spline parameters using observed LR structural evidence\n"
    "plus a bounded prior to the neural proposal; topology cannot change in this\n"
    "step and authored HR is never an inference input. The refined connected\n"
    "cubic-Hermite graph is queried as the production metric SDF, after which the\n"
    "baseline-relative residual authority decides whether the redraw can replace B.\n"
    "The retired whole-tile primitive classifier/regressor remains compatibility\n"
    "telemetry only and has no production structural authority.",
)

print("Applied V12 neural-proposal + explicit-geometry-refiner architecture")
