from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _central_difference(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = value.float()
    xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
    yp = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return (
        0.5 * (xp[:, :, :, 2:] - xp[:, :, :, :-2]),
        0.5 * (yp[:, :, 2:, :] - yp[:, :, :-2, :]),
    )


class ConnectedSplineGraph(nn.Module):
    """Hard-connectivity marching-squares graph with shared cubic-Hermite nodes.

    The learned topology field decides which control-lattice edges are crossed.
    Every crossing edge owns one shared movable node and tangent, reused by both
    adjacent cells. Cells connect those shared nodes deterministically with the
    marching-squares topology rule and render cubic-Hermite spans. There is no
    independently movable per-cell or per-query zero-set.
    """

    def __init__(self, feature_channels: int, config) -> None:
        super().__init__()
        hidden = max(24, int(getattr(config, "spline_graph_hidden_channels", 96)))
        self.control_scale = max(1, int(getattr(config, "spline_graph_control_scale", 2)))
        self.output_scale = max(1, int(getattr(config, "target_scale", 4)))
        self.spacing_pixels = float(self.output_scale) / float(self.control_scale)
        self.origin_pixels = self.spacing_pixels
        self.max_distance_pixels = float(config.contour_sdf_max_distance_pixels)
        self.max_topology_delta_pixels = float(
            getattr(config, "spline_graph_max_topology_delta_pixels", 8.0)
        )
        self.topology_edit_band_pixels = float(
            getattr(config, "spline_graph_topology_edit_band_pixels", 4.0)
        )
        self.max_displacement_pixels = float(
            getattr(config, "spline_graph_max_displacement_pixels", 4.0)
        )
        self.max_tangent_residual = float(
            getattr(config, "spline_graph_max_tangent_residual", 0.75)
        )
        self.edit_band_pixels = float(getattr(config, "spline_graph_edit_band_pixels", 12.0))
        self.neighbour_radius = max(1, int(getattr(config, "spline_graph_neighbour_radius", 2)))
        self.samples_per_span = max(2, int(getattr(config, "spline_graph_samples_per_span", 4)))
        self.topology_head = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.geometry_head = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 8, 1),
        )
        nn.init.zeros_(self.topology_head[-1].weight)
        nn.init.zeros_(self.topology_head[-1].bias)
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        self._topology_locked = False

    def lock_topology(self) -> None:
        self._topology_locked = True
        for parameter in self.topology_head.parameters():
            parameter.requires_grad_(False)

    def unlock_topology(self) -> None:
        self._topology_locked = False
        for parameter in self.topology_head.parameters():
            parameter.requires_grad_(True)

    def restore_locked_topology_parameters(self) -> None:
        return None

    def _control_grid(
        self,
        batch: int,
        control_h: int,
        control_w: int,
        hr_h: int,
        hr_w: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(control_h, device=device, dtype=dtype),
            torch.arange(control_w, device=device, dtype=dtype),
            indexing="ij",
        )
        physical_x = self.origin_pixels + self.spacing_pixels * xx
        physical_y = self.origin_pixels + self.spacing_pixels * yy
        grid = torch.stack(
            (
                2.0 * physical_x / float(max(hr_w, 1)) - 1.0,
                2.0 * physical_y / float(max(hr_h, 1)) - 1.0,
            ),
            dim=-1,
        )
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    @staticmethod
    def _crossing_fraction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        denom = a - b
        eps = torch.where(
            denom >= 0.0,
            torch.full_like(denom, 1.0e-6),
            torch.full_like(denom, -1.0e-6),
        )
        return (a / (denom + eps)).clamp(0.0, 1.0)

    def _edge_graph(
        self,
        control_phi: torch.Tensor,
        geometry_raw: torch.Tensor,
        displacement_scale: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        batch, _channels, height, width = control_phi.shape
        gx, gy = _central_difference(control_phi)

        h_a, h_b = control_phi[:, :, :, :-1], control_phi[:, :, :, 1:]
        h_fraction = self._crossing_fraction(h_a, h_b)
        mask_h = ((h_a >= 0.0) != (h_b >= 0.0)).float().detach()
        h_y = torch.arange(
            height, device=control_phi.device, dtype=torch.float32
        ).view(1, 1, height, 1)
        h_x = torch.arange(
            width - 1, device=control_phi.device, dtype=torch.float32
        ).view(1, 1, 1, width - 1)
        source_h = torch.stack(
            (
                h_x.expand(batch, 1, height, width - 1) + h_fraction,
                h_y.expand(batch, 1, height, width - 1),
            ),
            dim=-1,
        )[:, 0]

        v_a, v_b = control_phi[:, :, :-1, :], control_phi[:, :, 1:, :]
        v_fraction = self._crossing_fraction(v_a, v_b)
        mask_v = ((v_a >= 0.0) != (v_b >= 0.0)).float().detach()
        v_y = torch.arange(
            height - 1, device=control_phi.device, dtype=torch.float32
        ).view(1, 1, height - 1, 1)
        v_x = torch.arange(
            width, device=control_phi.device, dtype=torch.float32
        ).view(1, 1, 1, width)
        source_v = torch.stack(
            (
                v_x.expand(batch, 1, height - 1, width),
                v_y.expand(batch, 1, height - 1, width) + v_fraction,
            ),
            dim=-1,
        )[:, 0]

        raw_h = 0.5 * (geometry_raw[:, :, :, :-1] + geometry_raw[:, :, :, 1:])
        raw_v = 0.5 * (geometry_raw[:, :, :-1, :] + geometry_raw[:, :, 1:, :])
        max_displacement_lattice = self.max_displacement_pixels / max(
            self.spacing_pixels, 1.0e-6
        )
        scale = torch.as_tensor(
            displacement_scale, device=control_phi.device, dtype=torch.float32
        )
        displacement_h = (
            torch.tanh(raw_h[:, 0:2]).permute(0, 2, 3, 1)
            * max_displacement_lattice
            * scale
        )
        displacement_v = (
            torch.tanh(raw_v[:, 2:4]).permute(0, 2, 3, 1)
            * max_displacement_lattice
            * scale
        )
        point_h = source_h + displacement_h
        point_v = source_v + displacement_v

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
            * self.max_tangent_residual,
            dim=-1,
            eps=1.0e-6,
        )
        tangent_v = F.normalize(
            tangent_v
            + torch.tanh(raw_v[:, 6:8]).permute(0, 2, 3, 1)
            * self.max_tangent_residual,
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

    def build_graph(
        self,
        topology_features: torch.Tensor,
        geometry_features: torch.Tensor,
        source_sdf_normalized: torch.Tensor,
        *,
        topology_scale: torch.Tensor | float = 1.0,
        displacement_scale: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        batch, _channels, h_lr, w_lr = source_sdf_normalized.shape
        control_h = (h_lr - 1) * self.control_scale + 1
        control_w = (w_lr - 1) * self.control_scale + 1
        hr_h, hr_w = h_lr * self.output_scale, w_lr * self.output_scale
        topology_control = F.interpolate(
            topology_features.float(),
            size=(control_h, control_w),
            mode="bilinear",
            align_corners=False,
        )
        geometry_control = F.interpolate(
            geometry_features.float(),
            size=(control_h, control_w),
            mode="bilinear",
            align_corners=False,
        )
        grid = self._control_grid(
            batch,
            control_h,
            control_w,
            hr_h,
            hr_w,
            device=source_sdf_normalized.device,
            dtype=torch.float32,
        )
        source_control = F.grid_sample(
            source_sdf_normalized.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ) * self.max_distance_pixels

        topology_raw = self.topology_head(topology_control)
        authority = (
            1.0
            - source_control.abs() / max(self.topology_edit_band_pixels, 1.0e-6)
        ).clamp(0.0, 1.0)
        topology_gain = torch.as_tensor(
            topology_scale, device=source_control.device, dtype=torch.float32
        )
        control_phi = source_control + (
            torch.tanh(topology_raw)
            * self.max_topology_delta_pixels
            * authority
            * topology_gain
        )
        geometry_raw = self.geometry_head(geometry_control)
        graph = self._edge_graph(control_phi, geometry_raw, displacement_scale)
        graph["spline_graph_control_phi_pixels"] = control_phi
        graph["spline_graph_source_control_phi_pixels"] = source_control
        graph["source_sdf_prior_lr"] = source_sdf_normalized.float()
        return graph

    @staticmethod
    def _gather_cell(
        value: torch.Tensor,
        ix: torch.Tensor,
        iy: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width = value.shape[:3]
        tail = value.shape[3:]
        flat = value.reshape(batch, height * width, *tail)
        index = (iy * width + ix).reshape(batch, -1)
        expand = index.view(
            batch, -1, *([1] * len(tail))
        ).expand(batch, index.shape[1], *tail)
        return torch.gather(flat, 1, expand).reshape(
            batch, *ix.shape[1:], *tail
        )

    def _cell_spans(
        self,
        graph: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        control_phi = graph["spline_graph_control_phi_pixels"].float()
        point_h = graph["spline_control_point_h_lr"].float()
        point_v = graph["spline_control_point_v_lr"].float()
        tangent_h = graph["spline_control_tangent_h"].float()
        tangent_v = graph["spline_control_tangent_v"].float()
        mask_h = graph["spline_graph_mask_h"][:, 0] > 0.5
        mask_v = graph["spline_graph_mask_v"][:, 0] > 0.5
        batch, _channels, height, width = control_phi.shape
        points = (
            point_h[:, :-1],
            point_v[:, :, 1:],
            point_h[:, 1:],
            point_v[:, :, :-1],
        )
        tangents = (
            tangent_h[:, :-1],
            tangent_v[:, :, 1:],
            tangent_h[:, 1:],
            tangent_v[:, :, :-1],
        )
        cross = torch.stack(
            (
                mask_h[:, :-1],
                mask_v[:, :, 1:],
                mask_h[:, 1:],
                mask_v[:, :, :-1],
            ),
            dim=-1,
        )
        count = cross.sum(dim=-1)
        ordinary = count == 2
        ambiguous = count == 4
        f00 = control_phi[:, 0, :-1, :-1]
        f10 = control_phi[:, 0, :-1, 1:]
        f01 = control_phi[:, 0, 1:, :-1]
        f11 = control_phi[:, 0, 1:, 1:]
        center = 0.25 * (f00 + f10 + f01 + f11)
        pair_a = ambiguous & ((f00 >= 0.0) == (center >= 0.0))
        pair_b = ambiguous & ~pair_a

        shape = (batch, height - 1, width - 1, 2, 2)
        p0 = control_phi.new_zeros(shape)
        p1 = control_phi.new_zeros(shape)
        t0 = control_phi.new_zeros(shape)
        t1 = control_phi.new_zeros(shape)
        active = torch.zeros(
            (batch, height - 1, width - 1, 2),
            device=control_phi.device,
            dtype=torch.bool,
        )
        for a_index, b_index in (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ):
            selected = ordinary & cross[..., a_index] & cross[..., b_index]
            p0[..., 0, :] = torch.where(
                selected[..., None], points[a_index], p0[..., 0, :]
            )
            p1[..., 0, :] = torch.where(
                selected[..., None], points[b_index], p1[..., 0, :]
            )
            t0[..., 0, :] = torch.where(
                selected[..., None], tangents[a_index], t0[..., 0, :]
            )
            t1[..., 0, :] = torch.where(
                selected[..., None], tangents[b_index], t1[..., 0, :]
            )
            active[..., 0] = active[..., 0] | selected
        for slot, a_index, b_index, selected in (
            (0, 0, 1, pair_a),
            (1, 2, 3, pair_a),
            (0, 0, 3, pair_b),
            (1, 1, 2, pair_b),
        ):
            p0[..., slot, :] = torch.where(
                selected[..., None], points[a_index], p0[..., slot, :]
            )
            p1[..., slot, :] = torch.where(
                selected[..., None], points[b_index], p1[..., slot, :]
            )
            t0[..., slot, :] = torch.where(
                selected[..., None], tangents[a_index], t0[..., slot, :]
            )
            t1[..., slot, :] = torch.where(
                selected[..., None], tangents[b_index], t1[..., slot, :]
            )
            active[..., slot] = active[..., slot] | selected
        return p0, p1, t0, t1, active

    def query(
        self,
        graph: dict[str, torch.Tensor],
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        control_phi = graph["spline_graph_control_phi_pixels"].float()
        source = graph["source_sdf_prior_lr"].float()
        batch, _channels, control_h, control_w = control_phi.shape
        query_h, query_w = query_grid.shape[1:3]
        physical_x = (
            query_grid[..., 0].float() + 1.0
        ) * (float(query_w) * 0.5)
        physical_y = (
            query_grid[..., 1].float() + 1.0
        ) * (float(query_h) * 0.5)
        query_x = (physical_x - self.origin_pixels) / self.spacing_pixels
        query_y = (physical_y - self.origin_pixels) / self.spacing_pixels
        base_x = torch.floor(query_x).long()
        base_y = torch.floor(query_y).long()

        p0, p1, t0, t1, active = self._cell_spans(graph)
        min_distance = torch.full(
            (batch, query_h, query_w),
            1.0e6,
            device=query_grid.device,
            dtype=torch.float32,
        )
        query = torch.stack((query_x, query_y), dim=-1).unsqueeze(-2)
        for offset_y in range(-self.neighbour_radius, self.neighbour_radius + 1):
            iy = (base_y + offset_y).clamp(0, control_h - 2)
            for offset_x in range(-self.neighbour_radius, self.neighbour_radius + 1):
                ix = (base_x + offset_x).clamp(0, control_w - 2)
                cell_p0 = self._gather_cell(p0, ix, iy)
                cell_p1 = self._gather_cell(p1, ix, iy)
                cell_t0 = self._gather_cell(t0, ix, iy)
                cell_t1 = self._gather_cell(t1, ix, iy)
                cell_active = self._gather_cell(active, ix, iy)
                chord = torch.linalg.vector_norm(
                    cell_p1 - cell_p0, dim=-1, keepdim=True
                ).clamp_min(1.0e-4)
                m0 = cell_t0 * chord
                m1 = cell_t1 * chord
                previous = cell_p0
                local_distance = torch.full(
                    cell_active.shape,
                    1.0e6,
                    device=query_grid.device,
                    dtype=torch.float32,
                )
                for sample_index in range(1, self.samples_per_span + 1):
                    s = float(sample_index) / float(self.samples_per_span)
                    s2, s3 = s * s, s * s * s
                    current = (
                        (2.0 * s3 - 3.0 * s2 + 1.0) * cell_p0
                        + (s3 - 2.0 * s2 + s) * m0
                        + (-2.0 * s3 + 3.0 * s2) * cell_p1
                        + (s3 - s2) * m1
                    )
                    segment = current - previous
                    projection = (
                        ((query - previous) * segment).sum(dim=-1)
                        / segment.square().sum(dim=-1).clamp_min(1.0e-8)
                    ).clamp(0.0, 1.0)
                    closest = previous + projection[..., None] * segment
                    distance = torch.linalg.vector_norm(
                        query - closest, dim=-1
                    ) * self.spacing_pixels
                    distance = torch.where(
                        cell_active,
                        distance,
                        torch.full_like(distance, 1.0e6),
                    )
                    local_distance = torch.minimum(local_distance, distance)
                    previous = current
                min_distance = torch.minimum(
                    min_distance, local_distance.min(dim=-1).values
                )

        control_grid = torch.stack(
            (
                2.0 * (query_x + 0.5) / float(max(control_w, 1)) - 1.0,
                2.0 * (query_y + 0.5) / float(max(control_h, 1)) - 1.0,
            ),
            dim=-1,
        )
        sampled_control = F.grid_sample(
            control_phi,
            control_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, 0]
        graph_sign = torch.where(sampled_control >= 0.0, 1.0, -1.0)
        graph_phi = graph_sign * min_distance
        sampled_source = F.grid_sample(
            source,
            query_grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, 0] * self.max_distance_pixels
        has_graph = min_distance < 1.0e5
        inner = max(2.0, self.edit_band_pixels * (2.0 / 3.0))
        outer = max(inner + 1.0, self.edit_band_pixels)
        authority = (
            (outer - sampled_source.abs()) / (outer - inner)
        ).clamp(0.0, 1.0)
        authority = authority * authority * (3.0 - 2.0 * authority)
        authority = authority * has_graph.float()
        phi = sampled_source * (1.0 - authority) + graph_phi * authority
        phi_field = phi.unsqueeze(1)
        graph_field = graph_phi.unsqueeze(1)

        gx, gy = _central_difference(phi_field)
        norm = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        normal = torch.cat((gx / norm, gy / norm), dim=1)
        nx_x, _ = _central_difference(normal[:, 0:1])
        _, ny_y = _central_difference(normal[:, 1:2])
        curvature = nx_x + ny_y
        source_field = sampled_source.unsqueeze(1)
        return {
            "phi_pixels": phi_field,
            "primitive_phi_pixels": graph_field,
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

    def forward(
        self,
        topology_features: torch.Tensor,
        geometry_features: torch.Tensor,
        source_sdf_normalized: torch.Tensor,
        query_grid: torch.Tensor,
        *,
        topology_scale: torch.Tensor | float = 1.0,
        displacement_scale: torch.Tensor | float = 1.0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        graph = self.build_graph(
            topology_features,
            geometry_features,
            source_sdf_normalized,
            topology_scale=topology_scale,
            displacement_scale=displacement_scale,
        )
        return {"graph": graph, "field": self.query(graph, query_grid)}
