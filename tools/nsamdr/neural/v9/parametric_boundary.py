"""Local analytic boundary primitives for NSAMDR V9.9.3.

The neural network predicts compact geometric parameters on the LR lattice;
it does not emit an independent HR SDF value per query.  Each control point
owns up to three local second-order signed-distance primitives (line/arc), plus
a small CSG selector that chooses a single primitive, a smooth union, or a
smooth intersection.  This covers straight edges, smooth curves, corners,
stripes and multi-branch junctions without reintroducing per-pixel Fourier or
modulo-phase geometry authority.

The renderer queries the resulting field at arbitrary output/subpixel
coordinates and applies plateau reconstruction only in a compact boundary band.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ParametricBoundaryService:
    # Purpose: Implement make query grid for ParametricBoundaryService.
    # Called by: supersample_coverage
    # Calls: No same-class helper methods.
    def make_query_grid(
        self,
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        offset_x_pixels: float = 0.0,
        offset_y_pixels: float = 0.0,
    ) -> torch.Tensor:
        yy = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5 + float(offset_y_pixels))
            * (2.0 / max(height, 1))
            - 1.0
        )
        xx = (
            (torch.arange(width, device=device, dtype=dtype) + 0.5 + float(offset_x_pixels))
            * (2.0 / max(width, 1))
            - 1.0
        )
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    # Purpose: Implement sample for ParametricBoundaryService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _sample(self, value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            value.float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=False
        )

    # Purpose: Implement central difference for ParametricBoundaryService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _central_difference(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = value.float()
        xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
        yp = F.pad(x, (0, 0, 1, 1), mode="replicate")
        gx = 0.5 * (xp[:, :, :, 2:] - xp[:, :, :, :-2])
        gy = 0.5 * (yp[:, :, 2:, :] - yp[:, :, :-2, :])
        return gx, gy

    # Purpose: Implement gather control for ParametricBoundaryService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _gather_control(self, field: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        b, c, h, w = field.shape
        ix = ix.clamp(0, w - 1)
        iy = iy.clamp(0, h - 1)
        index = (iy * w + ix).reshape(b, 1, -1).expand(-1, c, -1)
        return torch.gather(field.reshape(b, c, h * w), 2, index).reshape(
            b, c, ix.shape[-2], ix.shape[-1]
        )

    # Purpose: Implement rotate for ParametricBoundaryService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _rotate(self, nx: torch.Tensor, ny: torch.Tensor, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
        ca, sa = math.cos(angle), math.sin(angle)
        return ca * nx - sa * ny, sa * nx + ca * ny

    # Purpose: Implement supersample coverage for ParametricBoundaryService.
    # Called by: External callers and the owning workflow.
    # Calls: make_query_grid
    def supersample_coverage(
        self,
        query_fn,
        *,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        offsets: tuple[tuple[float, float], ...],
        transition_width: torch.Tensor,
        return_samples: bool = False,
    ):
        center_grid = self.make_query_grid(batch, height, width, device=device)
        center_phi = query_fn(center_grid).float()
        accum = torch.zeros_like(center_phi)
        samples: list[torch.Tensor] = []
        width_field = transition_width.float().clamp_min(0.20)
        for ox, oy in offsets:
            grid = self.make_query_grid(
                batch, height, width, device=device,
                offset_x_pixels=float(ox), offset_y_pixels=float(oy),
            )
            phi = query_fn(grid).float()
            if return_samples:
                samples.append(phi)
            t = (0.5 - phi / width_field).clamp(0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            accum = accum + t
        coverage = accum / float(max(len(offsets), 1))
        if return_samples:
            return coverage, center_phi, torch.cat(samples, dim=1) if samples else center_phi
        return coverage, center_phi

_parametric_boundary_service = ParametricBoundaryService()
make_query_grid = _parametric_boundary_service.make_query_grid
_sample = _parametric_boundary_service._sample
_central_difference = _parametric_boundary_service._central_difference
_gather_control = _parametric_boundary_service._gather_control
_rotate = _parametric_boundary_service._rotate
supersample_coverage = _parametric_boundary_service.supersample_coverage


class PrimitiveParameterHead(nn.Module):
    """Predict continuous geometry and topology through independent feature heads.

    B1a trains both branches. B1b freezes the topology branch completely while the
    continuous branch retains enough hidden capacity to de-rasterise anchor, normal,
    curvature, ribbon width and confidence without changing branch/CSG decisions.
    """

    BRANCH_STRIDE = 6
    OUTPUTS = 24
    TOPOLOGY_CHANNELS = (5, 11, 17, 18, 19, 20, 21, 22)
    GEOMETRY_CHANNELS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 23)

    # Purpose: Build one independent parameter-prediction branch.
    # Called by: __init__.
    # Calls: torch.nn.Conv2d(), torch.nn.GELU().
    @staticmethod
    def _make_branch(in_channels: int, hidden: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, outputs, 1),
        )

    # Purpose: Implement init for PrimitiveParameterHead.
    # Called by: LocalParametricBoundaryDecoder.__init__.
    # Calls: _make_branch.
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = max(24, int(hidden_channels))
        self.geometry_net = self._make_branch(
            in_channels, hidden, len(self.GEOMETRY_CHANNELS)
        )
        self.topology_net = self._make_branch(
            in_channels, hidden, len(self.TOPOLOGY_CHANNELS)
        )
        nn.init.zeros_(self.geometry_net[-1].weight)
        nn.init.zeros_(self.geometry_net[-1].bias)
        nn.init.zeros_(self.topology_net[-1].weight)
        nn.init.zeros_(self.topology_net[-1].bias)
        with torch.no_grad():
            self.topology_net[-1].bias[0:5].fill_(-1.5)
            self.topology_net[-1].bias[5] = 1.5
            self.topology_net[-1].bias[6:8].fill_(-1.5)
        self._topology_locked = False

    # Purpose: Merge independent geometry/topology channels into the public 24-channel layout.
    # Called by: forward.
    # Calls: torch.cat().
    def _merge_outputs(
        self,
        geometry: torch.Tensor,
        topology: torch.Tensor,
    ) -> torch.Tensor:
        geometry_parts = {
            channel: geometry[:, index:index + 1]
            for index, channel in enumerate(self.GEOMETRY_CHANNELS)
        }
        topology_parts = {
            channel: topology[:, index:index + 1]
            for index, channel in enumerate(self.TOPOLOGY_CHANNELS)
        }
        return torch.cat(
            [
                topology_parts[channel]
                if channel in topology_parts
                else geometry_parts[channel]
                for channel in range(self.OUTPUTS)
            ],
            dim=1,
        )

    # Purpose: Freeze the complete topology predictor after B1a qualification.
    # Called by: LocalBoundaryProductionStructure.lock_topology_for_proof.
    # Calls: torch.Tensor.requires_grad_().
    def lock_topology(self) -> None:
        self._topology_locked = True
        for parameter in self.topology_net.parameters():
            parameter.requires_grad_(False)

    # Purpose: Restore topology trainability for a fresh B1a bootstrap.
    # Called by: LocalBoundaryProductionStructure.unlock_topology_for_bootstrap.
    # Calls: torch.Tensor.requires_grad_().
    def unlock_topology(self) -> None:
        self._topology_locked = False
        for parameter in self.topology_net.parameters():
            parameter.requires_grad_(True)

    # Purpose: Preserve the legacy trainer callback after the heads are physically split.
    # Called by: LocalBoundaryProductionStructure.restore_locked_topology_parameters.
    # Calls: No same-class helper methods.
    def restore_locked_topology_parameters(self) -> None:
        return None

    # Purpose: Predict and merge continuous geometry with independent topology controls.
    # Called by: LocalParametricBoundaryDecoder.build_context.
    # Calls: _merge_outputs.
    def forward(
        self,
        x: torch.Tensor,
        topology_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if topology_x is None:
            topology_x = x
        geometry_raw = self.geometry_net(x)
        topology_raw = self.topology_net(topology_x)
        raw = self._merge_outputs(geometry_raw, topology_raw)
        smooth = F.avg_pool2d(raw, kernel_size=3, stride=1, padding=1)
        parts = []
        for branch in range(3):
            offset = branch * self.BRANCH_STRIDE
            analytic = raw[:, offset:offset + 4]
            width = (
                0.45 * raw[:, offset + 4:offset + 5]
                + 0.55 * smooth[:, offset + 4:offset + 5]
            )
            geom = torch.cat((analytic, width), dim=1)
            parts.append(torch.cat((geom, raw[:, offset + 5:offset + 6]), dim=1))
        tail = raw[:, 18:]
        return torch.cat((*parts, tail), dim=1)


class LocalParametricBoundaryDecoder(nn.Module):
    """Continuous metric SDF reconstructed from coherent topology-controlled scalar fields."""

    # Purpose: Implement init for LocalParametricBoundaryDecoder.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        *,
        max_distance_pixels: float,
        max_offset_pixels: float = 6.0,
        max_normal_correction: float = 1.5,
        max_curvature_per_pixel: float = 0.35,
        max_ribbon_half_width_pixels: float = 6.0,
        control_scale: int = 1,
        output_scale: int = 4,
    ) -> None:
        super().__init__()
        self.max_distance_pixels = float(max_distance_pixels)
        self.max_offset_pixels = float(max_offset_pixels)
        self.max_normal_correction = float(max_normal_correction)
        self.max_curvature_per_pixel = float(max_curvature_per_pixel)
        self.max_ribbon_half_width_pixels = float(max_ribbon_half_width_pixels)
        self.control_scale = max(1, int(control_scale))
        self.output_scale = max(1, int(output_scale))
        self.parameter_head = PrimitiveParameterHead(feature_channels + 4, hidden_channels)

    # Purpose: Smooth a control-lattice geometry field without biasing an ideal affine line.
    # Called by: _coherent_branch_geometry.
    # Calls: torch.nn.functional.conv2d(), torch.nn.functional.pad().
    @staticmethod
    def _smooth_geometry_field(value: torch.Tensor) -> torch.Tensor:
        channels = int(value.shape[1])
        binomial = value.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0
        kernel_x = binomial.view(1, 1, 1, 5).repeat(channels, 1, 1, 1)
        kernel_y = binomial.view(1, 1, 5, 1).repeat(channels, 1, 1, 1)
        result = F.conv2d(
            F.pad(value.float(), (2, 2, 0, 0), mode="replicate"),
            kernel_x,
            groups=channels,
        )
        return F.conv2d(
            F.pad(result, (0, 0, 2, 2), mode="replicate"),
            kernel_y,
            groups=channels,
        )

    # Purpose: Convert learned anchor distances into one spatially coherent analytic field.
    # Called by: build_context.
    # Calls: _smooth_geometry_field, _central_difference.
    def _coherent_branch_geometry(
        self,
        distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coherent_distance = self._smooth_geometry_field(distance)
        gx, gy = _central_difference(coherent_distance)
        norm = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        nx, ny = gx / norm, gy / norm
        nx_x, _ = _central_difference(nx)
        _, ny_y = _central_difference(ny)
        curvature = self._smooth_geometry_field(
            (nx_x + ny_y) / float(self.output_scale)
        )
        return coherent_distance, nx, ny, curvature

    # Purpose: Implement branch parameters for LocalParametricBoundaryDecoder.
    # Called by: build_context
    # Calls: No same-class helper methods.
    def _branch_parameters(
        self,
        raw: torch.Tensor,
        base_nx: torch.Tensor,
        base_ny: torch.Tensor,
        base_curvature: torch.Tensor,
        source_fit: torch.Tensor,
        branch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = branch * PrimitiveParameterHead.BRANCH_STRIDE
        distance = source_fit + torch.tanh(raw[:, start:start + 1]) * self.max_offset_pixels
        if branch == 0:
            rnx, rny = base_nx, base_ny
        elif branch == 1:
            rnx, rny = _rotate(base_nx, base_ny, 2.0 * math.pi / 3.0)
        else:
            rnx, rny = _rotate(base_nx, base_ny, -2.0 * math.pi / 3.0)
        corr = torch.tanh(raw[:, start + 1:start + 3]) * self.max_normal_correction
        nx = rnx + corr[:, 0:1]
        ny = rny + corr[:, 1:2]
        norm = torch.sqrt(nx.square() + ny.square() + 1.0e-6)
        nx, ny = nx / norm, ny / norm
        curvature = base_curvature + torch.tanh(raw[:, start + 3:start + 4]) * self.max_curvature_per_pixel
        # A ribbon is the signed distance to a centre line/arc minus a learned
        # half width.  This explicitly represents both edges of thin strokes.
        half_width = torch.sigmoid(raw[:, start + 4:start + 5]) * self.max_ribbon_half_width_pixels
        ribbon_mode = torch.sigmoid(raw[:, start + 5:start + 6])
        return distance, nx, ny, curvature, half_width, ribbon_mode

    # Purpose: Implement build context for LocalParametricBoundaryDecoder.
    # Called by: forward
    # Calls: _branch_parameters
    def build_context(
        self,
        feature_grid: torch.Tensor,
        source_sdf_normalized: torch.Tensor,
        topology_feature_grid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source = source_sdf_normalized.float().clamp(-1.0, 1.0)
        source_pixels = source * self.max_distance_pixels
        # The LR prior is evidence, not the desired contour. Build a phase-neutral
        # local polynomial basis before deriving anchor normals/curvature so one LR
        # staircase cannot become the analytic tangent field. A separable 5-tap
        # binomial filter removes Nyquist raster phase while preserving lines/arcs
        # and keeps the learned bounded residual responsible for exact placement.
        binomial = source_pixels.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0
        kernel_x = binomial.view(1, 1, 1, 5)
        kernel_y = binomial.view(1, 1, 5, 1)
        source_fit = F.conv2d(
            F.pad(source_pixels, (2, 2, 0, 0), mode="replicate"), kernel_x
        )
        source_fit = F.conv2d(
            F.pad(source_fit, (0, 0, 2, 2), mode="replicate"), kernel_y
        )
        gx, gy = _central_difference(source_fit)
        grad = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        base_nx, base_ny = gx / grad, gy / grad
        nx_x, _ = _central_difference(base_nx)
        _, ny_y = _central_difference(base_ny)
        base_curvature = (nx_x + ny_y) / float(self.output_scale)

        if topology_feature_grid is None:
            topology_feature_grid = feature_grid
        common_evidence = (
            (source_fit / max(self.max_distance_pixels, 1.0e-6)).clamp(-1.0, 1.0),
            base_nx,
            base_ny,
            (base_curvature / max(self.max_curvature_per_pixel, 1.0e-6)).clamp(-1.0, 1.0),
        )
        geometry_head_input = torch.cat((feature_grid.float(), *common_evidence), dim=1)
        topology_head_input = torch.cat(
            (topology_feature_grid.float(), *common_evidence), dim=1
        )
        raw = self.parameter_head(geometry_head_input, topology_head_input)

        branch_data = [
            self._branch_parameters(raw, base_nx, base_ny, base_curvature, source_fit, i)
            for i in range(3)
        ]
        raw_branch_distance = torch.cat([v[0] for v in branch_data], dim=1)
        (
            branch_distance,
            branch_normal_x,
            branch_normal_y,
            branch_curvature,
        ) = self._coherent_branch_geometry(raw_branch_distance)
        branch_half_width = torch.cat([v[4] for v in branch_data], dim=1)
        branch_ribbon_mode = torch.cat([v[5] for v in branch_data], dim=1)
        # Primary always active; extra branches learn activation only where a
        # corner/junction composition is actually required.
        branch_activation = torch.cat((
            torch.ones_like(raw[:, 18:19]),
            torch.sigmoid(raw[:, 18:19]),
            torch.sigmoid(raw[:, 19:20]),
        ), dim=1)
        csg_logits = raw[:, 20:23]
        confidence = torch.sigmoid(raw[:, 23:24])

        control_size = (
            source.shape[-2] * self.control_scale,
            source.shape[-1] * self.control_scale,
        )
        def up(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(x, size=control_size, mode="bilinear", align_corners=False)

        d = up(branch_distance)
        nx = up(branch_normal_x)
        ny = up(branch_normal_y)
        norm = torch.sqrt(nx.square() + ny.square() + 1.0e-6)
        nx, ny = nx / norm, ny / norm
        return {
            "source_sdf_prior_lr": source,
            "branch_anchor_distance_pixels": d,
            "branch_normal_x": nx,
            "branch_normal_y": ny,
            "branch_curvature_per_pixel": up(branch_curvature),
            "branch_half_width_pixels": up(branch_half_width),
            "branch_ribbon_mode": up(branch_ribbon_mode),
            "branch_activation": up(branch_activation),
            "csg_logits": up(csg_logits),
            "confidence": up(confidence),
            # Primary aliases used by losses/telemetry.
            "anchor_distance_pixels": d[:, 0:1],
            "normal_x": nx[:, 0:1],
            "normal_y": ny[:, 0:1],
            "curvature_per_pixel": up(branch_curvature[:, 0:1]),
            "ribbon_half_width_pixels": up(branch_half_width[:, 0:1]),
            "ribbon_mode": up(branch_ribbon_mode[:, 0:1]),
            "distance_delta_pixels": up(branch_distance[:, 0:1] - source_fit),
            "junction_hint": up(torch.maximum(branch_activation[:, 1:2], branch_activation[:, 2:3])),
        }

    # Purpose: Sample one control-lattice scalar field as a globally stitched C1 surface.
    # Called by: query.
    # Calls: _central_difference, _gather_control.
    @staticmethod
    def _hermite_sample_scalar(
        field: torch.Tensor,
        query_grid: torch.Tensor,
    ) -> torch.Tensor:
        _batch, _channels, height, width = field.shape
        if height < 2 or width < 2:
            return F.grid_sample(
                field.float(), query_grid.float(), mode="bilinear",
                padding_mode="border", align_corners=False,
            )

        # One scalar lattice owns values and every derivative. Shared node jets make
        # adjacent bicubic cells agree in both value and first derivative (C1).
        fx, fy = _central_difference(field)
        fx = fx.clone()
        fy = fy.clone()
        fx[:, :, :, 0] = field[:, :, :, 1] - field[:, :, :, 0]
        fx[:, :, :, -1] = field[:, :, :, -1] - field[:, :, :, -2]
        fy[:, :, 0, :] = field[:, :, 1, :] - field[:, :, 0, :]
        fy[:, :, -1, :] = field[:, :, -1, :] - field[:, :, -2, :]
        _unused_fxx, fxy = _central_difference(fx)
        fxy = fxy.clone()
        fxy[:, :, 0, :] = fx[:, :, 1, :] - fx[:, :, 0, :]
        fxy[:, :, -1, :] = fx[:, :, -1, :] - fx[:, :, -2, :]

        gx = query_grid[..., 0].float()
        gy = query_grid[..., 1].float()
        cx = (gx + 1.0) * (float(width) * 0.5) - 0.5
        cy = (gy + 1.0) * (float(height) * 0.5) - 0.5
        x0 = torch.floor(cx).long().clamp(0, width - 2)
        y0 = torch.floor(cy).long().clamp(0, height - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        tx = cx - x0.float()
        ty = cy - y0.float()

        def basis(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
            t2 = t * t
            t3 = t2 * t
            return (
                2.0 * t3 - 3.0 * t2 + 1.0,
                t3 - 2.0 * t2 + t,
                -2.0 * t3 + 3.0 * t2,
                t3 - t2,
            )

        hx00, hx10, hx01, hx11 = [v.unsqueeze(1) for v in basis(tx)]
        hy00, hy10, hy01, hy11 = [v.unsqueeze(1) for v in basis(ty)]

        def gather(value: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
            return _gather_control(value, ix, iy)

        f00, f10 = gather(field, x0, y0), gather(field, x1, y0)
        f01, f11 = gather(field, x0, y1), gather(field, x1, y1)
        fx00, fx10 = gather(fx, x0, y0), gather(fx, x1, y0)
        fx01, fx11 = gather(fx, x0, y1), gather(fx, x1, y1)
        fy00, fy10 = gather(fy, x0, y0), gather(fy, x1, y0)
        fy01, fy11 = gather(fy, x0, y1), gather(fy, x1, y1)
        fxy00, fxy10 = gather(fxy, x0, y0), gather(fxy, x1, y0)
        fxy01, fxy11 = gather(fxy, x0, y1), gather(fxy, x1, y1)

        row0 = hx00 * f00 + hx10 * fx00 + hx01 * f10 + hx11 * fx10
        row1 = hx00 * f01 + hx10 * fx01 + hx01 * f11 + hx11 * fx11
        dy0 = hx00 * fy00 + hx10 * fxy00 + hx01 * fy10 + hx11 * fxy10
        dy1 = hx00 * fy01 + hx10 * fxy01 + hx01 * fy11 + hx11 * fxy11
        return hy00 * row0 + hy10 * dy0 + hy01 * row1 + hy11 * dy1

    # Purpose: Implement smooth min for LocalParametricBoundaryDecoder.
    # Called by: query
    # Calls: No same-class helper methods.
    @staticmethod
    def _smooth_min(values: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
        return -float(tau) * torch.logsumexp(-values / float(tau), dim=1, keepdim=True)

    # Purpose: Implement smooth max for LocalParametricBoundaryDecoder.
    # Called by: query
    # Calls: No same-class helper methods.
    @staticmethod
    def _smooth_max(values: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
        return float(tau) * torch.logsumexp(values / float(tau), dim=1, keepdim=True)

    # Purpose: Implement query for LocalParametricBoundaryDecoder.
    # Called by: forward
    # Calls: _central_difference, _gather_control, _sample, _smooth_max, _smooth_min
    # Purpose: Query the shared C1 topology-branch scalar fields.
    # Called by: forward.
    # Calls: _hermite_sample_scalar, _sample, _central_difference, _smooth_max, _smooth_min.
    def query(
        self,
        context: dict[str, torch.Tensor],
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        d_field = context["branch_anchor_distance_pixels"]
        half_width_field = context["branch_half_width_pixels"]
        ribbon_mode_field = context["branch_ribbon_mode"]
        activation_field = context["branch_activation"]
        csg_field = context["csg_logits"]
        confidence_field = context["confidence"]
        source = context["source_sdf_prior_lr"]

        b, _branches, _hc, _wc = d_field.shape
        hq, wq = query_grid.shape[1:3]

        # Geometry is one stitched C1 scalar surface per topology branch. No local
        # primitive owns an independent query-time tangent, curvature or zero-set.
        branch_center = self._hermite_sample_scalar(d_field, query_grid)
        branch_half_width = _sample(half_width_field, query_grid)
        branch_ribbon_mode = _sample(ribbon_mode_field, query_grid)
        branch_activation = _sample(activation_field, query_grid)
        csg_logits = _sample(csg_field, query_grid)
        confidence = _sample(confidence_field, query_grid)
        ribbon_surface = branch_center.abs() - branch_half_width
        branch_phi = (
            branch_center * (1.0 - branch_ribbon_mode)
            + ribbon_surface * branch_ribbon_mode
        )

        inactive_penalty = 16.0
        union_values = branch_phi.clone()
        union_values[:, 1:] = union_values[:, 1:] + (1.0 - branch_activation[:, 1:]) * inactive_penalty
        inter_values = branch_phi.clone()
        inter_values[:, 1:] = inter_values[:, 1:] - (1.0 - branch_activation[:, 1:]) * inactive_penalty
        single_phi = branch_phi[:, 0:1]
        union_phi = self._smooth_min(union_values)
        intersection_phi = self._smooth_max(inter_values)
        csg_weight = torch.softmax(csg_logits, dim=1)
        phi_param = (
            csg_weight[:, 0:1] * single_phi
            + csg_weight[:, 1:2] * union_phi
            + csg_weight[:, 2:3] * intersection_phi
        )

        sampled_source = _sample(source, query_grid) * self.max_distance_pixels
        # A 4x-quantised LR prior has roughly a two-pixel contour uncertainty band.
        # The learned C1 field owns that band; stable source signs outside it remain
        # a fail-closed topology envelope.
        zero_crossing_guard = 2.0
        zero_crossing_epsilon = 0.05
        phi_param = torch.where(
            sampled_source >= zero_crossing_guard,
            phi_param.clamp_min(zero_crossing_epsilon),
            phi_param,
        )
        phi_param = torch.where(
            sampled_source <= -zero_crossing_guard,
            phi_param.clamp_max(-zero_crossing_epsilon),
            phi_param,
        )
        inner_radius = 8.0
        outer_radius = 12.0
        u = ((outer_radius - sampled_source.abs()) / (outer_radius - inner_radius)).clamp(0.0, 1.0)
        authority = u * u * (3.0 - 2.0 * u)
        phi = sampled_source * (1.0 - authority) + phi_param * authority

        # Normals/curvature are telemetry derived from that exact sampled surface.
        gx_hr, gy_hr = _central_difference(branch_center)
        grad_norm = torch.sqrt(gx_hr.square() + gy_hr.square() + 1.0e-6)
        branch_nx, branch_ny = gx_hr / grad_norm, gy_hr / grad_norm
        nx_x, _ = _central_difference(branch_nx)
        _, ny_y = _central_difference(branch_ny)
        branch_k = nx_x + ny_y
        primary_normal = torch.cat((branch_nx[:, 0:1], branch_ny[:, 0:1]), dim=1)
        source_primary = _sample(context["distance_delta_pixels"], query_grid)
        return {
            "phi_pixels": phi,
            "primitive_phi_pixels": phi_param,
            "primitive_normal": primary_normal,
            "primitive_curvature": branch_k[:, 0:1],
            "primitive_confidence": confidence,
            "primitive_junction_hint": branch_activation[:, 1:].amax(dim=1, keepdim=True),
            "primitive_distance_delta_pixels": source_primary,
            "primitive_branch_half_width_pixels": branch_half_width,
            "primitive_branch_ribbon_mode": branch_ribbon_mode,
            "primitive_branch_activation": branch_activation,
            "primitive_csg_weights": csg_weight,
            "transport_pixels": torch.zeros((b, 2, hq, wq), device=phi.device, dtype=phi.dtype),
            "dilation_pixels": torch.zeros_like(phi),
            "residual_pixels": source_primary,
            "direct_delta_pixels": phi - sampled_source,
            "warped_source_pixels": sampled_source,
            "implicit_authority": authority,
        }

    # Purpose: Implement forward for LocalParametricBoundaryDecoder.
    # Called by: External callers and the owning workflow.
    # Calls: build_context, query
    def forward(
        self,
        feature_grid: torch.Tensor,
        source_sdf_normalized: torch.Tensor,
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.query(self.build_context(feature_grid, source_sdf_normalized), query_grid)
