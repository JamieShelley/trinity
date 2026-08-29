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
    """Predict three local primitives and a compact CSG composition.

    Each branch predicts an analytic centre surface plus a soft half-space /
    ribbon mode.  Ribbon mode represents the two sides of a thin authored line
    with one coherent primitive, so opposite SDF normals cannot fight each
    other.  Three ribbon branches combined by smooth union represent Y/T
    junctions without giving the network independent per-pixel SDF authority.
    """

    BRANCH_STRIDE = 6
    OUTPUTS = 24

    # Purpose: Implement init for PrimitiveParameterHead.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = max(24, int(hidden_channels))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.OUTPUTS, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Start as one ordinary boundary primitive.  The logits are kept
        # deliberately unsaturated so thin-line / junction evidence can turn
        # on ribbon branches early in training.
        with torch.no_grad():
            for branch in range(3):
                self.net[-1].bias[branch * self.BRANCH_STRIDE + 5] = -1.5  # ribbon mode
            self.net[-1].bias[18:20].fill_(-1.5)  # extra-branch activation
            self.net[-1].bias[20] = 1.5           # single
            self.net[-1].bias[21] = -1.5          # union
            self.net[-1].bias[22] = -1.5          # intersection

    # Purpose: Implement forward for PrimitiveParameterHead.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        smooth = F.avg_pool2d(raw, kernel_size=3, stride=1, padding=1)
        # CSG/mode/activation logits must retain their initialization at the
        # image boundary; smoothing a constant bias with zero padding would
        # alter it. Smooth only geometric residuals (d,nx,ny,kappa,width).
        parts = []
        for branch in range(3):
            start = branch * self.BRANCH_STRIDE
            geom = 0.45 * raw[:, start:start + 5] + 0.55 * smooth[:, start:start + 5]
            parts.append(torch.cat((geom, raw[:, start + 5:start + 6]), dim=1))
        tail = raw[:, 18:]
        geom = torch.cat(parts, dim=1)
        return torch.cat((geom, tail), dim=1)


class LocalParametricBoundaryDecoder(nn.Module):
    """Continuous metric SDF assembled from local analytic line/arc primitives."""

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
    ) -> dict[str, torch.Tensor]:
        source = source_sdf_normalized.float().clamp(-1.0, 1.0)
        source_pixels = source * self.max_distance_pixels
        # The source prior initializes geometry but is not the learned contour.
        # Mild local fitting removes quantized plateaus before parameterization.
        source_fit = 0.40 * source_pixels + 0.60 * F.avg_pool2d(
            source_pixels, kernel_size=3, stride=1, padding=1
        )
        gx, gy = _central_difference(source_fit)
        grad = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        base_nx, base_ny = gx / grad, gy / grad
        nx_x, _ = _central_difference(base_nx)
        _, ny_y = _central_difference(base_ny)
        base_curvature = (nx_x + ny_y) / float(self.output_scale)

        head_input = torch.cat((
            feature_grid.float(),
            (source_fit / max(self.max_distance_pixels, 1.0e-6)).clamp(-1.0, 1.0),
            base_nx,
            base_ny,
            (base_curvature / max(self.max_curvature_per_pixel, 1.0e-6)).clamp(-1.0, 1.0),
        ), dim=1)
        raw = self.parameter_head(head_input)

        branch_data = [
            self._branch_parameters(raw, base_nx, base_ny, base_curvature, source_fit, i)
            for i in range(3)
        ]
        branch_distance = torch.cat([v[0] for v in branch_data], dim=1)
        branch_normal_x = torch.cat([v[1] for v in branch_data], dim=1)
        branch_normal_y = torch.cat([v[2] for v in branch_data], dim=1)
        branch_curvature = torch.cat([v[3] for v in branch_data], dim=1)
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
    # Calls: _smooth_max, _smooth_min
    def query(
        self,
        context: dict[str, torch.Tensor],
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        d_field = context["branch_anchor_distance_pixels"]
        nx_field = context["branch_normal_x"]
        ny_field = context["branch_normal_y"]
        k_field = context["branch_curvature_per_pixel"]
        half_width_field = context["branch_half_width_pixels"]
        ribbon_mode_field = context["branch_ribbon_mode"]
        activation_field = context["branch_activation"]
        csg_field = context["csg_logits"]
        confidence_field = context["confidence"]
        source = context["source_sdf_prior_lr"]

        b, branches, hc, wc = d_field.shape
        hq, wq = query_grid.shape[1:3]
        gx = query_grid[..., 0].float()
        gy = query_grid[..., 1].float()
        cx = (gx + 1.0) * (float(wc) * 0.5) - 0.5
        cy = (gy + 1.0) * (float(hc) * 0.5) - 0.5
        qx = (gx + 1.0) * (float(wq) * 0.5) - 0.5
        qy = (gy + 1.0) * (float(hq) * 0.5) - 0.5
        x0, y0 = torch.floor(cx).long(), torch.floor(cy).long()
        wx, wy = (cx - x0.float()).clamp(0.0, 1.0), (cy - y0.float()).clamp(0.0, 1.0)
        sx, sy = float(wq) / float(wc), float(hq) / float(hc)

        branch_phi = torch.zeros((b, branches, hq, wq), device=query_grid.device)
        branch_nx = torch.zeros_like(branch_phi)
        branch_ny = torch.zeros_like(branch_phi)
        branch_k = torch.zeros_like(branch_phi)
        branch_half_width = torch.zeros_like(branch_phi)
        branch_ribbon_mode = torch.zeros_like(branch_phi)
        branch_activation = torch.zeros_like(branch_phi)
        csg_logits = torch.zeros((b, 3, hq, wq), device=query_grid.device)
        confidence = torch.zeros((b, 1, hq, wq), device=query_grid.device)

        for ox, oy, weight in (
            (0, 0, (1.0 - wx) * (1.0 - wy)),
            (1, 0, wx * (1.0 - wy)),
            (0, 1, (1.0 - wx) * wy),
            (1, 1, wx * wy),
        ):
            ix, iy = x0 + ox, y0 + oy
            d = _gather_control(d_field, ix, iy)
            nx = _gather_control(nx_field, ix, iy)
            ny = _gather_control(ny_field, ix, iy)
            kappa = _gather_control(k_field, ix, iy)
            half_width = _gather_control(half_width_field, ix, iy)
            ribbon_mode = _gather_control(ribbon_mode_field, ix, iy)
            act = _gather_control(activation_field, ix, iy)
            ops = _gather_control(csg_field, ix, iy)
            conf = _gather_control(confidence_field, ix, iy)
            anchor_x = (ix.float() + 0.5) * sx - 0.5
            anchor_y = (iy.float() + 0.5) * sy - 0.5
            dx = (qx - anchor_x).unsqueeze(1)
            dy = (qy - anchor_y).unsqueeze(1)
            tangent = -ny * dx + nx * dy
            centre_surface = d + nx * dx + ny * dy + 0.5 * kappa * tangent.square()
            ribbon_surface = centre_surface.abs() - half_width
            primitive = centre_surface * (1.0 - ribbon_mode) + ribbon_surface * ribbon_mode
            w = weight.unsqueeze(1)
            branch_phi = branch_phi + w * primitive
            branch_nx = branch_nx + w * nx
            branch_ny = branch_ny + w * ny
            branch_k = branch_k + w * kappa
            branch_half_width = branch_half_width + w * half_width
            branch_ribbon_mode = branch_ribbon_mode + w * ribbon_mode
            branch_activation = branch_activation + w * act
            csg_logits = csg_logits + w * ops
            confidence = confidence + w * conf

        nrm = torch.sqrt(branch_nx.square() + branch_ny.square() + 1.0e-6)
        branch_nx, branch_ny = branch_nx / nrm, branch_ny / nrm

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
        # Local primitives own the entire renderer-relevant contour tube. The
        # source prior is only a far-field sign/distance fallback; partially
        # mixing quantized LR distance inside the proof band would reintroduce
        # the very staircase the parametric representation removes.
        inner_radius = 8.0
        outer_radius = 12.0
        u = ((outer_radius - sampled_source.abs()) / (outer_radius - inner_radius)).clamp(0.0, 1.0)
        authority = u * u * (3.0 - 2.0 * u)
        phi = sampled_source * (1.0 - authority) + phi_param * authority

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
