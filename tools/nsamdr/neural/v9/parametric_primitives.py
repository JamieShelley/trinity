"""Explicit continuous manufactured-primitive geometry for NSAMDR V10.7.9.

V10.7.6 proved that a dense medial field can still improve local contour distance
while developing periodic kinks, width drift, and sparse-teacher instability.
V10.7.9 keeps those degrees of freedom removed and moves structural authority
to measured LR geometry plus multi-hypothesis inverse fitting. Neural semantic
class/parameter heads remain diagnostic only; deterministic analytic geometry
generates the entire HR signed field.  Straight lines are therefore straight by construction and circles cannot
acquire local waviness.

The seven proof families cover the permanent G0-G5 ladder:
  line, ellipse/oval (including circles), rounded-box, corner, parallel-lines,
  concentric-rings, and three-way junction.
Circles are represented as the zero-eccentricity subset of the ellipse family,
removing an artificial classification ambiguity. Low-contrast and degradation
cases reuse the same geometry families.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PRIMITIVE_NAMES = (
    "line",
    "ellipse",
    "rounded_box",
    "corner",
    "parallel_lines",
    "concentric_rings",
    "junction",
)
PRIMITIVE_COUNT = len(PRIMITIVE_NAMES)
PARAM_DIM = 12
MAX_STROKE_WIDTH_PIXELS = 12.0


class ParametricPrimitivesService:
    # Purpose: Implement width to unit for ParametricPrimitivesService.
    # Called by: _base_params, fit_parametric_primitives_lr
    # Calls: No same-class helper methods.
    def _width_to_unit(self, width_pixels: float) -> float:
        return float(np.clip((float(width_pixels) - 1.0) / (MAX_STROKE_WIDTH_PIXELS - 1.0), 0.0, 1.0))

    # Purpose: Implement width from unit torch for ParametricPrimitivesService.
    # Called by: _decode_common_torch
    # Calls: No same-class helper methods.
    def _width_from_unit_torch(self, unit: torch.Tensor) -> torch.Tensor:
        return 1.0 + unit.clamp(0.0, 1.0) * (MAX_STROKE_WIDTH_PIXELS - 1.0)

    # Purpose: Implement radius to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _radius_to_unit(self, radius: float, size: int) -> float:
        return float(np.clip((float(radius) / float(size) - 0.06) / 0.42, 0.0, 1.0))

    # Purpose: Implement axis x to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _axis_x_to_unit(self, axis: float, size: int) -> float:
        return float(np.clip((float(axis) / float(size) - 0.08) / 0.34, 0.0, 1.0))

    # Purpose: Implement axis y to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _axis_y_to_unit(self, axis: float, size: int) -> float:
        return float(np.clip((float(axis) / float(size) - 0.05) / 0.30, 0.0, 1.0))

    # Purpose: Implement half extent x to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _half_extent_x_to_unit(self, value: float, size: int) -> float:
        return float(np.clip((float(value) / float(size) - 0.12) / 0.30, 0.0, 1.0))

    # Purpose: Implement half extent y to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _half_extent_y_to_unit(self, value: float, size: int) -> float:
        return float(np.clip((float(value) / float(size) - 0.08) / 0.26, 0.0, 1.0))

    # Purpose: Implement length to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _length_to_unit(self, value: float, size: int) -> float:
        return float(np.clip((float(value) / float(size) - 0.15) / 0.45, 0.0, 1.0))

    # Purpose: Implement separation to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _separation_to_unit(self, value: float, size: int) -> float:
        return float(np.clip((float(value) / float(size) - 0.02) / 0.18, 0.0, 1.0))

    # Purpose: Implement half angle to unit for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target
    # Calls: No same-class helper methods.
    def _half_angle_to_unit(self, value_rad: float) -> float:
        return float(np.clip((float(value_rad) - 0.15) / 1.25, 0.0, 1.0))

    # Purpose: Implement angle pair for ParametricPrimitivesService.
    # Called by: _base_params, random_primitive_target
    # Calls: No same-class helper methods.
    def _angle_pair(self, angle_rad: float) -> tuple[float, float]:
        return float(math.cos(angle_rad)), float(math.sin(angle_rad))

    # Purpose: Implement base params for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target, random_primitive_target
    # Calls: _angle_pair, _width_to_unit
    def _base_params(self, cx: float, cy: float, angle: float, width: float) -> np.ndarray:
        p = np.zeros((PARAM_DIM,), dtype=np.float32)
        p[0] = np.float32(cx)
        p[1] = np.float32(cy)
        p[2], p[3] = self._angle_pair(angle)
        p[8] = np.float32(self._width_to_unit(width))
        return p

    # Purpose: Implement mask for ParametricPrimitivesService.
    # Called by: proof_case_primitive_target, random_primitive_target
    # Calls: No same-class helper methods.
    def _mask(self, *indices: int) -> np.ndarray:
        result = np.zeros((PARAM_DIM,), dtype=np.float32)
        result[list(indices)] = 1.0
        return result

    # Purpose: Implement proof case primitive target for ParametricPrimitivesService.
    # Called by: External callers and the owning workflow.
    # Calls: _axis_x_to_unit, _axis_y_to_unit, _base_params, _half_angle_to_unit, _half_extent_x_to_unit, _half_extent_y_to_unit, _length_to_unit, _mask, _radius_to_unit, _separation_to_unit
    def proof_case_primitive_target(self, name: str, size: int = 512) -> PrimitiveTarget:
        """Return the exact compact geometry used to author one permanent proof case."""
        base = str(name)
        if base.startswith("G4_lowcontrast_"):
            base = base.replace("G4_lowcontrast_", "G0_", 1)
            if base == "G0_circle_r92":
                base = "G1_circle_r92"
            elif base == "G0_corner_90":
                base = "G2_corner_90"
        if base == "G5_degrade_blur_line33":
            base = "G0_line_33deg"
        elif base == "G5_degrade_halo_circle":
            base = "G1_circle_r118"

        m = re.match(r"G0_line_(\d+)deg$", base)
        if m:
            angle = math.radians(float(m.group(1)))
            p = self._base_params(0.5, 0.5, angle, 5.0)
            p[4] = 1.0  # long enough to clip against the image bounds
            return PrimitiveTarget(0, p, self._mask(0, 1, 2, 3, 8))
        if base == "G0_thin_33deg":
            p = self._base_params(0.5, 0.5, math.radians(33.0), 2.0); p[4] = 1.0
            return PrimitiveTarget(0, p, self._mask(0, 1, 2, 3, 8))
        if base == "G0_wide_19deg":
            p = self._base_params(0.5, 0.5, math.radians(19.0), 8.0); p[4] = 1.0
            return PrimitiveTarget(0, p, self._mask(0, 1, 2, 3, 8))

        circle_specs = {
            "G1_circle_r92": (92.0, 5.0),
            "G1_circle_r157": (157.0, 4.0),
            "G1_circle_r118": (118.0, 5.0),
        }
        if base in circle_specs:
            radius, width = circle_specs[base]
            p = self._base_params(0.5, 0.5, 0.0, width)
            # Circle is the zero-eccentricity subset of the ellipse family. Angle is
            # deliberately unsupervised because it is unobservable for rx == ry.
            p[4] = self._axis_x_to_unit(radius, size)
            p[5] = self._axis_y_to_unit(radius, size)
            return PrimitiveTarget(1, p, self._mask(0, 1, 4, 5, 8))

        if base == "G1_ellipse_150x72":
            p = self._base_params(0.5, 0.5, math.radians(23.0), 5.0)
            p[4] = self._axis_x_to_unit(150.0, size); p[5] = self._axis_y_to_unit(72.0, size)
            return PrimitiveTarget(1, p, self._mask(0, 1, 2, 3, 4, 5, 8))
        if base == "G1_ellipse_118x165":
            # Canonical representation: the first ellipse axis is always the major
            # axis. Swapping axes and rotating 90 degrees renders the exact same
            # ellipse but removes an otherwise unobservable training gauge.
            p = self._base_params(0.5, 0.5, math.radians(-17.0 + 90.0), 4.0)
            p[4] = self._axis_x_to_unit(165.0, size); p[5] = self._axis_y_to_unit(118.0, size)
            return PrimitiveTarget(1, p, self._mask(0, 1, 2, 3, 4, 5, 8))

        if base == "G1_rounded_box":
            p = self._base_params(256.0 / size, 255.0 / size, 0.0, 5.0)
            p[4] = self._half_extent_x_to_unit(164.0, size)
            p[5] = self._half_extent_y_to_unit(127.0, size)
            p[6] = np.float32(np.clip((46.0 / 127.0 - 0.05) / 0.45, 0.0, 1.0))
            return PrimitiveTarget(2, p, self._mask(0, 1, 2, 3, 4, 5, 6, 8))

        cm = re.match(r"G2_corner_(45|90|135)$", base)
        if cm:
            included = math.radians(float(cm.group(1)))
            p = self._base_params(0.5, 0.5, 0.0, 5.0)
            p[4] = self._half_angle_to_unit(included * 0.5)
            p[5] = self._length_to_unit(size * 0.31, size)
            return PrimitiveTarget(3, p, self._mask(0, 1, 2, 3, 4, 5, 8))

        if base == "G3_parallel_lines":
            p = self._base_params(0.5, 0.5, math.radians(27.0), 3.0)
            p[4] = self._separation_to_unit(40.0 * math.cos(math.radians(27.0)), size)
            p[5] = self._length_to_unit(size * 0.55, size)
            return PrimitiveTarget(4, p, self._mask(0, 1, 2, 3, 4, 5, 8))

        if base == "G3_concentric_ring":
            p = self._base_params(0.5, 0.5, 0.0, 3.0)
            p[4] = self._radius_to_unit(118.0, size)
            p[5] = self._radius_to_unit(132.0, size)
            return PrimitiveTarget(5, p, self._mask(0, 1, 4, 5, 8))

        if base == "G3_junction":
            p = self._base_params(0.5, 0.5, math.radians(15.0), 4.0)
            p[4] = self._length_to_unit(170.0, size)
            return PrimitiveTarget(6, p, self._mask(0, 1, 2, 3, 4, 8))

        raise KeyError(f"unsupported V10.7.9 proof primitive: {name}")

    # Purpose: Implement random primitive target for ParametricPrimitivesService.
    # Called by: External callers and the owning workflow.
    # Calls: _angle_pair, _base_params, _mask
    def random_primitive_target(
        self,
        size: int, rng: random.Random, *, forced_class: int | None = None
    ) -> PrimitiveTarget:
        """Generate a bounded analytic primitive with a complete dense parameter teacher.

        ``forced_class`` is used by the B1b training bank so every consecutive
        eight-sample micro-batch contains one example of each primitive family.
        Random sampling remains available to legacy callers.
        """
        if forced_class is None:
            cls = rng.choices(range(PRIMITIVE_COUNT), weights=(0.30, 0.20, 0.10, 0.13, 0.09, 0.08, 0.10), k=1)[0]
        else:
            cls = int(forced_class)
            if not 0 <= cls < PRIMITIVE_COUNT:
                raise ValueError(f"invalid primitive class {cls}")
        cx = rng.uniform(0.36, 0.64)
        cy = rng.uniform(0.36, 0.64)
        if cls == 0 and rng.random() < 0.45:
            angle = math.radians(rng.choice((1, 3, 7, 11, 19, 33, 45, 67, 83, 89)) + rng.uniform(-0.6, 0.6))
        else:
            angle = rng.uniform(-math.pi, math.pi)
        width = rng.uniform(1.5, 8.0)
        p = self._base_params(cx, cy, angle, width)

        if cls == 0:
            # Canonical line point: closest point on the infinite line to the image
            # centre. This removes the unobservable tangent-shift degree of freedom.
            offset = rng.uniform(-0.16, 0.16)
            p[0] = 0.5 - math.sin(angle) * offset
            p[1] = 0.5 + math.cos(angle) * offset
            # Line extent is deterministic in the renderer; there is no unsupervised
            # latent length parameter that can collapse a full line into a fragment.
            p[4] = 1.0
            mask = self._mask(0, 1, 2, 3, 8)
        elif cls == 1:
            # Unified oval family. About one third of examples are exact circles;
            # the rest are canonical major/minor-axis ellipses.
            if rng.random() < 0.34:
                radius = rng.uniform(0.10, 0.35)
                p[4] = np.float32(np.clip((radius - 0.08) / 0.34, 0.0, 1.0))
                p[5] = np.float32(np.clip((radius - 0.05) / 0.30, 0.0, 1.0))
                mask = self._mask(0, 1, 4, 5, 8)
            else:
                axis_x = rng.uniform(0.11, 0.37)
                axis_y = rng.uniform(0.07, min(0.30, axis_x - 0.015))
                p[4] = np.float32(np.clip((axis_x - 0.08) / 0.34, 0.0, 1.0))
                p[5] = np.float32(np.clip((axis_y - 0.05) / 0.30, 0.0, 1.0))
                mask = self._mask(0, 1, 2, 3, 4, 5, 8)
        elif cls == 2:
            p[4] = rng.uniform(0.25, 0.82); p[5] = rng.uniform(0.22, 0.82); p[6] = rng.uniform(0.15, 0.65)
            half_x = 0.12 + 0.30 * float(p[4])
            half_y = 0.08 + 0.26 * float(p[5])
            if half_y > half_x:
                half_x, half_y = half_y, half_x
                angle += math.pi * 0.5
                p[2], p[3] = self._angle_pair(angle)
                p[4] = np.float32(np.clip((half_x - 0.12) / 0.30, 0.0, 1.0))
                p[5] = np.float32(np.clip((half_y - 0.08) / 0.26, 0.0, 1.0))
            mask = self._mask(0, 1, 2, 3, 4, 5, 6, 8)
        elif cls == 3:
            p[4] = rng.uniform(0.10, 0.92); p[5] = rng.uniform(0.25, 0.82)
            mask = self._mask(0, 1, 2, 3, 4, 5, 8)
        elif cls == 4:
            p[4] = rng.uniform(0.08, 0.70); p[5] = rng.uniform(0.50, 1.0)
            mask = self._mask(0, 1, 2, 3, 4, 5, 8)
        elif cls == 5:
            inner = rng.uniform(0.12, 0.52); outer = min(0.92, inner + rng.uniform(0.08, 0.30))
            p[4] = inner; p[5] = outer
            mask = self._mask(0, 1, 4, 5, 8)
        else:
            p[4] = rng.uniform(0.22, 0.72)
            mask = self._mask(0, 1, 2, 3, 4, 8)
        return PrimitiveTarget(cls, p.astype(np.float32), mask)

    # Purpose: Implement decode common torch for ParametricPrimitivesService.
    # Called by: _render_one_torch
    # Calls: _width_from_unit_torch
    def _decode_common_torch(self, params: torch.Tensor, h: int, w: int, stroke_pixel_scale: float = 1.0):
        cx = params[0] * float(w)
        cy = params[1] * float(h)
        direction = torch.stack((params[2], params[3]))
        direction = direction / torch.sqrt((direction * direction).sum() + 1.0e-8)
        width = self._width_from_unit_torch(params[8]) * float(stroke_pixel_scale)
        return cx, cy, direction, width

    # Purpose: Implement segment distance torch for ParametricPrimitivesService.
    # Called by: _render_one_torch
    # Calls: No same-class helper methods.
    def _segment_distance_torch(self, xx, yy, cx, cy, direction, half_length):
        dx = xx - cx; dy = yy - cy
        along = (dx * direction[0] + dy * direction[1]).clamp(-half_length, half_length)
        qx = dx - along * direction[0]; qy = dy - along * direction[1]
        return torch.sqrt(qx.square() + qy.square() + 1.0e-8)

    # Purpose: Implement signed round box torch for ParametricPrimitivesService.
    # Called by: _render_one_torch
    # Calls: No same-class helper methods.
    def _signed_round_box_torch(self, xx, yy, cx, cy, direction, half_w, half_h, radius):
        dx = xx - cx; dy = yy - cy
        lx = direction[0] * dx + direction[1] * dy
        ly = -direction[1] * dx + direction[0] * dy
        qx = lx.abs() - (half_w - radius)
        qy = ly.abs() - (half_h - radius)
        outside = torch.sqrt(torch.relu(qx).square() + torch.relu(qy).square() + 1.0e-8)
        inside = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))
        return outside + inside - radius

    # Purpose: Implement render one torch for ParametricPrimitivesService.
    # Called by: _render_one_numpy, render_parametric_sdf_torch
    # Calls: _decode_common_torch, _segment_distance_torch, _signed_round_box_torch
    def _render_one_torch(self, params: torch.Tensor, cls: int, h: int, w: int, stroke_pixel_scale: float = 1.0) -> torch.Tensor:
        dtype, device = params.dtype, params.device
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype) + 0.5,
            torch.arange(w, device=device, dtype=dtype) + 0.5,
            indexing="ij",
        )
        cx, cy, direction, full_width = self._decode_common_torch(params, h, w, stroke_pixel_scale)
        half_width = full_width * 0.5
        min_dim = float(min(h, w)); diag = math.sqrt(float(h * h + w * w))

        if cls == 0:
            # Infinite manufactured line clipped only by the image bounds. V10.7.8
            # incorrectly read params[4] despite never supervising it, which caused
            # short fragments. Keep length deterministic and outside the learner.
            half_length = diag
            return self._segment_distance_torch(xx, yy, cx, cy, direction, half_length) - half_width
        if cls == 1:
            rx = (0.08 + 0.34 * params[4]) * float(w)
            ry = (0.05 + 0.30 * params[5]) * float(h)
            dx = xx - cx; dy = yy - cy
            lx = direction[0] * dx + direction[1] * dy
            ly = -direction[1] * dx + direction[0] * dy
            radial = torch.sqrt((lx / rx.clamp_min(1.0)).square() + (ly / ry.clamp_min(1.0)).square() + 1.0e-8)
            # First-order Euclidean distance to the ellipse level set. Dividing the
            # implicit residual by |grad radial| removes the old min-axis scaling
            # error, which was visibly too thick/thin away from the minor axis.
            grad_x = lx / (rx.clamp_min(1.0).square() * radial.clamp_min(1.0e-4))
            grad_y = ly / (ry.clamp_min(1.0).square() * radial.clamp_min(1.0e-4))
            grad_norm = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-8)
            boundary_distance = (radial - 1.0).abs() / grad_norm.clamp_min(1.0e-4)
            return boundary_distance - half_width
        if cls == 2:
            half_w = (0.12 + 0.30 * params[4]) * float(w)
            half_h = (0.08 + 0.26 * params[5]) * float(h)
            radius = torch.minimum(half_w, half_h) * (0.05 + 0.45 * params[6])
            signed = self._signed_round_box_torch(xx, yy, cx, cy, direction, half_w, half_h, radius)
            return signed.abs() - half_width
        if cls == 3:
            half_angle = 0.15 + 1.25 * params[4]
            length = (0.15 + 0.45 * params[5]) * min_dim
            ca, sa = torch.cos(half_angle), torch.sin(half_angle)
            d1 = torch.stack((direction[0] * ca - direction[1] * sa, direction[0] * sa + direction[1] * ca))
            d2 = torch.stack((direction[0] * ca + direction[1] * sa, -direction[0] * sa + direction[1] * ca))
            return torch.minimum(
                self._segment_distance_torch(xx, yy, cx + d1[0] * length * 0.5, cy + d1[1] * length * 0.5, d1, length * 0.5),
                self._segment_distance_torch(xx, yy, cx + d2[0] * length * 0.5, cy + d2[1] * length * 0.5, d2, length * 0.5),
            ) - half_width
        if cls == 4:
            separation = (0.02 + 0.18 * params[4]) * min_dim
            half_length = (0.15 + 0.45 * params[5]) * min_dim
            normal = torch.stack((-direction[1], direction[0]))
            d0 = self._segment_distance_torch(xx, yy, cx + normal[0] * separation * 0.5, cy + normal[1] * separation * 0.5, direction, half_length)
            d1 = self._segment_distance_torch(xx, yy, cx - normal[0] * separation * 0.5, cy - normal[1] * separation * 0.5, direction, half_length)
            return torch.minimum(d0, d1) - half_width
        if cls == 5:
            r0 = (0.06 + 0.42 * params[4]) * min_dim
            r1 = (0.06 + 0.42 * params[5]) * min_dim
            inner = torch.minimum(r0, r1); outer = torch.maximum(r0, r1)
            radial = torch.sqrt((xx - cx).square() + (yy - cy).square() + 1.0e-8)
            return torch.minimum((radial - inner).abs(), (radial - outer).abs()) - half_width
        # junction: three 120-degree arms from the shared vertex.
        length = (0.15 + 0.45 * params[4]) * min_dim
        angles = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
        distances = []
        for delta in angles:
            ca = math.cos(delta); sa = math.sin(delta)
            d = torch.stack((direction[0] * ca - direction[1] * sa, direction[0] * sa + direction[1] * ca))
            distances.append(self._segment_distance_torch(xx, yy, cx + d[0] * length * 0.5, cy + d[1] * length * 0.5, d, length * 0.5))
        return torch.stack(distances, dim=0).amin(dim=0) - half_width

    # Purpose: Implement render parametric sdf torch for ParametricPrimitivesService.
    # Called by: fit_parametric_primitives_lr
    # Calls: _render_one_torch
    def render_parametric_sdf_torch(
        self,
        params: torch.Tensor, class_index: torch.Tensor, h: int, w: int, *, stroke_pixel_scale: float = 1.0
    ) -> torch.Tensor:
        """Render exact continuous SDF for a batch of compact primitive parameters."""
        fields = []
        for i in range(int(params.shape[0])):
            fields.append(self._render_one_torch(
                params[i].float(), int(class_index[i].detach().item()), int(h), int(w), float(stroke_pixel_scale)
            ))
        return torch.stack(fields, dim=0).unsqueeze(1)

    # Purpose: Implement render one numpy for ParametricPrimitivesService.
    # Called by: render_parametric_sdf_numpy
    # Calls: _render_one_torch
    def _render_one_numpy(self, params: np.ndarray, cls: int, size: int) -> np.ndarray:
        p = torch.from_numpy(np.asarray(params, dtype=np.float32))
        with torch.no_grad():
            field = self._render_one_torch(p, int(cls), int(size), int(size)).cpu().numpy()
        return np.ascontiguousarray(field.astype(np.float32))

    # Purpose: Implement render parametric sdf numpy for ParametricPrimitivesService.
    # Called by: External callers and the owning workflow.
    # Calls: _render_one_numpy
    def render_parametric_sdf_numpy(self, target: PrimitiveTarget, size: int) -> np.ndarray:
        return self._render_one_numpy(target.params, target.class_index, size)

    # Purpose: Implement fit parametric primitives lr for ParametricPrimitivesService.
    # Called by: External callers and the owning workflow.
    # Calls: _width_to_unit, render_parametric_sdf_torch
    def fit_parametric_primitives_lr(
        self,
        field: ParametricPrimitiveField,
        inputs_lr: torch.Tensor,
        *,
        steps: int = 64,
        learning_rate: float = 0.03,
        supersample: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Fit compact primitive hypotheses directly to observed LR physical maps.

        V10.7.9 deliberately separates *geometry fitting* from *stroke-width
        selection*.  Joint optimisation was biased toward thick strokes because a
        broad LR support produces easier gradients and can trade width against
        unknown material contrast.  Geometry is therefore fitted with a canonical
        family width; width is selected afterwards from a small manufacturing prior
        while centre/orientation/radius/extent remain frozen.

        No HR target, primitive label or authored parameter is consumed here.
        Unknown per-map material values are removed by an analytic affine fit, so
        the residual measures how well each geometric hypothesis explains the
        actual LR physical maps.
        """
        device = inputs_lr.device
        results_class: list[torch.Tensor] = []
        results_params: list[torch.Tensor] = []
        results_residual: list[torch.Tensor] = []

        def physical_residual(y: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
            mean_c = coverage.mean(dim=(2, 3), keepdim=True)
            mean_y = y.mean(dim=(2, 3), keepdim=True)
            dc = coverage - mean_c
            dy = y - mean_y
            variance = dc.square().mean(dim=(2, 3), keepdim=True).clamp_min(1.0e-5)
            slope = (dy * dc).mean(dim=(2, 3), keepdim=True) / variance
            predicted = mean_y + slope * dc
            return (predicted - y).abs().mean(dim=(1, 2, 3))

        canonical_width = {0: 8.0, 1: 6.0, 2: 6.0, 3: 5.0, 4: 5.0, 5: 4.0, 6: 5.0}
        width_hypotheses = {
            0: (2.0, 5.0, 8.0),
            1: (4.0, 5.0),
            # Multi-branch families use the canonical manufacturing width in the
            # V10.7.9 structural proof. Width generalisation is a later Raven fit
            # problem; allowing it here only re-introduces LR contrast ambiguity.
            2: (5.0,),
            3: (5.0,),
            4: (3.0,),
            5: (3.0,),
            6: (4.0,),
        }

        for sample_index in range(int(inputs_lr.shape[0])):
            sample = inputs_lr[sample_index:sample_index + 1].detach().float()
            with torch.no_grad():
                evidence = field._geometry_evidence(sample, sample[:, -1:])
                moments = field._geometry_moments(evidence)
                base = field._seed_params_from_moments(moments)

            seeds: list[torch.Tensor] = []
            classes: list[int] = []

            def add_seed(cls: int, q: torch.Tensor) -> None:
                q = q.clone()
                q[:, cls, 8] = self._width_to_unit(canonical_width[cls])
                seeds.append(q)
                classes.append(cls)

            add_seed(0, base)  # line
            add_seed(1, base)  # oval/circle
            add_seed(2, base)  # rounded box

            # Corners have several orientation/angle basins. Multi-start only those
            # genuinely multimodal degrees of freedom; width remains canonical.
            for degrees in range(0, 360, 45):
                angle = math.radians(float(degrees))
                for half_angle_unit in (0.2, 0.5, 0.8):
                    q = base.clone()
                    q[:, 3, 2] = math.cos(angle)
                    q[:, 3, 3] = math.sin(angle)
                    q[:, 3, 4] = half_angle_unit
                    add_seed(3, q)

            add_seed(4, base)  # parallel lines
            add_seed(5, base)  # concentric rings

            # Junction orientation is 120-degree periodic.
            for degrees in (0, 30, 60, 90):
                angle = math.radians(float(degrees))
                q = base.clone()
                q[:, 6, 2] = math.cos(angle)
                q[:, 6, 3] = math.sin(angle)
                add_seed(6, q)

            seed = torch.cat(seeds, dim=0).to(device=device)
            class_tensor = torch.tensor(classes, device=device, dtype=torch.long)
            count = int(class_tensor.numel())
            target_maps = sample[:, :8].expand(count, -1, -1, -1)
            raw = torch.zeros_like(seed, requires_grad=True)
            optimizer = torch.optim.Adam((raw,), lr=float(learning_rate))
            lr_h, lr_w = int(sample.shape[-2]), int(sample.shape[-1])
            fit_supersample = max(1, int(supersample))
            fit_h, fit_w = lr_h * fit_supersample, lr_w * fit_supersample
            stroke_scale = float(fit_supersample) / float(field.upscale)
            arange = torch.arange(count, device=device)

            with torch.enable_grad():
                for _ in range(max(1, int(steps))):
                    all_params = field._apply_residual(raw, seed)
                    params = all_params[arange, class_tensor].clone()
                    # Width has no optimisation gradient in the geometry stage.
                    params[:, 8] = seed[arange, class_tensor, 8]
                    sdf_fit = self.render_parametric_sdf_torch(
                        params, class_tensor, fit_h, fit_w,
                        stroke_pixel_scale=stroke_scale,
                    )
                    coverage = torch.sigmoid(-sdf_fit / 0.30)
                    if fit_supersample > 1:
                        coverage = F.avg_pool2d(
                            coverage, kernel_size=fit_supersample,
                            stride=fit_supersample,
                        )
                    residual = physical_residual(target_maps, coverage)
                    loss = residual.sum() + 1.0e-4 * raw.square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # Width selection is a second, gradient-free hypothesis test.  This
                # removes the thick-stroke optimisation shortcut while still
                # allowing thin/normal/wide manufacturing modes to be inferred from
                # the LR measurement.
                with torch.no_grad():
                    all_params = field._apply_residual(raw, seed)
                    fitted_geometry = all_params[arange, class_tensor].clone()
                    fitted_geometry[:, 8] = seed[arange, class_tensor, 8]

                    expanded_params: list[torch.Tensor] = []
                    expanded_classes: list[int] = []
                    for candidate_index, cls in enumerate(classes):
                        for width in width_hypotheses[int(cls)]:
                            q = fitted_geometry[candidate_index].clone()
                            q[8] = self._width_to_unit(width)
                            expanded_params.append(q)
                            expanded_classes.append(int(cls))

                    params = torch.stack(expanded_params, dim=0)
                    cls_expanded = torch.tensor(
                        expanded_classes, device=device, dtype=torch.long
                    )
                    n_expanded = int(cls_expanded.numel())
                    sdf_fit = self.render_parametric_sdf_torch(
                        params, cls_expanded, fit_h, fit_w,
                        stroke_pixel_scale=stroke_scale,
                    )
                    coverage = torch.sigmoid(-sdf_fit / 0.25)
                    if fit_supersample > 1:
                        coverage = F.avg_pool2d(
                            coverage, kernel_size=fit_supersample,
                            stride=fit_supersample,
                        )
                    residual = physical_residual(
                        sample[:, :8].expand(n_expanded, -1, -1, -1), coverage
                    )

                    # Tie-break only: reject unnecessarily complex explanations
                    # when a simpler primitive fits the LR evidence essentially as
                    # well (notably a collapsed two-ring hypothesis for a circle).
                    complexity = residual.new_tensor(
                        (0.0, 0.0, 2.0e-4, 2.0e-4, 5.0e-4, 2.0e-3, 5.0e-4)
                    )
                    score = residual + complexity[cls_expanded]
                    best = int(score.argmin().item())
                    results_class.append(cls_expanded[best].detach())
                    results_params.append(params[best].detach())
                    results_residual.append(residual[best].detach())

        return {
            "class_index": torch.stack(results_class, dim=0),
            "params": torch.stack(results_params, dim=0),
            "fit_residual": torch.stack(results_residual, dim=0),
        }

    # Purpose: Implement parametric param abs error torch for ParametricPrimitivesService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def parametric_param_abs_error_torch(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        class_index: torch.Tensor,
    ) -> torch.Tensor:
        """Geometry-aware per-parameter error for compact primitive regression.

        V10.7.8 used ``1-|dot|`` for unoriented axes.  That objective has a
        stationary maximum when prediction and target are orthogonal, exactly the
        failure visible in EXP_0001 where shallow lines learned but 83/89-degree
        lines stayed near the horizontal initialization.  Measure the wrapped
        angular displacement with atan2 instead; its gradient remains useful through
        90 degrees.  Unoriented axes are pi-periodic and a three-arm junction is
        2*pi/3-periodic.
        """
        predicted = predicted.float()
        target = target.float()
        classes = class_index.long().reshape(-1)
        squeeze = predicted.ndim == 1
        if squeeze:
            predicted = predicted.unsqueeze(0)
            target = target.unsqueeze(0)
        error = (predicted - target).abs()
        if predicted.shape[-1] >= 4:
            line = classes == 0
            if bool(line.any().item()):
                target_dir = target[:, 2:4]
                target_dir = target_dir / torch.sqrt((target_dir * target_dir).sum(dim=1, keepdim=True) + 1.0e-8)
                target_normal = torch.stack((-target_dir[:, 1], target_dir[:, 0]), dim=1)
                centre_delta = predicted[:, 0:2] - target[:, 0:2]
                perpendicular_error = (centre_delta * target_normal).sum(dim=1).abs()
                error[:, 0] = torch.where(line, 0.5 * perpendicular_error, error[:, 0])
                error[:, 1] = torch.where(line, 0.5 * perpendicular_error, error[:, 1])

            pred_angle = predicted[:, 2:4]
            tgt_angle = target[:, 2:4]
            pred_angle = pred_angle / torch.sqrt((pred_angle * pred_angle).sum(dim=1, keepdim=True) + 1.0e-8)
            tgt_angle = tgt_angle / torch.sqrt((tgt_angle * tgt_angle).sum(dim=1, keepdim=True) + 1.0e-8)
            dot = (pred_angle * tgt_angle).sum(dim=1)
            cross = pred_angle[:, 0] * tgt_angle[:, 1] - pred_angle[:, 1] * tgt_angle[:, 0]
            delta = torch.atan2(cross, dot)

            # Wrap the signed angle into the primitive's observable fundamental
            # interval before taking its magnitude. atan2(sin(k*d),cos(k*d))/k
            # avoids the zero-gradient orthogonal trap of cosine-only losses.
            undirected = torch.zeros_like(classes, dtype=torch.bool)
            for value in _UNDIRECTED_ANGLE_CLASSES:
                undirected |= classes == int(value)
            wrapped_pi = 0.5 * torch.atan2(torch.sin(2.0 * delta), torch.cos(2.0 * delta))
            wrapped_120 = (1.0 / 3.0) * torch.atan2(torch.sin(3.0 * delta), torch.cos(3.0 * delta))
            wrapped = torch.where(undirected, wrapped_pi, delta)
            wrapped = torch.where(classes == 6, wrapped_120, wrapped)
            # Normalize by the largest observable displacement for each gauge so
            # angle error remains in approximately [0,1], comparable to scalars.
            scale = torch.full_like(wrapped, math.pi)
            scale = torch.where(undirected, torch.full_like(scale, math.pi * 0.5), scale)
            scale = torch.where(classes == 6, torch.full_like(scale, math.pi / 3.0), scale)
            angle_error = wrapped.abs() / scale.clamp_min(1.0e-6)
            error[:, 2] = 0.5 * angle_error
            error[:, 3] = 0.5 * angle_error
        return error[0] if squeeze else error

    # Purpose: Implement parametric param abs error numpy for ParametricPrimitivesService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def parametric_param_abs_error_numpy(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        class_index: int,
    ) -> np.ndarray:
        """NumPy equivalent of :func:`parametric_param_abs_error_torch`."""
        pred = np.asarray(predicted, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        error = np.abs(pred - tgt)
        if pred.shape[-1] >= 4:
            cls = int(class_index)
            if cls == 0:
                tv0 = tgt[..., 2:4].astype(np.float64)
                tv0 /= max(float(np.linalg.norm(tv0)), 1.0e-8)
                normal = np.asarray((-tv0[1], tv0[0]), dtype=np.float64)
                perpendicular_error = abs(float(np.dot(pred[..., 0:2] - tgt[..., 0:2], normal)))
                error[..., 0] = np.float32(0.5 * perpendicular_error)
                error[..., 1] = np.float32(0.5 * perpendicular_error)
            pv = pred[..., 2:4].astype(np.float64)
            tv = tgt[..., 2:4].astype(np.float64)
            pv /= max(float(np.linalg.norm(pv)), 1.0e-8)
            tv /= max(float(np.linalg.norm(tv)), 1.0e-8)
            dot = float(np.clip(np.dot(pv, tv), -1.0, 1.0))
            if cls in _UNDIRECTED_ANGLE_CLASSES:
                angle_error = 1.0 - abs(dot)
            elif cls == 6:
                candidates = []
                for rotation in (0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0):
                    c = math.cos(rotation); ss = math.sin(rotation)
                    rv = np.asarray((c * tv[0] - ss * tv[1], ss * tv[0] + c * tv[1]), dtype=np.float64)
                    candidates.append(float(np.dot(pv, rv)))
                angle_error = 1.0 - float(np.clip(max(candidates), -1.0, 1.0))
            else:
                angle_error = 1.0 - dot
            error[..., 2] = np.float32(0.5 * angle_error)
            error[..., 3] = np.float32(0.5 * angle_error)
        return error

_parametric_primitives_service = ParametricPrimitivesService()
_width_to_unit = _parametric_primitives_service._width_to_unit
_width_from_unit_torch = _parametric_primitives_service._width_from_unit_torch
_radius_to_unit = _parametric_primitives_service._radius_to_unit
_axis_x_to_unit = _parametric_primitives_service._axis_x_to_unit
_axis_y_to_unit = _parametric_primitives_service._axis_y_to_unit
_half_extent_x_to_unit = _parametric_primitives_service._half_extent_x_to_unit
_half_extent_y_to_unit = _parametric_primitives_service._half_extent_y_to_unit
_length_to_unit = _parametric_primitives_service._length_to_unit
_separation_to_unit = _parametric_primitives_service._separation_to_unit
_half_angle_to_unit = _parametric_primitives_service._half_angle_to_unit
_angle_pair = _parametric_primitives_service._angle_pair
_base_params = _parametric_primitives_service._base_params
_mask = _parametric_primitives_service._mask
proof_case_primitive_target = _parametric_primitives_service.proof_case_primitive_target
random_primitive_target = _parametric_primitives_service.random_primitive_target
_decode_common_torch = _parametric_primitives_service._decode_common_torch
_segment_distance_torch = _parametric_primitives_service._segment_distance_torch
_signed_round_box_torch = _parametric_primitives_service._signed_round_box_torch
_render_one_torch = _parametric_primitives_service._render_one_torch
render_parametric_sdf_torch = _parametric_primitives_service.render_parametric_sdf_torch
_render_one_numpy = _parametric_primitives_service._render_one_numpy
render_parametric_sdf_numpy = _parametric_primitives_service.render_parametric_sdf_numpy
fit_parametric_primitives_lr = _parametric_primitives_service.fit_parametric_primitives_lr
parametric_param_abs_error_torch = _parametric_primitives_service.parametric_param_abs_error_torch
parametric_param_abs_error_numpy = _parametric_primitives_service.parametric_param_abs_error_numpy


@dataclass(frozen=True)
class PrimitiveTarget:
    class_index: int
    params: np.ndarray
    mask: np.ndarray


class _Residual(nn.Module):
    # Purpose: Implement init for _Residual.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    # Purpose: Implement forward for _Residual.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.scale


class ParametricPrimitiveField(nn.Module):
    """V10.7.9 analytic primitive field with diagnostic neural heads.

    EXP_0002 showed semantic classification can overfit and collapse. Structural
    inference therefore uses deterministic LR evidence and inverse fitting; the
    split neural class/regression heads are retained only as independent
    telemetry/experimentation paths.
    """

    # Purpose: Implement init for ParametricPrimitiveField.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def __init__(self, input_channels: int, hidden_channels: int = 96, *, upscale: int = 4, max_distance_pixels: float = 24.0) -> None:
        super().__init__()
        if int(upscale) != 4:
            raise ValueError("V10.7.9 ParametricPrimitiveField requires 4x output")
        self.upscale = int(upscale)
        self.max_distance_pixels = float(max_distance_pixels)
        hidden = max(48, int(hidden_channels))
        moment_dim = 22
        # B1b does not consume raw random synthetic colour values. It uses the
        # explicit LR SDF plus five already-normalized physical-map guidance
        # channels (luma gx/gy, normal edge, material edge, curvature). Those
        # channels preserve observable stroke width/profile information that the
        # scalar SDF prior intentionally discards. Absolute x/y are appended.
        _ = int(input_channels)  # retained in the constructor contract
        in_channels = 12

        def make_encoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden, 5, stride=2, padding=2), nn.GELU(), _Residual(hidden),
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU(), _Residual(hidden),
                nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1), nn.GELU(), _Residual(hidden * 2),
                nn.Conv2d(hidden * 2, hidden * 2, 3, stride=1, padding=1), nn.GELU(), _Residual(hidden * 2),
            )

        def make_trunk() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden * 2 * 64 + moment_dim, hidden * 4), nn.GELU(),
                nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
            )

        # Classification and regression intentionally do not share learned
        # features. The classifier can be frozen once qualified while the
        # parameter branch continues learning exact geometry.
        self.class_encoder = make_encoder()
        self.class_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.class_trunk = make_trunk()
        self.class_head = nn.Linear(hidden * 2, PRIMITIVE_COUNT)

        self.param_encoder = make_encoder()
        self.param_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.param_trunk = make_trunk()
        self.param_head = nn.Linear(hidden * 2, PRIMITIVE_COUNT * PARAM_DIM)

        nn.init.zeros_(self.class_head.bias)
        # Zero residual means "use the deterministic LR estimate".
        nn.init.zeros_(self.param_head.weight)
        nn.init.zeros_(self.param_head.bias)

    # Purpose: Implement seed params from moments for ParametricPrimitiveField.
    # Called by: forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _seed_params_from_moments(moments: torch.Tensor) -> torch.Tensor:
        """Build class-specific deterministic geometry seeds from LR contour moments.

        These are initial measurements, not oracle targets.  The learned branch
        predicts bounded corrections.  The formulas are scale-normalized so the
        same seed works at LR training and 512px proof resolution.
        """
        batch = int(moments.shape[0])
        seed = moments.new_full((batch, PRIMITIVE_COUNT, PARAM_DIM), 0.5)
        cx = ((moments[:, 0] + 1.0) * 0.5).clamp(0.0, 1.0)
        cy = ((moments[:, 1] + 1.0) * 0.5).clamp(0.0, 1.0)
        seed[:, :, 0] = cx[:, None]
        seed[:, :, 1] = cy[:, None]

        cxx, cyy, cxy = moments[:, 2], moments[:, 3], moments[:, 4]
        trace = (cxx + cyy).clamp_min(1.0e-8)
        disc = torch.sqrt((cxx - cyy).square() + 4.0 * cxy.square() + 1.0e-8)
        eig_major = ((trace + disc) * 0.5).clamp_min(1.0e-8)
        eig_minor = ((trace - disc) * 0.5).clamp_min(1.0e-8)
        theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
        direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        seed[:, :, 2:4] = direction[:, None, :]

        def put(index: int, value: torch.Tensor) -> None:
            seed[:, :, index] = value[:, None].clamp(0.0, 1.0)

        # Five HR pixels is the modal manufactured-stroke width in the permanent
        # proof ladder and close to the centre of the randomized training range.
        put(8, moments.new_full((batch,), _width_to_unit(5.0)))

        # Oval family: covariance eigenvalues estimate ellipse axes. Near-
        # isotropic contours use radial mean, which is substantially less biased
        # for small circles after LR prior extraction.
        anisotropy = moments[:, 14]
        radial_fraction = (0.5 * moments[:, 5]).clamp(0.05, 0.45)
        major_fraction = torch.sqrt(eig_major * 0.5).clamp(0.08, 0.42)
        minor_fraction = (torch.sqrt(eig_minor * 0.5) - 0.04).clamp(0.05, 0.35)
        near_circle = anisotropy < 0.08
        axis_x = torch.where(near_circle, radial_fraction, major_fraction)
        axis_y = torch.where(near_circle, radial_fraction, minor_fraction)
        seed[:, 1, 4] = ((axis_x - 0.08) / 0.34).clamp(0.0, 1.0)
        seed[:, 1, 5] = ((axis_y - 0.05) / 0.30).clamp(0.0, 1.0)

        # Rounded-box half extents from boundary covariance. The 1.5 factor is
        # the perimeter-distribution analogue of the rectangle variance term.
        box_half_major = 0.5 * torch.sqrt(1.5 * eig_major)
        box_half_minor = 0.5 * torch.sqrt(1.5 * eig_minor)
        seed[:, 2, 4] = ((box_half_major - 0.12) / 0.30).clamp(0.0, 1.0)
        seed[:, 2, 5] = ((box_half_minor - 0.08) / 0.26).clamp(0.0, 1.0)

        # Parallel-line span follows the major-axis variance of a uniform
        # segment. Separation remains a learned correction because LR edge blur
        # biases the minor eigenvalue strongly.
        parallel_half_length = 0.5 * torch.sqrt(3.0 * eig_major)
        seed[:, 4, 5] = ((parallel_half_length - 0.15) / 0.45).clamp(0.0, 1.0)

        # Two concentric rings: radial mean is a stable average-radius estimate.
        # Seed a conservative small separation; the residual head resolves the
        # actual spacing from guidance edge profiles.
        ring_average = radial_fraction
        ring_inner = (ring_average - 0.03).clamp(0.06, 0.48)
        ring_outer = (ring_average + 0.03).clamp(0.06, 0.48)
        seed[:, 5, 4] = ((ring_inner - 0.06) / 0.42).clamp(0.0, 1.0)
        seed[:, 5, 5] = ((ring_outer - 0.06) / 0.42).clamp(0.0, 1.0)

        # Symmetric three-arm junction covariance is approximately L^2/6 in
        # normalized coordinates. Empirical LR-prior broadening is compensated
        # by 0.78; the network retains full bounded correction authority.
        junction_length_fraction = (0.39 * torch.sqrt(6.0 * trace * 0.5)).clamp(0.15, 0.60)
        seed[:, 6, 4] = ((junction_length_fraction - 0.15) / 0.45).clamp(0.0, 1.0)
        return seed

    # Purpose: Implement apply residual for ParametricPrimitiveField.
    # Called by: forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _apply_residual(raw: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        """Apply bounded learned corrections around deterministic geometry."""
        result = seed.clone()
        residual = torch.tanh(raw)
        result[..., 0:2] = (seed[..., 0:2] + 0.22 * residual[..., 0:2]).clamp(0.0, 1.0)
        for index in (4, 5, 6, 7, 8, 9, 10, 11):
            result[..., index] = (seed[..., index] + 0.55 * residual[..., index]).clamp(0.0, 1.0)
        direction = seed[..., 2:4] + 1.5 * residual[..., 2:4]
        direction = direction / torch.sqrt((direction * direction).sum(dim=-1, keepdim=True) + 1.0e-8)
        result[..., 2:4] = direction
        return result

    # Purpose: Implement geometry evidence for ParametricPrimitiveField.
    # Called by: forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _geometry_evidence(inputs_lr: torch.Tensor, source_sdf_prior_lr: torch.Tensor) -> torch.Tensor:
        sdf = source_sdf_prior_lr.float()
        px = F.pad(sdf, (1, 1, 0, 0), mode="replicate")
        py = F.pad(sdf, (0, 0, 1, 1), mode="replicate")
        gx = 0.5 * (px[:, :, :, 2:] - px[:, :, :, :-2])
        gy = 0.5 * (py[:, :, 2:, :] - py[:, :, :-2, :])
        grad = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
        # build_model_input channels 8..15 are normalized guidance. Keep only
        # geometry-bearing channels 9..13; severity/UV/chart flags are omitted.
        guidance = inputs_lr[:, 9:14].float()
        # [signed distance, unsigned distance, dx, dy, SDF edge strength,
        #  luma gx, luma gy, normal edge, material edge, luma curvature]
        return torch.cat((sdf, sdf.abs(), gx, gy, grad, guidance), dim=1)

    # Purpose: Implement geometry moments for ParametricPrimitiveField.
    # Called by: forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _geometry_moments(evidence_lr: torch.Tensor) -> torch.Tensor:
        """Low-order global geometry statistics in normalized LR coordinates."""
        sdf = evidence_lr[:, 0:1].float()
        gx = evidence_lr[:, 2:3].float()
        gy = evidence_lr[:, 3:4].float()
        batch, _channels, height, width = sdf.shape
        yy = torch.linspace(-1.0, 1.0, height, device=sdf.device, dtype=sdf.dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=sdf.device, dtype=sdf.dtype).view(1, 1, 1, width)
        weight = torch.exp(-6.0 * sdf.abs())
        norm = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0e-6)
        mx = (weight * xx).sum(dim=(2, 3), keepdim=True) / norm
        my = (weight * yy).sum(dim=(2, 3), keepdim=True) / norm
        dx = xx - mx
        dy = yy - my
        cxx = (weight * dx.square()).sum(dim=(2, 3), keepdim=True) / norm
        cyy = (weight * dy.square()).sum(dim=(2, 3), keepdim=True) / norm
        cxy = (weight * dx * dy).sum(dim=(2, 3), keepdim=True) / norm
        radius = torch.sqrt(dx.square() + dy.square() + 1.0e-8)
        rmean = (weight * radius).sum(dim=(2, 3), keepdim=True) / norm
        r2 = (weight * radius.square()).sum(dim=(2, 3), keepdim=True) / norm
        rstd = torch.sqrt((r2 - rmean.square()).clamp_min(0.0) + 1.0e-8)
        grad2 = gx.square() + gy.square() + 1.0e-8
        orient_x = (weight * (gx.square() - gy.square()) / grad2).sum(dim=(2, 3), keepdim=True) / norm
        orient_y = (weight * (2.0 * gx * gy) / grad2).sum(dim=(2, 3), keepdim=True) / norm
        edge_density = weight.mean(dim=(2, 3), keepdim=True)
        abs_mean = sdf.abs().mean(dim=(2, 3), keepdim=True)
        abs_std = sdf.abs().std(dim=(2, 3), keepdim=True, unbiased=False)
        signed_mean = sdf.mean(dim=(2, 3), keepdim=True)
        signed_std = sdf.std(dim=(2, 3), keepdim=True, unbiased=False)
        trace = (cxx + cyy).clamp_min(1.0e-8)
        anisotropy = torch.sqrt((cxx - cyy).square() + 4.0 * cxy.square() + 1.0e-8) / trace
        negative_fraction = (sdf < 0.0).float().mean(dim=(2, 3), keepdim=True)
        border = torch.zeros_like(weight)
        border[:, :, :2, :] = 1.0; border[:, :, -2:, :] = 1.0
        border[:, :, :, :2] = 1.0; border[:, :, :, -2:] = 1.0
        border_fraction = (weight * border).sum(dim=(2, 3), keepdim=True) / norm
        centered_r = radius - rmean
        r3 = (weight * centered_r.pow(3)).sum(dim=(2, 3), keepdim=True) / norm
        r4 = (weight * centered_r.pow(4)).sum(dim=(2, 3), keepdim=True) / norm
        rskew = r3 / rstd.clamp_min(1.0e-4).pow(3)
        rkurt = r4 / rstd.clamp_min(1.0e-4).pow(4)
        orient_coherence = torch.sqrt(orient_x.square() + orient_y.square() + 1.0e-8)
        signed_min = sdf.amin(dim=(2, 3), keepdim=True)
        signed_max = sdf.amax(dim=(2, 3), keepdim=True)
        features = torch.cat((
            mx, my, cxx, cyy, cxy, rmean, rstd, orient_x, orient_y,
            edge_density, abs_mean, abs_std, signed_mean, signed_std,
            anisotropy, negative_fraction, border_fraction, rskew, rkurt,
            orient_coherence, signed_min, signed_max,
        ), dim=1)
        return features.flatten(1)

    # Purpose: Implement with coordinates for ParametricPrimitiveField.
    # Called by: forward
    # Calls: No same-class helper methods.
    @staticmethod
    def _with_coordinates(evidence_lr: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = evidence_lr.shape
        yy = torch.linspace(-1.0, 1.0, height, device=evidence_lr.device, dtype=evidence_lr.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=evidence_lr.device, dtype=evidence_lr.dtype)
        ymap = yy.view(1, 1, height, 1).expand(batch, 1, height, width)
        xmap = xx.view(1, 1, 1, width).expand(batch, 1, height, width)
        return torch.cat((evidence_lr, xmap, ymap), dim=1)

    # Purpose: Implement classifier parameters for ParametricPrimitiveField.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def classifier_parameters(self):
        for module in (self.class_encoder, self.class_trunk, self.class_head):
            yield from module.parameters()

    # Purpose: Implement regressor parameters for ParametricPrimitiveField.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def regressor_parameters(self):
        for module in (self.param_encoder, self.param_trunk, self.param_head):
            yield from module.parameters()

    # Purpose: Implement forward for ParametricPrimitiveField.
    # Called by: External callers and the owning workflow.
    # Calls: _apply_residual, _geometry_evidence, _geometry_moments, _seed_params_from_moments, _with_coordinates
    def forward(self, inputs_lr: torch.Tensor, source_sdf_prior_lr: torch.Tensor) -> dict[str, torch.Tensor]:
        geometry_evidence = self._geometry_evidence(inputs_lr, source_sdf_prior_lr)
        moments = self._geometry_moments(geometry_evidence)
        evidence = self._with_coordinates(geometry_evidence)

        class_feature = self.class_pool(self.class_encoder(evidence)).flatten(1)
        class_feature = self.class_trunk(torch.cat((class_feature, moments), dim=1))
        logits = self.class_head(class_feature).float()

        param_feature = self.param_pool(self.param_encoder(evidence)).flatten(1)
        param_feature = self.param_trunk(torch.cat((param_feature, moments), dim=1))
        raw_params = self.param_head(param_feature).float().view(-1, PRIMITIVE_COUNT, PARAM_DIM)
        seed_params = self._seed_params_from_moments(moments)
        params_by_class = self._apply_residual(raw_params, seed_params)

        predicted_class = logits.argmax(dim=1)
        batch_index = torch.arange(logits.shape[0], device=logits.device)
        params = params_by_class[batch_index, predicted_class]
        h = int(inputs_lr.shape[-2]) * self.upscale
        w = int(inputs_lr.shape[-1]) * self.upscale
        phi = render_parametric_sdf_torch(params, predicted_class, h, w).clamp(-self.max_distance_pixels, self.max_distance_pixels)
        source_pixels = F.interpolate(
            source_sdf_prior_lr.float() * self.max_distance_pixels,
            size=(h, w), mode="bilinear", align_corners=False,
        )
        probability = torch.softmax(logits, dim=1)
        confidence = probability.max(dim=1, keepdim=True).values[:, :, None, None].expand(-1, -1, h, w)
        return {
            "phi_pixels": phi,
            "source_sdf_prior_pixels": source_pixels,
            "class_logits": logits,
            "class_index": predicted_class,
            "params": params,
            "params_by_class": params_by_class,
            "seed_params_by_class": seed_params,
            "confidence": confidence,
        }


_UNDIRECTED_ANGLE_CLASSES = frozenset((0, 1, 2, 4))
