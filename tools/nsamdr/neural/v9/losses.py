"""V10.7.9 deterministic geometry-redraw proof objective.

The objective directly supervises the bounded 2x topology proposal, projects
its shared graph nodes onto the GT contour, matches tangent lines, supervises
the resulting spline SDF, then scores the same
two-sided renderer used by the GT-SDF oracle. Appearance residuals remain
excluded so geometry cannot be solved by repainting the raster staircase.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import V9Config
from .contours import sobel_tensor
from .redistance import redistance_zero_contour, sdf_gradient_components
from .oracle_patch_distillation import extract_target_patches, extract_target_patch_validity
from .seam_restoration import multi_map_ridge_response
from .parametric_primitives import parametric_param_abs_error_torch, render_parametric_sdf_torch


class LossesService:
    # Purpose: Implement mean fp32 for LossesService.
    # Called by: _fine_zero_mean, charbonnier, compute_losses, normal_cosine_loss
    # Calls: No same-class helper methods.
    def _mean_fp32(self, value: torch.Tensor) -> torch.Tensor:
        return value.float().mean()

    # Purpose: Implement sum fp32 for LossesService.
    # Called by: _topology_sign_margin_loss, _weighted_mean, compute_losses, masked_mean
    # Calls: No same-class helper methods.
    def _sum_fp32(self, value: torch.Tensor) -> torch.Tensor:
        return value.float().sum()

    # Purpose: Implement target like for LossesService.
    # Called by: charbonnier, gradient_loss, normal_cosine_loss, pyramid_loss, seam_loss
    # Calls: No same-class helper methods.
    def _target_like(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return target.to(device=prediction.device, dtype=prediction.dtype, non_blocking=True)

    # Purpose: Implement charbonnier for LossesService.
    # Called by: compute_losses, gradient_loss, pyramid_loss, seam_loss
    # Calls: _mean_fp32, _target_like
    def charbonnier(self, prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
        target = self._target_like(target, prediction)
        return self._mean_fp32(torch.sqrt((prediction - target).square() + epsilon * epsilon))

    # Purpose: Implement gradient loss for LossesService.
    # Called by: compute_losses
    # Calls: _target_like, charbonnier
    def gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._target_like(target, prediction)
        pgx, pgy = sobel_tensor(prediction)
        tgx, tgy = sobel_tensor(target)
        return self.charbonnier(pgx, tgx) + self.charbonnier(pgy, tgy)

    # Purpose: Implement gradient magnitude for LossesService.
    # Called by: _coverage_profile_width, _projected_view_seam_loss, compute_losses
    # Calls: No same-class helper methods.
    def gradient_magnitude(self, value: torch.Tensor) -> torch.Tensor:
        gx, gy = sobel_tensor(value)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    # Purpose: Implement subpixel target stack for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _subpixel_target_stack(self, value: torch.Tensor, grid_n: int = 3) -> torch.Tensor:
        """Sample a metric target field at the same subpixel lattice as the implicit renderer."""
        b, _c, h, w = value.shape
        coords = [((i + 0.5) / grid_n - 0.5) for i in range(grid_n)]
        yy = (torch.arange(h, device=value.device, dtype=torch.float32) + 0.5) * (2.0 / max(h, 1)) - 1.0
        xx = (torch.arange(w, device=value.device, dtype=torch.float32) + 0.5) * (2.0 / max(w, 1)) - 1.0
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        base = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        samples = []
        for oy in coords:
            for ox in coords:
                grid = base.clone()
                grid[..., 0] += float(ox) * (2.0 / max(w, 1))
                grid[..., 1] += float(oy) * (2.0 / max(h, 1))
                samples.append(F.grid_sample(
                    value.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
                ))
        return torch.cat(samples, dim=1)

    # Purpose: Implement coverage profile width for LossesService.
    # Called by: compute_losses
    # Calls: gradient_magnitude
    def _coverage_profile_width(
        self,
        coverage: torch.Tensor,
        target_sdf_pixels: torch.Tensor,
        *,
        band_pixels: float = 6.0,
    ) -> torch.Tensor:
        """Differentiable per-sample RMS distance of coverage-gradient energy.

        This mirrors the intent of the audit profile-width metric, but operates on
        the shared coverage field before plateau colour can obscure the signal.
        """
        grad = self.gradient_magnitude(coverage.float())
        distance = target_sdf_pixels.float().abs()
        band = (distance <= float(band_pixels)).float()
        weight = grad * band
        numerator = (weight * distance.square()).flatten(1).sum(dim=1)
        denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0e-6)
        return torch.sqrt(numerator / denominator + 1.0e-8)

    # Purpose: Implement metricize level set pixels for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _metricize_level_set_pixels(
        self,
        sdf_pixels: torch.Tensor,
        max_distance_pixels: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility alias for the V9.9.3 explicit zero-contour redistance.

        Kept under the historical helper name so older diagnostics fail softly.
        There is no gradient denominator in V9.9.3.
        """
        raw = sdf_pixels.float()
        gx, gy = sdf_gradient_components(raw)
        grad = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
        metric = redistance_zero_contour(raw, float(max_distance_pixels))
        return metric, grad, torch.ones_like(metric)

    # Purpose: Implement pyramid loss for LossesService.
    # Called by: compute_losses
    # Calls: _target_like, charbonnier
    def pyramid_loss(self, prediction: torch.Tensor, target: torch.Tensor, levels: int = 3) -> torch.Tensor:
        target = self._target_like(target, prediction)
        total = prediction.new_zeros((), dtype=torch.float32)
        weight = 1.0
        for _ in range(levels):
            total = total + self.charbonnier(prediction, target) * weight
            if min(prediction.shape[-2:]) <= 8:
                break
            prediction = F.avg_pool2d(prediction, 2)
            target = F.avg_pool2d(target, 2)
            weight *= 0.5
        return total

    # Purpose: Implement normal cosine loss for LossesService.
    # Called by: compute_losses
    # Calls: _mean_fp32, _target_like
    def normal_cosine_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._target_like(target, prediction)
        pz = torch.sqrt((1.0 - prediction.square().sum(dim=1, keepdim=True)).clamp_min(1e-6))
        tz = torch.sqrt((1.0 - target.square().sum(dim=1, keepdim=True)).clamp_min(1e-6))
        p = F.normalize(torch.cat((prediction, pz), dim=1), dim=1, eps=1e-4)
        t = F.normalize(torch.cat((target, tz), dim=1), dim=1, eps=1e-4)
        return self._mean_fp32(1.0 - (p * t).sum(dim=1))

    # Purpose: Implement masked mean for LossesService.
    # Called by: compute_losses
    # Calls: _sum_fp32
    def masked_mean(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        while valid.ndim < value.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.to(device=value.device, dtype=value.dtype, non_blocking=True)
        expanded = valid.expand_as(value)
        return self._sum_fp32(value * expanded) / self._sum_fp32(expanded).clamp_min(1.0)

    # Purpose: Implement weighted mean for LossesService.
    # Called by: _projected_view_seam_loss, compute_losses
    # Calls: _sum_fp32
    def _weighted_mean(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
        if weight.shape[1] == 1 and value.shape[1] != 1:
            weight = weight.expand(-1, value.shape[1], -1, -1)
        return self._sum_fp32(value * weight) / self._sum_fp32(weight).clamp_min(1.0)

    # Purpose: Implement project contour offset to source lattice for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _project_contour_offset_to_source_lattice(
        self,
        target_offset_pixels: torch.Tensor,
        weight_hr: torch.Tensor,
        source_size: tuple[int, int],
        max_offset_pixels: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project a dense HR normal-offset teacher into the LR actuator lattice.

        The GeometryNet coarse actuator owns exactly one scalar per LR/source
        lattice cell.  Supervising its bilinear HR expansion against every HR
        target independently gives one actuator coefficient mutually inconsistent
        gradients.  Weighted first/second moments produce the least-squares target
        that the actuator can actually represent.  Local target variance lowers
        authority at junctions or multiple-contour cells where one scalar is not a
        valid geometric description.
        """
        target = target_offset_pixels.float()
        weight = weight_hr.float().clamp_min(0.0)
        pooled_weight = F.adaptive_avg_pool2d(weight, source_size)
        pooled_first = F.adaptive_avg_pool2d(target.detach() * weight, source_size) / (
            pooled_weight.clamp_min(1.0e-6)
        )
        pooled_second = F.adaptive_avg_pool2d(
            target.detach().square() * weight, source_size
        ) / pooled_weight.clamp_min(1.0e-6)
        variance = (pooled_second - pooled_first.square()).clamp_min(0.0)
        confidence = (
            pooled_weight.clamp(0.0, 1.0) * torch.exp(-variance / 2.25)
        ).detach()
        projected = pooled_first.clamp(
            -float(max_offset_pixels), float(max_offset_pixels)
        ).detach()
        projected_hr = F.interpolate(
            projected, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        confidence_hr = F.interpolate(
            confidence, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        return projected, confidence, projected_hr, confidence_hr

    # Purpose: Implement project contour vector to control lattice for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _project_contour_vector_to_control_lattice(
        self,
        target_flow_pixels: torch.Tensor,
        weight_hr: torch.Tensor,
        control_size: tuple[int, int],
        max_flow_pixels: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project an HR transport teacher into the continuous control lattice.

        Opposite sides of a translated thin feature carry opposite SDF normals but
        the *same vector transport*.  Projecting vectors therefore preserves motion
        that scalar normal offsets cancel.  Vector variance lowers confidence where
        one local transport is not a valid description (junctions/overlaps).
        """
        target = target_flow_pixels.float()
        weight = weight_hr.float().clamp_min(0.0)
        pooled_weight = F.adaptive_avg_pool2d(weight, control_size)
        pooled_first = F.adaptive_avg_pool2d(target.detach() * weight, control_size) / (
            pooled_weight.clamp_min(1.0e-6)
        )
        pooled_second = F.adaptive_avg_pool2d(
            target.detach().square().sum(dim=1, keepdim=True) * weight, control_size
        ) / pooled_weight.clamp_min(1.0e-6)
        variance = (pooled_second - pooled_first.square().sum(dim=1, keepdim=True)).clamp_min(0.0)
        confidence = (
            pooled_weight.clamp(0.0, 1.0) * torch.exp(-variance / 3.0)
        ).detach()
        magnitude = torch.sqrt(pooled_first.square().sum(dim=1, keepdim=True) + 1.0e-8)
        scale = torch.clamp(float(max_flow_pixels) / magnitude.clamp_min(1.0e-6), max=1.0)
        projected = (pooled_first * scale).detach()
        projected_hr = F.interpolate(
            projected, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        confidence_hr = F.interpolate(
            confidence, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        return projected, confidence, projected_hr, confidence_hr

    # Purpose: Implement sdf global polarity for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _sdf_global_polarity(
        self,
        predicted_pixels: torch.Tensor,
        target_pixels: torch.Tensor,
        metric_band_pixels: float,
    ) -> torch.Tensor:
        """Choose one physically equivalent SDF polarity per sample.

        A contour has no intrinsic positive/negative side in the EVE texture input.
        The renderer is invariant to a global SDF sign flip when its two side
        plateaus are swapped, so training must not punish that unobservable gauge.
        """
        band = max(float(metric_band_pixels), 1.0e-3)
        weight = torch.exp(-target_pixels.abs() / max(band * 0.55, 1.0e-3))
        denom = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        positive = ((predicted_pixels - target_pixels).abs() * weight).sum(
            dim=(1, 2, 3), keepdim=True
        ) / denom
        negative = ((predicted_pixels + target_pixels).abs() * weight).sum(
            dim=(1, 2, 3), keepdim=True
        ) / denom
        return torch.where(positive <= negative, torch.ones_like(positive), -torch.ones_like(negative)).detach()

    # Purpose: Implement balanced metric band mean for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _balanced_metric_band_mean(
        self,
        value: torch.Tensor,
        target_pixels: torch.Tensor,
        metric_band_pixels: float,
        *,
        near_pixels: float = 2.0,
    ) -> torch.Tensor:
        """Balance contour, positive-side and negative-side SDF supervision.

        This prevents a large background region from overwhelming thin lines, holes
        and rings. Every sample contributes fixed authority to the contour and to
        both metric sides inside the useful geometry band.
        """
        abs_target = target_pixels.abs()
        band = (abs_target <= float(metric_band_pixels)).float()
        near = (abs_target <= float(near_pixels)).float() * band
        positive = (target_pixels > float(near_pixels)).float() * band
        negative = (target_pixels < -float(near_pixels)).float() * band

        def normalized(mask: torch.Tensor) -> torch.Tensor:
            return mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

        weight = 0.60 * normalized(near) + 0.20 * normalized(positive) + 0.20 * normalized(negative)
        per_sample = (value.float() * weight.float()).sum(dim=(1, 2, 3))
        return per_sample.mean()

    # Purpose: Implement balanced sign mean for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _balanced_sign_mean(
        self,
        value: torch.Tensor,
        target_pixels: torch.Tensor,
        metric_band_pixels: float,
        *,
        near_pixels: float = 2.0,
    ) -> torch.Tensor:
        """Balance topology/sign supervision for thin and sparse geometry.

        Unlike area-weighted BCE, this gives the negative and positive material sides
        fixed authority even when one side is only a one- or two-pixel stripe.
        """
        abs_target = target_pixels.abs()
        band = (abs_target <= float(metric_band_pixels)).float()
        near = (abs_target <= float(near_pixels)).float() * band
        positive = (target_pixels > 0.0).float() * band
        negative = (target_pixels < 0.0).float() * band

        def normalized(mask: torch.Tensor) -> torch.Tensor:
            return mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

        weight = 0.40 * normalized(near) + 0.30 * normalized(positive) + 0.30 * normalized(negative)
        per_sample = (value.float() * weight.float()).sum(dim=(1, 2, 3))
        return per_sample.mean()

    # Purpose: Implement topology sign margin loss for LossesService.
    # Called by: compute_losses
    # Calls: _sum_fp32
    def _topology_sign_margin_loss(
        self,
        predicted_pixels: torch.Tensor,
        target_pixels: torch.Tensor,
        *,
        margin_pixels: float = 0.35,
        core_pixels: float = 0.75,
        band_pixels: float = 6.0,
        worst_fraction: float = 0.002,
    ) -> torch.Tensor:
        """Protect material connectivity while allowing the zero contour to move.

        Mean sign BCE can hide a one-pixel cut through a long thin feature because
        only a few pixels are wrong.  Such a cut is catastrophic for topology even
        when Chamfer improves.  This hinge loss looks only at confident target-side
        pixels near the contour and gives explicit authority to the worst local sign
        violations.  Global SDF polarity must already be aligned before calling it.
        """
        predicted = predicted_pixels.float()
        target = target_pixels.float()
        margin = max(float(margin_pixels), 0.0)
        core = max(float(core_pixels), margin + 1.0e-4)
        band = max(float(band_pixels), core + 1.0e-4)

        negative_core = (target <= -core) & (target.abs() <= band)
        positive_core = (target >= core) & (target.abs() <= band)
        violation = torch.where(
            negative_core,
            torch.relu(predicted + margin),
            torch.where(positive_core, torch.relu(margin - predicted), torch.zeros_like(predicted)),
        )
        active = (negative_core | positive_core).float()
        mean_violation = self._sum_fp32(violation * active) / self._sum_fp32(active).clamp_min(1.0)

        flat = violation.reshape(violation.shape[0], -1)
        active_count = active.reshape(active.shape[0], -1).sum(dim=1)
        per_sample_worst = []
        for sample_index in range(flat.shape[0]):
            count = int(active_count[sample_index].detach().item())
            if count <= 0:
                per_sample_worst.append(flat[sample_index].new_zeros((), dtype=torch.float32))
                continue
            k = max(1, min(count, int(round(count * max(float(worst_fraction), 1.0e-4)))))
            values = flat[sample_index]
            per_sample_worst.append(torch.topk(values, k=k, largest=True).values.float().mean())
        worst_violation = torch.stack(per_sample_worst).mean() if per_sample_worst else mean_violation.new_zeros(())
        return 0.35 * mean_violation + 0.65 * worst_violation

    # Purpose: Implement seam loss for LossesService.
    # Called by: compute_losses
    # Calls: _target_like, charbonnier
    def seam_loss(self, prediction: torch.Tensor, target: torch.Tensor, border: int = 8) -> torch.Tensor:
        target = self._target_like(target, prediction)
        border = max(1, min(border, prediction.shape[-1] // 4, prediction.shape[-2] // 4))
        regions = [
            (prediction[..., :border, :], target[..., :border, :]),
            (prediction[..., -border:, :], target[..., -border:, :]),
            (prediction[..., :, :border], target[..., :, :border]),
            (prediction[..., :, -border:], target[..., :, -border:]),
        ]
        return sum(self.charbonnier(p, t) for p, t in regions) / len(regions)

    # Purpose: Implement local error for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _local_error(self, value: torch.Tensor, target: torch.Tensor, patch: int) -> torch.Tensor:
        error = (value - target).abs().mean(dim=1, keepdim=True)
        patch = max(1, min(int(patch), error.shape[-1], error.shape[-2]))
        return F.avg_pool2d(error, kernel_size=patch, stride=patch)

    # Purpose: Implement local scalar error for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _local_scalar_error(self, value: torch.Tensor, patch: int) -> torch.Tensor:
        patch = max(1, min(int(patch), value.shape[-1], value.shape[-2]))
        return F.avg_pool2d(value, kernel_size=patch, stride=patch)

    # Purpose: Implement fine zero mean for LossesService.
    # Called by: compute_losses
    # Calls: _mean_fp32
    def _fine_zero_mean(self, delta: torch.Tensor, patch: int = 16) -> torch.Tensor:
        patch = max(2, min(int(patch), delta.shape[-1], delta.shape[-2]))
        local_mean = F.avg_pool2d(delta, patch, patch)
        return self._mean_fp32(local_mean.abs())

    # Purpose: Implement laplacian tensor for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def laplacian_tensor(self, value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        kernel = value.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        return F.conv2d(
            value,
            kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3),
            padding=1,
            groups=channels,
        )

    # Purpose: Implement axial encoding for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _axial_encoding(self, direction: torch.Tensor) -> torch.Tensor:
        unit = F.normalize(direction.float(), dim=1, eps=1e-4)
        tx, ty = unit[:, 0:1], unit[:, 1:2]
        return torch.cat((tx.square() - ty.square(), 2.0 * tx * ty), dim=1)

    # Purpose: Implement image tangent for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _image_tangent(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = value.float().mean(dim=1, keepdim=True)
        vector = scalar.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = (vector[:, None] * vector[None, :]) / 256.0
        smooth = F.conv2d(F.pad(scalar, (2, 2, 2, 2), mode="reflect"), kernel.view(1, 1, 5, 5))
        gx, gy = sobel_tensor(smooth)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
        tangent = torch.cat((-gy / magnitude, gx / magnitude), dim=1)
        return tangent, magnitude

    # Purpose: Implement field variation for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _field_variation(self, value: torch.Tensor) -> torch.Tensor:
        gx, gy = sobel_tensor(value)
        return torch.sqrt(gx.square() + gy.square() + 1.0e-8).mean(dim=1, keepdim=True)

    # Purpose: Implement normalise edge for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _normalise_edge(self, value: torch.Tensor) -> torch.Tensor:
        # Fixed monotonic mapping; unlike per-image normalisation it does not let a
        # weak/blurred contour look perfect merely because it is the strongest edge.
        return (value.float() * 2.5).clamp(0.0, 1.0)

    # Purpose: Implement directional derivatives for LossesService.
    # Called by: compute_losses
    # Calls: No same-class helper methods.
    def _directional_derivatives(self, value: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = value.float().mean(dim=1, keepdim=True)
        gx, gy = sobel_tensor(scalar)
        t = F.normalize(tangent.float(), dim=1, eps=1.0e-5)
        tx, ty = t[:, 0:1], t[:, 1:2]
        tangential = gx * tx + gy * ty
        normal = -gx * ty + gy * tx
        return tangential, normal

    # Purpose: Implement projected view seam loss for LossesService.
    # Called by: compute_losses
    # Calls: _weighted_mean, gradient_magnitude
    def _projected_view_seam_loss(
        self,
        prediction: torch.Tensor, target: torch.Tensor, target_edge: torch.Tensor
    ) -> torch.Tensor:
        """Cheap differentiable oblique-view proxy for seam realism.

        Actual mesh/view-space backprojection is a later stage; this proxy catches
        texture-space stair steps that become especially visible under rotated,
        anisotropically minified sampling.  It runs at half resolution to keep the
        structural proof memory-safe.
        """
        size = (min(256, prediction.shape[-2]), min(256, prediction.shape[-1]))
        pred = F.interpolate(prediction.float(), size=size, mode="bilinear", align_corners=False)
        tgt = F.interpolate(target.float(), size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(target_edge.float(), size=size, mode="bilinear", align_corners=False)
        total = pred.new_zeros(())
        # Rotation plus anisotropic scale approximates oblique UV projection.
        for degrees, sx, sy in ((23.0, 0.78, 1.0), (-41.0, 1.0, 0.72)):
            a = degrees * 3.141592653589793 / 180.0
            c, q = float(torch.cos(pred.new_tensor(a))), float(torch.sin(pred.new_tensor(a)))
            theta = pred.new_tensor([[c / sx, -q / sy, 0.0], [q / sx, c / sy, 0.0]]).unsqueeze(0).expand(pred.shape[0], -1, -1)
            grid = F.affine_grid(theta, pred.shape, align_corners=False)
            wp = F.grid_sample(pred, grid, mode="bilinear", padding_mode="border", align_corners=False)
            wt = F.grid_sample(tgt, grid, mode="bilinear", padding_mode="border", align_corners=False)
            we = F.grid_sample(edge, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0.0, 1.0)
            gp = self.gradient_magnitude(wp.mean(dim=1, keepdim=True))
            gt = self.gradient_magnitude(wt.mean(dim=1, keepdim=True))
            total = total + self._weighted_mean((gp - gt).abs(), 0.05 + 1.95 * we)
        return total * 0.5

    # Purpose: Implement compute losses for LossesService.
    # Called by: External callers and the owning workflow.
    # Calls: _axial_encoding, _balanced_metric_band_mean, _balanced_sign_mean, _coverage_profile_width, _directional_derivatives, _field_variation, _fine_zero_mean, _image_tangent, _local_error, _local_scalar_error, _mean_fp32, _metricize_level_set_pixels, _normalise_edge, _project_contour_offset_to_source_lattice, _project_contour_vector_to_control_lattice, _projected_view_seam_loss, _sdf_global_polarity, _subpixel_target_stack, _sum_fp32, _topology_sign_margin_loss, _weighted_mean, charbonnier, gradient_loss, gradient_magnitude, laplacian_tensor, masked_mean, normal_cosine_loss, pyramid_loss, seam_loss
    def compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        config: V9Config,
        phase: str,
    ) -> dict[str, torch.Tensor]:
        dtype = outputs["albedo"].dtype
        target_albedo = batch["target_albedo"].to(dtype=dtype, non_blocking=True)
        target_normal = batch["target_normal"].to(dtype=dtype, non_blocking=True)
        target_roughness = batch["target_roughness"].to(dtype=dtype, non_blocking=True)
        target_emissive = batch["target_emissive"].to(dtype=dtype, non_blocking=True)
        target_material_class = batch["target_material_class"].long()
        target_sdf = batch["target_sdf"].to(dtype=dtype, non_blocking=True)
        target_orientation = batch["target_orientation"].to(dtype=dtype, non_blocking=True)
        target_edge = batch["target_edge"].to(dtype=dtype, non_blocking=True)
        auxiliary_valid = batch["auxiliary_valid"].to(
            dtype=dtype, non_blocking=True
        ).view(-1, 1, 1, 1)
        geometry_exact = batch.get("geometry_exact")
        if geometry_exact is None:
            geometry_exact = torch.zeros(
                (target_albedo.shape[0], 1),
                device=target_albedo.device,
                dtype=dtype,
            )
        geometry_exact = geometry_exact.to(
            device=target_albedo.device, dtype=dtype, non_blocking=True
        ).view(-1, 1, 1, 1)
        geometry_need = batch.get("geometry_need")
        if geometry_need is None:
            geometry_need = torch.ones_like(target_sdf)
        geometry_need = geometry_need.to(
            device=target_albedo.device, dtype=dtype, non_blocking=True
        ).clamp(0.0, 1.0)
        need_weight = (
            float(config.geometry_need_floor)
            + (1.0 - float(config.geometry_need_floor)) * geometry_need.float()
        ).detach()

        losses: dict[str, torch.Tensor] = {}
        zero = outputs["albedo"].new_zeros((), dtype=torch.float32)

        # Final-output metrics remain available in every phase; the V10.6 detail branch is isolated until geometry/profile qualification.
        losses["albedo"] = self.charbonnier(outputs["albedo"], target_albedo)
        losses["albedo_gradient"] = self.gradient_loss(outputs["albedo"], target_albedo)
        losses["albedo_pyramid"] = self.pyramid_loss(outputs["albedo"], target_albedo)
        losses["normal"] = self.normal_cosine_loss(outputs["normal_xy"], target_normal)
        losses["normal_gradient"] = self.gradient_loss(outputs["normal_xy"], target_normal)
        roughness_error = torch.sqrt(
            (outputs["roughness"] - target_roughness).square() + 1e-6
        )
        emissive_error = torch.sqrt(
            (outputs["emissive"] - target_emissive).square() + 1e-6
        )
        losses["roughness"] = self.masked_mean(roughness_error, auxiliary_valid)
        losses["emissive"] = self.masked_mean(emissive_error, auxiliary_valid)
        material_ce = F.cross_entropy(
            outputs["material_logits"].float(),
            target_material_class,
            reduction="none",
        ).unsqueeze(1)
        losses["material"] = self.masked_mean(material_ce, auxiliary_valid.float())

        # Continuous contour supervision.
        near_contour = torch.exp(-target_sdf.abs() * 5.0)
        geometry_boost = 1.0 + geometry_exact * float(
            config.synthetic_geometry_loss_boost
        )
        contour_weight = (
            near_contour * target_edge.clamp(0.05, 1.0) * geometry_boost
        ).detach()

        predicted_sdf_pixels = outputs.get(
            "predicted_sdf_pixels",
            outputs["sdf"] * float(config.contour_sdf_max_distance_pixels),
        ).float()
        raw_target_sdf_pixels = (
            target_sdf.float() * float(config.contour_sdf_max_distance_pixels)
        )
        source_sdf_prior_pixels = outputs.get("source_sdf_prior_pixels")
        if source_sdf_prior_pixels is None:
            source_sdf_prior_pixels = batch.get("source_sdf", target_sdf).float() * float(
                config.contour_sdf_max_distance_pixels
            )
        else:
            source_sdf_prior_pixels = source_sdf_prior_pixels.float()
        if bool(config.sdf_sign_gauge_invariant):
            # Stable gauge: align HR target polarity to the observed LR prior, not
            # to the moving prediction. The network then learns an actual delta.
            sdf_polarity = self._sdf_global_polarity(
                source_sdf_prior_pixels.detach(), raw_target_sdf_pixels, config.sdf_metric_band_pixels
            )
        else:
            sdf_polarity = torch.ones(
                (predicted_sdf_pixels.shape[0], 1, 1, 1),
                device=predicted_sdf_pixels.device, dtype=predicted_sdf_pixels.dtype,
            )
        target_sdf_pixels = raw_target_sdf_pixels * sdf_polarity
        target_sdf_aligned = target_sdf.float() * sdf_polarity

        # V10.7.9 medial-stroke supervision.  The target is not two independent
        # boundary sides: for every deterministic LR ridge seed, find the authored
        # HR medial minimum along the seed normal, supervise one shared centre and
        # one half-width, then compare the analytic capsule union to the target SDF.
        stroke_active = outputs.get("stroke_centerline_active")
        if stroke_active is not None and float(stroke_active.detach().float().mean().item()) > 0.5:
            ridge = outputs["stroke_ridge_mask_lr"].float()
            seed_center = outputs["stroke_seed_center_lr"].float()
            center = outputs["stroke_center_lr"].float()
            seed_normal = outputs["stroke_seed_normal"].float()
            tangent = outputs["stroke_tangent"].float()
            seed_tangent = outputs["stroke_seed_tangent"].float()
            half_width = outputs["stroke_half_width_pixels"].float()
            bsz, _one, lr_h, lr_w = ridge.shape
            hr_h, hr_w = raw_target_sdf_pixels.shape[-2:]
            scale_x = float(hr_w) / float(max(lr_w, 1))
            scale_y = float(hr_h) / float(max(lr_h, 1))
            if abs(scale_x - scale_y) > 1.0e-5:
                raise RuntimeError("V10.7.9 stroke teacher requires isotropic 4x geometry")
            scale = scale_x
            offsets_hr = torch.linspace(
                -7.0, 7.0, 29, device=center.device, dtype=torch.float32
            )
            offsets_lr = offsets_hr / scale
            points = (
                seed_center.unsqueeze(-1)
                + seed_normal.unsqueeze(-1) * offsets_lr.view(1, 1, 1, 1, -1)
            )
            # B,2,H,W,S -> B,H,W*S,2 grid. LR coordinates are measured from
            # image edges, so align_corners=False normalisation is simply 2*x/W-1.
            grid_x = 2.0 * points[:, 0] / float(max(lr_w, 1)) - 1.0
            grid_y = 2.0 * points[:, 1] / float(max(lr_h, 1)) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1).reshape(bsz, lr_h, lr_w * len(offsets_hr), 2)
            teacher_samples = F.grid_sample(
                raw_target_sdf_pixels.detach(), grid, mode="bilinear",
                padding_mode="border", align_corners=False,
            ).reshape(bsz, 1, lr_h, lr_w, len(offsets_hr))[:, 0]
            # Soft minimum gives a subpixel medial target but remains a fixed teacher.
            weights = torch.softmax(-teacher_samples / 0.20, dim=-1).detach()
            teacher_offset_hr = (weights * offsets_hr.view(1, 1, 1, -1)).sum(dim=-1)
            teacher_depth = -(weights * teacher_samples).sum(dim=-1)
            teacher_valid = (teacher_samples.min(dim=-1).values < -0.20).float() * ridge[:, 0]
            teacher_count = teacher_valid.sum().clamp_min(1.0)

            predicted_offset_hr = (
                (center - seed_center) * seed_normal
            ).sum(dim=1) * scale
            losses["stroke_center"] = (
                (predicted_offset_hr - teacher_offset_hr).abs() * teacher_valid
            ).sum() / teacher_count
            losses["stroke_width"] = (
                (half_width[:, 0] - teacher_depth.clamp(0.25, 12.0)).abs() * teacher_valid
            ).sum() / teacher_count
            tangent_dot = (tangent * seed_tangent).sum(dim=1).abs().clamp(0.0, 1.0)
            losses["stroke_tangent"] = ((1.0 - tangent_dot) * teacher_valid).sum() / teacher_count
            stroke_band = (raw_target_sdf_pixels.abs() <= 6.0).float().detach()
            losses["stroke_render"] = self._weighted_mean(
                F.smooth_l1_loss(
                    predicted_sdf_pixels, raw_target_sdf_pixels.detach(), beta=0.12, reduction="none"
                ),
                (0.10 + 1.90 * stroke_band).detach(),
            )
            losses["stroke_teacher_valid_fraction"] = teacher_valid.mean().detach()
            losses["stroke_width_mean_pixels"] = self._weighted_mean(half_width[:, 0:1].detach(), ridge.detach())
        else:
            losses["stroke_center"] = zero
            losses["stroke_width"] = zero
            losses["stroke_tangent"] = zero
            losses["stroke_render"] = zero
            losses["stroke_teacher_valid_fraction"] = zero
            losses["stroke_width_mean_pixels"] = zero

        # V10.7.9 complete parametric supervision. Every synthetic B1b tile has a
        # dense class/parameter teacher; no spatial valid-mask sparsity is involved.
        primitive_active = outputs.get("parametric_primitive_active")
        primitive_valid = batch.get("primitive_valid")
        if (
            primitive_active is not None
            and primitive_valid is not None
            and float(primitive_active.detach().float().mean().item()) > 0.5
            and bool((primitive_valid.detach().float().reshape(-1) > 0.5).any().item())
        ):
            valid = primitive_valid.float().reshape(-1) > 0.5
            logits = outputs["primitive_class_logits"].float()
            target_class = batch["primitive_class"].long().reshape(-1)
            params_by_class = outputs.get("primitive_params_by_class")
            if params_by_class is not None:
                all_params = params_by_class.float()
                batch_index = torch.arange(all_params.shape[0], device=all_params.device)
                params = all_params[batch_index, target_class]
            else:
                params = outputs["primitive_params"].float()
            target_params = batch["primitive_params"].float()
            param_mask = batch["primitive_param_mask"].float() * valid.float().unsqueeze(1)
            losses["primitive_class"] = F.cross_entropy(logits[valid], target_class[valid])
            param_abs = parametric_param_abs_error_torch(params, target_params, target_class) * param_mask
            losses["primitive_param"] = param_abs.sum() / param_mask.sum().clamp_min(1.0)
            losses["primitive_param_mae"] = losses["primitive_param"].detach()
            losses["primitive_class_accuracy"] = (
                logits[valid].argmax(dim=1) == target_class[valid]
            ).float().mean().detach()
            gt_class_render = render_parametric_sdf_torch(
                params[valid], target_class[valid],
                int(raw_target_sdf_pixels.shape[-2]), int(raw_target_sdf_pixels.shape[-1]),
            ).clamp(-float(config.contour_sdf_max_distance_pixels), float(config.contour_sdf_max_distance_pixels))
            target_render_sdf = raw_target_sdf_pixels[valid].detach()
            render_band = (target_render_sdf.abs() <= 8.0).float()
            losses["primitive_render"] = self._weighted_mean(
                F.smooth_l1_loss(gt_class_render, target_render_sdf, beta=0.12, reduction="none"),
                render_band.detach(),
            )
        else:
            losses["primitive_class"] = zero
            losses["primitive_param"] = zero
            losses["primitive_param_mae"] = zero
            losses["primitive_class_accuracy"] = zero
            losses["primitive_render"] = zero

        metricized_predicted_sdf_pixels = outputs.get("predicted_sdf_redistanced_pixels", outputs.get("predicted_sdf_metric_pixels"))
        if metricized_predicted_sdf_pixels is None:
            metricized_predicted_sdf_pixels, raw_grad_for_metricization, metricization_denominator = (
                self._metricize_level_set_pixels(
                    predicted_sdf_pixels, float(config.contour_sdf_max_distance_pixels)
                )
            )
        else:
            metricized_predicted_sdf_pixels = metricized_predicted_sdf_pixels.float()
            raw_gx_metric, raw_gy_metric = sdf_gradient_components(predicted_sdf_pixels)
            raw_grad_for_metricization = torch.sqrt(
                raw_gx_metric.square() + raw_gy_metric.square() + 1.0e-8
            )
            metricization_denominator = torch.ones_like(metricized_predicted_sdf_pixels)

        # V9.9.3 keeps raw-field regularization, but position-sensitive
        # supervision is evaluated in redistanced physical pixel distance. EXP_0001 showed
        # that |phi| at the target contour is not a positional error when |grad phi|
        # is far below one.
        # sign/topology objectives retain authority, but far-field pixel count may
        # no longer dominate the contour itself.
        metric_band = float(config.sdf_metric_band_pixels)
        raw_surface_error = F.smooth_l1_loss(
            predicted_sdf_pixels, target_sdf_pixels, beta=0.35, reduction="none"
        )
        metricized_surface_error = F.smooth_l1_loss(
            metricized_predicted_sdf_pixels, target_sdf_pixels, beta=0.35, reduction="none"
        )
        losses["sdf_raw_surface"] = self._balanced_metric_band_mean(
            raw_surface_error, target_sdf_pixels, metric_band
        )
        # Main surface authority is physical pixel distance, not raw level-set units.
        losses["sdf_surface"] = self._balanced_metric_band_mean(
            metricized_surface_error, target_sdf_pixels, metric_band
        )
        losses["sdf"] = self._balanced_metric_band_mean(
            (metricized_predicted_sdf_pixels / max(float(config.contour_sdf_max_distance_pixels), 1.0e-6)
             - target_sdf_aligned).abs(),
            target_sdf_pixels, metric_band
        )

        # LR->HR geometric correction authority. Source prior stays in its observed
        # gauge; the target was aligned to it above. GeometryNet therefore predicts
        # the physical correction required by this pair instead of memorising an
        # absolute synthetic field.
        target_delta_pixels = target_sdf_pixels - source_sdf_prior_pixels
        predicted_delta_pixels = outputs.get(
            "sdf_delta_pixels", predicted_sdf_pixels - source_sdf_prior_pixels
        ).float()
        delta_error = F.smooth_l1_loss(
            predicted_delta_pixels, target_delta_pixels, beta=0.35, reduction="none"
        ) * need_weight
        losses["sdf_delta_surface"] = self._balanced_metric_band_mean(
            delta_error, target_sdf_pixels, metric_band, near_pixels=3.0
        )

        # V9.9.3: Chamfer can improve while a shallow-angle contour develops
        # visible one/two-pixel kinks. Supervise the *shape of the correction field*
        # in physical pixel units. Matching the target delta's tangential derivative
        # and Laplacian removes short-period wobble without flattening authored
        # corners: any legitimate target curvature remains present in target_delta.
        pred_delta_gx, pred_delta_gy = sdf_gradient_components(predicted_delta_pixels)
        target_delta_gx, target_delta_gy = sdf_gradient_components(target_delta_pixels.detach())
        target_tangent_delta = F.normalize(target_orientation.float(), dim=1, eps=1.0e-4)
        pred_delta_dt = (
            pred_delta_gx * target_tangent_delta[:, 0:1]
            + pred_delta_gy * target_tangent_delta[:, 1:2]
        )
        target_delta_dt = (
            target_delta_gx * target_tangent_delta[:, 0:1]
            + target_delta_gy * target_tangent_delta[:, 1:2]
        ).detach()
        delta_shape_weight = (
            (target_sdf_pixels.abs() <= 4.0).float()
            * (0.35 + 0.65 * need_weight)
        ).detach()
        losses["sdf_delta_tangent"] = self._weighted_mean(
            F.smooth_l1_loss(pred_delta_dt, target_delta_dt, beta=0.20, reduction="none"),
            delta_shape_weight,
        )
        pred_delta_lap = self.laplacian_tensor(predicted_delta_pixels)
        target_delta_lap = self.laplacian_tensor(target_delta_pixels.detach()).detach()
        losses["sdf_delta_laplacian"] = self._weighted_mean(
            F.smooth_l1_loss(pred_delta_lap, target_delta_lap, beta=0.25, reduction="none"),
            delta_shape_weight,
        )

        # V9.9.3 continuous-implicit representation. A scalar normal offset is
        # ambiguous for thin features: the two sides of a translated stripe need
        # opposite scalar offsets and can cancel in one LR cell.  Convert the
        # source->target normal correction into a 2-D transport vector first.  A
        # coherent translation then has the same vector on both sides; residual
        # same-sign motion is represented separately as dilation/erosion.
        target_normal_offset_pixels = source_sdf_prior_pixels - target_sdf_pixels
        src_gx, src_gy = sdf_gradient_components(source_sdf_prior_pixels.detach())
        src_norm = torch.sqrt(src_gx.square() + src_gy.square() + 1.0e-6)
        src_nx = src_gx / src_norm
        src_ny = src_gy / src_norm
        target_transport_hr = torch.cat((
            target_normal_offset_pixels.detach() * src_nx,
            target_normal_offset_pixels.detach() * src_ny,
        ), dim=1)

        source_contour_tube = torch.exp(-source_sdf_prior_pixels.abs() / 2.5)
        offset_weight = (
            source_contour_tube
            * (0.30 + 0.70 * need_weight)
            * (0.35 + 0.65 * target_edge.float().clamp(0.0, 1.0))
        ).detach()

        predicted_transport_control = outputs.get("contour_transport_control_pixels")
        predicted_transport_hr = outputs.get("contour_transport_pixels")
        predicted_dilation_control = outputs.get("contour_dilation_control_pixels")
        predicted_dilation_hr = outputs.get("contour_dilation_pixels")

        if predicted_transport_control is not None and predicted_transport_hr is not None:
            predicted_transport_control = predicted_transport_control.float()
            predicted_transport_hr = predicted_transport_hr.float()
            projected_transport, transport_confidence, projected_transport_hr, transport_confidence_hr = (
                self._project_contour_vector_to_control_lattice(
                    target_transport_hr,
                    offset_weight,
                    tuple(predicted_transport_control.shape[-2:]),
                    float(config.contour_transport_max_pixels),
                )
            )
            transport_error = F.smooth_l1_loss(
                predicted_transport_control, projected_transport, beta=0.20, reduction="none"
            ).mean(dim=1, keepdim=True)
            losses["contour_transport"] = self._weighted_mean(
                transport_error, transport_confidence + 0.03
            )
            target_transport_normal = (
                projected_transport_hr[:, 0:1] * src_nx
                + projected_transport_hr[:, 1:2] * src_ny
            )
        else:
            projected_transport_hr = torch.zeros(
                (predicted_sdf_pixels.shape[0], 2, *predicted_sdf_pixels.shape[-2:]),
                device=predicted_sdf_pixels.device,
                dtype=predicted_sdf_pixels.dtype,
            )
            transport_confidence_hr = torch.zeros_like(predicted_sdf_pixels)
            target_transport_normal = torch.zeros_like(predicted_sdf_pixels)
            losses["contour_transport"] = predicted_sdf_pixels.new_zeros(())

        target_dilation_hr = (
            target_normal_offset_pixels.detach() - target_transport_normal.detach()
        ).clamp(
            -float(config.contour_dilation_max_pixels),
            float(config.contour_dilation_max_pixels),
        )
        if predicted_dilation_control is not None and predicted_dilation_hr is not None:
            predicted_dilation_control = predicted_dilation_control.float()
            predicted_dilation_hr = predicted_dilation_hr.float()
            projected_dilation, dilation_confidence, projected_dilation_hr, dilation_confidence_hr = (
                self._project_contour_offset_to_source_lattice(
                    target_dilation_hr,
                    offset_weight * (0.35 + 0.65 * (1.0 - transport_confidence_hr)),
                    tuple(predicted_dilation_control.shape[-2:]),
                    float(config.contour_dilation_max_pixels),
                )
            )
            losses["contour_dilation"] = self._weighted_mean(
                F.smooth_l1_loss(
                    predicted_dilation_control, projected_dilation, beta=0.15, reduction="none"
                ),
                dilation_confidence + 0.03,
            )
        else:
            projected_dilation_hr = torch.zeros_like(predicted_sdf_pixels)
            dilation_confidence_hr = torch.zeros_like(predicted_sdf_pixels)
            losses["contour_dilation"] = predicted_sdf_pixels.new_zeros(())

        predicted_normal_offset_pixels = outputs.get(
            "contour_normal_offset_pixels", -predicted_delta_pixels
        ).float()
        losses["contour_normal_offset"] = self._weighted_mean(
            F.smooth_l1_loss(
                predicted_normal_offset_pixels,
                target_normal_offset_pixels.detach(),
                beta=0.25,
                reduction="none",
            ),
            offset_weight,
        )

        # Prevent transport folds.  A positive pull-back Jacobian keeps a connected
        # source contour connected; this directly attacks the dashed shallow-line
        # failure observed in EXP_0002 rather than merely penalising it after render.
        if predicted_transport_hr is not None:
            flow_x = predicted_transport_hr[:, 0:1]
            flow_y = predicted_transport_hr[:, 1:2]
            fx_x, fx_y = sdf_gradient_components(flow_x)
            fy_x, fy_y = sdf_gradient_components(flow_y)
            jac_det = (1.0 - fx_x) * (1.0 - fy_y) - fx_y * fy_x
            fold_penalty = F.relu(float(config.contour_transport_min_jacobian) - jac_det)
            losses["contour_transport_fold"] = self._weighted_mean(
                fold_penalty,
                (0.15 + 0.85 * source_contour_tube).detach(),
            )
        else:
            losses["contour_transport_fold"] = predicted_sdf_pixels.new_zeros(())

        # Deprecated V9.9.3 scalar/phase telemetry is kept at zero so old log readers
        # remain parseable while the new transport/dilation terms are authoritative.
        losses["contour_projected_offset"] = losses["contour_transport"].detach()
        losses["contour_phase_refine"] = losses["contour_dilation"].detach()

        # Direct differentiable sub-pixel occupancy target.  This bypasses the
        # discrete zero-crossing/redistance path during optimisation while matching
        # the same hard-profile geometry that Panel 2 renders.  Once the raw zero set
        # is correct, deterministic redistance gives the physical metric SDF.
        coverage_temperature = 0.45
        target_soft_coverage = torch.sigmoid(-target_sdf_pixels.detach() / coverage_temperature)
        soft_coverage_error = F.binary_cross_entropy_with_logits(
            -predicted_sdf_pixels / coverage_temperature,
            target_soft_coverage,
            reduction="none",
        )
        soft_coverage_weight = (
            torch.exp(-target_sdf_pixels.abs() / 2.5)
            + 0.35 * torch.exp(-source_sdf_prior_pixels.detach().abs() / 2.5)
        ) * (0.45 + 0.55 * need_weight)
        losses["contour_soft_coverage"] = self._weighted_mean(
            soft_coverage_error, soft_coverage_weight.detach()
        )

        # V9.9.3: supervise the actual arbitrary-coordinate implicit field, not
        # only its values at output-pixel centres.  The previous model could reduce
        # mean Chamfer while retaining the LR staircase between centres.
        implicit_samples = outputs.get("implicit_phi_samples_pixels")
        if implicit_samples is not None and implicit_samples.shape[1] == 9:
            pred_samples = implicit_samples.float()
            target_samples = self._subpixel_target_stack(target_sdf_pixels.detach(), 3)
            sample_weight = (
                0.15 + 1.85 * torch.exp(-target_samples.abs() / 4.0)
            ).detach()
            losses["implicit_subpixel_surface"] = self._weighted_mean(
                F.smooth_l1_loss(pred_samples, target_samples, beta=0.20, reduction="none"),
                sample_weight,
            )

            # 3x3 ordering is row-major: estimate metric gradient directly over the
            # subpixel cell. This penalises stair-step and radial wobble where it is
            # created, before rendering or contour extraction.
            step = 2.0 / 3.0
            pred_gx_q = (pred_samples[:, 5:6] - pred_samples[:, 3:4]) / step
            pred_gy_q = (pred_samples[:, 7:8] - pred_samples[:, 1:2]) / step
            tgt_gx_q = (target_samples[:, 5:6] - target_samples[:, 3:4]) / step
            tgt_gy_q = (target_samples[:, 7:8] - target_samples[:, 1:2]) / step
            center_weight = (
                0.20 + 1.80 * torch.exp(-target_samples[:, 4:5].abs() / 4.0)
            ).detach()
            losses["implicit_subpixel_gradient"] = self._weighted_mean(
                F.smooth_l1_loss(pred_gx_q, tgt_gx_q, beta=0.12, reduction="none")
                + F.smooth_l1_loss(pred_gy_q, tgt_gy_q, beta=0.12, reduction="none"),
                center_weight,
            )
            pred_grad_q = torch.sqrt(pred_gx_q.square() + pred_gy_q.square() + 1.0e-8)
            losses["implicit_subpixel_eikonal"] = self._weighted_mean(
                (pred_grad_q - 1.0).abs(), center_weight
            )
        else:
            losses["implicit_subpixel_surface"] = predicted_sdf_pixels.new_zeros(())
            losses["implicit_subpixel_gradient"] = predicted_sdf_pixels.new_zeros(())
            losses["implicit_subpixel_eikonal"] = predicted_sdf_pixels.new_zeros(())

        # V10 oracle-distilled overlapping patch supervision.  Every LR location
        # predicts a local HR SDF/coverage patch and receives the exact GT patch as
        # its teacher.  The overlap-consistency term prevents adjacent local
        # predictions from forming disconnected contour fragments even when their
        # individual average error is low.
        oracle_patch_sdf = outputs.get("oracle_patch_sdf_patches_pixels")
        oracle_patch_cov = outputs.get("oracle_patch_coverage_patches")
        if oracle_patch_sdf is not None and oracle_patch_cov is not None:
            patch_size = int(getattr(config, "oracle_patch_footprint_lr", 3)) * 4
            target_patch_sdf = extract_target_patches(
                target_sdf_pixels.detach(), patch_size=patch_size,
                upscale=4, footprint_lr=int(getattr(config, "oracle_patch_footprint_lr", 3)),
            )
            patch_valid = extract_target_patch_validity(
                target_sdf_pixels.detach(), patch_size=patch_size,
                upscale=4, footprint_lr=int(getattr(config, "oracle_patch_footprint_lr", 3)),
            )
            patch_weight = (
                (0.04 + 1.96 * torch.exp(-target_patch_sdf.abs() / 4.0)) * patch_valid
            ).detach()
            patch_surface = F.smooth_l1_loss(
                oracle_patch_sdf.float(), target_patch_sdf, beta=0.15, reduction="none"
            )
            losses["oracle_patch_sdf"] = self._weighted_mean(patch_surface, patch_weight)

            target_inside_patch = (target_patch_sdf < 0.0).float()
            patch_sign = F.binary_cross_entropy_with_logits(
                -oracle_patch_sdf.float() / 0.35,
                target_inside_patch,
                reduction="none",
            )
            losses["oracle_patch_sign"] = self._weighted_mean(
                patch_sign, (0.08 + 1.92 * torch.exp(-target_patch_sdf.abs() / 2.0)).detach()
            )

            teacher_cov = outputs.get("sdf_teacher_coverage_negative")
            if teacher_cov is None:
                teacher_cov = torch.sigmoid(-target_sdf_pixels.detach() / 0.45)
            target_patch_cov = extract_target_patches(
                teacher_cov.float().detach(), patch_size=patch_size,
                upscale=4, footprint_lr=int(getattr(config, "oracle_patch_footprint_lr", 3)),
            ).clamp(0.0, 1.0)
            pred_patch_cov = oracle_patch_cov.float().clamp(1.0e-5, 1.0 - 1.0e-5)
            losses["oracle_patch_coverage"] = self._weighted_mean(
                (pred_patch_cov - target_patch_cov).abs(), patch_weight
            )
            patch_cov_bce = F.binary_cross_entropy(
                pred_patch_cov, target_patch_cov, reduction="none"
            )
            losses["oracle_patch_coverage_bce"] = self._weighted_mean(
                patch_cov_bce, patch_weight
            )

            aggregate_patches = extract_target_patches(
                predicted_sdf_pixels.detach(), patch_size=patch_size,
                upscale=4, footprint_lr=int(getattr(config, "oracle_patch_footprint_lr", 3)),
            )
            losses["oracle_patch_overlap_consistency"] = self._weighted_mean(
                F.smooth_l1_loss(
                    oracle_patch_sdf.float(), aggregate_patches,
                    beta=0.10, reduction="none",
                ),
                patch_weight,
            )
            op_gx, op_gy = sdf_gradient_components(predicted_sdf_pixels)
            ot_gx, ot_gy = sdf_gradient_components(target_sdf_pixels.detach())
            oracle_grad_weight = (
                0.10 + 1.90 * torch.exp(-target_sdf_pixels.detach().abs() / 4.0)
            ).detach()
            losses["oracle_patch_gradient"] = self._weighted_mean(
                F.smooth_l1_loss(op_gx, ot_gx, beta=0.10, reduction="none")
                + F.smooth_l1_loss(op_gy, ot_gy, beta=0.10, reduction="none"),
                oracle_grad_weight,
            )
            direct_cov = outputs.get("oracle_patch_coverage")
            if direct_cov is not None:
                direct_cov = direct_cov.float().clamp(1.0e-5, 1.0 - 1.0e-5)
                losses["oracle_coverage_aggregate"] = self._weighted_mean(
                    (direct_cov - teacher_cov.float().detach()).abs(),
                    (0.04 + 1.96 * torch.exp(-target_sdf_pixels.detach().abs() / 4.0)).detach(),
                )
            else:
                losses["oracle_coverage_aggregate"] = predicted_sdf_pixels.new_zeros(())
        else:
            losses["oracle_patch_sdf"] = predicted_sdf_pixels.new_zeros(())
            losses["oracle_patch_sign"] = predicted_sdf_pixels.new_zeros(())
            losses["oracle_patch_coverage"] = predicted_sdf_pixels.new_zeros(())
            losses["oracle_patch_coverage_bce"] = predicted_sdf_pixels.new_zeros(())
            losses["oracle_patch_overlap_consistency"] = predicted_sdf_pixels.new_zeros(())
            losses["oracle_patch_gradient"] = predicted_sdf_pixels.new_zeros(())
            losses["oracle_coverage_aggregate"] = predicted_sdf_pixels.new_zeros(())


        # V10.4 topology-safe branch-smooth spline-graph supervision.  A bounded 2x control
        # field learns local connectivity repairs; each shared contour node may then
        # move freely in 2-D. GT SDF projects every graph node directly onto the HR
        # oracle contour and also supplies its target tangent line.
        spline_control = outputs.get("spline_graph_control_phi_pixels")
        spline_source_control = outputs.get("spline_graph_source_control_phi_pixels")
        spline_h = outputs.get("spline_control_point_h_lr")
        spline_v = outputs.get("spline_control_point_v_lr")
        spline_src_h = outputs.get("spline_source_control_point_h_lr")
        spline_src_v = outputs.get("spline_source_control_point_v_lr")
        spline_tan_h = outputs.get("spline_control_tangent_h")
        spline_tan_v = outputs.get("spline_control_tangent_v")
        spline_disp_h = outputs.get("spline_control_displacement_h_lr")
        spline_disp_v = outputs.get("spline_control_displacement_v_lr")
        spline_mask_h = outputs.get("spline_graph_mask_h")
        spline_mask_v = outputs.get("spline_graph_mask_v")
        if (spline_control is not None and spline_source_control is not None
                and spline_h is not None and spline_v is not None and spline_src_h is not None
                and spline_src_v is not None and spline_tan_h is not None and spline_tan_v is not None
                and spline_disp_h is not None and spline_disp_v is not None
                and spline_mask_h is not None and spline_mask_v is not None):
            target_detached = target_sdf_pixels.detach().float()
            tgx, tgy = sdf_gradient_components(target_detached)
            hr_h, hr_w = target_detached.shape[-2:]
            control_scale = float(getattr(config, "spline_graph_control_scale", 2))
            control_spacing_hr = 4.0 / max(control_scale, 1.0)
            control_origin = 2.0

            # Supervise the denser topology/level-set proposal directly.  Unlike the
            # hard marching-squares masks this path is fully differentiable, so the
            # network can learn to split/restore locally ambiguous LR components
            # before spline-node geometry is fitted.
            ch, cw = spline_control.shape[-2:]
            cy, cx = torch.meshgrid(
                torch.arange(ch, device=spline_control.device, dtype=torch.float32),
                torch.arange(cw, device=spline_control.device, dtype=torch.float32),
                indexing="ij",
            )
            physical_x = control_origin + control_spacing_hr * cx
            physical_y = control_origin + control_spacing_hr * cy
            control_grid = torch.stack(
                (2.0 * physical_x / float(hr_w) - 1.0,
                 2.0 * physical_y / float(hr_h) - 1.0),
                dim=-1,
            ).unsqueeze(0).expand(target_detached.shape[0], -1, -1, -1)
            target_control = F.grid_sample(
                target_detached, control_grid, mode="bilinear", padding_mode="border", align_corners=False
            ).detach()
            control_weight = (0.20 + 1.80 * torch.exp(-target_control.abs() / 4.0)).detach()
            losses["spline_graph_topology_control"] = self._weighted_mean(
                F.smooth_l1_loss(spline_control.float(), target_control, beta=0.12, reduction="none"),
                control_weight,
            )
            target_inside_control = (target_control < 0.0).float()
            losses["spline_graph_topology_sign"] = self._weighted_mean(
                F.binary_cross_entropy_with_logits(
                    -spline_control.float() / 0.75, target_inside_control, reduction="none"
                ),
                control_weight,
            )

            def _sample_at_lr_points(field: torch.Tensor, points_lr: torch.Tensor) -> torch.Tensor:
                physical_x = control_origin + control_spacing_hr * points_lr[..., 0]
                physical_y = control_origin + control_spacing_hr * points_lr[..., 1]
                grid_x = 2.0 * physical_x / float(hr_w) - 1.0
                grid_y = 2.0 * physical_y / float(hr_h) - 1.0
                grid = torch.stack((grid_x, grid_y), dim=-1).float()
                return F.grid_sample(
                    field.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
                )[:, 0]

            def _point_teacher(source_points_lr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                phi = _sample_at_lr_points(target_detached, source_points_lr)
                gx = _sample_at_lr_points(tgx, source_points_lr)
                gy = _sample_at_lr_points(tgy, source_points_lr)
                norm = torch.sqrt(gx.square() + gy.square() + 1.0e-8)
                nx = gx / norm
                ny = gy / norm
                source_x_hr = control_origin + control_spacing_hr * source_points_lr[..., 0]
                source_y_hr = control_origin + control_spacing_hr * source_points_lr[..., 1]
                target_x_hr = source_x_hr - phi * nx
                target_y_hr = source_y_hr - phi * ny
                target_lr = torch.stack(
                    ((target_x_hr - control_origin) / control_spacing_hr, (target_y_hr - control_origin) / control_spacing_hr),
                    dim=-1,
                ).detach()
                tangent = torch.stack((-ny, nx), dim=-1).detach()
                return target_lr, tangent, phi.abs().detach()

            target_h, target_tan_h, target_err_h = _point_teacher(spline_src_h.float())
            target_v, target_tan_v, target_err_v = _point_teacher(spline_src_v.float())
            mh = spline_mask_h[:, 0].float()
            mv = spline_mask_v[:, 0].float()
            # V10.7.9 B1b uses the same direct node/tangent objective as the
            # successful startup proof. Do not reweight points by their original
            # raster error and do not change the objective into HR-pixel SmoothL1;
            # both differences made the production optimizer chase local raster
            # irregularity instead of the globally smooth ordered branch.
            point_h_error = (spline_h.float() - target_h).abs().sum(dim=-1)
            point_v_error = (spline_v.float() - target_v).abs().sum(dim=-1)
            losses["spline_graph_point"] = (
                (point_h_error * mh).sum() + (point_v_error * mv).sum()
            ) / (mh.sum() + mv.sum()).clamp_min(1.0)

            dot_h = (spline_tan_h.float() * target_tan_h).sum(dim=-1).abs().clamp(0.0, 1.0)
            dot_v = (spline_tan_v.float() * target_tan_v).sum(dim=-1).abs().clamp(0.0, 1.0)
            losses["spline_graph_tangent"] = (
                ((1.0 - dot_h) * mh).sum() + ((1.0 - dot_v) * mv).sum()
            ) / (mh.sum() + mv.sum()).clamp_min(1.0)

            losses["spline_graph_displacement"] = (
                ((spline_h.float() - spline_src_h.float()).norm(dim=-1) * mh).sum()
                + ((spline_v.float() - spline_src_v.float()).norm(dim=-1) * mv).sum()
            ) * control_spacing_hr / (mh.sum() + mv.sum()).clamp_min(1.0)

            # V10.4 graph-space regularity. Marching-squares connectivity is used
            # directly so neighbouring endpoints on the same contour span cannot
            # learn independent one-cell zigzags. Smooth tangent authority is applied
            # only where the GT branch itself is smooth; corners/junctions are exempt.
            disp_h_hr = spline_disp_h.float() * control_spacing_hr
            disp_v_hr = spline_disp_v.float() * control_spacing_hr
            cell_disp = (
                disp_h_hr[:, :-1, :, :],
                disp_v_hr[:, :, 1:, :],
                disp_h_hr[:, 1:, :, :],
                disp_v_hr[:, :, :-1, :],
            )
            cell_point = (
                spline_h.float()[:, :-1, :, :],
                spline_v.float()[:, :, 1:, :],
                spline_h.float()[:, 1:, :, :],
                spline_v.float()[:, :, :-1, :],
            )
            cell_target = (
                target_h[:, :-1, :, :],
                target_v[:, :, 1:, :],
                target_h[:, 1:, :, :],
                target_v[:, :, :-1, :],
            )
            cell_tangent = (
                spline_tan_h.float()[:, :-1, :, :],
                spline_tan_v.float()[:, :, 1:, :],
                spline_tan_h.float()[:, 1:, :, :],
                spline_tan_v.float()[:, :, :-1, :],
            )
            cell_target_tangent = (
                target_tan_h[:, :-1, :, :],
                target_tan_v[:, :, 1:, :],
                target_tan_h[:, 1:, :, :],
                target_tan_v[:, :, :-1, :],
            )
            cross = torch.stack(
                (
                    spline_mask_h[:, 0, :-1, :] > 0.5,
                    spline_mask_v[:, 0, :, 1:] > 0.5,
                    spline_mask_h[:, 0, 1:, :] > 0.5,
                    spline_mask_v[:, 0, :, :-1] > 0.5,
                ), dim=-1,
            )
            count = cross.sum(dim=-1)
            ordinary = count == 2
            ambiguous = count == 4
            f00 = spline_control[:, 0, :-1, :-1]
            f10 = spline_control[:, 0, :-1, 1:]
            f01 = spline_control[:, 0, 1:, :-1]
            f11 = spline_control[:, 0, 1:, 1:]
            center = 0.25 * (f00 + f10 + f01 + f11)
            pair_a = ambiguous & ((f00 >= 0.0) == (center >= 0.0))
            pair_b = ambiguous & ~pair_a
            pair_ids = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            smooth_sum = predicted_sdf_pixels.new_zeros(())
            tangent_sum = predicted_sdf_pixels.new_zeros(())
            separation_sum = predicted_sdf_pixels.new_zeros(())
            smooth_count = predicted_sdf_pixels.new_zeros(())
            tangent_count = predicted_sdf_pixels.new_zeros(())
            for a_idx, b_idx in pair_ids:
                active = ordinary & cross[..., a_idx] & cross[..., b_idx]
                if (a_idx, b_idx) in ((0, 1), (2, 3)):
                    active = active | pair_a
                if (a_idx, b_idx) in ((0, 3), (1, 2)):
                    active = active | pair_b
                active_f = active.float()
                if active_f.numel() == 0:
                    continue
                delta_diff = F.smooth_l1_loss(
                    cell_disp[a_idx], cell_disp[b_idx], beta=0.10, reduction="none"
                ).sum(dim=-1)
                smooth_sum = smooth_sum + (delta_diff * active_f).sum()
                smooth_count = smooth_count + active_f.sum()

                teacher_dot = (
                    cell_target_tangent[a_idx] * cell_target_tangent[b_idx]
                ).sum(dim=-1).abs().clamp(0.0, 1.0)
                smooth_branch = active_f * (teacher_dot >= 0.90).float()
                pred_dot = (cell_tangent[a_idx] * cell_tangent[b_idx]).sum(dim=-1).abs().clamp(0.0, 1.0)
                tangent_sum = tangent_sum + ((1.0 - pred_dot) * smooth_branch).sum()
                tangent_count = tangent_count + smooth_branch.sum()

                pred_len = (cell_point[a_idx] - cell_point[b_idx]).norm(dim=-1) * control_spacing_hr
                teacher_len = (cell_target[a_idx] - cell_target[b_idx]).norm(dim=-1) * control_spacing_hr
                collapse = F.relu(0.40 * teacher_len.detach() - pred_len)
                separation_sum = separation_sum + (collapse * active_f).sum()

            losses["spline_graph_span_smoothness"] = smooth_sum / smooth_count.clamp_min(1.0)
            losses["spline_graph_span_tangent"] = tangent_sum / tangent_count.clamp_min(1.0)
            losses["spline_graph_span_separation"] = separation_sum / smooth_count.clamp_min(1.0)

            field_weight = (0.08 + 1.92 * torch.exp(-target_detached.abs() / 4.0)).detach()
            losses["spline_graph_sdf"] = self._weighted_mean(
                F.smooth_l1_loss(predicted_sdf_pixels, target_detached, beta=0.10, reduction="none"),
                field_weight,
            )
            sgx, sgy = sdf_gradient_components(predicted_sdf_pixels)
            losses["spline_graph_gradient"] = self._weighted_mean(
                F.smooth_l1_loss(sgx, tgx, beta=0.07, reduction="none")
                + F.smooth_l1_loss(sgy, tgy, beta=0.07, reduction="none"),
                field_weight,
            )
            snorm = torch.sqrt(sgx.square() + sgy.square() + 1.0e-8)
            losses["spline_graph_eikonal"] = self._weighted_mean((snorm - 1.0).abs(), field_weight)

            # V10.7.1 explicit signed-offset metric calibration.  Dense target-SDF
            # bands are equivalent to sampling the contour at +/-0.25, 0.5, 1.0,
            # 1.5 and 2.0 px, but retain a stable vectorised loss on every tile.
            abs_target = target_detached.abs()
            metric_offset_weight = torch.where(
                abs_target <= 0.35, torch.full_like(abs_target, 8.0),
                torch.where(abs_target <= 0.75, torch.full_like(abs_target, 6.0),
                torch.where(abs_target <= 1.25, torch.full_like(abs_target, 4.5),
                torch.where(abs_target <= 1.75, torch.full_like(abs_target, 3.0),
                torch.where(abs_target <= 2.25, torch.full_like(abs_target, 2.0),
                            torch.full_like(abs_target, 0.10)))))
            ).detach()
            losses["spline_metric_offset"] = self._weighted_mean(
                F.smooth_l1_loss(predicted_sdf_pixels, target_detached, beta=0.06, reduction="none"),
                metric_offset_weight,
            )
            near_eikonal_weight = torch.where(
                abs_target <= 0.50, torch.full_like(abs_target, 8.0),
                torch.where(abs_target <= 1.50, torch.full_like(abs_target, 4.0),
                torch.where(abs_target <= 3.00, torch.full_like(abs_target, 1.5),
                            torch.zeros_like(abs_target)))
            ).detach()
            losses["spline_metric_eikonal_near"] = self._weighted_mean(
                (snorm - 1.0).abs(), near_eikonal_weight
            )
            metric_scale = outputs.get("spline_metric_scale")
            metric_bias = outputs.get("spline_metric_bias_pixels")
            calibration_gate = outputs.get("spline_metric_calibration_gate")
            if metric_scale is not None and metric_bias is not None:
                reg_weight = calibration_gate.detach() if calibration_gate is not None else (abs_target <= 3.0).float().detach()
                losses["spline_metric_scale_regularization"] = self._weighted_mean(
                    (metric_scale.float() - 1.0).abs(), reg_weight
                )
                losses["spline_metric_bias_regularization"] = self._weighted_mean(
                    metric_bias.float().abs(), reg_weight
                )
            else:
                losses["spline_metric_scale_regularization"] = predicted_sdf_pixels.new_zeros(())
                losses["spline_metric_bias_regularization"] = predicted_sdf_pixels.new_zeros(())
            losses["spline_graph_curvature"] = self._weighted_mean(
                (self.laplacian_tensor(predicted_sdf_pixels) - self.laplacian_tensor(target_detached)).abs(),
                field_weight,
            )
            losses["spline_graph_topology_invariant"] = predicted_sdf_pixels.new_zeros(())
        else:
            zero_spline = predicted_sdf_pixels.new_zeros(())
            losses["spline_graph_topology_control"] = zero_spline
            losses["spline_graph_topology_sign"] = zero_spline
            losses["spline_graph_point"] = zero_spline
            losses["spline_graph_tangent"] = zero_spline
            losses["spline_graph_displacement"] = zero_spline
            losses["spline_graph_span_smoothness"] = zero_spline
            losses["spline_graph_span_tangent"] = zero_spline
            losses["spline_graph_span_separation"] = zero_spline
            losses["spline_graph_sdf"] = zero_spline
            losses["spline_graph_gradient"] = zero_spline
            losses["spline_graph_eikonal"] = zero_spline
            losses["spline_metric_offset"] = zero_spline
            losses["spline_metric_eikonal_near"] = zero_spline
            losses["spline_metric_scale_regularization"] = zero_spline
            losses["spline_metric_bias_regularization"] = zero_spline
            losses["spline_graph_curvature"] = zero_spline
            losses["spline_graph_topology_invariant"] = zero_spline

        # Historical V10.2 edge-crossing metric names remain zero-valued telemetry.
        zero_edge = predicted_sdf_pixels.new_zeros(())
        losses["edge_crossing_fraction"] = zero_edge
        losses["edge_crossing_displacement"] = zero_edge
        losses["edge_crossing_sdf"] = zero_edge
        losses["edge_crossing_gradient"] = zero_edge
        losses["edge_crossing_eikonal"] = zero_edge
        losses["edge_crossing_curvature"] = zero_edge
        losses["edge_crossing_topology_invariant"] = zero_edge

        # Historical V10.2 metric names remain zero-valued telemetry only.
        losses["topology_field_sdf"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_field_control"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_field_gradient"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_field_eikonal"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_field_curvature"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_field_crossing"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_field_sign_invariant"] = predicted_sdf_pixels.new_zeros(())
        losses["topology_saddle_projection_fraction"] = predicted_sdf_pixels.new_zeros(())

        # V9.9.3 local analytic primitive supervision. The renderer-facing zero set
        # is no longer an arbitrary pointwise delta field: a normal, anchor distance
        # and curvature define each local line/arc primitive. These losses teach the
        # compact parameters directly while the subpixel phi loss remains the final
        # metric authority.
        primitive_anchor = outputs.get("parametric_anchor_distance_pixels")
        if primitive_anchor is not None:
            target_anchor = F.interpolate(
                target_sdf_pixels.detach(), size=primitive_anchor.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            anchor_weight = (0.20 + 1.80 * torch.exp(-target_anchor.abs() / 4.0)).detach()
            losses["parametric_anchor"] = self._weighted_mean(
                F.smooth_l1_loss(primitive_anchor.float(), target_anchor, beta=0.12, reduction="none"),
                anchor_weight,
            )
        else:
            losses["parametric_anchor"] = predicted_sdf_pixels.new_zeros(())

        primitive_normal = outputs.get("primitive_normal")
        primitive_curvature = outputs.get("primitive_curvature_hr")
        if primitive_normal is not None and primitive_curvature is not None:
            tgt_gx, tgt_gy = sdf_gradient_components(target_sdf_pixels.detach())
            tgt_norm = torch.sqrt(tgt_gx.square() + tgt_gy.square() + 1.0e-6)
            tgt_nx, tgt_ny = tgt_gx / tgt_norm, tgt_gy / tgt_norm
            dot = (
                primitive_normal[:, 0:1].float() * tgt_nx
                + primitive_normal[:, 1:2].float() * tgt_ny
            ).abs().clamp(0.0, 1.0)
            primitive_weight = (
                0.20 + 1.80 * torch.exp(-target_sdf_pixels.detach().abs() / 4.0)
            ).detach()
            losses["parametric_normal"] = self._weighted_mean(1.0 - dot, primitive_weight)

            tnx_x, _tnx_y = sdf_gradient_components(tgt_nx)
            _tny_x, tny_y = sdf_gradient_components(tgt_ny)
            target_curvature = tnx_x + tny_y
            losses["parametric_curvature"] = self._weighted_mean(
                F.smooth_l1_loss(
                    primitive_curvature.float().abs(), target_curvature.detach().abs(),
                    beta=0.03, reduction="none",
                ),
                primitive_weight,
            )
        else:
            losses["parametric_normal"] = predicted_sdf_pixels.new_zeros(())
            losses["parametric_curvature"] = predicted_sdf_pixels.new_zeros(())

        primitive_delta = outputs.get("parametric_distance_delta_pixels")
        if primitive_delta is not None:
            pdx, pdy = sdf_gradient_components(primitive_delta.float())
            # First-order smoothness on the sparse control lattice is deliberately
            # stronger than a generic SDF Laplacian: it directly prevents adjacent
            # primitives from chasing different subpixel phases along one edge.
            losses["parametric_offset_smoothness"] = 0.5 * (
                pdx.abs().mean() + pdy.abs().mean()
            )
        else:
            losses["parametric_offset_smoothness"] = predicted_sdf_pixels.new_zeros(())

        # Smooth the *transport* in vector space instead of forcing a scalar offset
        # to be constant through the normal tube.  The old normal-consistency term
        # was physically wrong for translated thin stripes because opposite SDF
        # normals require opposite scalar offsets for the same translation.
        if predicted_transport_hr is not None:
            ptx_x, ptx_y = sdf_gradient_components(predicted_transport_hr[:, 0:1])
            pty_x, pty_y = sdf_gradient_components(predicted_transport_hr[:, 1:2])
            ttx_x, ttx_y = sdf_gradient_components(projected_transport_hr[:, 0:1].detach())
            tty_x, tty_y = sdf_gradient_components(projected_transport_hr[:, 1:2].detach())
            transport_grad_error = (
                F.smooth_l1_loss(ptx_x, ttx_x, beta=0.12, reduction="none")
                + F.smooth_l1_loss(ptx_y, ttx_y, beta=0.12, reduction="none")
                + F.smooth_l1_loss(pty_x, tty_x, beta=0.12, reduction="none")
                + F.smooth_l1_loss(pty_y, tty_y, beta=0.12, reduction="none")
            ) * 0.25
            losses["contour_transport_smoothness"] = self._weighted_mean(
                transport_grad_error,
                offset_weight * (0.25 + 0.75 * transport_confidence_hr),
            )
        else:
            losses["contour_transport_smoothness"] = predicted_sdf_pixels.new_zeros(())

        if predicted_dilation_hr is not None:
            dil_gx, dil_gy = sdf_gradient_components(predicted_dilation_hr)
            tgt_dil_gx, tgt_dil_gy = sdf_gradient_components(projected_dilation_hr.detach())
            losses["contour_dilation_smoothness"] = self._weighted_mean(
                0.5 * (
                    F.smooth_l1_loss(dil_gx, tgt_dil_gx, beta=0.10, reduction="none")
                    + F.smooth_l1_loss(dil_gy, tgt_dil_gy, beta=0.10, reduction="none")
                ),
                offset_weight,
            )
        else:
            losses["contour_dilation_smoothness"] = predicted_sdf_pixels.new_zeros(())

        losses["contour_offset_tangent"] = losses["contour_transport_smoothness"].detach()
        losses["contour_offset_normal_consistency"] = losses["contour_dilation_smoothness"].detach()
        losses["contour_offset_rms_pixels"] = torch.sqrt(
            self._weighted_mean(predicted_normal_offset_pixels.square(), offset_weight).clamp_min(0.0)
            + 1.0e-8
        ).detach()

        source_abs_error = (source_sdf_prior_pixels - target_sdf_pixels).abs().detach()
        predicted_abs_error = (predicted_sdf_pixels - target_sdf_pixels).abs()
        improvement_regret = F.relu(
            predicted_abs_error - source_abs_error - float(config.sdf_improvement_margin_pixels)
        )
        losses["sdf_improvement_regret"] = self._balanced_metric_band_mean(
            improvement_regret * need_weight, target_sdf_pixels, metric_band, near_pixels=3.0
        )
        losses["sdf_source_surface"] = self._balanced_metric_band_mean(
            source_abs_error, target_sdf_pixels, metric_band, near_pixels=3.0
        ).detach()
        losses["geometry_need_mean"] = self._mean_fp32(geometry_need.float()).detach()

        coarse_sdf_pixels = outputs.get("coarse_sdf_pixels", predicted_sdf_pixels).float()
        coarse_surface_error = F.smooth_l1_loss(
            coarse_sdf_pixels, target_sdf_pixels, beta=0.75, reduction="none"
        )
        losses["coarse_sdf_surface"] = self._balanced_metric_band_mean(
            coarse_surface_error, target_sdf_pixels, metric_band, near_pixels=3.0
        )
        losses["sdf_residual_l1"] = self._mean_fp32(
            outputs.get("sdf_residual_pixels", torch.zeros_like(predicted_sdf_pixels)).abs()
        )

        # Sign is evaluated only after the global gauge has been aligned. The far
        # field only needs the correct side and no extra zero-sets.
        target_inside = (target_sdf_pixels < 0.0).float()
        sign_error = F.binary_cross_entropy_with_logits(
            -predicted_sdf_pixels / 1.5, target_inside, reduction="none"
        )
        losses["sdf_sign"] = self._balanced_sign_mean(
            sign_error, target_sdf_pixels, metric_band, near_pixels=2.0
        )
        losses["sdf_topology_sign"] = self._topology_sign_margin_loss(
            predicted_sdf_pixels,
            target_sdf_pixels,
            margin_pixels=float(config.sdf_topology_margin_pixels),
            core_pixels=float(config.sdf_topology_core_pixels),
            band_pixels=float(config.sdf_topology_band_pixels),
            worst_fraction=float(config.sdf_topology_worst_fraction),
        )

        pred_pad = F.pad(predicted_sdf_pixels, (1, 1, 1, 1), mode="replicate")
        pred_gx = (pred_pad[:, :, 1:-1, 2:] - pred_pad[:, :, 1:-1, :-2]) * 0.5
        pred_gy = (pred_pad[:, :, 2:, 1:-1] - pred_pad[:, :, :-2, 1:-1]) * 0.5
        pred_grad_norm = torch.sqrt(pred_gx.square() + pred_gy.square() + 1.0e-6)
        metric_pred_pad = F.pad(
            metricized_predicted_sdf_pixels, (1, 1, 1, 1), mode="replicate"
        )
        metric_pred_gx = (
            metric_pred_pad[:, :, 1:-1, 2:] - metric_pred_pad[:, :, 1:-1, :-2]
        ) * 0.5
        metric_pred_gy = (
            metric_pred_pad[:, :, 2:, 1:-1] - metric_pred_pad[:, :, :-2, 1:-1]
        ) * 0.5
        metric_pred_grad_norm = torch.sqrt(
            metric_pred_gx.square() + metric_pred_gy.square() + 1.0e-6
        )
        metric_mask = (target_sdf_pixels.abs() <= metric_band).float().detach()
        near_metric_mask = (target_sdf_pixels.abs() <= 4.0).float().detach()
        # Raw Eikonal remains a training authority: eventually GeometryNet should
        # produce a true metric field even though the renderer can calibrate it.
        losses["sdf_eikonal"] = self._weighted_mean(
            (pred_grad_norm - 1.0).abs(), metric_mask
        )
        redistanced_eikonal = self._weighted_mean(
            (metric_pred_grad_norm - 1.0).abs(), metric_mask
        ).detach()
        redistanced_near_eikonal = self._weighted_mean(
            (metric_pred_grad_norm - 1.0).abs(), near_metric_mask
        ).detach()
        losses["sdf_redistanced_eikonal"] = redistanced_eikonal
        losses["sdf_redistanced_near_eikonal"] = redistanced_near_eikonal
        # Compatibility aliases for V9.8.4 telemetry readers.
        losses["sdf_metricized_eikonal"] = redistanced_eikonal
        losses["sdf_metricized_near_eikonal"] = redistanced_near_eikonal

        # Metric-gradient supervision is the bootstrap authority: unlike Eikonal it
        # supplies a direction even near the almost-flat initial field.
        target_pad = F.pad(target_sdf_pixels, (1, 1, 1, 1), mode="replicate")
        target_gx = (target_pad[:, :, 1:-1, 2:] - target_pad[:, :, 1:-1, :-2]) * 0.5
        target_gy = (target_pad[:, :, 2:, 1:-1] - target_pad[:, :, :-2, 1:-1]) * 0.5
        metric_gradient_error = (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
        losses["sdf_metric_gradient"] = self._balanced_metric_band_mean(
            metric_gradient_error, target_sdf_pixels, metric_band, near_pixels=3.0
        )

        pred_normal = torch.cat(
            (pred_gx / pred_grad_norm, pred_gy / pred_grad_norm), dim=1
        )
        target_tangent_unit = F.normalize(target_orientation.float(), dim=1, eps=1.0e-4)
        target_normal_unit = torch.cat(
            (-target_tangent_unit[:, 1:2], target_tangent_unit[:, 0:1]), dim=1
        )
        sdf_grad_alignment = 1.0 - (
            pred_normal * target_normal_unit
        ).sum(dim=1, keepdim=True).abs().clamp(0.0, 1.0)
        losses["sdf_gradient_alignment"] = self._weighted_mean(
            sdf_grad_alignment, (metric_mask * (0.25 + near_contour.float() * 1.75)).detach()
        )

        pred_sdf_curvature = self.laplacian_tensor(outputs["sdf"].float())
        target_sdf_curvature = self.laplacian_tensor(target_sdf_aligned).detach()
        losses["sdf_curvature"] = self._weighted_mean(
            (pred_sdf_curvature - target_sdf_curvature).abs(), contour_weight
        )

        zero_band = max(0.10, float(config.sdf_zero_band_pixels))
        zero_distance = raw_target_sdf_pixels.abs()
        zero_mask = torch.where(
            zero_distance <= zero_band,
            torch.exp(-0.5 * (zero_distance / max(zero_band * 0.55, 1.0e-4)).square()),
            torch.zeros_like(zero_distance),
        ).detach()
        raw_zero_sq = predicted_sdf_pixels.square()
        metric_zero_sq = metricized_predicted_sdf_pixels.square()
        losses["boundary_sdf_zero"] = self._weighted_mean(
            F.smooth_l1_loss(
                metricized_predicted_sdf_pixels,
                torch.zeros_like(metricized_predicted_sdf_pixels),
                beta=0.25, reduction="none"
            ),
            zero_mask * geometry_boost.float(),
        )
        # Dense differentiable zero-set transport. Redistance is intentionally
        # scale-invariant, but its crossing mask cannot create a missing crossing:
        # when no sign-changing edge exists near the target, the propagated distance
        # is a constant saturated field and supplies no gradient to move phi through
        # zero. This raw gauge-fixing term provides that missing transport gradient.
        losses["boundary_sdf_raw_zero"] = self._weighted_mean(
            F.smooth_l1_loss(
                predicted_sdf_pixels,
                torch.zeros_like(predicted_sdf_pixels),
                beta=0.25, reduction="none"
            ),
            zero_mask * geometry_boost.float(),
        )

        # Collapse/position telemetry. ``sdf_zero_rms_pixels`` now means actual
        # redistanced pixel displacement at the target contour; keep the raw value
        # separately so field compression can never masquerade as accurate placement.
        losses["sdf_raw_zero_rms_levelset_units"] = torch.sqrt(
            self._weighted_mean(raw_zero_sq, zero_mask).clamp_min(0.0) + 1.0e-12
        )
        losses["sdf_zero_rms_pixels"] = torch.sqrt(
            self._weighted_mean(metric_zero_sq, zero_mask).clamp_min(0.0) + 1.0e-12
        )
        losses["sdf_grad_norm_mean"] = self._weighted_mean(pred_grad_norm, metric_mask)
        redistanced_grad_norm_mean = self._weighted_mean(
            metric_pred_grad_norm, metric_mask
        ).detach()
        losses["sdf_redistanced_grad_norm_mean"] = redistanced_grad_norm_mean
        losses["sdf_metricized_grad_norm_mean"] = redistanced_grad_norm_mean
        losses["sdf_metricization_denominator_mean"] = self._weighted_mean(
            metricization_denominator, near_metric_mask
        ).detach()
        losses["sdf_positive_fraction"] = self._mean_fp32((predicted_sdf_pixels > 0.0).float())
        losses["sdf_negative_fraction"] = self._mean_fp32((predicted_sdf_pixels < 0.0).float())
        losses["sdf_polarity_positive_fraction"] = self._mean_fp32((sdf_polarity > 0.0).float())

        predicted_edge_probability = torch.sigmoid(outputs["edge_logits"].float())
        sdf_edge_probability = torch.exp(
            -predicted_sdf_pixels.abs() / 0.85
        ).clamp(0.0, 1.0)
        losses["edge_sdf_consistency"] = self._weighted_mean(
            (predicted_edge_probability - sdf_edge_probability).abs(),
            (0.05 + target_edge.float() * 2.0).detach(),
        )

        edge_loss = F.binary_cross_entropy_with_logits(
            outputs["edge_logits"].float(),
            target_edge.float(),
            reduction="none",
        )
        edge_class_weight = (
            0.02 + target_edge.float() * 5.0
        )
        losses["edge"] = self._weighted_mean(
            edge_loss * geometry_boost.float(),
            edge_class_weight.detach(),
        )

        pred_orientation_axial = self._axial_encoding(outputs["orientation"])
        target_orientation_axial = self._axial_encoding(target_orientation)
        orientation_cosine = 1.0 - (
            pred_orientation_axial * target_orientation_axial
        ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        losses["orientation"] = self._weighted_mean(
            orientation_cosine, contour_weight
        )

        # Hardness is a boundary-profile property, not a colour residual. Synthetic
        # analytic edges are deliberately hard; real targets retain softer profiles
        # when their fine/coarse gradient concentration is low.
        target_gray = target_albedo.float().mean(dim=1, keepdim=True)
        target_gradient = self.gradient_magnitude(target_gray)
        coarse_gradient = F.avg_pool2d(
            target_gradient, kernel_size=5, stride=1, padding=2
        )
        concentration = (
            target_gradient / (coarse_gradient * 2.0 + 1.0e-4)
        ).clamp(0.0, 1.0)
        target_hardness = (
            0.55 + 0.45 * concentration
        ).clamp(0.0, 1.0)
        target_hardness = (
            target_hardness * (1.0 - geometry_exact)
            + torch.ones_like(target_hardness) * geometry_exact
        )
        hardness_bce = F.binary_cross_entropy_with_logits(
            outputs["hardness_logits"].float(),
            target_hardness.float(),
            reduction="none",
        )
        losses["hardness"] = self._weighted_mean(
            hardness_bce,
            (0.15 + near_contour.float() * 1.85).detach(),
        )

        # Geometry is judged on the actual implicit-boundary renderer, never on an
        # detail residual. The same SDF/gate/coverage is applied to every map.
        reconstructed_albedo = outputs["boundary_reconstructed_albedo"]
        candidate_albedo = outputs.get("boundary_candidate_albedo", reconstructed_albedo)
        reconstructed_normal = outputs["boundary_reconstructed_normal"]
        reconstructed_material = outputs["boundary_reconstructed_material"]
        baseline_albedo = outputs["baseline_albedo"]
        baseline_normal = outputs["baseline_normal"]
        seam_source_albedo = outputs.get("boundary_pre_seam_albedo", baseline_albedo)
        seam_source_normal = outputs.get("boundary_pre_seam_normal", baseline_normal)
        seam_source_material = outputs.get("boundary_pre_seam_material", outputs["baseline_material"])

        # V10.8.8.1: bind seam authority before the B3 teacher override below.
        # V10.8.6 introduced the direct phase-SR B3 path but the variable was
        # still assigned later in the directional-loss block, causing an
        # UnboundLocalError on the first real Raven seam-proof batch.
        seam_authority = outputs.get("seam_authority")

        # V10.7.1 seam teacher. Positive authority is derived from authored HR
        # structure that is missing or damaged in the bicubic source. This prevents
        # the old regularisation-only optimum where the safest detector response was
        # to close authority everywhere.
        target_material_for_seam = torch.cat((
            ((target_material_class.float().unsqueeze(1) + 0.5) / max(float(config.material_classes), 1.0)).clamp(0.0, 1.0),
            target_emissive.float().clamp(0.0, 1.0),
            target_roughness.float().clamp(0.0, 1.0),
        ), dim=1)
        target_ridge = multi_map_ridge_response(target_albedo, target_normal, target_material_for_seam).detach()
        baseline_ridge = multi_map_ridge_response(seam_source_albedo, seam_source_normal, seam_source_material).detach()
        pixel_need = (target_albedo.float() - seam_source_albedo.float()).abs().mean(dim=1, keepdim=True)
        pixel_need = (pixel_need * float(getattr(config, "seam_missing_detail_scale", 8.0))).clamp(0.0, 1.0)
        ridge_need = (target_ridge - baseline_ridge).clamp(0.0, 1.0)
        seam_teacher = torch.maximum(
            target_edge.float().clamp(0.0, 1.0) * (0.25 + 0.75 * pixel_need),
            ridge_need * (0.35 + 0.65 * pixel_need),
        ).clamp(0.0, 1.0).detach()
        seam_teacher_radius = int(getattr(config, "seam_teacher_dilation_pixels", 2))
        if seam_teacher_radius > 0:
            seam_teacher = F.max_pool2d(
                seam_teacher, kernel_size=seam_teacher_radius * 2 + 1, stride=1, padding=seam_teacher_radius
            ).detach()
        if phase == "seam-proof" and seam_authority is not None:
            # The B3 representation proof must be scored on exactly the region it
            # is allowed to edit. V10.8.4 used a broader ridge-weighted metric than
            # its edge-only forced authority, making 70% recovery partly impossible.
            seam_teacher = seam_authority.detach().float().clamp(0.0, 1.0)
        losses["seam_teacher_mean"] = seam_teacher.mean().detach()

        # V10.7.1 manufactured-seam objective.  The target orientation is a tangent
        # field; straight/curved coherent seams should vary little *along* the seam
        # while matching the authored transition across its normal.  This explicitly
        # penalises the saw-tooth phase pattern that can have acceptable pixel MAE.
        seam_tangent = outputs.get("seam_tangent")
        seam_coherence = outputs.get("seam_coherence")
        if seam_tangent is not None and seam_authority is not None and seam_coherence is not None:
            target_tangent = F.normalize(target_orientation.float(), dim=1, eps=1.0e-5)
            seam_tangent_unit = F.normalize(seam_tangent.float(), dim=1, eps=1.0e-5)
            target_axial = self._axial_encoding(target_tangent)
            seam_axial = self._axial_encoding(seam_tangent_unit)
            seam_mask = (0.05 + target_edge.float() * 1.95) * (0.25 + seam_coherence.float().detach() * 0.75)
            losses["seam_directional"] = self._weighted_mean(
                1.0 - (target_axial * seam_axial).sum(dim=1, keepdim=True).clamp(-1.0, 1.0), seam_mask.detach()
            )
            pred_t, pred_n = self._directional_derivatives(reconstructed_albedo, target_tangent)
            tgt_t, tgt_n = self._directional_derivatives(target_albedo, target_tangent)
            losses["seam_tangent_smoothness"] = self._weighted_mean(
                (pred_t - tgt_t).abs(), seam_mask.detach()
            )
            losses["seam_normal_profile"] = self._weighted_mean(
                (pred_n.abs() - tgt_n.abs()).abs(), seam_mask.detach()
            )
            off_seam = (1.0 - target_edge.float()).clamp(0.0, 1.0)
            losses["seam_authority_regularization"] = self._weighted_mean(
                seam_authority.float(), (0.10 + off_seam).detach()
            )
            learned_authority = outputs.get("seam_learned_authority", seam_authority).float().clamp(1.0e-5, 1.0 - 1.0e-5)
            losses["seam_authority_teacher"] = F.binary_cross_entropy(learned_authority, seam_teacher)
            seam_weight = (0.05 + seam_teacher * 1.95).detach()
            losses["seam_reconstruction"] = self._weighted_mean(
                (reconstructed_albedo.float() - target_albedo.float()).abs(), seam_weight
            )
            phase_delta = outputs.get("seam_phase_delta")
            if phase_delta is not None:
                # V10.8.8: supervise the actual 8-channel 4x residual directly.
                # This removes the previous indirect-gradient ambiguity where B3
                # could spend its tiny budget tuning proposal mixing while the
                # zero-initialised phase-SR head remained effectively identity.
                target_delta = torch.cat((
                    target_albedo.float() - seam_source_albedo.float(),
                    target_normal.float() - seam_source_normal.float(),
                    target_material_for_seam.float() - seam_source_material.float(),
                ), dim=1)
                phase_weight = (0.05 + seam_teacher * 1.95).detach().expand_as(target_delta)
                losses["seam_phase_residual"] = self._weighted_mean(
                    (phase_delta.float() - target_delta).abs(), phase_weight
                )
            else:
                losses["seam_phase_residual"] = zero
            with torch.no_grad():
                base_err = self._weighted_mean((seam_source_albedo.float() - target_albedo.float()).abs(), seam_weight)
                cand_err = self._weighted_mean((reconstructed_albedo.float() - target_albedo.float()).abs(), seam_weight)
                losses["seam_recovery"] = ((base_err - cand_err) / base_err.clamp_min(1.0e-6)).detach()
                pred_mask = (learned_authority > 0.5).float()
                true_mask = (seam_teacher > 0.35).float()
                inter = (pred_mask * true_mask).sum()
                union = ((pred_mask + true_mask) > 0.0).float().sum().clamp_min(1.0)
                losses["seam_authority_iou"] = (inter / union).detach()
            if phase in {"sdf-proof", "seam-proof"}:
                losses["seam_projected_view"] = self._projected_view_seam_loss(
                    reconstructed_albedo, target_albedo, target_edge
                )
            else:
                losses["seam_projected_view"] = zero
        else:
            losses["seam_directional"] = zero
            losses["seam_tangent_smoothness"] = zero
            losses["seam_normal_profile"] = zero
            losses["seam_authority_regularization"] = zero
            losses["seam_authority_teacher"] = zero
            losses["seam_reconstruction"] = zero
            losses["seam_phase_residual"] = zero
            losses["seam_recovery"] = zero.detach()
            losses["seam_authority_iou"] = zero.detach()
            losses["seam_projected_view"] = zero

        # V9.9.3 Panel-2 teacher.  training.py supplies a detached render made by
        # the same BoundaryRenderer using aligned GT SDF + forced gate + forced hard
        # profile.  Stage-B therefore optimises Panel 3 directly toward Panel 2.
        teacher_albedo = outputs.get("sdf_teacher_boundary_albedo")
        teacher_normal = outputs.get("sdf_teacher_boundary_normal")
        teacher_material = outputs.get("sdf_teacher_boundary_material")
        if teacher_albedo is not None:
            teacher_albedo = teacher_albedo.detach().float()
            teacher_normal = teacher_normal.detach().float() if teacher_normal is not None else reconstructed_normal.detach().float()
            teacher_material = teacher_material.detach().float() if teacher_material is not None else reconstructed_material.detach().float()
            teacher_band = (
                0.08 + torch.exp(-target_sdf_pixels.abs() / 3.0) * (1.70 + geometry_exact.float())
            ).detach()
            teacher_albedo_error = (
                reconstructed_albedo.float() - teacher_albedo
            ).abs().mean(dim=1, keepdim=True)
            teacher_normal_error = (
                reconstructed_normal.float() - teacher_normal
            ).abs().mean(dim=1, keepdim=True)
            teacher_material_error = (
                reconstructed_material.float() - teacher_material
            ).abs().mean(dim=1, keepdim=True)
            losses["sdf_teacher_render"] = self._weighted_mean(
                teacher_albedo_error + teacher_normal_error * 0.35 + teacher_material_error * 0.20,
                teacher_band,
            )
            teacher_gray = teacher_albedo.mean(dim=1, keepdim=True)
            recon_gray_teacher = reconstructed_albedo.float().mean(dim=1, keepdim=True)
            teacher_grad = self.gradient_magnitude(teacher_gray)
            recon_grad_teacher = self.gradient_magnitude(recon_gray_teacher)
            losses["sdf_teacher_gradient"] = self._weighted_mean(
                (recon_grad_teacher - teacher_grad).abs(),
                teacher_band,
            )
            teacher_edge_proxy = self._normalise_edge(teacher_grad)
            recon_teacher_edge_proxy = self._normalise_edge(recon_grad_teacher)
            losses["sdf_teacher_profile"] = self._weighted_mean(
                (recon_teacher_edge_proxy - teacher_edge_proxy).abs(),
                teacher_band,
            )
            # Direct profile-shape constraints. Width and core/halo distribution are
            # properties of the rendered output, so Panel 3 cannot pass merely by
            # putting a zero crossing at the right centreline.
            target_distance = target_sdf_pixels.abs().detach()
            profile_band = (target_distance <= 6.0).float().detach()
            teacher_energy = teacher_grad * profile_band
            recon_energy = recon_grad_teacher * profile_band
            teacher_energy_sum = teacher_energy.flatten(1).sum(dim=1).clamp_min(1.0e-6)
            recon_energy_sum = recon_energy.flatten(1).sum(dim=1).clamp_min(1.0e-6)
            teacher_width = torch.sqrt(
                (teacher_energy * target_distance.square()).flatten(1).sum(dim=1) / teacher_energy_sum
            )
            recon_width = torch.sqrt(
                (recon_energy * target_distance.square()).flatten(1).sum(dim=1) / recon_energy_sum
            )
            losses["sdf_teacher_width"] = ((recon_width - teacher_width).abs() / teacher_width.clamp_min(0.10)).mean()
            teacher_peak = (teacher_grad * (target_distance <= 1.0).float()).flatten(1).amax(dim=1)
            recon_peak = (recon_grad_teacher * (target_distance <= 1.0).float()).flatten(1).amax(dim=1)
            losses["sdf_teacher_peak"] = ((recon_peak - teacher_peak).abs() / teacher_peak.clamp_min(1.0e-3)).mean()
            halo_band = ((target_distance > 1.5) & (target_distance <= 6.0)).float().detach()
            teacher_halo = (teacher_grad * halo_band).flatten(1).sum(dim=1) / teacher_energy_sum
            recon_halo = (recon_grad_teacher * halo_band).flatten(1).sum(dim=1) / recon_energy_sum
            losses["sdf_teacher_halo"] = (recon_halo - teacher_halo).abs().mean()
            t = teacher_grad * profile_band
            r = recon_grad_teacher * profile_band
            t_mean = t.flatten(1).sum(dim=1, keepdim=True) / profile_band.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
            r_mean = r.flatten(1).sum(dim=1, keepdim=True) / profile_band.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
            tc = (t.flatten(1) - t_mean) * profile_band.flatten(1)
            rc = (r.flatten(1) - r_mean) * profile_band.flatten(1)
            corr = (tc * rc).sum(dim=1) / torch.sqrt(
                tc.square().sum(dim=1).clamp_min(1.0e-8) * rc.square().sum(dim=1).clamp_min(1.0e-8)
            )
            losses["sdf_teacher_profile_correlation"] = corr.mean().detach()
            losses["sdf_teacher_profile_correlation_loss"] = (1.0 - corr.clamp(-1.0, 1.0)).mean()
            baseline_teacher_error = self._weighted_mean(
                (baseline_albedo.float() - teacher_albedo).abs().mean(dim=1, keepdim=True),
                teacher_band,
            )
            predicted_teacher_error = self._weighted_mean(
                teacher_albedo_error,
                teacher_band,
            )
            losses["sdf_teacher_baseline_mae"] = baseline_teacher_error.detach()
            losses["sdf_teacher_predicted_mae"] = predicted_teacher_error.detach()
            losses["sdf_teacher_recovery"] = (
                (baseline_teacher_error - predicted_teacher_error)
                / baseline_teacher_error.clamp_min(1.0e-6)
            ).detach()
        else:
            zero = predicted_sdf_pixels.new_zeros(())
            losses["sdf_teacher_render"] = zero
            losses["sdf_teacher_gradient"] = zero
            losses["sdf_teacher_profile"] = zero
            losses["sdf_teacher_width"] = zero
            losses["sdf_teacher_peak"] = zero
            losses["sdf_teacher_halo"] = zero
            losses["sdf_teacher_profile_correlation"] = zero.detach()
            losses["sdf_teacher_profile_correlation_loss"] = zero
            losses["sdf_teacher_baseline_mae"] = zero.detach()
            losses["sdf_teacher_predicted_mae"] = zero.detach()
            losses["sdf_teacher_recovery"] = zero.detach()

        # V9.9.3 direct boundary-profile specialist. It cannot paint RGB; it
        # predicts the shared coverage profile in logit-correction space. Training
        # is anchored directly to the exact Panel-2 coverage and its spatial
        # derivative/profile moment, so a visually soft LR-like edge cannot satisfy
        # the specialist objective merely by moving the contour centre.
        refined_coverage = outputs.get("boundary_refined_coverage")
        initial_coverage = outputs.get("boundary_initial_coverage")
        if refined_coverage is not None:
            teacher_coverage = outputs.get("sdf_teacher_coverage_negative")
            if teacher_coverage is not None:
                target_cov = teacher_coverage.float().detach().clamp(0.0, 1.0)
            else:
                width = max(float(config.boundary_renderer_hard_width_pixels), 0.20)
                target_cov = (0.5 - target_sdf_pixels.float() / width).clamp(0.0, 1.0)
                target_cov = target_cov * target_cov * (3.0 - 2.0 * target_cov)
            pred_cov = refined_coverage.float().clamp(1.0e-5, 1.0 - 1.0e-5)
            specialist_band = (
                0.02
                + torch.exp(
                    -target_sdf_pixels.abs()
                    / max(float(config.boundary_specialist_band_pixels), 0.5)
                ) * 1.98
            ).detach()
            losses["boundary_specialist_coverage"] = self._weighted_mean(
                (pred_cov - target_cov).abs(), specialist_band
            )
            bce = F.binary_cross_entropy(pred_cov, target_cov, reduction="none")
            losses["boundary_specialist_coverage_bce"] = self._weighted_mean(bce, specialist_band)
            pred_gx, pred_gy = sobel_tensor(pred_cov)
            target_gx, target_gy = sobel_tensor(target_cov)
            losses["boundary_specialist_coverage_gradient"] = self._weighted_mean(
                (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs(),
                specialist_band,
            )
            pred_width = self._coverage_profile_width(pred_cov, target_sdf_pixels, band_pixels=6.0)
            target_width = self._coverage_profile_width(target_cov, target_sdf_pixels, band_pixels=6.0).detach()
            losses["boundary_specialist_profile_moment"] = (pred_width - target_width).abs().mean()
            losses["boundary_specialist_profile_width"] = pred_width.mean().detach()
            losses["boundary_specialist_teacher_profile_width"] = target_width.mean().detach()
            if initial_coverage is not None:
                initial_error = self._weighted_mean(
                    (initial_coverage.float() - target_cov).abs(), specialist_band
                )
                refined_error = losses["boundary_specialist_coverage"]
                losses["boundary_specialist_recovery"] = (
                    (initial_error - refined_error) / initial_error.clamp_min(1.0e-6)
                ).detach()
            else:
                losses["boundary_specialist_recovery"] = zero.detach()
            losses["boundary_specialist_gradient"] = losses["sdf_teacher_gradient"]
            losses["boundary_specialist_profile"] = losses["sdf_teacher_profile"]
        else:
            losses["boundary_specialist_coverage"] = zero
            losses["boundary_specialist_coverage_bce"] = zero
            losses["boundary_specialist_coverage_gradient"] = zero
            losses["boundary_specialist_profile_moment"] = zero
            losses["boundary_specialist_profile_width"] = zero.detach()
            losses["boundary_specialist_teacher_profile_width"] = zero.detach()
            losses["boundary_specialist_gradient"] = zero
            losses["boundary_specialist_profile"] = zero
            losses["boundary_specialist_recovery"] = zero.detach()

        reconstructed_tangent, reconstructed_edge_magnitude = self._image_tangent(
            reconstructed_albedo
        )
        baseline_tangent, baseline_edge_magnitude = self._image_tangent(
            baseline_albedo
        )
        reconstructed_tangent_axial = self._axial_encoding(reconstructed_tangent)
        target_orientation_geometry = target_orientation_axial.float()
        contour_weight_geometry = contour_weight.float()

        image_alignment = 1.0 - (
            reconstructed_tangent_axial * target_orientation_geometry
        ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        edge_presence = (
            0.35 + reconstructed_edge_magnitude.detach().clamp(0.0, 1.0)
        ).clamp(0.35, 1.35)
        losses["geometric_alignment"] = self._weighted_mean(
            image_alignment,
            contour_weight_geometry * edge_presence,
        )

        reconstructed_tangent_variation = self._field_variation(
            reconstructed_tangent_axial
        )
        target_tangent_variation = self._field_variation(
            target_orientation_geometry
        ).detach()
        tangent_excess = F.relu(
            reconstructed_tangent_variation
            - target_tangent_variation
            - float(config.tangent_variation_margin)
        )
        losses["tangent_coherence"] = self._weighted_mean(
            tangent_excess, contour_weight_geometry
        )

        reconstructed_curvature_change = self.laplacian_tensor(
            reconstructed_tangent_axial
        ).abs().mean(dim=1, keepdim=True)
        target_curvature_change = self.laplacian_tensor(
            target_orientation_geometry
        ).abs().mean(dim=1, keepdim=True).detach()
        curvature_excess = F.relu(
            reconstructed_curvature_change
            - target_curvature_change
            - float(config.curvature_variation_margin)
        )
        losses["curvature_coherence"] = self._weighted_mean(
            curvature_excess, contour_weight_geometry
        )

        boundary_photo_weight = (
            0.08 + near_contour.float() * (1.65 + geometry_exact.float())
        ).detach()
        losses["boundary_photometric"] = self._weighted_mean(
            (
                reconstructed_albedo.float()
                - target_albedo.float()
            ).abs().mean(dim=1, keepdim=True),
            boundary_photo_weight,
        )
        # Legacy diagnostic name now points to the renderer result.
        losses["geometry_photometric"] = losses["boundary_photometric"]
        baseline_boundary_mae = self._weighted_mean(
            (baseline_albedo.float() - target_albedo.float()).abs().mean(dim=1, keepdim=True),
            boundary_photo_weight,
        )
        stageb_boundary_mae = losses["boundary_photometric"]
        losses["sdf_stageb_baseline_mae"] = baseline_boundary_mae.detach()
        losses["sdf_stageb_renderer_mae"] = stageb_boundary_mae.detach()
        losses["sdf_stageb_renderer_improvement"] = (
            (baseline_boundary_mae - stageb_boundary_mae)
            / baseline_boundary_mae.clamp_min(1.0e-6)
        ).detach()

        target_edge_proxy = self._normalise_edge(
            self.gradient_magnitude(target_gray)
        )
        reconstructed_edge_proxy = self._normalise_edge(
            reconstructed_edge_magnitude
        )
        baseline_edge_proxy = self._normalise_edge(
            baseline_edge_magnitude.detach()
        )
        profile_weight = (
            0.10 + near_contour.float() * 2.40
        ).detach()
        losses["boundary_profile"] = self._weighted_mean(
            (reconstructed_edge_proxy - target_edge_proxy).abs(),
            profile_weight,
        )


        # Explicit hard-edge fuzz objective.  For exact synthetic geometry the edge
        # energy should live in a narrow band around the analytic zero-set rather
        # than forming a multi-pixel pale outline.
        abs_target_sdf_pixels = target_sdf_pixels.abs()
        hard_core = (abs_target_sdf_pixels <= 1.35).float()
        outer_band = (
            (abs_target_sdf_pixels > 1.35) & (abs_target_sdf_pixels <= 5.0)
        ).float()
        recon_grad = self.gradient_magnitude(
            reconstructed_albedo.float().mean(dim=1, keepdim=True)
        )
        target_grad = self.gradient_magnitude(target_gray)
        recon_outer = self._sum_fp32(recon_grad * outer_band * geometry_exact.float())
        recon_total = self._sum_fp32(recon_grad * (hard_core + outer_band) * geometry_exact.float()).clamp_min(1.0e-6)
        target_outer = self._sum_fp32(target_grad * outer_band * geometry_exact.float())
        target_total = self._sum_fp32(target_grad * (hard_core + outer_band) * geometry_exact.float()).clamp_min(1.0e-6)
        losses["boundary_fuzz"] = F.relu(
            recon_outer / recon_total - target_outer / target_total - 0.015
        )

        # Halo/overshoot objective.  The exact synthetic target defines the valid
        # local material range.  Reconstruction is penalised if it creates a bright
        # or dark fringe outside those two plateaus.
        target_local_max = F.max_pool2d(target_albedo.float(), 9, 1, 4)
        target_local_min = -F.max_pool2d(-target_albedo.float(), 9, 1, 4)
        halo_excess = (
            F.relu(reconstructed_albedo.float() - target_local_max - (1.0 / 255.0))
            + F.relu(target_local_min - reconstructed_albedo.float() - (1.0 / 255.0))
        ).mean(dim=1, keepdim=True)
        losses["boundary_halo"] = self._weighted_mean(
            halo_excess,
            (near_contour.float() * geometry_exact.float()).detach(),
        )

        boundary_gate_applied = outputs["boundary_gate"].float().clamp(0.0, 1.0)
        boundary_gate_prediction = outputs.get(
            "boundary_gate_prediction", outputs["boundary_gate"]
        ).float().clamp(0.0, 1.0)
        boundary_gate_probability = outputs.get(
            "boundary_gate_probability", boundary_gate_prediction
        ).float().clamp(1.0e-5, 1.0 - 1.0e-5)

        target_edge_band = F.max_pool2d(
            target_edge.float(),
            kernel_size=9,
            stride=1,
            padding=4,
        )
        target_edge_band = torch.maximum(
            target_edge_band,
            (near_contour.float() * 0.85).clamp(0.0, 1.0),
        )

        # Only request renderer authority where the deterministic baseline actually
        # needs help. V9.5 trained the gate to activate on nearly every edge, which
        # produced real-Raven topology regressions even on already-good contours.
        baseline_photo_need = (
            (baseline_albedo.float() - target_albedo.float())
            .abs()
            .mean(dim=1, keepdim=True)
            / float(config.boundary_gate_need_scale)
        ).clamp(0.0, 1.0)
        baseline_edge_need = (
            (baseline_edge_proxy - target_edge_proxy).abs() / 0.35
        ).clamp(0.0, 1.0)
        benefit_need = torch.maximum(baseline_photo_need, baseline_edge_need)

        # V9.9.3 gate authority is based on *realised LR->HR benefit*, not merely
        # the existence of an edge. The BoundaryRenderer exposes its ungated
        # candidate, so the gate can learn to use a geometric correction where it
        # actually beats bicubic LR and suppress it where the current SDF/profile
        # would make the target worse. This is the same base-vs-after comparison
        # used by the final quality contract, but differentiable only through the
        # gate head (candidate benefit is detached).
        baseline_candidate_error = (
            baseline_albedo.float() - target_albedo.float()
        ).abs().mean(dim=1, keepdim=True)
        candidate_error = (
            candidate_albedo.float() - target_albedo.float()
        ).abs().mean(dim=1, keepdim=True)
        candidate_gain = baseline_candidate_error - candidate_error
        candidate_benefit = (
            candidate_gain / max(float(config.boundary_gate_need_scale), 1.0e-4)
        ).clamp(0.0, 1.0).detach()
        candidate_safe = (
            candidate_error <= baseline_candidate_error + (1.0 / 255.0)
        ).float().detach()
        losses["boundary_candidate_gain"] = self._weighted_mean(
            candidate_gain.detach(), (0.05 + target_edge_band).detach()
        )
        losses["boundary_candidate_win_fraction"] = self._weighted_mean(
            candidate_safe, (0.05 + target_edge_band).detach()
        ).detach()

        exact_floor = (
            geometry_exact.float()
            * float(config.boundary_gate_exact_floor)
            * candidate_safe
        )
        realised_need = torch.maximum(benefit_need * candidate_benefit, exact_floor)
        gate_target = (
            target_edge_band * realised_need
        ).clamp(0.0, 1.0).detach()

        gate_error = (boundary_gate_probability - gate_target).abs()
        edge_region_weight = (0.05 + target_edge_band * 0.95).detach()
        flat_region_weight = (1.0 - target_edge_band).clamp(0.0, 1.0).detach()
        # Balance contour and flat regions independently. A global pixel mean lets
        # the much larger flat area win the gate bias gradient and recreates the
        # V9.5 inactive-gate failure even when edge targets are correct.
        gate_edge_loss = self._weighted_mean(gate_error, edge_region_weight)
        gate_flat_loss = self._weighted_mean(boundary_gate_probability, flat_region_weight)
        losses["boundary_gate"] = gate_edge_loss + gate_flat_loss * 0.15
        losses["boundary_off_contour"] = gate_flat_loss

        losses["boundary_gate_edge_mean"] = self._weighted_mean(
            boundary_gate_prediction.detach(),
            (0.05 + near_contour.float()).detach(),
        )
        losses["boundary_gate_flat_mean"] = self._weighted_mean(
            boundary_gate_prediction.detach(),
            (1.0 - near_contour.float()).clamp(0.0, 1.0).detach(),
        )
        losses["boundary_gate_probability_edge_mean"] = self._weighted_mean(
            boundary_gate_probability.detach(),
            (0.05 + near_contour.float()).detach(),
        )
        losses["boundary_gate_probability_flat_mean"] = self._weighted_mean(
            boundary_gate_probability.detach(),
            (1.0 - near_contour.float()).clamp(0.0, 1.0).detach(),
        )
        losses["boundary_gate_applied_edge_mean"] = self._weighted_mean(
            boundary_gate_applied.detach(),
            (0.05 + near_contour.float()).detach(),
        )
        losses["boundary_hardness_mean"] = self._weighted_mean(
            outputs["hardness"].detach().float(),
            (0.05 + near_contour.float()).detach(),
        )
        losses["boundary_transition_width_mean"] = self._weighted_mean(
            outputs["transition_width"].detach().float(),
            (0.05 + near_contour.float()).detach(),
        )
        losses["boundary_delta_rms"] = torch.sqrt(
            self._mean_fp32(
                (
                    reconstructed_albedo.float()
                    - baseline_albedo.float()
                ).square()
            )
        ).detach()

        # Edge-space before/after regret: an edge may only be reconstructed if the
        # actual rendered output improves against the target.
        reconstructed_geometry_error = (
            reconstructed_edge_proxy - target_edge_proxy
        ).abs()
        baseline_geometry_error = (
            baseline_edge_proxy - target_edge_proxy
        ).abs()
        patch = config.local_regret_patch
        reconstructed_geometry_local = self._local_scalar_error(
            reconstructed_geometry_error, patch
        )
        baseline_geometry_local = self._local_scalar_error(
            baseline_geometry_error, patch
        )
        regret_contour = F.avg_pool2d(
            near_contour.float(),
            kernel_size=patch,
            stride=patch,
        )
        regret_protection = (
            1.0
            + float(config.edge_regret_multiplier) * regret_contour
        )
        geometry_regret_map = F.relu(
            reconstructed_geometry_local
            - baseline_geometry_local
            + float(config.geometry_regret_margin)
        )
        losses["geometry_regret"] = self._mean_fp32(
            geometry_regret_map * regret_protection
        )
        # Pixel-level regret catches fragmented/wavy regressions that can disappear
        # inside the older 8x8 local average.
        pixel_regret = F.relu(
            reconstructed_geometry_error
            - baseline_geometry_error
            + float(config.geometry_regret_margin)
        )
        losses["boundary_pixel_regret"] = self._weighted_mean(
            pixel_regret,
            (0.10 + near_contour.float() * (1.0 + float(config.edge_regret_multiplier))).detach(),
        )
        losses["regret"] = losses["geometry_regret"]
        losses["improvement_fraction"] = self._mean_fp32(
            (
                reconstructed_geometry_local
                < baseline_geometry_local - 1.0e-4
            ).float()
        )
        losses["regression_fraction"] = self._mean_fp32(
            (
                reconstructed_geometry_local
                > baseline_geometry_local + 1.0e-4
            ).float()
        )
        losses["geometry_proxy_improvement"] = self._mean_fp32(
            baseline_geometry_local - reconstructed_geometry_local
        ).detach()
        losses["baseline_albedo"] = self._mean_fp32(
            (baseline_albedo - target_albedo).abs()
        )

        # Keep the renderer exactly inert away from useful contour support.
        off_contour_weight = (
            1.0 - near_contour.float()
        ).clamp(0.0, 1.0)
        losses["boundary_identity"] = self._weighted_mean(
            (
                reconstructed_albedo.float()
                - baseline_albedo.float()
            ).abs().mean(dim=1, keepdim=True),
            (0.10 + off_contour_weight * 1.90).detach(),
        )

        # Physical-map alignment uses the same reconstructed boundary. Normal and
        # material edges should agree with the albedo/contour field rather than
        # drifting independently.
        reconstructed_albedo_edge = self.gradient_magnitude(
            reconstructed_albedo.mean(dim=1, keepdim=True)
        )
        reconstructed_normal_edge = self.gradient_magnitude(
            reconstructed_normal
        ).amax(dim=1, keepdim=True)
        reconstructed_material_edge = self.gradient_magnitude(
            reconstructed_material
        ).amax(dim=1, keepdim=True)
        contour_probability = torch.sigmoid(outputs["edge_logits"])
        losses["cross_map"] = (
            self.charbonnier(
                reconstructed_albedo_edge.clamp(0.0, 1.0),
                contour_probability,
            )
            + 0.5
            * self.charbonnier(
                reconstructed_normal_edge.clamp(0.0, 1.0),
                contour_probability,
            )
            + 0.35
            * self.charbonnier(
                reconstructed_material_edge.clamp(0.0, 1.0),
                contour_probability,
            )
        )
        losses["seam"] = (
            self.seam_loss(reconstructed_albedo, target_albedo)
            + 0.5 * self.seam_loss(reconstructed_normal, target_normal)
        )

        # Appearance terms remain defined for a later frozen-geometry stage.
        pred_local = self._local_error(outputs["albedo"], target_albedo, patch)
        base_local = self._local_error(
            reconstructed_albedo.detach(), target_albedo, patch
        )
        appearance_regret_map = F.relu(pred_local - base_local)
        losses["appearance_regret"] = self._mean_fp32(
            appearance_regret_map * regret_protection
        )
        pred_normal_local = self._local_error(
            outputs["normal_xy"], target_normal, patch
        )
        base_normal_local = self._local_error(
            reconstructed_normal.detach(), target_normal, patch
        )
        losses["normal_regret"] = self._mean_fp32(
            F.relu(pred_normal_local - base_normal_local)
        )

        baseline_pixel_error = (
            reconstructed_albedo.detach() - target_albedo
        ).abs().mean(dim=1, keepdim=True)
        unchanged = (
            baseline_pixel_error < config.unchanged_error_threshold
        ).to(dtype)
        unchanged_weight = unchanged * (
            1.0
            + near_contour.to(dtype)
            * float(config.edge_regret_multiplier)
        )
        unchanged_error = (
            outputs["albedo"] - target_albedo
        ).abs().mean(dim=1, keepdim=True)
        losses["unchanged"] = (
            self._sum_fp32(unchanged_error * unchanged_weight)
            / self._sum_fp32(unchanged_weight).clamp_min(1.0)
        )

        residual_terms = [
            outputs["albedo_delta_medium"].abs().mean(),
            outputs["albedo_delta_fine"].abs().mean(),
            outputs["normal_delta_medium"].abs().mean(),
            outputs["normal_delta_fine"].abs().mean(),
            outputs["material_delta"].abs().mean(),
        ]
        losses["residual_l1"] = torch.stack(
            [term.float() for term in residual_terms]
        ).mean()
        losses["fine_zero_mean"] = self._fine_zero_mean(
            outputs["albedo_delta_fine"]
        )

        pred_lap = self.laplacian_tensor(outputs["albedo"])
        target_lap = self.laplacian_tensor(target_albedo)
        boundary_lap = self.laplacian_tensor(
            reconstructed_albedo.detach()
        )
        detail_need = (
            target_lap - boundary_lap
        ).abs().detach().mean(dim=1, keepdim=True)
        detail_weight = (
            0.25 + detail_need * 6.0
        ).clamp(0.25, 2.0)
        losses["detail_laplacian"] = self._mean_fp32(
            (pred_lap - target_lap).abs() * detail_weight
        )
        ringing_excess = F.relu(
            pred_lap.abs() - target_lap.abs() - 0.015
        )
        losses["ringing_regret"] = self._mean_fp32(ringing_excess)

        # V10.6 full physical-detail proof. The geometry/profile candidate is the
        # frozen baseline for this stage; improvement must come from the 4x detail
        # decoder rather than from moving the contour or repainting with the selector.
        detail_candidate_albedo = outputs.get("detail_candidate_albedo", outputs["albedo"]).float()
        detail_candidate_normal = outputs.get("detail_candidate_normal", outputs["normal_xy"]).float()
        detail_candidate_material = outputs.get("detail_candidate_material", outputs["material"]).float()
        base_albedo_detached = reconstructed_albedo.detach().float()
        base_normal_detached = reconstructed_normal.detach().float()
        base_material_detached = reconstructed_material.detach().float()

        detail_base_error_map = (base_albedo_detached - target_albedo.float()).abs().mean(dim=1, keepdim=True)
        detail_final_error_map = (detail_candidate_albedo - target_albedo.float()).abs().mean(dim=1, keepdim=True)
        detail_base_error = self._mean_fp32(detail_base_error_map)
        detail_final_error = self._mean_fp32(detail_final_error_map)
        losses["detail_base_error"] = detail_base_error.detach()
        losses["detail_final_error"] = detail_final_error.detach()
        losses["detail_recovery"] = ((detail_base_error - detail_final_error) / detail_base_error.clamp_min(1.0e-6)).detach()
        losses["detail_win_fraction"] = self._mean_fp32((detail_final_error_map < detail_base_error_map).float()).detach()
        losses["detail_regression_fraction"] = self._mean_fp32(
            (detail_final_error_map > detail_base_error_map + 0.002).float()
        ).detach()

        detail_base_grad_error = self._mean_fp32((
            self.gradient_magnitude(base_albedo_detached) - self.gradient_magnitude(target_albedo.float())
        ).abs())
        detail_final_grad_error = self._mean_fp32((
            self.gradient_magnitude(detail_candidate_albedo) - self.gradient_magnitude(target_albedo.float())
        ).abs())
        losses["detail_gradient_recovery"] = (
            (detail_base_grad_error - detail_final_grad_error) / detail_base_grad_error.clamp_min(1.0e-6)
        ).detach()

        # V10.8.1 final-output safety metrics. These are deliberately measured on
        # outputs["albedo"] (after selector/gating), not on the ungated detail
        # candidate. A selector checkpoint cannot qualify merely because the detail
        # branch improved somewhere while the actual rendered output regressed.
        final_output_error_map = (
            outputs["albedo"].float() - target_albedo.float()
        ).abs().mean(dim=1, keepdim=True)
        final_output_error = self._mean_fp32(final_output_error_map)
        losses["final_output_error"] = final_output_error.detach()
        losses["final_recovery"] = (
            (detail_base_error - final_output_error) / detail_base_error.clamp_min(1.0e-6)
        ).detach()
        losses["final_win_fraction"] = self._mean_fp32(
            (final_output_error_map < detail_base_error_map).float()
        ).detach()
        losses["final_regression_fraction"] = self._mean_fp32(
            (final_output_error_map > detail_base_error_map + 0.002).float()
        ).detach()

        def _local_contrast(value: torch.Tensor) -> torch.Tensor:
            smooth = F.avg_pool2d(value.float(), kernel_size=5, stride=1, padding=2)
            return value.float() - smooth

        losses["detail_contrast"] = self.charbonnier(
            _local_contrast(detail_candidate_albedo), _local_contrast(target_albedo)
        )

        final_albedo_edge = self.gradient_magnitude(detail_candidate_albedo.mean(dim=1, keepdim=True))
        final_normal_edge = self.gradient_magnitude(detail_candidate_normal).amax(dim=1, keepdim=True)
        final_material_edge = self.gradient_magnitude(detail_candidate_material).amax(dim=1, keepdim=True)
        target_albedo_edge = self.gradient_magnitude(target_albedo.float().mean(dim=1, keepdim=True))
        target_normal_edge = self.gradient_magnitude(target_normal.float()).amax(dim=1, keepdim=True)
        # The dataset stores material identity as a class index rather than the
        # original continuous scalar channel. Reconstruct the class-centre scalar for
        # cross-map edge supervision so material boundaries remain geometrically
        # aligned without relying on an undefined continuous target tensor.
        target_material_scalar = (
            target_material_class.float().unsqueeze(1) + 0.5
        ) / float(max(int(config.material_classes), 1))
        target_material_rgb = torch.cat((
            target_material_scalar, target_emissive.float(), target_roughness.float()
        ), dim=1)
        losses["detail_candidate_albedo"] = self.charbonnier(
            detail_candidate_albedo, target_albedo.float()
        )
        losses["detail_candidate_albedo_gradient"] = self.gradient_loss(
            detail_candidate_albedo, target_albedo.float()
        )
        losses["detail_candidate_normal"] = self.normal_cosine_loss(
            detail_candidate_normal, target_normal.float()
        )
        losses["detail_candidate_normal_gradient"] = self.gradient_loss(
            detail_candidate_normal, target_normal.float()
        )
        losses["detail_candidate_roughness"] = self.masked_mean(
            torch.sqrt((detail_candidate_material[:, 2:3] - target_roughness.float()).square() + 1.0e-6),
            auxiliary_valid,
        )
        losses["detail_candidate_emissive"] = self.masked_mean(
            torch.sqrt((detail_candidate_material[:, 1:2] - target_emissive.float()).square() + 1.0e-6),
            auxiliary_valid,
        )
        detail_material_centres = torch.linspace(
            0.0, 1.0, int(config.material_classes),
            device=detail_candidate_material.device,
            dtype=detail_candidate_material.dtype,
        )
        detail_material_logits = -(
            detail_candidate_material[:, 0:1]
            - detail_material_centres.view(1, -1, 1, 1)
        ).square() * 40.0
        losses["detail_candidate_material"] = self.masked_mean(
            F.cross_entropy(
                detail_material_logits.float(), target_material_class,
                reduction="none",
            ).unsqueeze(1),
            auxiliary_valid.float(),
        )
        target_material_edge = self.gradient_magnitude(target_material_rgb).amax(dim=1, keepdim=True)
        losses["detail_cross_map"] = (
            self.charbonnier(final_albedo_edge.clamp(0.0, 1.0), target_albedo_edge.clamp(0.0, 1.0))
            + 0.5 * self.charbonnier(final_normal_edge.clamp(0.0, 1.0), target_normal_edge.clamp(0.0, 1.0))
            + 0.35 * self.charbonnier(final_material_edge.clamp(0.0, 1.0), target_material_edge.clamp(0.0, 1.0))
        )

        detail_confidence_logits = outputs.get("detail_confidence_logits")
        detail_regret_logits = outputs.get("detail_regret_logits")
        if detail_confidence_logits is not None and detail_regret_logits is not None:
            improvement = (detail_base_error_map - detail_final_error_map).detach()
            confidence_target = (0.5 + improvement / max(float(config.gate_error_scale), 1.0e-4)).clamp(0.0, 1.0)
            regret_target = (detail_final_error_map.detach() > detail_base_error_map.detach() + 0.001).float()
            losses["detail_confidence"] = F.binary_cross_entropy_with_logits(
                detail_confidence_logits.float(), confidence_target.float()
            )
            losses["detail_regret_classifier"] = F.binary_cross_entropy_with_logits(
                detail_regret_logits.float(), regret_target.float()
            )
        else:
            losses["detail_confidence"] = zero
            losses["detail_regret_classifier"] = zero

        # Compatibility metrics: V9.8 has no learned displacement actuator.
        losses["displacement_tangent"] = zero
        losses["displacement_smoothness"] = zero
        losses["displacement_off_contour"] = zero
        losses["displacement_sparsity"] = zero
        losses["gate_target"] = losses["boundary_gate"]
        losses["direct_flow"] = zero
        losses["flow_rms_source_pixels"] = zero
        losses["flow_p95_source_pixels"] = zero
        losses["flow_max_source_pixels"] = zero
        losses["gate_mean"] = losses["boundary_gate_edge_mean"].detach()
        losses["gate_p95"] = boundary_gate_prediction.detach().flatten().quantile(0.95)
        losses["gate_edge_mean"] = losses["boundary_gate_edge_mean"].detach()
        losses["gate_flat_mean"] = losses["boundary_gate_flat_mean"].detach()
        losses["edge_active_fraction"] = self._weighted_mean(
            (boundary_gate_prediction.detach() > 0.20).float(),
            (0.05 + near_contour.float()).detach(),
        )
        losses["off_edge_active_fraction"] = self._weighted_mean(
            (boundary_gate_prediction.detach() > 0.20).float(),
            (1.0 - near_contour.float()).clamp(0.0, 1.0).detach(),
        )
        losses["flow_normal_component"] = zero
        losses["flow_tangent_component"] = zero
        losses["target_flow_rms_source_pixels"] = zero

        sdf_bootstrap = phase == "sdf-bootstrap"
        sdf_proof = phase == "sdf-proof"
        seam_proof = phase == "seam-proof"
        seam_authority_phase = phase == "seam-authority"
        gate_proof = phase == "gate-proof"
        boundary_hardening = phase in {"boundary-hardening", "physical-finetune"}
        detail_phase = bool(
            getattr(config, "detail_reconstruction_enabled", True)
            and phase == "detail-reconstruction"
        )

        if sdf_bootstrap:
            # B1a trains the exact production geometry/conditioning path. There is
            # no auxiliary spline or stroke checkpoint branch.
            total = (
                losses["sdf_surface"] * config.sdf_surface_weight
                + losses["sdf_sign"] * config.sdf_sign_weight
                + losses["sdf_topology_sign"] * 8.0
                + losses["sdf_eikonal"] * config.sdf_eikonal_weight
                + losses["edge"] * config.edge_weight
                + losses["edge_sdf_consistency"] * config.boundary_edge_sdf_consistency_weight
                + losses["orientation"] * config.orientation_weight
                + losses["hardness"] * config.boundary_hardness_weight
            )
        elif sdf_proof:
            # V10.7.9 B1b: compact primitive class + global continuous parameters.
            # Analytic rasterisation guarantees manufactured-line smoothness by
            # construction; there is no dense centreline/SDF deformation authority.
            total = (
                losses["primitive_class"] * float(getattr(config, "parametric_primitive_class_weight", 8.0))
                + losses["primitive_param"] * float(getattr(config, "parametric_primitive_param_weight", 48.0))
                + losses["primitive_render"] * float(getattr(config, "parametric_primitive_render_weight", 6.0))
            )
        elif seam_proof:
            # B3: geometry is frozen and the trainer forces the exact GT seam mask
            # and tangent. This proves the directional/phase reconstruction itself.
            total = (
                losses["seam_reconstruction"] * float(getattr(config, "seam_reconstruction_weight", 28.0))
                + losses["seam_phase_residual"] * float(getattr(config, "seam_phase_residual_weight", 36.0))
                + losses["seam_tangent_smoothness"] * float(getattr(config, "seam_tangent_smoothness_weight", 16.0))
                + losses["seam_normal_profile"] * float(getattr(config, "seam_normal_profile_weight", 18.0))
                + losses["seam_projected_view"] * float(getattr(config, "seam_projected_view_weight", 6.0))
            )
        elif seam_authority_phase:
            # B4: reconstruction filters remain frozen; train only the positive/
            # negative seam authority teacher derived from HR missing detail/ridges.
            total = (
                losses["seam_authority_teacher"] * float(getattr(config, "seam_authority_teacher_weight", 18.0))
                + losses["seam_authority_regularization"] * float(getattr(config, "seam_authority_regularization_weight", 0.20))
            )
        elif gate_proof:
            # Stage 3: freeze structural geometry and train only the small local
            # coverage specialist against Panel 2. No SDF, gate or appearance term
            # can move the already-proven contour in this phase.
            total = (
                losses["boundary_specialist_coverage"] * config.boundary_specialist_coverage_weight
                + losses["boundary_specialist_coverage_bce"] * config.boundary_specialist_coverage_bce_weight
                + losses["boundary_specialist_coverage_gradient"] * config.boundary_specialist_coverage_gradient_weight
                + losses["boundary_specialist_profile_moment"] * config.boundary_specialist_profile_moment_weight
                + losses["sdf_teacher_render"] * config.sdf_teacher_render_weight
                + losses["boundary_specialist_gradient"] * config.boundary_specialist_gradient_weight
                + losses["boundary_specialist_profile"] * config.boundary_specialist_profile_weight
                + losses["boundary_fuzz"] * config.boundary_fuzz_weight
                + losses["boundary_halo"] * config.boundary_halo_weight
            )
        elif detail_phase:
            # Stage C: geometry and the profile specialist are frozen. The explicit
            # 2x/4x detail decoder must reconstruct authored physical-map content on
            # held-out Raven tiles before any selector is allowed to hide mistakes.
            total = (
                losses["detail_candidate_albedo"] * config.albedo_weight
                + losses["detail_candidate_albedo_gradient"] * (config.albedo_gradient_weight * 1.5)
                + losses["detail_candidate_normal"] * config.normal_weight
                + losses["detail_candidate_normal_gradient"] * (config.normal_gradient_weight * 1.4)
                + losses["detail_candidate_roughness"] * config.roughness_weight
                + losses["detail_candidate_emissive"] * config.emissive_weight
                + losses["detail_candidate_material"] * config.material_weight
                + losses["appearance_regret"] * config.regret_weight
                + losses["normal_regret"] * config.normal_regret_weight
                + losses["residual_l1"] * config.residual_l1_weight
                + losses["unchanged"] * config.unchanged_region_weight
                + losses["detail_laplacian"] * config.detail_laplacian_weight
                + losses["detail_contrast"] * float(getattr(config, "detail_contrast_weight", 0.60))
                + losses["detail_cross_map"] * float(getattr(config, "detail_cross_map_weight", 0.35))
                + losses["detail_confidence"] * float(getattr(config, "detail_confidence_weight", 0.20))
                + losses["detail_regret_classifier"] * float(getattr(config, "detail_regret_classifier_weight", 0.40))
                + losses["ringing_regret"] * config.ringing_regret_weight
                + losses["seam"] * config.seam_weight
            )
        elif boundary_hardening:
            # Production Stage D is identical for Raven Quick and Full: the frozen
            # candidate is always consumed through BenefitSelector + confidence +
            # regret, never through a Raven-only raw-candidate objective.
            total = (
                losses["boundary_gate"] * config.benefit_selector_weight
                + losses["boundary_pixel_regret"] * config.boundary_pixel_regret_weight
                + losses["appearance_regret"] * config.regret_weight
                + losses["normal_regret"] * config.normal_regret_weight
            )
        else:
            # Defensive fallback for legacy phase names: preserve structural authority.
            total = (
                losses["contour_transport"] * config.contour_transport_weight
                + losses["contour_dilation"] * config.contour_dilation_weight
                + losses["contour_soft_coverage"] * config.contour_soft_coverage_weight
                + losses["sdf_topology_sign"] * config.sdf_topology_weight
            )

        losses["total"] = total.float()
        return losses

_losses_service = LossesService()
_mean_fp32 = _losses_service._mean_fp32
_sum_fp32 = _losses_service._sum_fp32
_target_like = _losses_service._target_like
charbonnier = _losses_service.charbonnier
gradient_loss = _losses_service.gradient_loss
gradient_magnitude = _losses_service.gradient_magnitude
_subpixel_target_stack = _losses_service._subpixel_target_stack
_coverage_profile_width = _losses_service._coverage_profile_width
_metricize_level_set_pixels = _losses_service._metricize_level_set_pixels
pyramid_loss = _losses_service.pyramid_loss
normal_cosine_loss = _losses_service.normal_cosine_loss
masked_mean = _losses_service.masked_mean
_weighted_mean = _losses_service._weighted_mean
_project_contour_offset_to_source_lattice = _losses_service._project_contour_offset_to_source_lattice
_project_contour_vector_to_control_lattice = _losses_service._project_contour_vector_to_control_lattice
_sdf_global_polarity = _losses_service._sdf_global_polarity
_balanced_metric_band_mean = _losses_service._balanced_metric_band_mean
_balanced_sign_mean = _losses_service._balanced_sign_mean
_topology_sign_margin_loss = _losses_service._topology_sign_margin_loss
seam_loss = _losses_service.seam_loss
_local_error = _losses_service._local_error
_local_scalar_error = _losses_service._local_scalar_error
_fine_zero_mean = _losses_service._fine_zero_mean
laplacian_tensor = _losses_service.laplacian_tensor
_axial_encoding = _losses_service._axial_encoding
_image_tangent = _losses_service._image_tangent
_field_variation = _losses_service._field_variation
_normalise_edge = _losses_service._normalise_edge
_directional_derivatives = _losses_service._directional_derivatives
_projected_view_seam_loss = _losses_service._projected_view_seam_loss
compute_losses = _losses_service.compute_losses
