"""V9.8 geometry-convergence reconstruction objective.

The objective first learns a continuous SDF/orientation/hardness field, then
scores the actual two-sided boundary renderer against analytic and authored
targets. Appearance residuals remain excluded so the network cannot solve the
task by repainting or sharpening the original staircase.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import V9Config
from .contours import sobel_tensor


def _mean_fp32(value: torch.Tensor) -> torch.Tensor:
    return value.float().mean()


def _sum_fp32(value: torch.Tensor) -> torch.Tensor:
    return value.float().sum()


def _target_like(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return target.to(device=prediction.device, dtype=prediction.dtype, non_blocking=True)


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    target = _target_like(target, prediction)
    return _mean_fp32(torch.sqrt((prediction - target).square() + epsilon * epsilon))


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _target_like(target, prediction)
    pgx, pgy = sobel_tensor(prediction)
    tgx, tgy = sobel_tensor(target)
    return charbonnier(pgx, tgx) + charbonnier(pgy, tgy)


def gradient_magnitude(value: torch.Tensor) -> torch.Tensor:
    gx, gy = sobel_tensor(value)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def pyramid_loss(prediction: torch.Tensor, target: torch.Tensor, levels: int = 3) -> torch.Tensor:
    target = _target_like(target, prediction)
    total = prediction.new_zeros((), dtype=torch.float32)
    weight = 1.0
    for _ in range(levels):
        total = total + charbonnier(prediction, target) * weight
        if min(prediction.shape[-2:]) <= 8:
            break
        prediction = F.avg_pool2d(prediction, 2)
        target = F.avg_pool2d(target, 2)
        weight *= 0.5
    return total


def normal_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _target_like(target, prediction)
    pz = torch.sqrt((1.0 - prediction.square().sum(dim=1, keepdim=True)).clamp_min(1e-6))
    tz = torch.sqrt((1.0 - target.square().sum(dim=1, keepdim=True)).clamp_min(1e-6))
    p = F.normalize(torch.cat((prediction, pz), dim=1), dim=1, eps=1e-4)
    t = F.normalize(torch.cat((target, tz), dim=1), dim=1, eps=1e-4)
    return _mean_fp32(1.0 - (p * t).sum(dim=1))


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.to(device=value.device, dtype=value.dtype, non_blocking=True)
    expanded = valid.expand_as(value)
    return _sum_fp32(value * expanded) / _sum_fp32(expanded).clamp_min(1.0)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return _sum_fp32(value * weight) / _sum_fp32(weight).clamp_min(1.0)




def _sdf_global_polarity(
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


def _balanced_metric_band_mean(
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


def seam_loss(prediction: torch.Tensor, target: torch.Tensor, border: int = 8) -> torch.Tensor:
    target = _target_like(target, prediction)
    border = max(1, min(border, prediction.shape[-1] // 4, prediction.shape[-2] // 4))
    regions = [
        (prediction[..., :border, :], target[..., :border, :]),
        (prediction[..., -border:, :], target[..., -border:, :]),
        (prediction[..., :, :border], target[..., :, :border]),
        (prediction[..., :, -border:], target[..., :, -border:]),
    ]
    return sum(charbonnier(p, t) for p, t in regions) / len(regions)


def _local_error(value: torch.Tensor, target: torch.Tensor, patch: int) -> torch.Tensor:
    error = (value - target).abs().mean(dim=1, keepdim=True)
    patch = max(1, min(int(patch), error.shape[-1], error.shape[-2]))
    return F.avg_pool2d(error, kernel_size=patch, stride=patch)


def _local_scalar_error(value: torch.Tensor, patch: int) -> torch.Tensor:
    patch = max(1, min(int(patch), value.shape[-1], value.shape[-2]))
    return F.avg_pool2d(value, kernel_size=patch, stride=patch)


def _fine_zero_mean(delta: torch.Tensor, patch: int = 16) -> torch.Tensor:
    patch = max(2, min(int(patch), delta.shape[-1], delta.shape[-2]))
    local_mean = F.avg_pool2d(delta, patch, patch)
    return _mean_fp32(local_mean.abs())


def laplacian_tensor(value: torch.Tensor) -> torch.Tensor:
    channels = value.shape[1]
    kernel = value.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return F.conv2d(
        value,
        kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3),
        padding=1,
        groups=channels,
    )


def _axial_encoding(direction: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(direction.float(), dim=1, eps=1e-4)
    tx, ty = unit[:, 0:1], unit[:, 1:2]
    return torch.cat((tx.square() - ty.square(), 2.0 * tx * ty), dim=1)


def _image_tangent(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scalar = value.float().mean(dim=1, keepdim=True)
    vector = scalar.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    kernel = (vector[:, None] * vector[None, :]) / 256.0
    smooth = F.conv2d(F.pad(scalar, (2, 2, 2, 2), mode="reflect"), kernel.view(1, 1, 5, 5))
    gx, gy = sobel_tensor(smooth)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
    tangent = torch.cat((-gy / magnitude, gx / magnitude), dim=1)
    return tangent, magnitude


def _field_variation(value: torch.Tensor) -> torch.Tensor:
    gx, gy = sobel_tensor(value)
    return torch.sqrt(gx.square() + gy.square() + 1.0e-8).mean(dim=1, keepdim=True)


def _normalise_edge(value: torch.Tensor) -> torch.Tensor:
    # Fixed monotonic mapping; unlike per-image normalisation it does not let a
    # weak/blurred contour look perfect merely because it is the strongest edge.
    return (value.float() * 2.5).clamp(0.0, 1.0)



def compute_losses(
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

    losses: dict[str, torch.Tensor] = {}
    zero = outputs["albedo"].new_zeros((), dtype=torch.float32)

    # Final-output metrics remain available even while appearance is disabled.
    losses["albedo"] = charbonnier(outputs["albedo"], target_albedo)
    losses["albedo_gradient"] = gradient_loss(outputs["albedo"], target_albedo)
    losses["albedo_pyramid"] = pyramid_loss(outputs["albedo"], target_albedo)
    losses["normal"] = normal_cosine_loss(outputs["normal_xy"], target_normal)
    losses["normal_gradient"] = gradient_loss(outputs["normal_xy"], target_normal)
    roughness_error = torch.sqrt(
        (outputs["roughness"] - target_roughness).square() + 1e-6
    )
    emissive_error = torch.sqrt(
        (outputs["emissive"] - target_emissive).square() + 1e-6
    )
    losses["roughness"] = masked_mean(roughness_error, auxiliary_valid)
    losses["emissive"] = masked_mean(emissive_error, auxiliary_valid)
    material_ce = F.cross_entropy(
        outputs["material_logits"].float(),
        target_material_class,
        reduction="none",
    ).unsqueeze(1)
    losses["material"] = masked_mean(material_ce, auxiliary_valid.float())

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
    if bool(config.sdf_sign_gauge_invariant):
        sdf_polarity = _sdf_global_polarity(
            predicted_sdf_pixels.detach(), raw_target_sdf_pixels, config.sdf_metric_band_pixels
        )
    else:
        sdf_polarity = torch.ones(
            (predicted_sdf_pixels.shape[0], 1, 1, 1),
            device=predicted_sdf_pixels.device, dtype=predicted_sdf_pixels.dtype,
        )
    target_sdf_pixels = raw_target_sdf_pixels * sdf_polarity
    target_sdf_aligned = target_sdf.float() * sdf_polarity

    # V9.8.3 trains the metric field where geometry matters. Beyond this band
    # sign/topology objectives retain authority, but far-field pixel count may
    # no longer dominate the contour itself.
    metric_band = float(config.sdf_metric_band_pixels)
    final_surface_error = F.smooth_l1_loss(
        predicted_sdf_pixels, target_sdf_pixels, beta=0.35, reduction="none"
    )
    losses["sdf_surface"] = _balanced_metric_band_mean(
        final_surface_error, target_sdf_pixels, metric_band
    )
    losses["sdf"] = _balanced_metric_band_mean(
        (outputs["sdf"].float() - target_sdf_aligned).abs(),
        target_sdf_pixels, metric_band
    )

    coarse_sdf_pixels = outputs.get("coarse_sdf_pixels", predicted_sdf_pixels).float()
    coarse_surface_error = F.smooth_l1_loss(
        coarse_sdf_pixels, target_sdf_pixels, beta=0.75, reduction="none"
    )
    losses["coarse_sdf_surface"] = _balanced_metric_band_mean(
        coarse_surface_error, target_sdf_pixels, metric_band, near_pixels=3.0
    )
    losses["sdf_residual_l1"] = _mean_fp32(
        outputs.get("sdf_residual_pixels", torch.zeros_like(predicted_sdf_pixels)).abs()
    )

    # Sign is evaluated only after the global gauge has been aligned. The far
    # field only needs the correct side and no extra zero-sets.
    target_inside = (target_sdf_pixels < 0.0).float()
    sign_weight = (
        0.20 + (target_sdf_pixels.abs() <= metric_band).float() * 1.80
    ).detach()
    losses["sdf_sign"] = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            -predicted_sdf_pixels / 1.5, target_inside, reduction="none"
        ),
        sign_weight,
    )

    pred_pad = F.pad(predicted_sdf_pixels, (1, 1, 1, 1), mode="replicate")
    pred_gx = (pred_pad[:, :, 1:-1, 2:] - pred_pad[:, :, 1:-1, :-2]) * 0.5
    pred_gy = (pred_pad[:, :, 2:, 1:-1] - pred_pad[:, :, :-2, 1:-1]) * 0.5
    pred_grad_norm = torch.sqrt(pred_gx.square() + pred_gy.square() + 1.0e-6)
    metric_mask = (target_sdf_pixels.abs() <= metric_band).float().detach()
    losses["sdf_eikonal"] = _weighted_mean(
        (pred_grad_norm - 1.0).abs(), metric_mask
    )

    # Metric-gradient supervision is the bootstrap authority: unlike Eikonal it
    # supplies a direction even near the almost-flat initial field.
    target_pad = F.pad(target_sdf_pixels, (1, 1, 1, 1), mode="replicate")
    target_gx = (target_pad[:, :, 1:-1, 2:] - target_pad[:, :, 1:-1, :-2]) * 0.5
    target_gy = (target_pad[:, :, 2:, 1:-1] - target_pad[:, :, :-2, 1:-1]) * 0.5
    metric_gradient_error = (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
    losses["sdf_metric_gradient"] = _balanced_metric_band_mean(
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
    losses["sdf_gradient_alignment"] = _weighted_mean(
        sdf_grad_alignment, (metric_mask * (0.25 + near_contour.float() * 1.75)).detach()
    )

    pred_sdf_curvature = laplacian_tensor(outputs["sdf"].float())
    target_sdf_curvature = laplacian_tensor(target_sdf_aligned).detach()
    losses["sdf_curvature"] = _weighted_mean(
        (pred_sdf_curvature - target_sdf_curvature).abs(), contour_weight
    )

    zero_band = max(0.10, float(config.sdf_zero_band_pixels))
    zero_distance = raw_target_sdf_pixels.abs()
    zero_mask = torch.where(
        zero_distance <= zero_band,
        torch.exp(-0.5 * (zero_distance / max(zero_band * 0.55, 1.0e-4)).square()),
        torch.zeros_like(zero_distance),
    ).detach()
    zero_sq = predicted_sdf_pixels.square()
    losses["boundary_sdf_zero"] = _weighted_mean(
        F.smooth_l1_loss(
            predicted_sdf_pixels, torch.zeros_like(predicted_sdf_pixels),
            beta=0.25, reduction="none"
        ),
        zero_mask * geometry_boost.float(),
    )

    # Collapse telemetry. These are metrics (zero loss authority) but travel in
    # the accumulator so every epoch reports whether the field contains a real
    # zero-set and metric gradients before the external Stage-B audit.
    losses["sdf_zero_rms_pixels"] = torch.sqrt(
        _weighted_mean(zero_sq, zero_mask).clamp_min(0.0) + 1.0e-12
    )
    losses["sdf_grad_norm_mean"] = _weighted_mean(pred_grad_norm, metric_mask)
    losses["sdf_positive_fraction"] = _mean_fp32((predicted_sdf_pixels > 0.0).float())
    losses["sdf_negative_fraction"] = _mean_fp32((predicted_sdf_pixels < 0.0).float())
    losses["sdf_polarity_positive_fraction"] = _mean_fp32((sdf_polarity > 0.0).float())

    predicted_edge_probability = torch.sigmoid(outputs["edge_logits"].float())
    sdf_edge_probability = torch.exp(
        -predicted_sdf_pixels.abs() / 0.85
    ).clamp(0.0, 1.0)
    losses["edge_sdf_consistency"] = _weighted_mean(
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
    losses["edge"] = _weighted_mean(
        edge_loss * geometry_boost.float(),
        edge_class_weight.detach(),
    )

    pred_orientation_axial = _axial_encoding(outputs["orientation"])
    target_orientation_axial = _axial_encoding(target_orientation)
    orientation_cosine = 1.0 - (
        pred_orientation_axial * target_orientation_axial
    ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    losses["orientation"] = _weighted_mean(
        orientation_cosine, contour_weight
    )

    # Hardness is a boundary-profile property, not a colour residual. Synthetic
    # analytic edges are deliberately hard; real targets retain softer profiles
    # when their fine/coarse gradient concentration is low.
    target_gray = target_albedo.float().mean(dim=1, keepdim=True)
    target_gradient = gradient_magnitude(target_gray)
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
    losses["hardness"] = _weighted_mean(
        hardness_bce,
        (0.15 + near_contour.float() * 1.85).detach(),
    )

    # Geometry is judged on the actual implicit-boundary renderer, never on an
    # appearance residual. The same SDF/gate/coverage is applied to every map.
    reconstructed_albedo = outputs["boundary_reconstructed_albedo"]
    reconstructed_normal = outputs["boundary_reconstructed_normal"]
    reconstructed_material = outputs["boundary_reconstructed_material"]
    baseline_albedo = outputs["baseline_albedo"]
    baseline_normal = outputs["baseline_normal"]

    reconstructed_tangent, reconstructed_edge_magnitude = _image_tangent(
        reconstructed_albedo
    )
    baseline_tangent, baseline_edge_magnitude = _image_tangent(
        baseline_albedo
    )
    reconstructed_tangent_axial = _axial_encoding(reconstructed_tangent)
    target_orientation_geometry = target_orientation_axial.float()
    contour_weight_geometry = contour_weight.float()

    image_alignment = 1.0 - (
        reconstructed_tangent_axial * target_orientation_geometry
    ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    edge_presence = (
        0.35 + reconstructed_edge_magnitude.detach().clamp(0.0, 1.0)
    ).clamp(0.35, 1.35)
    losses["geometric_alignment"] = _weighted_mean(
        image_alignment,
        contour_weight_geometry * edge_presence,
    )

    reconstructed_tangent_variation = _field_variation(
        reconstructed_tangent_axial
    )
    target_tangent_variation = _field_variation(
        target_orientation_geometry
    ).detach()
    tangent_excess = F.relu(
        reconstructed_tangent_variation
        - target_tangent_variation
        - float(config.tangent_variation_margin)
    )
    losses["tangent_coherence"] = _weighted_mean(
        tangent_excess, contour_weight_geometry
    )

    reconstructed_curvature_change = laplacian_tensor(
        reconstructed_tangent_axial
    ).abs().mean(dim=1, keepdim=True)
    target_curvature_change = laplacian_tensor(
        target_orientation_geometry
    ).abs().mean(dim=1, keepdim=True).detach()
    curvature_excess = F.relu(
        reconstructed_curvature_change
        - target_curvature_change
        - float(config.curvature_variation_margin)
    )
    losses["curvature_coherence"] = _weighted_mean(
        curvature_excess, contour_weight_geometry
    )

    boundary_photo_weight = (
        0.08 + near_contour.float() * (1.65 + geometry_exact.float())
    ).detach()
    losses["boundary_photometric"] = _weighted_mean(
        (
            reconstructed_albedo.float()
            - target_albedo.float()
        ).abs().mean(dim=1, keepdim=True),
        boundary_photo_weight,
    )
    # Legacy diagnostic name now points to the renderer result.
    losses["geometry_photometric"] = losses["boundary_photometric"]

    target_edge_proxy = _normalise_edge(
        gradient_magnitude(target_gray)
    )
    reconstructed_edge_proxy = _normalise_edge(
        reconstructed_edge_magnitude
    )
    baseline_edge_proxy = _normalise_edge(
        baseline_edge_magnitude.detach()
    )
    profile_weight = (
        0.10 + near_contour.float() * 2.40
    ).detach()
    losses["boundary_profile"] = _weighted_mean(
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
    recon_grad = gradient_magnitude(
        reconstructed_albedo.float().mean(dim=1, keepdim=True)
    )
    target_grad = gradient_magnitude(target_gray)
    recon_outer = _sum_fp32(recon_grad * outer_band * geometry_exact.float())
    recon_total = _sum_fp32(recon_grad * (hard_core + outer_band) * geometry_exact.float()).clamp_min(1.0e-6)
    target_outer = _sum_fp32(target_grad * outer_band * geometry_exact.float())
    target_total = _sum_fp32(target_grad * (hard_core + outer_band) * geometry_exact.float()).clamp_min(1.0e-6)
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
    losses["boundary_halo"] = _weighted_mean(
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
    exact_floor = geometry_exact.float() * float(config.boundary_gate_exact_floor)
    benefit_need = torch.maximum(benefit_need, exact_floor)
    gate_target = (
        target_edge_band * benefit_need
    ).clamp(0.0, 1.0).detach()

    gate_error = (boundary_gate_probability - gate_target).abs()
    edge_region_weight = (0.05 + target_edge_band * 0.95).detach()
    flat_region_weight = (1.0 - target_edge_band).clamp(0.0, 1.0).detach()
    # Balance contour and flat regions independently. A global pixel mean lets
    # the much larger flat area win the gate bias gradient and recreates the
    # V9.5 inactive-gate failure even when edge targets are correct.
    gate_edge_loss = _weighted_mean(gate_error, edge_region_weight)
    gate_flat_loss = _weighted_mean(boundary_gate_probability, flat_region_weight)
    losses["boundary_gate"] = gate_edge_loss + gate_flat_loss * 0.15
    losses["boundary_off_contour"] = gate_flat_loss

    losses["boundary_gate_edge_mean"] = _weighted_mean(
        boundary_gate_prediction.detach(),
        (0.05 + near_contour.float()).detach(),
    )
    losses["boundary_gate_flat_mean"] = _weighted_mean(
        boundary_gate_prediction.detach(),
        (1.0 - near_contour.float()).clamp(0.0, 1.0).detach(),
    )
    losses["boundary_gate_probability_edge_mean"] = _weighted_mean(
        boundary_gate_probability.detach(),
        (0.05 + near_contour.float()).detach(),
    )
    losses["boundary_gate_probability_flat_mean"] = _weighted_mean(
        boundary_gate_probability.detach(),
        (1.0 - near_contour.float()).clamp(0.0, 1.0).detach(),
    )
    losses["boundary_gate_applied_edge_mean"] = _weighted_mean(
        boundary_gate_applied.detach(),
        (0.05 + near_contour.float()).detach(),
    )
    losses["boundary_hardness_mean"] = _weighted_mean(
        outputs["hardness"].detach().float(),
        (0.05 + near_contour.float()).detach(),
    )
    losses["boundary_transition_width_mean"] = _weighted_mean(
        outputs["transition_width"].detach().float(),
        (0.05 + near_contour.float()).detach(),
    )
    losses["boundary_delta_rms"] = torch.sqrt(
        _mean_fp32(
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
    reconstructed_geometry_local = _local_scalar_error(
        reconstructed_geometry_error, patch
    )
    baseline_geometry_local = _local_scalar_error(
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
    losses["geometry_regret"] = _mean_fp32(
        geometry_regret_map * regret_protection
    )
    # Pixel-level regret catches fragmented/wavy regressions that can disappear
    # inside the older 8x8 local average.
    pixel_regret = F.relu(
        reconstructed_geometry_error
        - baseline_geometry_error
        + float(config.geometry_regret_margin)
    )
    losses["boundary_pixel_regret"] = _weighted_mean(
        pixel_regret,
        (0.10 + near_contour.float() * (1.0 + float(config.edge_regret_multiplier))).detach(),
    )
    losses["regret"] = losses["geometry_regret"]
    losses["improvement_fraction"] = _mean_fp32(
        (
            reconstructed_geometry_local
            < baseline_geometry_local - 1.0e-4
        ).float()
    )
    losses["regression_fraction"] = _mean_fp32(
        (
            reconstructed_geometry_local
            > baseline_geometry_local + 1.0e-4
        ).float()
    )
    losses["geometry_proxy_improvement"] = _mean_fp32(
        baseline_geometry_local - reconstructed_geometry_local
    ).detach()
    losses["baseline_albedo"] = _mean_fp32(
        (baseline_albedo - target_albedo).abs()
    )

    # Keep the renderer exactly inert away from useful contour support.
    off_contour_weight = (
        1.0 - near_contour.float()
    ).clamp(0.0, 1.0)
    losses["boundary_identity"] = _weighted_mean(
        (
            reconstructed_albedo.float()
            - baseline_albedo.float()
        ).abs().mean(dim=1, keepdim=True),
        (0.10 + off_contour_weight * 1.90).detach(),
    )

    # Physical-map alignment uses the same reconstructed boundary. Normal and
    # material edges should agree with the albedo/contour field rather than
    # drifting independently.
    reconstructed_albedo_edge = gradient_magnitude(
        reconstructed_albedo.mean(dim=1, keepdim=True)
    )
    reconstructed_normal_edge = gradient_magnitude(
        reconstructed_normal
    ).amax(dim=1, keepdim=True)
    reconstructed_material_edge = gradient_magnitude(
        reconstructed_material
    ).amax(dim=1, keepdim=True)
    contour_probability = torch.sigmoid(outputs["edge_logits"])
    losses["cross_map"] = (
        charbonnier(
            reconstructed_albedo_edge.clamp(0.0, 1.0),
            contour_probability,
        )
        + 0.5
        * charbonnier(
            reconstructed_normal_edge.clamp(0.0, 1.0),
            contour_probability,
        )
        + 0.35
        * charbonnier(
            reconstructed_material_edge.clamp(0.0, 1.0),
            contour_probability,
        )
    )
    losses["seam"] = (
        seam_loss(reconstructed_albedo, target_albedo)
        + 0.5 * seam_loss(reconstructed_normal, target_normal)
    )

    # Appearance terms remain defined for a later frozen-geometry stage.
    pred_local = _local_error(outputs["albedo"], target_albedo, patch)
    base_local = _local_error(
        reconstructed_albedo.detach(), target_albedo, patch
    )
    appearance_regret_map = F.relu(pred_local - base_local)
    losses["appearance_regret"] = _mean_fp32(
        appearance_regret_map * regret_protection
    )
    pred_normal_local = _local_error(
        outputs["normal_xy"], target_normal, patch
    )
    base_normal_local = _local_error(
        reconstructed_normal.detach(), target_normal, patch
    )
    losses["normal_regret"] = _mean_fp32(
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
        _sum_fp32(unchanged_error * unchanged_weight)
        / _sum_fp32(unchanged_weight).clamp_min(1.0)
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
    losses["fine_zero_mean"] = _fine_zero_mean(
        outputs["albedo_delta_fine"]
    )

    pred_lap = laplacian_tensor(outputs["albedo"])
    target_lap = laplacian_tensor(target_albedo)
    boundary_lap = laplacian_tensor(
        reconstructed_albedo.detach()
    )
    detail_need = (
        target_lap - boundary_lap
    ).abs().detach().mean(dim=1, keepdim=True)
    detail_weight = (
        0.25 + detail_need * 6.0
    ).clamp(0.25, 2.0)
    losses["detail_laplacian"] = _mean_fp32(
        (pred_lap - target_lap).abs() * detail_weight
    )
    ringing_excess = F.relu(
        pred_lap.abs() - target_lap.abs() - 0.015
    )
    losses["ringing_regret"] = _mean_fp32(ringing_excess)

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
    losses["edge_active_fraction"] = _weighted_mean(
        (boundary_gate_prediction.detach() > 0.20).float(),
        (0.05 + near_contour.float()).detach(),
    )
    losses["off_edge_active_fraction"] = _weighted_mean(
        (boundary_gate_prediction.detach() > 0.20).float(),
        (1.0 - near_contour.float()).clamp(0.0, 1.0).detach(),
    )
    losses["flow_normal_component"] = zero
    losses["flow_tangent_component"] = zero
    losses["target_flow_rms_source_pixels"] = zero

    sdf_bootstrap = phase == "sdf-bootstrap"
    sdf_proof = phase == "sdf-proof"
    gate_proof = phase == "gate-proof"
    boundary_hardening = phase == "boundary-hardening"
    appearance_phase = bool(
        config.appearance_enabled
        and phase in {"appearance-reconstruction", "physical-finetune"}
    )

    if sdf_bootstrap:
        # Stage 1: learn the continuous field itself. Renderer is disabled by
        # model.set_phase, so neither a gate nor a photometric shortcut can hide
        # a bad zero-set.
        total = (
            losses["sdf_surface"] * config.sdf_surface_weight
            + losses["coarse_sdf_surface"] * config.coarse_sdf_surface_weight
            + losses["sdf_residual_l1"] * config.sdf_residual_l1_weight
            + losses["sdf_sign"] * config.sdf_sign_weight
            + losses["sdf_eikonal"] * config.sdf_eikonal_weight
            + losses["sdf_gradient_alignment"] * config.sdf_gradient_alignment_weight
            + losses["sdf_metric_gradient"] * config.sdf_metric_gradient_weight
            + losses["edge"] * config.edge_weight
            + losses["orientation"] * config.orientation_weight
            + losses["sdf_curvature"] * config.sdf_curvature_weight
            + losses["boundary_sdf_zero"] * config.boundary_sdf_zero_weight
            + losses["edge_sdf_consistency"] * config.boundary_edge_sdf_consistency_weight
        )
    elif sdf_proof:
        # Stage 2: predicted SDF drives the real renderer, but training.py forces
        # the target contour gate. Any failure here is therefore an SDF/renderer
        # problem, not gate collapse.
        total = (
            losses["sdf_surface"] * config.sdf_surface_weight
            + losses["coarse_sdf_surface"] * config.coarse_sdf_surface_weight
            + losses["sdf_residual_l1"] * config.sdf_residual_l1_weight
            + losses["sdf_sign"] * config.sdf_sign_weight
            + losses["sdf_eikonal"] * config.sdf_eikonal_weight
            + losses["sdf_gradient_alignment"] * config.sdf_gradient_alignment_weight
            + losses["sdf_metric_gradient"] * config.sdf_metric_gradient_weight
            + losses["edge"] * config.edge_weight
            + losses["orientation"] * config.orientation_weight
            + losses["sdf_curvature"] * config.sdf_curvature_weight
            + losses["boundary_sdf_zero"] * config.boundary_sdf_zero_weight
            + losses["edge_sdf_consistency"] * config.boundary_edge_sdf_consistency_weight
            + losses["hardness"] * config.boundary_hardness_weight
            + losses["boundary_photometric"]
            * config.boundary_photometric_weight
            * config.sdf_proof_renderer_weight
            + losses["boundary_profile"]
            * config.boundary_profile_weight
            * config.sdf_proof_renderer_weight
            + losses["boundary_fuzz"] * config.boundary_fuzz_weight
            + losses["boundary_halo"] * config.boundary_halo_weight
            + losses["geometric_alignment"] * config.geometric_alignment_weight
            + losses["tangent_coherence"] * config.tangent_coherence_weight
            + losses["curvature_coherence"] * config.curvature_coherence_weight
            + losses["boundary_gate"] * (config.boundary_gate_weight * 0.25)
        )
    elif not appearance_phase:
        profile_scale = 1.45 if boundary_hardening else 1.0
        photo_scale = 0.85 if boundary_hardening else 1.0
        total = (
            losses["sdf_surface"] * config.sdf_surface_weight
            + losses["coarse_sdf_surface"] * config.coarse_sdf_surface_weight
            + losses["sdf_residual_l1"] * config.sdf_residual_l1_weight
            + losses["sdf_sign"] * config.sdf_sign_weight
            + losses["sdf_eikonal"] * config.sdf_eikonal_weight
            + losses["sdf_gradient_alignment"] * config.sdf_gradient_alignment_weight
            + losses["sdf_metric_gradient"] * config.sdf_metric_gradient_weight
            + losses["sdf"] * (config.sdf_weight * 0.35)
            + losses["edge"] * config.edge_weight
            + losses["orientation"] * config.orientation_weight
            + losses["sdf_curvature"] * config.sdf_curvature_weight
            + losses["boundary_sdf_zero"] * config.boundary_sdf_zero_weight
            + losses["edge_sdf_consistency"] * config.boundary_edge_sdf_consistency_weight
            + losses["hardness"]
            * config.boundary_hardness_weight
            * profile_scale
            + losses["geometric_alignment"]
            * config.geometric_alignment_weight
            + losses["tangent_coherence"]
            * config.tangent_coherence_weight
            + losses["curvature_coherence"]
            * config.curvature_coherence_weight
            + losses["boundary_photometric"]
            * config.boundary_photometric_weight
            * photo_scale
            + losses["boundary_profile"]
            * config.boundary_profile_weight
            * profile_scale
            + losses["boundary_fuzz"] * config.boundary_fuzz_weight * profile_scale
            + losses["boundary_halo"] * config.boundary_halo_weight * profile_scale
            + losses["boundary_gate"]
            * config.boundary_gate_weight
            + losses["boundary_off_contour"]
            * config.boundary_off_contour_weight
            + losses["boundary_identity"]
            * config.boundary_off_contour_weight
            + losses["geometry_regret"]
            * config.boundary_regret_weight
            + losses["boundary_pixel_regret"]
            * config.boundary_pixel_regret_weight
            + losses["cross_map"] * config.cross_map_weight
            + losses["seam"] * config.seam_weight
        )
        if phase == "physical-finetune":
            total = (
                total
                + losses["normal"] * (config.normal_weight * 0.35)
                + losses["normal_gradient"]
                * (config.normal_gradient_weight * 0.35)
                + losses["roughness"]
                * (config.roughness_weight * 0.20)
                + losses["emissive"]
                * (config.emissive_weight * 0.20)
                + losses["material"]
                * (config.material_weight * 0.20)
            )
    else:
        # GeometryNet is frozen before this branch becomes reachable.
        total = (
            losses["albedo"] * config.albedo_weight
            + losses["albedo_gradient"]
            * config.albedo_gradient_weight
            + losses["albedo_pyramid"]
            * config.albedo_pyramid_weight
            + losses["normal"] * config.normal_weight
            + losses["normal_gradient"]
            * config.normal_gradient_weight
            + losses["roughness"] * config.roughness_weight
            + losses["emissive"] * config.emissive_weight
            + losses["material"] * config.material_weight
            + losses["appearance_regret"] * config.regret_weight
            + losses["normal_regret"] * config.normal_regret_weight
            + losses["residual_l1"] * config.residual_l1_weight
            + losses["unchanged"]
            * config.unchanged_region_weight
            + losses["fine_zero_mean"]
            * config.fine_zero_mean_weight
            + losses["detail_laplacian"]
            * config.detail_laplacian_weight
            + losses["ringing_regret"]
            * config.ringing_regret_weight
            + losses["seam"] * config.seam_weight
        )

    losses["total"] = total.float()
    return losses
