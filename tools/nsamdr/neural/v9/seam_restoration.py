"""Directional seam restoration for NSAMDR V10.7.1.

The LR EVE maps contain long manufactured seams whose raster staircase is not a
material-boundary topology problem.  This module treats them as local vector
features.  A multi-map structure tensor estimates the shared seam normal/tangent,
then a small RAISR/BLADE-like directional branch smooths *along* the seam while
sharpening only across the seam.  Authority is bounded and concentrated on
coherent edge evidence; the ordered spline/SDF geometry remains the contour
location authority.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _sobel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = value.shape[1]
    kx = value.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = kx.t()
    x = F.pad(value.float(), (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x, kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3), groups=channels)
    gy = F.conv2d(x, ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3), groups=channels)
    return gx, gy


def _smooth(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(1, int(radius))
    kernel = radius * 2 + 1
    return F.avg_pool2d(value, kernel_size=kernel, stride=1, padding=radius)


def multi_map_structure_tensor(
    albedo: torch.Tensor,
    normal_xy: torch.Tensor,
    material: torch.Tensor,
    *,
    radius: int = 2,
    strength_gain: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Return an axial normal/tangent field shared by all physical maps."""
    luma = (
        albedo[:, 0:1].float() * 0.2126
        + albedo[:, 1:2].float() * 0.7152
        + albedo[:, 2:3].float() * 0.0722
    )
    # Normal/material edges are deliberately lower weight than albedo but can
    # rescue a physically real seam that is low contrast in diffuse colour.
    features = torch.cat((luma, normal_xy.float() * 0.55, material.float() * 0.45), dim=1)
    gx, gy = _sobel(features)
    jxx = _smooth(gx.square().sum(dim=1, keepdim=True), radius)
    jyy = _smooth(gy.square().sum(dim=1, keepdim=True), radius)
    jxy = _smooth((gx * gy).sum(dim=1, keepdim=True), radius)

    trace = jxx + jyy
    disc = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1.0e-12)
    lambda1 = 0.5 * (trace + disc)
    lambda2 = 0.5 * (trace - disc)
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1.0e-6)).clamp(0.0, 1.0)
    strength = (1.0 - torch.exp(-torch.sqrt(lambda1.clamp_min(0.0) + 1.0e-10) * float(strength_gain))).clamp(0.0, 1.0)

    # Dominant eigenvector of the structure tensor is the edge normal.  The
    # half-angle form is stable and axial, so +/- direction is equivalent.
    theta = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy + 1.0e-12)
    nx = torch.cos(theta)
    ny = torch.sin(theta)
    normal = torch.cat((nx, ny), dim=1)
    tangent = torch.cat((-ny, nx), dim=1)

    # Axial tangent curvature: t and -t describe the same seam direction.
    tx, ty = tangent[:, 0:1], tangent[:, 1:2]
    axial = torch.cat((tx.square() - ty.square(), 2.0 * tx * ty), dim=1)
    ax, ay = _sobel(axial)
    curvature = torch.sqrt(ax.square() + ay.square() + 1.0e-8).mean(dim=1, keepdim=True)
    straightness = (coherence * torch.exp(-curvature * 6.0)).clamp(0.0, 1.0)
    curve_support = (coherence * (1.0 - straightness) * torch.exp(-curvature * 0.75)).clamp(0.0, 1.0)
    irregular = (1.0 - torch.maximum(straightness, curve_support)).clamp(0.0, 1.0)
    classes = torch.cat((straightness, curve_support, irregular), dim=1)
    classes = classes / classes.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return {
        "normal": normal,
        "tangent": tangent,
        "strength": strength,
        "coherence": coherence,
        "curvature": curvature,
        "primitive_class": classes,
    }


def multi_map_ridge_response(
    albedo: torch.Tensor, normal_xy: torch.Tensor, material: torch.Tensor
) -> torch.Tensor:
    """Return a bounded ridge/groove response, not just a step-edge response.

    Manufactured panel seams are frequently bright-dark-bright or normal-map
    grooves and therefore need second-derivative evidence even when no material
    boundary exists.
    """
    luma = (
        albedo[:, 0:1].float() * 0.2126
        + albedo[:, 1:2].float() * 0.7152
        + albedo[:, 2:3].float() * 0.0722
    )
    features = torch.cat((luma, normal_xy.float() * 0.70, material.float() * 0.35), dim=1)
    gx, gy = _sobel(features)
    gxx, gxy = _sobel(gx)
    gyx, gyy = _sobel(gy)
    # Frobenius norm of the symmetric Hessian is sign-independent, so both
    # engraved valleys and raised ridges receive support.
    hxy = 0.5 * (gxy + gyx)
    ridge = torch.sqrt((gxx.square() + 2.0 * hxy.square() + gyy.square()).mean(dim=1, keepdim=True) + 1.0e-10)
    return (1.0 - torch.exp(-ridge * 10.0)).clamp(0.0, 1.0)


def _sample_offset(value: torch.Tensor, vector: torch.Tensor, pixels: float) -> torch.Tensor:
    """Bilinearly sample `value` at a spatially varying vector offset."""
    b, _c, h, w = value.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=value.device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, w, device=value.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1).clone()
    sx = 2.0 * float(pixels) / max(float(w - 1), 1.0)
    sy = 2.0 * float(pixels) / max(float(h - 1), 1.0)
    grid[..., 0] += vector[:, 0].float() * sx
    grid[..., 1] += vector[:, 1].float() * sy
    return F.grid_sample(
        value.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    )


class DirectionalKernelBank(nn.Module):
    """Small RAISR/BLADE-style axial filter bank.

    Each orientation bin owns a DC-preserving residual kernel.  The observed
    seam tangent softly selects the appropriate bin, so shallow diagonals are
    reconstructed by an orientation-specific filter instead of a generic
    isotropic convolution.  Kernels initialise to identity and must earn any
    correction from the HR teacher.
    """

    def __init__(self, bins: int = 12, kernel_size: int = 7, residual_scale: float = 0.10) -> None:
        super().__init__()
        self.bins = max(4, int(bins))
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.residual_scale = float(residual_scale)
        self.logits = nn.Parameter(torch.zeros(self.bins, self.kernel_size, self.kernel_size))
        angles = torch.arange(self.bins, dtype=torch.float32) * (math.pi / float(self.bins))
        self.register_buffer("bin_tangent", torch.stack((torch.cos(angles), torch.sin(angles)), dim=1), persistent=False)

    def kernels(self) -> torch.Tensor:
        residual = torch.tanh(self.logits) * self.residual_scale
        residual = residual - residual.mean(dim=(1, 2), keepdim=True)
        base = torch.zeros_like(residual)
        c = self.kernel_size // 2
        base[:, c, c] = 1.0
        return base + residual

    def forward(self, value: torch.Tensor, tangent: torch.Tensor, coherence: torch.Tensor) -> torch.Tensor:
        # Axial orientation: t and -t are the same seam.  Soft hashing avoids
        # bin-boundary discontinuities while retaining RAISR-like specialisation.
        t = F.normalize(tangent.float(), dim=1, eps=1.0e-5)
        dots = []
        for i in range(self.bins):
            b = self.bin_tangent[i].to(device=value.device, dtype=torch.float32)
            d = t[:, 0:1] * b[0] + t[:, 1:2] * b[1]
            dots.append(2.0 * d.square() - 1.0)
        axial_score = torch.cat(dots, dim=1)
        concentration = 2.0 + coherence.float().clamp(0.0, 1.0) * 10.0
        weights = torch.softmax(axial_score * concentration, dim=1)

        kernels = self.kernels().to(device=value.device, dtype=torch.float32)
        channels = value.shape[1]
        pad = self.kernel_size // 2
        padded = F.pad(value.float(), (pad, pad, pad, pad), mode="reflect")
        result = torch.zeros_like(value, dtype=torch.float32)
        for i in range(self.bins):
            kernel = kernels[i].view(1, 1, self.kernel_size, self.kernel_size).expand(channels, 1, -1, -1)
            filtered = F.conv2d(padded, kernel, groups=channels)
            result = result + filtered * weights[:, i:i+1]
        return result


class PhaseAwareSeamSR(nn.Module):
    """4x seam residual predictor with explicit output-pixel phase authority.

    PixelShuffle makes the sixteen 4x subpixel phases separate learned outputs,
    matching the classical RAISR observation that an SR filter must depend on
    both local edge geometry and the output pixel's phase inside the LR texel.
    The zero-initialised head makes this branch exact identity at startup.
    """

    def __init__(self, hidden: int = 32, max_delta: float = 0.25) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        # LR physical RGB/XY/RGB + geometry normal XY + SDF + edge support.
        self.body = nn.Sequential(
            nn.Conv2d(12, hidden, 5, padding=2), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden, 8 * 16, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        source_albedo: torch.Tensor,
        source_normal: torch.Tensor,
        source_material: torch.Tensor,
        geometry_normal_hr: torch.Tensor,
        sdf_pixels_hr: torch.Tensor,
        edge_probability_hr: torch.Tensor,
        *,
        hr_size: tuple[int, int],
    ) -> torch.Tensor:
        lr_size = source_albedo.shape[-2:]
        gn = F.interpolate(geometry_normal_hr.float(), size=lr_size, mode="bilinear", align_corners=False)
        sdf = F.interpolate(sdf_pixels_hr.float(), size=lr_size, mode="bilinear", align_corners=False)
        sdf = (sdf / 6.0).clamp(-1.0, 1.0)
        edge = F.interpolate(edge_probability_hr.float(), size=lr_size, mode="area")
        x = torch.cat((source_albedo.float(), source_normal.float(), source_material.float(), gn, sdf, edge), dim=1)
        delta = F.pixel_shuffle(self.head(self.body(x)), 4)
        delta = torch.tanh(delta) * self.max_delta
        if delta.shape[-2:] != hr_size:
            delta = F.interpolate(delta, size=hr_size, mode="bilinear", align_corners=False)
        return delta


class DirectionalSeamRestorer(nn.Module):
    """Bounded local vector-seam cleanup after deterministic boundary rendering.

    The branch cannot move spline nodes or alter SDF topology.  It produces a
    directional physical-map proposal and a learned authority scalar.  Initial
    weights make it nearly identity so Stage-B must earn any seam correction.
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden = int(getattr(config, "seam_directional_channels", 24))
        self.tensor_radius = int(getattr(config, "seam_structure_tensor_radius", 2))
        self.strength_gain = float(getattr(config, "seam_structure_strength_gain", 4.0))
        self.coherence_floor = float(getattr(config, "seam_coherence_floor", 0.45))
        self.tangent_sample_pixels = float(getattr(config, "seam_tangent_sample_pixels", 1.35))
        self.normal_sample_pixels = float(getattr(config, "seam_normal_sample_pixels", 0.90))
        self.max_sharpen = float(getattr(config, "seam_max_normal_sharpen", 1.35))
        self.max_authority = float(getattr(config, "seam_max_authority", 0.90))
        self.geometry_band = float(getattr(config, "seam_geometry_band_pixels", 4.0))
        self.ridge_weight = float(getattr(config, "seam_ridge_weight", 0.65))
        self.kernel_bank = DirectionalKernelBank(
            bins=int(getattr(config, "seam_directional_angle_bins", 12)),
            kernel_size=int(getattr(config, "seam_directional_kernel_size", 7)),
            residual_scale=float(getattr(config, "seam_directional_kernel_residual_scale", 0.10)),
        )
        self.phase_sr = PhaseAwareSeamSR(
            hidden=int(getattr(config, "seam_phase_sr_channels", 32)),
            max_delta=float(getattr(config, "seam_phase_sr_max_delta", 0.25)),
        )
        # strength, coherence, ridge/groove, straight, curve, geometry support,
        # coverage, profile confidence, model edge probability.
        self.authority = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 4, 1),
        )
        nn.init.zeros_(self.authority[-1].weight)
        nn.init.constant_(self.authority[-1].bias[0], -1.75)
        nn.init.zeros_(self.authority[-1].bias[1])
        nn.init.zeros_(self.authority[-1].bias[2])
        nn.init.zeros_(self.authority[-1].bias[3])

    @staticmethod
    def _normalise_xy(value: torch.Tensor) -> torch.Tensor:
        length2 = value.float().square().sum(dim=1, keepdim=True)
        scale = torch.where(length2 > 0.999**2, 0.999 / torch.sqrt(length2 + 1.0e-8), torch.ones_like(length2))
        return value.float() * scale

    def _proposal(
        self,
        value: torch.Tensor,
        tangent: torch.Tensor,
        normal: torch.Tensor,
        sharpen: torch.Tensor,
    ) -> torch.Tensor:
        t0 = _sample_offset(value, tangent, -self.tangent_sample_pixels)
        t1 = _sample_offset(value, tangent, +self.tangent_sample_pixels)
        along = (t0 + 2.0 * value.float() + t1) * 0.25
        n0 = _sample_offset(value, normal, -self.normal_sample_pixels)
        n1 = _sample_offset(value, normal, +self.normal_sample_pixels)
        across_blur = (n0 + 2.0 * value.float() + n1) * 0.25
        normal_detail = value.float() - across_blur
        return along + normal_detail * sharpen

    def forward(
        self,
        albedo: torch.Tensor,
        normal_xy: torch.Tensor,
        material: torch.Tensor,
        *,
        sdf_pixels: torch.Tensor,
        coverage: torch.Tensor,
        profile_confidence: torch.Tensor,
        edge_probability: torch.Tensor,
        geometry_normal: torch.Tensor | None = None,
        source_albedo: torch.Tensor | None = None,
        source_normal: torch.Tensor | None = None,
        source_material: torch.Tensor | None = None,
        authority_override: torch.Tensor | None = None,
        tangent_override: torch.Tensor | None = None,
        phase_only: bool = False,
        enabled: bool,
    ) -> dict[str, torch.Tensor]:
        # V10.7.1 measures seam evidence on the native LR observations whenever
        # available.  This prevents the 4x bicubic staircase from becoming the
        # orientation teacher.
        if source_albedo is not None and source_normal is not None and source_material is not None:
            tensor_lr = multi_map_structure_tensor(
                source_albedo, source_normal, source_material,
                radius=self.tensor_radius, strength_gain=self.strength_gain,
            )
            ridge_lr = multi_map_ridge_response(source_albedo, source_normal, source_material)
            hr_size = albedo.shape[-2:]
            tensor = {
                key: F.interpolate(value.float(), size=hr_size, mode="bilinear", align_corners=False)
                for key, value in tensor_lr.items() if key != "primitive_class"
            }
            tensor["normal"] = F.normalize(tensor["normal"], dim=1, eps=1.0e-5)
            tensor["tangent"] = F.normalize(tensor["tangent"], dim=1, eps=1.0e-5)
            ridge = F.interpolate(ridge_lr.float(), size=hr_size, mode="bilinear", align_corners=False)
        else:
            tensor = multi_map_structure_tensor(
                albedo, normal_xy, material,
                radius=self.tensor_radius, strength_gain=self.strength_gain,
            )
            ridge = multi_map_ridge_response(albedo, normal_xy, material)
        geometry_support = torch.exp(-sdf_pixels.float().abs() / max(self.geometry_band, 0.25))
        if geometry_normal is not None:
            gn = F.normalize(geometry_normal.float(), dim=1, eps=1.0e-5)
            sn = F.normalize(tensor["normal"].float(), dim=1, eps=1.0e-5)
            # SDF normal sign is a gauge. Align it to the observed structure
            # tensor before blending so opposite signs cannot cancel.
            sign = torch.where((gn * sn).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
            gn = gn * sign
            vector_blend = geometry_support * tensor["coherence"]
            normal_field = F.normalize(sn * (1.0 - vector_blend) + gn * vector_blend, dim=1, eps=1.0e-5)
            tangent_field = torch.cat((-normal_field[:, 1:2], normal_field[:, 0:1]), dim=1)
        else:
            normal_field = tensor["normal"]
            tangent_field = tensor["tangent"]
        if tangent_override is not None:
            tangent_field = tangent_override.float()
            if tangent_field.shape[-2:] != albedo.shape[-2:]:
                tangent_field = F.interpolate(tangent_field, size=albedo.shape[-2:], mode="bilinear", align_corners=False)
            tangent_field = F.normalize(tangent_field, dim=1, eps=1.0e-5)
            normal_field = torch.cat((tangent_field[:, 1:2], -tangent_field[:, 0:1]), dim=1)
        # Recompute straight/curve classes from the final geometry-aware tangent
        # so the vector primitive classifier sees the same orientation used by
        # the directional sampler.
        tx, ty = tangent_field[:, 0:1], tangent_field[:, 1:2]
        axial = torch.cat((tx.square() - ty.square(), 2.0 * tx * ty), dim=1)
        ax, ay = _sobel(axial)
        curvature = torch.sqrt(ax.square() + ay.square() + 1.0e-8).mean(dim=1, keepdim=True)
        straightness = (tensor["coherence"] * torch.exp(-curvature * 6.0)).clamp(0.0, 1.0)
        curve_support = (tensor["coherence"] * (1.0 - straightness) * torch.exp(-curvature * 0.75)).clamp(0.0, 1.0)
        irregular = (1.0 - torch.maximum(straightness, curve_support)).clamp(0.0, 1.0)
        classes = torch.cat((straightness, curve_support, irregular), dim=1)
        classes = classes / classes.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        learned = self.authority(torch.cat((
            tensor["strength"], tensor["coherence"], ridge, classes[:, 0:1], classes[:, 1:2],
            geometry_support, coverage.float(), profile_confidence.float(), edge_probability.float(),
        ), dim=1))
        learned_authority = torch.sigmoid(learned[:, 0:1])
        sharpen = 1.0 + torch.tanh(learned[:, 1:2]) * self.max_sharpen
        coherence_gate = ((tensor["coherence"] - self.coherence_floor) / max(1.0 - self.coherence_floor, 1.0e-4)).clamp(0.0, 1.0)
        # Geometry support boosts known boundaries but is not mandatory: engraved
        # panel lines may be visible in albedo/normal while not being material
        # segmentation boundaries.
        structure_support = torch.maximum(
            tensor["strength"] * coherence_gate,
            ridge * float(getattr(self, "ridge_weight", 0.65)),
        ).clamp(0.0, 1.0)
        geometry_authority = geometry_support * (0.25 + 0.75 * coverage.float().clamp(0.0, 1.0))
        # A proven SDF boundary is itself strong seam evidence. Requiring strong
        # texture gradients as well would suppress exactly the low-contrast and
        # codec-damaged seams this branch exists to restore. Engraved seams that
        # are not SDF boundaries still enter through structure_support.
        support = torch.maximum(structure_support, geometry_authority).clamp(0.0, 1.0)
        authority = learned_authority * support * self.max_authority
        authority_forced = torch.zeros_like(authority)
        if authority_override is not None:
            forced = authority_override.float()
            if forced.shape[-2:] != authority.shape[-2:]:
                forced = F.interpolate(forced, size=authority.shape[-2:], mode="bilinear", align_corners=False)
            authority = forced.clamp(0.0, 1.0)
            authority_forced = torch.ones_like(authority)
        if not enabled:
            authority = torch.zeros_like(authority)

        kernel_mix = torch.sigmoid(learned[:, 2:3])
        albedo_analytic = self._proposal(albedo, tangent_field, normal_field, sharpen)
        normal_analytic = self._proposal(normal_xy, tangent_field, normal_field, sharpen)
        material_analytic = self._proposal(material, tangent_field, normal_field, sharpen)
        albedo_kernel = self.kernel_bank(albedo, tangent_field, tensor["coherence"])
        normal_kernel = self.kernel_bank(normal_xy, tangent_field, tensor["coherence"])
        material_kernel = self.kernel_bank(material, tangent_field, tensor["coherence"])
        albedo_prop = (albedo_analytic * (1.0 - kernel_mix) + albedo_kernel * kernel_mix).clamp(0.0, 1.0)
        normal_prop = self._normalise_xy(normal_analytic * (1.0 - kernel_mix) + normal_kernel * kernel_mix)
        material_prop = (material_analytic * (1.0 - kernel_mix) + material_kernel * kernel_mix).clamp(0.0, 1.0)
        phase_mix = torch.sigmoid(learned[:, 3:4])
        if source_albedo is not None and source_normal is not None and source_material is not None:
            phase_delta = self.phase_sr(
                source_albedo, source_normal, source_material,
                normal_field, sdf_pixels, edge_probability, hr_size=albedo.shape[-2:],
            )
            phase_albedo = (albedo.float() + phase_delta[:, 0:3]).clamp(0.0, 1.0)
            phase_normal = self._normalise_xy(normal_xy.float() + phase_delta[:, 3:5])
            phase_material = (material.float() + phase_delta[:, 5:8]).clamp(0.0, 1.0)
            if phase_only:
                # V10.8.6: the representation proof and all qualified downstream
                # consumers use the exact same trained 4x phase-SR proposal.
                # The old analytic/kernel blend could overwhelm a good learned
                # residual and made B3 judge a different signal than the one
                # we actually wanted to train.
                phase_mix = torch.ones_like(phase_mix)
                kernel_mix = torch.zeros_like(kernel_mix)
                albedo_prop = phase_albedo
                normal_prop = phase_normal
                material_prop = phase_material
            else:
                albedo_prop = albedo_prop * (1.0 - phase_mix) + phase_albedo * phase_mix
                normal_prop = self._normalise_xy(normal_prop * (1.0 - phase_mix) + phase_normal * phase_mix)
                material_prop = material_prop * (1.0 - phase_mix) + phase_material * phase_mix
        else:
            phase_delta = torch.zeros(
                (albedo.shape[0], 8, albedo.shape[-2], albedo.shape[-1]),
                device=albedo.device, dtype=torch.float32,
            )
            phase_mix = torch.zeros_like(phase_mix)
        out_albedo = albedo.float() * (1.0 - authority) + albedo_prop * authority
        out_normal = self._normalise_xy(normal_xy.float() * (1.0 - authority) + normal_prop * authority)
        out_material = material.float() * (1.0 - authority) + material_prop * authority
        return {
            "albedo": out_albedo.to(albedo.dtype),
            "normal_xy": out_normal.to(normal_xy.dtype),
            "material": out_material.to(material.dtype),
            "authority": authority.to(albedo.dtype),
            "learned_authority": learned_authority.to(albedo.dtype),
            "normal": normal_field.to(albedo.dtype),
            "tangent": tangent_field.to(albedo.dtype),
            "strength": tensor["strength"].to(albedo.dtype),
            "coherence": tensor["coherence"].to(albedo.dtype),
            "ridge": ridge.to(albedo.dtype),
            "authority_forced": authority_forced.to(albedo.dtype),
            "curvature": curvature.to(albedo.dtype),
            "primitive_class": classes.to(albedo.dtype),
            "sharpen": sharpen.to(albedo.dtype),
            "kernel_mix": kernel_mix.to(albedo.dtype),
            "phase_mix": phase_mix.to(albedo.dtype),
            "phase_delta": phase_delta.to(albedo.dtype),
            "phase_only": torch.full_like(authority, 1.0 if phase_only else 0.0).to(albedo.dtype),
        }
