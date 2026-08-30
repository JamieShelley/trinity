"""Diagnostic zero-contour redistancing retained for NSAMDR V9.9.3.

GeometryNet owns only the sign / zero set.  Physical pixel distance is rebuilt
from explicit sub-pixel zero crossings, so learned level-set magnitude and
slope cannot change the renderer's boundary width.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


class RedistanceService:
    # Purpose: Implement sdf gradient components for RedistanceService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def sdf_gradient_components(self, sdf_pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(sdf_pixels.float(), (1, 1, 1, 1), mode="replicate")
        gx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) * 0.5
        gy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) * 0.5
        return gx, gy

    # Purpose: Implement zero crossing seed distance for RedistanceService.
    # Called by: redistance_zero_contour
    # Calls: No same-class helper methods.
    def zero_crossing_seed_distance(self, level_set_pixels: torch.Tensor) -> torch.Tensor:
        """Distance from each pixel centre to an incident interpolated zero crossing.

        A strict sign-changing edge has a zero at fraction
        ``abs(phi0) / (abs(phi0) + abs(phi1))`` from the current pixel centre.
        Uniformly scaling the level-set therefore leaves the seed geometry exactly
        unchanged. Pixels not incident to a zero crossing receive +inf.
        """
        raw = level_set_pixels.float()
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"level_set_pixels must be Nx1xHxW, got {tuple(raw.shape)}")

        inf = torch.full_like(raw, float("inf"))
        seed = torch.where(raw == 0.0, torch.zeros_like(raw), inf)

        # Replication is safe because equal border samples cannot introduce a new
        # strict sign crossing.  Each current pixel obtains its own fractional
        # distance to crossings on its four incident grid edges.
        padded = F.pad(raw, (1, 1, 1, 1), mode="replicate")
        neighbours = (
            padded[:, :, 1:-1, :-2],
            padded[:, :, 1:-1, 2:],
            padded[:, :, :-2, 1:-1],
            padded[:, :, 2:, 1:-1],
        )
        raw_abs = raw.abs()
        for neighbour in neighbours:
            crossing = ((raw < 0.0) & (neighbour > 0.0)) | ((raw > 0.0) & (neighbour < 0.0))
            denominator = raw_abs + neighbour.abs()
            fraction = raw_abs / denominator.clamp_min(1.0e-12)
            candidate = torch.where(crossing, fraction, inf)
            seed = torch.minimum(seed, candidate)
        return seed

    # Purpose: Implement redistance zero contour for RedistanceService.
    # Called by: External callers and the owning workflow.
    # Calls: zero_crossing_seed_distance
    def redistance_zero_contour(
        self,
        level_set_pixels: torch.Tensor,
        max_distance_pixels: float,
    ) -> torch.Tensor:
        """Rebuild a truncated signed distance field from the predicted zero set.

        The zero set is first extracted on pixel-grid edges with linear sub-pixel
        interpolation.  A deterministic parallel Eikonal solve then propagates
        unsigned distance away from those seeds.  The original predicted sign is
        restored afterwards.  No gradient-magnitude floor or learned SDF magnitude
        participates in physical distance reconstruction.
        """
        raw = level_set_pixels.float()
        max_distance = max(float(max_distance_pixels), 0.5)
        seed = self.zero_crossing_seed_distance(raw)

        far = max_distance + 2.0
        distance = torch.where(torch.isfinite(seed), seed, torch.full_like(seed, far))

        # One Jacobi wave advances at most one pixel per iteration.  The field is
        # truncated, so propagation beyond max_distance is unnecessary.
        iterations = int(math.ceil(max_distance)) + 2
        for _ in range(iterations):
            padded = F.pad(distance, (1, 1, 1, 1), mode="constant", value=far)
            left = padded[:, :, 1:-1, :-2]
            right = padded[:, :, 1:-1, 2:]
            up = padded[:, :, :-2, 1:-1]
            down = padded[:, :, 2:, 1:-1]
            a = torch.minimum(left, right)
            b = torch.minimum(up, down)
            lo = torch.minimum(a, b)
            hi = torch.maximum(a, b)
            delta = hi - lo

            # First-order Godunov update for |grad d| = 1 on a unit grid.
            one_axis = lo + 1.0
            radicand = (2.0 - delta.square()).clamp_min(0.0)
            two_axis = 0.5 * (lo + hi + torch.sqrt(radicand))
            candidate = torch.where(delta >= 1.0, one_axis, two_axis)
            distance = torch.minimum(distance, candidate).clamp_max(far)

        distance = distance.clamp(0.0, max_distance)
        signed = torch.where(raw < 0.0, -distance, distance)
        return torch.where(raw == 0.0, torch.zeros_like(signed), signed)

_redistance_service = RedistanceService()
sdf_gradient_components = _redistance_service.sdf_gradient_components
zero_crossing_seed_distance = _redistance_service.zero_crossing_seed_distance
redistance_zero_contour = _redistance_service.redistance_zero_contour
