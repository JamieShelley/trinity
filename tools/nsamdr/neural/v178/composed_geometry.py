"""V178-only, deterministic synthetic junction/parallel geometry.

This module supplements—not modifies—the frozen V17.7 generator. Every target
comes from continuous geometry evaluated on the 4x grid; neither native edge
observations nor independent material auxiliaries contain the HR target.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .contour_model import physical_edge_evidence

KINDS = ("parallel", "T_junction", "X_junction", "diagonal", "curve",
         "circle", "protected_texture")


def _distance_to_segment(x: torch.Tensor, y: torch.Tensor,
                         start: tuple[float, float],
                         end: tuple[float, float]) -> torch.Tensor:
    vx, vy = end[0] - start[0], end[1] - start[1]
    scale = vx * vx + vy * vy
    t = ((x - start[0]) * vx + (y - start[1]) * vy) / scale
    t = t.clamp(0, 1)
    return torch.sqrt((x - start[0] - t * vx).square()
                      + (y - start[1] - t * vy).square() + 1e-8)


def _auxiliary(patch: int, seed: int) -> torch.Tensor:
    """Exactly V5's seeded, independent nuisance-observation policy."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x5A17832)
    coarse = torch.rand((1, 2, 4, 4), generator=g)
    maps = F.interpolate(coarse, size=(patch, patch), mode="bilinear",
                         align_corners=False)
    level = torch.rand((1, 2, 1, 1), generator=g)
    maps = (level + .25 * (maps - .5)).clamp(0, 1)
    absent = torch.rand((1, 2, 1, 1), generator=g) < .5
    return torch.where(absent, torch.tensor([0., .5]).view(1, 2, 1, 1), maps)


def make_case(patch: int, seed: int, *, kind: str | None = None,
              sampling: str = "seeded", phase: tuple[int, int] | None = None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                         torch.Tensor, dict]:
    """Return native RGB, normal XY, observation, clean HR coverage, metadata.

    Sampling may be forced to area or point for controlled observability tests.
    The random draw sequence for geometry is unchanged when sampling changes.
    """
    if not 16 <= patch <= 128:
        raise ValueError("synthetic patches must be 16..128 native pixels")
    if sampling not in ("seeded", "area", "point"):
        raise ValueError("sampling must be seeded, area or point")
    if kind is not None and kind not in KINDS:
        raise ValueError("unknown composed geometry kind")
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def rnd(lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=g))

    chosen = KINDS[int(rnd(0, len(KINDS)))] if kind is None else kind
    grid = (torch.arange(4 * patch, dtype=torch.float32) + .5) / 4.
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    cx, cy = rnd(.44, .56) * patch, rnd(.44, .56) * patch
    theta = rnd(.22, math.pi - .22)
    ux, uy = math.cos(theta), math.sin(theta)
    nx, ny = -uy, ux
    width = rnd(.18, .48)
    length = rnd(.58, .77) * patch
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    paths: list[tuple[tuple[float, float], tuple[float, float]]] = []
    separation = None
    junction = None

    def seg(mx: float, my: float, vx: float, vy: float, radius: float
            ) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((mx - radius * vx, my - radius * vy),
                (mx + radius * vx, my + radius * vy))

    if chosen == "parallel":
        separation = rnd(2.2, 4.1)
        for sign in (-1, 1):
            item = seg(cx + sign * nx * separation / 2,
                       cy + sign * ny * separation / 2, ux, uy, length / 2)
            segments.append(item)
            paths.append(item)
    elif chosen == "T_junction":
        across = seg(cx, cy, ux, uy, length / 2)
        branch = ((cx, cy), (cx + nx * length * .46, cy + ny * length * .46))
        segments = [across, branch]
        paths = list(segments)
        junction = (cx, cy)
    elif chosen == "X_junction":
        opening = rnd(.60, 1.12)
        for sign in (-1, 1):
            a = theta + sign * opening / 2
            item = seg(cx, cy, math.cos(a), math.sin(a), length / 2)
            segments.append(item)
            paths.append(item)
        junction = (cx, cy)
    elif chosen == "diagonal":
        item = seg(cx, cy, ux, uy, length / 2)
        segments = [item]
        paths = [item]
    elif chosen == "curve":
        # Sample a smooth quadratic arc in native coordinates. Segment length
        # stays below 0.20 native pixels; geometric polygon error < HR pixel.
        bend = rnd(1.5, 4.2) * (1 if rnd(0, 1) > .5 else -1)
        centers = []
        for index in range(65):
            t = (index / 64. - .5) * length
            offset = bend * (4 * t * t / (length * length) - .5)
            centers.append((cx + t * ux + offset * nx,
                            cy + t * uy + offset * ny))
        paths = list(zip(centers[:-1], centers[1:]))
        longitudinal = (x - cx) * ux + (y - cy) * uy
        lateral = (x - cx) * nx + (y - cy) * ny
        signed = (lateral - bend * (4 * longitudinal.square()
                                   / (length * length) - .5)).abs() - width
    elif chosen == "circle":
        radius = rnd(.18, .29) * patch
        signed = ((x - cx).square() + (y - cy).square() + 1e-8).sqrt()
        signed = (signed - radius).abs() - width
    # protected_texture intentionally contains no geometry or positive target.

    if chosen not in ("circle", "curve"):
        signed = (torch.stack([_distance_to_segment(x, y, p, q) - width
                               for p, q in segments]).amin(dim=0)
                  if segments else torch.full_like(x, 10.))
    coverage = (.5 - 4 * signed).clamp(0, 1)
    bg = torch.tensor([rnd(.12, .36), rnd(.15, .42), rnd(.19, .48)]).view(3, 1, 1)
    edge = torch.tensor([rnd(.08, .50), rnd(.09, .48), rnd(.11, .55)]).view(3, 1, 1)
    texture = .008 * torch.sin(x * .23 + y * .12) + .006 * torch.cos(x * .09 - y * .20)
    if chosen == "protected_texture":
        texture = texture + .035 * torch.sin(x * 2.1 + y * .8)
        texture = texture + .025 * torch.cos(x * .9 - y * 1.8)
    rgb = (bg + edge * coverage.unsqueeze(0) + texture.unsqueeze(0)).clamp(0, 1)
    gy, gx = torch.gradient(signed, spacing=(.25, .25))
    magnitude = (gx.square() + gy.square() + 1e-6).sqrt()
    normal = torch.stack((.38 * coverage * gx / magnitude,
                          .38 * coverage * gy / magnitude))

    if sampling == "seeded":
        sampling = "area" if rnd(0, 1) < .5 else "point"
    if phase is None:
        phase = (int(rnd(0, 4)), int(rnd(0, 4)))
    if not all(isinstance(v, int) and 0 <= v < 4 for v in phase):
        raise ValueError("point phase must contain two integers in 0..3")

    def native(t: torch.Tensor) -> torch.Tensor:
        if sampling == "area":
            return F.avg_pool2d(t.unsqueeze(0), 4).squeeze(0)
        return t[:, phase[1]::4, phase[0]::4]

    a = (native(rgb) * 255).round().clamp(0, 255) / 255
    n = ((native(normal) + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1
    observation = torch.cat((physical_edge_evidence(a[None], n[None]),
                              _auxiliary(patch, seed)), dim=1)[0]
    metadata = {"kind": chosen, "seed": seed, "sampling": sampling,
                "phase": phase, "width": width, "separation": separation,
                "segments": paths, "junction": junction,
                "nativePatch": patch, "scale": 4}
    return a, n, observation, coverage[None], metadata


def mixed_batch(count: int, patch: int, seed: int, device: torch.device,
                *, augment_rgb: bool) -> tuple[torch.Tensor, ...]:
    """Mix original V5 and composed examples without modifying V5 semantics.

    The V5 inputs retain their exact historical implementation; new samples
    use independent auxiliary maps and the same seeded 50% RGB policy.
    """
    from tools.nsamdr.neural.train_nsamdr_v178_contour import _batch
    if count < 1:
        raise ValueError("batch must be positive")
    values = []
    for i in range(count):
        sample_seed = seed + i * 1723
        # Alternation is deterministic and yields both distributions each batch
        # for the intended batch=4 experiment.
        if i % 2 == 0:
            result = _batch(1, patch, sample_seed, device,
                            augment_rgb=augment_rgb)
            values.append(tuple(t[0] for t in result))
        else:
            a, n, m, target, _ = make_case(patch, sample_seed)
            if augment_rgb:
                contrast = torch.Generator(device="cpu").manual_seed(
                    int(sample_seed) ^ 0x5A17842)
                if float(torch.rand((), generator=contrast)) < .5:
                    a = 1. - a
            values.append(tuple(t.to(device) for t in (a, n, m, target)))
    return tuple(torch.stack([row[i] for row in values]) for i in range(4))
