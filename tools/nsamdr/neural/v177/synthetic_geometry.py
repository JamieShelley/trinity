"""Analytic clean 4x geometry with native-resolution aliased observations.

No EVE authored high-resolution target or sealed test image enters this
generator. The HR reference is computed from continuous SDF geometry directly,
not sharpened from LR inputs or reusing the model's output.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def _sample(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator)


def make_batch(
    batch: int, native_patch: int, *, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch < 1 or not 16 <= native_patch <= 128:
        raise ValueError("synthetic patches must be 16..128 native pixels")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    hr = native_patch * 4
    grid = (torch.arange(hr, dtype=torch.float32) + 0.5) / 4.0
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    source_a, source_n, source_m = [], [], []
    target_a, target_n, target_m, preserve = [], [], [], []
    for _ in range(batch):
        def rnd(lo: float, hi: float) -> float:
            return float(lo + (hi - lo) * _sample((1,), g).item())

        kind = int(rnd(0, 4))
        theta = rnd(0.12, math.pi - 0.12)
        ct, st = math.cos(theta), math.sin(theta)
        cx, cy = rnd(.26,.74)*native_patch, rnd(.26,.74)*native_patch
        dx, dy = x-cx, y-cy
        if kind == 0:
            # Subpixel-width diagonal groove / painted seam.
            signed = torch.abs(dx * ct + dy * st) - rnd(.13,.70)
        elif kind == 1:
            # Thin circular panel trim.
            radius = rnd(.14,.32) * native_patch
            signed = torch.abs(torch.sqrt(dx.square()+dy.square()+1.e-8)-radius)-rnd(.14,.63)
        elif kind == 2:
            # Rounded rectangular panel boundary.
            wx, wy = rnd(.10,.32)*native_patch, rnd(.10,.30)*native_patch
            qx, qy = dx.abs()-wx, dy.abs()-wy
            sdf = torch.sqrt(qx.clamp_min(0).square()+qy.clamp_min(0).square()+1.e-8) + torch.maximum(qx,qy).clamp_max(0)
            signed = torch.abs(sdf)-rnd(.13,.60)
        else:
            # Slightly curved hull groove, with fractional-pixel phase.
            aligned = dx * ct + dy * st
            tangent = -dx*st + dy*ct
            signed = torch.abs(aligned + rnd(.002,.025)*tangent.square()/native_patch) - rnd(.14,.55)

        # Exact 4x reference edge coverage; do not reconstruct it from LR.
        coverage = (.5 - signed * 4.0).clamp(0, 1)
        # A secondary repeated detail line tests two parallel subpixel grooves.
        if kind in (0,3):
            second = (.5 - (signed + rnd(1.1, 2.4)) * 4.0).clamp(0,1)
            coverage = torch.maximum(coverage, second*.7)

        bg = torch.tensor([rnd(.12,.36),rnd(.15,.42),rnd(.19,.48)]).view(3,1,1)
        edge = torch.tensor([rnd(.08,.50),rnd(.09,.48),rnd(.11,.55)]).view(3,1,1)
        # Continuous faint material texture: the target must preserve it.
        texture = (.008*torch.sin(x*.23+y*.12)+.006*torch.cos(x*.09-y*.20)).unsqueeze(0)
        rgb = (bg + edge*coverage.unsqueeze(0) + texture).clamp(0,1)

        gy, gx = torch.gradient(signed, spacing=(.25,.25))
        magnitude = (gx.square()+gy.square()+1e-6).sqrt()
        nx = (.38*coverage*gx/magnitude).unsqueeze(0)
        ny = (.38*coverage*gy/magnitude).unsqueeze(0)
        normal = torch.cat((nx,ny), dim=0)
        # "legacy_mask" only: first material channel is continuous coverage.
        material = torch.stack((
            coverage,
            (.09 + .6*coverage).clamp(0,1),
            (.6 - .18*coverage).clamp(0,1),
        ), dim=0)
        target_a.append(rgb)
        target_n.append(normal)
        target_m.append(material)
        preserve.append((signed.abs() > 1.25).unsqueeze(0).float())

        # Source texels: alternate area integration and point sampling to
        # expose both antialiased and jagged authored rasterisation.
        if int(_sample((1,),g).item()*2) == 0:
            def to_native(t: torch.Tensor) -> torch.Tensor:
                return F.avg_pool2d(t.unsqueeze(0),4).squeeze(0)
        else:
            phase_x = int(_sample((1,),g).item()*4)
            phase_y = int(_sample((1,),g).item()*4)
            def to_native(t: torch.Tensor) -> torch.Tensor:
                return t[:,phase_y::4,phase_x::4]
        source_a.append((to_native(rgb)*255).round().clamp(0,255)/255)
        source_n.append(((to_native(normal)+1)*127.5).round().clamp(0,255)/127.5-1)
        source_m.append((to_native(material)*255).round().clamp(0,255)/255)

    return tuple(torch.stack(z) for z in (
        source_a, source_n, source_m, target_a, target_n, target_m, preserve
    ))
