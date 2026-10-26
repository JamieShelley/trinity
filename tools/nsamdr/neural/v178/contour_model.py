"""G2 structural contour head: native physical evidence -> smooth 4× coverage.

This predicts a geometry representation, NOT sharpened RGB. The historical
backbone and synthetic-only V17.7 specialist remain untouched.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "NSAMDR_V178_CONTOUR_HEAD_V2_AUTO_EVIDENCE"
SCALE = 4


def physical_edge_evidence(
    albedo: torch.Tensor, normal: torch.Tensor,
) -> torch.Tensor:
    """Native geometry observation, not a high-resolution training target.

    Shared between analytic training inputs and authentic EVE inputs.
    Derivative magnitude is computed in NATIVE texels, without using any
    high-resolution target or V5 categorical material selector as an edge.
    """
    if (albedo.ndim != 4 or normal.ndim != 4 or albedo.shape[1] != 3
            or normal.shape[1] != 2 or albedo.shape[0] != normal.shape[0]
            or albedo.shape[-2:] != normal.shape[-2:]):
        raise ValueError("aligned native RGB and normal XY required")
    lum=(albedo[:,0:1]*.2126+albedo[:,1:2]*.7152
         +albedo[:,2:3]*.0722)
    def magnitude(signal: torch.Tensor) -> torch.Tensor:
        # Padded symmetric differences approximate boundaries at their
        # native coordinates; do not infer a fake clean 4x contour.
        gx=torch.nn.functional.pad(signal[:,:,:,2:]-signal[:,:,:,:-2],
                                  (1,1,0,0),mode="replicate")*.5
        gy=torch.nn.functional.pad(signal[:,:,2:,:]-signal[:,:,:-2,:],
                                  (0,0,1,1),mode="replicate")*.5
        return torch.sqrt(gx.square()+gy.square()+1.e-10)
    colour=magnitude(lum)
    normal_edges=magnitude(normal).mean(dim=1,keepdim=True)
    return torch.clamp(3.0*colour+1.2*normal_edges,0,1)


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + .2 * self.conv2(F.silu(self.conv1(value)))


class StructuralContourHead(nn.Module):
    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, width, 3, padding=1), nn.SiLU(),
            Residual(width), Residual(width), Residual(width),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=SCALE, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
        )
        self.delta = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self, albedo: torch.Tensor, normal: torch.Tensor,
        material: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (albedo.ndim != 4 or albedo.shape[1] != 3
            or normal.shape[1] != 2 or material.shape[1] != 3
            or normal.ndim != 4 or material.ndim != 4
            or albedo.shape[0] != normal.shape[0]
            or albedo.shape[0] != material.shape[0]
            or albedo.shape[-2:] != normal.shape[-2:]
            or albedo.shape[-2:] != material.shape[-2:]):
            raise ValueError("G2 contour head requires aligned native NCHW physical maps")
        # material[0] is the calibrated native edge evidence shared by
        # synthetic and real sources, NEVER a categorical V5 material ID.
        # Bilinear only as an explicitly labelled observation baseline. The
        # output corrects its geometry rather than treating it as clean truth.
        reference = F.interpolate(
            material[:, :1].float(), scale_factor=SCALE,
            mode="bilinear", align_corners=False,
        ).clamp(0, 1)
        features = self.features(torch.cat((
            albedo.float(), normal.float(), material.float()
        ), dim=1))
        # Signed ±1 range permits learning an independently traced contour
        # even when a real V5 selector is semantically unrelated to the line.
        # Initial zero residual still reproduces the observation baseline.
        correction = torch.tanh(self.delta(self.up(features)))
        candidate = (reference + correction).clamp(0, 1)
        return {"baseline_contour": reference, "candidate_contour": candidate}


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[:,:,:,1:] - value[:,:,:,:-1], value[:,:,1:,:] - value[:,:,:-1,:]


def contour_loss(
    candidate: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if candidate.shape != target.shape or candidate.ndim != 4 or candidate.shape[1] != 1:
        raise ValueError("contour target must match enhanced coverage dimensions")
    regression = F.l1_loss(candidate,target)
    cx,cy = _gradient(candidate)
    tx,ty = _gradient(target)
    tangent = .5*(F.l1_loss(cx,tx)+F.l1_loss(cy,ty))
    # Bound cross-phase grid noise: second differences must agree with the
    # analytic clean reference instead of forming four-pixel periodic blocks.
    dxx = candidate[:,:,:,2:]-2*candidate[:,:,:,1:-1]+candidate[:,:,:,:-2]
    dyy = candidate[:,:,2:,:]-2*candidate[:,:,1:-1,:]+candidate[:,:,:-2,:]
    txx = target[:,:,:,2:]-2*target[:,:,:,1:-1]+target[:,:,:,:-2]
    tyy = target[:,:,2:,:]-2*target[:,:,1:-1,:]+target[:,:,:-2,:]
    curvature = .5*(F.l1_loss(dxx,txx)+F.l1_loss(dyy,tyy))
    result = regression + .8*tangent + .3*curvature
    return result, {
        "coverage": float(regression.detach()), "tangent":float(tangent.detach()),
        "curvature":float(curvature.detach()),
    }
