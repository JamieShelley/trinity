from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import V14Config


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(dim=1, keepdim=True)
    kernel = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1)


def _pyramid_l1(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = []
    for scale in (2, 4):
        values.append(F.l1_loss(
            F.avg_pool2d(candidate.float(), scale, scale),
            F.avg_pool2d(target.float(), scale, scale),
        ))
    return sum(values) / len(values)


def candidate_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V14Config,
) -> dict[str, torch.Tensor]:
    ca = outputs["candidate_albedo"].float()
    cn = outputs["candidate_normal"].float()
    cm = outputs["candidate_material"].float()
    ba = outputs["baseline_albedo"].detach().float()
    bn = outputs["baseline_normal"].detach().float()
    bm = outputs["baseline_material"].detach().float()
    ta = batch["target_albedo"].float()
    tn = batch["target_normal"].float()
    tm = batch["target_material"].float()

    reconstruction = F.l1_loss(ca, ta)
    cgx, cgy = _gradient(ca)
    tgx, tgy = _gradient(ta)
    gradient = (cgx - tgx).abs().mean() + (cgy - tgy).abs().mean()
    laplacian = F.l1_loss(_laplacian(ca), _laplacian(ta))
    pyramid = _pyramid_l1(ca, ta)

    target_residual_a = (ta - ba).clamp(-config.albedo_residual_cap, config.albedo_residual_cap)
    target_residual_n = (tn - bn).clamp(-config.normal_residual_cap, config.normal_residual_cap)
    target_residual_m = (tm - bm).clamp(-config.material_residual_cap, config.material_residual_cap)
    residual = (
        F.l1_loss(outputs["candidate_residual_albedo"].float(), target_residual_a)
        + 0.25 * F.l1_loss(outputs["candidate_residual_normal"].float(), target_residual_n)
        + 0.25 * F.l1_loss(outputs["candidate_residual_material"].float(), target_residual_m)
    )
    normal = F.l1_loss(cn, tn)
    material = F.l1_loss(cm, tm)

    total = (
        reconstruction
        + 0.50 * gradient
        + 0.25 * laplacian
        + 0.25 * pyramid
        + residual
        + 0.25 * normal
        + 0.25 * material
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "gradient": gradient,
        "laplacian": laplacian,
        "pyramid": pyramid,
        "residual": residual,
        "normal": normal,
        "material": material,
    }


def selector_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    b = outputs["baseline_albedo"].detach().float()
    c = outputs["candidate_albedo"].detach().float()
    f = outputs["albedo"].float()
    target = batch["target_albedo"].float()
    logits = outputs["selector_logits"].float()

    b_error = (b - target).abs().mean(dim=1, keepdim=True)
    c_error = (c - target).abs().mean(dim=1, keepdim=True)
    oracle = (c_error + 1.0e-5 < b_error).float()
    protected = ((b - target).abs().amax(dim=1, keepdim=True) <= (2.0 / 255.0))
    excessive_candidate_drift = ((c - b).abs().amax(dim=1, keepdim=True) > (1.0 / 255.0))
    oracle = torch.where(protected & excessive_candidate_drift, torch.zeros_like(oracle), oracle)

    classification = F.binary_cross_entropy_with_logits(logits, oracle)
    reconstruction = F.l1_loss(f, target)
    fgx, fgy = _gradient(f)
    tgx, tgy = _gradient(target)
    gradient = (fgx - tgx).abs().mean() + (fgy - tgy).abs().mean()
    total = classification + reconstruction * 2.0 + gradient * 0.25
    return {
        "total": total,
        "classification": classification,
        "reconstruction": reconstruction,
        "gradient": gradient,
        "oracle_gate_mean": oracle.mean().detach(),
    }
