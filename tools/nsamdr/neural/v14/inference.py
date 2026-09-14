from __future__ import annotations

import torch

from .baseline import normalize_xy
from .model import NSAMDRV16


def _positions(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    step = tile - overlap
    values = list(range(0, max(1, size - tile + 1), step))
    last = size - tile
    if values[-1] != last:
        values.append(last)
    return values


def _window(height: int, width: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(height, periodic=False, device=device).clamp_min(0.05)
    wx = torch.hann_window(width, periodic=False, device=device).clamp_min(0.05)
    return (wy[:, None] * wx[None, :]).view(1, 1, height, width)


@torch.no_grad()
def tiled_inference(
    model: NSAMDRV16,
    lr_albedo: torch.Tensor,
    lr_normal: torch.Tensor,
    lr_material: torch.Tensor,
    *,
    tile_lr: int,
    overlap_lr: int,
) -> dict[str, torch.Tensor]:
    if lr_albedo.shape[0] != 1:
        raise ValueError("V16 tiled inference currently expects batch size 1")
    h, w = lr_albedo.shape[-2:]
    scale = model.config.scale
    ys = _positions(h, tile_lr, overlap_lr)
    xs = _positions(w, tile_lr, overlap_lr)
    out_h, out_w = h * scale, w * scale
    keys = (
        "baseline_albedo",
        "baseline_normal",
        "baseline_material",
        "candidate_albedo",
        "candidate_normal",
        "candidate_material",
        "albedo",
        "normal",
        "material",
        "selector_probability",
    )
    channels = {
        "baseline_albedo": 3,
        "baseline_normal": 2,
        "baseline_material": 3,
        "candidate_albedo": 3,
        "candidate_normal": 2,
        "candidate_material": 3,
        "albedo": 3,
        "normal": 2,
        "material": 3,
        "selector_probability": 1,
    }
    accum = {
        key: torch.zeros(
            (1, channels[key], out_h, out_w),
            device=lr_albedo.device,
            dtype=torch.float32,
        )
        for key in keys
    }
    weight = torch.zeros(
        (1, 1, out_h, out_w),
        device=lr_albedo.device,
        dtype=torch.float32,
    )

    for y in ys:
        for x in xs:
            a = lr_albedo[..., y : y + tile_lr, x : x + tile_lr]
            n = lr_normal[..., y : y + tile_lr, x : x + tile_lr]
            m = lr_material[..., y : y + tile_lr, x : x + tile_lr]
            outputs = model(a, n, m)
            ph, pw = outputs["albedo"].shape[-2:]
            win = _window(ph, pw, lr_albedo.device)
            oy, ox = y * scale, x * scale
            for key in keys:
                accum[key][..., oy : oy + ph, ox : ox + pw] += (
                    outputs[key].float() * win
                )
            weight[..., oy : oy + ph, ox : ox + pw] += win

    result = {
        key: value / weight.clamp_min(1.0e-6)
        for key, value in accum.items()
    }
    for key in ("baseline_normal", "candidate_normal", "normal"):
        result[key] = normalize_xy(result[key])
    return result
