from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .dataset import _canonical_native_family
from .inference import tiled_inference


def _batch(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.transpose(2, 0, 1).copy()).unsqueeze(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bake a V14 4x physical-map preview from an immutable checkpoint")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--target-size", type=int, default=4096)
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    experiment = root / "artifacts" / "nsamdr" / "experiments" / args.experiment
    final_manifest = json.loads((experiment / "final_manifest.json").read_text(encoding="utf-8"))
    checkpoint = experiment / final_manifest["checkpoint"]["path"]
    device = torch.device("cuda" if args.device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint, device)
    manifest = json.loads((root / model.config.dataset_manifest).read_text(encoding="utf-8"))
    family = manifest["families"][0]
    canonical = _canonical_native_family(family)
    if canonical is None:
        raise RuntimeError("V14 preview could not resolve the first native physical-map family")
    albedo, normal, material = canonical
    lr_a = _batch(albedo).to(device)
    lr_n = _batch(normal).to(device)
    lr_m = _batch(material).to(device)
    model.eval()
    outputs = tiled_inference(
        model, lr_a, lr_n, lr_m,
        tile_lr=model.config.production_tile_lr,
        overlap_lr=model.config.production_overlap_lr,
    )
    natural_target = int(albedo.shape[1] * model.config.scale)
    if int(args.target_size) != natural_target:
        print(
            f"[v14-preview] requested target-size={args.target_size}; V14 production scale is fixed 4x, "
            f"so native {albedo.shape[1]} produces {natural_target}",
            flush=True,
        )
    out = experiment / "previews" / "final_v14"
    out.mkdir(parents=True, exist_ok=True)
    value = outputs["albedo"][0].detach().cpu()
    image = np.round(value.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(out / "albedo_4x.png"), image[:, :, ::-1])
    np.save(out / "normal_xy_4x.npy", outputs["normal"][0].detach().cpu().numpy())
    value = outputs["material"][0].detach().cpu()
    image = np.round(value.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(out / "material_4x.png"), image[:, :, ::-1])
    print(f"[v14-preview] baked native {albedo.shape[1]} -> {outputs['albedo'].shape[-1]} physical maps: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
