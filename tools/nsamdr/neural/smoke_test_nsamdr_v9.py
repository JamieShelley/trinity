#!/usr/bin/env python3
"""Historical command shim: smoke-test the V14 production architecture."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import V14Config
from v14.losses import candidate_loss
from v14.model import MODEL_SCHEMA, NSAMDRV14


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    config = V14Config(
        train_lr_size=32,
        train_hr_size=128,
        validation_lr_size=32,
        validation_hr_size=128,
        lr_context_channels=12,
        lr_blocks=2,
        hr_channels=12,
        hr_blocks=2,
        selector_channels=8,
        production_tile_lr=32,
        production_overlap_lr=4,
        tiles_per_epoch=1,
        validation_tiles=1,
        clean_epochs=1,
        robust_epochs=1,
        selector_epochs=1,
    )
    model = NSAMDRV14(config).to(device)
    contract = model.architecture_contract()
    if contract["schema"] != MODEL_SCHEMA or tuple(contract["retiredComponents"]) != ():
        raise SystemExit("V14 clean architecture contract failed")
    torch.manual_seed(14)
    albedo = torch.rand(1, 3, 32, 32, device=device)
    normal = torch.rand(1, 2, 32, 32, device=device) * 0.8 - 0.4
    material = torch.rand(1, 3, 32, 32, device=device)
    outputs = model(albedo, normal, material)
    if tuple(outputs["albedo"].shape[-2:]) != (128, 128):
        raise SystemExit("V14 model did not perform exact 4x reconstruction")
    if not torch.equal(outputs["candidate_albedo"], outputs["baseline_albedo"]):
        raise SystemExit("V14 zero-init candidate is not exact baseline B")
    batch = {
        "target_albedo": outputs["baseline_albedo"].detach(),
        "target_normal": outputs["baseline_normal"].detach(),
        "target_material": outputs["baseline_material"].detach(),
    }
    losses = candidate_loss(outputs, batch, config)
    if not torch.isfinite(losses["total"]):
        raise SystemExit("V14 candidate loss is non-finite")
    print("NSAMDR V14 architecture smoke test passed")
    print(f"  device={device}")
    print(f"  parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"  schema={MODEL_SCHEMA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
