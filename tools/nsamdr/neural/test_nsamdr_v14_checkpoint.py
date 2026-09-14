#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.checkpoint import load_checkpoint
from v14.model import MODEL_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one NSAMDR V14.4 checkpoint")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing V14.4 checkpoint: {checkpoint}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA checkpoint validation requested but CUDA is unavailable")
    device = torch.device(args.device)
    model, payload = load_checkpoint(checkpoint, device)
    if payload.get("schema") != MODEL_SCHEMA:
        raise SystemExit("V14.4 checkpoint schema mismatch")

    size = 16
    albedo = torch.rand(1, 3, size, size, device=device)
    normal = torch.rand(1, 2, size, size, device=device) * 0.6 - 0.3
    material = torch.rand(1, 3, size, size, device=device)
    with torch.no_grad():
        output = model(albedo, normal, material)
    expected = size * 4
    for key in (
        "baseline_albedo",
        "candidate_albedo",
        "albedo",
        "normal",
        "material",
        "predicted_residual_albedo",
        "predicted_residual_normal",
        "predicted_residual_material",
    ):
        if tuple(output[key].shape[-2:]) != (expected, expected):
            raise SystemExit(
                f"V14.4 checkpoint output shape mismatch for {key}: {tuple(output[key].shape)}"
            )
        if not bool(torch.isfinite(output[key]).all().item()):
            raise SystemExit(f"V14.4 checkpoint produced non-finite output: {key}")

    contract = model.architecture_contract()
    if (
        contract.get("schema") != MODEL_SCHEMA
        or contract.get("revision") != "V14.4"
        or tuple(contract.get("retiredComponents", ())) != ()
        or contract.get("pixelShuffleUsed") is not False
        or contract.get("transposedConvolutionUsed") is not False
        or contract.get("residualBounding") != "straight-through-clamp"
        or contract.get("residualSupervision")
        != "bounded-pre-physical-projection"
        or abs(float(contract.get("residualGroupScale", -1.0)) - 0.10) > 1.0e-9
        or contract.get("identityInitializedDeepResiduals") is not True
    ):
        raise SystemExit("V14.4 architecture contract mismatch")

    if any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()):
        raise SystemExit("V14.4 checkpoint unexpectedly contains PixelShuffle")
    if any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules()):
        raise SystemExit("V14.4 checkpoint unexpectedly contains ConvTranspose2d")

    print("NSAMDR V14.4 checkpoint validation passed")
    print(f"  checkpoint={checkpoint}")
    print(f"  schema={MODEL_SCHEMA}")
    print(f"  parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"  phase={payload.get('phase')}")
    print(f"  epoch={payload.get('epoch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
