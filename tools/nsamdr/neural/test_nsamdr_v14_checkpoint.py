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
from v14.refinement import SwinTransformerLayer, WindowAttention


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one NSAMDR V16.0 checkpoint")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing V16.0 checkpoint: {checkpoint}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA checkpoint validation requested but CUDA is unavailable")
    device = torch.device(args.device)
    model, payload = load_checkpoint(checkpoint, device)
    if payload.get("schema") != MODEL_SCHEMA:
        raise SystemExit("V16.0 checkpoint schema mismatch")

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
                f"V16.0 checkpoint output shape mismatch for {key}: {tuple(output[key].shape)}"
            )
        if not bool(torch.isfinite(output[key]).all().item()):
            raise SystemExit(f"V16.0 checkpoint produced non-finite output: {key}")

    contract = model.architecture_contract()
    if (
        contract.get("schema") != MODEL_SCHEMA
        or contract.get("revision") != "V16.0"
        or contract.get("backbone") != "SwinIR-style-fixed-HR-RSTB"
        or contract.get("windowAttentionUsed") is not True
        or contract.get("shiftedWindowAttentionUsed") is not True
        or contract.get("pixelShuffleUsed") is not False
        or contract.get("transposedConvolutionUsed") is not False
        or contract.get("batchNormalizationUsed") is not False
        or contract.get("multiscaleFeatureHierarchy") is not False
        or contract.get("residualBounding") != "tanh"
        or contract.get("residualSupervision") != "bounded-pre-physical-projection"
    ):
        raise SystemExit("V16.0 architecture contract mismatch")

    if not any(isinstance(module, WindowAttention) for module in model.modules()):
        raise SystemExit("V16.0 checkpoint has no WindowAttention")
    if not any(isinstance(module, SwinTransformerLayer) for module in model.modules()):
        raise SystemExit("V16.0 checkpoint has no SwinTransformerLayer")
    if any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()):
        raise SystemExit("V16.0 checkpoint unexpectedly contains PixelShuffle")
    if any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules()):
        raise SystemExit("V16.0 checkpoint unexpectedly contains ConvTranspose2d")
    if any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()):
        raise SystemExit("V16.0 checkpoint unexpectedly contains BatchNorm2d")

    print("NSAMDR V16.0 checkpoint validation passed")
    print(f"  checkpoint={checkpoint}")
    print(f"  schema={MODEL_SCHEMA}")
    print(f"  parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"  phase={payload.get('phase')}")
    print(f"  epoch={payload.get('epoch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
