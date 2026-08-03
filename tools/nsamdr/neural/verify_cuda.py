#!/usr/bin/env python3
"""Verify that the NSAMDR Python environment can execute on an NVIDIA GPU."""
from __future__ import annotations

import argparse
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is not installed in this Python environment.") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--require-arch", default="sm_120")
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'none'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: This is a CPU-only or unusable CUDA PyTorch installation.", file=sys.stderr)
        return 2
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        print(f"ERROR: CUDA device index {args.device_index} is invalid.", file=sys.stderr)
        return 3

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    capability = f"sm_{props.major}{props.minor}"
    architectures = torch.cuda.get_arch_list()
    print(f"CUDA device: {device} - {props.name}")
    print(f"Compute capability: {props.major}.{props.minor} ({capability})")
    print(f"VRAM: {props.total_memory / (1024 ** 3):.1f} GiB")
    print(f"Compiled architectures: {' '.join(architectures)}")

    if args.require_arch and args.require_arch not in architectures:
        print(
            f"ERROR: The installed PyTorch wheel does not include {args.require_arch}. "
            "Install the CUDA 12.8 wheel using scripts\\build\\setup_nsamdr_cuda.bat.",
            file=sys.stderr,
        )
        return 4
    if capability not in architectures and f"compute_{props.major}{props.minor}" not in architectures:
        print(f"ERROR: No executable kernel target for {capability}.", file=sys.stderr)
        return 5

    a = torch.randn((2048, 2048), device=device)
    b = torch.randn((2048, 2048), device=device)
    c = a @ b
    checksum = float(c.mean().cpu())
    torch.cuda.synchronize(device)
    print(f"CUDA matrix test passed; checksum={checksum:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
