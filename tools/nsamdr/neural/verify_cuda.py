#!/usr/bin/env python3
"""Verify that the NSAMDR Python environment can execute on an NVIDIA GPU."""
from __future__ import annotations

import argparse
import sys
import time


def _status(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--require-arch", default="sm_120")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the lightweight training-start CUDA execution test.",
    )
    args = parser.parse_args()

    _status("[cuda-preflight] Python process entered.")
    _status("[cuda-preflight] Importing PyTorch/CUDA runtime DLLs...")
    import_started = time.perf_counter()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in this Python environment."
        ) from exc
    _status(
        f"[cuda-preflight] PyTorch import complete in "
        f"{time.perf_counter() - import_started:.1f}s."
    )

    _status(f"PyTorch version: {torch.__version__}")
    _status(f"PyTorch CUDA runtime: {torch.version.cuda or 'none'}")

    _status("[cuda-preflight] Querying CUDA driver...")
    query_started = time.perf_counter()
    cuda_available = torch.cuda.is_available()
    _status(
        f"[cuda-preflight] CUDA driver query complete in "
        f"{time.perf_counter() - query_started:.1f}s."
    )
    _status(f"CUDA available: {cuda_available}")
    if not cuda_available:
        print(
            "ERROR: This is a CPU-only or unusable CUDA PyTorch installation.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    device_count = torch.cuda.device_count()
    if args.device_index < 0 or args.device_index >= device_count:
        print(
            f"ERROR: CUDA device index {args.device_index} is invalid "
            f"(device_count={device_count}).",
            file=sys.stderr,
            flush=True,
        )
        return 3

    device = torch.device(f"cuda:{args.device_index}")
    _status(f"[cuda-preflight] Initializing CUDA context on {device}...")
    context_started = time.perf_counter()
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    # mem_get_info forces a real runtime/context query. Free VRAM is a
    # device-wide measurement; occupied memory may belong to any application.
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    _status(
        f"[cuda-preflight] CUDA context ready in "
        f"{time.perf_counter() - context_started:.1f}s."
    )

    capability = f"sm_{props.major}{props.minor}"
    _status(f"CUDA device: {device} - {props.name}")
    _status(f"Compute capability: {props.major}.{props.minor} ({capability})")
    _status(f"VRAM: {props.total_memory / (1024 ** 3):.1f} GiB")
    _status(
        f"VRAM currently free: {free_bytes / (1024 ** 3):.1f} GiB / "
        f"{total_bytes / (1024 ** 3):.1f} GiB"
    )

    free_fraction = free_bytes / max(float(total_bytes), 1.0)
    if free_fraction < 0.50:
        _status(
            "[cuda-preflight] GPU is already shared: less than 50% of VRAM is "
            "currently free. V9 elastic training will start conservatively and "
            "react to device-wide memory pressure; no assumption is made about "
            "which application owns the occupied VRAM."
        )

    _status("[cuda-preflight] Checking compiled CUDA architectures...")
    arch_started = time.perf_counter()
    architectures = torch.cuda.get_arch_list()
    _status(
        f"[cuda-preflight] Architecture query complete in "
        f"{time.perf_counter() - arch_started:.1f}s."
    )
    _status(f"Compiled architectures: {' '.join(architectures)}")

    if args.require_arch and args.require_arch not in architectures:
        print(
            f"ERROR: The installed PyTorch wheel does not include "
            f"{args.require_arch}. Install the CUDA 12.8 wheel using "
            "scripts\\build\\nsamdr.bat setup cuda.",
            file=sys.stderr,
            flush=True,
        )
        return 4
    if (
        capability not in architectures
        and f"compute_{props.major}{props.minor}" not in architectures
    ):
        print(
            f"ERROR: No executable kernel target for {capability}.",
            file=sys.stderr,
            flush=True,
        )
        return 5

    dimension = 256 if args.quick else 2048
    mode = "lightweight" if args.quick else "full"
    _status(
        f"[cuda-preflight] Running {mode} CUDA execution test "
        f"({dimension}x{dimension} matrix multiply)..."
    )
    test_started = time.perf_counter()
    a = torch.randn((dimension, dimension), device=device)
    b = torch.randn((dimension, dimension), device=device)
    c = a @ b
    checksum = float(c.mean().cpu())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - test_started
    _status(
        f"[cuda-preflight] CUDA execution test complete in {elapsed:.2f}s."
    )
    _status(f"CUDA matrix test passed; checksum={checksum:.6f}")
    _status("[cuda-preflight] Preflight complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
