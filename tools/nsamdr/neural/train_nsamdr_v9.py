#!/usr/bin/env python3
"""Train and expose the NSAMDR V9 fidelity-first residual pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(r"Missing PyTorch. Run scripts\build\nsamdr.bat setup cuda") from exc

from v9.config import V9Config
from v9.dataset import PhysicalTileDatasetV9, load_dataset_manifest, prepare_dataset
from v9.inference import infer_tiled, load_trained_model, resolve_device
from v9.losses import compute_losses
from v9.model import (
    INPUT_CHANNELS, MODEL_SCHEMA, UPSCALE_FACTOR, FidelityResidualNetV9,
    architecture_summary, build_model_input, model_hash, parameter_count,
)
from v9.training import train_v9

TrainingConfig = V9Config
MaterialTileContextNet = FidelityResidualNetV9
CachePBRNetV9 = FidelityResidualNetV9


def _semantic_maps_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb[..., :3], 0.0, 1.0)
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    gy, gx = np.gradient(luma)
    length = np.sqrt(gx * gx + gy * gy + 1e-8)
    limiter = np.maximum(1.0, length / 0.999)
    normal_x, normal_y = gx / limiter, gy / limiter
    contrast = np.clip(rgb.max(axis=-1) - rgb.min(axis=-1), 0.0, 1.0)
    emissive = np.clip((luma - 0.72) * 3.5, 0.0, 1.0)
    roughness = np.clip(0.78 - luma * 0.42 + length * 0.15, 0.0, 1.0)
    return np.stack((normal_x, normal_y, contrast, emissive, roughness), axis=-1).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train NSAMDR V9 fidelity-first residual reconstruction")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--performance-profile", choices=("fast", "optimized", "tuned", "compiled", "channels-last", "balanced", "compatibility"))
    parser.add_argument("--tuning-file", type=Path)
    parser.add_argument("--loss-precision", choices=("mixed", "fp32"))
    parser.add_argument("--torch-compile", choices=("off", "default", "reduce-overhead", "max-autotune"))
    parser.add_argument("--channels-last", choices=("auto", "on", "off"))
    parser.add_argument("--optimizer-kernel", choices=("auto", "fused", "foreach"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--amp-precision", choices=("auto", "fp16", "bf16"))
    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--prepare-dataset-only", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--shared-cache")
    parser.add_argument("--source-root", type=Path)
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--auto", action="store_true")
    state.add_argument("--resume", action="store_true")
    state.add_argument("--restart", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    config = V9Config.load(args.config.resolve() if args.config else None)
    if args.device: config.device = args.device
    if args.performance_profile: config.apply_performance_profile(args.performance_profile)
    if args.workers is not None: config.data_loader_workers = args.workers
    if args.prefetch_factor is not None: config.data_loader_prefetch_factor = args.prefetch_factor
    if args.amp_precision: config.amp_dtype = args.amp_precision
    if args.tuning_file:
        tuning_path = args.tuning_file if args.tuning_file.is_absolute() else repo_root / args.tuning_file
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
        selected = payload.get("selected", payload)
        mapping = {
            "workers": "data_loader_workers", "prefetchFactor": "data_loader_prefetch_factor",
            "persistentWorkers": "data_loader_persistent_workers", "cudaPrefetch": "cuda_prefetch",
            "channelsLast": "channels_last", "ampDtype": "amp_dtype", "fusedOptimizer": "fused_optimizer",
            "cudnnBenchmark": "cudnn_benchmark", "allowTf32": "allow_tf32",
            "lossPrecision": "loss_precision", "torchCompileMode": "torch_compile_mode",
        }
        for key, target in mapping.items():
            if key in selected: setattr(config, target, selected[key])
        config.performance_profile = "tuned"
    if args.loss_precision: config.loss_precision = args.loss_precision
    if args.torch_compile: config.torch_compile_mode = args.torch_compile
    if args.channels_last and args.channels_last != "auto": config.channels_last = args.channels_last == "on"
    if args.optimizer_kernel and args.optimizer_kernel != "auto": config.fused_optimizer = args.optimizer_kernel == "fused"
    config.validate()
    if args.prepare_dataset or args.prepare_dataset_only:
        prepare_dataset(repo_root, config, shared_cache=args.shared_cache, source_root=args.source_root, rebuild=args.rebuild_dataset)
    if args.prepare_dataset_only:
        return 0

    # Safe default: auto-resume. A missing state means a genuinely fresh run;
    # an existing state is never silently discarded.
    auto_control = bool(args.auto or (not args.resume and not args.restart))
    effective_resume = bool(args.resume)
    effective_restart = bool(args.restart)

    if auto_control:
        output_dir = (repo_root / config.output_dir).resolve()
        state_path = output_dir / config.training_state_name
        checkpoint_path = output_dir / config.checkpoint_name

        if state_path.is_file():
            effective_resume = True
            print(
                f"[startup] Auto training control: resumable state found: {state_path}",
                flush=True,
            )
            print("[startup] Auto training control: RESUME selected.", flush=True)
        elif checkpoint_path.is_file():
            raise RuntimeError(
                "V9 auto training control found a final checkpoint but no "
                f"training state: {checkpoint_path}. Refusing to overwrite it "
                "automatically. Use --restart only when a fresh run is intentional."
            )
        else:
            print(
                "[startup] Auto training control: no training state found; "
                "starting a fresh run without destructive cleanup.",
                flush=True,
            )

    train_v9(
        config,
        repo_root,
        args.device,
        resume=effective_resume,
        restart=effective_restart,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
