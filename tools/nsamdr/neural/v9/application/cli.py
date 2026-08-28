"""Command-line parsing for the canonical NSAMDR training application."""
from __future__ import annotations

import argparse
from pathlib import Path

from .domain import TrainingOptions


def build_parser() -> argparse.ArgumentParser:
    """Build the stable Quick/Full training command-line interface.

    Purpose:
        Keep argument declarations separate from experiment/training orchestration.
    Called by:
        parse_options().
    Calls:
        argparse.ArgumentParser().
    """
    parser = argparse.ArgumentParser(
        description="Train the complete production NSAMDR model with a Quick or Full work budget"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("tools/nsamdr/neural/configs/v9_fidelity_full.json"),
        help="Production semantic config used by both Quick and Full",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="Dataset scope config; Quick normally uses v9_preview_raven.json",
    )
    parser.add_argument("--experiment", default="new", help="new or an in-progress EXP_####")
    parser.add_argument("--control", choices=("auto", "resume"), default="auto")
    parser.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--preset", default="Production")
    parser.add_argument("--tiles-per-epoch", type=int)
    parser.add_argument("--validation-tiles", type=int)
    parser.add_argument(
        "--performance-profile",
        choices=("optimized", "fast", "balanced", "compatibility"),
        default="optimized",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--allocate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop-after-phase",
        choices=(
            "sdf-bootstrap",
            "sdf-proof",
            "seam-proof",
            "seam-authority",
            "gate-proof",
            "detail-reconstruction",
            "boundary-hardening",
            "physical-finetune",
        ),
        help=argparse.SUPPRESS,
    )
    return parser


def parse_options(argv: list[str] | None = None) -> TrainingOptions:
    """Parse CLI arguments into an immutable application value object.

    Purpose:
        Remove argparse.Namespace from the internal object graph.
    Called by:
        application.main().
    Calls:
        build_parser(), TrainingOptions().
    """
    args = build_parser().parse_args(argv)
    return TrainingOptions(
        repo_root=args.repo_root.resolve(),
        base_config=args.base_config,
        dataset_config=args.dataset_config,
        experiment=str(args.experiment),
        control=str(args.control),
        training_mode=str(args.training_mode),
        preset=str(args.preset),
        tiles_per_epoch=args.tiles_per_epoch,
        validation_tiles=args.validation_tiles,
        performance_profile=str(args.performance_profile),
        workers=int(args.workers),
        prefetch_factor=int(args.prefetch_factor),
        amp_precision=str(args.amp_precision),
        device=str(args.device),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        result_file=args.result_file,
        allocate_only=bool(args.allocate_only),
        stop_after_phase=args.stop_after_phase,
    )
