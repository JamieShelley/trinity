#!/usr/bin/env python3
"""Dispatch non-promotable V14.3 mini diagnostics.

Capacity has its own executable. This dispatcher exposes only the multi-region and
selector stages and delegates implementation to their dedicated OOP classes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.diagnostic_support import archive_run, device_from_name
    from v14.multiregion_diagnostic import MultiRegionDiagnostic
    from v14.selector_diagnostic import SelectorRetentionDiagnostic
else:
    from .diagnostic_support import archive_run, device_from_name
    from .multiregion_diagnostic import MultiRegionDiagnostic
    from .selector_diagnostic import SelectorRetentionDiagnostic


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V14.3 focused Raven mini diagnostics")
    p.add_argument("--mode", choices=("multiregion", "selector"), required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--rebuild-dataset", action="store_true")
    p.add_argument("--prepare-train-regions", type=int, default=16)
    p.add_argument("--prepare-validation-regions", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")

    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)
    p.add_argument("--train-regions", type=int, default=4)
    p.add_argument("--validation-regions", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tiles-per-epoch", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)

    p.add_argument("--selector-epochs", type=int, default=2)
    p.add_argument("--selector-learning-rate", type=float, default=2.0e-4)
    p.add_argument("--required-retention", type=float, default=0.90)
    p.add_argument("--protected-preservation", type=float, default=0.99)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    device = device_from_name(args.device)
    diagnostic = (
        MultiRegionDiagnostic(args, repo_root, device)
        if args.mode == "multiregion"
        else SelectorRetentionDiagnostic(args, repo_root, device)
    )
    code, run_dir = diagnostic.run()
    archive = archive_run(run_dir)
    print(f"[v14.3-mini] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v14.3-mini] diagnostics : {archive}", flush=True)
    print(f"[v14.3-mini] result      : {'PASS' if code == 0 else 'FAIL'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
