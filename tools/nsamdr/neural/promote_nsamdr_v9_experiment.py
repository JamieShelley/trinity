#!/usr/bin/env python3
"""Promote an immutable Raven tuning experiment configuration to production."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.experiments import promote_experiment, selected_promotion


class ExperimentPromotionApplication:
    # Purpose: Implement main for ExperimentPromotionApplication.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def main(self) -> int:
        parser = argparse.ArgumentParser(description="Promote NSAMDR V9 tuning experiment to full training")
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--experiment")
        parser.add_argument(
            "--print-selected-config",
            action="store_true",
            help="Print only the currently promoted full-config path and exit.",
        )
        parser.add_argument(
            "--full-base-config",
            type=Path,
            default=Path("tools/nsamdr/neural/configs/v9_fidelity_full.json"),
        )
        args = parser.parse_args()
        root = args.repo_root.resolve()
        if args.print_selected_config:
            record = selected_promotion(root)
            if not record:
                raise RuntimeError("no NSAMDR V9 tuning experiment has been promoted")
            raw = str(record.get("promotedConfig") or "")
            if not raw:
                raise RuntimeError("selected promotion record has no promotedConfig")
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise RuntimeError(f"selected promoted config is missing: {path}")
            print(path.resolve())
            return 0

        if not args.experiment:
            parser.error("--experiment is required unless --print-selected-config is used")
        base = args.full_base_config if args.full_base_config.is_absolute() else root / args.full_base_config
        config_path, record = promote_experiment(root, args.experiment, full_base_config_path=base.resolve())
        print("=" * 64)
        print("NSAMDR V9 TUNING CONFIG PROMOTED")
        print(f"Source experiment         : {record['sourceExperiment']}")
        print(f"Promoted config           : {config_path}")
        print("Semantic hyperparameters  : preserved exactly")
        print("Production-only data/work : restored from full base config")
        print("Full training may now be launched from the promoted config.")
        print("=" * 64)
        return 0

_experiment_promotion_application = ExperimentPromotionApplication()
main = _experiment_promotion_application.main


if __name__ == "__main__":
    raise SystemExit(main())
