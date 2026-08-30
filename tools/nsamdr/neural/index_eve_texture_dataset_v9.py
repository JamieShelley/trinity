#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from v9.config import V9Config
from v9.dataset import prepare_dataset


class EveTextureDatasetIndexApplication:
    # Purpose: Implement main for EveTextureDatasetIndexApplication.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def main(self) -> int:
        parser = argparse.ArgumentParser(description="Build NSAMDR V9 authored texture crop bundles")
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--config", type=Path)
        parser.add_argument("--shared-cache")
        parser.add_argument("--source-root", type=Path)
        parser.add_argument("--rebuild", action="store_true")
        parser.add_argument("--max-families", type=int)
        parser.add_argument("--crops-per-family", type=int)
        args = parser.parse_args()
        config = V9Config.load(args.config.resolve() if args.config else None)
        if args.max_families is not None:
            config.max_families = args.max_families
        if args.crops_per_family is not None:
            config.crops_per_family = args.crops_per_family
        config.validate()
        prepare_dataset(
            args.repo_root.resolve(), config, shared_cache=args.shared_cache,
            source_root=args.source_root, rebuild=args.rebuild)
        return 0

_eve_texture_dataset_index_application = EveTextureDatasetIndexApplication()
main = _eve_texture_dataset_index_application.main


if __name__ == "__main__":
    raise SystemExit(main())
