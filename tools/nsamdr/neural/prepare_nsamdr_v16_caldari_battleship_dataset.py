#!/usr/bin/env python3
"""V16 Stage 2 data authority using four distinct Caldari battleship families.

The proven V16 model, losses and qualification rules are unchanged.  This module
only broadens Stage 2 authored evidence.  It reuses the bounded multi-family
preparation path and replaces Raven-only family discovery with deterministic
Caldari-battleship discovery.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
if str(NSAMDR_ROOT) not in sys.path:
    sys.path.insert(0, str(NSAMDR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eve_asset_test as eve  # type: ignore
import prepare_nsamdr_v16_multifamily_dataset as base
import prepare_nsamdr_v16_raven_dataset as spatial
from discover_nsamdr_v16_caldari_battleship_diversity import _candidate_entries


REQUIRED_AUTHORED_FAMILIES = 4
DATASET_SCHEMA = "NSAMDR_BATTLESHIP_DEVELOPMENT_DATASET_V7_MULTI_FAMILY_SPATIAL_DOMAIN_DISJOINT"
CROP_SCHEMA = "NSAMDR_BATTLESHIP_DEVELOPMENT_CROP_V7_MULTI_FAMILY_SPATIAL_DOMAIN_DISJOINT"
BUILDER_VERSION = "caldari-battleship-native-authored-multi-family-spatial-domain-v7"


def _select_distinct_assets(
    app: spatial.V16RavenDatasetPreparationApplication,
    rows: list[eve.ResourceRow],
    repo_root: Any,
) -> list[eve.ShipCatalogEntry]:
    entries = eve._build_sde_ship_catalog(rows, repo_root)  # noqa: SLF001
    anchor = app._find_navy_raven(rows, repo_root)
    selected: list[eve.ShipCatalogEntry] = []
    seen_pairs: set[tuple[str, str]] = set()

    for entry in _candidate_entries(entries, anchor, 64):
        try:
            pair = base._texture_pair_identity(rows, entry)  # noqa: SLF001
        except Exception as exc:
            print(
                f"[v16-battleship] SKIP {entry.display_name}: "
                f"texture authority failed: {exc}",
                flush=True,
            )
            continue
        if pair is None or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        selected.append(entry)
        print(
            f"[v16-battleship] Candidate family {len(selected)}/{REQUIRED_AUTHORED_FAMILIES}: "
            f"{entry.display_name}",
            flush=True,
        )
        if len(selected) >= REQUIRED_AUTHORED_FAMILIES:
            break

    if len(selected) < REQUIRED_AUTHORED_FAMILIES:
        raise RuntimeError(
            "V16 Stage 2 requires at least four distinct native Caldari battleship "
            f"albedo+normal families; found {len(selected)}. Run battleship-diversity first."
        )
    return selected


# Patch only the Stage 2 data-authority globals used by the existing bounded builder.
base.DATASET_SCHEMA = DATASET_SCHEMA
base.CROP_SCHEMA = CROP_SCHEMA
base.BUILDER_VERSION = BUILDER_VERSION
base.MINIMUM_AUTHORED_FAMILIES = REQUIRED_AUTHORED_FAMILIES
base._select_distinct_assets = _select_distinct_assets  # type: ignore[attr-defined]

prepare = base.prepare


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
