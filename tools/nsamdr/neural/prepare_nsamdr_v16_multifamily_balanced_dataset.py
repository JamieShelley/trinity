#!/usr/bin/env python3
"""Build the V16 Stage 2 dataset with four-family balanced split coverage."""
from __future__ import annotations

import sys
from typing import Any

import prepare_nsamdr_v16_caldari_battleship_dataset as base
import prepare_nsamdr_v16_raven_dataset as spatial


REQUIRED_AUTHORED_FAMILIES = 4


def _balanced_from_family_pools(
    application: spatial.V16RavenDatasetPreparationApplication,
    pools: dict[str, list[dict[str, Any]]],
    requested: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    target = min(sum(len(values) for values in pools.values()), max(1, int(requested)))
    queues: dict[str, list[dict[str, Any]]] = {}
    for offset, family_id in enumerate(sorted(pools)):
        queues[family_id] = application._stratified_subset(
            list(pools[family_id]),
            len(pools[family_id]),
            seed + offset * 1009,
        )

    selected: list[dict[str, Any]] = []
    while len(selected) < target:
        progressed = False
        for family_id in sorted(queues):
            queue = queues[family_id]
            if not queue or len(selected) >= target:
                continue
            selected.append(queue.pop(0))
            progressed = True
        if not progressed:
            break
    return selected


def _balanced_select_fixed_regions(
    self: spatial.V16RavenDatasetPreparationApplication,
    candidates: list[dict[str, Any]],
    *,
    max_train_crops: int,
    max_validation_crops: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not candidates:
        raise RuntimeError("V16 Stage 2 balanced split has no candidate windows")

    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        by_family.setdefault(str(item["familyId"]), []).append(item)
    if len(by_family) < REQUIRED_AUTHORED_FAMILIES:
        raise RuntimeError(
            "V16 Stage 2 balanced split requires at least four authored families; "
            f"found {len(by_family)}"
        )

    train_pools: dict[str, list[dict[str, Any]]] = {}
    validation_pools: dict[str, list[dict[str, Any]]] = {}
    self._last_domains = {}
    for family_id, items in sorted(by_family.items()):
        train, validation, domain = self._family_domain_pools(items, seed=seed)
        train_pools[family_id] = train
        validation_pools[family_id] = validation
        self._last_domains[family_id] = domain

    selected_train = _balanced_from_family_pools(
        self, train_pools, max_train_crops, seed=seed
    )
    selected_validation = _balanced_from_family_pools(
        self, validation_pools, max_validation_crops, seed=seed + 77
    )
    if not selected_train or not selected_validation:
        raise RuntimeError(
            "V16 Stage 2 balanced split could not produce training and validation windows"
        )

    crop_size = int(selected_train[0]["albedo"].shape[0])
    for train in selected_train:
        for validation in selected_validation:
            if self._rectangles_overlap(train, validation, crop_size):
                raise RuntimeError(
                    "V16 Stage 2 balanced train/validation pixel domains overlap"
                )

    train_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    for item in selected_train:
        family = str(item["familyId"])
        train_counts[family] = train_counts.get(family, 0) + 1
    for item in selected_validation:
        family = str(item["familyId"])
        validation_counts[family] = validation_counts.get(family, 0) + 1

    family_ids = sorted(by_family)
    if int(max_validation_crops) >= len(family_ids):
        missing = [family for family in family_ids if validation_counts.get(family, 0) < 1]
        if missing:
            raise RuntimeError(
                "V16 Stage 2 held-out selection does not cover every authored family: "
                f"counts={validation_counts} missing={missing}"
            )
    if int(max_train_crops) >= 2 * len(family_ids):
        missing = [family for family in family_ids if train_counts.get(family, 0) < 2]
        if missing:
            raise RuntimeError(
                "V16 Stage 2 training selection needs at least two regions per family: "
                f"counts={train_counts} missing={missing}"
            )

    print(
        "[v16-multifamily] Balanced selected regions: "
        f"train={train_counts} held-out={validation_counts}",
        flush=True,
    )
    return selected_train, selected_validation


spatial.V16RavenDatasetPreparationApplication._select_fixed_regions = (
    _balanced_select_fixed_regions
)


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    # V7 data authority and four-family selection must not reuse an old V6 manifest.
    if "--rebuild" not in values:
        values.append("--rebuild")
    return base.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
