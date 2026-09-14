#!/usr/bin/env python3
"""Build the V16 Raven development dataset with spatial-domain holdout.

The legacy Raven source extractor remains the authority for EVE asset discovery,
semantic-map decoding, and native-resolution supervision.  This V16 wrapper changes
only spatial sampling and split policy.

Training and validation pixels are separated by a hard spatial domain boundary.
Within one split, 512x512 windows may overlap so a native 1024x1024 map can provide
multiple deterministic regions without leaking any training pixel into validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import prepare_nsamdr_v9_raven_preview_dataset as legacy


DATASET_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_DATASET_V5_SPATIAL_DOMAIN_DISJOINT"
CROP_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_CROP_V5_SPATIAL_DOMAIN_DISJOINT"
BUILDER_VERSION = "raven-native-authored-spatial-domain-sliding-v5"
SLIDING_STRIDE_DIVISOR = 4
MIN_MULTI_REGION_TRAIN = 4
MIN_MULTI_REGION_VALIDATION = 4


class V16RavenDatasetPreparationApplication(
    legacy.RavenPreviewDatasetPreparationApplication
):
    """Use native Raven authority with disjoint train/validation spatial domains."""

    def __init__(self) -> None:
        super().__init__()
        self._last_domains: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _sliding_positions(length: int, crop_size: int) -> list[int]:
        if length < crop_size:
            return [0]
        last = length - crop_size
        stride = max(1, crop_size // SLIDING_STRIDE_DIVISOR)
        positions = list(range(0, last + 1, stride))
        if not positions or positions[-1] != last:
            positions.append(last)
        return sorted(set(positions))

    def _grid_positions(self, length: int, crop_size: int) -> list[int]:
        """Return deterministic sliding starts instead of one crop-size grid."""

        return self._sliding_positions(length, crop_size)

    @staticmethod
    def _stable_key(item: dict[str, Any], seed: int) -> tuple[str, str, int, int]:
        identity = (
            f"{seed}|{item['familyId']}|{item['x']}|{item['y']}|"
            f"{int(bool(item['materialValid']))}"
        )
        return (
            hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            str(item["familyId"]),
            int(item["y"]),
            int(item["x"]),
        )

    @classmethod
    def _stratified_subset(
        cls,
        pool: list[dict[str, Any]],
        requested: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        requested = min(len(pool), max(1, int(requested)))
        ordered_by_detail = sorted(
            pool,
            key=lambda item: (
                float(item["detailScore"]),
                str(item["familyId"]),
                int(item["y"]),
                int(item["x"]),
            ),
        )
        strata: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
        denominator = max(1, len(ordered_by_detail))
        for rank, item in enumerate(ordered_by_detail):
            stratum = min(3, (rank * 4) // denominator)
            item["detailStratum"] = stratum
            strata[stratum].append(item)
        for values in strata.values():
            values.sort(key=lambda item: cls._stable_key(item, seed))

        selected: list[dict[str, Any]] = []
        family_counts: dict[str, int] = {}
        for _ in range(requested + 4):
            made_progress = False
            for stratum in (0, 3, 1, 2):
                values = strata[stratum]
                if not values or len(selected) >= requested:
                    continue
                choice_index = min(
                    range(len(values)),
                    key=lambda index: (
                        family_counts.get(str(values[index]["familyId"]), 0),
                        cls._stable_key(values[index], seed),
                    ),
                )
                choice = values.pop(choice_index)
                selected.append(choice)
                family = str(choice["familyId"])
                family_counts[family] = family_counts.get(family, 0) + 1
                made_progress = True
            if len(selected) >= requested or not made_progress:
                break
        return selected

    @staticmethod
    def _rectangles_overlap(
        left: dict[str, Any],
        right: dict[str, Any],
        crop_size: int,
    ) -> bool:
        if str(left["familyId"]) != str(right["familyId"]):
            return False
        lx0, ly0 = int(left["x"]), int(left["y"])
        rx0, ry0 = int(right["x"]), int(right["y"])
        lx1, ly1 = lx0 + crop_size, ly0 + crop_size
        rx1, ry1 = rx0 + crop_size, ry0 + crop_size
        return lx0 < rx1 and rx0 < lx1 and ly0 < ry1 and ry0 < ly1

    def _family_domain_pools(
        self,
        family_items: list[dict[str, Any]],
        *,
        seed: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        sample = family_items[0]
        crop_size = int(sample["albedo"].shape[0])
        x_values = sorted({int(item["x"]) for item in family_items})
        y_values = sorted({int(item["y"]) for item in family_items})
        source_width = max(x_values) + crop_size
        source_height = max(y_values) + crop_size
        family_id = str(sample["familyId"])

        axes: list[tuple[str, int]] = []
        if source_width >= crop_size * 2:
            axes.append(("x", source_width // 2))
        if source_height >= crop_size * 2:
            axes.append(("y", source_height // 2))
        if not axes:
            raise RuntimeError(
                f"Raven family {family_id} cannot provide two pixel-disjoint "
                f"{crop_size}x{crop_size} domains from {source_width}x{source_height}"
            )

        # Prefer the axis that provides the most candidate windows after the hard split.
        choices: list[
            tuple[int, str, int, list[dict[str, Any]], list[dict[str, Any]]]
        ] = []
        for axis, boundary in axes:
            if axis == "x":
                low = [item for item in family_items if int(item["x"]) + crop_size <= boundary]
                high = [item for item in family_items if int(item["x"]) >= boundary]
            else:
                low = [item for item in family_items if int(item["y"]) + crop_size <= boundary]
                high = [item for item in family_items if int(item["y"]) >= boundary]
            if low and high:
                choices.append((min(len(low), len(high)), axis, boundary, low, high))

        if not choices:
            raise RuntimeError(
                f"Raven family {family_id} has no candidate windows on both sides of a "
                "hard spatial split"
            )

        _capacity, axis, boundary, low, high = max(
            choices,
            key=lambda value: (value[0], value[1] == "x"),
        )

        # Alternate the validation side by a stable family/seed hash.  The domains
        # remain disjoint; this avoids a permanent left/right bias when more families
        # are added later.
        swap = int(
            hashlib.sha256(f"{seed}|{family_id}|domain-side".encode("utf-8")).hexdigest(),
            16,
        ) & 1
        train_pool, validation_pool = (high, low) if swap else (low, high)

        for item in train_pool:
            item["spatialDomain"] = "train"
        for item in validation_pool:
            item["spatialDomain"] = "validation"

        if axis == "x":
            low_box = [0, 0, boundary, source_height]
            high_box = [boundary, 0, source_width, source_height]
        else:
            low_box = [0, 0, source_width, boundary]
            high_box = [0, boundary, source_width, source_height]
        train_box, validation_box = (high_box, low_box) if swap else (low_box, high_box)

        domain = {
            "familyId": family_id,
            "axis": axis,
            "boundary": boundary,
            "sourceSize": [source_width, source_height],
            "cropSize": crop_size,
            "stride": max(1, crop_size // SLIDING_STRIDE_DIVISOR),
            "trainDomain": train_box,
            "validationDomain": validation_box,
            "trainCandidateWindows": len(train_pool),
            "validationCandidateWindows": len(validation_pool),
        }
        return train_pool, validation_pool, domain

    def _select_fixed_regions(
        self,
        candidates: list[dict[str, Any]],
        *,
        max_train_crops: int,
        max_validation_crops: int,
        seed: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            raise RuntimeError("Raven V16 spatial-domain split has no candidate windows")

        by_family: dict[str, list[dict[str, Any]]] = {}
        for item in candidates:
            by_family.setdefault(str(item["familyId"]), []).append(item)

        training_pool: list[dict[str, Any]] = []
        validation_pool: list[dict[str, Any]] = []
        self._last_domains = {}
        for family_id, items in sorted(by_family.items()):
            train, validation, domain = self._family_domain_pools(items, seed=seed)
            training_pool.extend(train)
            validation_pool.extend(validation)
            self._last_domains[family_id] = domain

        selected_train = self._stratified_subset(training_pool, max_train_crops, seed)
        selected_validation = self._stratified_subset(
            validation_pool,
            max_validation_crops,
            seed + 77,
        )
        if not selected_train or not selected_validation:
            raise RuntimeError(
                "Raven V16 spatial-domain split could not produce training and validation windows"
            )

        crop_size = int(selected_train[0]["albedo"].shape[0])
        overlap_pairs: list[tuple[str, int, int, int, int]] = []
        for train in selected_train:
            for validation in selected_validation:
                if self._rectangles_overlap(train, validation, crop_size):
                    overlap_pairs.append(
                        (
                            str(train["familyId"]),
                            int(train["x"]),
                            int(train["y"]),
                            int(validation["x"]),
                            int(validation["y"]),
                        )
                    )
        if overlap_pairs:
            raise RuntimeError(
                "Raven V16 train/validation pixel domains overlap: "
                f"{overlap_pairs[:8]}"
            )
        return selected_train, selected_validation

    def prepare(
        self,
        repo_root: Path,
        config: legacy.V9Config,
        *,
        shared_cache: str,
        rebuild: bool,
        train_crops: int,
        validation_crops: int,
    ) -> dict[str, Any]:
        # Inherited code reads these module globals.  Set them before source
        # fingerprinting so V4 manifests cannot be reused as V5 datasets.
        legacy.PREVIEW_DATASET_SCHEMA = DATASET_SCHEMA
        legacy.PREVIEW_CROP_SCHEMA = CROP_SCHEMA
        legacy.BUILDER_VERSION = BUILDER_VERSION

        payload = super().prepare(
            repo_root,
            config,
            shared_cache=shared_cache,
            rebuild=rebuild,
            train_crops=train_crops,
            validation_crops=validation_crops,
        )

        split_policy = dict(payload.get("splitPolicy") or {})
        split_policy.update(
            {
                "type": "feature-stratified-spatial-domain-disjoint-sliding-v5",
                "windowStrideDivisor": SLIDING_STRIDE_DIVISOR,
                "trainValidationPixelOverlap": False,
                "intraSplitWindowOverlapAllowed": True,
                "domainRule": "per-family-hard-axis-bisection",
                "minimumMultiRegionTrainCrops": MIN_MULTI_REGION_TRAIN,
                "minimumMultiRegionValidationCrops": MIN_MULTI_REGION_VALIDATION,
                "spatialDomains": list(self._last_domains.values())
                or list(split_policy.get("spatialDomains") or []),
            }
        )
        payload["splitPolicy"] = split_policy
        payload["schema"] = DATASET_SCHEMA
        payload["builderVersion"] = BUILDER_VERSION

        for record in payload.get("crops", []):
            if isinstance(record, dict):
                record["spatial_domain"] = str(record.get("split") or "")
                record["train_validation_pixel_overlap"] = False

        for family in payload.get("families", []):
            if not isinstance(family, dict):
                continue
            family.pop("nonOverlappingGridCells", None)
            selected = list(family.get("selectedRegions") or [])
            family["slidingWindowCandidates"] = len(selected)
            domain = self._last_domains.get(str(family.get("familyId") or ""))
            if domain is not None:
                family["spatialDomain"] = domain

        manifest_path = (repo_root.resolve() / config.dataset_manifest).resolve()
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "[v16-dataset] Split policy: hard pixel-disjoint train/validation domains; "
            "512x512 windows may overlap only inside the same split.",
            flush=True,
        )
        print(
            f"[v16-dataset] Selected windows: train={payload.get('counts', {}).get('trainCrops', 0)} "
            f"held-out={payload.get('counts', {}).get('validationCrops', 0)}",
            flush=True,
        )
        return payload

    def main(self) -> int:
        # Reuse the legacy command surface.  Dynamic dispatch returns to this
        # class for prepare(), _grid_positions(), and _select_fixed_regions().
        return super().main()


_application = V16RavenDatasetPreparationApplication()
prepare = _application.prepare
main = _application.main


if __name__ == "__main__":
    raise SystemExit(main())
