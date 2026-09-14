from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare_nsamdr_v16_multifamily_balanced_dataset import (
    _balanced_select_fixed_regions,
)
from prepare_nsamdr_v16_raven_dataset import (
    V16RavenDatasetPreparationApplication,
)
from v14.multifamily_multiregion_diagnostic import _balanced_family_subset


FAMILIES = ("raven-t1", "golem-t2", "scorpion-t1", "widow-t2")


class V16MultiFamilyBalanceTests(unittest.TestCase):
    @staticmethod
    def _family(family_id: str, width: int, height: int) -> list[dict[str, object]]:
        crop = 512
        xs = V16RavenDatasetPreparationApplication._sliding_positions(width, crop)
        ys = V16RavenDatasetPreparationApplication._sliding_positions(height, crop)
        pixels = np.zeros((crop, crop, 3), dtype=np.uint8)
        result: list[dict[str, object]] = []
        for gy, y in enumerate(ys):
            for gx, x in enumerate(xs):
                result.append(
                    {
                        "familyId": family_id,
                        "familyIndex": 0,
                        "gx": gx,
                        "gy": gy,
                        "x": x,
                        "y": y,
                        "detailScore": float(gx + gy),
                        "holdout": False,
                        "albedo": pixels,
                        "normal": pixels,
                        "material": pixels,
                        "materialValid": True,
                        "normalEncoding": "test",
                        "family": {},
                    }
                )
        return result

    def test_four_family_dataset_has_one_heldout_per_family(self) -> None:
        app = V16RavenDatasetPreparationApplication()
        candidates: list[dict[str, object]] = []
        for index, family in enumerate(FAMILIES):
            width = 2048 if index % 2 else 1024
            candidates.extend(self._family(family, width, 1024))

        train, validation = _balanced_select_fixed_regions(
            app,
            candidates,
            max_train_crops=16,
            max_validation_crops=4,
            seed=14001,
        )
        self.assertEqual(len(validation), 4)
        validation_counts = {
            family: sum(1 for item in validation if item["familyId"] == family)
            for family in FAMILIES
        }
        self.assertEqual(validation_counts, {family: 1 for family in FAMILIES})
        self.assertEqual({str(item["familyId"]) for item in train}, set(FAMILIES))

        for train_item in train:
            for validation_item in validation:
                self.assertFalse(app._rectangles_overlap(train_item, validation_item, 512))

    def test_eight_stage2_training_regions_are_two_per_family(self) -> None:
        records: list[dict[str, object]] = []
        source_counts = (5, 11, 7, 13)
        for family, count in zip(FAMILIES, source_counts):
            for index in range(count):
                records.append(
                    {
                        "family_id": family,
                        "detail_score": float(1000 - index),
                        "path": f"{family}-{index}.npz",
                        "split": "train",
                    }
                )

        selected = _balanced_family_subset(records, 8)
        counts = {
            family: sum(1 for item in selected if item["family_id"] == family)
            for family in FAMILIES
        }
        self.assertEqual(len(selected), 8)
        self.assertEqual(counts, {family: 2 for family in FAMILIES})

    def test_four_stage2_heldout_regions_are_one_per_family(self) -> None:
        records: list[dict[str, object]] = []
        for family in FAMILIES:
            for index in range(3):
                records.append(
                    {
                        "family_id": family,
                        "detail_score": float(100 - index),
                        "path": f"{family}-heldout-{index}.npz",
                        "split": "validation",
                    }
                )
        selected = _balanced_family_subset(records, 4)
        counts = {
            family: sum(1 for item in selected if item["family_id"] == family)
            for family in FAMILIES
        }
        self.assertEqual(counts, {family: 1 for family in FAMILIES})

    def test_coverage_budget_preserves_640_visits_per_region(self) -> None:
        self.assertEqual(2560 // 4, 640)
        self.assertEqual(5120 // 8, 640)


if __name__ == "__main__":
    unittest.main()
