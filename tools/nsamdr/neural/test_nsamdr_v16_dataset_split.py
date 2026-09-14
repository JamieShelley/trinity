from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare_nsamdr_v16_raven_dataset import (
    V16RavenDatasetPreparationApplication,
)
from v14.multifamily_multiregion_diagnostic import _balanced_family_subset
from v14.multiregion_diagnostic import MultiRegionDiagnostic, balanced_region_index


class V16RavenSpatialSplitTests(unittest.TestCase):
    @staticmethod
    def _candidates(
        family_id: str = "raven-test-family",
        *,
        width: int = 1024,
        height: int = 1024,
        family_index: int = 0,
    ) -> list[dict[str, object]]:
        crop_size = 512
        x_positions = V16RavenDatasetPreparationApplication._sliding_positions(
            width,
            crop_size,
        )
        y_positions = V16RavenDatasetPreparationApplication._sliding_positions(
            height,
            crop_size,
        )
        pixels = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        result: list[dict[str, object]] = []
        for gy, y in enumerate(y_positions):
            for gx, x in enumerate(x_positions):
                result.append(
                    {
                        "familyId": family_id,
                        "familyIndex": family_index,
                        "gx": gx,
                        "gy": gy,
                        "x": x,
                        "y": y,
                        "detailScore": float(gx + gy + family_index * 0.01),
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

    def test_1024_source_produces_four_by_four_pixel_disjoint_split(self) -> None:
        application = V16RavenDatasetPreparationApplication()
        train, validation = application._select_fixed_regions(
            self._candidates(),
            max_train_crops=4,
            max_validation_crops=4,
            seed=14001,
        )

        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 4)
        domain = application._last_domains["raven-test-family"]
        self.assertEqual(domain["sourceSize"], [1024, 1024])
        self.assertEqual(domain["cropSize"], 512)
        self.assertEqual(domain["stride"], 128)
        self.assertGreaterEqual(domain["trainCandidateWindows"], 4)
        self.assertGreaterEqual(domain["validationCandidateWindows"], 4)

        for train_item in train:
            for validation_item in validation:
                self.assertFalse(
                    application._rectangles_overlap(
                        train_item,
                        validation_item,
                        512,
                    )
                )

    def test_two_authored_families_keep_independent_disjoint_domains(self) -> None:
        application = V16RavenDatasetPreparationApplication()
        candidates = [
            *self._candidates("t1", width=1024, height=1024, family_index=0),
            *self._candidates("t2", width=2048, height=1024, family_index=1),
        ]
        train, validation = application._select_fixed_regions(
            candidates,
            max_train_crops=8,
            max_validation_crops=8,
            seed=14001,
        )

        self.assertEqual({item["familyId"] for item in train}, {"t1", "t2"})
        self.assertEqual(
            {item["familyId"] for item in validation},
            {"t1", "t2"},
        )
        self.assertEqual(set(application._last_domains), {"t1", "t2"})
        for train_item in train:
            for validation_item in validation:
                self.assertFalse(
                    application._rectangles_overlap(
                        train_item,
                        validation_item,
                        512,
                    )
                )

    def test_sliding_positions_include_last_valid_origin(self) -> None:
        self.assertEqual(
            V16RavenDatasetPreparationApplication._sliding_positions(1024, 512),
            [0, 128, 256, 384, 512],
        )
        self.assertEqual(
            V16RavenDatasetPreparationApplication._sliding_positions(1100, 512)[-1],
            588,
        )

    def test_multi_region_schedule_is_balanced_round_robin(self) -> None:
        indices = [balanced_region_index(step, 4) for step in range(1, 17)]
        self.assertEqual(indices, [0, 1, 2, 3] * 4)
        counts = [indices.count(index) for index in range(4)]
        self.assertEqual(counts, [4, 4, 4, 4])

    def test_multi_region_schedule_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            balanced_region_index(0, 4)
        with self.assertRaises(ValueError):
            balanced_region_index(1, 0)

    def test_stage2_region_selection_balances_authored_families(self) -> None:
        records: list[dict[str, object]] = []
        for family, offset in (("t1", 0.0), ("t2", 100.0)):
            for index in range(6):
                records.append(
                    {
                        "family_id": family,
                        "detail_score": offset + float(index),
                        "path": f"{family}_{index}.npz",
                    }
                )
        selected = _balanced_family_subset(records, 4)
        counts = {
            family: sum(1 for item in selected if item["family_id"] == family)
            for family in ("t1", "t2")
        }
        self.assertEqual(counts, {"t1": 2, "t2": 2})

    def test_multi_region_diagnosis_requires_train_and_validation_pass(self) -> None:
        self.assertEqual(
            MultiRegionDiagnostic._diagnosis({"passed": True}, {"passed": True}),
            "passed",
        )
        self.assertEqual(
            MultiRegionDiagnostic._diagnosis({"passed": True}, {"passed": False}),
            "generalisation-failure",
        )
        self.assertEqual(
            MultiRegionDiagnostic._diagnosis({"passed": False}, {"passed": True}),
            "validation-pass-train-anomaly",
        )
        self.assertEqual(
            MultiRegionDiagnostic._diagnosis({"passed": False}, {"passed": False}),
            "training-capacity-or-optimization-failure",
        )


if __name__ == "__main__":
    unittest.main()
