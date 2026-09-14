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


class V16RavenSpatialSplitTests(unittest.TestCase):
    @staticmethod
    def _candidates() -> list[dict[str, object]]:
        crop_size = 512
        positions = (0, 128, 256, 384, 512)
        pixels = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        result: list[dict[str, object]] = []
        for gy, y in enumerate(positions):
            for gx, x in enumerate(positions):
                result.append(
                    {
                        "familyId": "raven-test-family",
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

    def test_sliding_positions_include_last_valid_origin(self) -> None:
        self.assertEqual(
            V16RavenDatasetPreparationApplication._sliding_positions(1024, 512),
            [0, 128, 256, 384, 512],
        )
        self.assertEqual(
            V16RavenDatasetPreparationApplication._sliding_positions(1100, 512)[-1],
            588,
        )


if __name__ == "__main__":
    unittest.main()
