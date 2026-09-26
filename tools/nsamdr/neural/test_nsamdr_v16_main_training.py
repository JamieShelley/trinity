from __future__ import annotations

import unittest

import numpy as np

from tools.nsamdr.neural.train_nsamdr_v16_main import (
    _stage_schedule,
    _training_geometry,
)
from tools.nsamdr.neural.v16.broad_prior import _augment_d4


class NSAMDRV16MainTrainingTests(unittest.TestCase):
    def test_one_d4_pass_stage_schedule_matches_full_corpus_coverage(self) -> None:
        self.assertEqual(
            _stage_schedule(596, 1),
            [596, 1192, 2384, 4768],
        )

    def test_second_d4_pass_extends_from_first_complete_pass(self) -> None:
        self.assertEqual(
            _stage_schedule(596, 2),
            [596, 1192, 2384, 4768, 9536],
        )

    def test_training_geometry_requires_uniform_crops_per_authority(self) -> None:
        manifest = {
            "crops": [
                {"split": "train", "family_id": "a"},
                {"split": "train", "family_id": "a"},
                {"split": "train", "family_id": "b"},
                {"split": "train", "family_id": "b"},
                {"split": "validation", "family_id": "held"},
            ]
        }
        self.assertEqual(_training_geometry(manifest), (2, 2, 4))

        bad = {
            "crops": [
                {"split": "train", "family_id": "a"},
                {"split": "train", "family_id": "a"},
                {"split": "train", "family_id": "b"},
            ]
        }
        with self.assertRaises(RuntimeError):
            _training_geometry(bad)

    def test_d4_normal_vector_transform_matches_rotation_and_mirror(self) -> None:
        albedo = np.zeros((2, 2, 3), dtype=np.float32)
        material = np.zeros((2, 2, 3), dtype=np.float32)
        normal = np.zeros((2, 2, 2), dtype=np.float32)
        normal[..., 0] = 1.0

        _, rotated, _ = _augment_d4(albedo, normal.copy(), material, 1)
        self.assertTrue(np.allclose(rotated[..., 0], 0.0, atol=1.0e-6))
        self.assertTrue(np.allclose(rotated[..., 1], 1.0, atol=1.0e-6))

        _, mirrored, _ = _augment_d4(albedo, normal.copy(), material, 4)
        self.assertTrue(np.allclose(mirrored[..., 0], -1.0, atol=1.0e-6))
        self.assertTrue(np.allclose(mirrored[..., 1], 0.0, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
