import unittest

import torch

from tools.nsamdr.neural.probe_nsamdr_v16_augmented_two_crop_fit import (
    AUGMENTATION_VARIANTS,
    _augment_batch,
    _normal_transform,
    _spatial_transform,
)


class AugmentedTwoCropFitProbeTests(unittest.TestCase):
    def test_identity_variant_leaves_spatial_tensor_unchanged(self) -> None:
        value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        result = _spatial_transform(value, 0, False)
        self.assertTrue(torch.equal(result, value))

    def test_four_quarter_turns_return_normal_vector(self) -> None:
        normal = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        normal[:, 0] = 0.6
        normal[:, 1] = 0.8
        result = normal
        for _ in range(4):
            result = _normal_transform(result, 1, False)
        self.assertTrue(torch.allclose(result, normal))

    def test_horizontal_mirror_negates_normal_x(self) -> None:
        normal = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        normal[:, 0] = 0.25
        normal[:, 1] = 0.75
        result = _normal_transform(normal, 0, True)
        self.assertTrue(torch.allclose(result[:, 0], torch.full((1, 2, 3), -0.25)))
        self.assertTrue(torch.allclose(result[:, 1], torch.full((1, 2, 3), 0.75)))

    def test_all_eight_variants_are_accepted(self) -> None:
        batch = {
            "lr_albedo": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
            "lr_normal": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "lr_material": torch.ones((1, 1, 4, 4), dtype=torch.float32),
            "target_albedo": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
            "target_normal": torch.zeros((1, 2, 4, 4), dtype=torch.float32),
            "target_material": torch.ones((1, 1, 4, 4), dtype=torch.float32),
            "record_index": torch.tensor([0]),
        }
        for variant in range(AUGMENTATION_VARIANTS):
            result = _augment_batch(batch, variant)
            self.assertEqual(result["lr_albedo"].shape, batch["lr_albedo"].shape)
            self.assertEqual(result["target_normal"].shape, batch["target_normal"].shape)
            self.assertTrue(torch.equal(result["record_index"], batch["record_index"]))

    def test_invalid_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _augment_batch({}, AUGMENTATION_VARIANTS)


if __name__ == "__main__":
    unittest.main()
