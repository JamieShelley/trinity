import unittest

from tools.nsamdr.neural.probe_nsamdr_v16_sibling_crop_transfer import (
    _authority_crop_pair_records,
    _authority_records,
)


class SiblingCropTransferProbeTests(unittest.TestCase):
    def test_authority_records_filters_train_and_sorts(self) -> None:
        manifest = {
            "crops": [
                {"split": "train", "family_id": "a", "crop_id": "a_001"},
                {"split": "validation", "family_id": "a", "crop_id": "a_999"},
                {"split": "train", "family_id": "b", "crop_id": "b_000"},
                {"split": "train", "family_id": "a", "crop_id": "a_000"},
            ]
        }
        rows = _authority_records(manifest, "a")
        self.assertEqual([row["crop_id"] for row in rows], ["a_000", "a_001"])

    def test_crop_pair_uses_first_and_second_sorted_crops(self) -> None:
        manifest = {
            "crops": [
                {"split": "train", "family_id": "a", "crop_id": "a_001"},
                {"split": "train", "family_id": "a", "crop_id": "a_000"},
                {"split": "train", "family_id": "b", "crop_id": "b_000"},
                {"split": "train", "family_id": "b", "crop_id": "b_001"},
            ]
        }
        pairs = _authority_crop_pair_records(manifest, ["a", "b"])
        self.assertEqual(pairs[0]["trained"]["crop_id"], "a_000")
        self.assertEqual(pairs[0]["sibling"]["crop_id"], "a_001")
        self.assertEqual(pairs[1]["trained"]["crop_id"], "b_000")
        self.assertEqual(pairs[1]["sibling"]["crop_id"], "b_001")

    def test_crop_pair_rejects_single_crop_authority(self) -> None:
        manifest = {
            "crops": [
                {"split": "train", "family_id": "a", "crop_id": "a_000"},
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "requires at least two"):
            _authority_crop_pair_records(manifest, ["a"])


if __name__ == "__main__":
    unittest.main()
