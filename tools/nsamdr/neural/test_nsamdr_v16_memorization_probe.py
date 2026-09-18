from __future__ import annotations

import unittest

from tools.nsamdr.neural.probe_nsamdr_v16_memorization import (
    _authority_records,
    _parse_stages,
)


class ExactMemorizationProbeTests(unittest.TestCase):
    def test_stage_parser_sorts_and_deduplicates(self) -> None:
        self.assertEqual(
            _parse_stages("32,1,8,8,64"),
            [1, 8, 32, 64],
        )
        self.assertEqual(
            _parse_stages(["64", "128", "256"]),
            [64, 128, 256],
        )

    def test_authority_records_select_train_and_sort_crops(self) -> None:
        manifest = {
            "crops": [
                {
                    "family_id": "alpha",
                    "crop_id": "alpha_001",
                    "split": "train",
                    "path": "b.npz",
                },
                {
                    "family_id": "alpha",
                    "crop_id": "alpha_000",
                    "split": "train",
                    "path": "a.npz",
                },
                {
                    "family_id": "alpha",
                    "crop_id": "alpha_val",
                    "split": "validation",
                    "path": "v.npz",
                },
                {
                    "family_id": "beta",
                    "crop_id": "beta_000",
                    "split": "train",
                    "path": "x.npz",
                },
            ]
        }
        rows = _authority_records(manifest, "alpha")
        self.assertEqual(
            [row["crop_id"] for row in rows],
            ["alpha_000", "alpha_001"],
        )

    def test_authority_records_reject_missing_authority(self) -> None:
        with self.assertRaises(RuntimeError):
            _authority_records({"crops": []}, "missing")


if __name__ == "__main__":
    unittest.main()
