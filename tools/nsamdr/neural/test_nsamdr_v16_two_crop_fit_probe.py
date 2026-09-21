import unittest

from tools.nsamdr.neural.probe_nsamdr_v16_two_crop_fit import (
    _heldout_delta,
    _select_fixed_crops,
)


class TwoCropFitProbeTests(unittest.TestCase):
    def test_select_fixed_crops_uses_first_two_sorted_train_crops(self) -> None:
        manifest = {
            "crops": [
                {"split": "train", "family_id": "a", "crop_id": "a_002"},
                {"split": "train", "family_id": "a", "crop_id": "a_000"},
                {"split": "train", "family_id": "a", "crop_id": "a_001"},
                {"split": "train", "family_id": "b", "crop_id": "b_001"},
                {"split": "train", "family_id": "b", "crop_id": "b_000"},
            ]
        }
        rows = _select_fixed_crops(
            manifest,
            ["a", "b"],
            crops_per_authority=2,
        )
        self.assertEqual(
            [(row["authorityId"], row["record"]["crop_id"]) for row in rows],
            [("a", "a_000"), ("a", "a_001"), ("b", "b_000"), ("b", "b_001")],
        )

    def test_select_fixed_crops_rejects_missing_second_crop(self) -> None:
        manifest = {
            "crops": [
                {"split": "train", "family_id": "a", "crop_id": "a_000"},
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "2 required"):
            _select_fixed_crops(
                manifest,
                ["a"],
                crops_per_authority=2,
            )

    def test_heldout_delta_is_candidate_minus_source(self) -> None:
        source = {
            "median_global_recovery": 0.05,
            "median_edge_recovery": 0.04,
            "median_gradient_recovery": 0.03,
            "median_normal_recovery": 0.10,
            "median_lattice_cell_excess": 0.50,
            "median_detail_recovery_1px": 0.01,
            "median_detail_recovery_2px": 0.02,
            "median_detail_recovery_4px": 0.03,
        }
        candidate = {
            "median_global_recovery": 0.10,
            "median_edge_recovery": 0.12,
            "median_gradient_recovery": 0.11,
            "median_normal_recovery": 0.18,
            "median_lattice_cell_excess": 0.10,
            "median_detail_recovery_1px": 0.04,
            "median_detail_recovery_2px": 0.06,
            "median_detail_recovery_4px": 0.08,
        }
        delta = _heldout_delta(source, candidate)
        self.assertAlmostEqual(delta["global_recovery"], 0.05)
        self.assertAlmostEqual(delta["edge_recovery"], 0.08)
        self.assertAlmostEqual(delta["lattice_cell_excess"], -0.40)
        self.assertAlmostEqual(delta["detail_recovery_1px"], 0.03)


if __name__ == "__main__":
    unittest.main()
