from __future__ import annotations

import unittest

from tools.nsamdr.neural.probe_nsamdr_v16_interference import (
    _candidate_authority_order,
    _train_authority_ids,
)


class InterferenceProbeTests(unittest.TestCase):
    def test_train_authority_ids_filters_and_sorts(self) -> None:
        manifest = {
            "crops": [
                {"family_id": "b", "split": "train"},
                {"family_id": "a", "split": "train"},
                {"family_id": "a", "split": "train"},
                {"family_id": "z", "split": "validation"},
            ]
        }
        self.assertEqual(_train_authority_ids(manifest), ["a", "b"])

    def test_candidate_order_keeps_anchor_first_and_is_deterministic(self) -> None:
        manifest = {
            "crops": [
                {"family_id": value, "split": "train"}
                for value in ("a", "b", "c", "d", "e")
            ]
        }
        first = _candidate_authority_order(
            manifest,
            anchor_authority="c",
            seed=123,
        )
        second = _candidate_authority_order(
            manifest,
            anchor_authority="c",
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0], "c")
        self.assertEqual(sorted(first), ["a", "b", "c", "d", "e"])

    def test_candidate_order_rejects_missing_anchor(self) -> None:
        manifest = {"crops": [{"family_id": "a", "split": "train"}]}
        with self.assertRaises(RuntimeError):
            _candidate_authority_order(
                manifest,
                anchor_authority="missing",
                seed=123,
            )


if __name__ == "__main__":
    unittest.main()
