from __future__ import annotations

import random
import unittest

import numpy as np

from tools.nsamdr.neural.v16.broad_prior import (
    _detail_crop_origin,
    _detail_map,
    authority_balanced_record_indices,
)


class BroadPriorAuthoritySamplingTests(unittest.TestCase):
    @staticmethod
    def _records() -> list[dict[str, str]]:
        return [
            {
                "family_id": family,
                "crop_id": f"{family}_{crop}",
                "path": f"{family}_{crop}.npz",
            }
            for family in ("a", "b", "c", "d")
            for crop in ("0", "1")
        ]

    def test_each_authority_is_seen_before_any_repeat(self) -> None:
        records = self._records()
        indices = authority_balanced_record_indices(records, 8, seed=16201)
        first = [records[index]["family_id"] for index in indices[:4]]
        second = [records[index]["family_id"] for index in indices[4:8]]
        self.assertEqual(set(first), {"a", "b", "c", "d"})
        self.assertEqual(set(second), {"a", "b", "c", "d"})

    def test_later_cycles_rotate_authority_crops(self) -> None:
        records = self._records()
        indices = authority_balanced_record_indices(records, 8, seed=16201)
        selected: dict[str, list[str]] = {family: [] for family in ("a", "b", "c", "d")}
        for index in indices:
            record = records[index]
            selected[record["family_id"]].append(record["crop_id"])
        for crop_ids in selected.values():
            self.assertEqual(len(crop_ids), 2)
            self.assertEqual(len(set(crop_ids)), 2)

    def test_detail_map_prefers_structured_half(self) -> None:
        albedo = np.zeros((128, 128, 3), dtype=np.float32)
        normal = np.zeros((128, 128, 2), dtype=np.float32)
        pattern = (np.indices((128, 64)).sum(axis=0) % 2).astype(np.float32)
        albedo[:, 64:, :] = pattern[..., None]
        score = _detail_map(albedo, normal)
        middle = score.shape[1] // 2
        self.assertGreater(float(score[:, middle:].mean()), float(score[:, :middle].mean()))

    def test_detail_crop_origin_is_deterministic(self) -> None:
        albedo = np.zeros((128, 128, 3), dtype=np.float32)
        normal = np.zeros((128, 128, 2), dtype=np.float32)
        pattern = (np.indices((128, 64)).sum(axis=0) % 2).astype(np.float32)
        albedo[:, 64:, :] = pattern[..., None]
        first = _detail_crop_origin(albedo, normal, 64, random.Random(99))
        second = _detail_crop_origin(albedo, normal, 64, random.Random(99))
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 32)

    def test_schedule_is_deterministic(self) -> None:
        records = self._records()
        first = authority_balanced_record_indices(records, 13, seed=99)
        second = authority_balanced_record_indices(records, 13, seed=99)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
