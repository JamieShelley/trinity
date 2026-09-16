from __future__ import annotations

import unittest

import numpy as np

from tools.nsamdr.neural.audit_nsamdr_v16_boundary_profiles import _knn_predict
from tools.nsamdr.neural.v16.profiles import PHYSICAL_CHANNELS, PROFILE_OFFSETS


class BoundaryProfileAuditTests(unittest.TestCase):
    def test_profile_layout_is_stable_for_reproducible_negative_evidence(self) -> None:
        self.assertEqual(PROFILE_OFFSETS, (-8, -6, -4, -2, 0, 2, 4, 6, 8))
        self.assertEqual(PHYSICAL_CHANNELS, 8)

    def test_knn_exact_match_returns_training_profile(self) -> None:
        train_x = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        train_y = np.asarray([[2.0, 3.0], [7.0, 11.0]], dtype=np.float32)
        predicted = _knn_predict(
            train_x,
            train_y,
            np.asarray([[1.0, 1.0]], dtype=np.float32),
            1,
        )
        np.testing.assert_allclose(predicted[0], train_y[1], rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
