from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.nsamdr.neural.audit_nsamdr_v16_stage2_recoverability import (
    _knn_probe,
    _spectral_partition,
)


class RecoverabilityAuditTests(unittest.TestCase):
    def test_spectral_partition_separates_low_and_high_frequency(self) -> None:
        size = 64
        x = torch.arange(size, dtype=torch.float32)
        low = torch.sin(2.0 * torch.pi * 4.0 * x / size).repeat(size, 1)
        high = torch.sin(2.0 * torch.pi * 16.0 * x / size).repeat(size, 1)
        low_report = _spectral_partition(low.unsqueeze(0))
        high_report = _spectral_partition(high.unsqueeze(0))
        self.assertGreater(low_report["belowFraction"], 0.95)
        self.assertGreater(high_report["aboveFraction"], 0.95)
        self.assertAlmostEqual(
            low_report["belowFraction"]
            + low_report["nearFraction"]
            + low_report["aboveFraction"],
            1.0,
            places=5,
        )

    def test_knn_probe_recovers_repeated_residual_mapping(self) -> None:
        train_features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
        held_features = np.asarray([[0.0], [2.0]], dtype=np.float32)
        train_residual = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
        held_residual = np.asarray([[0.1], [0.3]], dtype=np.float32)

        def sample(features: np.ndarray, residual: np.ndarray) -> dict[str, object]:
            return {
                "features": features,
                "inputDetail": np.asarray([0.0, 1.0][: len(features)], dtype=np.float32),
                "residual": {
                    "albedo": residual,
                    "normal": residual,
                    "material": residual,
                },
                "residualMagnitude": {
                    "albedo": np.abs(residual[:, 0]),
                    "normal": np.abs(residual[:, 0]),
                    "material": np.abs(residual[:, 0]),
                },
            }

        train = sample(train_features, train_residual)
        held = sample(held_features, held_residual)
        result = _knn_probe(train, held, k=1)
        self.assertGreater(result["maps"]["albedo"]["recovery"], 0.999)
        self.assertGreater(
            result["maps"]["albedo"]["predictionResidualCorrelation"], 0.999
        )


if __name__ == "__main__":
    unittest.main()
