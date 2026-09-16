from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.nsamdr.neural.audit_nsamdr_v16_stage2_recovery_sweep import (
    DESCRIPTOR_GRID,
    _pooled_descriptor,
    _spectral_partition,
)


class RecoverySweepTests(unittest.TestCase):
    def test_descriptor_dimension_is_context_independent(self) -> None:
        small = np.random.default_rng(1).normal(size=(8, 5, 5)).astype(np.float32)
        large = np.random.default_rng(2).normal(size=(8, 33, 33)).astype(np.float32)
        a = _pooled_descriptor(small)
        b = _pooled_descriptor(large)
        expected = 8 * DESCRIPTOR_GRID * DESCRIPTOR_GRID + 8 + 8
        self.assertEqual(a.shape, (expected,))
        self.assertEqual(b.shape, (expected,))

    def test_high_frequency_residual_has_more_above_nyquist_energy(self) -> None:
        size = 128
        y = torch.arange(size, dtype=torch.float32)[:, None]
        x = torch.arange(size, dtype=torch.float32)[None, :]
        low = torch.sin(2.0 * torch.pi * x / 64.0).expand(size, size)
        high = ((x.long() + y.long()) % 2).float() * 2.0 - 1.0
        low_result = _spectral_partition(low.unsqueeze(0), 4.0)
        high_result = _spectral_partition(high.unsqueeze(0), 4.0)
        self.assertLess(low_result["aboveFraction"], high_result["aboveFraction"])
        self.assertGreater(high_result["aboveFraction"], 0.90)


if __name__ == "__main__":
    unittest.main()
