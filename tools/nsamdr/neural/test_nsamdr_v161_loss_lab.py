#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import V16Config
from v14 import safe_live_resume_monitored_v161_fourfamily_multiregion_diagnostic as lab


class V161LossLabTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "loss_normalization": "baseline-relative-bounded",
            "frequency_loss": "off",
            "frequency_loss_weight": 0.05,
            "normal_loss": "xy-l1",
            "angular_normal_weight": 0.25,
            "gradient_conflict": "telemetry-only",
            "normalization_min_weight": 0.50,
            "normalization_max_weight": 2.50,
            "normalization_reference_albedo": 0.040,
            "normalization_reference_normal": 0.060,
            "normalization_reference_material": 0.020,
            "gradient_sketch_values": 1024,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _case():
        torch.manual_seed(7)
        target_a = torch.rand(1, 3, 16, 16)
        target_n = torch.rand(1, 2, 16, 16) * 1.2 - 0.6
        target_m = torch.rand(1, 3, 16, 16)
        baseline_a = (target_a * 0.85).clamp(0.0, 1.0)
        baseline_n = target_n * 0.80
        baseline_m = (target_m * 0.90).clamp(0.0, 1.0)
        candidate_a = (baseline_a + 0.02).clamp(0.0, 1.0)
        candidate_n = (baseline_n + 0.01).clamp(-1.0, 1.0)
        candidate_m = (baseline_m + 0.01).clamp(0.0, 1.0)
        outputs = {
            "candidate_albedo": candidate_a,
            "candidate_normal": candidate_n,
            "candidate_material": candidate_m,
            "baseline_albedo": baseline_a,
            "baseline_normal": baseline_n,
            "baseline_material": baseline_m,
            "predicted_residual_albedo": candidate_a - baseline_a,
            "predicted_residual_normal": candidate_n - baseline_n,
            "predicted_residual_material": candidate_m - baseline_m,
        }
        batch = {
            "target_albedo": target_a,
            "target_normal": target_n,
            "target_material": target_m,
        }
        return outputs, batch

    def setUp(self):
        lab._ACTIVE_STEP = 0
        lab._REGION_FAMILIES = [{"familyId": "test", "family": "Test"}]

    def test_normalisation_weights_are_bounded(self):
        lab._ACTIVE_OPTIONS = lab._options_from_args(self._args())
        outputs, batch = self._case()
        losses = lab._v161_candidate_loss(outputs, batch, V16Config())
        for key in ("weight_albedo", "weight_normal", "weight_material"):
            value = float(losses[key].item())
            self.assertGreaterEqual(value, 0.50)
            self.assertLessEqual(value, 2.50)
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_wavelet_and_fourier_are_finite(self):
        for mode in ("wavelet", "focal-fourier"):
            lab._ACTIVE_STEP = 0
            lab._ACTIVE_OPTIONS = lab._options_from_args(
                self._args(frequency_loss=mode)
            )
            outputs, batch = self._case()
            losses = lab._v161_candidate_loss(outputs, batch, V16Config())
            self.assertTrue(torch.isfinite(losses["frequency"]))
            self.assertTrue(torch.isfinite(losses["total"]))

    def test_angular_normal_mode_is_finite(self):
        lab._ACTIVE_OPTIONS = lab._options_from_args(
            self._args(normal_loss="angular+l1")
        )
        outputs, batch = self._case()
        losses = lab._v161_candidate_loss(outputs, batch, V16Config())
        self.assertGreaterEqual(float(losses["normal_angular"].item()), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main()
