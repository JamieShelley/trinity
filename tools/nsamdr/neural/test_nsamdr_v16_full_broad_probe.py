from __future__ import annotations

import unittest

import torch

from tools.nsamdr.neural.probe_nsamdr_v16_full_broad import (
    _distribution,
    _full_config,
    _oracle_scalar_albedo_outputs,
    _parse_stages,
    _phase_abs_energy,
    _proof_loss,
    _proof_loss_terms,
    _residual_diagnostics,
    _safe_name,
    _unit_slope_albedo_outputs,
    _unit_slope_bounded_residual,
    parser,
)


class FullBroadProbeTests(unittest.TestCase):
    def test_stage_parser_sorts_and_deduplicates(self) -> None:
        self.assertEqual(_parse_stages("1024,256,512,512"), [256, 512, 1024])

    def test_preview_defaults_are_enabled(self) -> None:
        args = parser().parse_args([])
        self.assertEqual(args.preview_samples, 4)
        self.assertEqual(args.train_validation_samples, 0)
        self.assertFalse(args.preview_only)

    def test_preview_path_names_are_filesystem_safe(self) -> None:
        self.assertEqual(_safe_name("authority/a:b c"), "authority_a_b_c")

    def test_distribution_reports_range_and_positive_fraction(self) -> None:
        result = _distribution([-2.0, 0.0, 2.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["min"], -2.0)
        self.assertEqual(result["median"], 1.0)
        self.assertEqual(result["max"], 4.0)
        self.assertEqual(result["positiveFraction"], 0.5)

    def test_phase_energy_reports_all_4x_phases(self) -> None:
        value = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        phases = _phase_abs_energy(value, 4)
        self.assertEqual(len(phases), 16)
        self.assertTrue(all(abs(item - 1.0) < 1.0e-6 for item in phases.values()))

    def test_unit_slope_bound_preserves_small_signal_scale(self) -> None:
        raw = torch.tensor([-0.01, 0.0, 0.01], dtype=torch.float32)
        bounded = _unit_slope_bounded_residual(raw, 0.40)
        self.assertTrue(torch.all(bounded.abs() <= 0.40))
        self.assertAlmostEqual(float(bounded[1]), 0.0, places=7)
        self.assertAlmostEqual(float(bounded[2]), 0.01, places=4)

    def test_unit_slope_ablation_only_changes_albedo_candidate(self) -> None:
        config = _full_config("manifest.json", 32)
        baseline = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        raw = torch.full_like(baseline, 0.02)
        current = torch.tanh(raw) * config.albedo_residual_cap
        outputs = {
            "baseline_albedo": baseline,
            "candidate_albedo": baseline + current,
            "candidate_raw_residual_albedo": raw,
            "predicted_residual_albedo": current,
            "candidate_normal": torch.zeros((1, 2, 32, 32)),
            "candidate_material": torch.zeros((1, 3, 32, 32)),
        }
        result = _unit_slope_albedo_outputs(outputs, config)
        self.assertIs(result["candidate_normal"], outputs["candidate_normal"])
        self.assertIs(result["candidate_material"], outputs["candidate_material"])
        self.assertGreater(
            float(result["predicted_residual_albedo"].abs().mean()),
            float(current.abs().mean()),
        )

    def test_oracle_scalar_gain_recovers_matching_residual_shape(self) -> None:
        baseline = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        candidate = torch.full_like(baseline, 0.10)
        target = torch.full_like(baseline, 0.20)
        outputs = {
            "baseline_albedo": baseline,
            "candidate_albedo": candidate,
            "predicted_residual_albedo": candidate,
        }
        batch = {"target_albedo": target}
        result, gain = _oracle_scalar_albedo_outputs(outputs, batch)
        self.assertAlmostEqual(gain, 2.0, places=5)
        self.assertTrue(torch.allclose(result["candidate_albedo"], target))

    def test_loss_decomposition_preserves_existing_total(self) -> None:
        config = _full_config("manifest.json", 32)
        baseline_albedo = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        target_albedo = torch.full_like(baseline_albedo, 0.20)
        predicted_albedo = torch.full_like(baseline_albedo, 0.10)
        candidate_albedo = baseline_albedo + predicted_albedo

        baseline_normal = torch.zeros((1, 2, 32, 32), dtype=torch.float32)
        target_normal = torch.full_like(baseline_normal, 0.05)
        predicted_normal = torch.full_like(baseline_normal, 0.025)
        candidate_normal = baseline_normal + predicted_normal

        outputs = {
            "baseline_albedo": baseline_albedo,
            "candidate_albedo": candidate_albedo,
            "candidate_raw_residual_albedo": predicted_albedo,
            "predicted_residual_albedo": predicted_albedo,
            "baseline_normal": baseline_normal,
            "candidate_normal": candidate_normal,
            "predicted_residual_normal": predicted_normal,
        }
        batch = {
            "target_albedo": target_albedo,
            "target_normal": target_normal,
        }

        terms = _proof_loss_terms(outputs, batch, config)
        self.assertEqual(
            set(terms),
            {
                "reconstruction",
                "gradient",
                "normal",
                "residual_albedo",
                "residual_normal",
                "total",
            },
        )
        self.assertTrue(torch.allclose(_proof_loss(outputs, batch, config), terms["total"]))

        diagnostics = _residual_diagnostics(outputs, batch, config)
        self.assertAlmostEqual(
            diagnostics["candidate_to_target_residual_ratio"],
            0.5,
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["residual_cap_saturation"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["candidate_phase_energy_spread"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["residual_cosine_similarity"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["target_weighted_sign_agreement"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["least_squares_residual_gain"],
            2.0,
            places=5,
        )

    def test_full_config_keeps_production_v16_capacity(self) -> None:
        config = _full_config("manifest.json", 512)
        self.assertEqual(config.train_lr_size, 128)
        self.assertEqual(config.train_hr_size, 512)
        self.assertEqual(config.lr_context_channels, 32)
        self.assertEqual(config.lr_blocks, 5)
        self.assertEqual(config.hr_channels, 96)
        self.assertEqual(config.swin_groups, 6)
        self.assertEqual(config.swin_blocks_per_group, 6)
        self.assertEqual(config.swin_depth, 36)
        self.assertEqual(config.swin_num_heads, 6)
        self.assertEqual(config.map_tail_blocks, 2)
        self.assertTrue(config.use_gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
