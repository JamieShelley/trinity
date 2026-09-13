from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import MODEL_SCHEMA, V14Config
from v14.inference import tiled_inference
from v14.losses import candidate_loss, selector_loss
from v14.model import BenefitSelector, NSAMDRV14
from v14.qualification import aggregate_candidate, sample_metrics


class NSAMDRV14ContractTests(unittest.TestCase):
    def _config(self) -> V14Config:
        config = V14Config(
            train_lr_size=8,
            train_hr_size=32,
            validation_lr_size=8,
            validation_hr_size=32,
            lr_context_channels=8,
            lr_blocks=1,
            hr_channels=8,
            half_channels=12,
            quarter_channels=16,
            hr_encoder_blocks=1,
            half_encoder_blocks=1,
            quarter_encoder_blocks=1,
            bottleneck_blocks=1,
            half_decoder_blocks=1,
            hr_decoder_blocks=1,
            map_tail_blocks=1,
            attention_reduction=4,
            use_gradient_checkpointing=False,
            selector_channels=8,
            production_tile_lr=8,
            production_overlap_lr=2,
            tiles_per_epoch=1,
            validation_tiles=4,
            minimum_heldout_samples=4,
            clean_epochs=1,
            robust_epochs=1,
            selector_epochs=1,
        )
        config.validate()
        return config

    def _batch(
        self,
        size: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(14)
        albedo = torch.rand(1, 3, size, size)
        normal = torch.rand(1, 2, size, size) * 0.8 - 0.4
        material = torch.rand(1, 3, size, size)
        return albedo, normal, material

    @staticmethod
    def _gradient_sum(module: torch.nn.Module) -> float:
        return sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    def test_initial_candidate_is_exact_baseline(self) -> None:
        model = NSAMDRV14(self._config()).eval()
        albedo, normal, material = self._batch()
        with torch.no_grad():
            outputs = model(albedo, normal, material)

        self.assertEqual(
            tuple(outputs["candidate_albedo"].shape[-2:]),
            (32, 32),
        )
        self.assertTrue(
            torch.equal(
                outputs["candidate_albedo"],
                outputs["baseline_albedo"],
            )
        )
        self.assertTrue(
            torch.equal(
                outputs["candidate_material"],
                outputs["baseline_material"],
            )
        )
        self.assertLess(
            float(
                (
                    outputs["candidate_normal"]
                    - outputs["baseline_normal"]
                )
                .abs()
                .max()
            ),
            1.0e-6,
        )

    def test_context_and_decoder_are_phase_neutral(self) -> None:
        config = self._config()
        model = NSAMDRV14(config).eval()
        albedo, normal, material = self._batch()
        lr_maps = torch.cat((albedo, normal, material), dim=1)

        with torch.no_grad():
            context_lr = model.context_encoder(lr_maps)

        self.assertEqual(
            tuple(context_lr.shape[-2:]),
            tuple(albedo.shape[-2:]),
        )
        self.assertFalse(
            any(
                isinstance(module, torch.nn.PixelShuffle)
                for module in model.modules()
            )
        )
        self.assertFalse(
            any(
                isinstance(module, torch.nn.ConvTranspose2d)
                for module in model.modules()
            )
        )
        contract = model.architecture_contract()
        self.assertEqual(contract["revision"], "V14.2")
        self.assertEqual(
            contract["contextUpsampling"],
            "bilinear-phase-neutral + HR 3x3 adapter",
        )
        self.assertEqual(
            contract["decoderUpsampling"],
            "bilinear-phase-neutral + HR convolution",
        )
        self.assertFalse(contract["lrPhaseGridPixelAuthority"])
        self.assertFalse(contract["pixelShuffleUsed"])
        self.assertFalse(contract["transposedConvolutionUsed"])

    def test_architecture_contract_is_clean(self) -> None:
        model = NSAMDRV14(self._config())
        contract = model.architecture_contract()

        self.assertEqual(contract["schema"], MODEL_SCHEMA)
        self.assertEqual(model.config.schema, MODEL_SCHEMA)
        self.assertEqual(
            tuple(contract["activeComponents"]),
            (
                "baseline",
                "context_encoder",
                "context_adapter",
                "multiscale_hr_refiner",
                "albedo_tail",
                "normal_tail",
                "material_tail",
                "selector",
            ),
        )
        self.assertEqual(tuple(contract["retiredComponents"]), ())
        self.assertFalse(contract["geometryPixelAuthority"])
        self.assertFalse(contract["seamPixelAuthority"])
        self.assertFalse(contract["profilePixelAuthority"])
        self.assertTrue(contract["selectorUsesPhysicalMaps"])
        self.assertEqual(
            model.selector.net[0].in_channels,
            BenefitSelector.FEATURE_CHANNELS,
        )
        self.assertEqual(BenefitSelector.FEATURE_CHANNELS, 35)
        for forbidden in (
            "geometry_net",
            "boundary_renderer",
            "seam_restorer",
        ):
            self.assertFalse(hasattr(model, forbidden))

    def test_candidate_training_reaches_all_v14_2_paths(self) -> None:
        config = self._config()
        model = NSAMDRV14(config)
        model.set_candidate_training()
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
        albedo, normal, material = self._batch()

        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(albedo, normal, material)
            batch = {
                "target_albedo": (
                    outputs["baseline_albedo"].detach() * 0.88 + 0.06
                ).clamp(0, 1),
                "target_normal": (
                    outputs["baseline_normal"].detach() * 0.90
                ).clamp(-0.999, 0.999),
                "target_material": (
                    outputs["baseline_material"].detach() * 0.85 + 0.075
                ).clamp(0, 1),
            }
            losses = candidate_loss(outputs, batch, config)
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()
            optimizer.step()

        refiner = model.hr_refiner
        modules = (
            model.context_encoder,
            model.context_adapter,
            refiner.hr_encoder,
            refiner.half_encoder,
            refiner.quarter_encoder,
            refiner.bottleneck,
            refiner.half_decoder,
            refiner.hr_decoder,
            refiner.albedo_tail,
            refiner.normal_tail,
            refiner.material_tail,
        )
        for module in modules:
            self.assertGreater(self._gradient_sum(module), 0.0)

    def test_selector_training_freezes_reconstruction(self) -> None:
        model = NSAMDRV14(self._config())
        model.set_selector_training()
        albedo, normal, material = self._batch()
        outputs = model(albedo, normal, material)
        batch = {
            "target_albedo": (
                outputs["baseline_albedo"].detach() * 0.92 + 0.04
            ).clamp(0, 1),
            "target_normal": (
                outputs["baseline_normal"].detach() * 0.95
            ).clamp(-0.999, 0.999),
            "target_material": (
                outputs["baseline_material"].detach() * 0.90 + 0.05
            ).clamp(0, 1),
        }
        losses = selector_loss(outputs, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

        self.assertGreater(self._gradient_sum(model.selector), 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.hr_refiner.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.context_adapter.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.context_encoder.parameters()
            )
        )

    def test_detail_telemetry_does_not_change_qualification_gate(self) -> None:
        config = self._config()
        passing = {
            "global_recovery": 0.80,
            "edge_recovery": 0.80,
            "gradient_recovery": 0.80,
            "detail_recovery_1px": -0.50,
            "detail_recovery_2px": -0.50,
            "detail_recovery_4px": -0.50,
            "normal_recovery": 0.20,
            "material_recovery": 0.20,
            "lattice_cell_excess": 0.0,
            "lattice_candidate_fraction": 0.1,
            "lattice_target_fraction": 0.1,
            "protected_preservation": 1.0,
            "target_size": 32.0,
            "heldout_sample": 1.0,
        }
        report = aggregate_candidate(
            [dict(passing) for _ in range(4)],
            config,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["medianDetailRecovery1px"], -0.50)

    def test_lattice_metric_penalizes_extra_cell_constant_residual(self) -> None:
        config = self._config()
        model = NSAMDRV14(config).eval()
        albedo, normal, material = self._batch()
        with torch.no_grad():
            outputs = model(albedo, normal, material)

        baseline = outputs["baseline_albedo"].detach()
        target = baseline.clone()
        target[..., 10:22, 9:23] = (
            target[..., 10:22, 9:23] + 0.08
        ).clamp(0, 1)
        candidate = baseline.clone()
        candidate[..., 8:24, 8:24] = (
            candidate[..., 8:24, 8:24] + 0.08
        ).clamp(0, 1)

        fake = dict(outputs)
        fake["candidate_albedo"] = candidate
        batch = {
            "target_albedo": target,
            "target_normal": outputs["baseline_normal"].detach(),
            "target_material": outputs["baseline_material"].detach(),
        }
        metrics = sample_metrics(fake, batch, final=False)
        self.assertGreater(metrics["lattice_cell_excess"], 0.0)
        self.assertIn("detail_recovery_1px", metrics)
        self.assertIn("detail_recovery_2px", metrics)
        self.assertIn("detail_recovery_4px", metrics)

    def test_native_scale_telemetry_cannot_satisfy_heldout_coverage(self) -> None:
        config = self._config()
        passing = {
            "global_recovery": 0.80,
            "edge_recovery": 0.80,
            "gradient_recovery": 0.80,
            "normal_recovery": 0.20,
            "material_recovery": 0.20,
            "lattice_cell_excess": 0.0,
            "lattice_candidate_fraction": 0.1,
            "lattice_target_fraction": 0.1,
            "protected_preservation": 1.0,
            "target_size": 1024.0,
        }
        native_only = [
            {**passing, "heldout_sample": 0.0}
            for _ in range(8)
        ]
        report = aggregate_candidate(native_only, config)
        self.assertEqual(report["sampleCount"], 0)
        self.assertEqual(report["nativeScaleTelemetry"]["sampleCount"], 8)
        self.assertFalse(report["heldOutCoveragePass"])
        self.assertFalse(report["passed"])

        heldout = [
            {
                **passing,
                "heldout_sample": 1.0,
                "target_size": 32.0,
            }
            for _ in range(4)
        ]
        report = aggregate_candidate([*native_only, *heldout], config)
        self.assertEqual(report["sampleCount"], 4)
        self.assertTrue(report["heldOutCoveragePass"])
        self.assertTrue(report["passed"])

    def test_tiled_inference_keeps_exact_four_x_dimensions_and_normalizes_normals(self) -> None:
        model = NSAMDRV14(self._config()).eval()
        albedo, normal, material = self._batch(size=12)
        with torch.no_grad():
            outputs = tiled_inference(
                model,
                albedo,
                normal,
                material,
                tile_lr=8,
                overlap_lr=2,
            )
        self.assertEqual(tuple(outputs["albedo"].shape[-2:]), (48, 48))
        self.assertEqual(tuple(outputs["normal"].shape[-2:]), (48, 48))
        self.assertEqual(tuple(outputs["material"].shape[-2:]), (48, 48))
        for key in (
            "baseline_normal",
            "candidate_normal",
            "normal",
        ):
            length = torch.sqrt((outputs[key].float() ** 2).sum(dim=1))
            self.assertLessEqual(float(length.max()), 0.9991)


if __name__ == "__main__":
    unittest.main()
