from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import MODEL_SCHEMA, V15Config
from v14.inference import tiled_inference
from v14.losses import candidate_loss, selector_loss
from v14.model import BenefitSelector, NSAMDRV15
from v14.qualification import aggregate_candidate, sample_metrics
from v14.refinement import EDSRResidualBlock, SingleScaleHRRefinementTrunk
from v14.training_stability import DivergenceMonitor


class NSAMDRV15ContractTests(unittest.TestCase):
    def _config(self) -> V15Config:
        config = V15Config(
            train_lr_size=8,
            train_hr_size=32,
            validation_lr_size=8,
            validation_hr_size=32,
            lr_context_channels=8,
            lr_blocks=1,
            hr_channels=8,
            hr_blocks=4,
            map_tail_blocks=1,
            residual_scale=0.10,
            checkpoint_segment_blocks=2,
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
        torch.manual_seed(15)
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
        model = NSAMDRV15(self._config()).eval()
        albedo, normal, material = self._batch()
        with torch.no_grad():
            outputs = model(albedo, normal, material)

        self.assertEqual(tuple(outputs["candidate_albedo"].shape[-2:]), (32, 32))
        self.assertTrue(torch.equal(outputs["candidate_albedo"], outputs["baseline_albedo"]))
        self.assertTrue(torch.equal(outputs["candidate_material"], outputs["baseline_material"]))
        self.assertLess(
            float((outputs["candidate_normal"] - outputs["baseline_normal"]).abs().max()),
            1.0e-6,
        )
        for key in (
            "predicted_residual_albedo",
            "predicted_residual_normal",
            "predicted_residual_material",
        ):
            self.assertEqual(float(outputs[key].abs().max()), 0.0)

    def test_tanh_residuals_stay_inside_caps(self) -> None:
        config = self._config()
        model = NSAMDRV15(config).eval()
        with torch.no_grad():
            for head in (
                model.hr_refiner.albedo_tail.head,
                model.hr_refiner.normal_tail.head,
                model.hr_refiner.material_tail.head,
            ):
                head.bias.fill_(100.0)
        albedo, normal, material = self._batch()
        with torch.no_grad():
            outputs = model(albedo, normal, material)

        self.assertLessEqual(
            float(outputs["predicted_residual_albedo"].abs().max()),
            config.albedo_residual_cap,
        )
        self.assertLessEqual(
            float(outputs["predicted_residual_normal"].abs().max()),
            config.normal_residual_cap,
        )
        self.assertLessEqual(
            float(outputs["predicted_residual_material"].abs().max()),
            config.material_residual_cap,
        )

    def test_backbone_is_single_resolution_and_bn_free(self) -> None:
        model = NSAMDRV15(self._config()).eval()
        refiner = model.hr_refiner
        self.assertIsInstance(refiner, SingleScaleHRRefinementTrunk)
        self.assertTrue(
            all(isinstance(block, EDSRResidualBlock) for block in refiner.body.blocks)
        )
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()))
        self.assertFalse(any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()))
        self.assertFalse(any(isinstance(module, torch.nn.ConvTranspose2d) for module in model.modules()))

        albedo, normal, material = self._batch()
        lr_maps = torch.cat((albedo, normal, material), dim=1)
        with torch.no_grad():
            context_lr = model.context_encoder(lr_maps)
            baseline = model.baseline(albedo, normal, material)[0]
            context_hr = model._phase_neutral_context(lr_maps, baseline.shape[-2:])
        self.assertEqual(tuple(context_lr.shape[-2:]), (8, 8))
        self.assertEqual(tuple(context_hr.shape[-2:]), (32, 32))

    def test_architecture_contract_matches_v15(self) -> None:
        model = NSAMDRV15(self._config())
        contract = model.architecture_contract()
        self.assertEqual(contract["schema"], MODEL_SCHEMA)
        self.assertEqual(contract["revision"], "V15.0")
        self.assertEqual(contract["backbone"], "EDSR-style-single-resolution-HR")
        self.assertEqual(contract["decoderUpsampling"], "none")
        self.assertEqual(contract["residualBounding"], "tanh")
        self.assertFalse(contract["multiscaleFeatureHierarchy"])
        self.assertFalse(contract["batchNormalizationUsed"])
        self.assertFalse(contract["channelAttentionUsed"])
        self.assertFalse(contract["pixelShuffleUsed"])
        self.assertFalse(contract["transposedConvolutionUsed"])
        self.assertTrue(contract["selectorUsesPhysicalMaps"])
        self.assertEqual(model.selector.net[0].in_channels, BenefitSelector.FEATURE_CHANNELS)
        self.assertEqual(BenefitSelector.FEATURE_CHANNELS, 35)

    def test_candidate_training_reaches_backbone_and_map_tails(self) -> None:
        config = self._config()
        model = NSAMDRV15(config)
        model.set_candidate_training()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=2.0e-4)
        albedo, normal, material = self._batch()

        for _step in range(5):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(albedo, normal, material)
            batch = {
                "target_albedo": (outputs["baseline_albedo"].detach() * 0.88 + 0.06).clamp(0, 1),
                "target_normal": (outputs["baseline_normal"].detach() * 0.90).clamp(-0.999, 0.999),
                "target_material": (outputs["baseline_material"].detach() * 0.85 + 0.075).clamp(0, 1),
            }
            losses = candidate_loss(outputs, batch, config)
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()
            optimizer.step()

        modules = (
            model.context_encoder,
            model.context_adapter,
            model.hr_refiner.body,
            model.hr_refiner.albedo_tail,
            model.hr_refiner.normal_tail,
            model.hr_refiner.material_tail,
        )
        for module in modules:
            self.assertGreater(self._gradient_sum(module), 0.0)

    def test_selector_training_freezes_reconstruction(self) -> None:
        model = NSAMDRV15(self._config())
        model.set_selector_training()
        albedo, normal, material = self._batch()
        outputs = model(albedo, normal, material)
        batch = {
            "target_albedo": (outputs["baseline_albedo"].detach() * 0.92 + 0.04).clamp(0, 1),
            "target_normal": (outputs["baseline_normal"].detach() * 0.95).clamp(-0.999, 0.999),
            "target_material": (outputs["baseline_material"].detach() * 0.90 + 0.05).clamp(0, 1),
        }
        losses = selector_loss(outputs, batch)
        losses["total"].backward()
        self.assertGreater(self._gradient_sum(model.selector), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.hr_refiner.parameters()))

    def test_divergence_monitor_rejects_persistent_saturation(self) -> None:
        monitor = DivergenceMonitor(saturation_patience_reports=3)
        saturation = {"albedo": 0.10, "normal": 0.99, "material": 0.10}
        for _ in range(2):
            self.assertFalse(
                monitor.observe_report(
                    loss=1.0,
                    gradient_norm=0.5,
                    saturation=saturation,
                ).diverged
            )
        decision = monitor.observe_report(
            loss=1.0,
            gradient_norm=0.5,
            saturation=saturation,
        )
        self.assertTrue(decision.diverged)

    def test_lattice_metric_penalizes_extra_cell_constant_residual(self) -> None:
        config = self._config()
        model = NSAMDRV15(config).eval()
        albedo, normal, material = self._batch()
        with torch.no_grad():
            outputs = model(albedo, normal, material)
        baseline = outputs["baseline_albedo"].detach()
        target = baseline.clone()
        target[..., 10:22, 9:23] = (target[..., 10:22, 9:23] + 0.08).clamp(0, 1)
        candidate = baseline.clone()
        candidate[..., 8:24, 8:24] = (candidate[..., 8:24, 8:24] + 0.08).clamp(0, 1)
        fake = dict(outputs)
        fake["candidate_albedo"] = candidate
        batch = {
            "target_albedo": target,
            "target_normal": outputs["baseline_normal"].detach(),
            "target_material": outputs["baseline_material"].detach(),
        }
        metrics = sample_metrics(fake, batch, final=False)
        self.assertGreater(metrics["lattice_cell_excess"], 0.0)

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
        native_only = [{**passing, "heldout_sample": 0.0} for _ in range(8)]
        report = aggregate_candidate(native_only, config)
        self.assertFalse(report["heldOutCoveragePass"])
        self.assertFalse(report["passed"])

    def test_tiled_inference_keeps_four_x_and_normalizes_normals(self) -> None:
        model = NSAMDRV15(self._config()).eval()
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
        for key in ("baseline_normal", "candidate_normal", "normal"):
            length = torch.sqrt((outputs[key].float() ** 2).sum(dim=1))
            self.assertLessEqual(float(length.max()), 0.9991)


if __name__ == "__main__":
    unittest.main()
