from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import V14Config
from v14.inference import tiled_inference
from v14.losses import candidate_loss
from v14.model import MODEL_SCHEMA, NSAMDRV14
from v14.qualification import sample_metrics


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
            hr_blocks=1,
            selector_channels=8,
            production_tile_lr=8,
            production_overlap_lr=2,
            tiles_per_epoch=1,
            validation_tiles=1,
            clean_epochs=1,
            robust_epochs=1,
            selector_epochs=1,
        )
        config.validate()
        return config

    def _batch(self, size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(14)
        albedo = torch.rand(1, 3, size, size)
        normal = torch.rand(1, 2, size, size) * 0.8 - 0.4
        material = torch.rand(1, 3, size, size)
        return albedo, normal, material

    def test_initial_candidate_is_exact_baseline(self) -> None:
        model = NSAMDRV14(self._config()).eval()
        albedo, normal, material = self._batch()
        with torch.no_grad():
            outputs = model(albedo, normal, material)
        self.assertEqual(tuple(outputs["candidate_albedo"].shape[-2:]), (32, 32))
        self.assertTrue(torch.equal(outputs["candidate_albedo"], outputs["baseline_albedo"]))
        self.assertTrue(torch.equal(outputs["candidate_material"], outputs["baseline_material"]))
        self.assertLess(float((outputs["candidate_normal"] - outputs["baseline_normal"]).abs().max()), 1.0e-6)

    def test_architecture_contains_no_retired_pixel_authority(self) -> None:
        contract = NSAMDRV14(self._config()).architecture_contract()
        self.assertEqual(contract["schema"], MODEL_SCHEMA)
        self.assertEqual(tuple(contract["retiredComponents"]), ())
        self.assertFalse(contract["geometryPixelAuthority"])
        self.assertFalse(contract["seamPixelAuthority"])
        self.assertFalse(contract["profilePixelAuthority"])
        for forbidden in ("geometry_net", "boundary_renderer", "seam_restorer"):
            self.assertFalse(hasattr(NSAMDRV14(self._config()), forbidden))

    def test_candidate_loss_backpropagates_through_hr_refiner(self) -> None:
        config = self._config()
        model = NSAMDRV14(config)
        model.set_candidate_training()
        albedo, normal, material = self._batch()
        outputs = model(albedo, normal, material)
        batch = {
            "target_albedo": (outputs["baseline_albedo"].detach() * 0.9 + 0.05).clamp(0, 1),
            "target_normal": outputs["baseline_normal"].detach(),
            "target_material": outputs["baseline_material"].detach(),
        }
        losses = candidate_loss(outputs, batch, config)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.hr_refiner.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)

    def test_lattice_metric_penalizes_extra_cell_constant_residual(self) -> None:
        config = self._config()
        model = NSAMDRV14(config).eval()
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

    def test_tiled_inference_keeps_exact_four_x_dimensions(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
