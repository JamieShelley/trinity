from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from tools.nsamdr.neural.v14.config import V16Config
from tools.nsamdr.neural.v16.conditioning import StructureConditionedV16Candidate
from tools.nsamdr.neural.v16.structure import (
    StructureConditioningEncoder,
    derive_structure_targets,
    structure_channels,
)


class StructureTargetTests(unittest.TestCase):
    def test_shared_boundary_uses_all_physical_maps(self) -> None:
        h = w = 64
        albedo = np.zeros((h, w, 3), dtype=np.float32)
        normal = np.zeros((h, w, 2), dtype=np.float32)
        material = np.zeros((h, w, 3), dtype=np.float32)
        albedo[:, 32:, :] = 1.0
        normal[16:, :, 0] = 0.75
        material[:, :20, 2] = 1.0

        targets = derive_structure_targets(albedo, normal, material)
        self.assertGreater(float(targets["boundaryStrength"].max()), 0.9)
        self.assertTrue(bool(targets["boundaryMask"].any()))
        self.assertEqual(structure_channels(targets).shape, (h, w, 5))
        self.assertAlmostEqual(float(targets["distancePixels"][32, 31]), 0.0, places=5)

    def test_conditioning_encoder_is_phase_neutral_and_shape_stable(self) -> None:
        model = StructureConditioningEncoder(
            scale=4,
            channels=16,
            output_channels=12,
            blocks=2,
        ).eval()
        evidence = torch.randn(1, 8, 16, 20)
        with torch.no_grad():
            result = model(evidence)
            edges = model.analytic_edges(evidence)

        self.assertEqual(tuple(result.shape), (1, 12, 64, 80))
        self.assertEqual(tuple(edges.shape), (1, 3, 16, 20))
        forbidden = (nn.PixelShuffle, nn.ConvTranspose2d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))

    def test_conditioned_candidate_keeps_v16_physical_output_contract(self) -> None:
        config = V16Config(
            train_lr_size=8,
            train_hr_size=32,
            validation_lr_size=8,
            validation_hr_size=32,
            lr_context_channels=8,
            lr_blocks=1,
            hr_channels=24,
            swin_groups=1,
            swin_blocks_per_group=1,
            swin_num_heads=4,
            swin_window_size=4,
            map_tail_blocks=1,
            selector_channels=8,
            use_gradient_checkpointing=False,
            tiles_per_epoch=1,
            validation_tiles=4,
            minimum_heldout_samples=4,
            clean_epochs=1,
            robust_epochs=1,
            selector_epochs=1,
            production_tile_lr=8,
            production_overlap_lr=2,
        )
        model = StructureConditionedV16Candidate(
            config,
            structure_channels=8,
            structure_blocks=1,
        ).eval()
        albedo = torch.rand(1, 3, 8, 8)
        normal = torch.rand(1, 2, 8, 8) * 0.5 - 0.25
        material = torch.rand(1, 3, 8, 8)
        with torch.no_grad():
            output = model(albedo, normal, material)

        self.assertEqual(tuple(output["candidate_albedo"].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(output["candidate_normal"].shape), (1, 2, 32, 32))
        self.assertEqual(tuple(output["candidate_material"].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(output["structure_context"].shape), (1, 8, 32, 32))
        forbidden = (nn.PixelShuffle, nn.ConvTranspose2d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.structure.modules()))


if __name__ == "__main__":
    unittest.main()
