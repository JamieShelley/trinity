from __future__ import annotations

import unittest

import torch
from torch import nn

from tools.nsamdr.neural.v14.config import V16Config
from tools.nsamdr.neural.v17.model import (
    NSAMDRV17,
    RelativeQueryDecoder,
    _query_axis_geometry,
)


def _config() -> V16Config:
    return V16Config(
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


class V17ModelTests(unittest.TestCase):
    def test_local_ensemble_geometry_is_continuous_and_weights_partition_unity(self) -> None:
        position, lower, upper, fraction = _query_axis_geometry(
            source_size=4,
            scale=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(position.numel(), 16)
        self.assertTrue(bool(torch.all(position[1:] >= position[:-1])))
        self.assertTrue(bool(torch.all(fraction >= 0.0)))
        self.assertTrue(bool(torch.all(fraction <= 1.0)))
        self.assertTrue(bool(torch.all(upper >= lower)))

        wy0 = 1.0 - fraction
        wy1 = fraction
        weights = (
            wy0.view(-1, 1) * wy0.view(1, -1),
            wy0.view(-1, 1) * wy1.view(1, -1),
            wy1.view(-1, 1) * wy0.view(1, -1),
            wy1.view(-1, 1) * wy1.view(1, -1),
        )
        total = sum(weights)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1.0e-6))

    def test_decoder_uses_only_two_relative_coordinate_channels(self) -> None:
        self.assertEqual(RelativeQueryDecoder.COORD_CHANNELS, 2)

    def test_candidate_starts_exactly_at_deterministic_baseline(self) -> None:
        model = NSAMDRV17(
            _config(),
            encoder_channels=16,
            neighbourhood_channels=24,
            decoder_hidden_channels=32,
        ).eval()
        albedo = torch.rand(1, 3, 8, 8)
        normal = torch.rand(1, 2, 8, 8) * 0.5 - 0.25
        material = torch.rand(1, 3, 8, 8)
        with torch.no_grad():
            output = model(albedo, normal, material)

        self.assertTrue(
            torch.equal(output["candidate_albedo"], output["baseline_albedo"])
        )
        self.assertTrue(
            torch.equal(output["candidate_normal"], output["baseline_normal"])
        )
        self.assertTrue(
            torch.equal(output["candidate_material"], output["baseline_material"])
        )
        self.assertEqual(tuple(output["candidate_albedo"].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(output["candidate_normal"].shape), (1, 2, 32, 32))
        self.assertEqual(tuple(output["candidate_material"].shape), (1, 3, 32, 32))

    def test_v17_has_no_phase_decoder_pixelshuffle_or_transposed_convolution(self) -> None:
        model = NSAMDRV17(
            _config(),
            encoder_channels=16,
            neighbourhood_channels=24,
            decoder_hidden_channels=32,
        )
        forbidden = (nn.PixelShuffle, nn.ConvTranspose2d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        contract = model.architecture_contract()
        self.assertEqual(contract["revision"], "V17.1-proof")
        self.assertTrue(contract["relativeSubpixelCoordinatesUsed"])
        self.assertFalse(contract["absoluteUvCoordinatesUsed"])
        self.assertFalse(contract["periodicPhaseEncodingUsed"])
        self.assertTrue(contract["localEnsembleUsed"])
        self.assertEqual(contract["localEnsembleAnchors"], 4)
        self.assertTrue(contract["learnedHrReconstructionDecoder"])
        self.assertFalse(contract["fixedHrSwinRefinerUsed"])


if __name__ == "__main__":
    unittest.main()
