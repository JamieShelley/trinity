from __future__ import annotations

import unittest

import torch
from torch import nn

from tools.nsamdr.neural.v14.config import V16Config
from tools.nsamdr.neural.v17.model import (
    NSAMDRV17,
    _relative_coordinate_channels,
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
    def test_relative_coordinates_repeat_per_lr_texel_without_absolute_uv(self) -> None:
        coords = _relative_coordinate_channels(
            height=8,
            width=8,
            scale=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(coords.shape), (1, 9, 8, 8))
        self.assertTrue(torch.equal(coords[:, :, :4, :4], coords[:, :, 4:, :4]))
        self.assertTrue(torch.equal(coords[:, :, :4, :4], coords[:, :, :4, 4:]))
        self.assertEqual(
            [round(float(v), 2) for v in coords[0, 0, 0, :4]],
            [-0.75, -0.25, 0.25, 0.75],
        )

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

    def test_v17_has_no_pixelshuffle_or_transposed_convolution(self) -> None:
        model = NSAMDRV17(
            _config(),
            encoder_channels=16,
            neighbourhood_channels=24,
            decoder_hidden_channels=32,
        )
        forbidden = (nn.PixelShuffle, nn.ConvTranspose2d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        contract = model.architecture_contract()
        self.assertTrue(contract["relativeSubpixelCoordinatesUsed"])
        self.assertFalse(contract["absoluteUvCoordinatesUsed"])
        self.assertTrue(contract["learnedHrReconstructionDecoder"])
        self.assertFalse(contract["fixedHrSwinRefinerUsed"])


if __name__ == "__main__":
    unittest.main()
