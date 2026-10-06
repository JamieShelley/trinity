from __future__ import annotations

import unittest

import torch
from torch import nn

from tools.nsamdr.neural.v14.config import V16Config
from tools.nsamdr.neural.v17.model import MultiScaleResidualDecoder, NSAMDRV17


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
    def test_multiscale_decoder_emits_mid_and_hr_residual_bands(self) -> None:
        decoder = MultiScaleResidualDecoder(
            feature_channels=16,
            mid_channels=12,
            detail_channels=10,
            blocks_per_stage=1,
            scale=4,
        ).eval()
        features = torch.rand(1, 16, 8, 8)
        maps = torch.rand(1, 8, 8, 8)
        with torch.no_grad():
            output = decoder(features, maps)
        self.assertEqual(tuple(output["mid_residual"].shape), (1, 8, 16, 16))
        self.assertEqual(tuple(output["mid_residual_hr"].shape), (1, 8, 32, 32))
        self.assertEqual(tuple(output["detail_residual"].shape), (1, 8, 32, 32))
        self.assertEqual(tuple(output["albedo"].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(output["normal"].shape), (1, 2, 32, 32))
        self.assertEqual(tuple(output["material"].shape), (1, 3, 32, 32))

    def test_candidate_starts_exactly_at_deterministic_baseline(self) -> None:
        model = NSAMDRV17(
            _config(),
            encoder_channels=16,
            mid_channels=12,
            detail_channels=10,
            decoder_blocks=1,
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

    def test_v17_2_has_no_phase_decoder_pixelshuffle_or_transposed_convolution(self) -> None:
        model = NSAMDRV17(
            _config(),
            encoder_channels=16,
            mid_channels=12,
            detail_channels=10,
            decoder_blocks=1,
        )
        forbidden = (nn.PixelShuffle, nn.ConvTranspose2d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        contract = model.architecture_contract()
        self.assertEqual(contract["revision"], "V17.2-proof")
        self.assertFalse(contract["absoluteUvCoordinatesUsed"])
        self.assertFalse(contract["relativeSubpixelCoordinatesUsed"])
        self.assertFalse(contract["periodicPhaseEncodingUsed"])
        self.assertFalse(contract["localEnsembleUsed"])
        self.assertTrue(contract["multiScaleResidualDecoderUsed"])
        self.assertEqual(contract["decoderScales"], [2, 4])
        self.assertTrue(contract["resizeConvolutionUsed"])
        self.assertTrue(contract["learnedHrDetailStageUsed"])
        self.assertTrue(contract["learnedHrReconstructionDecoder"])
        self.assertFalse(contract["fixedHrSwinRefinerUsed"])


if __name__ == "__main__":
    unittest.main()
