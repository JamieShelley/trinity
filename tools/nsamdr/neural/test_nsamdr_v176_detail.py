"""CPU-only V17.6 Stage 10A topology, wavelet and physical-safety tests.

No authenticated Stage 9 checkpoint or training corpus is needed. These tests
prove the new module behaves correctly before any GPU experiment is built.
"""
from __future__ import annotations

import unittest
import torch
from torch import nn
from tools.nsamdr.neural.test_nsamdr_v17_model import _config
from tools.nsamdr.neural.v17.model import NSAMDRV17
from tools.nsamdr.neural.v176.fine_detail import (
    FineDetailSpecialist,NSAMDRV176Detail,
)
from tools.nsamdr.neural.v176.physical_composition import (
    compose_albedo_detail,gaussian_highpass,
)
from tools.nsamdr.neural.v176.wavelet_supervision import (
    BAND_NAMES,LEVELS,LOSS_WEIGHTS,detail_supervision_loss,
    stationary_haar_details,
)


def _model() -> NSAMDRV176Detail:
    return NSAMDRV176Detail(
        NSAMDRV17(_config(),encoder_channels=16,mid_channels=12,
                  detail_channels=10,decoder_blocks=1),
        detail_channels=8,
    )


def _lr() -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    torch.manual_seed(100)
    return (
        torch.rand(1,3,8,8),
        torch.rand(1,2,8,8)*0.8-0.4,
        torch.rand(1,3,8,8),
    )


class FineDetailArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(min(torch.get_num_threads(),2))

    def test_backbone_authentic_and_frozen(self) -> None:
        model=_model()
        self.assertEqual(len(model.trainable_parameters()),
                         len(list(model.specialist.parameters())))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.specialist.parameters()))
        model.train()
        self.assertTrue(model.specialist.training)
        self.assertFalse(model.backbone.training)

    def test_init_exactly_reproduces_original_all_candidate_maps(self) -> None:
        model=_model().eval()
        lr=_lr()
        with torch.no_grad():
            original=model.backbone(*lr)
            proposed=model(*lr)
        for key in ("candidate_albedo","candidate_normal","candidate_material",
                    "baseline_albedo","baseline_normal","baseline_material",
                    "candidate_residual_albedo"):
            self.assertTrue(torch.equal(original[key],proposed[key]),key)
        self.assertEqual(tuple(proposed["fine_detail_raw"].shape),(1,1,32,32))
        self.assertEqual(tuple(proposed["fine_detail_highpass"].shape),(1,1,32,32))
        self.assertEqual(float(proposed["fine_detail_highpass"].abs().max()),0.)

    def test_one_channel_specialist_and_exact_input_accounting(self) -> None:
        model=_model().eval()
        self.assertEqual(FineDetailSpecialist.INPUT_CHANNELS,10)
        with torch.no_grad():
            original=model.backbone(*_lr())
            evidence=model._evidence(*_lr(),original)
        self.assertEqual(tuple(evidence.shape),(1,10,32,32))
        self.assertEqual(tuple(model.specialist(evidence).shape),(1,1,32,32))
        with self.assertRaisesRegex(ValueError,"N x 10"):
            model.specialist(torch.rand(1,9,32,32))

    def test_backbone_no_gradients_when_specialist_trains(self) -> None:
        model=_model().train()
        output=model(*_lr())
        target=(output["candidate_albedo"].detach()+
                torch.rand_like(output["candidate_albedo"])*.2-.1).clamp(0,1)
        total=detail_supervision_loss(output,{"target_albedo":target})["total"]
        total.backward()
        self.assertIsNotNone(model.specialist.head.weight.grad)
        self.assertGreater(float(model.specialist.head.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(torch.isfinite(total).item())

    def test_nonzero_detail_is_bounded_and_keeps_normal_material_unchanged(self) -> None:
        model=_model().eval()
        with torch.no_grad():
            model.specialist.head.weight.normal_(0.,.05)
            original=model.backbone(*_lr())
            outputs=model(*_lr())
        self.assertGreater(float(outputs["fine_detail_highpass"].abs().max()),0.)
        for key in ("candidate_normal","candidate_material",
                    "predicted_residual_normal","predicted_residual_material",
                    "candidate_raw_residual_normal","candidate_raw_residual_material"):
            self.assertTrue(torch.equal(original[key],outputs[key]),key)
        cap=model.backbone.config.albedo_residual_cap
        self.assertLessEqual(float(outputs["predicted_residual_albedo"].abs().max()),
                             cap+1e-6)
        self.assertEqual(tuple(outputs["candidate_albedo"].shape),(1,3,32,32))

    def test_scalar_correction_broadcasts_equally_to_each_legacy_albedo_channel(self) -> None:
        model=_model().eval()
        with torch.no_grad():
            source=model.backbone(*_lr())
            delta=torch.rand(1,1,32,32)*.02
            out=compose_albedo_detail(source,delta,
                 albedo_residual_cap=model.backbone.config.albedo_residual_cap)
        diff=out["candidate_raw_residual_albedo"]-source["candidate_raw_residual_albedo"]
        self.assertTrue(torch.allclose(diff[:,0:1],delta,atol=1e-7))
        self.assertTrue(torch.allclose(diff[:,1:2],delta,atol=1e-7))
        self.assertTrue(torch.allclose(diff[:,2:3],delta,atol=1e-7))

    def test_compositor_checks_both_albedo_shapes_and_accepts_matching_inputs(self) -> None:
        original = {
            "candidate_raw_residual_albedo": torch.zeros(2, 3, 16, 24),
            "baseline_albedo": torch.full((2, 3, 16, 24), 0.5),
            "candidate_albedo": torch.full((2, 3, 16, 24), 0.5),
        }
        delta = torch.zeros(2, 1, 16, 24)
        output = compose_albedo_detail(original, delta, albedo_residual_cap=0.4)
        self.assertTrue(torch.equal(
            output["candidate_albedo"], original["candidate_albedo"]
        ))
        self.assertEqual(tuple(output["candidate_albedo"].shape), (2, 3, 16, 24))
        wrong = dict(original)
        wrong["baseline_albedo"] = torch.zeros(2, 3, 16, 23)
        with self.assertRaisesRegex(ValueError, "aligned V17.2"):
            compose_albedo_detail(wrong, delta, albedo_residual_cap=0.4)

    def test_invalid_detail_shapes_and_cap_rejected(self) -> None:
        model=_model().eval()
        with torch.no_grad():
            original=model.backbone(*_lr())
        with self.assertRaises(ValueError):
            compose_albedo_detail(original,torch.randn(1,3,32,32),
                                  albedo_residual_cap=.4)
        with self.assertRaises(ValueError):
            compose_albedo_detail(original,torch.randn(1,1,32,32),
                                  albedo_residual_cap=-.4)

    def test_no_phase_pixelshuffle_or_transpose_or_quadtree(self) -> None:
        model=_model()
        self.assertFalse(any(isinstance(m,(nn.ConvTranspose2d,nn.PixelShuffle))
                             for m in model.specialist.modules()))
        c=model.architecture_contract()
        self.assertEqual(c["revision"],"V17.6-detail-10A-research")
        self.assertTrue(c["backboneFrozen"])
        self.assertTrue(c["candidateIdentityAtInitialization"])
        self.assertFalse(c["quadtreeEnabled"])
        self.assertFalse(c["absoluteCoordinatesUsed"])
        self.assertFalse(c["subpixelPhaseCoordinatesUsed"])
        self.assertEqual(c["detailDilations"],[1,2,4,1])

    def test_gaussian_highpass_removes_constant(self) -> None:
        image=torch.ones(1,1,32,32)*0.25
        out=gaussian_highpass(image)
        self.assertLess(float(out.abs().max()),1e-6)

    def test_highpass_shift_covariance_away_from_boundary(self) -> None:
        torch.manual_seed(12)
        image=torch.rand(1,1,40,40)
        a=gaussian_highpass(image)
        moved=torch.roll(image,shifts=(1,1),dims=(-2,-1))
        b=gaussian_highpass(moved)
        self.assertTrue(torch.allclose(
            torch.roll(a,shifts=(1,1),dims=(-2,-1))[...,8:-8,8:-8],
            b[...,8:-8,8:-8],atol=1e-6))

    def test_no_learned_change_in_original_backbone_modules(self) -> None:
        model=_model()
        self.assertEqual(model.backbone.architecture_contract()["revision"],
                         "V17.2-proof")
        self.assertEqual(model.architecture_contract()["outputChannels"],1)


class WaveletSupervisionTests(unittest.TestCase):
    def test_stationary_haar_preserves_size_and_six_directional_bands(self) -> None:
        source=torch.rand(2,1,32,32)
        bands=stationary_haar_details(source)
        self.assertEqual(set(bands),{
            f"L{level}_{name}" for level in LEVELS for name in BAND_NAMES})
        self.assertTrue(all(t.shape==source.shape for t in bands.values()))

    def test_constant_input_has_no_high_frequency_response(self) -> None:
        source=torch.ones(1,1,32,32)*.2
        details=stationary_haar_details(source)
        self.assertTrue(all(float(x.abs().max())<1e-6 for x in details.values()))

    def test_wavelets_are_shift_covariant_in_interior(self) -> None:
        torch.manual_seed(2)
        source=torch.rand(1,1,40,40)
        a=stationary_haar_details(source)
        b=stationary_haar_details(torch.roll(source,(1,1),(-2,-1)))
        for name in a:
            self.assertTrue(torch.allclose(
                torch.roll(a[name],(1,1),(-2,-1))[...,8:-8,8:-8],
                b[name][...,8:-8,8:-8],atol=1e-6),name)

    def test_wavelet_coefficients_are_differentiable(self) -> None:
        value=torch.rand(1,1,16,16,requires_grad=True)
        loss=sum(b.abs().mean() for b in stationary_haar_details(value).values())
        loss.backward()
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(value.grad).all().item())

    def test_supervision_target_does_not_enter_model_forward(self) -> None:
        model=_model()
        self.assertEqual(set(model.forward.__code__.co_varnames[:4]),
                         {"self","lr_albedo","lr_normal","lr_material"})

    def test_wavelet_loss_is_zero_when_candidate_matches_authored_target(self) -> None:
        model=_model().eval()
        with torch.no_grad():
            result=model(*_lr())
        batch={"target_albedo":result["candidate_albedo"].clone()}
        terms=detail_supervision_loss(result,batch)
        self.assertLess(float(terms["wavelet"]),1e-6)
        self.assertLess(float(terms["edge"]),1e-6)
        self.assertLess(float(terms["pixel"]),1e-6)
        self.assertTrue(torch.isfinite(terms["total"]).item())

    def test_supervision_detects_missing_high_frequency_target(self) -> None:
        model=_model().eval()
        with torch.no_grad():
            result=model(*_lr())
        x=torch.arange(32).view(1,1,1,32).float()
        stripes=(x.remainder(2)*.06-.03).expand(1,3,32,32)
        batch={"target_albedo":(result["candidate_albedo"]+stripes).clamp(0,1)}
        terms=detail_supervision_loss(result,batch)
        self.assertGreater(float(terms["wavelet"]),0.)
        self.assertEqual(set(LOSS_WEIGHTS),
                         {"wavelet","edge","pixel","protected","detail_magnitude"})

    def test_invalid_loss_reference_shape_is_rejected(self) -> None:
        model=_model().eval()
        with torch.no_grad():
            result=model(*_lr())
        with self.assertRaises(ValueError):
            detail_supervision_loss(result,{"target_albedo":torch.rand(1,1,32,32)})
        with self.assertRaises(ValueError):
            detail_supervision_loss(result,{"target_albedo":result["candidate_albedo"]},
                                    denominator_floor=0.)


if __name__=="__main__":
    unittest.main()
