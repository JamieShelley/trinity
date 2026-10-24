"""Synthetic-only V17.7 model, reference and training gate tests."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest

import torch

from tools.nsamdr.neural.v177.enhancement_model import NativeEnhancer4x
from tools.nsamdr.neural.v177.synthetic_geometry import make_batch
from tools.nsamdr.neural.train_nsamdr_v177_enhancement import _loss, _score, _contour_error, source_digest


class NativeEnhancerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2, torch.get_num_threads()))

    def test_zero_initialized_c1_equals_semantic_b(self):
        a,n,m,*_ = make_batch(2,16,seed=73)
        model=NativeEnhancer4x()
        p=model(a,n,m,material_policy="legacy_mask")
        for key in ("albedo", "normal", "material"):
            self.assertTrue(torch.allclose(
                p["candidate_"+key],p["baseline_"+key],atol=1e-6
            ), key)

    def test_model_receives_full_native_input_and_expands_exactly_four(self):
        a,n,m,*_ = make_batch(1,21,seed=72)
        p=NativeEnhancer4x()(a,n,m,material_policy="legacy_mask")
        self.assertEqual(tuple(a.shape), (1,3,21,21))
        self.assertEqual(tuple(p["candidate_albedo"].shape),(1,3,84,84))
        self.assertEqual(tuple(p["candidate_normal"].shape),(1,2,84,84))
        self.assertEqual(tuple(p["candidate_material"].shape),(1,3,84,84))

    def test_v5_material_selector_cannot_be_modified(self):
        a,n,m,*_ = make_batch(1,16,seed=74)
        model=NativeEnhancer4x()
        with torch.no_grad():
            model.delta.bias.fill_(1.0)
        p=model(a,n,m,material_policy="v5_selector")
        self.assertTrue(torch.equal(
            p["candidate_material"][:,0],p["baseline_material"][:,0]))
        self.assertTrue(torch.isfinite(p["candidate_normal"]).all())
        self.assertLessEqual(float(p["candidate_normal"].square().sum(1).max()),.999**2+1e-5)

    def test_synthetic_targets_come_from_clean_hr_geometry(self):
        a,n,m,ta,tn,tm,protect=make_batch(2,20,seed=90)
        self.assertEqual(tuple(ta.shape),(2,3,80,80))
        self.assertEqual(tuple(tn.shape),(2,2,80,80))
        self.assertEqual(tuple(tm.shape),(2,3,80,80))
        self.assertEqual(tuple(protect.shape),(2,1,80,80))
        self.assertFalse(torch.allclose(ta, torch.nn.functional.interpolate(
            a,size=(80,80),mode="nearest")))
        a2,n2,m2,ta2,tn2,tm2,_=make_batch(2,20,seed=91)
        self.assertFalse(torch.equal(ta,ta2))
        self.assertTrue(torch.equal(ta,make_batch(2,20,seed=90)[3]))

    def test_backprop_updates_multichannel_head(self):
        torch.manual_seed(5)
        a,n,m,ta,tn,tm,protect=make_batch(2,16,seed=55)
        model=NativeEnhancer4x()
        initial=model.delta.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
        prediction=model(a,n,m,material_policy="legacy_mask")
        loss,_=_loss(prediction,(ta,tn,tm),protect)
        self.assertTrue(bool(torch.isfinite(loss).item()))
        loss.backward()
        self.assertGreater(float(model.delta.weight.grad.abs().sum()),0)
        optimizer.step()
        self.assertFalse(torch.equal(initial,model.delta.weight.detach()))

    def test_shared_spatial_upsampler_and_contour_quality_gate(self):
        from tools.nsamdr.neural.v177.enhancement_model import SCHEMA
        self.assertIn("V2_CONTOUR",SCHEMA)
        model=NativeEnhancer4x()
        self.assertFalse(any(isinstance(layer,torch.nn.PixelShuffle)
                             for layer in model.modules()))
        self.assertTrue(any(isinstance(layer,torch.nn.Upsample)
                            for layer in model.modules()))
        target=torch.zeros((1,1,32,32))
        target[:,:,:,16:]=1.0
        exact=_contour_error(target,target)
        stepped=target.clone()
        stepped[:,:,::2,:]=torch.roll(stepped[:,:,::2,:],3,dims=3)
        self.assertEqual(float(exact),0.0)
        self.assertGreater(float(_contour_error(stepped,target)),0.0)

    def test_contour_metric_reported_separately_from_combined_recovery(self):
        model=NativeEnhancer4x()
        score=_score(model,torch.device("cpu"),seed=190,patch=16)
        self.assertAlmostEqual(score["contourRecovery"],0.0,places=5)
        self.assertGreater(score["contourBaseline"],0.0)
        self.assertAlmostEqual(score["contourCandidate"],score["contourBaseline"],places=6)

    def test_validation_is_held_out_and_source_is_hashed(self):
        model=NativeEnhancer4x()
        report=_score(model,torch.device("cpu"),seed=131,patch=16)
        self.assertAlmostEqual(report["relativeRecovery"],0.0,places=5)
        self.assertGreater(report["baselineLoss"],0.0)
        self.assertEqual(report["syntheticHoldoutSeed"],1000131)
        self.assertEqual(len(source_digest()),64)


if __name__=="__main__":
    unittest.main()
