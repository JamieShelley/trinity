"""G0/G1 real-pixel evidence, independent trace and G2 synthetic-contour tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from tools.nsamdr.neural.v178.golden_patch import (
    TRACE_SCHEMA, create, render_contour_trace, sha256,
    validate_trace,
)
from tools.nsamdr.neural.v178.contour_model import (
    SCHEMA, StructuralContourHead, contour_loss,
)
from tools.nsamdr.neural.v177.synthetic_geometry import make_batch
from tools.nsamdr.neural.train_nsamdr_v178_contour import evaluate


class GoldenEvidenceTests(unittest.TestCase):
    def _fixture(self, folder:Path)->Path:
        h,w=96,112
        yy,xx=np.indices((h,w))
        a=np.empty((h,w,4),dtype=np.uint8)
        a[:,:,0]=np.uint8((xx*2)%256)
        a[:,:,1]=np.uint8((yy*3)%256)
        a[:,:,2]=np.uint8((xx+yy)%256)
        a[:,:,3]=255
        n=np.full((48,56,4),127,dtype=np.uint8)
        p=np.zeros((h,w,4),dtype=np.uint8)
        p[:,w//2:,2]=255
        self.assertTrue(cv2.imwrite(str(folder/"a.png"),a))
        self.assertTrue(cv2.imwrite(str(folder/"n.png"),n))
        self.assertTrue(cv2.imwrite(str(folder/"m.png"),p))
        manifest=folder/"ship.materials.tsv"
        manifest.write_text(
            "albedo\tnormal\tmaterial\tshader_family\tmaterial_channel\n"
            "a.png\tn.png\tm.png\tv5_packed\t0\n",
            encoding="utf-8",
        )
        return manifest

    def test_real_launcher_import_and_prepared_manifest_routing(self):
        # Runs the actual launched entry point, not merely create().
        # Fake only external SharedCache preparation; this catches import-path
        # failures on the normal Windows python -u execution route.
        from unittest.mock import patch
        from tools.nsamdr.neural.golden_nsamdr_v178 import main
        with tempfile.TemporaryDirectory() as directory:
            repo=Path(directory)
            manifest=self._fixture(repo)
            prepared=(None,None,None,None,None,None,manifest)
            with patch("tools.nsamdr.eve_asset_test.prepare_asset",
                       return_value=prepared) as prepare:
                rc=main(["--repo-root",str(repo),"--action","prepare",
                         "--shared-cache",str(repo/"mock_cache"),
                         "--source-name","a.png","--crop-x","8",
                         "--crop-y","12","--size","64"])
            self.assertEqual(rc,0)
            self.assertEqual(prepare.call_count,1)
            report=repo/"artifacts/nsamdr/golden_patch/a_x8_y12_s64/evidence.json"
            self.assertTrue(report.is_file())
            self.assertEqual(json.loads(report.read_text())["decision"],
                             "AWAITING_CURATED_CONTOUR_REFERENCE")

    def test_native_evidence_unchanged_and_explicitly_unqualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=self._fixture(root)
            before=(root/"a.png").read_bytes()
            result=create(manifest,root/"golden",crop_x=8,crop_y=12,size=64)
            self.assertEqual(result["decision"],"AWAITING_CURATED_CONTOUR_REFERENCE")
            self.assertFalse(result["trainingAccessAllowed"])
            self.assertFalse(result["productionPromotionAllowed"])
            self.assertEqual(result["identity"]["nativeRoi"],[8,12,64,64])
            saved=root/"golden/a_x8_y12_s64"
            source=cv2.imread(str(root/"a.png"),cv2.IMREAD_UNCHANGED)
            native=cv2.imread(str(saved/"native_albedo.png"),cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(native,source[12:76,8:72])
            self.assertEqual(cv2.imread(str(saved/"native_normal.png"),
                                           cv2.IMREAD_UNCHANGED).shape[:2],(32,32))
            self.assertEqual((root/"a.png").read_bytes(),before)
            repeated=create(manifest,root/"golden",crop_x=8,crop_y=12,size=64)
            self.assertEqual(repeated["identity"],result["identity"])
            # Source evidence cannot silently mutate after registration.
            with (saved/"native_albedo.png").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError,"source evidence changed"):
                create(manifest,root/"golden",crop_x=8,crop_y=12,size=64)

    def test_real_single_patch_fit_emits_visual_evidence_but_never_promotes(self):
        from tools.nsamdr.neural.v178.real_fit import fit, SCHEMA_FIT
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=self._fixture(root)
            result=create(manifest,root/"golden",crop_x=8,crop_y=12,size=64)
            patch=root/"golden/a_x8_y12_s64"
            trace=json.loads((patch/"trace.template.json").read_text())
            trace["author"]="independent reviewer"
            trace["paths"]=[{"points":[[5.,10.],[19.,24.],[38.,42.]],
                             "widthNative":1.25}]
            (patch/"trace.json").write_text(json.dumps(trace),encoding="utf-8")
            output=fit(patch,device_name="cpu",steps=2,patch_size=16,seed=3001)
            self.assertEqual(output["schema"],SCHEMA_FIT)
            self.assertIn(output["decision"],(
                "SINGLE_PATCH_CAPACITY_PASS_ONLY","REAL_PATCH_FIT_INSUFFICIENT"
            ))
            self.assertFalse(output["realEveGeneralisationVerified"])
            self.assertFalse(output["productionPromotionAllowed"])
            self.assertTrue(output["targetUsedInTraining"])
            self.assertTrue(Path(output["comparisonSheet"]).is_file())
            self.assertTrue(Path(output["checkpoint"]).is_file())
            self.assertEqual(output["sourceSha256"],result["identity"]["sourceSha256"])

    def test_template_is_not_a_clean_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=self._fixture(root)
            create(manifest,root/"golden",crop_x=8,crop_y=12,size=64)
            patch=root/"golden/a_x8_y12_s64"
            with self.assertRaisesRegex(RuntimeError,"human-curated trace"):
                render_contour_trace(patch)
            template=json.loads((patch/"trace.template.json").read_text())
            self.assertEqual(template["paths"],[])
            template.update({"author":"human operator",
                             "paths":[{"points":[[5.25,10.5],[20.5,18.25],[38.5,41.5]],
                                       "widthNative":.75}]})
            (patch/"trace.json").write_text(json.dumps(template),encoding="utf-8")
            output=render_contour_trace(patch)
            image=cv2.imread(str(output),cv2.IMREAD_GRAYSCALE)
            self.assertEqual(image.shape,(256,256))
            self.assertGreater(int(np.count_nonzero(image)),0)
            self.assertTrue(((image>0)&(image<255)).any())
            provenance=json.loads((patch/"curated_contours.provenance.json").read_text())
            self.assertFalse(provenance["referenceIsNetworkGenerated"])
            self.assertFalse(provenance["qualifiedForRealEveTraining"])
            template["sourceSha256"]="f"*64
            (patch/"trace.json").write_text(json.dumps(template),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                validate_trace(patch)


class ContourHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2,torch.get_num_threads()))

    def test_native_evidence_to_smooth_4x_and_exact_initial_fallback(self):
        a,n,m,*_=make_batch(1,16,seed=171)
        net=StructuralContourHead()
        p=net(a,n,m)
        self.assertEqual(tuple(p["candidate_contour"].shape),(1,1,64,64))
        self.assertLess(
            float((p["baseline_contour"]-p["candidate_contour"]).abs().max()),
            .00011
        )
        self.assertFalse(any(isinstance(v,torch.nn.PixelShuffle)
                             for v in net.modules()))
        with self.assertRaisesRegex(ValueError,"aligned native"):
            net(a,n[:,:,:-1,:],m)

    def test_contour_geometry_loss_penalizes_grid_pattern(self):
        target=torch.zeros((1,1,64,64))
        target[:,:,:,26:31]=1.
        exact,parts=contour_loss(target,target)
        distorted=target.clone()
        distorted[:,:,::4,:]=torch.roll(distorted[:,:,::4,:],4,dims=-1)
        worse,_=contour_loss(distorted,target)
        self.assertAlmostEqual(float(exact),0.,places=6)
        self.assertGreater(float(worse),.001)
        self.assertEqual(set(parts),{"coverage","tangent","curvature"})

    def test_foreground_aware_loss_rejects_all_black_contours(self):
        from tools.nsamdr.neural.v178.contour_model import (
            contour_supervised_loss,foreground_metrics
        )
        target=torch.zeros((1,1,64,64))
        target[:,:,:,28:32]=1.
        blank=torch.zeros_like(target)
        exact=target.clone()
        loss_blank,_=contour_supervised_loss(blank,target)
        loss_exact,_=contour_supervised_loss(exact,target)
        self.assertGreater(float(loss_blank),float(loss_exact)+.2)
        metrics=foreground_metrics(blank,target)
        self.assertEqual(metrics["contourRecall"],0.)
        self.assertEqual(metrics["contourIoU"],0.)
        self.assertEqual(metrics["predictedActiveFraction"],0.)
        self.assertAlmostEqual(foreground_metrics(exact,target)["contourRecall"],1.)

    def test_contour_head_has_training_gradient_and_independent_holdout(self):
        a,n,m,_,_,target,_=make_batch(1,16,seed=173)
        model=StructuralContourHead(width=16)
        prediction=model(a,n,m)
        loss,_=contour_loss(prediction["candidate_contour"],target[:,:1])
        loss.backward()
        self.assertGreater(float(model.delta.weight.grad.abs().sum()),0)
        scores=evaluate(model,torch.device("cpu"),patch=16,seed=131,cases=2)
        self.assertEqual(scores["holdoutSeedStart"],1000131)
        self.assertAlmostEqual(scores["relativeContourRecovery"],0.,places=5)
        self.assertEqual(SCHEMA,"NSAMDR_V178_CONTOUR_HEAD_V4_AUXILIARY_INDEPENDENT")

    def test_contour_inputs_do_not_reuse_auxiliary_target_copies(self):
        from unittest.mock import patch
        from tools.nsamdr.neural.train_nsamdr_v178_contour import _batch
        images=make_batch(2,16,seed=173)
        changed=list(images)
        changed[2]=images[2].clone()
        changed[2][:,1:]=1.-changed[2][:,1:]
        batch_factory="tools.nsamdr.neural.train_nsamdr_v178_contour.make_batch"
        with patch(batch_factory,return_value=images):
            first=_batch(2,16,444,torch.device("cpu"))
            torch.manual_seed(999)
            repeated=_batch(2,16,444,torch.device("cpu"))
            other_seed=_batch(2,16,445,torch.device("cpu"))
        with patch(batch_factory,return_value=tuple(changed)):
            altered=_batch(2,16,444,torch.device("cpu"))
        for before,after in zip(first,altered):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        torch.testing.assert_close(first[0],images[0],rtol=0,atol=0)
        torch.testing.assert_close(first[1],images[1],rtol=0,atol=0)
        torch.testing.assert_close(first[3],images[5][:,:1],rtol=0,atol=0)
        torch.testing.assert_close(first[2],repeated[2],rtol=0,atol=0)
        self.assertGreater(float((first[2][:,1:]-other_seed[2][:,1:]).abs().max()),.1)
        self.assertGreaterEqual(float(first[2][:,1:].min()),0.)
        self.assertLessEqual(float(first[2][:,1:].max()),1.)


if __name__=="__main__":
    unittest.main()
