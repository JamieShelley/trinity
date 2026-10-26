"""No-trace Raven inference: real source verification and fail-closed model gates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from tools.nsamdr.neural.v178.auto_preview import run, _predict_tiled
from tools.nsamdr.neural.v178.contour_model import (
    SCHEMA, StructuralContourHead, physical_edge_evidence,
)
from tools.nsamdr.neural.v178.golden_patch import create, sha256
from tools.nsamdr.neural.train_nsamdr_v178_contour import source_hash


class AutoRavenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(),2))

    def _folder(self, root:Path)->Path:
        h=w=64
        ys,xs=np.indices((h,w))
        img=np.full((h,w,4),88,dtype=np.uint8)
        img[:,:,3]=255
        # Diagonal authored panel edge; the source is not used as its own target.
        img[xs>ys*.65+13,:3]=180
        normal=np.full((h,w,4),127,dtype=np.uint8)
        material=np.zeros((h,w,4),dtype=np.uint8)
        material[:,:,3]=255
        material[:,32:,2]=255
        for name,im in (("a.png",img),("n.png",normal),("m.png",material)):
            self.assertTrue(cv2.imwrite(str(root/name),im))
        manifest=root/"ship.materials.tsv"
        manifest.write_text(
            "albedo\tnormal\tmaterial\tshader_family\tmaterial_channel\n"
            "a.png\tn.png\tm.png\tv5_packed\t0\n",
            encoding="utf-8",
        )
        create(manifest,root/"golden",crop_x=0,crop_y=0,size=64)
        return root/"golden/a_x0_y0_s64"

    def _checkpoint(self, root:Path, passed:bool)->Path:
        ck=root/"model.pt"
        torch.save({
            "schema":SCHEMA,
            "config":{"sourceHash":source_hash(Path(__file__).resolve().parents[3])},
            "syntheticPass":passed,
            "model":StructuralContourHead().state_dict(),
            "productionPromotionAllowed":False,
        },ck)
        return ck

    def test_structural_native_edges_are_not_material_ids(self):
        rgb=torch.zeros((1,3,32,32))
        rgb[:,:,:,16:]=1.
        normal=torch.zeros((1,2,32,32))
        observation=physical_edge_evidence(rgb,normal)
        self.assertEqual(observation.shape,(1,1,32,32))
        self.assertGreater(float(observation.max()),.3)
        self.assertLess(float(observation[:,:,:,:8].max()),.01)
        with self.assertRaisesRegex(ValueError,"aligned native"):
            physical_edge_evidence(rgb,normal[:,:,:-1,:])

    def test_automatic_preview_no_trace_and_no_false_quality_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=self._folder(root)
            self.assertFalse((folder/"trace.json").exists())
            ck=self._checkpoint(root,True)
            result=run(folder,ck,device_name="cpu",tile=32)
            self.assertEqual(result["decision"],"REAL_EVE_UNQUALIFIED_AUTOMATIC_CONTOUR_PREVIEW")
            self.assertTrue(result["nativeEvidenceVerified"])
            self.assertFalse(result["humanContourTracingRequired"])
            self.assertFalse(result["realEveAccuracyProven"])
            self.assertFalse(result["referenceSynthesizedFromInput"])
            self.assertFalse(result["materialIdUsedAsContourInput"])
            self.assertFalse(result["productionPromotionAllowed"])
            self.assertEqual(result["outputDimensions"],[256,256])
            self.assertTrue(Path(result["candidateMask"]).is_file())
            self.assertTrue(Path(result["visualComparison"]).is_file())
            self.assertIn("sourceSha256",result)
            self.assertNotIn("contourIoU",result)

    def test_real_black_contour_is_rejected_with_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=self._folder(root)
            ck=self._checkpoint(root,True)
            saved=torch.load(ck,weights_only=True)
            # Deliberately suppress every contour logit. It must NEVER be
            # labelled a successful real Raven reconstruction.
            saved["model"]["delta.bias"]=torch.tensor([-70.])
            torch.save(saved,ck)
            report=run(folder,ck,device_name="cpu",tile=32)
            self.assertTrue(report["realContourCollapsed"])
            self.assertEqual(report["decision"],"REAL_EVE_CONTOUR_COLLAPSE_REJECTED")
            self.assertLess(report["signalRetentionRatio"],.05)
            self.assertLess(report["candidateActiveFraction"],.001)
            self.assertFalse(report["productionPromotionAllowed"])
            self.assertTrue((folder/"automatic_contour_preview/report.json").exists())

    def test_failed_or_wrong_generation_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=self._folder(root)
            ck=self._checkpoint(root,False)
            with self.assertRaisesRegex(RuntimeError,"synthetic contour proof failed"):
                run(folder,ck,device_name="cpu",tile=32)
            state=torch.load(ck,weights_only=True)
            state["schema"]="NSAMDR_V178_CONTOUR_HEAD_V1"
            state["syntheticPass"]=True
            torch.save(state,ck)
            with self.assertRaisesRegex(RuntimeError,"old V1"):
                run(folder,ck,device_name="cpu",tile=32)

    def test_tiled_matches_dense_same_native_evidence(self):
        torch.manual_seed(1234)
        a=torch.rand((1,3,40,48))
        n=torch.rand((1,2,40,48))*2-1
        m=torch.rand((1,3,40,48))
        m[:,0:1]=physical_edge_evidence(a,n)
        model=StructuralContourHead()
        # Exercise non-zero learned contour corrections; the zero-initialised
        # network alone would trivially hide tile seams.
        with torch.no_grad():
            model.delta.weight.normal_(0,.02)
            model.delta.bias.fill_(.08)
            dense=model(a,n,m)
        base,cand=_predict_tiled(model,a,n,m,tile=20,halo=20)
        self.assertGreater(float((dense["candidate_contour"] -
                                  dense["baseline_contour"]).abs().max()),.001)
        self.assertLess(float((base-dense["baseline_contour"]).abs().max()),1.e-5)
        self.assertLess(float((cand-dense["candidate_contour"]).abs().max()),1.e-5)

    def test_native_source_tampering_blocks_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=self._folder(root)
            with (folder/"native_albedo.png").open("ab") as stream:
                stream.write(b"mutated")
            with self.assertRaisesRegex(RuntimeError,"evidence changed"):
                run(folder,self._checkpoint(root,True),device_name="cpu",tile=32)


if __name__=="__main__":
    unittest.main()
