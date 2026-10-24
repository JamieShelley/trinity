"""Material coverage and semantic-mip contracts for the V17.7 Raven renderer."""
from __future__ import annotations
import csv
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.nsamdr.neural.v177.material_coverage import (
    _quantize_selector, rgba_coverage, semantic_half, selector_has_discrete_slots, weights_to_bgra,
    write_coverage, write_semantic_mips,
)
from tools.nsamdr.neural.v176.raven_renderer import _write_material_manifest


class V177MaterialCoverageTests(unittest.TestCase):
    def test_different_nonadjacent_v5_labels_mix_without_new_identity(self):
        native=np.zeros((24,24),dtype=np.uint8)
        native[:,12:]=255
        rgba=rgba_coverage(native)
        self.assertEqual(rgba.shape,(96,96,4))
        self.assertTrue(np.all(rgba[:,:,1:3]==0))  # absent slots never invented
        self.assertEqual(set(np.unique(rgba[:,:35,0]).tolist()),{255})
        self.assertEqual(set(np.unique(rgba[:,65:,3]).tolist()),{255})
        self.assertTrue(np.any((rgba[:,:,0]>0)&(rgba[:,:,0]<255)))
        self.assertTrue(np.any((rgba[:,:,3]>0)&(rgba[:,:,3]<255)))
        self.assertTrue(np.all(rgba.astype(np.int16).sum(axis=2)==255))

    def test_authored_intermediate_selector_remains_unmodified(self):
        pure=np.zeros((32,32),dtype=np.uint8)
        pure[:,16:]=255
        self.assertTrue(selector_has_discrete_slots(pure))
        authored=np.full((32,32),127,dtype=np.uint8)
        self.assertFalse(selector_has_discrete_slots(authored))
        partly=authored.copy()
        partly[:,:31]=0
        self.assertTrue(selector_has_discrete_slots(partly))

    def test_flat_selector_preserved_exactly(self):
        native=np.full((16,16),170,dtype=np.uint8)
        rgba=rgba_coverage(native)
        self.assertTrue(np.all(rgba[:,:,2]==255))
        self.assertTrue(np.all(rgba[:,:,[0,1,3]]==0))
        with self.assertRaises(ValueError):
            rgba_coverage(native,scale=2)

    def test_reference_smooths_staircase_without_changing_interior_classes(self):
        h=w=48
        ys,xs=np.indices((h,w))
        native=np.where(xs>ys*.65+9,255,0).astype(np.uint8)
        coverage=rgba_coverage(native,smoothing=.7).astype(np.float32)/255
        nearest=cv2.resize(native,(w*4,h*4),interpolation=cv2.INTER_NEAREST)
        # Boundary coverage must not simply enlarge each pixel into a 4x4
        # constant block as the old baseline did.
        self.assertTrue(np.any((coverage[:,:,0]>0)&(coverage[:,:,0]<1)))
        self.assertTrue(np.all(coverage[16:48,0:16,0]==1))
        self.assertGreater(float(np.mean(np.abs(coverage[:,:,0] -
            (nearest==0).astype(np.float32)))),.0001)

    def test_coverage_mips_average_weights_and_keep_sum(self):
        native=np.zeros((12,12),dtype=np.uint8)
        native[:,6:]=255
        full=weights_to_bgra(rgba_coverage(native))
        current=full
        while min(current.shape[:2])>1:
            current=semantic_half(current,mode="coverage")
            rgba=current[:,:,[2,1,0,3]].astype(np.int16)
            self.assertTrue(np.all(rgba.sum(axis=2)==255))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"map.png"
            write_coverage(native,path)
            levels=write_semantic_mips(path,mode="coverage")
            self.assertGreater(levels,3)
            for level in range(1,levels+1):
                self.assertTrue((Path(folder)/f"map.mip{level}.png").is_file())

    def test_native_coverage_tiling_matches_whole_image_across_tile_seams(self):
        ys,xs=np.indices((67,89))
        native=np.where(xs>ys*.65+9,255,0).astype(np.uint8)
        expected=weights_to_bgra(rgba_coverage(native))
        with tempfile.TemporaryDirectory() as folder:
            result=Path(folder)/"tiled.png"
            write_coverage(native,result,tile_native=32)
            got=cv2.imread(str(result),cv2.IMREAD_UNCHANGED)
            self.assertEqual(got.shape,expected.shape)
            difference=np.abs(got.astype(np.int16)-expected.astype(np.int16))
            self.assertLessEqual(int(difference.max()),2)
            self.assertTrue(np.all(got.astype(np.int16).sum(axis=2)==255))

    def test_categorical_mip_retains_identity_but_other_channels_are_filtered(self):
        img=np.zeros((8,8,4),dtype=np.uint8)
        img[:4,:4,0]=255
        img[::2,::2,1]=255
        down=semantic_half(img,mode="categorical",selector_channel=0)
        self.assertEqual(set(np.unique(down[:,:,0]).tolist()),{0,255})
        self.assertTrue(np.any((down[:,:,1]>0)&(down[:,:,1]<255)))

    def test_normal_mip_valid_and_packed_overlap_rejected(self):
        img=np.full((8,8,4),127,dtype=np.uint8)
        img[:,:,0]=255
        img[:,:,1]=255
        out=semantic_half(img,mode="normal",normal_channels=(0,1))
        xy=out[:,:,:2].astype(np.float32)/127.5-1
        self.assertTrue(np.all(np.sum(xy*xy,axis=2)<=1.0+1e-4))
        with self.assertRaisesRegex(ValueError,"overlap"):
            semantic_half(img,mode="packed",selector_channel=0,normal_channels=(0,1))

    def test_coverage_paths_optional_and_never_in_authored_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            image=root/"source.png"
            image.write_bytes(b"not read by manifest writer")
            fields=["albedo","shader_family","material"]
            rows=[{"albedo":str(image),"shader_family":"v5_packed","material":str(image)}]
            original=root/"c0.tsv"
            candidate=root/"c1.tsv"
            _write_material_manifest(original,fields,rows,[],{},root,"FROZEN")
            _write_material_manifest(candidate,fields,rows,[],{},root,"B",
                                     material_weights=[str(root/"weights.png")])
            with original.open(newline="",encoding="utf-8") as handle:
                text=handle.read()
            self.assertNotIn("material_weights",text)
            with candidate.open(newline="",encoding="utf-8") as handle:
                text=handle.read()
            self.assertIn("material_weights",text)
            self.assertIn("weights.png",text)

    def test_native_shader_uses_coverage_only_when_explicitly_bound(self):
        root=Path(__file__).resolve().parents[3]
        shader=(root/"trinityal/tests/nsamdr/NSAMDRPreview.hlsl").read_text()
        pipeline=(root/"trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp").read_text()
        processor=(root/"trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp").read_text()
        self.assertIn("Texture2D gMaterialWeights : register(t9)",shader)
        self.assertIn("gDebug.x > 0.5 && shaderFamily >= 2",shader)
        self.assertIn("PSSetShaderResources(0, 10, textureViews)",pipeline)
        self.assertIn('value("material_weights")',processor)


if __name__=="__main__":
    unittest.main()
