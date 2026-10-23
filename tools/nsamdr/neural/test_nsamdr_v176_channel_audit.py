"""Native Raven channel-audit regressions; synthetic local textures only."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.nsamdr.neural.v176 import channel_audit as audit


class V176ChannelAuditTests(unittest.TestCase):
    def test_native_roi_preserves_source_pixels_and_coordinates(self):
        source = np.arange(12*16, dtype=np.uint8).reshape(12,16)
        actual = audit._native_roi(source, (3,4), 6)
        self.assertEqual(actual.shape, (6,6))
        np.testing.assert_array_equal(actual, source[4:10,3:9])

    def test_interpolation_audit_exports_truthful_difference(self):
        with tempfile.TemporaryDirectory() as root:
            image = np.zeros((16,16), dtype=np.uint8)
            image[:, 8:] = 255
            record = audit._write_one(image, Path(root)/"material.png", "material")
            self.assertTrue(Path(record["sheet"]).is_file())
            self.assertGreater(record["differenceMAE_8bit"], 0.0)
            self.assertFalse(record["semanticWarning"].startswith("Corrected"))

    def test_audit_reads_native_dimensions_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            albedo = np.zeros((64,96,4), dtype=np.uint8)
            albedo[:, 32:, :3] = 255
            albedo[:, :, 3] = 255
            material = np.zeros((32,48,4), dtype=np.uint8)
            material[:,24:,:3] = 255
            self.assertTrue(cv2.imwrite(str(root/"a.png"), albedo))
            self.assertTrue(cv2.imwrite(str(root/"m.png"), material))
            manifest = root / "materials.tsv"
            manifest.write_text(
                "albedo\tmaterial\tmaterial_channel\n"
                "a.png\tm.png\t0\n",
                encoding="utf-8",
            )
            result = audit.run(manifest, root/"audit", crop_size=32, max_textures=1,
                               origin=(16,16))
            self.assertEqual(result["schema"], audit.SCHEMA)
            self.assertFalse(result["neuralInferencePerformed"])
            self.assertFalse(result["modelWeightsUpdated"])
            self.assertFalse(result["productionPromotionAllowed"])
            self.assertEqual(result["textures"][0]["sourceNativeDimensions"], [96,64])
            self.assertEqual(result["textures"][0]["roiOriginNative"], [16,16])
            self.assertTrue((root/"audit/report.json").is_file())
            self.assertIn("material", [x["semantic"] for x in result["textures"][0]["channels"]])

    def test_rejects_outside_roi_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            image=np.full((32,32,4),96,np.uint8)
            cv2.imwrite(str(root/"a.png"),image)
            manifest=root/"materials.tsv"
            manifest.write_text("albedo\na.png\n",encoding="utf-8")
            before=(root/"a.png").read_bytes()
            with self.assertRaisesRegex(ValueError,"outside"):
                audit.run(manifest,root/"audit",origin=(999,0),crop_size=16,max_textures=1)
            self.assertEqual((root/"a.png").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
