from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "eve_asset_test.py"
spec = importlib.util.spec_from_file_location("eve_asset_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SemanticLayoutTests(unittest.TestCase):
    def test_v5_packed_layout(self) -> None:
        textures = {
            "AlbedoMap": "res:/ship/cb1_t1_ar.dds",
            "NormalMap": "res:/ship/cb1_t1_no.dds",
            "PmdgMap": "res:/ship/cb1_t1_pmdg.dds",
        }
        layout = module._semantic_texture_layout({"shader": "quadV5"}, textures)
        self.assertEqual(layout["shaderFamily"], "v5_packed")
        self.assertTrue(layout["roughnessMap"].endswith("_ar.dds"))
        self.assertTrue(layout["ao"].endswith("_no.dds"))
        self.assertTrue(layout["glow"].endswith("_pmdg.dds"))
        self.assertEqual(layout["channels"], {
            "normalX": 3, "normalY": 1, "roughness": 3, "material": 1,
            "ao": 2, "paint": 0, "dirt": 2, "glow": 3,
        })
        self.assertTrue(layout["semanticComplete"])

    def test_v5_separate_layout(self) -> None:
        textures = {
            "AlbedoMap": "res:/ship/albedo.dds",
            "NormalMap": "res:/ship/normal.dds",
            "MaterialMap": "res:/ship/material.dds",
            "RoughnessMap": "res:/ship/roughness.dds",
            "PaintMaskMap": "res:/ship/paint.dds",
            "DirtMap": "res:/ship/dirt.dds",
            "GlowMap": "res:/ship/glow.dds",
            "AoMap": "res:/ship/ao.dds",
        }
        layout = module._semantic_texture_layout({"shader": "quadV5"}, textures)
        self.assertEqual(layout["shaderFamily"], "v5_separate")
        self.assertEqual(layout["channels"]["material"], 0)
        self.assertTrue(layout["semanticComplete"])

    def test_legacy_pgs_layout(self) -> None:
        textures = {
            "DiffuseMap": "res:/ship/cb1_t1_d.dds",
            "NormalMap": "res:/ship/cb1_t1_n.dds",
            "PgsMap": "res:/ship/cb1_t1_pgs.dds",
        }
        layout = module._semantic_texture_layout({"shader": "quad"}, textures)
        self.assertEqual(layout["shaderFamily"], "legacy_pgs")
        self.assertEqual(layout["channels"]["glow"], 3)
        self.assertTrue(layout["semanticComplete"])

    def test_unknown_layout_blocks_baseline(self) -> None:
        layout = module._semantic_texture_layout({"shader": ""}, {"DiffuseMap": "res:/ship/d.dds"})
        self.assertEqual(layout["shaderFamily"], "unknown")
        self.assertFalse(layout["semanticComplete"])


if __name__ == "__main__":
    unittest.main()
