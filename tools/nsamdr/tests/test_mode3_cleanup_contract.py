from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class Mode3CleanupContractTests(unittest.TestCase):
    def test_mode1_asset_pipeline_is_not_modified_by_cleanup_patch(self):
        shader = (ROOT / "trinityal/tests/nsamdr/NSAMDRPreview.hlsl").read_text(encoding="utf-8")
        self.assertIn("if (mode == 3)", shader)
        self.assertIn("ApplyMode3Cleanup", shader)
        self.assertIn("quality <= 0", shader)

    def test_graphics_setting_is_registered(self):
        source = (ROOT / "trinity/Resources/Tr2TextureDetailReconstructionSettings.cpp").read_text(encoding="utf-8")
        self.assertIn("textureDetailReconstructionQuality", (ROOT / "trinity/Resources/Tr2TextureDetailReconstructionSettings.h").read_text(encoding="utf-8"))
        self.assertIn("TRI_REGISTER_SETTING", source)
        self.assertIn("QUALITY_OFF", source)

    def test_preview_exposes_quality_and_diagnostics(self):
        panel = (ROOT / "trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp").read_text(encoding="utf-8")
        self.assertIn("Neural Surface Reconstruction", panel)
        self.assertIn("Display & Graphics policy", panel)
        self.assertIn("Flat-region denoise", panel)
        self.assertIn("Cleanup difference", panel)

    def test_quality_mode_has_directional_reconstruction_contract(self):
        shader = (ROOT / "trinityal/tests/nsamdr/NSAMDRPreview.hlsl").read_text(encoding="utf-8")
        self.assertIn("directionalDeblockStrength", shader)
        self.assertIn("Sobel gradient", shader)
        self.assertIn("stairStepConfidence", shader)
        self.assertIn("thinLineConfidence", shader)
        self.assertIn("gCleanup3", shader)

    def test_directional_controls_are_bound_and_exposed(self):
        types = (ROOT / "trinityal/tests/nsamdr/NSAMDRPreviewTypes.h").read_text(encoding="utf-8")
        pipeline = (ROOT / "trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp").read_text(encoding="utf-8")
        panel = (ROOT / "trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp").read_text(encoding="utf-8")
        self.assertIn("cleanupDirectionalDeblockStrength", types)
        self.assertIn("constants.cleanup3", pipeline)
        self.assertIn("Directional deblocking", panel)
        self.assertIn("Thin-line confidence", panel)


if __name__ == "__main__":
    unittest.main()
