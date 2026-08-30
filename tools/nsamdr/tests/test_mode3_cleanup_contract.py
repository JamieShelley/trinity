from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class TestMode3CleanupContract:
    # Purpose: Implement read for TestMode3CleanupContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

_test_mode3_cleanup_contract = TestMode3CleanupContract()
read = _test_mode3_cleanup_contract.read


class FixedABRendererContractTests(unittest.TestCase):
    # Purpose: Implement test final candidate has no live cleanup or mode branch for FixedABRendererContractTests.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_final_candidate_has_no_live_cleanup_or_mode_branch(self):
        shader = read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        pipeline = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        types = read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")

        self.assertNotIn("ApplyMode3Cleanup", shader)
        self.assertNotIn("gCleanup", shader)
        self.assertNotIn("normalVariation", shader)
        self.assertNotIn("mode == 3", shader)
        self.assertNotIn("constants.cleanup", pipeline)
        self.assertNotIn("cleanupMasterStrength", types)
        self.assertIn(
            "gAlbedo.SampleGrad(gTextureSampler, uv, ddx(uv), ddy(uv))",
            shader,
        )

    # Purpose: Implement test production ui is fixed raw vs final for FixedABRendererContractTests.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_production_ui_is_fixed_raw_vs_final(self):
        panel = read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
        scene = read("trinityal/tests/nsamdr/NSAMDRSceneController.cpp")

        self.assertIn("Fixed production comparison: A RAW SOURCE and B NSAMDR FINAL", panel)
        self.assertIn("A RAW SOURCE - 16x AF / LOD 0", panel)
        self.assertIn("B NSAMDR FINAL - 16x AF / LOD 0", panel)
        self.assertNotIn("Strategy mode", panel)
        self.assertNotIn("UV/stretch", panel)
        self.assertNotIn("Offline retraining", panel)
        self.assertNotIn("ProcessNumberHotkeys", scene)
        self.assertNotIn("VK_SPACE", scene)

    # Purpose: Implement test obsolete mode and training controllers are not compiled for FixedABRendererContractTests.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_obsolete_mode_and_training_controllers_are_not_compiled(self):
        cmake = read("scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake")

        self.assertNotIn('"${_NSAMDR_OBJ_DIR}/NSAMDRStrategyModes.cpp"', cmake)
        self.assertNotIn('"${_NSAMDR_OBJ_DIR}/NSAMDRMode3Pipeline.cpp"', cmake)
        self.assertNotIn('"${_NSAMDR_OBJ_DIR}/NSAMDRTrainingController.cpp"', cmake)

    # Purpose: Implement test native candidate availability is provenance gated for FixedABRendererContractTests.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_native_candidate_availability_is_provenance_gated(self):
        processing = read("trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp")

        self.assertIn('GetEnvironmentString("NSAMDR_PROVENANCE_STATUS") != "VERIFIED"', processing)
        self.assertIn('GetEnvironmentString("NSAMDR_PREVIEW_CHECKPOINT_SHA256")', processing)
        self.assertIn('GetEnvironmentString("NSAMDR_FINAL_OBJ")', processing)
        self.assertIn('GetEnvironmentString("NSAMDR_FINAL_MATERIALS")', processing)
        self.assertIn('GetEnvironmentString("NSAMDR_FINAL_ANALYSIS")', processing)
        self.assertIn('GetEnvironmentString("NSAMDR_FINAL_VALIDATION")', processing)
        self.assertNotIn("NSAMDR_MODE3_", processing)
        self.assertIn('authority.find("final")', processing)
        self.assertIn('authority.find("intermediate")', processing)
        self.assertIn('authority.find("paused")', processing)
        self.assertIn("BaselineContainsAlbedo", processing)
        self.assertIn("CandidateContainsAlbedo", processing)
        self.assertIn("candidate.available = false", processing)
        self.assertIn("provenance gate blocked NSAMDR FINAL", processing)


if __name__ == "__main__":
    unittest.main()
