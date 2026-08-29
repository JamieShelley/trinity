from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestLegacyBaselineProofContract:
    # Purpose: Implement read for TestLegacyBaselineProofContract.
    # Called by: test_baseline_normal_uses_gradient_sampling, test_neural_proof_is_two_pane_raw_vs_final, test_no_pane_specific_shader_postprocessing_or_identity_tint, test_preview_has_one_shared_high_quality_sampler_only, test_real_eve_legacy_pgs_material_semantics_remain
    # Calls: No same-class helper methods.
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    # Purpose: Implement test preview has one shared high quality sampler only for TestLegacyBaselineProofContract.
    # Called by: External callers and the owning workflow.
    # Calls: read
    def test_preview_has_one_shared_high_quality_sampler_only(self) -> None:
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        asset = self.read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        assert "baselineTextureSampler" not in types
        assert "emulateLegacyEveBaseline" not in types
        assert "baselineSamplerDescription" not in asset
        assert "samplerDescription.MaxAnisotropy = 16" in asset
        assert "samplerDescription.MipLODBias = 0.0f" in asset
        assert "2x anisotropic" not in asset
        assert "baselineTextureSampler" not in render
        assert "emulateLegacyEveBaseline" not in render

    # Purpose: Implement test neural proof is two pane raw vs final for TestLegacyBaselineProofContract.
    # Called by: External callers and the owning workflow.
    # Calls: read
    def test_neural_proof_is_two_pane_raw_vs_final(self) -> None:
        panel = self.read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        assert "Fixed production comparison: A RAW SOURCE and B NSAMDR FINAL" in panel
        assert "A RAW SOURCE - 16x AF / LOD 0" in panel
        assert "B NSAMDR FINAL - 16x AF / LOD 0" in panel
        assert "B LEGACY EMULATION" not in panel
        assert "legacy-sampler" not in panel.lower()
        assert "three-pane" not in panel.lower()
        assert "drawPane(candidatePane, finalAsset);" in render
        assert "drawPane(rawControlPane, baselineAsset);" in render
        assert "m_strategyModes" not in render
        assert "paneTextureSampler = textureSampler" in render
        assert "baselineAsset.vertexBuffer" in render
        assert "baselineAsset.indexBuffer" in render
        assert "requireSourceDrawRange" in render
        assert "candidate.vertexBuffer" not in render
        assert "candidate.indexBuffer" not in render

    # Purpose: Implement test baseline normal uses gradient sampling for TestLegacyBaselineProofContract.
    # Called by: External callers and the owning workflow.
    # Calls: read
    def test_baseline_normal_uses_gradient_sampling(self) -> None:
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        assert "SampleNormalXYGrad(uv, ddx(uv), ddy(uv))" in shader
        assert ": SampleNormalXY(uv, 0.0)" not in shader

    # Purpose: Implement test no pane specific shader postprocessing or identity tint for TestLegacyBaselineProofContract.
    # Called by: External callers and the owning workflow.
    # Calls: read
    def test_no_pane_specific_shader_postprocessing_or_identity_tint(self) -> None:
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        assert "verifyPaneIdentity" not in types
        assert "verifyPaneIdentity" not in render
        assert "ApplyMode3Cleanup" not in shader
        assert "normalVariation" not in shader
        assert "gOptions.w < -0.5" not in shader
        assert "gOptions.w > 0.5" not in shader

    # Purpose: Implement test real eve legacy pgs material semantics remain for TestLegacyBaselineProofContract.
    # Called by: External callers and the owning workflow.
    # Calls: read
    def test_real_eve_legacy_pgs_material_semantics_remain(self) -> None:
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        asset = self.read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        assert "LegacyPgs = 1" in types
        assert 'lower == "legacy_pgs"' in asset
        assert "Legacy PGS uses R=sub-mask, B=mask" in shader
        assert "shaderFamily == 1" in shader

_test_legacy_baseline_proof_contract = TestLegacyBaselineProofContract()
read = _test_legacy_baseline_proof_contract.read
