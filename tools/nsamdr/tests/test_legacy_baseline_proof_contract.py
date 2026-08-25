from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_preview_has_one_shared_high_quality_sampler_only() -> None:
    types = read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    asset = read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
    render = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    assert "baselineTextureSampler" not in types
    assert "emulateLegacyEveBaseline" not in types
    assert "baselineSamplerDescription" not in asset
    assert "samplerDescription.MaxAnisotropy = 16" in asset
    assert "samplerDescription.MipLODBias = 0.0f" in asset
    assert "2x anisotropic" not in asset
    assert "baselineTextureSampler" not in render
    assert "emulateLegacyEveBaseline" not in render


def test_neural_proof_is_two_pane_raw_vs_final() -> None:
    panel = read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
    render = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
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


def test_baseline_normal_uses_gradient_sampling() -> None:
    shader = read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
    assert "SampleNormalXYGrad(uv, ddx(uv), ddy(uv))" in shader
    assert ": SampleNormalXY(uv, 0.0)" not in shader


def test_no_pane_specific_shader_postprocessing_or_identity_tint() -> None:
    types = read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    render = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    shader = read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
    assert "verifyPaneIdentity" not in types
    assert "verifyPaneIdentity" not in render
    assert "ApplyMode3Cleanup" not in shader
    assert "normalVariation" not in shader
    assert "gOptions.w < -0.5" not in shader
    assert "gOptions.w > 0.5" not in shader


def test_real_eve_legacy_pgs_material_semantics_remain() -> None:
    types = read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    asset = read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
    shader = read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
    assert "LegacyPgs = 1" in types
    assert 'lower == "legacy_pgs"' in asset
    assert "Legacy PGS uses R=sub-mask, B=mask" in shader
    assert "shaderFamily == 1" in shader
