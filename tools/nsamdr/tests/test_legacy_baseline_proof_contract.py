from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_distinct_sampler_resources_exist() -> None:
    types = read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    asset = read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
    render = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    assert "baselineTextureSampler" in types
    assert "baselineSamplerDescription.MipLODBias = 1.0f" in asset
    assert "baselineSamplerDescription.MaxAnisotropy = 2" in asset
    assert "useLegacySampler ? baselineTextureSampler : textureSampler" in render
    assert "state.emulateLegacyEveBaseline);" in render
    assert "? baselineTextureSampler" in render


def test_right_candidate_keeps_high_quality_sampler() -> None:
    asset = read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
    render = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    assert "samplerDescription.MaxAnisotropy = 16" in asset
    assert "drawPane(candidatePane, state.mode, selectedAsset, false);" in render
    assert ": textureSampler" in render


def test_baseline_normal_uses_gradient_sampling() -> None:
    shader = read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
    assert "SampleNormalXYGrad(uv, ddx(uv), ddy(uv))" in shader
    assert ": SampleNormalXY(uv, 0.0)" not in shader


def test_ab_isolation_is_visible() -> None:
    panel = read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
    assert "A/B albedo resource isolation: %s" in panel
    assert "differentPath && differentSrv" in panel
    assert "Left SRV:" in panel
    assert "Right SRV:" in panel


def test_identity_tint_is_opt_in() -> None:
    types = read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    render = read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    shader = read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
    assert "bool verifyPaneIdentity = false" in types
    assert "state.verifyPaneIdentity" in render
    assert "gOptions.w < -0.5" in shader
    assert "gOptions.w > 0.5" in shader
