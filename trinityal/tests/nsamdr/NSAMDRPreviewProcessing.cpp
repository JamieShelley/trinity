#include "StdAfx.h"
#include "NSAMDRPreviewProcessing.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
PreviewProcessing::PreviewProcessing(
    PreviewRenderer& renderer,
    AssetProcessor& assetProcessor,
    StrategyModes& strategyModes,
    NSAMDRPipeline& pipeline,
    SceneController& sceneController)
    : m_renderer(renderer),
      m_assetProcessor(assetProcessor),
      m_strategyModes(strategyModes),
      m_pipeline(pipeline),
      m_sceneController(sceneController)
{
}

bool PreviewProcessing::LoadCandidates(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    StrategyCandidateSet& candidates)
{
    for (const StrategyDescriptor& descriptor : m_strategyModes.Registry())
    {
        if (!descriptor.physicalCandidate) continue;
        CandidateAssetGpu& candidate = candidates.At(static_cast<int>(descriptor.mode));
        if (!m_assetProcessor.LoadCandidateAsset(
                device,
                context,
                descriptor.candidateLabel,
                GetEnvironmentString(descriptor.objEnvironment),
                GetEnvironmentString(descriptor.materialsEnvironment),
                candidate))
        {
            return false;
        }
    }
    return true;
}

bool PreviewProcessing::InitializeState(
    PreviewState& state,
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    PreviewResources& resources,
    const ObjMesh& mesh)
{
    const bool hasAreaAlbedo = std::any_of(
        resources.areaMaterials.begin(),
        resources.areaMaterials.end(),
        [](const AreaMaterialGpu& material) { return material.hasAlbedo; });
    const bool hasAreaNormal = std::any_of(
        resources.areaMaterials.begin(),
        resources.areaMaterials.end(),
        [](const AreaMaterialGpu& material) { return material.hasNormal; });
    const bool hasAreaPgs = std::any_of(
        resources.areaMaterials.begin(),
        resources.areaMaterials.end(),
        [](const AreaMaterialGpu& material) { return material.hasPgs; });

    state.useTexture = resources.hasExternalAlbedo || hasAreaAlbedo || !resources.areaMaterials.empty();
    state.useNormalMap = resources.hasNormalMap || hasAreaNormal;
    state.usePgsMap = resources.hasPgsMap || hasAreaPgs;
    state.useEnvironment = resources.hasEnvironment;
    if (!m_renderer.EnsureEnvironment(device, context, resources, state)) return false;
    m_sceneController.ApplyGameLightingPreset(state);
    m_sceneController.ResetView(state, mesh);
    return true;
}

void PreviewProcessing::PrintDiagnostics(
    const ObjMesh& mesh,
    const PreviewResources& resources,
    const StrategyCandidateSet& candidates,
    const std::string& albedoPath,
    const std::string& normalPath,
    const std::string& pgsPath,
    const std::string& environmentPath,
    const std::string& materialManifestPath) const
{
    std::printf("NSAMDR OBJ loaded: %s\n", mesh.path.c_str());
    std::printf("  triangles=%u vertices=%zu sourcePositions=%u sourceUVs=%u sourceNormals=%u\n",
        mesh.triangleCount,
        mesh.vertices.size(),
        mesh.sourcePositionCount,
        mesh.sourceTexcoordCount,
        mesh.sourceNormalCount);
    std::printf("  uvStretchAverage=%.4f uvStretchMaximum=%.4f degenerateUvTriangles=%u calibrationRawP50=%.4f calibrationRawP95=%.4f\n",
        mesh.averageStretch,
        mesh.maximumStretch,
        mesh.degenerateUvTriangles,
        mesh.stretchCalibrationLow,
        mesh.stretchCalibrationHigh);

    if (resources.hasExternalAlbedo)
        std::printf("  albedo=%s (%ux%u)\n", albedoPath.c_str(), resources.textureWidth, resources.textureHeight);
    else
        std::printf("  albedo=<neutral fallback>\n");

    if (resources.hasNormalMap)
        std::printf("  normal=%s (%ux%u)\n", normalPath.c_str(), resources.normalWidth, resources.normalHeight);
    if (resources.hasPgsMap)
        std::printf("  pgs=%s (%ux%u)\n", pgsPath.c_str(), resources.pgsWidth, resources.pgsHeight);

    if (!resources.areaMaterials.empty())
        std::printf("  sofMaterials=%s (draws=%zu, groups=%zu)\n", materialManifestPath.c_str(), resources.areaMaterials.size(), mesh.drawRanges.size());
    else
        std::printf("  sofMaterials=<legacy global fallback> (groups=%zu)\n", mesh.drawRanges.size());

    std::printf(
        "  publicModes=1-3 splitCompare=available sharedCamera=true nsamdrConfigured=%s\n",
        m_pipeline.IsConfigured(candidates) ? "true" : "false");
    for (const StrategyDescriptor& descriptor : m_strategyModes.Registry())
    {
        if (!descriptor.physicalCandidate) continue;
        PrintCandidate(static_cast<int>(descriptor.mode), candidates.At(static_cast<int>(descriptor.mode)));
    }

    if (resources.hasEnvironment)
        std::printf("  environment=%s (%ux%u)\n", environmentPath.c_str(), resources.environmentWidth, resources.environmentHeight);
    else
        std::printf("  environment=<procedural fallback>\n");
}


void PreviewProcessing::PrintCandidate(int mode, const CandidateAssetGpu& candidate) const
{
    if (candidate.available)
    {
        std::printf("  publicMode%dCandidate=loaded texture=%ux%u triangles=%u draws=%zu obj=%s\n",
            mode,
            candidate.maximumTextureWidth,
            candidate.maximumTextureHeight,
            candidate.mesh.triangleCount,
            candidate.areaMaterials.size(),
            candidate.objPath.c_str());
    }
    else
    {
        std::printf("  publicMode%dCandidate=unavailable reason=%s\n", mode, candidate.status.c_str());
    }
}

} // namespace nsamdr
