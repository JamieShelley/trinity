#pragma once

#include "NSAMDRMode3Pipeline.h"
#include "NSAMDRPreviewRenderer.h"
#include "NSAMDRSceneController.h"

namespace nsamdr
{
class PreviewProcessing final
{
public:
    PreviewProcessing(
        PreviewRenderer& renderer,
        AssetProcessor& assetProcessor,
        StrategyModes& strategyModes,
        NSAMDRPipeline& pipeline,
        SceneController& sceneController);

    bool LoadCandidates(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        StrategyCandidateSet& candidates);
    bool InitializeState(
        PreviewState& state,
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        PreviewResources& resources,
        const ObjMesh& mesh);
    void PrintDiagnostics(
        const ObjMesh& mesh,
        const PreviewResources& resources,
        const StrategyCandidateSet& candidates,
        const std::string& albedoPath,
        const std::string& normalPath,
        const std::string& pgsPath,
        const std::string& environmentPath,
        const std::string& materialManifestPath) const;

private:
    void PrintCandidate(int mode, const CandidateAssetGpu& candidate) const;

    PreviewRenderer& m_renderer;
    AssetProcessor& m_assetProcessor;
    StrategyModes& m_strategyModes;
    NSAMDRPipeline& m_pipeline;
    SceneController& m_sceneController;
};
} // namespace nsamdr
