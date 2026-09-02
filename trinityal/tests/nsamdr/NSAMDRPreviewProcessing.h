#pragma once

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
        SceneController& sceneController);

    bool LoadCandidates(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        const PreviewResources& resources,
        const std::string& rawAlbedoPath,
        FinalCandidateSet& candidates);
    bool RefreshLiveCandidate(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        const PreviewResources& resources,
        const std::string& rawAlbedoPath,
        FinalCandidateSet& candidates);
    bool InitializeState(
        PreviewState& state,
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        PreviewResources& resources,
        const ObjMesh& mesh);
    void PrintDiagnostics(
        const ObjMesh& mesh,
        const PreviewResources& resources,
        const FinalCandidateSet& candidates,
        const std::string& albedoPath,
        const std::string& normalPath,
        const std::string& pgsPath,
        const std::string& environmentPath,
        const std::string& materialManifestPath) const;

private:
    void PrintCandidate(const CandidateAssetGpu& candidate) const;

    PreviewRenderer& m_renderer;
    AssetProcessor& m_assetProcessor;
    SceneController& m_sceneController;
    std::string m_liveCandidateToken;
};
} // namespace nsamdr
