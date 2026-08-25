#pragma once

#include "NSAMDRAssetProcessor.h"
#include "NSAMDRRenderPipeline.h"

namespace nsamdr
{
class PreviewRenderer final
{
public:
    PreviewRenderer(AssetProcessor& assetProcessor, RenderPipeline& renderPipeline);

    bool CreateResources(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        IDXGISwapChain* swapChain,
        const ObjMesh& mesh,
        const std::string& albedoPath,
        const std::string& normalPath,
        const std::string& pgsPath,
        const std::vector<std::string>& environmentPaths,
        const std::string& materialManifestPath,
        PreviewResources& resources);
    bool RecreateTargets(ID3D11Device* device, IDXGISwapChain* swapChain, PreviewResources& resources);
    bool EnsureEnvironment(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        PreviewResources& resources,
        PreviewState& state);
    bool UpdateSceneConstants(
        ID3D11DeviceContext* context,
        ID3D11Buffer* constantBuffer,
        const PreviewResources& resources,
        const PreviewState& state,
        float elapsedSeconds,
        SceneConstants& constants);
    void Render(
        ID3D11DeviceContext* context,
        const PreviewResources& resources,
        const FinalCandidateSet& candidates,
        const PreviewState& state,
        const SceneConstants& constants);

private:
    AssetProcessor& m_assetProcessor;
    RenderPipeline& m_renderPipeline;
};
} // namespace nsamdr
