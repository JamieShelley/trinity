#include "StdAfx.h"
#include "NSAMDRPreviewRenderer.h"

namespace nsamdr
{
PreviewRenderer::PreviewRenderer(AssetProcessor& assetProcessor, RenderPipeline& renderPipeline)
    : m_assetProcessor(assetProcessor),
      m_renderPipeline(renderPipeline)
{
}

bool PreviewRenderer::CreateResources(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    IDXGISwapChain* swapChain,
    const ObjMesh& mesh,
    const std::string& albedoPath,
    const std::string& normalPath,
    const std::string& pgsPath,
    const std::vector<std::string>& environmentPaths,
    const std::string& materialManifestPath,
    PreviewResources& resources)
{
    return m_assetProcessor.CreatePreviewResources(
        device,
        context,
        swapChain,
        mesh,
        albedoPath,
        normalPath,
        pgsPath,
        environmentPaths,
        materialManifestPath,
        resources);
}

bool PreviewRenderer::RecreateTargets(
    ID3D11Device* device,
    IDXGISwapChain* swapChain,
    PreviewResources& resources)
{
    return m_assetProcessor.CreatePreviewTargets(device, swapChain, resources);
}

bool PreviewRenderer::EnsureEnvironment(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    PreviewResources& resources,
    PreviewState& state)
{
    return m_assetProcessor.EnsureSelectedEnvironmentLoaded(device, context, resources, state);
}

bool PreviewRenderer::UpdateSceneConstants(
    ID3D11DeviceContext* context,
    ID3D11Buffer* constantBuffer,
    const PreviewResources& resources,
    const PreviewState& state,
    float elapsedSeconds,
    SceneConstants& constants)
{
    return m_renderPipeline.UpdateSceneConstants(
        context,
        constantBuffer,
        resources,
        state,
        elapsedSeconds,
        constants);
}

void PreviewRenderer::Render(
    ID3D11DeviceContext* context,
    const PreviewResources& resources,
    const FinalCandidateSet& candidates,
    const PreviewState& state,
    const SceneConstants& constants)
{
    m_renderPipeline.RenderShip(context, resources, candidates, state, constants);
}
} // namespace nsamdr
