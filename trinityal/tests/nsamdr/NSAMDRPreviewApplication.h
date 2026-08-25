#pragma once

#include "NSAMDRMeshProcessor.h"
#include "NSAMDRPreviewPanel.h"
#include "NSAMDRPreviewProcessing.h"

namespace nsamdr
{
struct PreviewHost
{
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    HWND window = nullptr;
    std::function<bool(uint32_t width, uint32_t height)> resize;
    std::function<bool()> present;
    std::function<void(const std::function<void()>& frame)> runLoop;
    std::function<void(const std::string& path)> screenshot;
};

class PreviewApplication final
{
public:
    explicit PreviewApplication(PreviewHost host);
    void Run();

private:
    PreviewHost m_host;
    PreviewShaderLibrary m_shaderLibrary;
    CameraController m_cameraController;
    MeshProcessor m_meshProcessor;
    InputController m_inputController;
    AssetProcessor m_assetProcessor;
    SceneController m_sceneController;
    RenderPipeline m_renderPipeline;
    PreviewRenderer m_renderer;
    PreviewProcessing m_processing;
    PreviewPanel m_panel;
};
} // namespace nsamdr
