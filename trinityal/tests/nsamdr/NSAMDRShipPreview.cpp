// Copyright © 2026
// Granny-free NSAMDR visual test using real ship geometry converted to Wavefront OBJ.

#include "StdAfx.h"
#include "WithValidRenderContextFixture.h"
#include "RenderWindow.h"

#if defined(_WIN32) && TRINITY_PLATFORM == TRINITY_DIRECTX11

#include "NSAMDRPreviewApplication.h"

namespace
{
struct NSAMDRRendering : public WithValidRenderContext
{
};
} // namespace

TEST_F(NSAMDRRendering, RealObjShipPreview)
{
    ENSURE_GPU_OR_SKIP

    ASSERT_NE(renderContext, nullptr);
    ASSERT_TRUE(renderContext->IsValid());
    ASSERT_TRUE(renderContext->m_d3dDevice11);
    ASSERT_TRUE(renderContext->m_context);
    ASSERT_TRUE(renderContext->m_swapChain);

    nsamdr::PreviewHost host;
    host.device = renderContext->m_d3dDevice11;
    host.context = renderContext->m_context;
    host.swapChain = renderContext->m_swapChain;
    host.window = static_cast<HWND>(GetWindowHandle());
    host.resize = [this](uint32_t width, uint32_t height) {
        GetWindow()->Resize(width, height);
        presentParameters.mode.width = width;
        presentParameters.mode.height = height;
        return SUCCEEDED(renderContext->SetPresentParameters(
            Tr2VideoAdapterInfo::DEFAULT_ADAPTER,
            presentParameters));
    };
    host.present = [this]() {
        return SUCCEEDED(renderContext->Present());
    };
    host.runLoop = [this](const std::function<void()>& frame) {
        RunLoop(frame);
    };
    host.screenshot = [this](const std::string& path) {
        MakeScreenShot(path.c_str());
    };

    nsamdr::PreviewApplication application(std::move(host));
    application.Run();
}

#endif // _WIN32 && TRINITY_DIRECTX11
