#include "StdAfx.h"
#include "NSAMDRPreviewApplication.h"
#include "NSAMDRPreviewUtilities.h"
#include "NSAMDRWindowIcon.h"

namespace nsamdr
{
PreviewApplication::PreviewApplication(PreviewHost host)
    : m_host(std::move(host)),
      m_inputController(m_cameraController),
      m_strategyModes(m_inputController),
      m_pipeline(m_strategyModes),
      m_assetProcessor(m_shaderLibrary, m_meshProcessor),
      m_sceneController(m_inputController, m_strategyModes),
      m_renderPipeline(m_cameraController, m_strategyModes),
      m_renderer(m_assetProcessor, m_renderPipeline),
      m_processing(m_renderer, m_assetProcessor, m_strategyModes, m_pipeline, m_sceneController),
      m_panel(m_strategyModes, m_pipeline, m_trainingController, m_sceneController)
{
}

void PreviewApplication::Run()
{
    ASSERT_NE(m_host.device, nullptr);
    ASSERT_NE(m_host.context, nullptr);
    ASSERT_NE(m_host.swapChain, nullptr);
    ASSERT_NE(m_host.window, nullptr);
    ASSERT_TRUE(static_cast<bool>(m_host.resize));
    ASSERT_TRUE(static_cast<bool>(m_host.present));
    ASSERT_TRUE(static_cast<bool>(m_host.runLoop));
    ASSERT_TRUE(static_cast<bool>(m_host.screenshot));

    const std::string objPath = GetEnvironmentString("NSAMDR_OBJ");
    ASSERT_FALSE(objPath.empty())
        << "NSAMDR_OBJ is not set. Launch through scripts\\build\\run_nsamdr_obj_preview_dx11.bat <ship.obj|ship.gr2> [albedo.png].";

    ObjMesh mesh;
    std::string meshError;
    ASSERT_TRUE(m_meshProcessor.LoadObjMesh(objPath, mesh, meshError)) << meshError;

    const std::string albedoPath = GetEnvironmentString("NSAMDR_ALBEDO");
    const std::string normalPath = GetEnvironmentString("NSAMDR_NORMAL");
    const std::string pgsPath = GetEnvironmentString("NSAMDR_PGS");
    const std::vector<std::string> environmentPaths = GetEnvironmentPaths();
    const std::string environmentPath = environmentPaths.empty() ? std::string() : environmentPaths.front();
    const std::string materialManifestPath = GetEnvironmentString("NSAMDR_MATERIALS");

    SetWindowTextW(
        m_host.window,
        L"NSAMDR \u2014 Neural Stretch-Aware Material Detail Reconstruction | Real EVE Ship Preview");
    ASSERT_TRUE(m_host.resize(1440U, 900U));
    WindowIcon windowIcon;
    std::string windowIconError;
    ASSERT_TRUE(windowIcon.Apply(m_host.window, windowIconError)) << windowIconError;

    PreviewState state;
    ASSERT_TRUE(m_inputController.Attach(m_host.window, state));

    PreviewResources resources;
    ASSERT_TRUE(m_renderer.CreateResources(
        m_host.device,
        m_host.context,
        m_host.swapChain,
        mesh,
        albedoPath,
        normalPath,
        pgsPath,
        environmentPaths,
        materialManifestPath,
        resources));

    StrategyCandidateSet candidates;
    ASSERT_TRUE(m_processing.LoadCandidates(m_host.device, m_host.context, candidates));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ASSERT_TRUE(ImGui_ImplWin32_Init(m_host.window));
    ASSERT_TRUE(ImGui_ImplDX11_Init(m_host.device, m_host.context));

    ASSERT_TRUE(m_processing.InitializeState(
        state,
        m_host.device,
        m_host.context,
        resources,
        mesh));
    m_trainingController.InitializeFromEnvironment(state.neuralTraining);
    if (m_pipeline.IsConfigured(candidates) && m_pipeline.IsReady(candidates))
    {
        state.neuralTraining.status =
            "Mode 3 candidate loaded. Its analysis identifies whether trained V4 tile-context inference or the deterministic bootstrap was baked.";
    }

    ShipCatalog shipCatalog;
    LoadShipCatalog(
        GetEnvironmentString("NSAMDR_EVE_CATALOG"),
        GetEnvironmentString("NSAMDR_EVE_QUERY"),
        shipCatalog);

    m_processing.PrintDiagnostics(
        mesh,
        resources,
        candidates,
        albedoPath,
        normalPath,
        pgsPath,
        environmentPath,
        materialManifestPath);

    const auto startTime = std::chrono::steady_clock::now();
    auto previousFrame = startTime;

    auto frame = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - previousFrame).count();
        const float elapsedSeconds = std::chrono::duration<float>(now - startTime).count();
        previousFrame = now;

        uint32_t resizeWidth = 0U;
        uint32_t resizeHeight = 0U;
        if (m_inputController.ConsumePendingResize(resizeWidth, resizeHeight) &&
            (resizeWidth != resources.width || resizeHeight != resources.height))
        {
            m_host.context->OMSetRenderTargets(0, nullptr, nullptr);
            resources.renderTargetView.Reset();
            resources.depthStencilView.Reset();
            resources.depthTexture.Reset();
            ASSERT_TRUE(m_host.resize(resizeWidth, resizeHeight));
            ASSERT_TRUE(m_renderer.RecreateTargets(m_host.device, m_host.swapChain, resources));
        }

        m_inputController.RefreshFocus();

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        if (m_inputController.IsFocused() &&
            !ImGui::GetIO().WantCaptureMouse &&
            ImGui::IsMouseDoubleClicked(0))
        {
            state.focusMouseX = static_cast<int>(ImGui::GetIO().MousePos.x);
            state.focusMouseY = static_cast<int>(ImGui::GetIO().MousePos.y);
            state.requestFocus = true;
        }

        m_sceneController.ProcessHotkeys(state, m_host.window, mesh, resources);
        if (state.autoOrbit) state.orbitYaw += deltaSeconds * state.orbitSpeed;
        if (state.requestFocus)
        {
            state.requestFocus = false;
            m_cameraController.FocusCameraAtScreenPoint(
                state,
                mesh,
                resources.width,
                resources.height,
                state.focusMouseX,
                state.focusMouseY);
        }
        m_cameraController.ApplyZoomRequest(state, mesh, resources.width, resources.height);

        m_panel.Draw(
            state,
            shipCatalog,
            m_host.window,
            mesh,
            resources,
            candidates,
            albedoPath,
            normalPath,
            pgsPath);

        ASSERT_TRUE(m_renderer.EnsureEnvironment(
            m_host.device,
            m_host.context,
            resources,
            state));

        SceneConstants sceneConstants{};
        ASSERT_TRUE(m_renderer.UpdateSceneConstants(
            m_host.context,
            resources.constantBuffer.Get(),
            resources,
            state,
            elapsedSeconds,
            sceneConstants));
        m_renderer.Render(m_host.context, resources, candidates, state, sceneConstants);
        m_panel.DrawSplitCompareOverlay(state, resources, candidates);

        ImGui::Render();
        m_host.context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        if (state.requestScreenshot)
        {
            state.requestScreenshot = false;
            const std::string screenshotPath = m_panel.BuildScreenshotPath(state);
            m_host.screenshot(screenshotPath);
            std::printf("Saved NSAMDR screenshot: %s\n", screenshotPath.c_str());
        }

        ASSERT_TRUE(m_host.present());
    };

    m_host.runLoop(frame);

    m_inputController.Detach();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    m_host.context->ClearState();
}
} // namespace nsamdr
