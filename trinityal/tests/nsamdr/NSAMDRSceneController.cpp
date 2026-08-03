#include "StdAfx.h"
#include "NSAMDRSceneController.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
SceneController::SceneController(InputController& inputController, StrategyModes& strategyModes)
    : m_inputController(inputController),
      m_strategyModes(strategyModes)
{
}

XMFLOAT3 SceneController::GetWorldSpaceBoundsCenter(const PreviewState& state, const ObjMesh& mesh)
{
    const XMMATRIX world =
        DirectX::XMMatrixRotationX(state.modelPitch) *
        DirectX::XMMatrixRotationY(state.modelYaw) *
        DirectX::XMMatrixRotationZ(state.modelRoll);
    XMFLOAT3 centre{};
    DirectX::XMStoreFloat3(
        &centre,
        DirectX::XMVector3TransformCoord(DirectX::XMLoadFloat3(&mesh.boundsCenter), world));
    return centre;
}

void SceneController::FrameShip(PreviewState& state, const ObjMesh& mesh)
{
    const XMFLOAT3 centre = GetWorldSpaceBoundsCenter(state, mesh);
    state.targetX = centre.x;
    state.targetY = centre.y;
    state.targetZ = centre.z;
    const float halfFov = DirectX::XMConvertToRadians(48.0f) * 0.5f;
    state.cameraDistance = std::max(mesh.boundsRadius / std::max(std::tan(halfFov), 0.1f) * 1.28f, 0.05f);
    state.nearClip = std::max(0.0005f, mesh.boundsRadius * 0.0002f);
    state.farClip = std::max(300.0f, state.cameraDistance + mesh.boundsRadius * 25.0f);
}

void SceneController::ResetView(PreviewState& state, const ObjMesh& mesh)
{
    state.orbitYaw = -0.55f;
    state.orbitPitch = 0.22f;
    FrameShip(state, mesh);
}

void SceneController::ApplyLightingPreset(PreviewState& state, int preset)
{
    state.lightingPreset = std::max(0, std::min(preset, 3));
    state.keyYaw = -0.65f;
    state.keyPitch = -0.55f;
    state.normalMapStrength = 0.90f;
    switch (state.lightingPreset)
    {
    case 1: // Studio
        state.keyIntensity = 2.00f;
        state.fillIntensity = 0.90f;
        state.rimIntensity = 0.62f;
        state.ambient = 0.58f;
        state.exposure = 1.28f;
        state.specularStrength = 0.78f;
        state.roughnessBias = 0.02f;
        state.environmentIntensity = 0.42f;
        state.backgroundIntensity = 0.28f;
        state.reflectionStrength = 0.58f;
        break;
    case 2: // Harsh inspection
        state.keyIntensity = 2.55f;
        state.fillIntensity = 0.30f;
        state.rimIntensity = 1.35f;
        state.ambient = 0.34f;
        state.exposure = 1.42f;
        state.specularStrength = 1.05f;
        state.roughnessBias = -0.12f;
        state.environmentIntensity = 0.48f;
        state.backgroundIntensity = 0.40f;
        state.reflectionStrength = 0.82f;
        break;
    case 3: // Dark silhouette
        state.keyIntensity = 0.48f;
        state.fillIntensity = 0.10f;
        state.rimIntensity = 2.05f;
        state.ambient = 0.10f;
        state.exposure = 0.95f;
        state.specularStrength = 0.62f;
        state.roughnessBias = 0.08f;
        state.environmentIntensity = 0.28f;
        state.backgroundIntensity = 0.16f;
        state.reflectionStrength = 0.38f;
        break;
    default: // Game-like, deliberately brighter than the previous test rig.
        state.keyIntensity = 1.90f;
        state.fillIntensity = 0.68f;
        state.rimIntensity = 1.05f;
        state.ambient = 0.62f;
        state.exposure = 1.42f;
        state.specularStrength = 0.92f;
        state.roughnessBias = -0.03f;
        state.environmentIntensity = 1.00f;
        state.backgroundIntensity = 0.82f;
        state.reflectionStrength = 1.00f;
        break;
    }
}

void SceneController::ApplyGameLightingPreset(PreviewState& state)
{
    ApplyLightingPreset(state, 0);
}

void SceneController::ProcessHotkeys(PreviewState& state, HWND hwnd, const ObjMesh& mesh, const PreviewResources& resources)
{
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard)
    {
        m_strategyModes.ProcessNumberHotkeys(state);
        if (m_inputController.KeyPressed(VK_SPACE) && state.mode != static_cast<int>(StrategyMode::OriginalBaseline))
        {
            state.splitCompare = !state.splitCompare;
        }
        if (m_inputController.KeyPressed('R')) ResetView(state, mesh);
        if (m_inputController.KeyPressed(VK_HOME) || m_inputController.KeyPressed('F')) FrameShip(state, mesh);
        if (m_inputController.KeyPressed('V')) state.flipV = !state.flipV;
        if (!resources.environments.empty() && m_inputController.KeyPressed(VK_OEM_4))
        {
            state.environmentIndex = state.environmentIndex == 0U
                ? static_cast<uint32_t>(resources.environments.size() - 1U)
                : state.environmentIndex - 1U;
        }
        if (!resources.environments.empty() && m_inputController.KeyPressed(VK_OEM_6))
        {
            state.environmentIndex = (state.environmentIndex + 1U) % static_cast<uint32_t>(resources.environments.size());
        }
    }

    if (m_inputController.KeyPressed(VK_F9)) state.requestScreenshot = true;
    if (m_inputController.KeyPressed(VK_ESCAPE)) PostMessage(hwnd, WM_CLOSE, 0, 0);
}

} // namespace nsamdr
