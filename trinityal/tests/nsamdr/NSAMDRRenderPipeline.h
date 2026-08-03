#pragma once

#include "NSAMDRCameraController.h"
#include "NSAMDRStrategyModes.h"

namespace nsamdr
{
class RenderPipeline final
{
public:
    RenderPipeline(CameraController& cameraController, StrategyModes& strategyModes);

    bool UpdateSceneConstants(
        ID3D11DeviceContext* context,
        ID3D11Buffer* constantBuffer,
        const PreviewResources& resources,
        const PreviewState& state,
        float elapsedSeconds,
        SceneConstants& outputConstants);
    void RenderShip(
        ID3D11DeviceContext* context,
        const PreviewResources& resources,
        const StrategyCandidateSet& candidates,
        const PreviewState& state,
        const SceneConstants& baseConstants);

private:
    bool UploadSceneConstants(ID3D11DeviceContext* context, ID3D11Buffer* constantBuffer, const SceneConstants& constants);
    SceneConstants BuildViewportSceneConstants( const PreviewState& state, const SceneConstants& baseConstants, uint32_t viewportWidth, uint32_t viewportHeight, int mode);

    CameraController& m_cameraController;
    StrategyModes& m_strategyModes;
};
} // namespace nsamdr
