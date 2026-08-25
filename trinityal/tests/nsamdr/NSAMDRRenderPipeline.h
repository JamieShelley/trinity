#pragma once

#include "NSAMDRCameraController.h"

namespace nsamdr
{
class RenderPipeline final
{
public:
    explicit RenderPipeline(CameraController& cameraController);

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
        const FinalCandidateSet& candidates,
        const PreviewState& state,
        const SceneConstants& baseConstants);

private:
    bool UploadSceneConstants(ID3D11DeviceContext* context, ID3D11Buffer* constantBuffer, const SceneConstants& constants);
    SceneConstants BuildViewportSceneConstants(
        const PreviewState& state,
        const SceneConstants& baseConstants,
        uint32_t viewportWidth,
        uint32_t viewportHeight);

    CameraController& m_cameraController;
};
} // namespace nsamdr
