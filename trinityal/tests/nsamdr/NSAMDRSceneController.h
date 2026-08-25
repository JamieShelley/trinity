#pragma once

#include "NSAMDRInputController.h"

namespace nsamdr
{
class SceneController final
{
public:
    explicit SceneController(InputController& inputController);
    XMFLOAT3 GetWorldSpaceBoundsCenter(const PreviewState& state, const ObjMesh& mesh);
    void FrameShip(PreviewState& state, const ObjMesh& mesh);
    void ResetView(PreviewState& state, const ObjMesh& mesh);
    void ApplyLightingPreset(PreviewState& state, int preset);
    void ApplyGameLightingPreset(PreviewState& state);
    void ProcessHotkeys(PreviewState& state, HWND hwnd, const ObjMesh& mesh, const PreviewResources& resources);

private:
    InputController& m_inputController;
};
} // namespace nsamdr
