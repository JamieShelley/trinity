#pragma once

#include "NSAMDRStrategyModes.h"

namespace nsamdr
{
class SceneController final
{
public:
    SceneController(InputController& inputController, StrategyModes& strategyModes);
    XMFLOAT3 GetWorldSpaceBoundsCenter(const PreviewState& state, const ObjMesh& mesh);
    void FrameShip(PreviewState& state, const ObjMesh& mesh);
    void ResetView(PreviewState& state, const ObjMesh& mesh);
    void ApplyLightingPreset(PreviewState& state, int preset);
    void ApplyGameLightingPreset(PreviewState& state);
    void ProcessHotkeys(PreviewState& state, HWND hwnd, const ObjMesh& mesh, const PreviewResources& resources);

private:
    InputController& m_inputController;
    StrategyModes& m_strategyModes;
};
} // namespace nsamdr
