#pragma once

#include "NSAMDRSceneController.h"

namespace nsamdr
{
class PreviewPanel final
{
public:
    explicit PreviewPanel(SceneController& sceneController);

    void Draw(
        PreviewState& state,
        ShipCatalog& catalog,
        HWND hwnd,
        const ObjMesh& mesh,
        const PreviewResources& resources,
        const FinalCandidateSet& candidates,
        const std::string& albedoPath,
        const std::string& normalPath,
        const std::string& pgsPath);
    void DrawSplitCompareOverlay(
        const PreviewState& state,
        const PreviewResources& resources,
        const FinalCandidateSet& candidates);
    std::string BuildScreenshotPath() const;

private:
    void DrawShipSelector(ShipCatalog& catalog, HWND hwnd);

    SceneController& m_sceneController;
};
} // namespace nsamdr
