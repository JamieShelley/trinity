#pragma once

#include "NSAMDRMode3Pipeline.h"
#include "NSAMDRSceneController.h"
#include "NSAMDRTrainingController.h"

namespace nsamdr
{
class PreviewPanel final
{
public:
    PreviewPanel(
        StrategyModes& strategyModes,
        NSAMDRPipeline& pipeline,
        NSAMDRTrainingController& trainingController,
        SceneController& sceneController);

    void Draw(
        PreviewState& state,
        ShipCatalog& catalog,
        HWND hwnd,
        const ObjMesh& mesh,
        const PreviewResources& resources,
        const StrategyCandidateSet& candidates,
        const std::string& albedoPath,
        const std::string& normalPath,
        const std::string& pgsPath);
    void DrawSplitCompareOverlay(
        const PreviewState& state,
        const PreviewResources& resources,
        const StrategyCandidateSet& candidates);
    std::string BuildScreenshotPath(const PreviewState& state) const;

private:
    void DrawShipSelector(ShipCatalog& catalog, HWND hwnd);

    StrategyModes& m_strategyModes;
    NSAMDRPipeline& m_pipeline;
    NSAMDRTrainingController& m_trainingController;
    SceneController& m_sceneController;
};
} // namespace nsamdr
