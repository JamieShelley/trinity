#pragma once

#include "NSAMDRStrategyModes.h"

namespace nsamdr
{
enum class NSAMDRPipelineStage
{
    SourceCorpusTraining = 1,
    TileCheckpointExport = 2,
    OverlappingTileInference = 3,
    BakedMaterialGeneration = 4,
    LiveShaderSampling = 5,
    CandidateValidation = 6,
};

struct NSAMDRPipelineContract
{
    const char* analysisEnvironment = "NSAMDR_MODE3_ANALYSIS";
    const char* validationEnvironment = "NSAMDR_MODE3_VALIDATION";
    const char* objEnvironment = "NSAMDR_MODE3_OBJ";
    const char* materialsEnvironment = "NSAMDR_MODE3_MATERIALS";
    const char* schema = "NSAMDR_MODE3_TILE_CONTEXT_V4";
};

class NSAMDRPipeline final
{
public:
    explicit NSAMDRPipeline(StrategyModes& strategyModes);

    const NSAMDRPipelineContract& Contract() const;
    bool IsConfigured(const StrategyCandidateSet& candidates) const;
    bool IsReady(const StrategyCandidateSet& candidates) const;
    const char* Summary() const;

private:
    StrategyModes& m_strategyModes;
};
} // namespace nsamdr
