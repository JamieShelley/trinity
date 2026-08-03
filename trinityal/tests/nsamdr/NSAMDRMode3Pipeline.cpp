#include "StdAfx.h"
#include "NSAMDRMode3Pipeline.h"

namespace nsamdr
{
NSAMDRPipeline::NSAMDRPipeline(StrategyModes& strategyModes)
    : m_strategyModes(strategyModes)
{
}

const NSAMDRPipelineContract& NSAMDRPipeline::Contract() const
{
    static const NSAMDRPipelineContract contract{};
    return contract;
}

bool NSAMDRPipeline::IsConfigured(const StrategyCandidateSet& candidates) const
{
    const CandidateAssetGpu* candidate = m_strategyModes.CandidateForMode(
        candidates,
        static_cast<int>(StrategyMode::NeuralReconstruction));
    return candidate && !candidate->objPath.empty() && !candidate->materialManifestPath.empty();
}

bool NSAMDRPipeline::IsReady(const StrategyCandidateSet& candidates) const
{
    const CandidateAssetGpu* candidate = m_strategyModes.CandidateForMode(
        candidates,
        static_cast<int>(StrategyMode::NeuralReconstruction));
    return candidate && candidate->available;
}

const char* NSAMDRPipeline::Summary() const
{
    return "Mode 3 is the active NSAMDR V4 path: deterministic 4K preparation, a "
           "trained fully convolutional tile-context network with a 125-pixel receptive "
           "field, overlapping offline inference, baked reconstructed material textures, "
           "and live shader sampling. Before the first V4 checkpoint, the same Mode 3 slot uses a deterministic prepared-4K bootstrap rather than disappearing. Mode 1 remains the untouched source comparison.";
}
} // namespace nsamdr
