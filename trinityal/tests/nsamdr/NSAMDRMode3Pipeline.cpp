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
    return "Mode 3 is the NSAMDR V9.8.2 metric-SDF geometry-convergence SDF path: a coordinate-conditioned GeometryNet predicts a continuous 4x signed-distance field, "
           "edge/tangent/hardness and an explicit benefit gate. The training/audit ladder isolates renderer, SDF and gate failures before a candidate is trusted. "
           "A deterministic two-sided BoundaryRenderer then applies the same zero-set to aligned albedo, normal and material semantics without giving GeometryNet RGB painting authority. "
           "FP16 overlapping CUDA inference bakes reconstructed resources before live shader sampling. Mode 3 fails closed when the selected checkpoint, provenance, or CUDA backend is unavailable; "
           "Mode 1 remains the SHA-256-verified untouched source comparison.";
}
} // namespace nsamdr
