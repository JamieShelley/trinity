#include "StdAfx.h"
#include "NSAMDRStrategyModes.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
StrategyModes::StrategyModes(InputController& inputController)
    : m_inputController(inputController)
{
}

const std::array<StrategyDescriptor, StrategyModes::kCount>& StrategyModes::Registry() const
{
    static const std::array<StrategyDescriptor, kCount> registry{{
        {
            StrategyMode::OriginalBaseline,
            "1 - Original source (no cleanup)",
            "Original source - no NSAMDR",
            "",
            "",
            "",
            false,
            true,
            0,
        },
        {
            StrategyMode::UvStretchDiagnostic,
            "2 - UV/stretch diagnostics",
            "Mode 2 - UV/stretch diagnostics",
            "",
            "",
            "",
            false,
            true,
            2,
        },
        {
            StrategyMode::NeuralReconstruction,
            "3 - NSAMDR cleanup",
            "NSAMDR cleanup",
            "Offline tile-context reconstruction + common live shader",
            "NSAMDR_MODE3_OBJ",
            "NSAMDR_MODE3_MATERIALS",
            true,
            true,
            3,
        },
    }};
    return registry;
}

const StrategyDescriptor& StrategyModes::Descriptor(int mode) const
{
    const int clamped = std::clamp(mode, First(), Last());
    return Registry()[static_cast<size_t>(clamped - First())];
}

const CandidateAssetGpu* StrategyModes::CandidateForMode(
    const StrategyCandidateSet& candidates,
    int mode) const
{
    const StrategyDescriptor& descriptor = Descriptor(mode);
    return descriptor.physicalCandidate ? &candidates.At(mode) : nullptr;
}

CandidateAssetGpu* StrategyModes::CandidateForMode(
    StrategyCandidateSet& candidates,
    int mode) const
{
    const StrategyDescriptor& descriptor = Descriptor(mode);
    return descriptor.physicalCandidate ? &candidates.At(mode) : nullptr;
}

bool StrategyModes::IsVisible(const StrategyCandidateSet& candidates, int mode) const
{
    const StrategyDescriptor& descriptor = Descriptor(mode);
    if (descriptor.visibleWithoutCandidate || !descriptor.physicalCandidate) return true;
    const CandidateAssetGpu* candidate = CandidateForMode(candidates, mode);
    return candidate && (
        candidate->available ||
        (!candidate->objPath.empty() && !candidate->materialManifestPath.empty()));
}

const char* StrategyModes::Label(int mode) const
{
    return Descriptor(mode).overlayLabel;
}

int StrategyModes::ShaderMode(int mode) const
{
    return Descriptor(mode).shaderMode;
}

void StrategyModes::Select(PreviewState& state, int mode) const
{
    state.mode = std::clamp(mode, First(), Last());
    if (state.mode != First()) state.previousEnabledMode = state.mode;
}

void StrategyModes::ProcessNumberHotkeys(PreviewState& state)
{
    for (const StrategyDescriptor& descriptor : Registry())
    {
        const int mode = static_cast<int>(descriptor.mode);
        if (m_inputController.KeyPressed('0' + mode))
        {
            Select(state, mode);
            return;
        }
    }
}
} // namespace nsamdr
