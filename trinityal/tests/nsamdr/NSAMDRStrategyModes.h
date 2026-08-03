#pragma once

#include "NSAMDRInputController.h"

namespace nsamdr
{
enum class StrategyMode : int
{
    OriginalBaseline = 1,
    UvStretchDiagnostic = 2,
    NeuralReconstruction = 3,
};

struct StrategyDescriptor
{
    StrategyMode mode = StrategyMode::OriginalBaseline;
    const char* panelLabel = "";
    const char* overlayLabel = "";
    const char* candidateLabel = "";
    const char* objEnvironment = "";
    const char* materialsEnvironment = "";
    bool physicalCandidate = false;
    bool visibleWithoutCandidate = true;
    int shaderMode = 0;
};

class StrategyModes final
{
public:
    explicit StrategyModes(InputController& inputController);

    static constexpr int First() { return static_cast<int>(StrategyMode::OriginalBaseline); }
    static constexpr int Last() { return static_cast<int>(StrategyMode::NeuralReconstruction); }
    static constexpr size_t kCount = 3U;
    static constexpr int Count() { return static_cast<int>(kCount); }

    const std::array<StrategyDescriptor, kCount>& Registry() const;
    const StrategyDescriptor& Descriptor(int mode) const;
    const CandidateAssetGpu* CandidateForMode(const StrategyCandidateSet& candidates, int mode) const;
    CandidateAssetGpu* CandidateForMode(StrategyCandidateSet& candidates, int mode) const;
    bool IsVisible(const StrategyCandidateSet& candidates, int mode) const;
    const char* Label(int mode) const;
    int ShaderMode(int mode) const;
    void Select(PreviewState& state, int mode) const;
    void ProcessNumberHotkeys(PreviewState& state);

private:
    InputController& m_inputController;
};
} // namespace nsamdr
