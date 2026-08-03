#pragma once

#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
class NSAMDRTrainingController final
{
public:
    void InitializeFromEnvironment(NSAMDRTrainingSettings& settings) const;
    bool LaunchRetrainBuildPreview(NSAMDRTrainingSettings& settings, std::string& error) const;

private:
    std::string EscapeJson(const std::string& value) const;
    std::string ResolveRepositoryRoot() const;
    bool EnsureDirectory(const std::string& path) const;
    bool WriteConfig(
        const std::string& repositoryRoot,
        const NSAMDRTrainingSettings& settings,
        std::string& configPath,
        std::string& error) const;
};
} // namespace nsamdr
