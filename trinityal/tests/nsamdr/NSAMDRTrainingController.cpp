#include "StdAfx.h"
#include "NSAMDRTrainingController.h"

namespace nsamdr
{
std::string NSAMDRTrainingController::EscapeJson(const std::string& value) const
{
    std::string escaped;
    escaped.reserve(value.size() + 16U);
    for (char character : value)
    {
        switch (character)
        {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

std::string NSAMDRTrainingController::ResolveRepositoryRoot() const
{
    std::string root = GetEnvironmentString("NSAMDR_EVE_REPO_ROOT");
    if (!root.empty()) return root;
    root = GetEnvironmentString("NSAMDR_REPO_ROOT");
    if (!root.empty()) return root;
    std::array<char, MAX_PATH> buffer{};
    const DWORD length = GetCurrentDirectoryA(static_cast<DWORD>(buffer.size()), buffer.data());
    return length > 0U && length < buffer.size() ? std::string(buffer.data(), length) : std::string();
}

bool NSAMDRTrainingController::EnsureDirectory(const std::string& path) const
{
    if (path.empty()) return false;
    const DWORD attributes = GetFileAttributesA(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES)
    {
        return (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }
    return CreateDirectoryA(path.c_str(), nullptr) != FALSE || GetLastError() == ERROR_ALREADY_EXISTS;
}

bool NSAMDRTrainingController::WriteConfig(
    const std::string& repositoryRoot,
    const NSAMDRTrainingSettings& settings,
    std::string& configPath,
    std::string& error) const
{
    const std::string artifacts = repositoryRoot + "\\artifacts";
    const std::string nsamdr = artifacts + "\\nsamdr";
    const std::string neural = nsamdr + "\\neural";
    if (!EnsureDirectory(artifacts) || !EnsureDirectory(nsamdr) || !EnsureDirectory(neural))
    {
        error = "Could not create artifacts\\nsamdr\\neural.";
        return false;
    }
    configPath = neural + "\\ui_training_profile.json";
    std::ofstream output(configPath, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        error = "Could not write the neural training profile.";
        return false;
    }
    output << "{\n"
        << "  \"epochs\": " << settings.epochs << ",\n"
        << "  \"tilesPerEpoch\": " << settings.tilesPerEpoch << ",\n"
        << "  \"batchSize\": " << settings.batchSize << ",\n"
        << "  \"learningRate\": " << settings.learningRate << ",\n"
        << "  \"baseChannels\": " << settings.baseChannels << ",\n"
        << "  \"residualBlocks\": " << settings.residualBlocks << ",\n"
        << "  \"tileSize\": " << settings.tileSize << ",\n"
        << "  \"reconstructionWeight\": " << settings.reconstructionWeight << ",\n"
        << "  \"edgeWeight\": " << settings.edgeWeight << ",\n"
        << "  \"identityWeight\": " << settings.identityWeight << ",\n"
        << "  \"confidenceWeight\": " << settings.confidenceWeight << ",\n"
        << "  \"flowSmoothnessWeight\": " << settings.flowSmoothnessWeight << ",\n"
        << "  \"maxResidual\": " << settings.maxResidual << ",\n"
        << "  \"maxOffsetPixels\": " << settings.maxOffsetPixels << ",\n"
        << "  \"maxSourceFiles\": " << settings.maxSourceFiles << ",\n"
        << "  \"realSourceFraction\": " << settings.realSourceFraction << ",\n"
        << "  \"artifactFraction\": " << settings.artifactFraction << ",\n"
        << "  \"seed\": " << settings.seed << ",\n"
        << "  \"sourceRoot\": \"" << EscapeJson(settings.sourceRoot.data()) << "\",\n"
        << "  \"sourceGlobs\": [\"**/*_d.png\", \"**/*_ar.png\", \"**/*.dds\"],\n"
        << "  \"outputDir\": \"artifacts/nsamdr/neural\",\n"
        << "  \"checkpointName\": \"nsamdr_tile_context.pt\",\n"
        << "  \"metadataName\": \"nsamdr_tile_context.json\",\n"
        << "  \"inferenceTileSize\": 512,\n"
        << "  \"inferenceOverlap\": 64,\n"
        << "  \"device\": \"cuda\",\n"
        << "  \"cudaDeviceIndex\": 0,\n"
        << "  \"matmulPrecision\": \"high\"\n"
        << "}\n";
    if (!output)
    {
        error = "Writing the neural training profile failed.";
        return false;
    }
    return true;
}


void NSAMDRTrainingController::InitializeFromEnvironment(NSAMDRTrainingSettings& settings) const
{
    const std::string sourceRoot = GetEnvironmentString("NSAMDR_TRAINING_SOURCE_ROOT");
    if (!sourceRoot.empty())
    {
        std::snprintf(settings.sourceRoot.data(), settings.sourceRoot.size(), "%s", sourceRoot.c_str());
    }
    else
    {
        const std::string cacheRoot = GetEnvironmentString("NSAMDR_EVE_CACHE");
        if (!cacheRoot.empty())
        {
            std::snprintf(settings.sourceRoot.data(), settings.sourceRoot.size(), "%s", cacheRoot.c_str());
        }
    }
}

bool NSAMDRTrainingController::LaunchRetrainBuildPreview(NSAMDRTrainingSettings& settings, std::string& error) const
{
    const std::string repositoryRoot = ResolveRepositoryRoot();
    if (repositoryRoot.empty())
    {
        error = "Could not determine the repository root.";
        return false;
    }
    const std::string script = repositoryRoot + "\\scripts\\build\\retrain_nsamdr_and_preview.bat";
    const DWORD attributes = GetFileAttributesA(script.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
    {
        error = "Missing retraining launcher: " + script;
        return false;
    }

    std::string configPath;
    if (!WriteConfig(repositoryRoot, settings, configPath, error)) return false;
    const std::wstring command =
        L"cmd.exe /d /s /c \"" + QuoteWindowsArgument(ToWidePath(script)) +
        L" --config " + QuoteWindowsArgument(ToWidePath(configPath)) +
        L" --wait-pid " + std::to_wstring(GetCurrentProcessId()) + L"\"";
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const std::wstring workingDirectory = ToWidePath(repositoryRoot);
    const BOOL launched = CreateProcessW(
        nullptr,
        mutableCommand.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP,
        nullptr,
        workingDirectory.c_str(),
        &startup,
        &process);
    if (!launched)
    {
        error = "Could not launch retraining. Windows error " + std::to_string(GetLastError()) + ".";
        return false;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    settings.status = "Retraining process launched. Closing this preview so the build can replace it.";
    return true;
}

} // namespace nsamdr
