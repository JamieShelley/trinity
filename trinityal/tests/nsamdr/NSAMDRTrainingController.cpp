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
    const std::string neural = nsamdr + "\\neural_v9";
    if (!EnsureDirectory(artifacts) || !EnsureDirectory(nsamdr) || !EnsureDirectory(neural))
    {
        error = "Could not create artifacts\\nsamdr\\neural_v9.";
        return false;
    }
    configPath = neural + "\\ui_training_profile_v9.json";
    std::ofstream output(configPath, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        error = "Could not write the V9 fidelity-first training profile.";
        return false;
    }
    output << "{\n"
        << "  \"datasetManifest\": \"artifacts/nsamdr/training_v9/dataset_manifest.json\",\n"
        << "  \"datasetRoot\": \"artifacts/nsamdr/training_v9\",\n"
        << "  \"maxFamilies\": " << settings.maxFamilies << ",\n"
        << "  \"cropsPerFamily\": " << settings.cropsPerFamily << ",\n"
        << "  \"sourceCropSize\": " << std::max(settings.sourceCropSize, 512) << ",\n"
        << "  \"minSourceDimension\": " << settings.minSourceDimension << ",\n"
        << "  \"validationFraction\": 0.10,\n"
        << "  \"requireCompletePbrFamily\": false,\n"
        << "  \"identityEpochs\": 1,\n"
        << "  \"residualEpochs\": " << settings.albedoBootstrapEpochs << ",\n"
        << "  \"boundaryEpochs\": " << std::max(settings.jointPbrEpochs / 3, 1) << ",\n"
        << "  \"detailEpochs\": " << std::max(settings.jointPbrEpochs - std::max(settings.jointPbrEpochs / 3, 1), 1) << ",\n"
        << "  \"physicalFinetuneEpochs\": " << settings.renderFinetuneEpochs << ",\n"
        << "  \"tilesPerEpoch\": " << settings.tilesPerEpoch << ",\n"
        << "  \"validationTiles\": " << settings.validationTiles << ",\n"
        << "  \"batchSize\": 1,\n"
        << "  \"learningRate\": " << settings.learningRate << ",\n"
        << "  \"finetuneLearningRate\": " << settings.renderFinetuneLearningRate << ",\n"
        << "  \"weightDecay\": 0.00001,\n"
        << "  \"tileSize\": 128,\n"
        << "  \"targetScale\": 4,\n"
        << "  \"boundaryCandidateCount\": 8,\n"
        << "  \"boundarySamplingProbability\": 0.62,\n"
        << "  \"widths\": [80, 128, 192, 256],\n"
        << "  \"blocksPerLevel\": [3, 4, 6, 6],\n"
        << "  \"decoderBlocks\": [4, 3, 3],\n"
        << "  \"attentionHeads\": 8,\n"
        << "  \"attentionWindow\": 8,\n"
        << "  \"dropPath\": 0.05,\n"
        << "  \"albedoWeight\": " << settings.albedoWeight << ",\n"
        << "  \"normalWeight\": " << settings.normalWeight << ",\n"
        << "  \"materialWeight\": " << settings.materialWeight << ",\n"
        << "  \"edgeWeight\": " << settings.edgeWeight << ",\n"
        << "  \"orientationWeight\": " << settings.orientationWeight << ",\n"
        << "  \"confidenceWeight\": " << settings.confidenceWeight << ",\n"
        << "  \"proposalWeight\": 0.55,\n"
        << "  \"sdfWeight\": 1.20,\n"
        << "  \"crossMapWeight\": 0.45,\n"
        << "  \"seamWeight\": 0.15,\n"
        << "  \"lodBiasMin\": 0.50,\n"
        << "  \"lodBiasMax\": 1.80,\n"
        << "  \"anisotropicBlurProbability\": 0.80,\n"
        << "  \"bcBlockProbability\": 0.85,\n"
        << "  \"chromaLossProbability\": 0.60,\n"
        << "  \"ringingProbability\": 0.45,\n"
        << "  \"haloProbability\": 0.45,\n"
        << "  \"seed\": " << settings.seed << ",\n"
        << "  \"outputDir\": \"artifacts/nsamdr/neural_v9\",\n"
        << "  \"checkpointName\": \"nsamdr_v9_fidelity.pt\",\n"
        << "  \"metadataName\": \"nsamdr_v9_fidelity.json\",\n"
        << "  \"trainingStateName\": \"nsamdr_v9_training_state.pt\",\n"
        << "  \"inferenceTileSize\": 128,\n"
        << "  \"inferenceOverlap\": 24,\n"
        << "  \"device\": \"cuda\",\n"
        << "  \"cudaDeviceIndex\": 0,\n"
        << "  \"matmulPrecision\": \"high\",\n"
        << "  \"ampInitialScale\": 512.0,\n"
        << "  \"gradientClipNorm\": 1.5\n"
        << "}\n";
    if (!output)
    {
        error = "Writing the V9 fidelity-first training profile failed.";
        return false;
    }
    return true;
}

void NSAMDRTrainingController::InitializeFromEnvironment(NSAMDRTrainingSettings& settings) const
{
    std::string cacheRoot = GetEnvironmentString("NSAMDR_EVE_CACHE");
    if (cacheRoot.empty()) cacheRoot = GetEnvironmentString("EVE_SHARED_CACHE");
    if (!cacheRoot.empty())
    {
        std::snprintf(settings.sourceRoot.data(), settings.sourceRoot.size(), "%s", cacheRoot.c_str());
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
    const std::string script = repositoryRoot + "\\scripts\\build\\nsamdr.bat";
    const DWORD attributes = GetFileAttributesA(script.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
    {
        error = "Missing NSAMDR command launcher: " + script;
        return false;
    }

    std::string configPath;
    if (!WriteConfig(repositoryRoot, settings, configPath, error)) return false;
    std::wstring scriptArguments =
        L" retrain-preview --config " + QuoteWindowsArgument(ToWidePath(configPath)) +
        L" --wait-pid " + std::to_wstring(GetCurrentProcessId());
    if (!settings.sourceRoot.empty() && settings.sourceRoot[0] != '\0')
    {
        scriptArguments += L" --shared-cache " + QuoteWindowsArgument(ToWidePath(settings.sourceRoot.data()));
    }
    const std::wstring command =
        L"cmd.exe /d /s /c \"" + QuoteWindowsArgument(ToWidePath(script)) + scriptArguments + L"\"";
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
