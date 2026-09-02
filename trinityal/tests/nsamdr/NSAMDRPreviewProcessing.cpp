#include "StdAfx.h"
#include "NSAMDRPreviewProcessing.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
namespace
{
std::string NormalizePathForComparison(std::string path)
{
    std::replace(path.begin(), path.end(), '/', '\\');
    return ToLowerAscii(std::move(path));
}

bool PathsMatch(const std::string& left, const std::string& right)
{
    return !left.empty() && !right.empty() &&
        NormalizePathForComparison(left) == NormalizePathForComparison(right);
}

bool IsSha256(const std::string& value)
{
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isxdigit(character) != 0;
    });
}

bool IsReadableFile(const std::string& path)
{
    return !path.empty() && static_cast<bool>(std::ifstream(path, std::ios::binary));
}

struct LiveCandidatePointer
{
    std::string token;
    std::string epoch;
    std::string phase;
    std::string checkpointSha;
    std::string candidateObj;
    std::string candidateMaterials;
};

bool ReadLiveCandidatePointer(const std::string& path, LiveCandidatePointer& pointer)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "NSAMDR_LIVE_CANDIDATE_POINTER_V1") return false;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1U);
        if (key == "token") pointer.token = value;
        else if (key == "epoch") pointer.epoch = value;
        else if (key == "phase") pointer.phase = value;
        else if (key == "checkpointSha256") pointer.checkpointSha = value;
        else if (key == "candidateObj") pointer.candidateObj = value;
        else if (key == "candidateMaterials") pointer.candidateMaterials = value;
    }
    return !pointer.token.empty() && !pointer.candidateObj.empty() &&
        !pointer.candidateMaterials.empty();
}

bool ValidationPassed(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    const std::string payload{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    std::string compact;
    compact.reserve(payload.size());
    for (unsigned char character : payload)
    {
        if (std::isspace(character) == 0)
            compact.push_back(static_cast<char>(std::tolower(character)));
    }
    return compact.find("\"passed\":true") != std::string::npos;
}

bool BaselineContainsAlbedo(
    const PreviewResources& resources,
    const std::string& rawAlbedoPath,
    const std::string& provenancePath)
{
    if (PathsMatch(rawAlbedoPath, provenancePath)) return true;
    return std::any_of(resources.areaMaterials.begin(), resources.areaMaterials.end(), [&](const AreaMaterialGpu& material) {
        return material.hasAlbedo && PathsMatch(material.source.albedoPath, provenancePath);
    });
}

bool CandidateContainsAlbedo(const CandidateAssetGpu& candidate, const std::string& provenancePath)
{
    return std::any_of(candidate.areaMaterials.begin(), candidate.areaMaterials.end(), [&](const AreaMaterialGpu& material) {
        return material.hasAlbedo && PathsMatch(material.source.albedoPath, provenancePath);
    });
}

bool CandidateUsesSourceDrawRanges(
    const PreviewResources& resources,
    const CandidateAssetGpu& candidate)
{
    return !resources.areaMaterials.empty() &&
        std::all_of(candidate.areaMaterials.begin(), candidate.areaMaterials.end(), [&](const AreaMaterialGpu& material) {
            return std::any_of(resources.areaMaterials.begin(), resources.areaMaterials.end(), [&](const AreaMaterialGpu& source) {
                return source.source.groupIndex == material.source.groupIndex;
            });
        });
}

std::string ValidateFinalCandidateProvenance(
    const PreviewResources& resources,
    const std::string& rawAlbedoPath,
    const CandidateAssetGpu& candidate)
{
    if (GetEnvironmentString("NSAMDR_PROVENANCE_STATUS") != "VERIFIED")
        return "NSAMDR_PROVENANCE_STATUS is not VERIFIED";

    const std::string sourcePath = GetEnvironmentString("NSAMDR_PROVENANCE_SOURCE");
    const std::string sourceSha = GetEnvironmentString("NSAMDR_PROVENANCE_SOURCE_SHA");
    const std::string candidatePath = GetEnvironmentString("NSAMDR_PROVENANCE_CANDIDATE");
    const std::string candidateSha = GetEnvironmentString("NSAMDR_PROVENANCE_CANDIDATE_SHA");
    const std::string provenanceFile = GetEnvironmentString("NSAMDR_PROVENANCE_FILE");
    const std::string analysisFile = GetEnvironmentString("NSAMDR_FINAL_ANALYSIS");
    const std::string validationFile = GetEnvironmentString("NSAMDR_FINAL_VALIDATION");
    const std::string checkpointPath = GetEnvironmentString("NSAMDR_PREVIEW_CHECKPOINT");
    const std::string checkpointSha = GetEnvironmentString("NSAMDR_PREVIEW_CHECKPOINT_SHA256");
    const std::string authority = ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY"));

    if (!IsSha256(sourceSha)) return "source SHA-256 is missing or malformed";
    if (!IsSha256(candidateSha)) return "candidate SHA-256 is missing or malformed";
    if (!IsSha256(checkpointSha)) return "checkpoint SHA-256 is missing or malformed";
    if (authority.find("final") == std::string::npos ||
        authority.find("intermediate") != std::string::npos ||
        authority.find("paused") != std::string::npos ||
        authority.find("latest") != std::string::npos)
        return "checkpoint authority is not a final selection";
    if (!IsReadableFile(checkpointPath)) return "bound checkpoint is missing or unreadable";
    if (!IsReadableFile(provenanceFile)) return "provenance evidence is missing or unreadable";
    if (!IsReadableFile(analysisFile)) return "final analysis is missing or unreadable";
    if (!ValidationPassed(validationFile)) return "final validation is missing or did not pass";
    if (!BaselineContainsAlbedo(resources, rawAlbedoPath, sourcePath))
        return "raw pane albedo does not match the proven source path";
    if (!CandidateContainsAlbedo(candidate, candidatePath))
        return "final pane albedo does not match the proven candidate path";
    if (!CandidateUsesSourceDrawRanges(resources, candidate))
        return "final material groups do not map to the raw source mesh draw ranges";
    return {};
}
} // namespace

PreviewProcessing::PreviewProcessing(
    PreviewRenderer& renderer,
    AssetProcessor& assetProcessor,
    SceneController& sceneController)
    : m_renderer(renderer),
      m_assetProcessor(assetProcessor),
      m_sceneController(sceneController)
{
}

bool PreviewProcessing::LoadCandidates(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const PreviewResources& resources,
    const std::string& rawAlbedoPath,
    FinalCandidateSet& candidates)
{
    CandidateAssetGpu& candidate = candidates.candidate;
    if (ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY")) == "training-intermediate")
    {
        candidate.status = "waiting for the first completed live training epoch";
        return RefreshLiveCandidate(device, context, resources, rawAlbedoPath, candidates);
    }
    if (!m_assetProcessor.LoadCandidateAsset(
            device,
            context,
            "NSAMDR FINAL",
            GetEnvironmentString("NSAMDR_FINAL_OBJ"),
            GetEnvironmentString("NSAMDR_FINAL_MATERIALS"),
            candidate))
    {
        return false;
    }
    if (!candidate.available) return true;

    const std::string provenanceFailure = ValidateFinalCandidateProvenance(
        resources,
        rawAlbedoPath,
        candidate);
    if (!provenanceFailure.empty())
    {
        candidate.available = false;
        candidate.status = "provenance gate blocked NSAMDR FINAL: " + provenanceFailure;
    }
    return true;
}

bool PreviewProcessing::RefreshLiveCandidate(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const PreviewResources& resources,
    const std::string& rawAlbedoPath,
    FinalCandidateSet& candidates)
{
    if (ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY")) != "training-intermediate")
        return true;
    const std::string pointerPath = GetEnvironmentString("NSAMDR_LIVE_CANDIDATE_POINTER");
    if (pointerPath.empty()) return true;

    LiveCandidatePointer pointer;
    if (!ReadLiveCandidatePointer(pointerPath, pointer)) return true;
    if (pointer.token == m_liveCandidateToken) return true;

    CandidateAssetGpu nextCandidate;
    const std::string label = "NSAMDR LIVE epoch " + pointer.epoch + " | " + pointer.phase;
    if (!m_assetProcessor.LoadCandidateAsset(
            device,
            context,
            label,
            pointer.candidateObj,
            pointer.candidateMaterials,
            nextCandidate))
    {
        std::printf("NSAMDR live preview: candidate GPU load failed for token %s\n", pointer.token.c_str());
        return true;
    }
    if (!nextCandidate.available)
    {
        std::printf("NSAMDR live preview: candidate unavailable for token %s: %s\n",
            pointer.token.c_str(), nextCandidate.status.c_str());
        return true;
    }
    if (!CandidateUsesSourceDrawRanges(resources, nextCandidate))
    {
        std::printf("NSAMDR live preview: rejected token %s because draw ranges differ from source\n",
            pointer.token.c_str());
        return true;
    }

    nextCandidate.status = "UNQUALIFIED INTERMEDIATE | epoch=" + pointer.epoch +
        " | phase=" + pointer.phase + " | checkpoint=" +
        (pointer.checkpointSha.empty() ? std::string("unknown") : pointer.checkpointSha.substr(0U, 12U));
    candidates.candidate = std::move(nextCandidate);
    m_liveCandidateToken = pointer.token;
    std::printf("NSAMDR live preview: hot-reloaded %s\n", candidates.candidate.status.c_str());
    return true;
}

bool PreviewProcessing::InitializeState(
    PreviewState& state,
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    PreviewResources& resources,
    const ObjMesh& mesh)
{
    const bool hasAreaAlbedo = std::any_of(
        resources.areaMaterials.begin(),
        resources.areaMaterials.end(),
        [](const AreaMaterialGpu& material) { return material.hasAlbedo; });
    const bool hasAreaNormal = std::any_of(
        resources.areaMaterials.begin(),
        resources.areaMaterials.end(),
        [](const AreaMaterialGpu& material) { return material.hasNormal; });
    const bool hasAreaPgs = std::any_of(
        resources.areaMaterials.begin(),
        resources.areaMaterials.end(),
        [](const AreaMaterialGpu& material) { return material.hasPgs; });

    state.useTexture = resources.hasExternalAlbedo || hasAreaAlbedo || !resources.areaMaterials.empty();
    state.useNormalMap = resources.hasNormalMap || hasAreaNormal;
    state.usePgsMap = resources.hasPgsMap || hasAreaPgs;
    state.useEnvironment = resources.hasEnvironment;
    if (!m_renderer.EnsureEnvironment(device, context, resources, state)) return false;
    m_sceneController.ApplyGameLightingPreset(state);
    m_sceneController.ResetView(state, mesh);
    return true;
}

void PreviewProcessing::PrintDiagnostics(
    const ObjMesh& mesh,
    const PreviewResources& resources,
    const FinalCandidateSet& candidates,
    const std::string& albedoPath,
    const std::string& normalPath,
    const std::string& pgsPath,
    const std::string& environmentPath,
    const std::string& materialManifestPath) const
{
    std::printf("NSAMDR OBJ loaded: %s\n", mesh.path.c_str());
    std::printf("  triangles=%u vertices=%zu sourcePositions=%u sourceUVs=%u sourceNormals=%u\n",
        mesh.triangleCount,
        mesh.vertices.size(),
        mesh.sourcePositionCount,
        mesh.sourceTexcoordCount,
        mesh.sourceNormalCount);
    std::printf("  uvStretchAverage=%.4f uvStretchMaximum=%.4f degenerateUvTriangles=%u calibrationRawP50=%.4f calibrationRawP95=%.4f\n",
        mesh.averageStretch,
        mesh.maximumStretch,
        mesh.degenerateUvTriangles,
        mesh.stretchCalibrationLow,
        mesh.stretchCalibrationHigh);

    if (resources.hasExternalAlbedo)
        std::printf("  albedo=%s (%ux%u)\n", albedoPath.c_str(), resources.textureWidth, resources.textureHeight);
    else
        std::printf("  albedo=<neutral fallback>\n");

    if (resources.hasNormalMap)
        std::printf("  normal=%s (%ux%u)\n", normalPath.c_str(), resources.normalWidth, resources.normalHeight);
    if (resources.hasPgsMap)
        std::printf("  pgs=%s (%ux%u)\n", pgsPath.c_str(), resources.pgsWidth, resources.pgsHeight);

    if (!resources.areaMaterials.empty())
        std::printf("  sofMaterials=%s (draws=%zu, groups=%zu)\n", materialManifestPath.c_str(), resources.areaMaterials.size(), mesh.drawRanges.size());
    else
        std::printf("  sofMaterials=<global texture fallback> (groups=%zu)\n", mesh.drawRanges.size());

    std::printf("  comparison=A_RAW_SOURCE_vs_B_NSAMDR_FINAL sharedMesh=true sharedCamera=true sharedShader=true sharedSampler=true\n");
    PrintCandidate(candidates.candidate);

    if (resources.hasEnvironment)
        std::printf("  environment=%s (%ux%u)\n", environmentPath.c_str(), resources.environmentWidth, resources.environmentHeight);
    else
        std::printf("  environment=<procedural fallback>\n");
}


void PreviewProcessing::PrintCandidate(const CandidateAssetGpu& candidate) const
{
    if (candidate.available)
    {
        std::printf("  finalCandidate=loaded provenance=verified texture=%ux%u draws=%zu obj=%s\n",
            candidate.maximumTextureWidth,
            candidate.maximumTextureHeight,
            candidate.areaMaterials.size(),
            candidate.objPath.c_str());
    }
    else
    {
        std::printf("  finalCandidate=unavailable reason=%s\n", candidate.status.c_str());
    }
}

} // namespace nsamdr
