#include "StdAfx.h"
#include "NSAMDRPreviewPanel.h"
#include "NSAMDRPreviewUtilities.h"
#include <cstdlib>

namespace nsamdr
{
namespace
{
std::string ReadEnvironmentVariable(const char* name)
{
#if defined(_WIN32)
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr)
    {
        if (value != nullptr) std::free(value);
        return {};
    }
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
#endif
}

}

PreviewPanel::PreviewPanel(SceneController& sceneController)
    : m_sceneController(sceneController)
{
}

void PreviewPanel::DrawShipSelector(ShipCatalog& catalog, HWND hwnd)
{
    if (!ImGui::CollapsingHeader("Select another ship from the EVE cache", ImGuiTreeNodeFlags_DefaultOpen)) return;

    ImGui::TextWrapped("Search by real ship name, class, faction or internal resource code. The preferred highest-detail mesh is selected automatically. Ship types that share a hull can resolve to the same geometry in this Granny-free test.");
    if (ImGui::InputTextWithHint("##shipSearch", "Search: Archon, Raven, carrier, Caldari...", catalog.search.data(), catalog.search.size()))
    {
        RebuildCatalogFilter(catalog);
    }
    if (ImGui::Checkbox("Show raw LOD / asset variants", &catalog.showRawVariants))
    {
        RebuildCatalogFilter(catalog);
    }
    ImGui::Text("Matches: %zu selections from %zu grouped ships", catalog.filtered.size(), catalog.entries.size());

    ImGui::BeginChild("ShipCatalog", ImVec2(570.0f, 210.0f), true);
    const size_t visibleLimit = std::min<size_t>(catalog.filtered.size(), 400U);
    for (size_t filteredIndex = 0; filteredIndex < visibleLimit; ++filteredIndex)
    {
        const CatalogSelection& selection = catalog.filtered[filteredIndex];
        if (selection.entryIndex >= catalog.entries.size()) continue;
        const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
        const bool selected = catalog.selectedFilteredIndex == static_cast<int>(filteredIndex);
        const std::string label = CatalogSelectionLabel(catalog, selection);
        ImGui::PushID(static_cast<int>(filteredIndex));
        if (ImGui::Selectable(label.c_str(), selected))
        {
            catalog.selectedFilteredIndex = static_cast<int>(filteredIndex);
        }
        std::string metadata = entry.groupName;
        if (!entry.factionName.empty()) metadata += " | " + entry.factionName;
        if (!entry.typeId.empty()) metadata += " | type " + entry.typeId;
        ImGui::TextDisabled("%s", metadata.c_str());
        if (catalog.showRawVariants)
        {
            ImGui::TextDisabled("%s", CatalogSelectionAsset(catalog, selection).c_str());
        }
        if (selected) ImGui::SetItemDefaultFocus();
        ImGui::PopID();
    }
    if (catalog.filtered.size() > visibleLimit)
    {
        ImGui::TextDisabled("Refine the search to see the remaining %zu matches.", catalog.filtered.size() - visibleLimit);
    }
    ImGui::EndChild();

    const bool canOpen = catalog.selectedFilteredIndex >= 0 &&
        static_cast<size_t>(catalog.selectedFilteredIndex) < catalog.filtered.size();
    if (canOpen && ImGui::Button("Convert and open selected ship"))
    {
        const CatalogSelection& selection = catalog.filtered[static_cast<size_t>(catalog.selectedFilteredIndex)];
        const std::string selectedAsset = CatalogSelectionAsset(catalog, selection);
        const std::string selectedLabel = CatalogSelectionLabel(catalog, selection);
        const ShipCatalogEntry& selectedEntry = catalog.entries[selection.entryIndex];
        const std::string selectionKey = selection.variantIndex >= 0 ? selectedAsset : selectedEntry.canonicalKey;
        std::string launchError;
        if (LaunchCachedShip(selectedAsset, selectionKey, launchError))
        {
            catalog.status = "Launching conversion for: " + selectedLabel;
            PostMessage(hwnd, WM_CLOSE, 0, 0);
        }
        else
        {
            catalog.status = launchError;
        }
    }
    else if (!canOpen)
    {
        ImGui::TextDisabled("Select a matching ship first.");
    }
    ImGui::SameLine();
    ImGui::TextWrapped("%s", catalog.status.c_str());
}



void PreviewPanel::Draw(
    PreviewState& state,
    ShipCatalog& catalog,
    HWND hwnd,
    const ObjMesh& mesh,
    const PreviewResources& resources,
    const FinalCandidateSet& candidates,
    const std::string& albedoPath,
    const std::string& normalPath,
    const std::string& pgsPath)
{
    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.94f);
    const float panelWidth = std::min(640.0f, std::max(420.0f, static_cast<float>(resources.width) * 0.45f));
    const float panelHeight = std::max(280.0f, static_cast<float>(resources.height) - 24.0f);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::Begin("NSAMDR Real EVE Ship Controls");

    ImGui::TextUnformatted("NSAMDR — Neural Stretch-Aware Material Detail Reconstruction");
    const std::string previewExperiment = ReadEnvironmentVariable("NSAMDR_PREVIEW_EXPERIMENT");
    const std::string previewCheckpoint = ReadEnvironmentVariable("NSAMDR_PREVIEW_CHECKPOINT");
    const std::string previewCheckpointSha = ReadEnvironmentVariable("NSAMDR_PREVIEW_CHECKPOINT_SHA256");
    const std::string previewAuthority = ReadEnvironmentVariable("NSAMDR_PREVIEW_AUTHORITY");
    const bool liveTrainingPreview = previewAuthority == "training-intermediate";
    const CandidateAssetGpu& finalCandidate = candidates.candidate;
    ImGui::TextWrapped(
        liveTrainingPreview
            ? "Live training comparison: A RAW SOURCE and B CURRENT NSAMDR EPOCH. Both panes keep the same EVE mesh, camera, lighting, background, material shader and sampler while only B hot-reloads after a completed epoch."
            : "Fixed production comparison: A RAW SOURCE and B NSAMDR FINAL. Both panes use the source mesh, one camera, one material shader and the same 16x anisotropic sampler at zero LOD bias. The renderer does not sharpen, denoise or otherwise alter the final candidate.");

    ImGui::Separator();
    if (liveTrainingPreview)
    {
        ImGui::TextColored(
            ImVec4(1.0f, 0.78f, 0.20f, 1.0f),
            "LIVE TRAINING PREVIEW — UNQUALIFIED INTERMEDIATE");
        ImGui::Text("Experiment: %s", previewExperiment.empty() ? "UNSET" : previewExperiment.c_str());
        ImGui::Text("Authority: training-intermediate (never promotable as a final)");
        ImGui::TextWrapped("Current B state: %s",
            finalCandidate.status.empty() ? "waiting for first completed epoch" : finalCandidate.status.c_str());
        ImGui::TextWrapped("To stop on a bad visual regression, use Stop current process in the NSAMDR Workflow window.");
    }
    else
    {
        const bool provenanceVerified =
            ReadEnvironmentVariable("NSAMDR_PROVENANCE_STATUS") == "VERIFIED" &&
            finalCandidate.available;
        ImGui::TextUnformatted("Immutable final provenance");
        ImGui::TextColored(
            provenanceVerified ? ImVec4(0.35f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.28f, 0.22f, 1.0f),
            "Native provenance gate: %s",
            provenanceVerified ? "VERIFIED" : "BLOCKED");
        ImGui::Text("Experiment: %s", previewExperiment.empty() ? "UNSET" : previewExperiment.c_str());
        ImGui::TextWrapped("Checkpoint: %s", previewCheckpoint.empty() ? "UNSET" : previewCheckpoint.c_str());
        ImGui::TextWrapped("Checkpoint SHA-256: %s", previewCheckpointSha.empty() ? "UNSET" : previewCheckpointSha.c_str());
        ImGui::Text("Authority: %s", previewAuthority.empty() ? "UNSET" : previewAuthority.c_str());
    }
    ImGui::Separator();
    ImGui::BeginChild("NSAMDRScrollableControls", ImVec2(0.0f, 0.0f), false);

    ImGui::TextWrapped("Model: %s", mesh.path.c_str());
    if (resources.hasExternalAlbedo)
    {
        ImGui::TextWrapped("Albedo: %s", albedoPath.c_str());
        ImGui::Text("Albedo size: %u x %u", resources.textureWidth, resources.textureHeight);
    }
    else ImGui::TextUnformatted("Albedo: neutral fallback");
    if (resources.hasNormalMap) ImGui::TextWrapped("Normal: %s (%u x %u)", normalPath.c_str(), resources.normalWidth, resources.normalHeight);
    if (resources.hasPgsMap) ImGui::TextWrapped("MaterialMap: %s (%u x %u)", pgsPath.c_str(), resources.pgsWidth, resources.pgsHeight);
    if (!resources.areaMaterials.empty())
    {
        ImGui::Text("SOF material draws: %zu", resources.areaMaterials.size());
        if (resources.baselineComplete)
        {
            ImGui::TextUnformatted("Visual baseline: COMPLETE");
        }
        else
        {
            ImGui::TextWrapped("Raw source material: INCOMPLETE (%d unresolved inputs). The A/B layout remains visible for diagnosis.", resources.baselineUnresolvedCount);
        }
        if (ImGui::TreeNode("Per-area baseline diagnostics"))
        {
            for (const AreaMaterialGpu& material : resources.areaMaterials)
            {
                ImGui::PushID(material.source.groupIndex);
                const char* status = material.source.baselineComplete ? "OK" : "BLOCKED";
                ImGui::Text("[%s] group %d | %s | %s | %s", status, material.source.groupIndex,
                    material.source.areaName.empty() ? material.source.areaType.c_str() : material.source.areaName.c_str(),
                    ShaderFamilyName(material.source.shaderFamily),
                    material.source.pass == MaterialPass::Opaque ? "opaque" :
                    material.source.pass == MaterialPass::Decal ? "decal" :
                    material.source.pass == MaterialPass::Transparent ? "transparent" : "additive");
                if (!material.source.unresolvedSemantics.empty())
                    ImGui::TextWrapped("Missing semantics: %s", material.source.unresolvedSemantics.c_str());
                if (!material.source.shaderPath.empty()) ImGui::TextWrapped("Shader: %s", material.source.shaderPath.c_str());
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
    }
    else
    {
        ImGui::TextUnformatted("Materials: global material fallback");
    }
    if (resources.hasEnvironment)
    {
        const EnvironmentGpu* selectedEnvironment = SelectedEnvironment(resources, state);
        if (selectedEnvironment)
        {
            ImGui::TextWrapped(
                "EVE nebula environment: %s (%u x %u)",
                selectedEnvironment->label.c_str(),
                selectedEnvironment->width,
                selectedEnvironment->height);
        }
    }
    else
    {
        ImGui::TextUnformatted("Environment: procedural EVE-like fallback");
    }
    ImGui::Text("OBJ source: %u positions, %u UVs, %u normals", mesh.sourcePositionCount, mesh.sourceTexcoordCount, mesh.sourceNormalCount);
    ImGui::Text("Render mesh: %u triangles, %zu corner vertices", mesh.triangleCount, mesh.vertices.size());
    ImGui::Text("UV metric: avg %.3f, max %.3f, degenerate %u", mesh.averageStretch, mesh.maximumStretch, mesh.degenerateUvTriangles);
    ImGui::Text("Mesh calibration: raw P50 %.3f, raw P95 %.3f", mesh.stretchCalibrationLow, mesh.stretchCalibrationHigh);

    ImGui::Separator();
    DrawShipSelector(catalog, hwnd);

    ImGui::Separator();
    ImGui::TextUnformatted(liveTrainingPreview ? "NSAMDR LIVE training candidate" : "NSAMDR FINAL candidate");
    if (finalCandidate.available)
    {
        if (liveTrainingPreview)
            ImGui::Text("Hot-reloaded completed epoch | %u x %u | %zu material draws",
                finalCandidate.maximumTextureWidth,
                finalCandidate.maximumTextureHeight,
                finalCandidate.areaMaterials.size());
        else
            ImGui::Text("Loaded and provenance-gated | %u x %u | %zu material draws",
                finalCandidate.maximumTextureWidth,
                finalCandidate.maximumTextureHeight,
                finalCandidate.areaMaterials.size());
        if (liveTrainingPreview) ImGui::TextWrapped("%s", finalCandidate.status.c_str());
        ImGui::TextWrapped("OBJ manifest source: %s", finalCandidate.objPath.c_str());
        ImGui::TextWrapped("Materials: %s", finalCandidate.materialManifestPath.c_str());
    }
    else
    {
        ImGui::TextColored(
            ImVec4(1.0f, 0.28f, 0.22f, 1.0f),
            "UNAVAILABLE - %s",
            finalCandidate.status.empty() ? "candidate was not configured" : finalCandidate.status.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted(liveTrainingPreview ? "Live epoch A/B comparison" : "Production A/B comparison");
    ImGui::Checkbox("Vertical panes", &state.splitVertical);
    ImGui::SameLine();
    ImGui::Checkbox("Swap A and B", &state.swapSplitSides);
    ImGui::TextWrapped(
        liveTrainingPreview
            ? "A RAW SOURCE stays fixed while B CURRENT NSAMDR EPOCH hot-reloads. Camera, transform, lighting, EVE environment, shader and sampler remain identical, so visual changes on the right are training changes."
            : "A RAW SOURCE and B NSAMDR FINAL are always visible. Both panes use the same source vertex/index buffers, camera, transform, lighting, environment, material shader, gradient sampling and 16x anisotropic sampler at zero LOD bias.");

    const AreaMaterialGpu* baselineProofMaterial = nullptr;
    const AreaMaterialGpu* candidateProofMaterial = nullptr;
    if (finalCandidate.available)
    {
        for (const AreaMaterialGpu& baselineMaterial : resources.areaMaterials)
        {
            if (!baselineMaterial.hasAlbedo) continue;
            const auto candidateIt = std::find_if(
                finalCandidate.areaMaterials.begin(),
                finalCandidate.areaMaterials.end(),
                [&](const AreaMaterialGpu& candidateMaterial) {
                    return candidateMaterial.hasAlbedo &&
                        candidateMaterial.source.groupIndex == baselineMaterial.source.groupIndex;
                });
            if (candidateIt != finalCandidate.areaMaterials.end())
            {
                baselineProofMaterial = &baselineMaterial;
                candidateProofMaterial = &(*candidateIt);
                break;
            }
        }
    }
    if (baselineProofMaterial && candidateProofMaterial)
    {
        const bool differentPath =
            baselineProofMaterial->source.albedoPath != candidateProofMaterial->source.albedoPath;
        const bool differentSrv =
            baselineProofMaterial->albedoView.Get() != candidateProofMaterial->albedoView.Get();
        const bool isolated = differentPath && differentSrv;
        ImGui::TextColored(
            isolated ? ImVec4(0.35f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.28f, 0.22f, 1.0f),
            "A/B texture resource isolation: %s",
            isolated ? "PASS" : "FAIL");
    }

    if (!liveTrainingPreview)
    {
        const std::string provenanceSource = ReadEnvironmentVariable("NSAMDR_PROVENANCE_SOURCE");
        const std::string provenanceSourceSha = ReadEnvironmentVariable("NSAMDR_PROVENANCE_SOURCE_SHA");
        const std::string provenanceCandidate = ReadEnvironmentVariable("NSAMDR_PROVENANCE_CANDIDATE");
        const std::string provenanceCandidateSha = ReadEnvironmentVariable("NSAMDR_PROVENANCE_CANDIDATE_SHA");
        const std::string provenanceFile = ReadEnvironmentVariable("NSAMDR_PROVENANCE_FILE");
        if (!provenanceSource.empty()) ImGui::TextWrapped("A source: %s", provenanceSource.c_str());
        if (!provenanceSourceSha.empty()) ImGui::TextWrapped("A SHA-256: %s", provenanceSourceSha.c_str());
        if (!provenanceCandidate.empty()) ImGui::TextWrapped("B final: %s", provenanceCandidate.c_str());
        if (!provenanceCandidateSha.empty()) ImGui::TextWrapped("B SHA-256: %s", provenanceCandidateSha.c_str());
        if (!provenanceFile.empty()) ImGui::TextWrapped("Evidence: %s", provenanceFile.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("EVE environment and inspection lighting");
    static constexpr const char* lightingPresets[] = {
        "Game-like",
        "Studio",
        "Harsh inspection",
        "Dark silhouette",
    };
    if (ImGui::Combo("Lighting preset", &state.lightingPreset, lightingPresets, 4))
    {
        m_sceneController.ApplyLightingPreset(state, state.lightingPreset);
    }
    if (ImGui::Button("Reset selected lighting preset")) m_sceneController.ApplyLightingPreset(state, state.lightingPreset);
    if (resources.hasEnvironment) ImGui::Checkbox("Use extracted EVE nebula", &state.useEnvironment);
    if (!resources.environments.empty())
    {
        int selectedEnvironment = static_cast<int>(std::min<size_t>(state.environmentIndex, resources.environments.size() - 1U));
        const char* preview = resources.environments[static_cast<size_t>(selectedEnvironment)].label.c_str();
        if (ImGui::BeginCombo("Background", preview))
        {
            for (size_t index = 0; index < resources.environments.size(); ++index)
            {
                const bool selected = index == static_cast<size_t>(selectedEnvironment);
                if (ImGui::Selectable(resources.environments[index].label.c_str(), selected))
                {
                    state.environmentIndex = static_cast<uint32_t>(index);
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("Previous background"))
        {
            state.environmentIndex = state.environmentIndex == 0U
                ? static_cast<uint32_t>(resources.environments.size() - 1U)
                : state.environmentIndex - 1U;
        }
        ImGui::SameLine();
        if (ImGui::Button("Next background"))
        {
            state.environmentIndex = (state.environmentIndex + 1U) % static_cast<uint32_t>(resources.environments.size());
        }
    }
    ImGui::SliderFloat("Environment light", &state.environmentIntensity, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Background intensity", &state.backgroundIntensity, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Environment reflections", &state.reflectionStrength, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Key yaw", &state.keyYaw, -3.14159f, 3.14159f, "%.2f");
    ImGui::SliderFloat("Key pitch", &state.keyPitch, -1.45f, 1.45f, "%.2f");
    ImGui::SliderFloat("Key intensity", &state.keyIntensity, 0.0f, 3.5f, "%.2f");
    ImGui::SliderFloat("Cool fill intensity", &state.fillIntensity, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Blue rim intensity", &state.rimIntensity, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Hemisphere ambient", &state.ambient, 0.0f, 1.5f, "%.2f");
    ImGui::SliderFloat("Exposure", &state.exposure, 0.25f, 3.0f, "%.2f");
    ImGui::SliderFloat("Specular strength", &state.specularStrength, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Roughness bias", &state.roughnessBias, -0.45f, 0.45f, "%.2f");
    if (resources.hasNormalMap)
    {
        ImGui::Checkbox("Use EVE normal map", &state.useNormalMap);
        ImGui::SliderFloat("Normal-map strength", &state.normalMapStrength, 0.0f, 2.0f, "%.2f");
    }
    if (resources.hasPgsMap) ImGui::Checkbox("Use EVE MaterialMap selector", &state.usePgsMap);

    ImGui::Separator();
    ImGui::Checkbox("Use albedo texture", &state.useTexture);
    ImGui::Checkbox("Debug invert baked texture V (V)", &state.flipV);
    ImGui::Checkbox("Wireframe", &state.wireframe);

    ImGui::Separator();
    ImGui::Checkbox("Automatic orbit", &state.autoOrbit);
    ImGui::SliderFloat("Automatic orbit speed", &state.orbitSpeed, -1.0f, 1.0f, "%.2f rad/s");
    ImGui::SliderFloat("Mouse orbit sensitivity", &state.orbitSensitivity, 0.15f, 3.0f, "%.2f");
    ImGui::SliderFloat("Camera distance", &state.cameraDistance, 0.01f, std::max(30.0f, mesh.boundsRadius * 20.0f), "%.3f");
    ImGui::SliderFloat("Pan speed", &state.panSpeed, 0.1f, 3.0f, "%.2f");
    ImGui::SliderFloat("Zoom speed", &state.zoomSpeed, 0.1f, 3.0f, "%.2f");
    ImGui::SliderFloat("Near clip", &state.nearClip, 0.0001f, std::max(0.25f, mesh.boundsRadius * 0.05f), "%.5f");
    ImGui::SliderFloat("Far clip", &state.farClip, 10.0f, std::max(1000.0f, mesh.boundsRadius * 100.0f), "%.1f");
    state.farClip = std::max(state.farClip, state.nearClip + 1.0f);
    ImGui::SliderFloat("Model pitch", &state.modelPitch, -3.14159f, 3.14159f, "%.2f");
    ImGui::SliderFloat("Model yaw", &state.modelYaw, -3.14159f, 3.14159f, "%.2f");
    ImGui::SliderFloat("Model roll", &state.modelRoll, -3.14159f, 3.14159f, "%.2f");
    if (ImGui::Button("Y-up model"))
    {
        state.modelPitch = 0.0f; state.modelYaw = 0.0f; state.modelRoll = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Z-up model"))
    {
        state.modelPitch = -DirectX::XM_PIDIV2; state.modelYaw = 0.0f; state.modelRoll = 0.0f;
    }

    if (ImGui::Button("Frame whole ship (F/Home)")) m_sceneController.FrameShip(state, mesh);
    ImGui::SameLine();
    if (ImGui::Button("Reset view (R)")) m_sceneController.ResetView(state, mesh);
    ImGui::SameLine();
    if (ImGui::Button("Save screenshot (F9)")) state.requestScreenshot = true;

    ImGui::Separator();
    ImGui::Text("Frame time: %.3f ms", 1000.0f / std::max(ImGui::GetIO().Framerate, 1.0f));
    ImGui::TextUnformatted("Mouse: RMB orbit | MMB or LMB+RMB pan | Shift+pan fine | wheel zoom-to-cursor | Ctrl+wheel fine zoom");
    ImGui::TextUnformatted("Double-click a hull panel to make it the orbit and zoom focus point.");
    ImGui::TextUnformatted("Keys: [ / ] backgrounds | F/Home frame ship | V flip UV | R reset | F9 capture | Esc exit");
    ImGui::TextWrapped("The comparison is fixed to A RAW SOURCE and B NSAMDR FINAL under one shared renderer and sampler.");

    ImGui::EndChild();

    const ImVec2 panelPosition = ImGui::GetWindowPos();
    const ImVec2 actualPanelSize = ImGui::GetWindowSize();
    const float desiredSceneX = panelPosition.x + actualPanelSize.x + 12.0f;
    state.sceneViewportX = static_cast<uint32_t>(ClampFloat(
        desiredSceneX,
        0.0f,
        resources.width > 320U ? static_cast<float>(resources.width - 320U) : 0.0f));

    ImGui::End();
}

void PreviewPanel::DrawSplitCompareOverlay(
    const PreviewState& state,
    const PreviewResources& resources,
    const FinalCandidateSet& candidates)
{
    const float sceneX = static_cast<float>(std::min(
        state.sceneViewportX,
        resources.width > 1U ? resources.width - 1U : 0U));
    const float sceneWidth = std::max(1.0f, static_cast<float>(resources.width) - sceneX);
    const float sceneHeight = std::max(1.0f, static_cast<float>(resources.height));
    ImDrawList* draw = ImGui::GetForegroundDrawList();

    auto addLabel = [&](const ImVec2& paneMin, const char* label)
    {
        const ImVec2 textPos(paneMin.x + 12.0f, paneMin.y + 12.0f);
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        draw->AddRectFilled(
            ImVec2(textPos.x - 6.0f, textPos.y - 4.0f),
            ImVec2(textPos.x + textSize.x + 6.0f, textPos.y + textSize.y + 4.0f),
            IM_COL32(0, 0, 0, 190),
            4.0f);
        draw->AddText(textPos, IM_COL32(255, 255, 255, 255), label);
    };

    ImVec2 firstMin(sceneX, 0.0f);
    ImVec2 firstMax(sceneX + sceneWidth, sceneHeight);
    ImVec2 secondMin = firstMin;
    ImVec2 secondMax = firstMax;
    ImVec2 dividerA;
    ImVec2 dividerB;
    if (state.splitVertical)
    {
        const float divider = sceneX + std::floor(sceneWidth * 0.5f);
        firstMax.x = divider;
        secondMin.x = divider;
        dividerA = ImVec2(divider, 0.0f);
        dividerB = ImVec2(divider, sceneHeight);
    }
    else
    {
        const float divider = std::floor(sceneHeight * 0.5f);
        firstMax.y = divider;
        secondMin.y = divider;
        dividerA = ImVec2(sceneX, divider);
        dividerB = ImVec2(sceneX + sceneWidth, divider);
    }

    const ImVec2 rawMin = state.swapSplitSides ? secondMin : firstMin;
    const ImVec2 finalMin = state.swapSplitSides ? firstMin : secondMin;
    const ImVec2 finalMax = state.swapSplitSides ? firstMax : secondMax;
    draw->AddLine(dividerA, dividerB, IM_COL32(255, 255, 255, 210), 2.0f);
    addLabel(rawMin, "A RAW SOURCE - 16x AF / LOD 0");
    addLabel(finalMin, "B NSAMDR FINAL - 16x AF / LOD 0");

    const CandidateAssetGpu& finalCandidate = candidates.candidate;
    if (!finalCandidate.available)
    {
        const char* warning = "B BLOCKED - immutable final provenance not verified";
        const ImVec2 warningPos(finalMin.x + 12.0f, finalMin.y + 48.0f);
        const ImVec2 warningSize = ImGui::CalcTextSize(warning);
        draw->AddRectFilled(finalMin, finalMax, IM_COL32(80, 0, 0, 78));
        draw->AddRectFilled(
            ImVec2(warningPos.x - 6.0f, warningPos.y - 4.0f),
            ImVec2(warningPos.x + warningSize.x + 6.0f, warningPos.y + warningSize.y + 4.0f),
            IM_COL32(130, 0, 0, 230),
            4.0f);
        draw->AddText(warningPos, IM_COL32(255, 235, 235, 255), warning);
        if (!finalCandidate.status.empty())
        {
            draw->AddText(
                ImVec2(finalMin.x + 12.0f, finalMin.y + 76.0f),
                IM_COL32(255, 210, 210, 255),
                finalCandidate.status.c_str());
        }
    }
}

std::string PreviewPanel::BuildScreenshotPath() const
{
    CreateDirectoryA("artifacts", nullptr);
    CreateDirectoryA("artifacts\\nsamdr", nullptr);
    CreateDirectoryA("artifacts\\nsamdr\\captures", nullptr);

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return "artifacts\\nsamdr\\captures\\raw_vs_final_" +
        std::to_string(milliseconds) + ".dds";
}

} // namespace nsamdr
