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

void ApplyNSAMDRGraphicsPreset(PreviewState& state, NSAMDRQuality quality)
{
    state.nsamdrGraphics = ResolveNSAMDRGraphicsSettings(quality);
    switch (quality)
    {
    case NSAMDRQuality::Off:
        state.textureDetailReconstructionQuality = 0;
        break;
    case NSAMDRQuality::Balanced:
        state.textureDetailReconstructionQuality = 1;
        state.cleanupMasterStrength = 0.72f;
        state.cleanupFlatDenoiseStrength = 0.28f;
        state.cleanupDarkSeparationStrength = 0.20f;
        state.cleanupHighlightStrength = 0.22f;
        state.cleanupDetailSharpenStrength = 0.04f;
        state.cleanupDirectionalDeblockStrength = 0.38f;
        state.cleanupContourReconstructionStrength = 0.34f;
        state.cleanupThinLineStrength = 0.26f;
        state.cleanupEdgeOvershootLimit = 0.010f;
        break;
    case NSAMDRQuality::High:
        state.textureDetailReconstructionQuality = 2;
        state.cleanupMasterStrength = 1.0f;
        state.cleanupFlatDenoiseStrength = 0.40f;
        state.cleanupDarkSeparationStrength = 0.25f;
        state.cleanupHighlightStrength = 0.30f;
        state.cleanupDetailSharpenStrength = 0.08f;
        state.cleanupDirectionalDeblockStrength = 0.55f;
        state.cleanupContourReconstructionStrength = 0.48f;
        state.cleanupThinLineStrength = 0.36f;
        state.cleanupEdgeOvershootLimit = 0.012f;
        break;
    case NSAMDRQuality::Ultra:
        state.textureDetailReconstructionQuality = 3;
        state.cleanupMasterStrength = 1.0f;
        state.cleanupFlatDenoiseStrength = 0.48f;
        state.cleanupDarkSeparationStrength = 0.34f;
        state.cleanupHighlightStrength = 0.38f;
        state.cleanupDetailSharpenStrength = 0.10f;
        state.cleanupDirectionalDeblockStrength = 0.70f;
        state.cleanupContourReconstructionStrength = 0.66f;
        state.cleanupThinLineStrength = 0.52f;
        state.cleanupEdgeOvershootLimit = 0.008f;
        break;
    }
}
}

PreviewPanel::PreviewPanel(
    StrategyModes& strategyModes,
    NSAMDRPipeline& pipeline,
    NSAMDRTrainingController& trainingController,
    SceneController& sceneController)
    : m_strategyModes(strategyModes),
      m_pipeline(pipeline),
      m_trainingController(trainingController),
      m_sceneController(sceneController)
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
    const StrategyCandidateSet& candidates,
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
    ImGui::TextWrapped("Three public modes only: Mode 1 is the untouched extracted source control rendered at 16x AF / LOD 0, Mode 2 is UV/stretch diagnostics, and Mode 3 is the V9.8.2 metric-SDF geometry-convergence CUDA 4x candidate. Mode 3 scientific comparison adds a separately labelled legacy-sampler context pane.");

    // Keep all three public mode controls outside the scrolling content. Mode 3
    // must remain visible and selectable even when the candidate is unavailable.
    ImGui::Separator();
    ImGui::TextUnformatted("Strategy mode");
    for (const StrategyDescriptor& descriptor : m_strategyModes.Registry())
    {
        const int mode = static_cast<int>(descriptor.mode);
        if (ImGui::RadioButton(descriptor.panelLabel, state.mode == mode))
        {
            m_strategyModes.Select(state, mode);
        }
    }
    ImGui::Separator();
    ImGui::TextUnformatted("Display & Graphics policy");
    static const char* nsamdrQualityLabels[] = { "Off", "Balanced", "High", "Ultra" };
    int nsamdrQualityIndex = static_cast<int>(state.nsamdrGraphics.quality);
    if (ImGui::Combo(
            "Neural Surface Reconstruction",
            &nsamdrQualityIndex,
            nsamdrQualityLabels,
            IM_ARRAYSIZE(nsamdrQualityLabels)))
    {
        const NSAMDRQuality selectedQuality = static_cast<NSAMDRQuality>(
            std::clamp(nsamdrQualityIndex, 0, 3));
        ApplyNSAMDRGraphicsPreset(state, selectedQuality);
        if (selectedQuality == NSAMDRQuality::Off)
            m_strategyModes.Select(state, static_cast<int>(StrategyMode::OriginalBaseline));
        else
            m_strategyModes.Select(state, static_cast<int>(StrategyMode::NeuralReconstruction));
    }
    const NSAMDRGraphicsSettings& graphicsPolicy = state.nsamdrGraphics;
    ImGui::TextDisabled(
        "%s | %ux target | ships=%s | structures=%s | cache %.0f GiB",
        NSAMDRQualityName(graphicsPolicy.quality),
        graphicsPolicy.targetScale,
        graphicsPolicy.reconstructShips ? "on" : "off",
        graphicsPolicy.reconstructStructures ? "on" : "off",
        static_cast<double>(graphicsPolicy.cacheBudgetBytes) / (1024.0 * 1024.0 * 1024.0));
    ImGui::TextWrapped("Functional Trinity developer scaffold for the planned EVE Display & Graphics setting. Off returns to the verified raw source; Balanced/High/Ultra select Mode 3 and apply matching live cleanup presets. The displayed target-scale/cache/confidence values are the client policy contract; this test viewer still uses the already-baked candidate and does not yet own the production cache/job service or EVE settings screen.");
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
            ImGui::TextWrapped("Visual baseline: INCOMPLETE (%d unresolved inputs). Modes remain available for diagnosis.", resources.baselineUnresolvedCount);
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
        ImGui::TextUnformatted("Materials: legacy global fallback");
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
    ImGui::TextUnformatted("Mode 3 NSAMDR candidate");
    const CandidateAssetGpu& nsamdrCandidate = candidates.At(
        static_cast<int>(StrategyMode::NeuralReconstruction));
    if (m_pipeline.IsConfigured(candidates))
    {
        if (nsamdrCandidate.available)
        {
            ImGui::Text("Loaded | %u x %u | %u triangles | %zu draws",
                nsamdrCandidate.maximumTextureWidth,
                nsamdrCandidate.maximumTextureHeight,
                nsamdrCandidate.mesh.triangleCount,
                nsamdrCandidate.areaMaterials.size());
        }
        else
        {
            ImGui::TextWrapped("UNAVAILABLE - %s", nsamdrCandidate.status.c_str());
        }
    }
    else
    {
        ImGui::TextUnformatted("Not configured. Run the candidate-generation/build workflow.");
    }
    if (ImGui::TreeNode("Mode 3 pipeline"))
    {
        ImGui::TextWrapped("%s", m_pipeline.Summary());
        ImGui::Text("Configured: %s", m_pipeline.IsConfigured(candidates) ? "yes" : "no");
        ImGui::Text("Candidate ready: %s", m_pipeline.IsReady(candidates) ? "yes" : "no");
        ImGui::BulletText("1. Learn a coordinate-conditioned continuous SDF from clean analytic and EVE-like PBR contours");
        ImGui::BulletText("2. Stage A proves the deterministic renderer with exact SDF + forced gate");
        ImGui::BulletText("3. Stage B proves predicted SDF with the gate still forced; Stage C enables the predicted gate");
        ImGui::BulletText("4. Rebuild one shared zero-set across albedo, normal and material semantics without RGB geometry authority");
        ImGui::BulletText("5. Run overlapping FP16 CUDA inference, verify provenance, audit fuzz/halo/topology, then bake the candidate");
        ImGui::TreePop();
    }

    if (ImGui::CollapsingHeader("Mode 3 developer cleanup tuning", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static const char* qualityLabels[] = { "Disabled", "Balanced", "High", "Ultra" };
        ImGui::TextWrapped("Developer-only post-reconstruction cleanup. The public Off/Balanced/High/Ultra policy is above; these controls expose the live shader pass used while evaluating the baked Mode 3 candidate.");
        ImGui::Combo(
            "Developer cleanup level",
            &state.textureDetailReconstructionQuality,
            qualityLabels,
            IM_ARRAYSIZE(qualityLabels));

        if (state.textureDetailReconstructionQuality <= 0)
        {
            ImGui::TextUnformatted("Disabled: the baked Mode 3 candidate is displayed unchanged.");
        }
        else
        {
            if (state.textureDetailReconstructionQuality == 1)
            {
                ImGui::TextWrapped("Balanced: restrained cleanup with conservative edge authority.");
            }
            else if (state.textureDetailReconstructionQuality == 2)
            {
                ImGui::TextWrapped("High: stronger directional deblocking, bounded contour reconstruction and thin-line continuity repair.");
            }
            else
            {
                ImGui::TextWrapped("Ultra: maximum developer cleanup preset with tighter overshoot bounds. Neural geometry still comes from the baked V9.8 candidate; this live pass is not a substitute for SDF reconstruction.");
            }
            ImGui::SliderFloat("Cleanup master strength", &state.cleanupMasterStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Flat-region denoise", &state.cleanupFlatDenoiseStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Dark-panel separation", &state.cleanupDarkSeparationStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Highlight cleanup", &state.cleanupHighlightStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Structure-limited sharpening", &state.cleanupDetailSharpenStrength, 0.0f, 0.5f, "%.2f");
            if (state.textureDetailReconstructionQuality >= 2)
            {
                ImGui::Separator();
                ImGui::TextUnformatted("Directional reconstruction");
                ImGui::SliderFloat("Directional deblocking", &state.cleanupDirectionalDeblockStrength, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Contour reconstruction", &state.cleanupContourReconstructionStrength, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Thin-line clarity", &state.cleanupThinLineStrength, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Edge overshoot limit", &state.cleanupEdgeOvershootLimit, 0.0f, 0.03f, "%.3f");
                ImGui::TextWrapped("The edge normal is estimated from a 3x3 Sobel field. Samples are blended along the line tangent, then contrast is restored across the line normal. Diagonal stair steps receive the strongest continuity correction.");
            }
        }

        static const char* debugLabels[] = {
            "Final cleanup",
            "Current Mode 3",
            "Luminance edges",
            "Normal edges",
            "Combined structure mask",
            "Flat-region confidence",
            "Highlight confidence",
            "Dark-region confidence",
            "Cleanup difference",
            "Edge direction",
            "Stair-step confidence",
            "Thin-line confidence",
            "Directional correction"
        };
        ImGui::Combo("Cleanup diagnostic", &state.cleanupDebugView, debugLabels, IM_ARRAYSIZE(debugLabels));

        if (ImGui::Button("Reset Mode 3 cleanup defaults"))
        {
            ApplyNSAMDRGraphicsPreset(state, NSAMDRQuality::High);
            state.cleanupDebugView = 0;
        }
    }

    if (ImGui::CollapsingHeader("Mode 3 CUDA neural reconstruction"))
    {
        ImGui::TextWrapped("The trained V9.8.2 metric-SDF geometry-convergence SDF network predicts continuous contour geometry from overlapping low-resolution PBR tiles during candidate generation. A deterministic shared BoundaryRenderer then rebuilds the same contour across albedo, normal and material semantics. The source control remains immutable and is SHA-256 verified before preview.");
        ImGui::TextWrapped("Changing neural behaviour still requires retraining and regenerating the candidate. Cleanup parameters can be tuned live without rebuilding the asset.");
    }

    if (ImGui::CollapsingHeader("Offline retraining (closes and rebuilds preview)"))
    {
        ImGui::TextWrapped("These values are written to a training profile. The button launches a separate process, closes this preview, trains on NVIDIA CUDA, validates the V9 checkpoint, executes direct 4x FP16 CUDA multi-head inference over overlapping tiles, rebuilds TrinityALTest_dx11, and opens the new result.");
        ImGui::TextWrapped("Real EVE cache crops are actively degraded and paired with their authored high-resolution targets. Authored normals and any available packed companions guide panel edges, bevels and material response. Missing packed maps are masked rather than fabricated, and families are split before crop generation to prevent validation leakage.");
        ImGui::TextWrapped("V9 trains a fidelity-first 4x reconstruction from the soft client-like input to the highest authored EVE cache texture. Baseline-anchored residual branches, local regret, contour supervision, cross-map alignment and anti-ringing losses constrain the learned correction.");
        ImGui::TextUnformatted("Training phases");
        ImGui::InputInt("Contour bootstrap epochs", &state.neuralTraining.albedoBootstrapEpochs, 1, 2);
        ImGui::InputInt("Multi-map reconstruction epochs", &state.neuralTraining.jointPbrEpochs, 1, 3);
        ImGui::InputInt("Physical fine-tune epochs", &state.neuralTraining.renderFinetuneEpochs, 1, 2);
        ImGui::InputInt("Tiles per epoch", &state.neuralTraining.tilesPerEpoch, 128, 512);
        ImGui::InputInt("Validation tiles", &state.neuralTraining.validationTiles, 16, 64);
        ImGui::TextWrapped("Batch size: 1 (fixed for the 128x128 to 512x512 production memory profile)");

        ImGui::TextUnformatted("EVE SharedCache dataset");
        ImGui::InputInt("Maximum authored texture families", &state.neuralTraining.maxFamilies, 8, 32);
        ImGui::InputInt("High-detail crops per family", &state.neuralTraining.cropsPerFamily, 2, 4);
        ImGui::InputInt("Stored authored-HR crop size", &state.neuralTraining.sourceCropSize, 32, 64);
        ImGui::InputInt("Minimum source dimension", &state.neuralTraining.minSourceDimension, 256, 512);
        ImGui::InputTextWithHint(
            "EVE SharedCache root",
            "blank = auto-detect; otherwise select the SharedCache directory",
            state.neuralTraining.sourceRoot.data(),
            state.neuralTraining.sourceRoot.size());

        ImGui::TextUnformatted("V9 fixed architecture");
        ImGui::TextWrapped("Widths: 96 / 160 / 256 / 384 / 512; attention at 16x16 and 8x8");
        ImGui::TextWrapped("Tile mapping: 128x128 LR -> 512x512 HR (4x)");
        ImGui::InputInt("Random seed", &state.neuralTraining.seed);
        ImGui::InputFloat("Main learning rate", &state.neuralTraining.learningRate, 0.00002f, 0.0001f, "%.6f");
        ImGui::InputFloat("Render fine-tune learning rate", &state.neuralTraining.renderFinetuneLearningRate, 0.00001f, 0.00005f, "%.6f");

        ImGui::TextUnformatted("V9.8.2 metric-SDF geometry-convergence losses");
        ImGui::SliderFloat("Albedo reconstruction", &state.neuralTraining.albedoWeight, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Normal reconstruction", &state.neuralTraining.normalWeight, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Packed material reconstruction", &state.neuralTraining.materialWeight, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Contour edge supervision", &state.neuralTraining.edgeWeight, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Boundary orientation", &state.neuralTraining.orientationWeight, 0.0f, 2.0f, "%.2f");
        ImGui::TextWrapped("Contour SDF weight is fixed at 1.20");
        ImGui::TextWrapped("Cross-map alignment weight is fixed at 0.45");
        ImGui::TextWrapped("Seam consistency weight is fixed at 0.15");
        ImGui::TextWrapped("Presentation rendering is intentionally excluded from V9.8 geometry training");
        ImGui::SliderFloat("Confidence supervision", &state.neuralTraining.confidenceWeight, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Ungated proposal learning", &state.neuralTraining.proposalWeight, 0.0f, 2.0f, "%.2f");
        ImGui::TextWrapped("Boundary-focused crop sampling: best of 8 candidates, 78% probability");
        ImGui::TextWrapped("Full-map proposals replace bounded local residual assumptions");

        state.neuralTraining.albedoBootstrapEpochs = std::clamp(state.neuralTraining.albedoBootstrapEpochs, 1, 100);
        state.neuralTraining.jointPbrEpochs = std::clamp(state.neuralTraining.jointPbrEpochs, 1, 200);
        state.neuralTraining.renderFinetuneEpochs = std::clamp(state.neuralTraining.renderFinetuneEpochs, 0, 100);
        state.neuralTraining.tilesPerEpoch = std::clamp(state.neuralTraining.tilesPerEpoch, 128, 1000000);
        state.neuralTraining.validationTiles = std::clamp(state.neuralTraining.validationTiles, 16, 8192);
        state.neuralTraining.batchSize = std::clamp(state.neuralTraining.batchSize, 1, 32);
        state.neuralTraining.maxFamilies = std::clamp(state.neuralTraining.maxFamilies, 8, 4096);
        state.neuralTraining.cropsPerFamily = std::clamp(state.neuralTraining.cropsPerFamily, 2, 128);
        state.neuralTraining.sourceCropSize = std::clamp(state.neuralTraining.sourceCropSize, 512, 2048);
        state.neuralTraining.minSourceDimension = std::clamp(state.neuralTraining.minSourceDimension, 256, 16384);
        state.neuralTraining.baseChannels = 96;
        state.neuralTraining.tileSize = 128;
        state.neuralTraining.learningRate = ClampFloat(state.neuralTraining.learningRate, 0.000001f, 0.01f);
        state.neuralTraining.renderFinetuneLearningRate = ClampFloat(state.neuralTraining.renderFinetuneLearningRate, 0.000001f, 0.01f);

        if (ImGui::Button("Retrain, rebuild and reopen preview"))
        {
            std::string launchError;
            if (m_trainingController.LaunchRetrainBuildPreview(state.neuralTraining, launchError))
            {
                PostMessage(hwnd, WM_CLOSE, 0, 0);
            }
            else
            {
                state.neuralTraining.status = launchError;
            }
        }
        ImGui::TextWrapped("%s", state.neuralTraining.status.c_str());
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Scientific comparison");
    const int baselineMode = static_cast<int>(StrategyMode::OriginalBaseline);
    const bool neuralMode = state.mode == static_cast<int>(StrategyMode::NeuralReconstruction);
    if (state.mode == baselineMode) ImGui::BeginDisabled();
    ImGui::Checkbox(
        neuralMode ? "Three-pane raw / legacy / NSAMDR proof" : "Split source vs selected mode",
        &state.splitCompare);
    if (state.mode == baselineMode)
    {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Mode 1 is already the untouched source control.");
        ImGui::EndDisabled();
    }
    if (state.mode != baselineMode && state.splitCompare)
    {
        ImGui::Checkbox("Vertical panes", &state.splitVertical);
        ImGui::SameLine();
        ImGui::Checkbox("Swap raw and NSAMDR", &state.swapSplitSides);
        if (!neuralMode)
        {
            ImGui::SliderFloat("Divider", &state.splitPosition, 0.20f, 0.80f, "%.2f");
            ImGui::Checkbox("Legacy EVE-like sampling on source pane", &state.emulateLegacyEveBaseline);
        }
        ImGui::Checkbox("Pane identity tint (proof)", &state.verifyPaneIdentity);
        if (neuralMode)
        {
            ImGui::TextWrapped("A RAW CONTROL: untouched extracted source, 16x AF, LOD 0. B LEGACY EMULATION: the same untouched source, 2x AF, +1 LOD. C NSAMDR: reconstructed candidate, 16x AF, LOD 0. A vs C is the authoritative neural comparison; B is presentation context only.");
        }
        else if (state.emulateLegacyEveBaseline)
        {
            ImGui::TextWrapped("Source pane uses 2x anisotropic filtering with +1.00 mip LOD bias; selected mode uses 16x anisotropic filtering with zero bias.");
        }
        else
        {
            ImGui::TextWrapped("Both panes use the high-quality 16x sampler with zero LOD bias.");
        }
        ImGui::TextWrapped("All panes use the same camera, transform, lighting, environment, geometry and material shader.");

        const AreaMaterialGpu* baselineProofMaterial = nullptr;
        const AreaMaterialGpu* candidateProofMaterial = nullptr;
        if (nsamdrCandidate.available)
        {
            for (const AreaMaterialGpu& baselineMaterial : resources.areaMaterials)
            {
                if (!baselineMaterial.hasAlbedo) continue;
                const auto candidateIt = std::find_if(
                    nsamdrCandidate.areaMaterials.begin(),
                    nsamdrCandidate.areaMaterials.end(),
                    [&](const AreaMaterialGpu& candidateMaterial) {
                        return candidateMaterial.hasAlbedo &&
                            candidateMaterial.source.groupIndex == baselineMaterial.source.groupIndex;
                    });
                if (candidateIt != nsamdrCandidate.areaMaterials.end())
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
                "A/B albedo resource isolation: %s",
                isolated ? "PASS" : "FAIL");
            ImGui::TextWrapped("Left path: %s", baselineProofMaterial->source.albedoPath.c_str());
            ImGui::Text("Left SRV:  %p", static_cast<void*>(baselineProofMaterial->albedoView.Get()));
            ImGui::TextWrapped("Right path: %s", candidateProofMaterial->source.albedoPath.c_str());
            ImGui::Text("Right SRV: %p", static_cast<void*>(candidateProofMaterial->albedoView.Get()));
        }
        else
        {
            ImGui::TextDisabled("A/B albedo isolation proof unavailable until matching baseline and Mode 3 material groups are loaded.");
        }

        if (neuralMode)
        {
            const std::string provenanceStatus = ReadEnvironmentVariable("NSAMDR_PROVENANCE_STATUS");
            const bool provenanceVerified = provenanceStatus == "VERIFIED";
            ImGui::TextColored(
                provenanceVerified ? ImVec4(0.35f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.28f, 0.22f, 1.0f),
                "Raw-control provenance: %s",
                provenanceVerified ? "VERIFIED" : "FAILED / MISSING");
            const std::string provenanceSource = ReadEnvironmentVariable("NSAMDR_PROVENANCE_SOURCE");
            const std::string provenanceSourceSha = ReadEnvironmentVariable("NSAMDR_PROVENANCE_SOURCE_SHA");
            const std::string provenanceCandidate = ReadEnvironmentVariable("NSAMDR_PROVENANCE_CANDIDATE");
            const std::string provenanceCandidateSha = ReadEnvironmentVariable("NSAMDR_PROVENANCE_CANDIDATE_SHA");
            const std::string provenanceFile = ReadEnvironmentVariable("NSAMDR_PROVENANCE_FILE");
            if (!provenanceSource.empty())
                ImGui::TextWrapped("Raw source: %s", provenanceSource.c_str());
            if (!provenanceSourceSha.empty())
                ImGui::TextWrapped("Raw SHA-256: %.16s...", provenanceSourceSha.c_str());
            if (!provenanceCandidate.empty())
                ImGui::TextWrapped("Candidate: %s", provenanceCandidate.c_str());
            if (!provenanceCandidateSha.empty())
                ImGui::TextWrapped("Candidate SHA-256: %.16s...", provenanceCandidateSha.c_str());
            if (baselineProofMaterial && !provenanceSource.empty())
            {
                const bool rawBindingMatches =
                    _stricmp(baselineProofMaterial->source.albedoPath.c_str(), provenanceSource.c_str()) == 0;
                ImGui::TextColored(
                    rawBindingMatches ? ImVec4(0.35f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.28f, 0.22f, 1.0f),
                    "Raw pane -> proven source binding: %s",
                    rawBindingMatches ? "PASS" : "FAIL");
            }
            if (candidateProofMaterial && !provenanceCandidate.empty())
            {
                const bool candidateBindingMatches =
                    _stricmp(candidateProofMaterial->source.albedoPath.c_str(), provenanceCandidate.c_str()) == 0;
                ImGui::TextColored(
                    candidateBindingMatches ? ImVec4(0.35f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.28f, 0.22f, 1.0f),
                    "NSAMDR pane -> proven candidate binding: %s",
                    candidateBindingMatches ? "PASS" : "FAIL");
            }
            if (!provenanceFile.empty())
                ImGui::TextWrapped("Provenance evidence: %s", provenanceFile.c_str());
        }
    }

    if (state.mode == static_cast<int>(StrategyMode::OriginalBaseline))
    {
        ImGui::TextWrapped("Mode 1 is the untouched extracted source rendered with the high-quality 16x sampler at zero LOD bias. Legacy sampling is shown separately in the Mode 3 scientific three-pane comparison.");
    }
    else if (state.mode == static_cast<int>(StrategyMode::UvStretchDiagnostic))
    {
        ImGui::SliderFloat("Diagnostic checker scale", &state.diagnosticCheckerScale, 2.0f, 64.0f, "%.1f");
        ImGui::TextWrapped("Mode 2 visualises UV texel density, principal stretch direction, anisotropy and mip pressure. It does not alter the asset.");
    }
    else if (state.mode == static_cast<int>(StrategyMode::NeuralReconstruction))
    {
        if (nsamdrCandidate.available)
        {
            ImGui::TextWrapped("Mode 3 uses the prepared V9.8.2 4K geometry-reconstruction candidate. The learned continuous boundary field drives one shared deterministic BoundaryRenderer for albedo, normal and material semantics so edge geometry remains spatially aligned. Missing checkpoints or failed candidate/provenance validation make Mode 3 unavailable rather than silently falling back to the source.");
            ImGui::TextWrapped("OBJ: %s", nsamdrCandidate.objPath.c_str());
            ImGui::TextWrapped("Materials: %s", nsamdrCandidate.materialManifestPath.c_str());
        }
        else
        {
            ImGui::TextWrapped("Mode 3 candidate is unavailable: %s", nsamdrCandidate.status.c_str());
        }
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
    ImGui::TextUnformatted("Keys: 1 raw source | 2 diagnostics | 3 NSAMDR | Space scientific compare | [ / ] backgrounds | F/Home frame ship | V flip UV | R reset | F9 capture | Esc exit");
    ImGui::TextWrapped("Mode 1 and Mode 3 use the same geometry, camera, lighting, environment and material shader. In the scientific three-pane view, raw source and NSAMDR use identical 16x/LOD0 sampling; the legacy pane is explicitly non-authoritative emulation.");

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
    const StrategyCandidateSet& candidates)
{
    if (state.mode == static_cast<int>(StrategyMode::OriginalBaseline) || !state.splitCompare) return;

    const float sceneX = static_cast<float>(std::min(
        state.sceneViewportX,
        resources.width > 1U ? resources.width - 1U : 0U));
    const float sceneWidth = std::max(1.0f, static_cast<float>(resources.width) - sceneX);
    const float sceneHeight = std::max(1.0f, static_cast<float>(resources.height));
    const bool neuralMode = state.mode == static_cast<int>(StrategyMode::NeuralReconstruction);
    ImDrawList* draw = ImGui::GetForegroundDrawList();

    auto addLabel = [&](const ImVec2& paneMin, const char* text)
    {
        const ImVec2 textPos(paneMin.x + 12.0f, paneMin.y + 12.0f);
        const ImVec2 textSize = ImGui::CalcTextSize(text);
        draw->AddRectFilled(
            ImVec2(textPos.x - 6.0f, textPos.y - 4.0f),
            ImVec2(textPos.x + textSize.x + 6.0f, textPos.y + textSize.y + 4.0f),
            IM_COL32(0, 0, 0, 190),
            4.0f);
        draw->AddText(textPos, IM_COL32(255, 255, 255, 255), text);
    };

    if (neuralMode)
    {
        // Match NSAMDRRenderPipeline.cpp exactly: equal thirds, with only raw and
        // candidate panes swapped. The legacy control stays in the centre.
        ImVec2 firstMin(sceneX, 0.0f), firstMax(sceneX + sceneWidth, sceneHeight);
        ImVec2 secondMin = firstMin, secondMax = firstMax;
        ImVec2 thirdMin = firstMin, thirdMax = firstMax;
        if (state.splitVertical)
        {
            const float firstWidth = std::floor(sceneWidth / 3.0f);
            const float secondWidth = std::floor((sceneWidth - firstWidth) / 2.0f);
            firstMax.x = sceneX + firstWidth;
            secondMin.x = firstMax.x;
            secondMax.x = secondMin.x + secondWidth;
            thirdMin.x = secondMax.x;
            draw->AddLine(ImVec2(firstMax.x, 0.0f), ImVec2(firstMax.x, sceneHeight), IM_COL32(255,255,255,210), 2.0f);
            draw->AddLine(ImVec2(secondMax.x, 0.0f), ImVec2(secondMax.x, sceneHeight), IM_COL32(255,255,255,210), 2.0f);
        }
        else
        {
            const float firstHeight = std::floor(sceneHeight / 3.0f);
            const float secondHeight = std::floor((sceneHeight - firstHeight) / 2.0f);
            firstMax.y = firstHeight;
            secondMin.y = firstMax.y;
            secondMax.y = secondMin.y + secondHeight;
            thirdMin.y = secondMax.y;
            draw->AddLine(ImVec2(sceneX, firstMax.y), ImVec2(sceneX + sceneWidth, firstMax.y), IM_COL32(255,255,255,210), 2.0f);
            draw->AddLine(ImVec2(sceneX, secondMax.y), ImVec2(sceneX + sceneWidth, secondMax.y), IM_COL32(255,255,255,210), 2.0f);
        }

        const ImVec2 rawMin = state.swapSplitSides ? thirdMin : firstMin;
        const ImVec2 candidateMin = state.swapSplitSides ? firstMin : thirdMin;
        const ImVec2 legacyMin = secondMin;
        const ImVec2 candidateMax = state.swapSplitSides ? firstMax : thirdMax;

        addLabel(rawMin, "A RAW CONTROL - source / 16x AF / LOD 0");
        addLabel(legacyMin, "B LEGACY EMULATION - same source / 2x AF / +1 LOD");
        addLabel(candidateMin, "C NSAMDR V9.8.2 - candidate / 16x AF / LOD 0");

        const std::string provenanceStatus = ReadEnvironmentVariable("NSAMDR_PROVENANCE_STATUS");
        const bool provenanceVerified = provenanceStatus == "VERIFIED";
        addLabel(
            ImVec2(rawMin.x, rawMin.y + 34.0f),
            provenanceVerified ? "SOURCE SHA-256 PROVENANCE: VERIFIED" : "SOURCE SHA-256 PROVENANCE: FAILED / MISSING");

        const CandidateAssetGpu* selectedCandidate = m_strategyModes.CandidateForMode(candidates, state.mode);
        if (selectedCandidate && !selectedCandidate->available)
        {
            const char* warning = "MODE 3 UNAVAILABLE - no source fallback";
            const ImVec2 warningPos(candidateMin.x + 12.0f, candidateMin.y + 48.0f);
            const ImVec2 warningSize = ImGui::CalcTextSize(warning);
            draw->AddRectFilled(candidateMin, candidateMax, IM_COL32(80, 0, 0, 78));
            draw->AddRectFilled(
                ImVec2(warningPos.x - 6.0f, warningPos.y - 4.0f),
                ImVec2(warningPos.x + warningSize.x + 6.0f, warningPos.y + warningSize.y + 4.0f),
                IM_COL32(130, 0, 0, 230),
                4.0f);
            draw->AddText(warningPos, IM_COL32(255, 235, 235, 255), warning);
            if (!selectedCandidate->status.empty())
            {
                const ImVec2 reasonPos(candidateMin.x + 12.0f, candidateMin.y + 76.0f);
                draw->AddText(reasonPos, IM_COL32(255, 210, 210, 255), selectedCandidate->status.c_str());
            }
        }
        return;
    }

    // Non-neural diagnostic modes retain the original two-pane split UI.
    const float split = std::clamp(state.splitPosition, 0.20f, 0.80f);
    ImVec2 firstMin(sceneX, 0.0f);
    ImVec2 firstMax(sceneX + sceneWidth, sceneHeight);
    ImVec2 secondMin = firstMin;
    ImVec2 secondMax = firstMax;
    ImVec2 dividerA;
    ImVec2 dividerB;
    if (state.splitVertical)
    {
        const float divider = sceneX + sceneWidth * split;
        firstMax.x = divider;
        secondMin.x = divider;
        dividerA = ImVec2(divider, 0.0f);
        dividerB = ImVec2(divider, sceneHeight);
    }
    else
    {
        const float divider = sceneHeight * split;
        firstMax.y = divider;
        secondMin.y = divider;
        dividerA = ImVec2(sceneX, divider);
        dividerB = ImVec2(sceneX + sceneWidth, divider);
    }

    const ImVec2 baselineMin = state.swapSplitSides ? secondMin : firstMin;
    const ImVec2 candidateMin = state.swapSplitSides ? firstMin : secondMin;
    draw->AddLine(dividerA, dividerB, IM_COL32(255, 255, 255, 210), 2.0f);
    addLabel(baselineMin, "RAW SOURCE CONTROL");
    if (state.emulateLegacyEveBaseline)
        addLabel(ImVec2(baselineMin.x, baselineMin.y + 32.0f), "Legacy sampler: 2x AF, +1.00 LOD");
    addLabel(candidateMin, m_strategyModes.Label(state.mode));
}


std::string PreviewPanel::BuildScreenshotPath(const PreviewState& state) const
{
    CreateDirectoryA("artifacts", nullptr);
    CreateDirectoryA("artifacts\\nsamdr", nullptr);
    CreateDirectoryA("artifacts\\nsamdr\\captures", nullptr);

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return "artifacts\\nsamdr\\captures\\mode_" + std::to_string(state.mode) +
        "_repair_" + std::to_string(state.repairMethod) +
        "_" + std::to_string(milliseconds) + ".dds";
}

} // namespace nsamdr
