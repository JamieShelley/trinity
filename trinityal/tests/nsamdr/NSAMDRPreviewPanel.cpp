#include "StdAfx.h"
#include "NSAMDRPreviewPanel.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
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
    ImGui::TextWrapped("Three public modes only: Mode 1 is the original extracted source with no NSAMDR cleanup, Mode 2 is UV/stretch diagnostics, and Mode 3 is the NSAMDR cleanup candidate. Mode 3 uses trained V4 tile-context output when available and a deterministic bootstrap before the first training run.");

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
        ImGui::BulletText("1. Train a fully convolutional tile-context model on clean contours and synthetic degradation");
        ImGui::BulletText("2. Predict continuous source transport, RGB residual and confidence across a 125-pixel receptive field");
        ImGui::BulletText("3. Run overlapping 4K inference with seam-free tile blending before launch");
        ImGui::BulletText("4. Bake the reconstructed albedo into the Mode 3 material manifest");
        ImGui::BulletText("5. Sample original and reconstructed materials through the same live shader");
        ImGui::TreePop();
    }

    if (ImGui::CollapsingHeader("Mode 3 baked tile reconstruction", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextWrapped("Mode 3 is no longer corrected by a per-pixel runtime kernel. The trained V4 model processes overlapping material tiles during candidate generation, so the loaded 4K texture already contains the reconstruction.");
        ImGui::TextWrapped("Changing model behaviour requires retraining and regenerating the candidate; the comparison renderer itself applies no hidden cleanup.");
    }

    if (ImGui::CollapsingHeader("Offline retraining (closes and rebuilds preview)"))
    {
        ImGui::TextWrapped("These values are written to a training profile. The button launches a separate process, closes this preview, trains on NVIDIA CUDA or CPU, validates the checkpoint, regenerates the overlapping-tile Mode 3 textures, rebuilds TrinityALTest_dx11, and opens the new result.");
        ImGui::TextWrapped("Real textures provide identity examples. Synthetic clean targets teach long contour continuity, removal of stair-stepped seams, and preservation of authored material edges.");
        ImGui::InputInt("Epochs", &state.neuralTraining.epochs, 1, 5);
        ImGui::InputInt("Tiles per epoch", &state.neuralTraining.tilesPerEpoch, 128, 512);
        ImGui::InputInt("Batch size", &state.neuralTraining.batchSize, 1, 4);
        ImGui::InputInt("Base channels", &state.neuralTraining.baseChannels, 4, 8);
        ImGui::InputInt("Residual blocks", &state.neuralTraining.residualBlocks, 1, 1);
        ImGui::InputInt("Training tile size", &state.neuralTraining.tileSize, 8, 16);
        ImGui::InputInt("Maximum transport radius", &state.neuralTraining.maxOffsetPixels, 1, 1);
        ImGui::InputInt("Maximum source files", &state.neuralTraining.maxSourceFiles, 8, 32);
        ImGui::InputInt("Random seed", &state.neuralTraining.seed);
        ImGui::InputFloat("Learning rate", &state.neuralTraining.learningRate, 0.00005f, 0.0002f, "%.6f");
        ImGui::SliderFloat("Reconstruction loss weight", &state.neuralTraining.reconstructionWeight, 0.0f, 8.0f, "%.2f");
        ImGui::SliderFloat("Contour edge loss weight", &state.neuralTraining.edgeWeight, 0.0f, 8.0f, "%.2f");
        ImGui::SliderFloat("Flat-region identity weight", &state.neuralTraining.identityWeight, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Confidence loss weight", &state.neuralTraining.confidenceWeight, 0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Flow smoothness weight", &state.neuralTraining.flowSmoothnessWeight, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Maximum RGB residual", &state.neuralTraining.maxResidual, 0.02f, 0.50f, "%.3f");
        ImGui::SliderFloat("Synthetic artifact frequency", &state.neuralTraining.artifactFraction, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Real-source identity fraction", &state.neuralTraining.realSourceFraction, 0.0f, 0.80f, "%.2f");
        ImGui::InputTextWithHint(
            "Source texture root",
            "EVE SharedCache or extracted texture root",
            state.neuralTraining.sourceRoot.data(),
            state.neuralTraining.sourceRoot.size());

        state.neuralTraining.epochs = std::clamp(state.neuralTraining.epochs, 1, 300);
        state.neuralTraining.tilesPerEpoch = std::clamp(state.neuralTraining.tilesPerEpoch, 128, 1000000);
        state.neuralTraining.batchSize = std::clamp(state.neuralTraining.batchSize, 1, 128);
        state.neuralTraining.baseChannels = std::clamp(state.neuralTraining.baseChannels, 16, 96);
        state.neuralTraining.residualBlocks = std::clamp(state.neuralTraining.residualBlocks, 4, 8);
        state.neuralTraining.tileSize = std::clamp(state.neuralTraining.tileSize, 48, 256);
        state.neuralTraining.maxOffsetPixels = std::clamp(state.neuralTraining.maxOffsetPixels, 1, 16);
        state.neuralTraining.maxSourceFiles = std::clamp(state.neuralTraining.maxSourceFiles, 0, 20000);
        state.neuralTraining.learningRate = ClampFloat(state.neuralTraining.learningRate, 0.000001f, 0.01f);

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
    ImGui::TextUnformatted("Synchronized comparison");
    const int baselineMode = static_cast<int>(StrategyMode::OriginalBaseline);
    if (state.mode == baselineMode) ImGui::BeginDisabled();
    ImGui::Checkbox("Split original vs cleanup", &state.splitCompare);
    if (state.mode == baselineMode)
    {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Mode 1 is already the original source view.");
        ImGui::EndDisabled();
    }
    if (state.mode != baselineMode && state.splitCompare)
    {
        ImGui::Checkbox("Vertical split", &state.splitVertical);
        ImGui::SameLine();
        ImGui::Checkbox("Swap sides", &state.swapSplitSides);
        ImGui::SliderFloat("Divider", &state.splitPosition, 0.20f, 0.80f, "%.2f");
        ImGui::TextWrapped("Both panes use the same camera, transform, lighting and background.");
    }

    if (state.mode == static_cast<int>(StrategyMode::OriginalBaseline))
    {
        ImGui::TextWrapped("Mode 1 renders the original extracted source textures with no NSAMDR cleanup. It uses the same mesh, camera, lighting and material shader as Mode 3.");
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
            ImGui::TextWrapped("Mode 3 uses the prepared 4K cleanup candidate. After V4 training it contains overlapping tile-context reconstruction baked into its albedo; before training it remains a deterministic bootstrap candidate. Normal, material, roughness, paint, AO, dirt and glow remain deterministic.");
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
    ImGui::TextUnformatted("Keys: 1 original source | 2 diagnostics | 3 NSAMDR cleanup | Space split compare | [ / ] backgrounds | F/Home frame ship | V flip UV | R reset | F9 capture | Esc exit");
    ImGui::TextWrapped("Mode 1 and Mode 3 use the same geometry, camera, lighting, material shader and render state. The comparison changes only original source resources versus NSAMDR cleanup output.");

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
    ImDrawList* draw = ImGui::GetForegroundDrawList();
    draw->AddLine(dividerA, dividerB, IM_COL32(255, 255, 255, 210), 2.0f);

    auto addLabel = [&](const ImVec2& paneMin, const char* text)
    {
        const ImVec2 textPos(paneMin.x + 12.0f, paneMin.y + 12.0f);
        const ImVec2 textSize = ImGui::CalcTextSize(text);
        draw->AddRectFilled(
            ImVec2(textPos.x - 6.0f, textPos.y - 4.0f),
            ImVec2(textPos.x + textSize.x + 6.0f, textPos.y + textSize.y + 4.0f),
            IM_COL32(0, 0, 0, 180),
            4.0f);
        draw->AddText(textPos, IM_COL32(255, 255, 255, 255), text);
    };

    addLabel(baselineMin, m_strategyModes.Label(static_cast<int>(StrategyMode::OriginalBaseline)));
    addLabel(candidateMin, m_strategyModes.Label(state.mode));
    const CandidateAssetGpu* selectedCandidate = m_strategyModes.CandidateForMode(candidates, state.mode);
    if (selectedCandidate && !selectedCandidate->available)
    {
        const char* warning = "MODE 3 UNAVAILABLE - no Mode 1 fallback";
        const ImVec2 warningPos(candidateMin.x + 12.0f, candidateMin.y + 42.0f);
        const ImVec2 warningSize = ImGui::CalcTextSize(warning);
        draw->AddRectFilled(
            ImVec2(candidateMin.x, candidateMin.y),
            ImVec2(candidateMin.x + (state.splitVertical ? sceneWidth * (1.0f - split) : sceneWidth),
                   candidateMin.y + (state.splitVertical ? sceneHeight : sceneHeight * (1.0f - split))),
            IM_COL32(80, 0, 0, 72));
        draw->AddRectFilled(
            ImVec2(warningPos.x - 6.0f, warningPos.y - 4.0f),
            ImVec2(warningPos.x + warningSize.x + 6.0f, warningPos.y + warningSize.y + 4.0f),
            IM_COL32(130, 0, 0, 230),
            4.0f);
        draw->AddText(warningPos, IM_COL32(255, 235, 235, 255), warning);
        if (!selectedCandidate->status.empty())
        {
            const ImVec2 reasonPos(candidateMin.x + 12.0f, candidateMin.y + 70.0f);
            draw->AddText(reasonPos, IM_COL32(255, 210, 210, 255), selectedCandidate->status.c_str());
        }
    }
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
