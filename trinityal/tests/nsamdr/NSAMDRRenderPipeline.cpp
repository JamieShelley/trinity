#include "StdAfx.h"
#include "NSAMDRRenderPipeline.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
RenderPipeline::RenderPipeline(CameraController& cameraController)
    : m_cameraController(cameraController)
{
}

bool RenderPipeline::UploadSceneConstants(ID3D11DeviceContext* context, ID3D11Buffer* constantBuffer, const SceneConstants& constants)
{
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context->Map(constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
    {
        ADD_FAILURE() << "Failed to update NSAMDR OBJ preview constants";
        return false;
    }
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    context->Unmap(constantBuffer, 0);
    return true;
}

bool RenderPipeline::UpdateSceneConstants(
    ID3D11DeviceContext* context,
    ID3D11Buffer* constantBuffer,
    const PreviewResources& resources,
    const PreviewState& state,
    float elapsedSeconds,
    SceneConstants& outputConstants)
{
    XMMATRIX world, view, projection;
    XMFLOAT3 eye, target;
    const uint32_t viewportX = std::min(state.sceneViewportX, resources.width > 1U ? resources.width - 1U : 0U);
    const uint32_t viewportWidth = std::max(1U, resources.width - viewportX);
    m_cameraController.BuildCameraMatrices(state, viewportWidth, resources.height, world, view, projection, eye, target);

    const auto travelDirection = [](float yaw, float pitch) {
        const float cosPitch = std::cos(pitch);
        return XMFLOAT3(
            std::sin(yaw) * cosPitch,
            std::sin(pitch),
            -std::cos(yaw) * cosPitch);
    };
    const XMFLOAT3 key = travelDirection(state.keyYaw, state.keyPitch);
    const XMFLOAT3 fill = travelDirection(state.keyYaw + 2.30f, 0.12f);
    const XMFLOAT3 rim = travelDirection(state.keyYaw + 3.05f, -0.28f);

    SceneConstants constants{};
    DirectX::XMStoreFloat4x4(&constants.world, world);
    DirectX::XMStoreFloat4x4(&constants.viewProjection, view * projection);
    constants.cameraTime = XMFLOAT4(eye.x, eye.y, eye.z, elapsedSeconds);
    constants.keyLight = XMFLOAT4(key.x, key.y, key.z, state.keyIntensity);
    constants.fillLight = XMFLOAT4(fill.x, fill.y, fill.z, state.fillIntensity);
    constants.rimLight = XMFLOAT4(rim.x, rim.y, rim.z, state.rimIntensity);
    constants.material = XMFLOAT4(
        state.useTexture && resources.hasExternalAlbedo ? 1.0f : 0.0f,
        state.flipV ? 1.0f : 0.0f,
        state.exposure,
        state.ambient);
    constants.surface = XMFLOAT4(
        0.0f,
        state.normalMapStrength,
        state.specularStrength,
        state.roughnessBias);
    constants.options = XMFLOAT4(
        state.useNormalMap && resources.hasNormalMap ? 1.0f : 0.0f,
        state.usePgsMap && resources.hasPgsMap ? 1.0f : 0.0f,
        0.0f,
        0.0f);

    XMFLOAT3 cameraRight, cameraUp, cameraForward;
    m_cameraController.GetCameraBasis(state, cameraRight, cameraUp, cameraForward);
    const float aspect = resources.height == 0U ? 1.0f :
        static_cast<float>(viewportWidth) / static_cast<float>(resources.height);
    const float tanHalfFov = std::tan(DirectX::XMConvertToRadians(48.0f) * 0.5f);
    constants.cameraRight = XMFLOAT4(cameraRight.x, cameraRight.y, cameraRight.z, aspect * tanHalfFov);
    constants.cameraUp = XMFLOAT4(cameraUp.x, cameraUp.y, cameraUp.z, tanHalfFov);
    constants.cameraForward = XMFLOAT4(cameraForward.x, cameraForward.y, cameraForward.z, 0.0f);
    constants.environment = XMFLOAT4(
        state.useEnvironment && SelectedEnvironment(resources, state) != nullptr ? 1.0f : 0.0f,
        state.environmentIntensity,
        state.backgroundIntensity,
        state.reflectionStrength);
    constants.areaTint = XMFLOAT4(0.045f, 0.105f, 0.145f, 0.0f);
    constants.areaSurface = XMFLOAT4(0.62f, 0.72f, 1.0f, 0.0f);
    constants.areaTextures = XMFLOAT4(
        resources.hasExternalAlbedo ? 1.0f : 0.0f,
        resources.hasNormalMap ? 1.0f : 0.0f,
        resources.hasPgsMap ? 1.0f : 0.0f,
        0.0f);
    constants.materialColor0 = XMFLOAT4(0.23f, 0.31f, 0.37f, 1.0f);
    constants.materialColor1 = XMFLOAT4(0.38f, 0.44f, 0.48f, 1.0f);
    constants.materialColor2 = XMFLOAT4(0.12f, 0.15f, 0.17f, 1.0f);
    constants.materialColor3 = XMFLOAT4(0.035f, 0.045f, 0.055f, 1.0f);
    constants.materialSurface0 = XMFLOAT4(0.045f, 0.048f, 0.052f, 0.52f);
    constants.materialSurface1 = XMFLOAT4(0.055f, 0.058f, 0.062f, 0.62f);
    constants.materialSurface2 = XMFLOAT4(0.04f, 0.04f, 0.04f, 0.38f);
    constants.materialSurface3 = XMFLOAT4(0.035f, 0.035f, 0.035f, 0.32f);
    constants.areaEffects = XMFLOAT4(0.34f, 0.58f, 0.95f, 1.0f);
    constants.auxTextures = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    constants.semanticChannels = XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
    constants.semanticChannels2 = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
    constants.debug = XMFLOAT4(0.0f, 0.0f, resources.baselineComplete ? 1.0f : 0.0f, 0.0f);

    outputConstants = constants;
    return UploadSceneConstants(context, constantBuffer, constants);
}
SceneConstants RenderPipeline::BuildViewportSceneConstants(
    const PreviewState& state,
    const SceneConstants& baseConstants,
    uint32_t viewportWidth,
    uint32_t viewportHeight)
{
    SceneConstants constants = baseConstants;
    XMMATRIX world, view, projection;
    XMFLOAT3 eye, target;
    m_cameraController.BuildCameraMatrices(
        state,
        std::max(viewportWidth, 1U),
        std::max(viewportHeight, 1U),
        world,
        view,
        projection,
        eye,
        target);
    DirectX::XMStoreFloat4x4(&constants.world, world);
    DirectX::XMStoreFloat4x4(&constants.viewProjection, view * projection);
    constants.cameraTime.x = eye.x;
    constants.cameraTime.y = eye.y;
    constants.cameraTime.z = eye.z;
    XMFLOAT3 cameraRight, cameraUp, cameraForward;
    m_cameraController.GetCameraBasis(state, cameraRight, cameraUp, cameraForward);
    const float aspect = viewportHeight == 0U ? 1.0f :
        static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight);
    const float tanHalfFov = std::tan(DirectX::XMConvertToRadians(48.0f) * 0.5f);
    constants.cameraRight = XMFLOAT4(cameraRight.x, cameraRight.y, cameraRight.z, aspect * tanHalfFov);
    constants.cameraUp = XMFLOAT4(cameraUp.x, cameraUp.y, cameraUp.z, tanHalfFov);
    constants.cameraForward = XMFLOAT4(cameraForward.x, cameraForward.y, cameraForward.z, 0.0f);
    return constants;
}

void RenderPipeline::RenderShip(
    ID3D11DeviceContext* context,
    const PreviewResources& resources,
    const FinalCandidateSet& candidates,
    const PreviewState& state,
    const SceneConstants& baseConstants)
{
    const float clearColour[4] = {0.004f, 0.007f, 0.012f, 1.0f};
    context->ClearRenderTargetView(resources.renderTargetView.Get(), clearColour);
    context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    D3D11_VIEWPORT viewport{};
    viewport.Width = static_cast<float>(resources.width);
    viewport.Height = static_cast<float>(resources.height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);

    ID3D11Buffer* constantBuffer = resources.constantBuffer.Get();
    ID3D11SamplerState* textureSampler = resources.textureSampler.Get();
    const EnvironmentGpu* selectedEnvironment = SelectedEnvironment(resources, state);
    ID3D11ShaderResourceView* environmentView = selectedEnvironment ? selectedEnvironment->view.Get() : nullptr;

    const uint32_t sceneX = std::min(state.sceneViewportX, resources.width > 1U ? resources.width - 1U : 0U);
    const uint32_t sceneWidth = std::max(1U, resources.width - sceneX);
    const uint32_t sceneHeight = std::max(1U, resources.height);
    // Scientific comparison is A authored source / B deterministic 4x baseline /
    // C current learned stage during live training, while immutable final preview
    // remains A source / C final. Every pane uses the same sampler, camera,
    // lighting, geometry and material shader.

    struct PaneRect
    {
        uint32_t x = 0U;
        uint32_t y = 0U;
        uint32_t width = 1U;
        uint32_t height = 1U;
    };

    struct AssetBinding
    {
        ID3D11Buffer* vertexBuffer = nullptr;
        ID3D11Buffer* indexBuffer = nullptr;
        uint32_t indexCount = 0U;
        const std::vector<AreaMaterialGpu>* areaMaterials = nullptr;
        bool useBaselineGlobals = false;
        bool requireSourceDrawRange = false;
        bool valid = true;
    };

    const AssetBinding baselineAsset{
        resources.vertexBuffer.Get(),
        resources.indexBuffer.Get(),
        resources.indexCount,
        &resources.areaMaterials,
        true,
        false,
        true,
    };

    const CandidateAssetGpu& deterministicBaseline = candidates.baseline;
    const AssetBinding deterministicBaselineAsset = deterministicBaseline.available
        ? AssetBinding{
            baselineAsset.vertexBuffer,
            baselineAsset.indexBuffer,
            baselineAsset.indexCount,
            &deterministicBaseline.areaMaterials,
            false,
            true,
            true,
        }
        : AssetBinding{nullptr, nullptr, 0U, nullptr, false, false, false};

    const CandidateAssetGpu& finalCandidate = candidates.candidate;
    const AssetBinding finalAsset = finalCandidate.available
        ? AssetBinding{
            baselineAsset.vertexBuffer,
            baselineAsset.indexBuffer,
            baselineAsset.indexCount,
            &finalCandidate.areaMaterials,
            false,
            true,
            true,
        }
        : AssetBinding{nullptr, nullptr, 0U, nullptr, false, false, false};

    const bool threeWay = deterministicBaseline.available;
    PaneRect rawControlPane{sceneX, 0U, sceneWidth, sceneHeight};
    PaneRect deterministicBaselinePane = rawControlPane;
    PaneRect candidatePane = rawControlPane;
    if (threeWay)
    {
        if (state.splitVertical)
        {
            const uint32_t firstWidth = std::max(1U, sceneWidth / 3U);
            const uint32_t secondWidth = std::max(1U, (sceneWidth - firstWidth) / 2U);
            rawControlPane = PaneRect{sceneX, 0U, firstWidth, sceneHeight};
            deterministicBaselinePane = PaneRect{sceneX + firstWidth, 0U, secondWidth, sceneHeight};
            candidatePane = PaneRect{
                sceneX + firstWidth + secondWidth, 0U,
                sceneWidth - firstWidth - secondWidth, sceneHeight};
        }
        else
        {
            const uint32_t firstHeight = std::max(1U, sceneHeight / 3U);
            const uint32_t secondHeight = std::max(1U, (sceneHeight - firstHeight) / 2U);
            rawControlPane = PaneRect{sceneX, 0U, sceneWidth, firstHeight};
            deterministicBaselinePane = PaneRect{sceneX, firstHeight, sceneWidth, secondHeight};
            candidatePane = PaneRect{
                sceneX, firstHeight + secondHeight, sceneWidth,
                sceneHeight - firstHeight - secondHeight};
        }
        if (state.swapSplitSides)
            std::swap(rawControlPane, candidatePane);
    }
    else
    {
        if (state.splitVertical)
        {
            const uint32_t firstWidth = std::max(1U, sceneWidth / 2U);
            rawControlPane = PaneRect{sceneX, 0U, firstWidth, sceneHeight};
            candidatePane = PaneRect{sceneX + firstWidth, 0U, sceneWidth - firstWidth, sceneHeight};
        }
        else
        {
            const uint32_t firstHeight = std::max(1U, sceneHeight / 2U);
            rawControlPane = PaneRect{sceneX, 0U, sceneWidth, firstHeight};
            candidatePane = PaneRect{sceneX, firstHeight, sceneWidth, sceneHeight - firstHeight};
        }
        if (state.swapSplitSides)
            std::swap(rawControlPane, candidatePane);
    }
    const float blendFactor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const UINT stride = sizeof(Vertex);
    const UINT offset = 0;

    auto drawBackgroundPane = [&](const PaneRect& pane)
    {
        SceneConstants backgroundConstants = BuildViewportSceneConstants(
            state,
            baseConstants,
            pane.width,
            pane.height);
        if (!UploadSceneConstants(context, constantBuffer, backgroundConstants)) return false;

        viewport.TopLeftX = static_cast<float>(pane.x);
        viewport.TopLeftY = static_cast<float>(pane.y);
        viewport.Width = static_cast<float>(pane.width);
        viewport.Height = static_cast<float>(pane.height);
        context->RSSetViewports(1, &viewport);
        context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), nullptr);
        context->RSSetState(resources.solidRasterizer.Get());
        context->IASetInputLayout(nullptr);
        context->IASetIndexBuffer(nullptr, DXGI_FORMAT_UNKNOWN, 0);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(resources.backgroundVertexShader.Get(), nullptr, 0);
        context->PSSetShader(resources.backgroundPixelShader.Get(), nullptr, 0);
        context->VSSetConstantBuffers(0, 1, &constantBuffer);
        context->PSSetConstantBuffers(0, 1, &constantBuffer);
        context->PSSetShaderResources(4, 1, &environmentView);
        context->PSSetSamplers(0, 1, &textureSampler);
        context->Draw(3, 0);
        return true;
    };

    auto drawPane = [&](const PaneRect& pane, const AssetBinding& asset)
    {
        // Every pane retains the same mesh, camera, lighting, material shader,
        // environment and one shared high-quality sampler.
        if (!asset.valid) return;
        SceneConstants paneConstants = BuildViewportSceneConstants(
            state, baseConstants, pane.width, pane.height);
        const PreviewState& renderState = state;
        ID3D11SamplerState* paneTextureSampler = textureSampler;

        viewport.TopLeftX = static_cast<float>(pane.x);
        viewport.TopLeftY = static_cast<float>(pane.y);
        viewport.Width = static_cast<float>(pane.width);
        viewport.Height = static_cast<float>(pane.height);
        context->RSSetViewports(1, &viewport);
        context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), resources.depthStencilView.Get());
        context->RSSetState(state.wireframe ? resources.wireRasterizer.Get() : resources.solidRasterizer.Get());
        context->IASetInputLayout(resources.inputLayout.Get());
        context->IASetVertexBuffers(0, 1, &asset.vertexBuffer, &stride, &offset);
        context->IASetIndexBuffer(asset.indexBuffer, DXGI_FORMAT_R32_UINT, 0);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(resources.vertexShader.Get(), nullptr, 0);
        context->PSSetShader(resources.pixelShader.Get(), nullptr, 0);
        context->VSSetConstantBuffers(0, 1, &constantBuffer);
        context->PSSetConstantBuffers(0, 1, &constantBuffer);
        context->PSSetSamplers(0, 1, &paneTextureSampler);

        auto drawArea = [&](const AreaMaterialGpu& material)
        {
            const ObjDrawRange* drawRange = &material.drawRange;
            if (asset.requireSourceDrawRange)
            {
                const auto sourceMaterial = std::find_if(
                    resources.areaMaterials.begin(),
                    resources.areaMaterials.end(),
                    [&](const AreaMaterialGpu& value) {
                        return value.source.groupIndex == material.source.groupIndex;
                    });
                if (sourceMaterial == resources.areaMaterials.end()) return false;
                drawRange = &sourceMaterial->drawRange;
            }

            SceneConstants constants = paneConstants;
            constants.material.x = renderState.useTexture && material.hasAlbedo ? 1.0f : 0.0f;
            constants.options.x = renderState.useNormalMap && material.hasNormal ? 1.0f : 0.0f;
            constants.options.y = renderState.usePgsMap && material.hasPgs ? 1.0f : 0.0f;
            constants.areaTint = XMFLOAT4(material.source.tint.x, material.source.tint.y, material.source.tint.z, 1.0f);
            constants.areaSurface = XMFLOAT4(
                material.source.roughness,
                material.source.specular,
                material.source.alpha,
                static_cast<float>(static_cast<int>(material.source.pass)));
            constants.areaTextures = XMFLOAT4(
                material.hasAlbedo ? 1.0f : 0.0f,
                material.hasNormal ? 1.0f : 0.0f,
                material.hasPgs ? 1.0f : 0.0f,
                material.hasGlow ? 1.0f : 0.0f);
            constants.materialColor0 = XMFLOAT4(material.source.slots[0].colour.x, material.source.slots[0].colour.y, material.source.slots[0].colour.z, 1.0f);
            constants.materialColor1 = XMFLOAT4(material.source.slots[1].colour.x, material.source.slots[1].colour.y, material.source.slots[1].colour.z, 1.0f);
            constants.materialColor2 = XMFLOAT4(material.source.slots[2].colour.x, material.source.slots[2].colour.y, material.source.slots[2].colour.z, 1.0f);
            constants.materialColor3 = XMFLOAT4(material.source.slots[3].colour.x, material.source.slots[3].colour.y, material.source.slots[3].colour.z, 1.0f);
            constants.materialSurface0 = XMFLOAT4(material.source.slots[0].f0.x, material.source.slots[0].f0.y, material.source.slots[0].f0.z, material.source.slots[0].gloss);
            constants.materialSurface1 = XMFLOAT4(material.source.slots[1].f0.x, material.source.slots[1].f0.y, material.source.slots[1].f0.z, material.source.slots[1].gloss);
            constants.materialSurface2 = XMFLOAT4(material.source.slots[2].f0.x, material.source.slots[2].f0.y, material.source.slots[2].f0.z, material.source.slots[2].gloss);
            constants.materialSurface3 = XMFLOAT4(material.source.slots[3].f0.x, material.source.slots[3].f0.y, material.source.slots[3].f0.z, material.source.slots[3].gloss);
            constants.areaEffects = XMFLOAT4(
                material.source.glowColour.x,
                material.source.glowColour.y,
                material.source.glowColour.z,
                material.source.generalDataX);
            constants.auxTextures = XMFLOAT4(
                material.hasDirt ? 1.0f : 0.0f,
                material.hasAo ? 1.0f : 0.0f,
                material.hasPaintMask ? 1.0f : 0.0f,
                material.hasRoughnessMap ? 1.0f : 0.0f);
            constants.semanticChannels = XMFLOAT4(
                static_cast<float>(material.source.normalXChannel),
                static_cast<float>(material.source.normalYChannel),
                static_cast<float>(material.source.roughnessChannel),
                static_cast<float>(material.source.materialChannel));
            constants.semanticChannels2 = XMFLOAT4(
                static_cast<float>(material.source.aoChannel),
                static_cast<float>(material.source.paintChannel),
                static_cast<float>(material.source.dirtChannel),
                static_cast<float>(material.source.glowChannel));
            constants.debug = XMFLOAT4(
                0.0f,
                static_cast<float>(material.source.groupIndex + 1),
                material.source.baselineComplete ? 1.0f : 0.0f,
                static_cast<float>(static_cast<int>(material.source.shaderFamily)));
            if (!UploadSceneConstants(context, constantBuffer, constants)) return false;

            // The final albedo is already reconstructed offline. Both panes sample
            // their own material manifests
            // through the same live shader with no hidden runtime correction pass.
            ID3D11ShaderResourceView* selectedAlbedoView = material.albedoView.Get();
            ID3D11ShaderResourceView* textureViews[9] = {
                selectedAlbedoView,
                material.normalView.Get(),
                material.pgsView.Get(),
                material.glowView.Get(),
                environmentView,
                material.dirtView.Get(),
                material.aoView.Get(),
                material.paintMaskView.Get(),
                material.roughnessMapView.Get(),
            };
            context->PSSetShaderResources(0, 9, textureViews);

            if (material.source.pass == MaterialPass::Opaque)
            {
                context->OMSetBlendState(resources.opaqueBlendState.Get(), blendFactor, 0xffffffffU);
                context->OMSetDepthStencilState(resources.depthWriteState.Get(), 0);
            }
            else if (material.source.pass == MaterialPass::Additive)
            {
                context->OMSetBlendState(resources.additiveBlendState.Get(), blendFactor, 0xffffffffU);
                context->OMSetDepthStencilState(resources.depthReadState.Get(), 0);
            }
            else
            {
                context->OMSetBlendState(resources.alphaBlendState.Get(), blendFactor, 0xffffffffU);
                context->OMSetDepthStencilState(resources.depthReadState.Get(), 0);
            }
            context->DrawIndexed(drawRange->indexCount, drawRange->startIndex, 0);
            return true;
        };

        if (asset.areaMaterials && !asset.areaMaterials->empty())
        {
            for (const AreaMaterialGpu& material : *asset.areaMaterials)
            {
                if (!drawArea(material)) break;
            }
        }
        else if (asset.useBaselineGlobals)
        {
            UploadSceneConstants(context, constantBuffer, paneConstants);
            ID3D11ShaderResourceView* textureViews[9] = {
                resources.albedoView.Get(),
                resources.normalView.Get(),
                resources.pgsView.Get(),
                nullptr,
                environmentView,
                nullptr, nullptr, nullptr, nullptr,
            };
            context->PSSetShaderResources(0, 9, textureViews);
            context->OMSetBlendState(resources.opaqueBlendState.Get(), blendFactor, 0xffffffffU);
            context->OMSetDepthStencilState(resources.depthWriteState.Get(), 0);
            context->DrawIndexed(asset.indexCount, 0, 0);
        }
    };

    // C: current learned stage (or immutable final outside live mode).
    drawBackgroundPane(candidatePane);
    drawPane(candidatePane, finalAsset);
    context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    // B: exact deterministic 4x reconstruction baseline from the same LR evidence.
    if (threeWay)
    {
        drawBackgroundPane(deterministicBaselinePane);
        drawPane(deterministicBaselinePane, deterministicBaselineAsset);
        context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
    }

    // A: authoritative authored source under the exact same draw path.
    drawBackgroundPane(rawControlPane);
    drawPane(rawControlPane, baselineAsset);

    // Release bound SRVs and restore neutral output-merger state after either
    // pane layout. This avoids carrying preview resources into later passes.
    ID3D11ShaderResourceView* nullViews[9] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    context->PSSetShaderResources(0, 9, nullViews);
    context->OMSetBlendState(nullptr, blendFactor, 0xffffffffU);
    context->OMSetDepthStencilState(nullptr, 0);
}

} // namespace nsamdr
