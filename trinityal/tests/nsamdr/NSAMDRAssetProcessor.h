#pragma once

#include "NSAMDRMeshProcessor.h"
#include "NSAMDRShaderLibrary.h"

namespace nsamdr
{
class AssetProcessor final
{
public:
    AssetProcessor(PreviewShaderLibrary& shaderLibrary, MeshProcessor& meshProcessor);

    bool EnsureSelectedEnvironmentLoaded(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        PreviewResources& resources,
        const PreviewState& state);
    bool CreatePreviewTargets(
        ID3D11Device* device,
        IDXGISwapChain* swapChain,
        PreviewResources& resources);
    bool CreatePreviewResources(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        IDXGISwapChain* swapChain,
        const ObjMesh& mesh,
        const std::string& albedoPath,
        const std::string& normalPath,
        const std::string& pgsPath,
        const std::vector<std::string>& environmentPaths,
        const std::string& materialManifestPath,
        PreviewResources& resources);
    bool LoadCandidateAsset(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        const std::string& label,
        const std::string& objPath,
        const std::string& materialManifestPath,
        CandidateAssetGpu& candidate);

private:
    bool CreateFallbackTexture(ID3D11Device* device, PreviewResources& resources);
    ShaderFamily ParseShaderFamily(const std::string& value);
    MaterialPass ParseMaterialPass(const std::string& value);
    std::vector<std::string> SplitTabs(const std::string& line);
    bool ParseFloatValue(const std::string& text, float& value);
    bool ParseIntValue(const std::string& text, int& value);
    bool LoadAreaMaterialSources(const std::string& path, std::vector<AreaMaterialSource>& materials, std::string& error);
    bool LoadWicTexture( ID3D11Device* device, ID3D11DeviceContext* context, const std::string& path, bool srgb, ComPtr<ID3D11ShaderResourceView>& view, uint32_t& outputWidth, uint32_t& outputHeight, const char* label);
    PreviewShaderLibrary& m_shaderLibrary;
    MeshProcessor& m_meshProcessor;
};
} // namespace nsamdr
