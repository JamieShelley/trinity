#pragma once

#include "NSAMDRPreviewPlatform.h"

namespace nsamdr
{
struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
    XMFLOAT2 uv;
    float stretchHint;
    XMFLOAT2 stretchAxis;
    XMFLOAT2 repairUvCenter;
    XMFLOAT2 repairUvScale;
};

struct SceneConstants
{
    XMFLOAT4X4 world;
    XMFLOAT4X4 viewProjection;
    XMFLOAT4 cameraTime;
    XMFLOAT4 keyLight;
    XMFLOAT4 fillLight;
    XMFLOAT4 rimLight;
    XMFLOAT4 material;
    XMFLOAT4 surface;
    XMFLOAT4 options;
    XMFLOAT4 cameraRight;
    XMFLOAT4 cameraUp;
    XMFLOAT4 cameraForward;
    XMFLOAT4 environment;
    XMFLOAT4 areaTint;
    XMFLOAT4 areaSurface;
    XMFLOAT4 areaTextures;
    XMFLOAT4 materialColor0;
    XMFLOAT4 materialColor1;
    XMFLOAT4 materialColor2;
    XMFLOAT4 materialColor3;
    XMFLOAT4 materialSurface0;
    XMFLOAT4 materialSurface1;
    XMFLOAT4 materialSurface2;
    XMFLOAT4 materialSurface3;
    XMFLOAT4 areaEffects;
    XMFLOAT4 auxTextures;
    XMFLOAT4 semanticChannels;
    XMFLOAT4 semanticChannels2;
    XMFLOAT4 debug;
};

static_assert((sizeof(SceneConstants) % 16U) == 0U, "D3D11 constant buffers must be 16-byte aligned");

struct ObjCorner
{
    int position = -1;
    int texcoord = -1;
    int normal = -1;
};

struct ObjTriangle
{
    std::array<ObjCorner, 3> corners{};
    int groupIndex = 0;
    std::string groupName;
    std::string materialName;
};

struct TriangleBuildData
{
    std::array<XMFLOAT3, 3> positions{};
    std::array<XMFLOAT3, 3> normals{};
    std::array<XMFLOAT2, 3> uvs{};
    float density = 0.0f;
    float anisotropy = 1.0f;
    XMFLOAT2 stretchAxis = XMFLOAT2(1.0f, 0.0f);
    XMFLOAT2 stretchMagnitudes = XMFLOAT2(1.0f, 1.0f);
    XMFLOAT2 repairUvCenter = XMFLOAT2(0.0f, 0.0f);
    bool degenerateUv = false;
    int groupIndex = 0;
    std::string groupName;
    std::string materialName;
};

struct ObjDrawRange
{
    int groupIndex = 0;
    std::string groupName;
    std::string materialName;
    uint32_t startIndex = 0;
    uint32_t indexCount = 0;
};

struct ObjMesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ObjDrawRange> drawRanges;
    uint32_t sourcePositionCount = 0;
    uint32_t sourceTexcoordCount = 0;
    uint32_t sourceNormalCount = 0;
    uint32_t triangleCount = 0;
    uint32_t degenerateUvTriangles = 0;
    float averageStretch = 0.0f;
    float maximumStretch = 0.0f;
    float stretchCalibrationLow = 0.0f;
    float stretchCalibrationHigh = 1.0f;
    XMFLOAT3 boundsCenter = XMFLOAT3(0.0f, 0.0f, 0.0f);
    float boundsRadius = 1.0f;
    std::string path;
};

struct PreviewState
{
    bool splitVertical = true;
    bool swapSplitSides = false;
    float orbitYaw = -0.55f;
    float orbitPitch = 0.22f;
    float cameraDistance = 8.2f;
    float targetX = 0.0f;
    float targetY = 0.0f;
    float targetZ = 0.0f;
    float modelPitch = 0.0f;
    float modelYaw = 0.0f;
    float modelRoll = 0.0f;
    float orbitSpeed = 0.18f;
    float orbitSensitivity = 1.0f;
    float panSpeed = 1.0f;
    float zoomSpeed = 1.0f;
    float nearClip = 0.005f;
    float farClip = 300.0f;
    float normalMapStrength = 0.85f;
    float keyYaw = -0.65f;
    float keyPitch = -0.55f;
    float keyIntensity = 1.45f;
    float fillIntensity = 0.38f;
    float rimIntensity = 0.82f;
    float exposure = 1.15f;
    float ambient = 0.28f;
    float specularStrength = 1.0f;
    float roughnessBias = 0.0f;
    float environmentIntensity = 0.72f;
    float backgroundIntensity = 0.68f;
    float reflectionStrength = 0.92f;
    int lightingPreset = 0;
    bool autoOrbit = false;
    bool wireframe = false;
    bool useTexture = true;
    bool useNormalMap = true;
    bool usePgsMap = true;
    bool useEnvironment = true;
    bool flipV = false;
    bool requestScreenshot = false;
    bool requestFocus = false;
    int focusMouseX = 0;
    int focusMouseY = 0;
    bool requestZoom = false;
    float zoomWheelSteps = 0.0f;
    int zoomMouseX = 0;
    int zoomMouseY = 0;
    uint32_t sceneViewportX = 660U;
    uint32_t environmentIndex = 0U;
};
enum class ShaderFamily
{
    Unknown = 0,
    LegacyPgs = 1,
    V5Separate = 2,
    V5Packed = 3
};

const char* ShaderFamilyName(ShaderFamily family);



enum class MaterialPass
{
    Opaque = 0,
    Decal = 1,
    Transparent = 2,
    Additive = 3
};

struct MaterialSlotSource
{
    XMFLOAT3 colour = XMFLOAT3(0.34f, 0.38f, 0.42f);
    XMFLOAT3 f0 = XMFLOAT3(0.04f, 0.04f, 0.04f);
    float gloss = 0.52f;
};

struct AreaMaterialSource
{
    int groupIndex = 0;
    MaterialPass pass = MaterialPass::Opaque;
    std::string areaType;
    std::string areaName;
    std::string shaderPath;
    ShaderFamily shaderFamily = ShaderFamily::Unknown;
    std::string albedoPath;
    std::string normalPath;
    std::string pgsPath;
    std::string glowPath;
    std::string dirtPath;
    std::string aoPath;
    std::string paintMaskPath;
    std::string roughnessMapPath;
    XMFLOAT3 tint = XMFLOAT3(0.34f, 0.38f, 0.42f);
    XMFLOAT3 glowColour = XMFLOAT3(0.34f, 0.58f, 0.95f);
    std::array<MaterialSlotSource, 4> slots{};
    float detailScale = 1.0f;
    float generalDataX = 1.0f;
    float roughness = 0.48f;
    float specular = 0.72f;
    float alpha = 1.0f;
    int normalXChannel = 0;
    int normalYChannel = 1;
    int roughnessChannel = 0;
    int materialChannel = 0;
    int aoChannel = 0;
    int paintChannel = 0;
    int dirtChannel = 0;
    int glowChannel = 0;
    bool semanticComplete = false;
    bool baselineComplete = false;
    int unresolvedCount = 0;
    std::string unresolvedSemantics;
};

struct AreaMaterialGpu
{
    AreaMaterialSource source;
    ObjDrawRange drawRange;
    ComPtr<ID3D11ShaderResourceView> albedoView;
    ComPtr<ID3D11ShaderResourceView> normalView;
    ComPtr<ID3D11ShaderResourceView> pgsView;
    ComPtr<ID3D11ShaderResourceView> glowView;
    ComPtr<ID3D11ShaderResourceView> dirtView;
    ComPtr<ID3D11ShaderResourceView> aoView;
    ComPtr<ID3D11ShaderResourceView> paintMaskView;
    ComPtr<ID3D11ShaderResourceView> roughnessMapView;
    uint32_t albedoWidth = 1;
    uint32_t albedoHeight = 1;
    bool hasAlbedo = false;
    bool hasNormal = false;
    bool hasPgs = false;
    bool hasGlow = false;
    bool hasDirt = false;
    bool hasAo = false;
    bool hasPaintMask = false;
    bool hasRoughnessMap = false;
};

struct EnvironmentGpu
{
    std::string path;
    std::string label;
    ComPtr<ID3D11ShaderResourceView> view;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct PreviewResources
{
    ComPtr<ID3D11RenderTargetView> renderTargetView;
    ComPtr<ID3D11Texture2D> depthTexture;
    ComPtr<ID3D11DepthStencilView> depthStencilView;
    ComPtr<ID3D11VertexShader> vertexShader;
    ComPtr<ID3D11PixelShader> pixelShader;
    ComPtr<ID3D11VertexShader> backgroundVertexShader;
    ComPtr<ID3D11PixelShader> backgroundPixelShader;
    ComPtr<ID3D11InputLayout> inputLayout;
    ComPtr<ID3D11Buffer> vertexBuffer;
    ComPtr<ID3D11Buffer> indexBuffer;
    ComPtr<ID3D11Buffer> constantBuffer;
    ComPtr<ID3D11RasterizerState> solidRasterizer;
    ComPtr<ID3D11RasterizerState> wireRasterizer;
    ComPtr<ID3D11BlendState> opaqueBlendState;
    ComPtr<ID3D11BlendState> alphaBlendState;
    ComPtr<ID3D11BlendState> additiveBlendState;
    ComPtr<ID3D11DepthStencilState> depthWriteState;
    ComPtr<ID3D11DepthStencilState> depthReadState;
    ComPtr<ID3D11ShaderResourceView> albedoView;
    ComPtr<ID3D11ShaderResourceView> normalView;
    ComPtr<ID3D11ShaderResourceView> pgsView;
    ComPtr<ID3D11ShaderResourceView> environmentView;
    ComPtr<ID3D11SamplerState> textureSampler;
    uint32_t indexCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t textureWidth = 0;
    uint32_t textureHeight = 0;
    uint32_t normalWidth = 0;
    uint32_t normalHeight = 0;
    uint32_t pgsWidth = 0;
    uint32_t pgsHeight = 0;
    uint32_t environmentWidth = 0;
    uint32_t environmentHeight = 0;
    bool hasExternalAlbedo = false;
    bool hasNormalMap = false;
    bool hasPgsMap = false;
    bool hasEnvironment = false;
    bool baselineComplete = false;
    int baselineUnresolvedCount = 0;
    std::vector<AreaMaterialGpu> areaMaterials;
    std::vector<EnvironmentGpu> environments;
};

struct CandidateAssetGpu
{
    std::vector<AreaMaterialGpu> areaMaterials;
    uint32_t maximumTextureWidth = 0;
    uint32_t maximumTextureHeight = 0;
    bool available = false;
    std::string label;
    std::string objPath;
    std::string materialManifestPath;
    std::string status;
};

struct FinalCandidateSet
{
    CandidateAssetGpu candidate;
};

struct ShipCatalogEntry
{
    std::string displayName;
    std::string groupName;
    std::string factionName;
    std::string typeId;
    std::string canonicalKey;
    std::string preferredAsset;
    std::vector<std::string> variants;
};

struct CatalogSelection
{
    size_t entryIndex = 0;
    int variantIndex = -1; // -1 selects preferredAsset
};

struct ShipCatalog
{
    std::vector<ShipCatalogEntry> entries;
    std::vector<CatalogSelection> filtered;
    std::array<char, 256> search{};
    int selectedFilteredIndex = -1;
    std::string currentQuery;
    std::string status;
    bool showRawVariants = false;
};

} // namespace nsamdr
