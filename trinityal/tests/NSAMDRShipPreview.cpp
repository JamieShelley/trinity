// Copyright © 2026
// Local NSAMDR visual prototype for Carbon TrinityAL DX11.

#include "StdAfx.h"
#include "WithValidRenderContextFixture.h"
#include "RenderWindow.h"

#if defined(_WIN32) && TRINITY_PLATFORM == TRINITY_DIRECTX11

#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using DirectX::XMFLOAT2;
using DirectX::XMFLOAT3;
using DirectX::XMFLOAT4;
using DirectX::XMFLOAT4X4;
using DirectX::XMMATRIX;
using DirectX::XMVECTOR;
using Microsoft::WRL::ComPtr;

extern bool g_exitInteractiveOnCharacter;

namespace
{

constexpr char kPreviewShader[] = R"HLSL(
cbuffer SceneConstants : register(b0)
{
    row_major float4x4 gWorld;
    row_major float4x4 gViewProjection;
    float4 gCameraTime;      // xyz = camera, w = elapsed seconds
    float4 gControls;        // x = mode, y = strength, z = damage low, w = damage high
    float4 gLighting;        // xyz = light direction, w = micro-normal strength
};

struct VSInput
{
    float3 position    : POSITION;
    float3 normal      : NORMAL;
    float2 uv          : TEXCOORD0;
    float stretchHint : TEXCOORD1;
};

struct VSOutput
{
    float4 position    : SV_POSITION;
    float3 worldPos    : TEXCOORD0;
    float3 localPos    : TEXCOORD1;
    float3 normal      : TEXCOORD2;
    float2 uv          : TEXCOORD3;
    float stretchHint : TEXCOORD4;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4 worldPosition = mul(float4(input.position, 1.0), gWorld);
    output.position = mul(worldPosition, gViewProjection);
    output.worldPos = worldPosition.xyz;
    output.localPos = input.position;
    output.normal = normalize(mul(float4(input.normal, 0.0), gWorld).xyz);
    output.uv = input.uv;
    output.stretchHint = input.stretchHint;
    return output;
}

float Hash21(float2 p)
{
    p = frac(p * float2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return frac(p.x * p.y);
}

float ValueNoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);

    float a = Hash21(i + float2(0.0, 0.0));
    float b = Hash21(i + float2(1.0, 0.0));
    float c = Hash21(i + float2(0.0, 1.0));
    float d = Hash21(i + float2(1.0, 1.0));

    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float Fbm(float2 p)
{
    float value = 0.0;
    float amplitude = 0.55;
    [unroll]
    for (int octave = 0; octave < 4; ++octave)
    {
        value += ValueNoise(p) * amplitude;
        p = mul(p, float2x2(1.58, -1.13, 1.13, 1.58)) + 17.17;
        amplitude *= 0.48;
    }
    return value;
}

float StochasticTriplanar(float3 p, float3 n)
{
    float3 weights = pow(abs(n), 4.0);
    weights /= max(weights.x + weights.y + weights.z, 1.0e-4);

    // Different rotations and offsets reduce visible repetition between projections.
    float xy = Fbm(mul(p.xy * 34.0, float2x2(0.80, -0.60, 0.60, 0.80)) + 19.3);
    float yz = Fbm(mul(p.yz * 37.0, float2x2(0.37, -0.93, 0.93, 0.37)) + 43.7);
    float zx = Fbm(mul(p.zx * 31.0, float2x2(0.94, -0.34, 0.34, 0.94)) + 71.1);

    return xy * weights.z + yz * weights.x + zx * weights.y;
}

float NeuralResidual(float4 features)
{
    // A tiny fixed-weight two-layer residual network. This is deliberately a
    // structural prototype, not trained production weights.
    float4 hidden;
    hidden.x = tanh(dot(features, float4( 1.34, -0.72,  0.48,  0.91)) - 0.18);
    hidden.y = tanh(dot(features, float4(-0.55,  1.27, -0.83,  0.39)) + 0.07);
    hidden.z = tanh(dot(features, float4( 0.76,  0.31,  1.11, -0.64)) - 0.24);
    hidden.w = tanh(dot(features, float4(-1.02,  0.58,  0.44,  1.16)) + 0.13);
    return dot(hidden, float4(0.31, -0.22, 0.27, 0.19));
}

float3 DamageHeatmap(float damage)
{
    float3 cold = float3(0.02, 0.08, 0.28);
    float3 warm = float3(1.00, 0.78, 0.03);
    float3 hot  = float3(1.00, 0.03, 0.01);
    return damage < 0.5
        ? lerp(cold, warm, damage * 2.0)
        : lerp(warm, hot, (damage - 0.5) * 2.0);
}

float4 PSMain(VSOutput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    float3 lightDirection = normalize(-gLighting.xyz);

    float uvFootprint = length(ddx(input.uv)) + length(ddy(input.uv));
    float worldFootprint = length(ddx(input.worldPos)) + length(ddy(input.worldPos));
    float worldUnitsPerUv = worldFootprint / max(uvFootprint, 1.0e-5);

    float damage = smoothstep(gControls.z, gControls.w, worldUnitsPerUv);
    damage = max(damage, input.stretchHint * 0.65);

    int mode = (int)round(gControls.x);
    if (mode == 2)
    {
        return float4(DamageHeatmap(damage), 1.0);
    }

    // Semantic panel structure remains independent of the reconstructed microdetail.
    float longitudinal = abs(frac((input.localPos.x + 4.0) * 0.58) - 0.5);
    float transverse = abs(frac((input.localPos.z + 2.0) * 1.35) - 0.5);
    float panelLine = 1.0 - smoothstep(0.025, 0.075, min(longitudinal, transverse));

    float baseMicro = Fbm(input.uv * 92.0);
    float stochastic = StochasticTriplanar(input.localPos, normal);
    float neural = NeuralResidual(float4(stochastic, baseMicro, damage, saturate(dot(normal, lightDirection))));

    float reconstructed = stochastic;
    if (mode == 1 || mode == 4)
    {
        reconstructed = saturate(stochastic + neural * 0.24);
    }

    float reconstructionAmount = saturate(damage * gControls.y);
    float micro = baseMicro;
    if (mode == 1 || mode == 3 || mode == 4)
    {
        micro = lerp(baseMicro, reconstructed, reconstructionAmount);
    }

    float3 dpdx = ddx(input.worldPos);
    float3 dpdy = ddy(input.worldPos);
    float3 tangentX = normalize(dpdx + 1.0e-5);
    float3 tangentY = normalize(dpdy + 1.0e-5);
    float microHeight = micro - 0.5;
    float3 microGradient = ddx(microHeight) * tangentX + ddy(microHeight) * tangentY;
    float3 shadedNormal = normalize(normal - microGradient * gLighting.w * reconstructionAmount);

    float diffuse = saturate(dot(shadedNormal, lightDirection));
    float rim = pow(1.0 - saturate(dot(shadedNormal, normalize(gCameraTime.xyz - input.worldPos))), 3.0);

    float sideTint = saturate(input.localPos.z * 0.22 + 0.5);
    float3 navyA = float3(0.035, 0.080, 0.115);
    float3 navyB = float3(0.075, 0.145, 0.190);
    float3 paint = lerp(navyA, navyB, sideTint);

    float grain = (micro - 0.5) * 0.24;
    float roughnessVariation = (Fbm(input.uv * 17.0 + 37.0) - 0.5) * 0.10;
    float3 colour = paint * (0.28 + diffuse * (0.68 + roughnessVariation));
    colour += grain;
    colour += rim * float3(0.12, 0.26, 0.34);
    colour = lerp(colour, colour * 0.28, panelLine * 0.70);

    // Engine-emissive hint on the rear face of the synthetic hull.
    float rearMask = 1.0 - smoothstep(-2.75, -2.35, input.localPos.x);
    float engineBand = 1.0 - smoothstep(0.10, 0.32, abs(input.localPos.y));
    colour += rearMask * engineBand * float3(0.08, 0.42, 0.95);

    return float4(saturate(colour), 1.0);
}
)HLSL";

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
    XMFLOAT2 uv;
    float stretchHint;
};

struct SceneConstants
{
    XMFLOAT4X4 world;
    XMFLOAT4X4 viewProjection;
    XMFLOAT4 cameraTime;
    XMFLOAT4 controls;
    XMFLOAT4 lighting;
};

static_assert((sizeof(SceneConstants) % 16U) == 0U, "D3D11 constant buffers must be 16-byte aligned");

struct PreviewState
{
    int mode = 0;
    int previousEnabledMode = 1;
    float strength = 1.0f;
    float damageLow = 1.8f;
    float damageHigh = 7.5f;
    float yaw = -0.52f;
    float pitch = 0.22f;
    float orbitSpeed = 0.22f;
    float microNormalStrength = 2.2f;
    bool autoOrbit = true;
    bool wireframe = false;
    bool requestScreenshot = false;
};

struct PreviewResources
{
    ComPtr<ID3D11RenderTargetView> renderTargetView;
    ComPtr<ID3D11Texture2D> depthTexture;
    ComPtr<ID3D11DepthStencilView> depthStencilView;
    ComPtr<ID3D11VertexShader> vertexShader;
    ComPtr<ID3D11PixelShader> pixelShader;
    ComPtr<ID3D11InputLayout> inputLayout;
    ComPtr<ID3D11Buffer> vertexBuffer;
    ComPtr<ID3D11Buffer> indexBuffer;
    ComPtr<ID3D11Buffer> constantBuffer;
    ComPtr<ID3D11RasterizerState> solidRasterizer;
    ComPtr<ID3D11RasterizerState> wireRasterizer;
    uint32_t indexCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

WNDPROC g_previousWindowProc = nullptr;

LRESULT CALLBACK NSAMDRPreviewWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, message, wParam, lParam))
    {
        return 1;
    }

    if (message == WM_CLOSE)
    {
        DestroyWindow(hwnd);
        return 0;
    }
    if (message == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }

    return CallWindowProc(g_previousWindowProc, hwnd, message, wParam, lParam);
}

bool KeyPressed(int virtualKey)
{
    return (GetAsyncKeyState(virtualKey) & 1) != 0;
}

void AddFace(
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    const std::array<XMFLOAT3, 4>& positions,
    const XMFLOAT3& normal,
    float uExtent,
    float vExtent,
    float stretchHint)
{
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    vertices.push_back({positions[0], normal, XMFLOAT2(0.0f, 0.0f), stretchHint});
    vertices.push_back({positions[1], normal, XMFLOAT2(uExtent, 0.0f), stretchHint});
    vertices.push_back({positions[2], normal, XMFLOAT2(uExtent, vExtent), stretchHint});
    vertices.push_back({positions[3], normal, XMFLOAT2(0.0f, vExtent), stretchHint});

    indices.insert(indices.end(), {base + 0U, base + 1U, base + 2U, base + 0U, base + 2U, base + 3U});
}

void AddBox(
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    const XMFLOAT3& centre,
    const XMFLOAT3& half,
    float uvDensity,
    float stretchHint)
{
    const float x0 = centre.x - half.x;
    const float x1 = centre.x + half.x;
    const float y0 = centre.y - half.y;
    const float y1 = centre.y + half.y;
    const float z0 = centre.z - half.z;
    const float z1 = centre.z + half.z;

    AddFace(vertices, indices, {{{x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}, {x1, y0, z1}}}, {1, 0, 0}, 2.0f * half.y * uvDensity, 2.0f * half.z * uvDensity, stretchHint);
    AddFace(vertices, indices, {{{x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}, {x0, y0, z0}}}, {-1, 0, 0}, 2.0f * half.y * uvDensity, 2.0f * half.z * uvDensity, stretchHint);
    AddFace(vertices, indices, {{{x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}}}, {0, 1, 0}, 2.0f * half.z * uvDensity, 2.0f * half.x * uvDensity, stretchHint);
    AddFace(vertices, indices, {{{x0, y0, z1}, {x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}}}, {0, -1, 0}, 2.0f * half.z * uvDensity, 2.0f * half.x * uvDensity, stretchHint);
    AddFace(vertices, indices, {{{x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}}}, {0, 0, 1}, 2.0f * half.x * uvDensity, 2.0f * half.y * uvDensity, stretchHint);
    AddFace(vertices, indices, {{{x1, y0, z0}, {x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0}}}, {0, 0, -1}, 2.0f * half.x * uvDensity, 2.0f * half.y * uvDensity, stretchHint);
}

void AddTaperedHullSection(
    std::vector<Vertex>& vertices,
    std::vector<uint32_t>& indices,
    float x0,
    float x1,
    float halfY0,
    float halfZ0,
    float halfY1,
    float halfZ1,
    float uvDensity,
    float stretchHint)
{
    const XMFLOAT3 a{x0, -halfY0, -halfZ0};
    const XMFLOAT3 b{x0,  halfY0, -halfZ0};
    const XMFLOAT3 c{x0,  halfY0,  halfZ0};
    const XMFLOAT3 d{x0, -halfY0,  halfZ0};
    const XMFLOAT3 e{x1, -halfY1, -halfZ1};
    const XMFLOAT3 f{x1,  halfY1, -halfZ1};
    const XMFLOAT3 g{x1,  halfY1,  halfZ1};
    const XMFLOAT3 h{x1, -halfY1,  halfZ1};

    const float length = std::abs(x1 - x0);
    AddFace(vertices, indices, {{a, b, c, d}}, {-1, 0, 0}, 2.0f * halfY0 * uvDensity, 2.0f * halfZ0 * uvDensity, stretchHint);
    AddFace(vertices, indices, {{h, g, f, e}}, {1, 0, 0}, 2.0f * halfY1 * uvDensity, 2.0f * halfZ1 * uvDensity, stretchHint);
    AddFace(vertices, indices, {{b, f, g, c}}, {0, 1, 0}, length * uvDensity, (halfZ0 + halfZ1) * uvDensity, stretchHint);
    AddFace(vertices, indices, {{d, h, e, a}}, {0, -1, 0}, length * uvDensity, (halfZ0 + halfZ1) * uvDensity, stretchHint);
    AddFace(vertices, indices, {{d, c, g, h}}, {0, 0, 1}, length * uvDensity, (halfY0 + halfY1) * uvDensity, stretchHint);
    AddFace(vertices, indices, {{e, f, b, a}}, {0, 0, -1}, length * uvDensity, (halfY0 + halfY1) * uvDensity, stretchHint);
}

void BuildRavenLikeMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    // Main battleship spine: reasonable texel density.
    AddTaperedHullSection(vertices, indices, -2.45f, 1.35f, 0.52f, 0.67f, 0.62f, 0.86f, 1.35f, 0.0f);

    // Broad forward armour section: intentionally damaged/stretch-mapped.
    AddTaperedHullSection(vertices, indices, 1.35f, 2.95f, 0.62f, 0.86f, 0.28f, 0.42f, 0.14f, 1.0f);

    // Rear engineering block and asymmetric side pods.
    AddBox(vertices, indices, {-2.67f, 0.00f, 0.00f}, {0.34f, 0.58f, 0.78f}, 1.20f, 0.0f);
    AddBox(vertices, indices, {-0.65f, 0.05f, 1.02f}, {1.45f, 0.22f, 0.27f}, 0.18f, 1.0f);
    AddBox(vertices, indices, {-0.35f, -0.02f, -0.96f}, {1.12f, 0.24f, 0.24f}, 1.30f, 0.0f);

    // Dorsal superstructure, bridge and weapon-like silhouette blocks.
    AddBox(vertices, indices, {-0.25f, 0.72f, 0.08f}, {0.86f, 0.20f, 0.38f}, 1.45f, 0.0f);
    AddBox(vertices, indices, {0.65f, 0.93f, 0.16f}, {0.36f, 0.16f, 0.28f}, 0.20f, 1.0f);
    AddBox(vertices, indices, {-1.35f, 0.82f, -0.20f}, {0.28f, 0.28f, 0.22f}, 1.25f, 0.0f);

    // Underside keel and fins add a recognisably heavy EVE-like profile.
    AddBox(vertices, indices, {-0.35f, -0.74f, 0.02f}, {1.25f, 0.16f, 0.30f}, 0.16f, 1.0f);
    AddBox(vertices, indices, {-1.95f, 0.02f, 1.22f}, {0.38f, 0.48f, 0.10f}, 1.10f, 0.0f);
    AddBox(vertices, indices, {-1.72f, 0.00f, -1.18f}, {0.42f, 0.42f, 0.10f}, 1.10f, 0.0f);
}

bool CompileShader(const char* entryPoint, const char* profile, ComPtr<ID3DBlob>& shaderBlob)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
#if defined(_DEBUG)
    flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    ComPtr<ID3DBlob> errorBlob;
    const HRESULT result = D3DCompile(
        kPreviewShader,
        sizeof(kPreviewShader) - 1U,
        "NSAMDRShipPreview.hlsl",
        nullptr,
        nullptr,
        entryPoint,
        profile,
        flags,
        0,
        shaderBlob.GetAddressOf(),
        errorBlob.GetAddressOf());

    if (FAILED(result))
    {
        const char* errorText = errorBlob ? static_cast<const char*>(errorBlob->GetBufferPointer()) : "Unknown HLSL compilation error";
        ADD_FAILURE() << errorText;
        return false;
    }
    return true;
}

bool CreatePreviewResources(ID3D11Device* device, IDXGISwapChain* swapChain, PreviewResources& resources)
{
    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(backBuffer.GetAddressOf()))))
    {
        ADD_FAILURE() << "Failed to retrieve TrinityAL DX11 back buffer";
        return false;
    }

    D3D11_TEXTURE2D_DESC backBufferDescription{};
    backBuffer->GetDesc(&backBufferDescription);
    resources.width = backBufferDescription.Width;
    resources.height = backBufferDescription.Height;

    if (FAILED(device->CreateRenderTargetView(backBuffer.Get(), nullptr, resources.renderTargetView.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview render-target view";
        return false;
    }

    D3D11_TEXTURE2D_DESC depthDescription{};
    depthDescription.Width = resources.width;
    depthDescription.Height = resources.height;
    depthDescription.MipLevels = 1;
    depthDescription.ArraySize = 1;
    depthDescription.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthDescription.SampleDesc.Count = 1;
    depthDescription.Usage = D3D11_USAGE_DEFAULT;
    depthDescription.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    if (FAILED(device->CreateTexture2D(&depthDescription, nullptr, resources.depthTexture.GetAddressOf())) ||
        FAILED(device->CreateDepthStencilView(resources.depthTexture.Get(), nullptr, resources.depthStencilView.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview depth buffer";
        return false;
    }

    ComPtr<ID3DBlob> vertexBlob;
    ComPtr<ID3DBlob> pixelBlob;
    if (!CompileShader("VSMain", "vs_5_0", vertexBlob) || !CompileShader("PSMain", "ps_5_0", pixelBlob))
    {
        return false;
    }

    if (FAILED(device->CreateVertexShader(vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), nullptr, resources.vertexShader.GetAddressOf())) ||
        FAILED(device->CreatePixelShader(pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), nullptr, resources.pixelShader.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview shaders";
        return false;
    }

    const D3D11_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, position)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, normal)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, uv)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, stretchHint)), D3D11_INPUT_PER_VERTEX_DATA, 0},
    };

    if (FAILED(device->CreateInputLayout(
            inputElements,
            static_cast<UINT>(sizeof(inputElements) / sizeof(inputElements[0])),
            vertexBlob->GetBufferPointer(),
            vertexBlob->GetBufferSize(),
            resources.inputLayout.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview input layout";
        return false;
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    BuildRavenLikeMesh(vertices, indices);
    resources.indexCount = static_cast<uint32_t>(indices.size());

    D3D11_BUFFER_DESC vertexBufferDescription{};
    vertexBufferDescription.ByteWidth = static_cast<UINT>(vertices.size() * sizeof(Vertex));
    vertexBufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    vertexBufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vertexData{};
    vertexData.pSysMem = vertices.data();

    D3D11_BUFFER_DESC indexBufferDescription{};
    indexBufferDescription.ByteWidth = static_cast<UINT>(indices.size() * sizeof(uint32_t));
    indexBufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    indexBufferDescription.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA indexData{};
    indexData.pSysMem = indices.data();

    D3D11_BUFFER_DESC constantBufferDescription{};
    constantBufferDescription.ByteWidth = static_cast<UINT>(sizeof(SceneConstants));
    constantBufferDescription.Usage = D3D11_USAGE_DYNAMIC;
    constantBufferDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constantBufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    if (FAILED(device->CreateBuffer(&vertexBufferDescription, &vertexData, resources.vertexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&indexBufferDescription, &indexData, resources.indexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&constantBufferDescription, nullptr, resources.constantBuffer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview buffers";
        return false;
    }

    D3D11_RASTERIZER_DESC rasterizerDescription{};
    rasterizerDescription.FillMode = D3D11_FILL_SOLID;
    rasterizerDescription.CullMode = D3D11_CULL_NONE;
    rasterizerDescription.DepthClipEnable = TRUE;
    if (FAILED(device->CreateRasterizerState(&rasterizerDescription, resources.solidRasterizer.GetAddressOf())))
    {
        return false;
    }
    rasterizerDescription.FillMode = D3D11_FILL_WIREFRAME;
    if (FAILED(device->CreateRasterizerState(&rasterizerDescription, resources.wireRasterizer.GetAddressOf())))
    {
        return false;
    }

    return true;
}

void ProcessHotkeys(PreviewState& state, HWND hwnd)
{
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard)
    {
        if (KeyPressed('0'))
        {
            state.mode = 0;
        }
        if (KeyPressed('1'))
        {
            state.mode = 1;
            state.previousEnabledMode = 1;
        }
        if (KeyPressed('2'))
        {
            state.mode = 2;
            state.previousEnabledMode = 2;
        }
        if (KeyPressed('3'))
        {
            state.mode = 3;
            state.previousEnabledMode = 3;
        }
        if (KeyPressed('4'))
        {
            state.mode = 4;
            state.previousEnabledMode = 4;
        }
        if (KeyPressed(VK_SPACE))
        {
            if (state.mode == 0)
            {
                state.mode = state.previousEnabledMode;
            }
            else
            {
                state.previousEnabledMode = state.mode;
                state.mode = 0;
            }
        }
        if (KeyPressed('R'))
        {
            state.yaw = -0.52f;
            state.pitch = 0.22f;
        }
    }

    if (KeyPressed(VK_F9))
    {
        state.requestScreenshot = true;
    }
    if (KeyPressed(VK_ESCAPE))
    {
        PostMessage(hwnd, WM_CLOSE, 0, 0);
    }
}

void DrawControlPanel(PreviewState& state)
{
    static constexpr const char* modeNames[] = {
        "Off / original stretched UV detail",
        "Full NSAMDR",
        "Damage mask",
        "Stochastic reconstruction",
        "Neural residual prototype",
    };

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.94f);
    ImGui::Begin("NSAMDR Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::TextUnformatted("Neural Stretch-Aware Material Detail Reconstruction");
    ImGui::Separator();
    for (int mode = 0; mode < static_cast<int>(sizeof(modeNames) / sizeof(modeNames[0])); ++mode)
    {
        if (ImGui::RadioButton(modeNames[mode], state.mode == mode))
        {
            state.mode = mode;
            if (mode != 0)
            {
                state.previousEnabledMode = mode;
            }
        }
    }

    ImGui::Separator();
    ImGui::SliderFloat("Strength", &state.strength, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Damage start", &state.damageLow, 0.5f, 6.0f, "%.2f");
    ImGui::SliderFloat("Damage full", &state.damageHigh, 2.0f, 16.0f, "%.2f");
    if (state.damageHigh < state.damageLow + 0.1f)
    {
        state.damageHigh = state.damageLow + 0.1f;
    }
    ImGui::SliderFloat("Micro-normal", &state.microNormalStrength, 0.0f, 5.0f, "%.2f");

    ImGui::Separator();
    ImGui::Checkbox("Automatic orbit", &state.autoOrbit);
    ImGui::SliderFloat("Orbit speed", &state.orbitSpeed, -1.0f, 1.0f, "%.2f rad/s");
    ImGui::SliderFloat("Yaw", &state.yaw, -3.14159f, 3.14159f, "%.2f");
    ImGui::SliderFloat("Pitch", &state.pitch, -0.75f, 0.75f, "%.2f");
    ImGui::Checkbox("Wireframe", &state.wireframe);

    if (ImGui::Button("Save screenshot (F9)"))
    {
        state.requestScreenshot = true;
    }

    ImGui::Separator();
    ImGui::Text("Frame time: %.3f ms", 1000.0f / std::max(ImGui::GetIO().Framerate, 1.0f));
    ImGui::TextUnformatted("Keys: 0-4 modes | Space off/on | R reset | F9 capture | Esc exit");
    ImGui::TextWrapped("Yellow/red regions in Damage Mask mode are the deliberately stretch-damaged armour panels.");
    ImGui::TextWrapped("The neural mode currently uses fixed prototype weights; production work will replace them with trained weights.");

    ImGui::End();
}

bool UpdateSceneConstants(
    ID3D11DeviceContext* context,
    ID3D11Buffer* constantBuffer,
    const PreviewResources& resources,
    const PreviewState& state,
    float elapsedSeconds)
{
    const float aspectRatio = resources.height == 0U ? 1.0f : static_cast<float>(resources.width) / static_cast<float>(resources.height);

    const XMMATRIX world = DirectX::XMMatrixRotationX(state.pitch) * DirectX::XMMatrixRotationY(state.yaw);
    const XMVECTOR eye = DirectX::XMVectorSet(0.0f, 2.15f, -8.6f, 1.0f);
    const XMVECTOR target = DirectX::XMVectorSet(0.0f, 0.05f, 0.0f, 1.0f);
    const XMVECTOR up = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    const XMMATRIX view = DirectX::XMMatrixLookAtLH(eye, target, up);
    const XMMATRIX projection = DirectX::XMMatrixPerspectiveFovLH(DirectX::XMConvertToRadians(48.0f), aspectRatio, 0.1f, 100.0f);

    SceneConstants constants{};
    DirectX::XMStoreFloat4x4(&constants.world, world);
    DirectX::XMStoreFloat4x4(&constants.viewProjection, view * projection);
    constants.cameraTime = XMFLOAT4(0.0f, 2.15f, -8.6f, elapsedSeconds);
    constants.controls = XMFLOAT4(static_cast<float>(state.mode), state.strength, state.damageLow, state.damageHigh);
    constants.lighting = XMFLOAT4(-0.45f, -0.78f, -0.35f, state.microNormalStrength);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context->Map(constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
    {
        ADD_FAILURE() << "Failed to update NSAMDR preview constants";
        return false;
    }
    memcpy(mapped.pData, &constants, sizeof(constants));
    context->Unmap(constantBuffer, 0);
    return true;
}

void RenderShip(ID3D11DeviceContext* context, const PreviewResources& resources, const PreviewState& state)
{
    const float clearColour[4] = {0.006f, 0.009f, 0.016f, 1.0f};
    context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), resources.depthStencilView.Get());
    context->ClearRenderTargetView(resources.renderTargetView.Get(), clearColour);
    context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    D3D11_VIEWPORT viewport{};
    viewport.Width = static_cast<float>(resources.width);
    viewport.Height = static_cast<float>(resources.height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);
    context->RSSetState(state.wireframe ? resources.wireRasterizer.Get() : resources.solidRasterizer.Get());

    const UINT stride = sizeof(Vertex);
    const UINT offset = 0;
    ID3D11Buffer* vertexBuffer = resources.vertexBuffer.Get();
    ID3D11Buffer* constantBuffer = resources.constantBuffer.Get();

    context->IASetInputLayout(resources.inputLayout.Get());
    context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    context->IASetIndexBuffer(resources.indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(resources.vertexShader.Get(), nullptr, 0);
    context->PSSetShader(resources.pixelShader.Get(), nullptr, 0);
    context->VSSetConstantBuffers(0, 1, &constantBuffer);
    context->PSSetConstantBuffers(0, 1, &constantBuffer);
    context->DrawIndexed(resources.indexCount, 0, 0);
}

std::string BuildScreenshotPath(int mode)
{
    CreateDirectoryA("artifacts", nullptr);
    CreateDirectoryA("artifacts\\nsamdr", nullptr);

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return "artifacts\\nsamdr\\raven_like_mode_" + std::to_string(mode) + "_" + std::to_string(milliseconds) + ".dds";
}

struct NSAMDRRendering : public WithValidRenderContext
{
};

} // namespace

TEST_F(NSAMDRRendering, ShipPreview)
{
    ENSURE_GPU_OR_SKIP

    ASSERT_NE(renderContext, nullptr);
    ASSERT_TRUE(renderContext->IsValid());
    ASSERT_TRUE(renderContext->m_d3dDevice11);
    ASSERT_TRUE(renderContext->m_context);
    ASSERT_TRUE(renderContext->m_swapChain);

    HWND hwnd = static_cast<HWND>(GetWindowHandle());
    SetWindowTextW(hwnd, L"NSAMDR Raven-like Ship Preview - TrinityAL DX11");

    GetWindow()->Resize(1280U, 720U);
    presentParameters.mode.width = 1280U;
    presentParameters.mode.height = 720U;
    ASSERT_HRESULT_SUCCEEDED(renderContext->SetPresentParameters(Tr2VideoAdapterInfo::DEFAULT_ADAPTER, presentParameters));

    g_exitInteractiveOnCharacter = false;
    g_previousWindowProc = reinterpret_cast<WNDPROC>(SetWindowLongPtr(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(NSAMDRPreviewWindowProc)));
    ASSERT_NE(g_previousWindowProc, nullptr);

    ID3D11Device* device = renderContext->m_d3dDevice11;
    ID3D11DeviceContext* context = renderContext->m_context;
    IDXGISwapChain* swapChain = renderContext->m_swapChain;

    PreviewResources resources;
    ASSERT_TRUE(CreatePreviewResources(device, swapChain, resources));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ASSERT_TRUE(ImGui_ImplWin32_Init(hwnd));
    ASSERT_TRUE(ImGui_ImplDX11_Init(device, context));

    PreviewState state;
    const auto startTime = std::chrono::steady_clock::now();
    auto previousFrame = startTime;

    auto frame = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - previousFrame).count();
        const float elapsedSeconds = std::chrono::duration<float>(now - startTime).count();
        previousFrame = now;

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        ProcessHotkeys(state, hwnd);
        if (state.autoOrbit)
        {
            state.yaw += deltaSeconds * state.orbitSpeed;
        }
        DrawControlPanel(state);

        ASSERT_TRUE(UpdateSceneConstants(context, resources.constantBuffer.Get(), resources, state, elapsedSeconds));
        RenderShip(context, resources, state);

        ImGui::Render();
        context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        if (state.requestScreenshot)
        {
            state.requestScreenshot = false;
            const std::string screenshotPath = BuildScreenshotPath(state.mode);
            MakeScreenShot(screenshotPath.c_str());
            printf("Saved NSAMDR screenshot: %s\n", screenshotPath.c_str());
        }

        ASSERT_HRESULT_SUCCEEDED(renderContext->Present());
    };

    RunLoop(frame);

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    if (IsWindow(hwnd))
    {
        SetWindowLongPtr(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(g_previousWindowProc));
    }
    g_previousWindowProc = nullptr;
    g_exitInteractiveOnCharacter = true;

    context->ClearState();
}

#endif // _WIN32 && TRINITY_DIRECTX11
