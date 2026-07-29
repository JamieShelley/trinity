// Copyright © 2026
// Granny-free NSAMDR visual test using real ship geometry converted to Wavefront OBJ.

#include "StdAfx.h"
#include "WithValidRenderContextFixture.h"
#include "RenderWindow.h"

#if defined(_WIN32) && TRINITY_PLATFORM == TRINITY_DIRECTX11

#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>
#include <wincodec.h>
#include <windowsx.h>
#include <wrl/client.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
    float4 gKeyLight;        // xyz = travel direction, w = intensity
    float4 gFillLight;       // xyz = travel direction, w = intensity
    float4 gRimLight;        // xyz = travel direction, w = intensity
    float4 gMaterial;        // x = use albedo, y = flip V, z = exposure, w = ambient
    float4 gSurface;         // x = micro normal, y = normal map strength, z = specular, w = roughness bias
    float4 gOptions;         // x = use normal, y = use PGS, z/w reserved
    float4 gCameraRight;     // xyz = camera right, w = aspect * tan(fov / 2)
    float4 gCameraUp;        // xyz = camera up, w = tan(fov / 2)
    float4 gCameraForward;   // xyz = camera forward
    float4 gEnvironment;     // x = use map, y = light intensity, z = background intensity, w = reflection strength
    float4 gDiagnostics;     // x = checker scale, y = reserved, z = texture width, w = texture height
    float4 gStructure;       // x = structure sharpness, y = structure scale, z = preserve-clean strength, w = difference scale
    float4 gAreaTint;        // rgb = SOF faction/material tint, w = per-area material enabled
    float4 gAreaSurface;     // x = roughness, y = specular, z = alpha, w = pass (0 opaque, 1 decal, 2 transparent, 3 additive)
    float4 gAreaTextures;    // x = albedo, y = normal, z = material selector, w = glow
    float4 gMaterialColor0;  // rgb = SOF material slot 1 base colour
    float4 gMaterialColor1;  // rgb = SOF material slot 2 base colour
    float4 gMaterialColor2;  // rgb = SOF material slot 3 base colour
    float4 gMaterialColor3;  // rgb = SOF material slot 4 base colour
    float4 gMaterialSurface0;// rgb = F0, w = roughness
    float4 gMaterialSurface1;// rgb = F0, w = roughness
    float4 gMaterialSurface2;// rgb = F0, w = roughness
    float4 gMaterialSurface3;// rgb = F0, w = roughness
    float4 gAreaEffects;     // rgb = resolved GeneralGlowColor, w = GeneralData.x
    float4 gAuxTextures;     // x = dirt, y = AO, z = paint mask, w = roughness map
    float4 gSemanticChannels;  // x = normal X, y = normal Y, z = roughness, w = material selector
    float4 gSemanticChannels2; // x = AO, y = paint, z = dirt, w = glow
    float4 gDebug;             // x = diagnostic view, y = area id, z = area complete, w = shader family
    float4 gRepair;            // x = repair method, y = sampling LOD bias, z = projection strength, w = transfer strength
};

Texture2D gAlbedo : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gPgsMap : register(t2);
Texture2D gGlowMap : register(t3);
Texture2D gEnvironmentMap : register(t4);
Texture2D gDirtMap : register(t5);
Texture2D gAoMap : register(t6);
Texture2D gPaintMaskMap : register(t7);
Texture2D gRoughnessMap : register(t8);
SamplerState gTextureSampler : register(s0);

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
    float a = Hash21(i);
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
    float xy = Fbm(mul(p.xy * 41.0, float2x2(0.80, -0.60, 0.60, 0.80)) + 19.3);
    float yz = Fbm(mul(p.yz * 43.0, float2x2(0.37, -0.93, 0.93, 0.37)) + 43.7);
    float zx = Fbm(mul(p.zx * 37.0, float2x2(0.94, -0.34, 0.34, 0.94)) + 71.1);
    return xy * weights.z + yz * weights.x + zx * weights.y;
}

float NeuralResidual(float4 features)
{
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
    float3 hot = float3(1.00, 0.03, 0.01);
    return damage < 0.5 ? lerp(cold, warm, damage * 2.0) : lerp(warm, hot, (damage - 0.5) * 2.0);
}

float Luminance(float3 colour)
{
    return dot(colour, float3(0.2126, 0.7152, 0.0722));
}

float Checker(float2 uv, float scale)
{
    float2 cell = floor(uv * max(scale, 1.0));
    return frac(cell.x + cell.y);
}

float GridMask(float2 uv, float scale)
{
    float2 f = abs(frac(uv * max(scale, 1.0)) - 0.5);
    float edge = min(f.x, f.y);
    return 1.0 - smoothstep(0.02, 0.08, edge);
}

float2 PrincipalStretchDirection(float2 duvdx, float2 duvdy, out float anisotropy)
{
    float a = dot(duvdx, duvdx);
    float b = dot(duvdx, duvdy);
    float c = dot(duvdy, duvdy);
    float trace = a + c;
    float determinant = a * c - b * b;
    float discriminant = sqrt(max(trace * trace * 0.25 - determinant, 0.0));
    float lambda1 = max(trace * 0.5 + discriminant, 1.0e-8);
    float lambda2 = max(trace * 0.5 - discriminant, 1.0e-8);
    anisotropy = sqrt(lambda1 / lambda2);
    if (abs(b) > 1.0e-6)
    {
        return normalize(float2(lambda1 - c, b));
    }
    return a >= c ? float2(1.0, 0.0) : float2(0.0, 1.0);
}

float PanelField2D(float2 p)
{
    float2 coarse = p;
    float2 fine = p * 2.7 + float2(17.3, 7.1);
    float2 f0 = abs(frac(coarse) - 0.5);
    float2 f1 = abs(frac(fine) - 0.5);
    float seam0 = 1.0 - smoothstep(0.018, 0.065, min(f0.x, f0.y));
    float seam1 = 1.0 - smoothstep(0.012, 0.045, min(f1.x, f1.y));
    float strakes = 0.5 + 0.5 * sin((coarse.x + coarse.y * 0.37) * 6.2831853);
    return saturate(seam0 * 0.65 + seam1 * 0.25 + strakes * 0.10);
}

float StructuralDetail(float3 p, float3 n, float scale)
{
    float3 weights = pow(abs(n), 6.0);
    weights /= max(weights.x + weights.y + weights.z, 1.0e-4);
    float xy = PanelField2D(p.xy * scale);
    float yz = PanelField2D(p.yz * scale * 0.95 + float2(5.1, 2.7));
    float zx = PanelField2D(p.zx * scale * 1.05 + float2(9.3, 6.4));
    return xy * weights.z + yz * weights.x + zx * weights.y;
}

float3 ApplyMappedNormal(float3 geometricNormal, float3 worldPos, float2 uv, float3 sampledNormal)
{
    float3 dp1 = ddx(worldPos);
    float3 dp2 = ddy(worldPos);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);
    float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(determinant) < 1.0e-7)
    {
        return geometricNormal;
    }
    float inverseDeterminant = 1.0 / determinant;
    float3 tangent = normalize((dp1 * duv2.y - dp2 * duv1.y) * inverseDeterminant);
    tangent = normalize(tangent - geometricNormal * dot(geometricNormal, tangent));
    float3 bitangent = normalize(cross(geometricNormal, tangent)) * (determinant < 0.0 ? -1.0 : 1.0);
    return normalize(tangent * sampledNormal.x + bitangent * sampledNormal.y + geometricNormal * sampledNormal.z);
}

float DistributionGGX(float3 n, float3 h, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = saturate(dot(n, h));
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denominator * denominator, 1.0e-5);
}

float GeometrySchlickGGX(float nDotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 1.0e-5);
}

float GeometrySmith(float3 n, float3 v, float3 l, float roughness)
{
    return GeometrySchlickGGX(saturate(dot(n, v)), roughness) * GeometrySchlickGGX(saturate(dot(n, l)), roughness);
}

float3 FresnelSchlick(float cosineTheta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - cosineTheta, 5.0);
}

float3 EvaluateLight(float3 n, float3 v, float3 l, float3 albedo, float3 lightColour, float intensity, float roughness, float3 f0)
{
    float nDotL = saturate(dot(n, l));
    if (nDotL <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 h = normalize(v + l);
    float3 f = FresnelSchlick(saturate(dot(h, v)), saturate(f0));
    float d = DistributionGGX(n, h, roughness);
    float g = GeometrySmith(n, v, l, roughness);
    float3 specular = d * g * f / max(4.0 * saturate(dot(n, v)) * nDotL, 1.0e-4);
    float3 diffuse = albedo * (1.0 - f) / 3.14159265;
    return (diffuse + specular) * lightColour * intensity * nDotL;
}

float3 ProceduralEnvironment(float3 direction)
{
    direction = normalize(direction);
    float horizon = pow(saturate(1.0 - abs(direction.y)), 2.4);
    float upper = saturate(direction.y * 0.5 + 0.5);
    float3 deepSpace = lerp(float3(0.002, 0.004, 0.012), float3(0.018, 0.045, 0.105), upper);
    float3 blueNebula = float3(0.055, 0.16, 0.34) * horizon;
    float warmCloud = pow(saturate(dot(direction, normalize(float3(-0.72, 0.18, 0.66)))), 7.0);
    return deepSpace + blueNebula + warmCloud * float3(0.38, 0.15, 0.055);
}

float3 SampleEnvironmentLod(float3 direction, float lod)
{
    direction = normalize(direction);
    if (gEnvironment.x > 0.5)
    {
        const float inverseTwoPi = 0.15915494309;
        const float inversePi = 0.31830988618;
        float2 uv;
        uv.x = atan2(direction.x, direction.z) * inverseTwoPi + 0.5;
        uv.y = acos(clamp(direction.y, -1.0, 1.0)) * inversePi;
        return max(gEnvironmentMap.SampleLevel(gTextureSampler, uv, max(lod, 0.0)).rgb, 0.0);
    }
    return ProceduralEnvironment(direction);
}

float3 SampleEnvironment(float3 direction)
{
    return SampleEnvironmentLod(direction, 0.0);
}

float SampleChannel(float4 sampleValue, float channel)
{
    if (channel < 0.5) return sampleValue.r;
    if (channel < 1.5) return sampleValue.g;
    if (channel < 2.5) return sampleValue.b;
    return sampleValue.a;
}

float3 AreaIdColour(float id)
{
    float3 seed = frac(float3(id * 0.1031, id * 0.11369, id * 0.13787));
    seed += dot(seed, seed.yzx + 19.19);
    return frac((seed.xxy + seed.yzz) * seed.zyx);
}

float4 NormalizedMaterialWeights(float2 uv)
{
    if (gAreaTextures.z <= 0.5) return float4(1.0, 0.0, 0.0, 0.0);

    const float4 materialSample = gPgsMap.Sample(gTextureSampler, uv);
    const int shaderFamily = (int)round(gDebug.w);
    if (shaderFamily == 1)
    {
        // Legacy PGS uses R=sub-mask, B=mask and three material slots. Keep
        // this path separate from the later V5 four-tone material selector.
        const float subMask = saturate(materialSample.r);
        const float mask = saturate(materialSample.b);
        float4 weights = float4(saturate(1.0 - max(mask, subMask)), mask * (1.0 - subMask), subMask, 0.0);
        const float total = dot(weights, float4(1.0, 1.0, 1.0, 1.0));
        return total > 1.0e-5 ? weights / total : float4(1.0, 0.0, 0.0, 0.0);
    }

    // V5 separate and V5++ packed textures both use one four-tone material
    // selector; the channel is supplied by the semantic manifest.
    const float selector = saturate(SampleChannel(materialSample, gSemanticChannels.w));
    const float4 centres = float4(0.0, 0.333333343, 0.666666687, 1.0);
    float4 weights = saturate(1.03191495 - abs(selector.xxxx - centres) * 3.19148946);
    const float total = dot(weights, float4(1.0, 1.0, 1.0, 1.0));
    return total > 1.0e-5 ? weights / total : float4(1.0, 0.0, 0.0, 0.0);
}

float3 BlendMaterialColour(float4 weights)
{
    return gMaterialColor0.rgb * weights.x + gMaterialColor1.rgb * weights.y +
           gMaterialColor2.rgb * weights.z + gMaterialColor3.rgb * weights.w;
}

float3 BlendMaterialF0(float4 weights)
{
    return gMaterialSurface0.rgb * weights.x + gMaterialSurface1.rgb * weights.y +
           gMaterialSurface2.rgb * weights.z + gMaterialSurface3.rgb * weights.w;
}

float BlendMaterialGloss(float4 weights)
{
    return dot(weights, float4(gMaterialSurface0.w, gMaterialSurface1.w, gMaterialSurface2.w, gMaterialSurface3.w));
}

struct BackgroundVSOutput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

BackgroundVSOutput VSBackground(uint vertexId : SV_VertexID)
{
    BackgroundVSOutput output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.9999, 1.0);
    output.uv = uv;
    return output;
}

float4 PSBackground(BackgroundVSOutput input) : SV_TARGET
{
    float2 ndc = float2(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0);
    float3 ray = normalize(
        gCameraForward.xyz +
        gCameraRight.xyz * (ndc.x * gCameraRight.w) +
        gCameraUp.xyz * (ndc.y * gCameraUp.w));
    float3 colour = SampleEnvironment(ray) * gEnvironment.z * gMaterial.z;
    colour = colour / (1.0 + colour);
    colour = pow(saturate(colour), 1.0 / 2.2);
    return float4(colour, 1.0);
}

float4 PSMain(VSOutput input) : SV_TARGET
{
    float3 geometricNormal = normalize(input.normal);
    float2 uv = input.uv;
    if (gMaterial.y > 0.5) uv.y = 1.0 - uv.y;

    int mode = (int)round(gControls.x);
    const int repairMethod = (int)round(gRepair.x);
    float4 materialWeights = NormalizedMaterialWeights(uv);
    float3 neutralPaint = gAreaTint.w > 0.5 ? max(gAreaTint.rgb, 0.001) : float3(0.045, 0.105, 0.145);
    float3 materialBase = max(BlendMaterialColour(materialWeights), 0.001);
    float3 materialF0 = clamp(BlendMaterialF0(materialWeights), 0.0, 1.0);
    float materialGloss = max(BlendMaterialGloss(materialWeights), 0.0);

    const float repairLodBias = (mode >= 3 && repairMethod == 0) ? gRepair.y : 0.0;
    float4 sampledAlbedoRgba = gAlbedo.SampleBias(gTextureSampler, uv, repairLodBias);
    float3 sampledAlbedo = sampledAlbedoRgba.rgb;
    float paintMask = gAuxTextures.z > 0.5
        ? SampleChannel(gPaintMaskMap.Sample(gTextureSampler, uv), gSemanticChannels2.y) : 0.0;

    // quadV5 multiplies the authored albedo detail by the material-library
    // diffuse colour. PaintMask removes that tint towards white; it is not a
    // faction-colour overlay and must not be inferred from albedo chroma.
    float3 materialTint = lerp(materialBase, float3(1.0, 1.0, 1.0), saturate(paintMask));
    float3 areaAlbedo = gMaterial.x > 0.5 ? sampledAlbedo * materialTint : materialTint;

    float dirtAmount = gAuxTextures.x > 0.5
        ? saturate(SampleChannel(gDirtMap.Sample(gTextureSampler, uv), gSemanticChannels2.z)) : 0.0;
    areaAlbedo *= lerp(1.0, 0.58, dirtAmount * 0.58);
    float aoSample = gAuxTextures.y > 0.5
        ? saturate(SampleChannel(gAoMap.Sample(gTextureSampler, uv), gSemanticChannels2.x)) : 1.0;
    float ao = lerp(0.32, 1.0, aoSample);
    float3 albedo = max(areaAlbedo, 0.0);

    float2 duvdx = ddx(uv);
    float2 duvdy = ddy(uv);
    float uvFootprint = length(duvdx) + length(duvdy);
    float worldFootprint = length(ddx(input.worldPos)) + length(ddy(input.worldPos));
    float worldUnitsPerUv = worldFootprint / max(uvFootprint, 1.0e-6);
    float derivativeMetric = log2(1.0 + worldUnitsPerUv);
    float derivativeDamage = smoothstep(gControls.z, gControls.w, derivativeMetric);

    float anisotropy = 1.0;
    float2 principalDir = PrincipalStretchDirection(duvdx, duvdy, anisotropy);
    float anisotropyDamage = saturate((anisotropy - 1.2) / 2.8);
    float damage = saturate(max(input.stretchHint, max(derivativeDamage * 0.60, anisotropyDamage)));

    float texelSpanX = max(abs(duvdx.x) * gDiagnostics.z, abs(duvdy.x) * gDiagnostics.z);
    float texelSpanY = max(abs(duvdx.y) * gDiagnostics.w, abs(duvdy.y) * gDiagnostics.w);
    float estimatedMip = log2(max(max(texelSpanX, texelSpanY), 1.0));
    float actualMipNormalized = saturate((estimatedMip + 1.0) / 7.0);

    if (mode == 1)
    {
        const int diagnosticView = (int)round(gDebug.x);
        float4 normalPacked = gNormalMap.Sample(gTextureSampler, uv);
        float normalX = SampleChannel(normalPacked, gSemanticChannels.x);
        float normalY = SampleChannel(normalPacked, gSemanticChannels.y);
        float2 normalXY = float2(normalX, normalY) * 2.0 - 1.0;
        float normalZ = sqrt(saturate(1.0 - dot(normalXY, normalXY)));
        float4 materialPacked = gPgsMap.Sample(gTextureSampler, uv);
        float selector = SampleChannel(materialPacked, gSemanticChannels.w);
        float roughnessSample = gAuxTextures.w > 0.5
            ? SampleChannel(gRoughnessMap.Sample(gTextureSampler, uv), gSemanticChannels.z) : 1.0;
        float glowSample = gAreaTextures.w > 0.5
            ? SampleChannel(gGlowMap.Sample(gTextureSampler, uv), gSemanticChannels2.w) : 0.0;

        if (diagnosticView == 1) return float4(sampledAlbedo, 1.0);
        if (diagnosticView == 2) return float4(roughnessSample.xxx, 1.0);
        if (diagnosticView == 3) return float4(normalX.xxx, 1.0);
        if (diagnosticView == 4) return float4(normalY.xxx, 1.0);
        if (diagnosticView == 5) return float4(normalZ.xxx, 1.0);
        if (diagnosticView == 6) return float4(aoSample.xxx, 1.0);
        if (diagnosticView == 7) return float4(paintMask.xxx, 1.0);
        if (diagnosticView == 8) return float4(selector.xxx, 1.0);
        if (diagnosticView == 9) return float4(dirtAmount.xxx, 1.0);
        if (diagnosticView == 10) return float4(glowSample.xxx, 1.0);
        if (diagnosticView == 11) return float4(saturate(materialBase), 1.0);
        if (diagnosticView == 12) return float4(AreaIdColour(gDebug.y), 1.0);
        if (diagnosticView == 13)
        {
            return gDebug.z > 0.5 ? float4(0.05, 0.65, 0.15, 1.0) : float4(1.0, 0.0, 0.75, 1.0);
        }
        if (diagnosticView == 14) return float4(actualMipNormalized, derivativeDamage, anisotropyDamage, 1.0);

        float checker = Checker(uv, gDiagnostics.x);
        float grid = GridMask(uv, gDiagnostics.x);
        float3 albedoPreview = saturate(albedo * lerp(0.82, 1.18, checker) + grid * 0.22);
        float3 normalPreview = float3(normalX, normalY, normalZ);
        float3 materialPreview = float3(selector, paintMask, glowSample);
        float3 colour = albedoPreview;
        if (uv.x >= (1.0 / 3.0) && uv.x < (2.0 / 3.0)) colour = normalPreview;
        if (uv.x >= (2.0 / 3.0)) colour = materialPreview;
        return float4(saturate(colour), 1.0);
    }

    if (mode == 2)
    {
        float3 dirColour = 0.5 + 0.5 * float3(principalDir.x, principalDir.y, 1.0 - abs(principalDir.x));
        float3 mipColour = float3(actualMipNormalized, derivativeDamage, anisotropyDamage);
        float checker = Checker(uv, gDiagnostics.x);
        float grid = GridMask(uv, gDiagnostics.x * 0.75);
        float3 colour = lerp(DamageHeatmap(damage), dirColour, 0.35);
        colour = lerp(colour, mipColour, 0.25);
        colour = lerp(colour, colour * 0.82 + 0.18, checker * 0.35);
        colour += grid * 0.18;
        return float4(saturate(colour), 1.0);
    }

    float3 mappedNormal = geometricNormal;
    if (gOptions.x > 0.5)
    {
        float4 packedNormal = gNormalMap.Sample(gTextureSampler, uv);
        float3 normalSample = float3(
            SampleChannel(packedNormal, gSemanticChannels.x) * 2.0 - 1.0,
            SampleChannel(packedNormal, gSemanticChannels.y) * 2.0 - 1.0,
            0.0);
        normalSample.xy *= gSurface.y;
        normalSample.z = sqrt(saturate(1.0 - dot(normalSample.xy, normalSample.xy)));
        mappedNormal = ApplyMappedNormal(geometricNormal, input.worldPos, uv, normalize(normalSample));
    }

    float sampleMip = clamp(max(estimatedMip, 1.0), 1.0, 6.0);
    float3 lowFrequencySample = gAlbedo.SampleLevel(gTextureSampler, uv, sampleMip).rgb;
    float lowFrequencyLuma = Luminance(lowFrequencySample);
    float lowFrequencyChroma = max(lowFrequencySample.r, max(lowFrequencySample.g, lowFrequencySample.b)) - min(lowFrequencySample.r, min(lowFrequencySample.g, lowFrequencySample.b));
    float3 lowFrequencyAlbedo = gMaterial.x > 0.5
        ? lerp(materialBase * lerp(0.42, 1.58, lowFrequencyLuma),
               lowFrequencySample * lerp(float3(0.80, 0.80, 0.80), materialBase * 1.20, 0.28),
               saturate((lowFrequencyChroma - 0.025) * 5.0))
        : materialBase;
    lowFrequencyAlbedo = lerp(lowFrequencyAlbedo, neutralPaint * lerp(0.42, 1.58, lowFrequencyLuma), saturate(paintMask * 0.72));
    lowFrequencyAlbedo *= lerp(1.0, 0.58, dirtAmount * 0.58);
    float3 sourceHigh = albedo - lowFrequencyAlbedo;

    float structureField = StructuralDetail(input.localPos, geometricNormal, max(gStructure.y, 0.1));
    float edgeSignal = saturate((abs(ddx(Luminance(albedo))) + abs(ddy(Luminance(albedo)))) * 32.0);
    float structureBlend = saturate(damage * gControls.y);
    float preserveClean = saturate(gStructure.z);
    float blend = saturate(structureBlend * preserveClean);

    float3 tint = normalize(max(lowFrequencyAlbedo + float3(0.08, 0.08, 0.08), float3(0.08, 0.08, 0.08)));
    float structureContrast = (structureField - 0.5) * gStructure.x;
    float sourceStructure = dot(sourceHigh, float3(0.3333, 0.3333, 0.3333));
    float reconstructedStructure = lerp(sourceStructure * 0.35 + edgeSignal * 0.10,
                                        structureContrast * (0.85 + edgeSignal * 0.35),
                                        saturate(anisotropyDamage + derivativeDamage * 0.5));
    float3 structuralAlbedo = saturate(lowFrequencyAlbedo + lerp(sourceHigh * 0.35,
                                                                  tint * reconstructedStructure,
                                                                  blend));

    float sourceMicro = Fbm(uv * 123.0 + albedo.rg * 7.0);
    float stochastic = StochasticTriplanar(input.localPos, mappedNormal);
    float3 keyDirection = normalize(-gKeyLight.xyz);
    float neural = NeuralResidual(float4(stochastic, sourceMicro, damage, saturate(dot(mappedNormal, keyDirection))));
    float reconstructedMicro = saturate(lerp(sourceMicro,
                                             structureField * 0.72 + stochastic * 0.28 + neural * 0.10,
                                             blend));
    float microHeight = reconstructedMicro - 0.5;

    float microBlend = (mode == 4) ? blend : 0.0;
    float3 dpdx = ddx(input.worldPos);
    float3 dpdy = ddy(input.worldPos);
    float3 tangentX = normalize(dpdx + float3(1.0e-5, 0.0, 0.0));
    float3 tangentY = normalize(dpdy + float3(0.0, 1.0e-5, 0.0));
    float3 microGradient = ddx(microHeight) * tangentX + ddy(microHeight) * tangentY;
    float3 n = normalize(mappedNormal - microGradient * gSurface.x * microBlend);
    float3 v = normalize(gCameraTime.xyz - input.worldPos);

    // quadV5 combines per-material gloss with RoughnessMap.r, then blends
    // towards the paint gloss constant (0.4) using PaintMask * GeneralData.x.
    float authoredRoughnessSample = gAuxTextures.w > 0.5
        ? SampleChannel(gRoughnessMap.Sample(gTextureSampler, uv), gSemanticChannels.z) : 1.0;
    float authoredGloss = saturate(materialGloss * authoredRoughnessSample);
    float paintGlossBlend = saturate(paintMask * gAreaEffects.w);
    float combinedGloss = lerp(authoredGloss, 0.4, paintGlossBlend);
    float roughness = clamp(1.0 - combinedGloss + gSurface.w + dirtAmount * 0.16, 0.04, 0.98);
    float3 f0 = clamp(materialF0 * max(gSurface.z, 0.05), 0.0, 1.0);

    float3 samplingCorrectedAlbedo = areaAlbedo;
    float projectedDetail = (StochasticTriplanar(input.localPos * max(gStructure.y, 0.1), geometricNormal) - 0.5) * gRepair.z;
    float3 projectedAlbedo = saturate(lowFrequencyAlbedo + tint * projectedDetail * blend);
    float3 edgeGuidedAlbedo = structuralAlbedo;
    float neuralColourResidual = NeuralResidual(float4(structureField, edgeSignal, damage, Luminance(lowFrequencyAlbedo)));
    float3 hybridAlbedo = saturate(edgeGuidedAlbedo + tint * neuralColourResidual * 0.08 * blend * gRepair.w);

    float3 selectedRepair = edgeGuidedAlbedo;
    if (repairMethod == 0) selectedRepair = samplingCorrectedAlbedo;
    else if (repairMethod == 1) selectedRepair = projectedAlbedo;
    else if (repairMethod == 3) selectedRepair = hybridAlbedo;

    float3 shadedAlbedo = (mode == 3 || mode == 4 || mode == 5) ? selectedRepair : albedo;

    float3 colour = 0.0;
    colour += EvaluateLight(n, v, normalize(-gKeyLight.xyz), shadedAlbedo, float3(1.00, 0.96, 0.90), gKeyLight.w, roughness, f0);
    colour += EvaluateLight(n, v, normalize(-gFillLight.xyz), shadedAlbedo, float3(0.46, 0.57, 0.76), gFillLight.w, min(roughness + 0.06, 0.98), f0 * 0.82);
    colour += EvaluateLight(n, v, normalize(-gRimLight.xyz), shadedAlbedo, float3(0.30, 0.45, 0.72), gRimLight.w, roughness, f0);

    float hemisphere = saturate(n.y * 0.5 + 0.5);
    float3 ambientColour = lerp(float3(0.012, 0.016, 0.024), float3(0.055, 0.075, 0.105), hemisphere);
    float3 environmentDiffuse = SampleEnvironmentLod(n, 5.0 + roughness * 2.0);
    colour += shadedAlbedo * (ambientColour + environmentDiffuse * (0.28 * gEnvironment.y)) * gMaterial.w * ao;

    float nDotV = saturate(dot(n, v));
    float3 reflectionFresnel = FresnelSchlick(nDotV, f0);
    float3 reflectedEnvironment = SampleEnvironmentLod(reflect(-v, n), roughness * 7.0);
    colour += reflectedEnvironment * reflectionFresnel * gEnvironment.y * gEnvironment.w * ao;

    if (mode == 4)
    {
        colour += (reconstructedMicro - 0.5) * 0.04 * blend;
    }
    if (gAreaTextures.w > 0.5)
    {
        float glowMask = SampleChannel(gGlowMap.Sample(gTextureSampler, uv), gSemanticChannels2.w);
        colour += glowMask * gAreaEffects.rgb * 1.8;
    }

    colour *= gMaterial.z;
    colour = colour / (1.0 + colour);
    colour = pow(saturate(colour), 1.0 / 2.2);

    if (mode == 5)
    {
        float3 diff = abs(selectedRepair - albedo) * gStructure.w;
        diff += DamageHeatmap(blend) * 0.15;
        return float4(saturate(diff), 1.0);
    }

    float textureAlpha = (gAreaTextures.x > 0.5 && (int)round(gDebug.w) == 1) ? sampledAlbedoRgba.a : 1.0;
    float outputAlpha = gAreaTint.w > 0.5 ? saturate(gAreaSurface.z * textureAlpha) : 1.0;
    return float4(colour, outputAlpha);
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
    XMFLOAT4 diagnostics;
    XMFLOAT4 structure;
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
    XMFLOAT4 repair;
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
    XMFLOAT3 boundsCenter = XMFLOAT3(0.0f, 0.0f, 0.0f);
    float boundsRadius = 1.0f;
    std::string path;
};

struct PreviewState
{
    int mode = 0;
    int previousEnabledMode = 1;
    float strength = 1.0f;
    float damageLow = 0.25f;
    float damageHigh = 1.55f;
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
    float microNormalStrength = 2.1f;
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
    float structureSharpness = 1.45f;
    float structureScale = 5.25f;
    float preserveClean = 0.92f;
    float differenceScale = 2.2f;
    float diagnosticCheckerScale = 18.0f;
    float samplingLodBias = -0.65f;
    float projectionStrength = 0.42f;
    float transferStrength = 0.72f;
    int diagnosticView = 0;
    int repairMethod = 2;
    int truthTarget = 0;
    int proofCaptureStep = -1;
    int proofRestoreMode = 0;
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

const char* ShaderFamilyName(ShaderFamily family)
{
    switch (family)
    {
        case ShaderFamily::LegacyPgs: return "legacy PGS";
        case ShaderFamily::V5Separate: return "V5 separate";
        case ShaderFamily::V5Packed: return "V5++ packed";
        default: return "unknown";
    }
}

static constexpr const char* kTruthTargetNames[] = {
    "A_overview_material_separation",
    "B_bright_underslung_module",
    "C_striped_panel",
    "D_markings_and_insignia",
    "E_known_uv_damage",
};

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

WNDPROC g_previousWindowProc = nullptr;
HWND g_previewWindow = nullptr;
PreviewState* g_previewState = nullptr;
bool g_previewInputFocused = false;
bool g_blockMouseUntilRelease = false;
bool g_orbitDragging = false;
bool g_panDragging = false;
bool g_leftMouseDown = false;
bool g_rightMouseDown = false;
bool g_middleMouseDown = false;
bool g_sceneMouseGesture = false;
std::array<bool, 256> g_keyWasDown{};
int g_lastMouseX = 0;
int g_lastMouseY = 0;
uint32_t g_pendingResizeWidth = 0U;
uint32_t g_pendingResizeHeight = 0U;

float ClampFloat(float value, float minimum, float maximum)
{
    return std::max(minimum, std::min(maximum, value));
}

XMFLOAT3 Add3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return XMFLOAT3(a.x + b.x, a.y + b.y, a.z + b.z);
}

XMFLOAT3 Subtract3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return XMFLOAT3(a.x - b.x, a.y - b.y, a.z - b.z);
}

XMFLOAT3 Multiply3(const XMFLOAT3& value, float scalar)
{
    return XMFLOAT3(value.x * scalar, value.y * scalar, value.z * scalar);
}

float Dot3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

XMFLOAT3 Cross3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return XMFLOAT3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float Length3(const XMFLOAT3& value)
{
    return std::sqrt(std::max(Dot3(value, value), 0.0f));
}

XMFLOAT3 Normalize3(const XMFLOAT3& value)
{
    const float length = Length3(value);
    if (length <= 1.0e-8f)
    {
        return XMFLOAT3(0.0f, 1.0f, 0.0f);
    }
    return Multiply3(value, 1.0f / length);
}

const EnvironmentGpu* SelectedEnvironment(const PreviewResources& resources, const PreviewState& state)
{
    if (resources.environments.empty()) return nullptr;
    const size_t index = std::min<size_t>(state.environmentIndex, resources.environments.size() - 1U);
    return &resources.environments[index];
}

std::string GetEnvironmentString(const char* name)
{
    const DWORD required = GetEnvironmentVariableA(name, nullptr, 0);
    if (required == 0)
    {
        return {};
    }

    std::vector<char> buffer(required, '\0');
    if (GetEnvironmentVariableA(name, buffer.data(), required) == 0)
    {
        return {};
    }
    return std::string(buffer.data());
}

std::wstring ToWidePath(const std::string& path)
{
    if (path.empty())
    {
        return {};
    }

    UINT codePage = CP_UTF8;
    DWORD flags = MB_ERR_INVALID_CHARS;
    int count = MultiByteToWideChar(codePage, flags, path.c_str(), -1, nullptr, 0);
    if (count <= 0)
    {
        codePage = CP_ACP;
        flags = 0;
        count = MultiByteToWideChar(codePage, flags, path.c_str(), -1, nullptr, 0);
    }
    if (count <= 0)
    {
        return {};
    }

    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(codePage, flags, path.c_str(), -1, &result[0], count);
    if (!result.empty() && result.back() == L'\0')
    {
        result.pop_back();
    }
    return result;
}


std::string ToLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::vector<std::string> SplitDelimited(const std::string& value, char delimiter)
{
    std::vector<std::string> result;
    std::string part;
    std::istringstream stream(value);
    while (std::getline(stream, part, delimiter))
    {
        result.push_back(part);
    }
    if (!value.empty() && value.back() == delimiter) result.emplace_back();
    return result;
}

std::string FileLabel(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1U);
}

std::vector<std::string> GetEnvironmentPaths()
{
    std::vector<std::string> paths;
    const std::string list = GetEnvironmentString("NSAMDR_ENVIRONMENTS");
    for (const std::string& value : SplitDelimited(list, ';'))
    {
        if (!value.empty() && std::find(paths.begin(), paths.end(), value) == paths.end()) paths.push_back(value);
    }
    const std::string primary = GetEnvironmentString("NSAMDR_ENVIRONMENT");
    if (!primary.empty() && std::find(paths.begin(), paths.end(), primary) == paths.end()) paths.insert(paths.begin(), primary);
    return paths;
}

std::string CatalogSelectionAsset(const ShipCatalog& catalog, const CatalogSelection& selection)
{
    if (selection.entryIndex >= catalog.entries.size()) return {};
    const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
    if (selection.variantIndex >= 0 && static_cast<size_t>(selection.variantIndex) < entry.variants.size())
    {
        return entry.variants[static_cast<size_t>(selection.variantIndex)];
    }
    return entry.preferredAsset;
}

std::string CatalogSelectionLabel(const ShipCatalog& catalog, const CatalogSelection& selection)
{
    if (selection.entryIndex >= catalog.entries.size()) return {};
    const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
    if (selection.variantIndex < 0) return entry.displayName;
    const std::string asset = CatalogSelectionAsset(catalog, selection);
    const size_t separator = asset.find_last_of("/\\");
    return entry.displayName + " - " + (separator == std::string::npos ? asset : asset.substr(separator + 1));
}

void RebuildCatalogFilter(ShipCatalog& catalog)
{
    std::string previousAsset;
    if (catalog.selectedFilteredIndex >= 0 &&
        static_cast<size_t>(catalog.selectedFilteredIndex) < catalog.filtered.size())
    {
        previousAsset = CatalogSelectionAsset(
            catalog,
            catalog.filtered[static_cast<size_t>(catalog.selectedFilteredIndex)]);
    }

    catalog.filtered.clear();
    const std::string search = ToLowerAscii(std::string(catalog.search.data()));
    for (size_t entryIndex = 0; entryIndex < catalog.entries.size(); ++entryIndex)
    {
        const ShipCatalogEntry& entry = catalog.entries[entryIndex];
        std::string searchable = entry.displayName + " " + entry.groupName + " " +
            entry.factionName + " " + entry.typeId + " " + entry.canonicalKey + " " + entry.preferredAsset;
        for (const std::string& variant : entry.variants) searchable += " " + variant;
        if (!search.empty() && ToLowerAscii(searchable).find(search) == std::string::npos) continue;

        if (catalog.showRawVariants && !entry.variants.empty())
        {
            for (size_t variantIndex = 0; variantIndex < entry.variants.size(); ++variantIndex)
            {
                catalog.filtered.push_back({entryIndex, static_cast<int>(variantIndex)});
            }
        }
        else
        {
            catalog.filtered.push_back({entryIndex, -1});
        }
    }

    catalog.selectedFilteredIndex = catalog.filtered.empty() ? -1 : 0;
    const std::string wanted = !previousAsset.empty() ? previousAsset : catalog.currentQuery;
    if (!wanted.empty())
    {
        const std::string loweredWanted = ToLowerAscii(wanted);
        for (size_t filteredIndex = 0; filteredIndex < catalog.filtered.size(); ++filteredIndex)
        {
            const CatalogSelection& selection = catalog.filtered[filteredIndex];
            const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
            const std::string asset = CatalogSelectionAsset(catalog, selection);
            bool matches = ToLowerAscii(asset) == loweredWanted ||
                ToLowerAscii(entry.canonicalKey) == loweredWanted;
            if (!matches && selection.variantIndex < 0)
            {
                matches = ToLowerAscii(entry.preferredAsset) == loweredWanted ||
                    std::any_of(entry.variants.begin(), entry.variants.end(), [&](const std::string& variant) {
                        return ToLowerAscii(variant) == loweredWanted;
                    });
            }
            if (matches)
            {
                catalog.selectedFilteredIndex = static_cast<int>(filteredIndex);
                break;
            }
        }
    }
}

bool LoadShipCatalog(const std::string& path, const std::string& currentQuery, ShipCatalog& catalog)
{
    catalog = ShipCatalog{};
    catalog.currentQuery = currentQuery;
    if (path.empty())
    {
        catalog.status = "Ship catalog is unavailable. Launch through the real EVE asset test script.";
        return false;
    }
    std::ifstream input(path);
    if (!input)
    {
        catalog.status = "Could not open ship catalog: " + path;
        return false;
    }

    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> fields = SplitDelimited(line, '\t');
        ShipCatalogEntry entry;
        if (fields.size() >= 7)
        {
            entry.displayName = fields[0];
            entry.groupName = fields[1];
            entry.factionName = fields[2];
            entry.typeId = fields[3] == "0" ? "" : fields[3];
            entry.canonicalKey = fields[4];
            entry.preferredAsset = fields[5];
            entry.variants = SplitDelimited(fields[6], '|');
            entry.variants.erase(
                std::remove_if(entry.variants.begin(), entry.variants.end(), [](const std::string& value) { return value.empty(); }),
                entry.variants.end());
        }
        else
        {
            // Legacy v1 catalog: one raw resource path per line.
            entry.displayName = line;
            entry.groupName = "Unmapped ship asset";
            entry.canonicalKey = line;
            entry.preferredAsset = line;
            entry.variants.push_back(line);
        }
        if (entry.displayName.empty()) entry.displayName = entry.preferredAsset;
        if (entry.preferredAsset.empty() && !entry.variants.empty()) entry.preferredAsset = entry.variants.front();
        if (entry.preferredAsset.empty()) continue;
        if (std::find(entry.variants.begin(), entry.variants.end(), entry.preferredAsset) == entry.variants.end())
        {
            entry.variants.insert(entry.variants.begin(), entry.preferredAsset);
        }
        catalog.entries.push_back(std::move(entry));
    }

    RebuildCatalogFilter(catalog);
    catalog.status = catalog.entries.empty()
        ? "No ship GR2 resources were found in the EVE cache index."
        : "Select a named ship. The highest-detail preferred asset is used unless raw variants are shown.";
    return !catalog.entries.empty();
}

std::wstring QuoteWindowsArgument(const std::wstring& value)
{
    std::wstring quoted = L"\"";
    for (wchar_t character : value)
    {
        if (character == L'\"') quoted += L'\\';
        quoted += character;
    }
    quoted += L"\"";
    return quoted;
}

bool LaunchCachedShip(const std::string& logicalAsset, const std::string& selectionKey, std::string& error)
{
    const std::string python = GetEnvironmentString("NSAMDR_PYTHON_EXE");
    const std::string tool = GetEnvironmentString("NSAMDR_EVE_TOOL");
    const std::string repoRoot = GetEnvironmentString("NSAMDR_EVE_REPO_ROOT");
    const std::string cacheRoot = GetEnvironmentString("NSAMDR_EVE_CACHE");
    const std::string launcher = GetEnvironmentString("NSAMDR_EVE_LAUNCHER");
    if (python.empty() || tool.empty() || repoRoot.empty() || cacheRoot.empty() || launcher.empty())
    {
        error = "The current viewer was not launched with the EVE cache selection environment.";
        return false;
    }

    const std::wstring command =
        QuoteWindowsArgument(ToWidePath(python)) + L" " +
        QuoteWindowsArgument(ToWidePath(tool)) + L" prepare-run --repo-root " +
        QuoteWindowsArgument(ToWidePath(repoRoot)) + L" --shared-cache " +
        QuoteWindowsArgument(ToWidePath(cacheRoot)) + L" --query " +
        QuoteWindowsArgument(ToWidePath(logicalAsset)) + L" --selection-key " +
        QuoteWindowsArgument(ToWidePath(selectionKey)) + L" --launcher " +
        QuoteWindowsArgument(ToWidePath(launcher));

    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const std::wstring workingDirectory = ToWidePath(repoRoot);
    const BOOL launched = CreateProcessW(
        nullptr,
        mutableCommand.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NEW_CONSOLE,
        nullptr,
        workingDirectory.empty() ? nullptr : workingDirectory.c_str(),
        &startup,
        &process);
    if (!launched)
    {
        error = "Could not launch the selected-ship converter. Windows error " + std::to_string(GetLastError()) + ".";
        return false;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return true;
}

void GetCameraBasis(const PreviewState& state, XMFLOAT3& right, XMFLOAT3& up, XMFLOAT3& forward)
{
    const float cosPitch = std::cos(state.orbitPitch);
    forward = Normalize3(XMFLOAT3(
        -std::sin(state.orbitYaw) * cosPitch,
        -std::sin(state.orbitPitch),
        std::cos(state.orbitYaw) * cosPitch));
    right = Normalize3(Cross3(XMFLOAT3(0.0f, 1.0f, 0.0f), forward));
    up = Normalize3(Cross3(forward, right));
}

void BuildCameraMatrices(
    const PreviewState& state,
    uint32_t width,
    uint32_t height,
    XMMATRIX& world,
    XMMATRIX& view,
    XMMATRIX& projection,
    XMFLOAT3& eye,
    XMFLOAT3& target)
{
    world =
        DirectX::XMMatrixRotationX(state.modelPitch) *
        DirectX::XMMatrixRotationY(state.modelYaw) *
        DirectX::XMMatrixRotationZ(state.modelRoll);
    target = XMFLOAT3(state.targetX, state.targetY, state.targetZ);
    const float cosPitch = std::cos(state.orbitPitch);
    eye = XMFLOAT3(
        target.x + std::sin(state.orbitYaw) * cosPitch * state.cameraDistance,
        target.y + std::sin(state.orbitPitch) * state.cameraDistance,
        target.z - std::cos(state.orbitYaw) * cosPitch * state.cameraDistance);
    view = DirectX::XMMatrixLookAtLH(
        DirectX::XMVectorSet(eye.x, eye.y, eye.z, 1.0f),
        DirectX::XMVectorSet(target.x, target.y, target.z, 1.0f),
        DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));
    const float aspect = height == 0U ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
    projection = DirectX::XMMatrixPerspectiveFovLH(
        DirectX::XMConvertToRadians(48.0f),
        aspect,
        std::max(state.nearClip, 0.0001f),
        std::max(state.farClip, state.nearClip + 1.0f));
}

bool RayTriangleIntersection(
    const XMFLOAT3& origin,
    const XMFLOAT3& direction,
    const XMFLOAT3& a,
    const XMFLOAT3& b,
    const XMFLOAT3& c,
    float& distance)
{
    const XMFLOAT3 edge1 = Subtract3(b, a);
    const XMFLOAT3 edge2 = Subtract3(c, a);
    const XMFLOAT3 p = Cross3(direction, edge2);
    const float determinant = Dot3(edge1, p);
    if (std::abs(determinant) < 1.0e-8f) return false;
    const float inverse = 1.0f / determinant;
    const XMFLOAT3 t = Subtract3(origin, a);
    const float u = Dot3(t, p) * inverse;
    if (u < 0.0f || u > 1.0f) return false;
    const XMFLOAT3 q = Cross3(t, edge1);
    const float v = Dot3(direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float hitDistance = Dot3(edge2, q) * inverse;
    if (hitDistance <= 0.0f) return false;
    distance = hitDistance;
    return true;
}

bool PickMeshAtScreenPoint(
    const PreviewState& state,
    const ObjMesh& mesh,
    uint32_t width,
    uint32_t height,
    int screenX,
    int screenY,
    XMFLOAT3& hitPoint,
    float& hitDistance)
{
    if (width == 0U || height == 0U) return false;
    const uint32_t viewportX = std::min(state.sceneViewportX, width > 1U ? width - 1U : 0U);
    const uint32_t viewportWidth = std::max(1U, width - viewportX);
    if (screenX < static_cast<int>(viewportX)) return false;
    const int viewportMouseX = screenX - static_cast<int>(viewportX);
    XMMATRIX world, view, projection;
    XMFLOAT3 eye, target;
    BuildCameraMatrices(state, viewportWidth, height, world, view, projection, eye, target);
    const XMVECTOR nearPoint = DirectX::XMVector3Unproject(
        DirectX::XMVectorSet(static_cast<float>(viewportMouseX), static_cast<float>(screenY), 0.0f, 1.0f),
        0.0f, 0.0f, static_cast<float>(viewportWidth), static_cast<float>(height), 0.0f, 1.0f,
        projection, view, DirectX::XMMatrixIdentity());
    const XMVECTOR farPoint = DirectX::XMVector3Unproject(
        DirectX::XMVectorSet(static_cast<float>(viewportMouseX), static_cast<float>(screenY), 1.0f, 1.0f),
        0.0f, 0.0f, static_cast<float>(viewportWidth), static_cast<float>(height), 0.0f, 1.0f,
        projection, view, DirectX::XMMatrixIdentity());
    XMFLOAT3 origin, farPosition;
    DirectX::XMStoreFloat3(&origin, nearPoint);
    DirectX::XMStoreFloat3(&farPosition, farPoint);
    const XMFLOAT3 direction = Normalize3(Subtract3(farPosition, origin));

    float nearest = std::numeric_limits<float>::max();
    XMFLOAT3 nearestPoint{};
    for (size_t triangle = 0; triangle + 2 < mesh.indices.size(); triangle += 3)
    {
        XMFLOAT3 worldPositions[3];
        for (size_t corner = 0; corner < 3; ++corner)
        {
            const Vertex& vertex = mesh.vertices[mesh.indices[triangle + corner]];
            DirectX::XMStoreFloat3(
                &worldPositions[corner],
                DirectX::XMVector3TransformCoord(DirectX::XMLoadFloat3(&vertex.position), world));
        }
        float distance = 0.0f;
        if (RayTriangleIntersection(origin, direction, worldPositions[0], worldPositions[1], worldPositions[2], distance) &&
            distance < nearest)
        {
            nearest = distance;
            nearestPoint = Add3(origin, Multiply3(direction, distance));
        }
    }
    if (nearest == std::numeric_limits<float>::max()) return false;
    hitPoint = nearestPoint;
    hitDistance = nearest;
    return true;
}

bool FocusCameraAtScreenPoint(
    PreviewState& state,
    const ObjMesh& mesh,
    uint32_t width,
    uint32_t height,
    int screenX,
    int screenY)
{
    XMFLOAT3 hitPoint{};
    float hitDistance = 0.0f;
    if (!PickMeshAtScreenPoint(state, mesh, width, height, screenX, screenY, hitPoint, hitDistance)) return false;
    state.targetX = hitPoint.x;
    state.targetY = hitPoint.y;
    state.targetZ = hitPoint.z;
    state.cameraDistance = std::max(0.02f, std::min(state.cameraDistance, hitDistance * 0.68f));
    return true;
}

void ApplyZoomRequest(
    PreviewState& state,
    const ObjMesh& mesh,
    uint32_t width,
    uint32_t height)
{
    if (!state.requestZoom) return;
    state.requestZoom = false;
    const float wheelSteps = state.zoomWheelSteps;
    state.zoomWheelSteps = 0.0f;
    if (std::abs(wheelSteps) < 1.0e-5f) return;

    if (wheelSteps > 0.0f)
    {
        XMFLOAT3 hitPoint{};
        float hitDistance = 0.0f;
        if (PickMeshAtScreenPoint(
                state,
                mesh,
                width,
                height,
                state.zoomMouseX,
                state.zoomMouseY,
                hitPoint,
                hitDistance))
        {
            const float blend = ClampFloat(1.0f - std::pow(0.72f, wheelSteps * state.zoomSpeed), 0.08f, 0.72f);
            state.targetX = state.targetX + (hitPoint.x - state.targetX) * blend;
            state.targetY = state.targetY + (hitPoint.y - state.targetY) * blend;
            state.targetZ = state.targetZ + (hitPoint.z - state.targetZ) * blend;
        }
    }

    const float maximumDistance = std::max(100.0f, mesh.boundsRadius * 100.0f);
    state.cameraDistance *= std::pow(0.84f, wheelSteps * state.zoomSpeed);
    state.cameraDistance = ClampFloat(state.cameraDistance, 0.01f, maximumDistance);
}

int ResolveObjIndex(int rawIndex, size_t count)
{
    if (rawIndex > 0)
    {
        const int resolved = rawIndex - 1;
        return resolved >= 0 && static_cast<size_t>(resolved) < count ? resolved : -1;
    }
    if (rawIndex < 0)
    {
        const int resolved = static_cast<int>(count) + rawIndex;
        return resolved >= 0 && static_cast<size_t>(resolved) < count ? resolved : -1;
    }
    return -1;
}

bool ParseIntegerPart(const std::string& text, int& value)
{
    if (text.empty())
    {
        value = 0;
        return true;
    }
    try
    {
        size_t consumed = 0;
        value = std::stoi(text, &consumed, 10);
        return consumed == text.size();
    }
    catch (...)
    {
        return false;
    }
}

bool ParseObjCornerToken(
    const std::string& token,
    size_t positionCount,
    size_t texcoordCount,
    size_t normalCount,
    ObjCorner& corner)
{
    std::array<std::string, 3> parts{};
    size_t partIndex = 0;
    size_t begin = 0;
    for (size_t i = 0; i <= token.size(); ++i)
    {
        if (i == token.size() || token[i] == '/')
        {
            if (partIndex >= parts.size())
            {
                return false;
            }
            parts[partIndex++] = token.substr(begin, i - begin);
            begin = i + 1;
        }
    }

    int rawPosition = 0;
    int rawTexcoord = 0;
    int rawNormal = 0;
    if (!ParseIntegerPart(parts[0], rawPosition) ||
        !ParseIntegerPart(parts[1], rawTexcoord) ||
        !ParseIntegerPart(parts[2], rawNormal))
    {
        return false;
    }

    corner.position = ResolveObjIndex(rawPosition, positionCount);
    corner.texcoord = rawTexcoord == 0 ? -1 : ResolveObjIndex(rawTexcoord, texcoordCount);
    corner.normal = rawNormal == 0 ? -1 : ResolveObjIndex(rawNormal, normalCount);
    return corner.position >= 0;
}

float ComputeTriangleDensityAndAnisotropy(
    const std::array<XMFLOAT3, 3>& p,
    const std::array<XMFLOAT2, 3>& uv,
    float& anisotropy,
    bool& degenerate)
{
    const XMFLOAT3 e1 = Subtract3(p[1], p[0]);
    const XMFLOAT3 e2 = Subtract3(p[2], p[0]);
    const float du1 = uv[1].x - uv[0].x;
    const float dv1 = uv[1].y - uv[0].y;
    const float du2 = uv[2].x - uv[0].x;
    const float dv2 = uv[2].y - uv[0].y;
    const float determinant = du1 * dv2 - dv1 * du2;

    if (std::abs(determinant) <= 1.0e-10f)
    {
        anisotropy = 1024.0f;
        degenerate = true;
        return 1024.0f;
    }

    const float inverse = 1.0f / determinant;
    const XMFLOAT3 dpdu = Multiply3(Subtract3(Multiply3(e1, dv2), Multiply3(e2, dv1)), inverse);
    const XMFLOAT3 dpdv = Multiply3(Add3(Multiply3(e1, -du2), Multiply3(e2, du1)), inverse);
    const float uLength = std::max(Length3(dpdu), 1.0e-8f);
    const float vLength = std::max(Length3(dpdv), 1.0e-8f);

    anisotropy = std::max(uLength, vLength) / std::max(std::min(uLength, vLength), 1.0e-8f);
    degenerate = false;
    return std::sqrt(uLength * vLength);
}

bool LoadObjMesh(const std::string& path, ObjMesh& mesh, std::string& error)
{
    std::ifstream input(path);
    if (!input)
    {
        error = "Could not open OBJ file: " + path;
        return false;
    }

    std::vector<XMFLOAT3> positions;
    std::vector<XMFLOAT2> texcoords;
    std::vector<XMFLOAT3> normals;
    std::vector<ObjTriangle> triangles;
    int currentGroupIndex = -1;
    std::string currentGroupName = "area_0";
    std::string currentMaterialName = "area_0";

    std::string line;
    size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::istringstream stream(line);
        std::string prefix;
        stream >> prefix;
        if (prefix.empty() || prefix[0] == '#')
        {
            continue;
        }

        if (prefix == "v")
        {
            XMFLOAT3 position{};
            if (!(stream >> position.x >> position.y >> position.z))
            {
                error = "Malformed OBJ position at line " + std::to_string(lineNumber);
                return false;
            }
            positions.push_back(position);
        }
        else if (prefix == "vt")
        {
            XMFLOAT2 uv{};
            if (!(stream >> uv.x >> uv.y))
            {
                error = "Malformed OBJ texture coordinate at line " + std::to_string(lineNumber);
                return false;
            }
            texcoords.push_back(uv);
        }
        else if (prefix == "vn")
        {
            XMFLOAT3 normal{};
            if (!(stream >> normal.x >> normal.y >> normal.z))
            {
                error = "Malformed OBJ normal at line " + std::to_string(lineNumber);
                return false;
            }
            normals.push_back(Normalize3(normal));
        }
        else if (prefix == "g")
        {
            ++currentGroupIndex;
            std::getline(stream, currentGroupName);
            currentGroupName.erase(0, currentGroupName.find_first_not_of(" 	"));
            if (currentGroupName.empty()) currentGroupName = "area_" + std::to_string(currentGroupIndex);
        }
        else if (prefix == "usemtl")
        {
            stream >> currentMaterialName;
            if (currentMaterialName.empty()) currentMaterialName = "area_" + std::to_string(std::max(currentGroupIndex, 0));
        }
        else if (prefix == "f")
        {
            std::vector<ObjCorner> polygon;
            std::string token;
            while (stream >> token)
            {
                ObjCorner corner;
                if (!ParseObjCornerToken(token, positions.size(), texcoords.size(), normals.size(), corner))
                {
                    error = "Malformed or out-of-range OBJ face index at line " + std::to_string(lineNumber);
                    return false;
                }
                polygon.push_back(corner);
            }

            if (polygon.size() < 3)
            {
                error = "OBJ face has fewer than three vertices at line " + std::to_string(lineNumber);
                return false;
            }

            for (size_t i = 1; i + 1 < polygon.size(); ++i)
            {
                ObjTriangle triangle;
                triangle.corners[0] = polygon[0];
                triangle.corners[1] = polygon[i];
                triangle.corners[2] = polygon[i + 1];
                triangle.groupIndex = std::max(currentGroupIndex, 0);
                triangle.groupName = currentGroupName;
                triangle.materialName = currentMaterialName;
                triangles.push_back(triangle);
            }
        }
    }

    if (positions.empty() || triangles.empty())
    {
        error = "OBJ file contains no renderable geometry: " + path;
        return false;
    }
    if (texcoords.empty())
    {
        error = "OBJ file contains no texture coordinates; NSAMDR stretch analysis requires UVs.";
        return false;
    }

    XMFLOAT3 minimum(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    XMFLOAT3 maximum(
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max());

    for (const XMFLOAT3& position : positions)
    {
        minimum.x = std::min(minimum.x, position.x);
        minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z);
        maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y);
        maximum.z = std::max(maximum.z, position.z);
    }

    const XMFLOAT3 centre = Multiply3(Add3(minimum, maximum), 0.5f);
    const XMFLOAT3 extent = Subtract3(maximum, minimum);
    const float maximumExtent = std::max(extent.x, std::max(extent.y, extent.z));
    if (maximumExtent <= 1.0e-8f)
    {
        error = "OBJ geometry has zero extent.";
        return false;
    }

    const float normalizationScale = 5.2f / maximumExtent;
    for (XMFLOAT3& position : positions)
    {
        position = Multiply3(Subtract3(position, centre), normalizationScale);
    }

    std::vector<TriangleBuildData> buildTriangles;
    buildTriangles.reserve(triangles.size());
    std::vector<float> validDensities;
    validDensities.reserve(triangles.size());

    for (const ObjTriangle& sourceTriangle : triangles)
    {
        TriangleBuildData triangle;
        triangle.groupIndex = sourceTriangle.groupIndex;
        triangle.groupName = sourceTriangle.groupName;
        triangle.materialName = sourceTriangle.materialName;
        bool hasAllNormals = true;
        bool hasAllUvs = true;

        for (size_t cornerIndex = 0; cornerIndex < 3; ++cornerIndex)
        {
            const ObjCorner& sourceCorner = sourceTriangle.corners[cornerIndex];
            triangle.positions[cornerIndex] = positions[static_cast<size_t>(sourceCorner.position)];

            if (sourceCorner.texcoord >= 0)
            {
                triangle.uvs[cornerIndex] = texcoords[static_cast<size_t>(sourceCorner.texcoord)];
            }
            else
            {
                triangle.uvs[cornerIndex] = XMFLOAT2(0.0f, 0.0f);
                hasAllUvs = false;
            }

            if (sourceCorner.normal >= 0)
            {
                triangle.normals[cornerIndex] = normals[static_cast<size_t>(sourceCorner.normal)];
            }
            else
            {
                hasAllNormals = false;
            }
        }

        const XMFLOAT3 edge1 = Subtract3(triangle.positions[1], triangle.positions[0]);
        const XMFLOAT3 edge2 = Subtract3(triangle.positions[2], triangle.positions[0]);
        const XMFLOAT3 faceNormal = Normalize3(Cross3(edge1, edge2));
        if (!hasAllNormals)
        {
            triangle.normals = {faceNormal, faceNormal, faceNormal};
        }

        if (!hasAllUvs)
        {
            triangle.degenerateUv = true;
            triangle.density = 1024.0f;
            triangle.anisotropy = 1024.0f;
        }
        else
        {
            triangle.density = ComputeTriangleDensityAndAnisotropy(
                triangle.positions,
                triangle.uvs,
                triangle.anisotropy,
                triangle.degenerateUv);
        }

        if (!triangle.degenerateUv && std::isfinite(triangle.density) && triangle.density > 0.0f)
        {
            validDensities.push_back(triangle.density);
        }
        buildTriangles.push_back(triangle);
    }

    float medianDensity = 1.0f;
    if (!validDensities.empty())
    {
        const size_t middle = validDensities.size() / 2;
        std::nth_element(validDensities.begin(), validDensities.begin() + middle, validDensities.end());
        medianDensity = std::max(validDensities[middle], 1.0e-8f);
    }

    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.drawRanges.clear();
    mesh.vertices.reserve(buildTriangles.size() * 3U);
    mesh.indices.reserve(buildTriangles.size() * 3U);
    mesh.degenerateUvTriangles = 0;
    mesh.averageStretch = 0.0f;
    mesh.maximumStretch = 0.0f;

    for (const TriangleBuildData& triangle : buildTriangles)
    {
        if (mesh.drawRanges.empty() ||
            mesh.drawRanges.back().groupIndex != triangle.groupIndex ||
            mesh.drawRanges.back().materialName != triangle.materialName)
        {
            ObjDrawRange range;
            range.groupIndex = triangle.groupIndex;
            range.groupName = triangle.groupName;
            range.materialName = triangle.materialName;
            range.startIndex = static_cast<uint32_t>(mesh.indices.size());
            mesh.drawRanges.push_back(std::move(range));
        }

        float stretch = 1.0f;
        if (!triangle.degenerateUv)
        {
            const float anisotropyScore = std::log2(std::max(triangle.anisotropy, 1.0f));
            const float densityScore = std::abs(std::log2(std::max(triangle.density / medianDensity, 1.0e-8f)));
            stretch = ClampFloat(std::max(anisotropyScore / 4.0f, densityScore / 4.0f), 0.0f, 1.0f);
        }
        else
        {
            ++mesh.degenerateUvTriangles;
        }

        mesh.averageStretch += stretch;
        mesh.maximumStretch = std::max(mesh.maximumStretch, stretch);

        for (size_t cornerIndex = 0; cornerIndex < 3; ++cornerIndex)
        {
            const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back({
                triangle.positions[cornerIndex],
                triangle.normals[cornerIndex],
                triangle.uvs[cornerIndex],
                stretch});
            mesh.indices.push_back(index);
        }
        mesh.drawRanges.back().indexCount += 3U;
    }

    mesh.triangleCount = static_cast<uint32_t>(buildTriangles.size());
    if (mesh.triangleCount > 0)
    {
        mesh.averageStretch /= static_cast<float>(mesh.triangleCount);
    }
    mesh.sourcePositionCount = static_cast<uint32_t>(positions.size());
    mesh.sourceTexcoordCount = static_cast<uint32_t>(texcoords.size());
    mesh.sourceNormalCount = static_cast<uint32_t>(normals.size());

    if (!mesh.vertices.empty())
    {
        XMFLOAT3 minimum = mesh.vertices.front().position;
        XMFLOAT3 maximum = mesh.vertices.front().position;
        for (const Vertex& vertex : mesh.vertices)
        {
            minimum.x = std::min(minimum.x, vertex.position.x);
            minimum.y = std::min(minimum.y, vertex.position.y);
            minimum.z = std::min(minimum.z, vertex.position.z);
            maximum.x = std::max(maximum.x, vertex.position.x);
            maximum.y = std::max(maximum.y, vertex.position.y);
            maximum.z = std::max(maximum.z, vertex.position.z);
        }
        mesh.boundsCenter = Multiply3(Add3(minimum, maximum), 0.5f);
        mesh.boundsRadius = 0.0f;
        for (const Vertex& vertex : mesh.vertices)
        {
            mesh.boundsRadius = std::max(mesh.boundsRadius, Length3(Subtract3(vertex.position, mesh.boundsCenter)));
        }
        mesh.boundsRadius = std::max(mesh.boundsRadius, 0.01f);
    }
    mesh.path = path;
    return true;
}


HWND GetPreviewRootWindow(HWND hwnd)
{
    HWND root = GetAncestor(hwnd, GA_ROOT);
    return root != nullptr ? root : hwnd;
}

bool IsPreviewWindowFocused(HWND hwnd)
{
    if (hwnd == nullptr || !IsWindow(hwnd)) return false;
    return GetForegroundWindow() == GetPreviewRootWindow(hwnd);
}

bool IsPreviewMouseInputMessage(UINT message)
{
    return message >= WM_MOUSEFIRST && message <= WM_MOUSELAST;
}

bool IsPreviewKeyboardInputMessage(UINT message)
{
    return (message >= WM_KEYFIRST && message <= WM_KEYLAST) ||
           message == WM_CHAR ||
           message == WM_DEADCHAR ||
           message == WM_SYSCHAR ||
           message == WM_SYSDEADCHAR ||
           message == WM_UNICHAR;
}

bool IsPreviewInputMessage(UINT message)
{
    return IsPreviewMouseInputMessage(message) ||
           IsPreviewKeyboardInputMessage(message);
}

bool AnyMouseButtonPhysicallyDown()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_XBUTTON1) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_XBUTTON2) & 0x8000) != 0;
}

void ResetSceneInput(HWND hwnd)
{
    g_leftMouseDown = false;
    g_rightMouseDown = false;
    g_middleMouseDown = false;
    g_sceneMouseGesture = false;
    g_orbitDragging = false;
    g_panDragging = false;

    if (GetCapture() == hwnd)
    {
        ReleaseCapture();
    }

    if (g_previewState != nullptr)
    {
        g_previewState->requestFocus = false;
        g_previewState->requestZoom = false;
        g_previewState->zoomWheelSteps = 0.0f;
    }
}

void SynchronizeHotkeyState()
{
    for (size_t index = 0; index < g_keyWasDown.size(); ++index)
    {
        g_keyWasDown[index] = (GetAsyncKeyState(static_cast<int>(index)) & 0x8000) != 0;
    }
}

void SetPreviewInputFocus(HWND hwnd, bool focused)
{
    if (g_previewInputFocused == focused) return;

    g_previewInputFocused = focused;
    if (!focused)
    {
        ResetSceneInput(hwnd);
        g_keyWasDown.fill(false);
        g_blockMouseUntilRelease = false;
        return;
    }

    // A key already held while another application was active must not become
    // a fresh preview hotkey when focus returns.
    SynchronizeHotkeyState();

    // WM_MOUSEACTIVATE sets this before focus changes when the activating click
    // lands in the client area. Preserve it so that click only activates the
    // window; it cannot also orbit, pan, zoom, focus, or operate the UI.
    g_blockMouseUntilRelease = g_blockMouseUntilRelease || AnyMouseButtonPhysicallyDown();
}

void RefreshPreviewInputFocus(HWND hwnd)
{
    SetPreviewInputFocus(hwnd, IsPreviewWindowFocused(hwnd));
}

bool MouseMessageHasAnyButtonDown(WPARAM buttonFlags)
{
    return (buttonFlags & (MK_LBUTTON | MK_RBUTTON | MK_MBUTTON | MK_XBUTTON1 | MK_XBUTTON2)) != 0 ||
           AnyMouseButtonPhysicallyDown();
}

bool IsMousePointInScene(HWND hwnd, int x, int y)
{
    if (g_previewState == nullptr) return false;
    RECT client{};
    if (!GetClientRect(hwnd, &client)) return false;
    return x >= static_cast<int>(g_previewState->sceneViewportX) &&
           x < client.right && y >= client.top && y < client.bottom;
}

void RefreshMouseButtonsFromMove(WPARAM buttonFlags)
{
    // WM_MOUSEMOVE carries the authoritative button chord. GetAsyncKeyState is
    // included as a recovery path when a button transition was consumed by the
    // host window procedure before this subclass saw it.
    g_leftMouseDown = (buttonFlags & MK_LBUTTON) != 0 || (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    g_rightMouseDown = (buttonFlags & MK_RBUTTON) != 0 || (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    g_middleMouseDown = (buttonFlags & MK_MBUTTON) != 0 || (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0;
}

void UpdateMouseDragMode(HWND hwnd, int x, int y)
{
    const bool chordPan = g_sceneMouseGesture && g_leftMouseDown && g_rightMouseDown;
    const bool shouldPan = g_sceneMouseGesture && (g_middleMouseDown || chordPan);
    const bool shouldOrbit = g_sceneMouseGesture && g_rightMouseDown && !shouldPan;
    const bool modeChanged = shouldPan != g_panDragging || shouldOrbit != g_orbitDragging;

    g_panDragging = shouldPan;
    g_orbitDragging = shouldOrbit;
    if (modeChanged)
    {
        // Rebase the cursor when changing between orbit and pan so the second
        // button cannot inject a stale movement delta. Do not rebase every move.
        g_lastMouseX = x;
        g_lastMouseY = y;
    }

    if (shouldPan || shouldOrbit)
    {
        if (GetCapture() != hwnd) SetCapture(hwnd);
    }
    else if (GetCapture() == hwnd)
    {
        ReleaseCapture();
    }
}

LRESULT CALLBACK NSAMDRPreviewWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    const bool activatingClientClick =
        message == WM_MOUSEACTIVATE &&
        !IsPreviewWindowFocused(hwnd) &&
        LOWORD(lParam) == HTCLIENT;

    switch (message)
    {
    case WM_ACTIVATEAPP:
        SetPreviewInputFocus(hwnd, wParam != FALSE);
        break;
    case WM_ACTIVATE:
        SetPreviewInputFocus(hwnd, LOWORD(wParam) != WA_INACTIVE);
        break;
    case WM_SETFOCUS:
        SetPreviewInputFocus(hwnd, true);
        break;
    case WM_KILLFOCUS:
        SetPreviewInputFocus(hwnd, false);
        break;
    default:
        RefreshPreviewInputFocus(hwnd);
        break;
    }

    // A click in an inactive client area should activate the preview only. The
    // same physical click must not also manipulate the camera or ImGui controls.
    if (activatingClientClick)
    {
        g_blockMouseUntilRelease = true;
    }

    const bool inputFocused = g_previewInputFocused && IsPreviewWindowFocused(hwnd);
    const bool mouseInput = IsPreviewMouseInputMessage(message);
    const bool previewInput = IsPreviewInputMessage(message);

    // Consume the complete mouse gesture that activated the window. Clearing the
    // block is deferred until every mouse button has been released.
    const bool mouseWasBlocked = mouseInput && inputFocused && g_blockMouseUntilRelease;
    if (mouseWasBlocked)
    {
        if ((message == WM_MOUSEMOVE ||
             message == WM_LBUTTONUP ||
             message == WM_RBUTTONUP ||
             message == WM_MBUTTONUP ||
             message == WM_XBUTTONUP) &&
            !MouseMessageHasAnyButtonDown(wParam))
        {
            g_blockMouseUntilRelease = false;
        }
    }

    LRESULT imguiResult = 0;
    if (ImGui::GetCurrentContext() != nullptr &&
        (!previewInput || (inputFocused && !mouseWasBlocked)))
    {
        imguiResult = ImGui_ImplWin32_WndProcHandler(hwnd, message, wParam, lParam);
    }
    const bool imguiWantsMouse =
        ImGui::GetCurrentContext() != nullptr &&
        ImGui::GetIO().WantCaptureMouse;

    // GetAsyncKeyState and mouse capture can otherwise make the preview react to
    // input intended for another application. Focus and activation messages are
    // still passed through, but render/UI input is discarded while inactive.
    if (previewInput && (!inputFocused || mouseWasBlocked))
    {
        return 0;
    }

    if (g_previewState != nullptr)
    {
        switch (message)
        {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED)
            {
                g_pendingResizeWidth = static_cast<uint32_t>(LOWORD(lParam));
                g_pendingResizeHeight = static_cast<uint32_t>(HIWORD(lParam));
            }
            break;
        case WM_LBUTTONDBLCLK:
            if (!imguiWantsMouse)
            {
                g_previewState->focusMouseX = GET_X_LPARAM(lParam);
                g_previewState->focusMouseY = GET_Y_LPARAM(lParam);
                g_previewState->requestFocus = true;
                return 0;
            }
            break;
        case WM_LBUTTONDOWN:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            if (g_sceneMouseGesture || IsMousePointInScene(hwnd, x, y))
            {
                g_sceneMouseGesture = true;
                g_leftMouseDown = true;
                UpdateMouseDragMode(hwnd, x, y);
                // Keep scene clicks away from the host procedure. Passing the first
                // button through allowed it to steal capture before the chord formed.
                return 0;
            }
            break;
        }
        case WM_RBUTTONDOWN:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            if (g_sceneMouseGesture || IsMousePointInScene(hwnd, x, y))
            {
                g_sceneMouseGesture = true;
                g_rightMouseDown = true;
                UpdateMouseDragMode(hwnd, x, y);
                return 0;
            }
            break;
        }
        case WM_MBUTTONDOWN:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            if (g_sceneMouseGesture || IsMousePointInScene(hwnd, x, y))
            {
                g_sceneMouseGesture = true;
                g_middleMouseDown = true;
                UpdateMouseDragMode(hwnd, x, y);
                return 0;
            }
            break;
        }
        case WM_LBUTTONUP:
            if (g_sceneMouseGesture)
            {
                g_leftMouseDown = false;
                UpdateMouseDragMode(hwnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
                if (!g_rightMouseDown && !g_middleMouseDown) g_sceneMouseGesture = false;
                return 0;
            }
            break;
        case WM_RBUTTONUP:
            if (g_sceneMouseGesture)
            {
                g_rightMouseDown = false;
                UpdateMouseDragMode(hwnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
                if (!g_leftMouseDown && !g_middleMouseDown) g_sceneMouseGesture = false;
                return 0;
            }
            break;
        case WM_MBUTTONUP:
            if (g_sceneMouseGesture)
            {
                g_middleMouseDown = false;
                UpdateMouseDragMode(hwnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
                if (!g_leftMouseDown && !g_rightMouseDown) g_sceneMouseGesture = false;
                return 0;
            }
            break;
        case WM_MOUSEMOVE:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            RefreshMouseButtonsFromMove(wParam);

            // Recover a chord even when one of its button-down messages was routed
            // through ImGui or the host procedure. Physical button state on motion
            // is sufficient to arm the scene gesture, but only while this preview
            // is the foreground window.
            const bool requestedDrag = g_middleMouseDown || g_rightMouseDown;
            if (!g_sceneMouseGesture && requestedDrag && IsMousePointInScene(hwnd, x, y))
            {
                g_sceneMouseGesture = true;
            }
            UpdateMouseDragMode(hwnd, x, y);

            if (g_orbitDragging || g_panDragging)
            {
                const int deltaX = x - g_lastMouseX;
                const int deltaY = y - g_lastMouseY;
                g_lastMouseX = x;
                g_lastMouseY = y;

                if (g_orbitDragging)
                {
                    const float sensitivity = 0.006f * g_previewState->orbitSensitivity;
                    g_previewState->orbitYaw += static_cast<float>(deltaX) * sensitivity;
                    g_previewState->orbitPitch = ClampFloat(
                        g_previewState->orbitPitch + static_cast<float>(deltaY) * sensitivity,
                        -1.45f,
                        1.45f);
                }
                if (g_panDragging)
                {
                    XMFLOAT3 right, up, forward;
                    GetCameraBasis(*g_previewState, right, up, forward);
                    const float fine = (GetKeyState(VK_SHIFT) & 0x8000) != 0 ? 0.22f : 1.0f;
                    const float scale = g_previewState->cameraDistance * 0.0016f * g_previewState->panSpeed * fine;
                    const XMFLOAT3 movement = Add3(
                        Multiply3(right, -static_cast<float>(deltaX) * scale),
                        Multiply3(up, static_cast<float>(deltaY) * scale));
                    g_previewState->targetX += movement.x;
                    g_previewState->targetY += movement.y;
                    g_previewState->targetZ += movement.z;
                }
                return 0;
            }

            if (g_sceneMouseGesture && !g_leftMouseDown && !g_rightMouseDown && !g_middleMouseDown)
            {
                g_sceneMouseGesture = false;
            }
            break;
        }
        case WM_MOUSEWHEEL:
            if (!imguiWantsMouse)
            {
                POINT cursor{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
                ScreenToClient(hwnd, &cursor);
                const float wheelSteps =
                    static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) /
                    static_cast<float>(WHEEL_DELTA);
                const float fine = (GetKeyState(VK_CONTROL) & 0x8000) != 0 ? 0.22f : 1.0f;
                g_previewState->requestZoom = true;
                g_previewState->zoomWheelSteps += wheelSteps * fine;
                g_previewState->zoomMouseX = cursor.x;
                g_previewState->zoomMouseY = cursor.y;
                return 0;
            }
            break;
        case WM_CAPTURECHANGED:
        case WM_CANCELMODE:
            ResetSceneInput(hwnd);
            break;
        default:
            break;
        }
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
    if (imguiResult != 0)
    {
        return imguiResult;
    }

    return CallWindowProc(g_previousWindowProc, hwnd, message, wParam, lParam);
}

bool KeyPressed(int virtualKey)
{
    if (g_previewWindow == nullptr ||
        !g_previewInputFocused ||
        !IsPreviewWindowFocused(g_previewWindow) ||
        virtualKey < 0 ||
        virtualKey >= static_cast<int>(g_keyWasDown.size()))
    {
        return false;
    }

    const bool down = (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
    const bool pressed = down && !g_keyWasDown[static_cast<size_t>(virtualKey)];
    g_keyWasDown[static_cast<size_t>(virtualKey)] = down;
    return pressed;
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
        std::ostringstream diagnostic;
        diagnostic << "NSAMDR HLSL compilation failed for " << entryPoint << " (" << profile << ")\n" << errorText;
        const std::string diagnosticText = diagnostic.str();
        std::fprintf(stderr, "%s\n", diagnosticText.c_str());
        OutputDebugStringA(diagnosticText.c_str());
        ADD_FAILURE() << diagnosticText;
        return false;
    }
    return true;
}

bool CreateFallbackTexture(ID3D11Device* device, PreviewResources& resources)
{
    const uint32_t pixels[4] = {
        0xff251b10U,
        0xff332419U,
        0xff332419U,
        0xff251b10U,
    };

    D3D11_TEXTURE2D_DESC textureDescription{};
    textureDescription.Width = 2;
    textureDescription.Height = 2;
    textureDescription.MipLevels = 1;
    textureDescription.ArraySize = 1;
    textureDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDescription.SampleDesc.Count = 1;
    textureDescription.Usage = D3D11_USAGE_IMMUTABLE;
    textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initialData{};
    initialData.pSysMem = pixels;
    initialData.SysMemPitch = 2U * sizeof(uint32_t);

    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(device->CreateTexture2D(&textureDescription, &initialData, texture.GetAddressOf())) ||
        FAILED(device->CreateShaderResourceView(texture.Get(), nullptr, resources.albedoView.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create fallback NSAMDR texture";
        return false;
    }

    resources.textureWidth = 2;
    resources.textureHeight = 2;
    resources.hasExternalAlbedo = false;
    return true;
}

ShaderFamily ParseShaderFamily(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "legacy_pgs") return ShaderFamily::LegacyPgs;
    if (lower == "v5_separate") return ShaderFamily::V5Separate;
    if (lower == "v5_packed") return ShaderFamily::V5Packed;
    return ShaderFamily::Unknown;
}

MaterialPass ParseMaterialPass(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "decal") return MaterialPass::Decal;
    if (lower == "transparent") return MaterialPass::Transparent;
    if (lower == "additive") return MaterialPass::Additive;
    return MaterialPass::Opaque;
}

std::vector<std::string> SplitTabs(const std::string& line)
{
    std::vector<std::string> values;
    size_t start = 0;
    for (;;)
    {
        const size_t separator = line.find('\t', start);
        values.push_back(line.substr(start, separator == std::string::npos ? std::string::npos : separator - start));
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    return values;
}

bool ParseFloatValue(const std::string& text, float& value)
{
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || (end && *end != '\0') || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

bool ParseIntValue(const std::string& text, int& value)
{
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || (end && *end != '\0')) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool LoadAreaMaterialSources(const std::string& path, std::vector<AreaMaterialSource>& materials, std::string& error)
{
    materials.clear();
    if (path.empty()) return true;
    std::ifstream input(path);
    if (!input)
    {
        error = "Could not open SOF material manifest: " + path;
        return false;
    }

    std::map<std::string, size_t> columns;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> values = SplitTabs(line);
        if (columns.empty())
        {
            for (size_t index = 0; index < values.size(); ++index) columns[values[index]] = index;
            continue;
        }

        auto value = [&](const char* name, const char* alias = nullptr) -> std::string {
            auto it = columns.find(name);
            if (it == columns.end() && alias) it = columns.find(alias);
            return it != columns.end() && it->second < values.size() ? values[it->second] : std::string();
        };
        auto parse = [&](const char* name, float& destination, const char* alias = nullptr) -> bool {
            const std::string text = value(name, alias);
            return text.empty() || ParseFloatValue(text, destination);
        };

        AreaMaterialSource material;
        try
        {
            const std::string group = value("group");
            if (group.empty()) throw std::runtime_error("missing group");
            material.groupIndex = std::stoi(group);
        }
        catch (...)
        {
            error = "Malformed group index in SOF material manifest";
            return false;
        }
        material.pass = ParseMaterialPass(value("pass"));
        material.areaType = value("area_type");
        material.areaName = value("area_name");
        material.shaderPath = value("shader");
        material.shaderFamily = ParseShaderFamily(value("shader_family"));
        material.albedoPath = value("albedo");
        material.normalPath = value("normal");
        material.pgsPath = value("material", "pgs");
        material.glowPath = value("glow");
        material.dirtPath = value("dirt");
        material.aoPath = value("ao");
        material.paintMaskPath = value("paint_mask");
        material.roughnessMapPath = value("roughness_map");

        auto parseChannel = [&](const char* name, int& destination) -> bool {
            const std::string text = value(name);
            return text.empty() || ParseIntValue(text, destination);
        };
        if (!parseChannel("normal_x_channel", material.normalXChannel) ||
            !parseChannel("normal_y_channel", material.normalYChannel) ||
            !parseChannel("roughness_channel", material.roughnessChannel) ||
            !parseChannel("material_channel", material.materialChannel) ||
            !parseChannel("ao_channel", material.aoChannel) ||
            !parseChannel("paint_channel", material.paintChannel) ||
            !parseChannel("dirt_channel", material.dirtChannel) ||
            !parseChannel("glow_channel", material.glowChannel))
        {
            error = "Malformed semantic channel in SOF material manifest";
            return false;
        }

        if (!parse("tint_r", material.tint.x) ||
            !parse("tint_g", material.tint.y) ||
            !parse("tint_b", material.tint.z) ||
            !parse("detail_scale", material.detailScale) ||
            !parse("roughness", material.roughness) ||
            !parse("specular", material.specular) ||
            !parse("alpha", material.alpha) ||
            !parse("glow_r", material.glowColour.x) ||
            !parse("glow_g", material.glowColour.y) ||
            !parse("glow_b", material.glowColour.z))
        {
            error = "Malformed numeric value in SOF material manifest";
            return false;
        }

        const std::array<XMFLOAT3, 4> fallbackColours = {
            material.tint,
            XMFLOAT3(material.tint.x * 1.15f, material.tint.y * 1.15f, material.tint.z * 1.15f),
            XMFLOAT3(material.tint.x * 0.62f, material.tint.y * 0.62f, material.tint.z * 0.62f),
            XMFLOAT3(0.035f, 0.045f, 0.055f),
        };
        for (size_t slotIndex = 0; slotIndex < material.slots.size(); ++slotIndex)
        {
            MaterialSlotSource& slot = material.slots[slotIndex];
            slot.colour = fallbackColours[slotIndex];
            const float legacyF0 = std::max(0.018f, std::min(0.22f, 0.04f * material.specular));
            slot.f0 = XMFLOAT3(legacyF0, legacyF0, legacyF0);
            slot.gloss = std::max(0.0f, 1.0f - material.roughness);

            const std::string prefix = "mtl" + std::to_string(slotIndex + 1U) + "_";
            if (!parse((prefix + "r").c_str(), slot.colour.x) ||
                !parse((prefix + "g").c_str(), slot.colour.y) ||
                !parse((prefix + "b").c_str(), slot.colour.z) ||
                !parse((prefix + "f0_r").c_str(), slot.f0.x) ||
                !parse((prefix + "f0_g").c_str(), slot.f0.y) ||
                !parse((prefix + "f0_b").c_str(), slot.f0.z))
            {
                error = "Malformed material-slot value in SOF material manifest";
                return false;
            }
            const std::string glossText = value((prefix + "gloss").c_str());
            if (!glossText.empty())
            {
                if (!ParseFloatValue(glossText, slot.gloss))
                {
                    error = "Malformed material gloss in SOF material manifest";
                    return false;
                }
            }
            else
            {
                float legacyRoughness = 1.0f - slot.gloss;
                const std::string roughnessText = value((prefix + "roughness").c_str());
                if (!roughnessText.empty() && !ParseFloatValue(roughnessText, legacyRoughness))
                {
                    error = "Malformed legacy material roughness in SOF material manifest";
                    return false;
                }
                slot.gloss = std::max(0.0f, 1.0f - legacyRoughness);
            }
        }
        parse("general_data_x", material.generalDataX);
        const std::string semanticComplete = value("semantic_complete");
        material.semanticComplete = semanticComplete == "1" || semanticComplete == "true" || semanticComplete == "TRUE";
        const std::string baselineComplete = value("baseline_complete");
        material.baselineComplete = baselineComplete == "1" || baselineComplete == "true" || baselineComplete == "TRUE";
        material.unresolvedSemantics = value("unresolved_semantics");
        const std::string unresolvedCount = value("unresolved_count");
        if (!unresolvedCount.empty())
        {
            try { material.unresolvedCount = std::max(0, std::stoi(unresolvedCount)); }
            catch (...) { material.unresolvedCount = 1; }
        }
        materials.push_back(std::move(material));
    }
    if (columns.empty())
    {
        error = "SOF material manifest has no column header";
        return false;
    }
    return true;
}

bool LoadWicTexture(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const std::string& path,
    bool srgb,
    ComPtr<ID3D11ShaderResourceView>& view,
    uint32_t& outputWidth,
    uint32_t& outputHeight,
    const char* label)
{
    const std::wstring widePath = ToWidePath(path);
    if (widePath.empty())
    {
        ADD_FAILURE() << "Could not convert " << label << " path to a Windows path: " << path;
        return false;
    }

    const HRESULT initialiseResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool mustUninitialise = SUCCEEDED(initialiseResult);
    if (FAILED(initialiseResult) && initialiseResult != RPC_E_CHANGED_MODE)
    {
        ADD_FAILURE() << "COM initialisation failed while loading " << label;
        return false;
    }

    bool success = false;
    do
    {
        ComPtr<IWICImagingFactory> factory;
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(factory.GetAddressOf()))))
        {
            ADD_FAILURE() << "Could not create Windows Imaging Component factory";
            break;
        }

        ComPtr<IWICBitmapDecoder> decoder;
        if (FAILED(factory->CreateDecoderFromFilename(widePath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, decoder.GetAddressOf())))
        {
            ADD_FAILURE() << "Could not decode " << label << ". Use PNG, JPG, BMP or TIFF: " << path;
            break;
        }

        ComPtr<IWICBitmapFrameDecode> frame;
        if (FAILED(decoder->GetFrame(0, frame.GetAddressOf())))
        {
            ADD_FAILURE() << "Could not read first frame from " << label << ": " << path;
            break;
        }

        UINT width = 0;
        UINT height = 0;
        if (FAILED(frame->GetSize(&width, &height)) || width == 0 || height == 0)
        {
            ADD_FAILURE() << label << " has invalid dimensions: " << path;
            break;
        }

        ComPtr<IWICFormatConverter> converter;
        if (FAILED(factory->CreateFormatConverter(converter.GetAddressOf())) ||
            FAILED(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)))
        {
            ADD_FAILURE() << "Could not convert " << label << " to RGBA8: " << path;
            break;
        }

        const size_t rowPitch = static_cast<size_t>(width) * 4U;
        const size_t byteCount = rowPitch * static_cast<size_t>(height);
        if (byteCount > static_cast<size_t>(std::numeric_limits<UINT>::max()))
        {
            ADD_FAILURE() << label << " is too large: " << path;
            break;
        }

        std::vector<uint8_t> pixels(byteCount);
        if (FAILED(converter->CopyPixels(nullptr, static_cast<UINT>(rowPitch), static_cast<UINT>(byteCount), pixels.data())))
        {
            ADD_FAILURE() << "Could not copy " << label << " pixels: " << path;
            break;
        }

        D3D11_TEXTURE2D_DESC textureDescription{};
        textureDescription.Width = width;
        textureDescription.Height = height;
        textureDescription.MipLevels = 0;
        textureDescription.ArraySize = 1;
        textureDescription.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        textureDescription.SampleDesc.Count = 1;
        textureDescription.Usage = D3D11_USAGE_DEFAULT;
        textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        textureDescription.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

        ComPtr<ID3D11Texture2D> texture;
        if (FAILED(device->CreateTexture2D(&textureDescription, nullptr, texture.GetAddressOf())))
        {
            ADD_FAILURE() << "Could not create DirectX " << label << " texture: " << path;
            break;
        }
        context->UpdateSubresource(texture.Get(), 0, nullptr, pixels.data(), static_cast<UINT>(rowPitch), 0);

        D3D11_SHADER_RESOURCE_VIEW_DESC viewDescription{};
        viewDescription.Format = srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
        viewDescription.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        viewDescription.Texture2D.MostDetailedMip = 0;
        viewDescription.Texture2D.MipLevels = UINT(-1);
        if (FAILED(device->CreateShaderResourceView(texture.Get(), &viewDescription, view.ReleaseAndGetAddressOf())))
        {
            ADD_FAILURE() << "Could not create DirectX " << label << " texture view: " << path;
            break;
        }
        context->GenerateMips(view.Get());

        outputWidth = width;
        outputHeight = height;
        success = true;
    } while (false);

    if (mustUninitialise) CoUninitialize();
    return success;
}

bool EnsureSelectedEnvironmentLoaded(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    PreviewResources& resources,
    const PreviewState& state)
{
    if (resources.environments.empty()) return true;
    const size_t selectedIndex = std::min<size_t>(state.environmentIndex, resources.environments.size() - 1U);
    EnvironmentGpu& selected = resources.environments[selectedIndex];
    if (!selected.view)
    {
        // Keep one large equirectangular background resident at a time.
        for (size_t index = 0; index < resources.environments.size(); ++index)
        {
            if (index != selectedIndex) resources.environments[index].view.Reset();
        }
        if (!LoadWicTexture(
                device,
                context,
                selected.path,
                true,
                selected.view,
                selected.width,
                selected.height,
                "EVE environment image"))
        {
            return false;
        }
    }
    resources.environmentView = selected.view;
    resources.environmentWidth = selected.width;
    resources.environmentHeight = selected.height;
    return true;
}

bool CreatePreviewTargets(
    ID3D11Device* device,
    IDXGISwapChain* swapChain,
    PreviewResources& resources)
{
    resources.renderTargetView.Reset();
    resources.depthStencilView.Reset();
    resources.depthTexture.Reset();

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
    return true;
}

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
    PreviewResources& resources)
{
    if (!CreatePreviewTargets(device, swapChain, resources)) return false;

    ComPtr<ID3DBlob> vertexBlob;
    ComPtr<ID3DBlob> pixelBlob;
    ComPtr<ID3DBlob> backgroundVertexBlob;
    ComPtr<ID3DBlob> backgroundPixelBlob;
    if (!CompileShader("VSMain", "vs_5_0", vertexBlob) ||
        !CompileShader("PSMain", "ps_5_0", pixelBlob) ||
        !CompileShader("VSBackground", "vs_5_0", backgroundVertexBlob) ||
        !CompileShader("PSBackground", "ps_5_0", backgroundPixelBlob))
    {
        return false;
    }

    if (FAILED(device->CreateVertexShader(vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), nullptr, resources.vertexShader.GetAddressOf())) ||
        FAILED(device->CreatePixelShader(pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), nullptr, resources.pixelShader.GetAddressOf())) ||
        FAILED(device->CreateVertexShader(backgroundVertexBlob->GetBufferPointer(), backgroundVertexBlob->GetBufferSize(), nullptr, resources.backgroundVertexShader.GetAddressOf())) ||
        FAILED(device->CreatePixelShader(backgroundPixelBlob->GetBufferPointer(), backgroundPixelBlob->GetBufferSize(), nullptr, resources.backgroundPixelShader.GetAddressOf())))
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

    if (mesh.vertices.empty() || mesh.indices.empty() ||
        mesh.vertices.size() * sizeof(Vertex) > static_cast<size_t>(std::numeric_limits<UINT>::max()) ||
        mesh.indices.size() * sizeof(uint32_t) > static_cast<size_t>(std::numeric_limits<UINT>::max()))
    {
        ADD_FAILURE() << "OBJ mesh is empty or exceeds D3D11 buffer limits";
        return false;
    }

    resources.indexCount = static_cast<uint32_t>(mesh.indices.size());

    D3D11_BUFFER_DESC vertexBufferDescription{};
    vertexBufferDescription.ByteWidth = static_cast<UINT>(mesh.vertices.size() * sizeof(Vertex));
    vertexBufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    vertexBufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vertexData{};
    vertexData.pSysMem = mesh.vertices.data();

    D3D11_BUFFER_DESC indexBufferDescription{};
    indexBufferDescription.ByteWidth = static_cast<UINT>(mesh.indices.size() * sizeof(uint32_t));
    indexBufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    indexBufferDescription.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA indexData{};
    indexData.pSysMem = mesh.indices.data();

    D3D11_BUFFER_DESC constantBufferDescription{};
    constantBufferDescription.ByteWidth = static_cast<UINT>(sizeof(SceneConstants));
    constantBufferDescription.Usage = D3D11_USAGE_DYNAMIC;
    constantBufferDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constantBufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    if (FAILED(device->CreateBuffer(&vertexBufferDescription, &vertexData, resources.vertexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&indexBufferDescription, &indexData, resources.indexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&constantBufferDescription, nullptr, resources.constantBuffer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR OBJ preview buffers";
        return false;
    }

    D3D11_RASTERIZER_DESC rasterizerDescription{};
    rasterizerDescription.FillMode = D3D11_FILL_SOLID;
    rasterizerDescription.CullMode = D3D11_CULL_NONE;
    rasterizerDescription.DepthClipEnable = TRUE;
    if (FAILED(device->CreateRasterizerState(&rasterizerDescription, resources.solidRasterizer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create solid rasterizer state";
        return false;
    }
    rasterizerDescription.FillMode = D3D11_FILL_WIREFRAME;
    if (FAILED(device->CreateRasterizerState(&rasterizerDescription, resources.wireRasterizer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create wireframe rasterizer state";
        return false;
    }

    D3D11_BLEND_DESC blendDescription{};
    blendDescription.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device->CreateBlendState(&blendDescription, resources.opaqueBlendState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create opaque blend state";
        return false;
    }
    blendDescription.RenderTarget[0].BlendEnable = TRUE;
    blendDescription.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDescription.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDescription.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDescription.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDescription.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    blendDescription.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    if (FAILED(device->CreateBlendState(&blendDescription, resources.alphaBlendState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create alpha blend state";
        return false;
    }
    blendDescription.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
    blendDescription.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    if (FAILED(device->CreateBlendState(&blendDescription, resources.additiveBlendState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create additive blend state";
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC depthStateDescription{};
    depthStateDescription.DepthEnable = TRUE;
    depthStateDescription.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthStateDescription.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    depthStateDescription.StencilEnable = FALSE;
    if (FAILED(device->CreateDepthStencilState(&depthStateDescription, resources.depthWriteState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create depth-write state";
        return false;
    }
    depthStateDescription.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    if (FAILED(device->CreateDepthStencilState(&depthStateDescription, resources.depthReadState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create depth-read state";
        return false;
    }

    D3D11_SAMPLER_DESC samplerDescription{};
    samplerDescription.Filter = D3D11_FILTER_ANISOTROPIC;
    samplerDescription.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDescription.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDescription.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDescription.MaxAnisotropy = 16;
    samplerDescription.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDescription.MinLOD = 0.0f;
    samplerDescription.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(device->CreateSamplerState(&samplerDescription, resources.textureSampler.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create albedo sampler";
        return false;
    }

    bool success = true;
    if (albedoPath.empty())
    {
        success = CreateFallbackTexture(device, resources);
    }
    else
    {
        success = LoadWicTexture(
            device, context, albedoPath, true, resources.albedoView, resources.textureWidth, resources.textureHeight, "albedo image");
        resources.hasExternalAlbedo = success;
    }
    if (!success) return false;

    if (!normalPath.empty())
    {
        resources.hasNormalMap = LoadWicTexture(
            device, context, normalPath, false, resources.normalView, resources.normalWidth, resources.normalHeight, "normal map");
        if (!resources.hasNormalMap) return false;
    }
    if (!pgsPath.empty())
    {
        resources.hasPgsMap = LoadWicTexture(
            device, context, pgsPath, false, resources.pgsView, resources.pgsWidth, resources.pgsHeight, "PGS material map");
        if (!resources.hasPgsMap) return false;
    }
    for (const std::string& environmentPath : environmentPaths)
    {
        EnvironmentGpu environment;
        environment.path = environmentPath;
        environment.label = FileLabel(environmentPath);
        resources.environments.push_back(std::move(environment));
    }
    resources.hasEnvironment = !resources.environments.empty();

    std::vector<AreaMaterialSource> materialSources;
    std::string materialError;
    if (!LoadAreaMaterialSources(materialManifestPath, materialSources, materialError))
    {
        ADD_FAILURE() << materialError;
        return false;
    }
    for (const AreaMaterialSource& source : materialSources)
    {
        for (const ObjDrawRange& range : mesh.drawRanges)
        {
            if (range.groupIndex != source.groupIndex) continue;
            AreaMaterialGpu material;
            material.source = source;
            material.drawRange = range;
            material.albedoView = resources.albedoView;
            material.normalView = resources.normalView;
            material.pgsView = resources.pgsView;
            material.albedoWidth = resources.textureWidth;
            material.albedoHeight = resources.textureHeight;
            material.hasAlbedo = resources.hasExternalAlbedo;
            material.hasNormal = resources.hasNormalMap;
            material.hasPgs = resources.hasPgsMap;
            if (!source.albedoPath.empty())
            {
                material.hasAlbedo = LoadWicTexture(
                    device, context, source.albedoPath, true, material.albedoView, material.albedoWidth, material.albedoHeight, "SOF area albedo");
                if (!material.hasAlbedo) return false;
            }
            if (!source.normalPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasNormal = LoadWicTexture(
                    device, context, source.normalPath, false, material.normalView, width, height, "SOF area normal map");
                if (!material.hasNormal) return false;
            }
            if (!source.pgsPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasPgs = LoadWicTexture(
                    device, context, source.pgsPath, false, material.pgsView, width, height, "SOF area material map");
                if (!material.hasPgs) return false;
            }
            if (!source.glowPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasGlow = LoadWicTexture(
                    device, context, source.glowPath, true, material.glowView, width, height, "SOF area glow map");
                if (!material.hasGlow) return false;
            }
            if (!source.dirtPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasDirt = LoadWicTexture(
                    device, context, source.dirtPath, false, material.dirtView, width, height, "SOF area dirt map");
                if (!material.hasDirt) return false;
            }
            if (!source.aoPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasAo = LoadWicTexture(
                    device, context, source.aoPath, false, material.aoView, width, height, "SOF area AO map");
                if (!material.hasAo) return false;
            }
            if (!source.paintMaskPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasPaintMask = LoadWicTexture(
                    device, context, source.paintMaskPath, false, material.paintMaskView, width, height, "SOF area paint mask");
                if (!material.hasPaintMask) return false;
            }
            if (!source.roughnessMapPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasRoughnessMap = LoadWicTexture(
                    device, context, source.roughnessMapPath, false, material.roughnessMapView, width, height, "SOF area roughness map");
                if (!material.hasRoughnessMap) return false;
            }
            resources.areaMaterials.push_back(std::move(material));
        }
    }
    std::stable_sort(resources.areaMaterials.begin(), resources.areaMaterials.end(), [](const AreaMaterialGpu& left, const AreaMaterialGpu& right) {
        return static_cast<int>(left.source.pass) < static_cast<int>(right.source.pass);
    });
    resources.baselineComplete = !resources.areaMaterials.empty();
    resources.baselineUnresolvedCount = 0;
    for (const AreaMaterialGpu& material : resources.areaMaterials)
    {
        const bool effectiveSemantics = material.source.semanticComplete && material.hasAlbedo && material.hasNormal && material.hasPgs;
        const bool roughnessResolved = material.hasRoughnessMap || material.source.shaderFamily == ShaderFamily::LegacyPgs;
        const bool coreTextures = effectiveSemantics && roughnessResolved;
        resources.baselineComplete = resources.baselineComplete && material.source.baselineComplete && coreTextures;
        resources.baselineUnresolvedCount += material.source.unresolvedCount + (coreTextures ? 0 : 1);
    }
    return true;
}

XMFLOAT3 GetWorldSpaceBoundsCenter(const PreviewState& state, const ObjMesh& mesh)
{
    const XMMATRIX world =
        DirectX::XMMatrixRotationX(state.modelPitch) *
        DirectX::XMMatrixRotationY(state.modelYaw) *
        DirectX::XMMatrixRotationZ(state.modelRoll);
    XMFLOAT3 centre{};
    DirectX::XMStoreFloat3(
        &centre,
        DirectX::XMVector3TransformCoord(DirectX::XMLoadFloat3(&mesh.boundsCenter), world));
    return centre;
}

void FrameShip(PreviewState& state, const ObjMesh& mesh)
{
    const XMFLOAT3 centre = GetWorldSpaceBoundsCenter(state, mesh);
    state.targetX = centre.x;
    state.targetY = centre.y;
    state.targetZ = centre.z;
    const float halfFov = DirectX::XMConvertToRadians(48.0f) * 0.5f;
    state.cameraDistance = std::max(mesh.boundsRadius / std::max(std::tan(halfFov), 0.1f) * 1.28f, 0.05f);
    state.nearClip = std::max(0.0005f, mesh.boundsRadius * 0.0002f);
    state.farClip = std::max(300.0f, state.cameraDistance + mesh.boundsRadius * 25.0f);
}

void ResetView(PreviewState& state, const ObjMesh& mesh)
{
    state.orbitYaw = -0.55f;
    state.orbitPitch = 0.22f;
    FrameShip(state, mesh);
}

void ApplyLightingPreset(PreviewState& state, int preset)
{
    state.lightingPreset = std::max(0, std::min(preset, 3));
    state.keyYaw = -0.65f;
    state.keyPitch = -0.55f;
    state.normalMapStrength = 0.90f;
    switch (state.lightingPreset)
    {
    case 1: // Studio
        state.keyIntensity = 2.00f;
        state.fillIntensity = 0.90f;
        state.rimIntensity = 0.62f;
        state.ambient = 0.58f;
        state.exposure = 1.28f;
        state.specularStrength = 0.78f;
        state.roughnessBias = 0.02f;
        state.environmentIntensity = 0.42f;
        state.backgroundIntensity = 0.28f;
        state.reflectionStrength = 0.58f;
        break;
    case 2: // Harsh inspection
        state.keyIntensity = 2.55f;
        state.fillIntensity = 0.30f;
        state.rimIntensity = 1.35f;
        state.ambient = 0.34f;
        state.exposure = 1.42f;
        state.specularStrength = 1.05f;
        state.roughnessBias = -0.12f;
        state.environmentIntensity = 0.48f;
        state.backgroundIntensity = 0.40f;
        state.reflectionStrength = 0.82f;
        break;
    case 3: // Dark silhouette
        state.keyIntensity = 0.48f;
        state.fillIntensity = 0.10f;
        state.rimIntensity = 2.05f;
        state.ambient = 0.10f;
        state.exposure = 0.95f;
        state.specularStrength = 0.62f;
        state.roughnessBias = 0.08f;
        state.environmentIntensity = 0.28f;
        state.backgroundIntensity = 0.16f;
        state.reflectionStrength = 0.38f;
        break;
    default: // Game-like, deliberately brighter than the previous test rig.
        state.keyIntensity = 1.90f;
        state.fillIntensity = 0.68f;
        state.rimIntensity = 1.05f;
        state.ambient = 0.62f;
        state.exposure = 1.42f;
        state.specularStrength = 0.92f;
        state.roughnessBias = -0.03f;
        state.environmentIntensity = 1.00f;
        state.backgroundIntensity = 0.82f;
        state.reflectionStrength = 1.00f;
        break;
    }
}

void ApplyGameLightingPreset(PreviewState& state)
{
    ApplyLightingPreset(state, 0);
}

void ProcessHotkeys(PreviewState& state, HWND hwnd, const ObjMesh& mesh, const PreviewResources& resources)
{
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard)
    {
        if (KeyPressed('0')) state.mode = 0;
        if (KeyPressed('1')) { state.mode = 1; state.previousEnabledMode = 1; }
        if (KeyPressed('2')) { state.mode = 2; state.previousEnabledMode = 2; }
        if (resources.baselineComplete && KeyPressed('3')) { state.mode = 3; state.previousEnabledMode = 3; }
        if (resources.baselineComplete && KeyPressed('4')) { state.mode = 4; state.previousEnabledMode = 4; }
        if (resources.baselineComplete && KeyPressed('5')) { state.mode = 5; state.previousEnabledMode = 5; }
        if (KeyPressed(VK_SPACE))
        {
            if (state.mode == 0) state.mode = (!resources.baselineComplete && state.previousEnabledMode >= 3) ? 2 : state.previousEnabledMode;
            else { state.previousEnabledMode = state.mode; state.mode = 0; }
        }
        if (KeyPressed('R')) ResetView(state, mesh);
        if (KeyPressed(VK_HOME) || KeyPressed('F')) FrameShip(state, mesh);
        if (KeyPressed('V')) state.flipV = !state.flipV;
        if (!resources.environments.empty() && KeyPressed(VK_OEM_4))
        {
            state.environmentIndex = state.environmentIndex == 0U
                ? static_cast<uint32_t>(resources.environments.size() - 1U)
                : state.environmentIndex - 1U;
        }
        if (!resources.environments.empty() && KeyPressed(VK_OEM_6))
        {
            state.environmentIndex = (state.environmentIndex + 1U) % static_cast<uint32_t>(resources.environments.size());
        }
    }

    if (KeyPressed(VK_F9)) state.requestScreenshot = true;
    if (KeyPressed(VK_ESCAPE)) PostMessage(hwnd, WM_CLOSE, 0, 0);
}

void DrawShipSelector(ShipCatalog& catalog, HWND hwnd)
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

void DrawControlPanel(
    PreviewState& state,
    ShipCatalog& catalog,
    HWND hwnd,
    const ObjMesh& mesh,
    const PreviewResources& resources,
    const std::string& albedoPath,
    const std::string& normalPath,
    const std::string& pgsPath)
{
    static constexpr const char* modeNames[] = {
        "Original EVE material",
        "Material / input validation",
        "UV texel-density and stretch diagnostics",
        "Selected repair candidate",
        "Selected repair + micro-normal",
        "Difference / reconstructed contribution",
    };

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.94f);
    const float panelWidth = std::min(640.0f, std::max(420.0f, static_cast<float>(resources.width) * 0.45f));
    const float panelHeight = std::max(280.0f, static_cast<float>(resources.height) - 24.0f);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::Begin("NSAMDR Real EVE Ship Controls");

    ImGui::TextUnformatted("NSAMDR — Neural Stretch-Aware Material Detail Reconstruction");
    ImGui::TextUnformatted("Detects UV-stretched regions, validates source inputs and reconstructs lost structure before adding microdetail.");
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
            ImGui::TextUnformatted("Visual baseline: COMPLETE - repair modes unlocked");
        }
        else
        {
            ImGui::TextWrapped("Visual baseline: INCOMPLETE (%d unresolved inputs). Repair modes are locked.", resources.baselineUnresolvedCount);
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

    ImGui::Separator();
    DrawShipSelector(catalog, hwnd);

    ImGui::Separator();
    if (!resources.baselineComplete)
    {
        ImGui::TextWrapped(
            "Modes 3-5 are intentionally locked until the original EVE material baseline is complete. "
            "Use modes 0-2 to diagnose unresolved SOF materials or textures first.");
    }
    for (int mode = 0; mode < static_cast<int>(sizeof(modeNames) / sizeof(modeNames[0])); ++mode)
    {
        const bool locked = mode >= 3 && !resources.baselineComplete;
        if (locked) ImGui::BeginDisabled();
        if (ImGui::RadioButton(modeNames[mode], state.mode == mode))
        {
            state.mode = mode;
            if (mode != 0) state.previousEnabledMode = mode;
        }
        if (locked && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        {
            ImGui::SetTooltip(
                "Locked because the visual baseline has %d unresolved material input%s. "
                "Any repair method would otherwise be tested against an incorrect render.",
                resources.baselineUnresolvedCount,
                resources.baselineUnresolvedCount == 1 ? "" : "s");
        }
        if (locked) ImGui::EndDisabled();
    }
    if (!resources.baselineComplete && state.mode >= 3) state.mode = 0;

    if (state.mode == 1)
    {
        static constexpr const char* diagnosticViews[] = {
            "Summary split", "Albedo", "Roughness", "Normal X", "Normal Y", "Normal Z",
            "Ambient occlusion", "Paint mask", "Material selector", "Dirt", "Glow",
            "Resolved material colour", "Area ID", "Unresolved overlay", "Mip / derivative evidence",
        };
        ImGui::Combo("Validation view", &state.diagnosticView, diagnosticViews, static_cast<int>(sizeof(diagnosticViews) / sizeof(diagnosticViews[0])));
    }

    static constexpr const char* repairMethods[] = {
        "Sampling / mip correction",
        "Object-space projected detail",
        "Edge-guided structural transfer",
        "Hybrid constrained neural residual",
    };
    ImGui::Combo("Repair candidate", &state.repairMethod, repairMethods, 4);
    if (state.repairMethod == 0) ImGui::SliderFloat("Sampling LOD bias", &state.samplingLodBias, -2.0f, 1.0f, "%.2f");
    if (state.repairMethod == 1) ImGui::SliderFloat("Projection strength", &state.projectionStrength, 0.0f, 1.5f, "%.2f");
    if (state.repairMethod >= 2) ImGui::SliderFloat("Transfer / residual strength", &state.transferStrength, 0.0f, 1.5f, "%.2f");

    ImGui::Separator();
    ImGui::SliderFloat("Reconstruction strength", &state.strength, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Derivative damage start", &state.damageLow, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Derivative damage full", &state.damageHigh, 0.1f, 4.0f, "%.2f");
    if (state.damageHigh < state.damageLow + 0.05f) state.damageHigh = state.damageLow + 0.05f;
    ImGui::SliderFloat("Structure sharpness", &state.structureSharpness, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Structure scale", &state.structureScale, 0.5f, 12.0f, "%.2f");
    ImGui::SliderFloat("Preserve clean regions", &state.preserveClean, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Difference view gain", &state.differenceScale, 0.5f, 6.0f, "%.2f");
    ImGui::SliderFloat("Diagnostic checker scale", &state.diagnosticCheckerScale, 2.0f, 64.0f, "%.1f");
    ImGui::SliderFloat("NSAMDR micro-normal", &state.microNormalStrength, 0.0f, 5.0f, "%.2f");

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
        ApplyLightingPreset(state, state.lightingPreset);
    }
    if (ImGui::Button("Reset selected lighting preset")) ApplyLightingPreset(state, state.lightingPreset);
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
    ImGui::Checkbox("Invert texture V (V)", &state.flipV);
    ImGui::Checkbox("Wireframe", &state.wireframe);

    ImGui::Separator();
    ImGui::TextUnformatted("Raven Navy Issue truth-set proof");
    ImGui::Combo("Truth target", &state.truthTarget, kTruthTargetNames, static_cast<int>(sizeof(kTruthTargetNames) / sizeof(kTruthTargetNames[0])));
    if (!resources.baselineComplete) ImGui::BeginDisabled();
    if (ImGui::Button("Capture baseline / repaired / difference"))
    {
        state.proofRestoreMode = state.mode;
        state.proofCaptureStep = 0;
    }
    if (!resources.baselineComplete) ImGui::EndDisabled();
    ImGui::TextWrapped("Adjust the camera to the matching in-game screenshot, select the target label, then capture the three-frame proof set.");

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

    if (ImGui::Button("Frame whole ship (F/Home)")) FrameShip(state, mesh);
    ImGui::SameLine();
    if (ImGui::Button("Reset view (R)")) ResetView(state, mesh);
    ImGui::SameLine();
    if (ImGui::Button("Save screenshot (F9)")) state.requestScreenshot = true;

    ImGui::Separator();
    ImGui::Text("Frame time: %.3f ms", 1000.0f / std::max(ImGui::GetIO().Framerate, 1.0f));
    ImGui::TextUnformatted("Mouse: RMB orbit | MMB or LMB+RMB pan | Shift+pan fine | wheel zoom-to-cursor | Ctrl+wheel fine zoom");
    ImGui::TextUnformatted("Double-click a hull panel to make it the orbit and zoom focus point.");
    ImGui::TextUnformatted("Keys: 0-5 modes | Space compare | [ / ] backgrounds | F/Home frame ship | V flip UV | R reset | F9 capture | Esc exit");
    ImGui::TextWrapped("Mode 1 shows validation splits for albedo/checker, normal and PGS-or-mip evidence. Mode 2 shows UV stretch direction, mip pressure and damage heat.");
    ImGui::TextWrapped("Modes 3-5 evaluate the selected repair candidate. Mode 4 adds micro-normal reconstruction; mode 5 shows only the candidate contribution.");

    const ImVec2 panelPosition = ImGui::GetWindowPos();
    const ImVec2 actualPanelSize = ImGui::GetWindowSize();
    const float desiredSceneX = panelPosition.x + actualPanelSize.x + 12.0f;
    state.sceneViewportX = static_cast<uint32_t>(ClampFloat(
        desiredSceneX,
        0.0f,
        resources.width > 320U ? static_cast<float>(resources.width - 320U) : 0.0f));

    ImGui::End();
}
bool UploadSceneConstants(ID3D11DeviceContext* context, ID3D11Buffer* constantBuffer, const SceneConstants& constants)
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

bool UpdateSceneConstants(
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
    BuildCameraMatrices(state, viewportWidth, resources.height, world, view, projection, eye, target);

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
    const int effectiveMode = (!resources.baselineComplete && state.mode >= 3) ? 0 : state.mode;
    constants.controls = XMFLOAT4(static_cast<float>(effectiveMode), state.strength, state.damageLow, state.damageHigh);
    constants.keyLight = XMFLOAT4(key.x, key.y, key.z, state.keyIntensity);
    constants.fillLight = XMFLOAT4(fill.x, fill.y, fill.z, state.fillIntensity);
    constants.rimLight = XMFLOAT4(rim.x, rim.y, rim.z, state.rimIntensity);
    constants.material = XMFLOAT4(
        state.useTexture && resources.hasExternalAlbedo ? 1.0f : 0.0f,
        state.flipV ? 1.0f : 0.0f,
        state.exposure,
        state.ambient);
    constants.surface = XMFLOAT4(
        state.microNormalStrength,
        state.normalMapStrength,
        state.specularStrength,
        state.roughnessBias);
    constants.options = XMFLOAT4(
        state.useNormalMap && resources.hasNormalMap ? 1.0f : 0.0f,
        state.usePgsMap && resources.hasPgsMap ? 1.0f : 0.0f,
        0.0f,
        0.0f);

    XMFLOAT3 cameraRight, cameraUp, cameraForward;
    GetCameraBasis(state, cameraRight, cameraUp, cameraForward);
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
    constants.diagnostics = XMFLOAT4(
        state.diagnosticCheckerScale,
        0.0f,
        static_cast<float>(std::max(resources.textureWidth, 1U)),
        static_cast<float>(std::max(resources.textureHeight, 1U)));
    constants.structure = XMFLOAT4(
        state.structureSharpness,
        state.structureScale,
        state.preserveClean,
        state.differenceScale);
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
    constants.debug = XMFLOAT4(static_cast<float>(state.diagnosticView), 0.0f, resources.baselineComplete ? 1.0f : 0.0f, 0.0f);
    constants.repair = XMFLOAT4(static_cast<float>(state.repairMethod), state.samplingLodBias, state.projectionStrength, state.transferStrength);

    outputConstants = constants;
    return UploadSceneConstants(context, constantBuffer, constants);
}
void RenderShip(
    ID3D11DeviceContext* context,
    const PreviewResources& resources,
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

    SceneConstants backgroundConstants = baseConstants;
    const float backgroundAspect = resources.height == 0U ? 1.0f :
        static_cast<float>(resources.width) / static_cast<float>(resources.height);
    const float tanHalfFov = std::tan(DirectX::XMConvertToRadians(48.0f) * 0.5f);
    backgroundConstants.cameraRight.w = backgroundAspect * tanHalfFov;
    UploadSceneConstants(context, constantBuffer, backgroundConstants);

    // Full-screen environment first. Null environment SRV is valid because the shader uses the procedural fallback.
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

    // Real ship mesh over the environment.
    UploadSceneConstants(context, constantBuffer, baseConstants);
    const uint32_t sceneX = std::min(state.sceneViewportX, resources.width > 1U ? resources.width - 1U : 0U);
    viewport.TopLeftX = static_cast<float>(sceneX);
    viewport.Width = static_cast<float>(std::max(1U, resources.width - sceneX));
    context->RSSetViewports(1, &viewport);
    context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), resources.depthStencilView.Get());
    context->RSSetState(state.wireframe ? resources.wireRasterizer.Get() : resources.solidRasterizer.Get());
    const UINT stride = sizeof(Vertex);
    const UINT offset = 0;
    ID3D11Buffer* vertexBuffer = resources.vertexBuffer.Get();
    context->IASetInputLayout(resources.inputLayout.Get());
    context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
    context->IASetIndexBuffer(resources.indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(resources.vertexShader.Get(), nullptr, 0);
    context->PSSetShader(resources.pixelShader.Get(), nullptr, 0);
    context->VSSetConstantBuffers(0, 1, &constantBuffer);
    context->PSSetConstantBuffers(0, 1, &constantBuffer);
    context->PSSetSamplers(0, 1, &textureSampler);

    const float blendFactor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto drawArea = [&](const AreaMaterialGpu& material) {
        SceneConstants constants = baseConstants;
        constants.material.x = state.useTexture && material.hasAlbedo ? 1.0f : 0.0f;
        constants.options.x = state.useNormalMap && material.hasNormal ? 1.0f : 0.0f;
        constants.options.y = state.usePgsMap && material.hasPgs ? 1.0f : 0.0f;
        constants.diagnostics.z = static_cast<float>(std::max(material.albedoWidth, 1U));
        constants.diagnostics.w = static_cast<float>(std::max(material.albedoHeight, 1U));
        constants.areaTint = XMFLOAT4(
            material.source.tint.x,
            material.source.tint.y,
            material.source.tint.z,
            1.0f);
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
            material.source.glowColour.x, material.source.glowColour.y, material.source.glowColour.z, material.source.generalDataX);
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
            static_cast<float>(state.diagnosticView),
            static_cast<float>(material.source.groupIndex + 1),
            material.source.baselineComplete ? 1.0f : 0.0f,
            static_cast<float>(static_cast<int>(material.source.shaderFamily)));
        constants.repair = XMFLOAT4(
            static_cast<float>(state.repairMethod), state.samplingLodBias, state.projectionStrength, state.transferStrength);
        if (!UploadSceneConstants(context, constantBuffer, constants)) return false;

        ID3D11ShaderResourceView* textureViews[9] = {
            material.albedoView.Get(),
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
        context->DrawIndexed(material.drawRange.indexCount, material.drawRange.startIndex, 0);
        return true;
    };

    if (!resources.areaMaterials.empty())
    {
        for (const AreaMaterialGpu& material : resources.areaMaterials)
        {
            if (!drawArea(material)) break;
        }
    }
    else
    {
        SceneConstants constants = baseConstants;
        UploadSceneConstants(context, constantBuffer, constants);
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
        context->DrawIndexed(resources.indexCount, 0, 0);
    }

    ID3D11ShaderResourceView* nullViews[9] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    context->PSSetShaderResources(0, 9, nullViews);
    context->OMSetBlendState(nullptr, blendFactor, 0xffffffffU);
    context->OMSetDepthStencilState(nullptr, 0);
}

std::string BuildScreenshotPath(const PreviewState& state)
{
    CreateDirectoryA("artifacts", nullptr);
    CreateDirectoryA("artifacts\\nsamdr", nullptr);
    CreateDirectoryA("artifacts\\nsamdr\\truth_set", nullptr);

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    const int target = std::max(0, std::min(state.truthTarget, static_cast<int>(sizeof(kTruthTargetNames) / sizeof(kTruthTargetNames[0])) - 1));
    return "artifacts\\nsamdr\\truth_set\\" + std::string(kTruthTargetNames[target]) +
        "_mode_" + std::to_string(state.mode) + "_repair_" + std::to_string(state.repairMethod) +
        "_" + std::to_string(milliseconds) + ".dds";
}

struct NSAMDRRendering : public WithValidRenderContext
{
};

} // namespace

TEST_F(NSAMDRRendering, RealObjShipPreview)
{
    ENSURE_GPU_OR_SKIP

    const std::string objPath = GetEnvironmentString("NSAMDR_OBJ");
    ASSERT_FALSE(objPath.empty())
        << "NSAMDR_OBJ is not set. Launch through scripts\\build\\run_nsamdr_obj_preview_dx11.bat <ship.obj|ship.gr2> [albedo.png].";

    ObjMesh mesh;
    std::string meshError;
    ASSERT_TRUE(LoadObjMesh(objPath, mesh, meshError)) << meshError;

    const std::string albedoPath = GetEnvironmentString("NSAMDR_ALBEDO");
    const std::string normalPath = GetEnvironmentString("NSAMDR_NORMAL");
    const std::string pgsPath = GetEnvironmentString("NSAMDR_PGS");
    const std::vector<std::string> environmentPaths = GetEnvironmentPaths();
    const std::string environmentPath = environmentPaths.empty() ? std::string() : environmentPaths.front();
    const std::string materialManifestPath = GetEnvironmentString("NSAMDR_MATERIALS");

    ASSERT_NE(renderContext, nullptr);
    ASSERT_TRUE(renderContext->IsValid());
    ASSERT_TRUE(renderContext->m_d3dDevice11);
    ASSERT_TRUE(renderContext->m_context);
    ASSERT_TRUE(renderContext->m_swapChain);

    HWND hwnd = static_cast<HWND>(GetWindowHandle());
    g_previewWindow = hwnd;
    SetWindowTextW(hwnd, L"NSAMDR \u2014 Neural Stretch-Aware Material Detail Reconstruction | Real EVE Ship Preview");

    GetWindow()->Resize(1440U, 900U);
    presentParameters.mode.width = 1440U;
    presentParameters.mode.height = 900U;
    ASSERT_HRESULT_SUCCEEDED(renderContext->SetPresentParameters(Tr2VideoAdapterInfo::DEFAULT_ADAPTER, presentParameters));

    g_exitInteractiveOnCharacter = false;
    g_previousWindowProc = reinterpret_cast<WNDPROC>(SetWindowLongPtr(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(NSAMDRPreviewWindowProc)));
    ASSERT_NE(g_previousWindowProc, nullptr);

    ID3D11Device* device = renderContext->m_d3dDevice11;
    ID3D11DeviceContext* context = renderContext->m_context;
    IDXGISwapChain* swapChain = renderContext->m_swapChain;

    PreviewResources resources;
    ASSERT_TRUE(CreatePreviewResources(
        device, context, swapChain, mesh, albedoPath, normalPath, pgsPath, environmentPaths, materialManifestPath, resources));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ASSERT_TRUE(ImGui_ImplWin32_Init(hwnd));
    ASSERT_TRUE(ImGui_ImplDX11_Init(device, context));

    PreviewState state;
    const bool hasAreaAlbedo = std::any_of(resources.areaMaterials.begin(), resources.areaMaterials.end(), [](const AreaMaterialGpu& material) { return material.hasAlbedo; });
    const bool hasAreaNormal = std::any_of(resources.areaMaterials.begin(), resources.areaMaterials.end(), [](const AreaMaterialGpu& material) { return material.hasNormal; });
    const bool hasAreaPgs = std::any_of(resources.areaMaterials.begin(), resources.areaMaterials.end(), [](const AreaMaterialGpu& material) { return material.hasPgs; });
    state.useTexture = resources.hasExternalAlbedo || hasAreaAlbedo || !resources.areaMaterials.empty();
    state.useNormalMap = resources.hasNormalMap || hasAreaNormal;
    state.usePgsMap = resources.hasPgsMap || hasAreaPgs;
    state.useEnvironment = resources.hasEnvironment;
    ASSERT_TRUE(EnsureSelectedEnvironmentLoaded(device, context, resources, state));
    ApplyGameLightingPreset(state);
    ResetView(state, mesh);

    ShipCatalog shipCatalog;
    LoadShipCatalog(
        GetEnvironmentString("NSAMDR_EVE_CATALOG"),
        GetEnvironmentString("NSAMDR_EVE_QUERY"),
        shipCatalog);
    g_previewState = &state;

    std::printf("NSAMDR OBJ loaded: %s\n", mesh.path.c_str());
    std::printf("  triangles=%u vertices=%zu sourcePositions=%u sourceUVs=%u sourceNormals=%u\n",
        mesh.triangleCount,
        mesh.vertices.size(),
        mesh.sourcePositionCount,
        mesh.sourceTexcoordCount,
        mesh.sourceNormalCount);
    std::printf("  uvStretchAverage=%.4f uvStretchMaximum=%.4f degenerateUvTriangles=%u\n",
        mesh.averageStretch,
        mesh.maximumStretch,
        mesh.degenerateUvTriangles);
    if (resources.hasExternalAlbedo)
    {
        std::printf("  albedo=%s (%ux%u)\n", albedoPath.c_str(), resources.textureWidth, resources.textureHeight);
    }
    else
    {
        std::printf("  albedo=<neutral fallback>\n");
    }
    if (resources.hasNormalMap)
    {
        std::printf("  normal=%s (%ux%u)\n", normalPath.c_str(), resources.normalWidth, resources.normalHeight);
    }
    if (resources.hasPgsMap)
    {
        std::printf("  pgs=%s (%ux%u)\n", pgsPath.c_str(), resources.pgsWidth, resources.pgsHeight);
    }
    if (!resources.areaMaterials.empty())
    {
        std::printf("  sofMaterials=%s (draws=%zu, groups=%zu)\n", materialManifestPath.c_str(), resources.areaMaterials.size(), mesh.drawRanges.size());
    }
    else
    {
        std::printf("  sofMaterials=<legacy global fallback> (groups=%zu)\n", mesh.drawRanges.size());
    }
    if (resources.hasEnvironment)
    {
        std::printf("  environment=%s (%ux%u)\n", environmentPath.c_str(), resources.environmentWidth, resources.environmentHeight);
    }
    else
    {
        std::printf("  environment=<procedural fallback>\n");
    }

    const auto startTime = std::chrono::steady_clock::now();
    auto previousFrame = startTime;

    auto frame = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - previousFrame).count();
        const float elapsedSeconds = std::chrono::duration<float>(now - startTime).count();
        previousFrame = now;

        if (g_pendingResizeWidth >= 64U && g_pendingResizeHeight >= 64U &&
            (g_pendingResizeWidth != resources.width || g_pendingResizeHeight != resources.height))
        {
            const uint32_t resizeWidth = g_pendingResizeWidth;
            const uint32_t resizeHeight = g_pendingResizeHeight;
            g_pendingResizeWidth = 0U;
            g_pendingResizeHeight = 0U;

            context->OMSetRenderTargets(0, nullptr, nullptr);
            resources.renderTargetView.Reset();
            resources.depthStencilView.Reset();
            resources.depthTexture.Reset();
            presentParameters.mode.width = resizeWidth;
            presentParameters.mode.height = resizeHeight;
            ASSERT_HRESULT_SUCCEEDED(renderContext->SetPresentParameters(
                Tr2VideoAdapterInfo::DEFAULT_ADAPTER,
                presentParameters));
            ASSERT_TRUE(CreatePreviewTargets(device, swapChain, resources));
        }

        RefreshPreviewInputFocus(hwnd);

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        if (g_previewInputFocused &&
            !ImGui::GetIO().WantCaptureMouse &&
            ImGui::IsMouseDoubleClicked(0))
        {
            state.focusMouseX = static_cast<int>(ImGui::GetIO().MousePos.x);
            state.focusMouseY = static_cast<int>(ImGui::GetIO().MousePos.y);
            state.requestFocus = true;
        }
        ProcessHotkeys(state, hwnd, mesh, resources);
        if (state.autoOrbit)
        {
            state.orbitYaw += deltaSeconds * state.orbitSpeed;
        }
        if (state.requestFocus)
        {
            state.requestFocus = false;
            FocusCameraAtScreenPoint(
                state, mesh, resources.width, resources.height, state.focusMouseX, state.focusMouseY);
        }
        ApplyZoomRequest(state, mesh, resources.width, resources.height);
        DrawControlPanel(
            state, shipCatalog, hwnd, mesh, resources, albedoPath, normalPath, pgsPath);
        ASSERT_TRUE(EnsureSelectedEnvironmentLoaded(device, context, resources, state));

        if (state.proofCaptureStep >= 0)
        {
            static constexpr int proofModes[] = {0, 3, 5};
            state.mode = proofModes[std::min(state.proofCaptureStep, 2)];
            state.requestScreenshot = true;
        }

        SceneConstants sceneConstants{};
        ASSERT_TRUE(UpdateSceneConstants(
            context, resources.constantBuffer.Get(), resources, state, elapsedSeconds, sceneConstants));
        RenderShip(context, resources, state, sceneConstants);

        ImGui::Render();
        context->OMSetRenderTargets(1, resources.renderTargetView.GetAddressOf(), nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        if (state.requestScreenshot)
        {
            state.requestScreenshot = false;
            const std::string screenshotPath = BuildScreenshotPath(state);
            MakeScreenShot(screenshotPath.c_str());
            std::printf("Saved NSAMDR screenshot: %s\n", screenshotPath.c_str());
            if (state.proofCaptureStep >= 0)
            {
                ++state.proofCaptureStep;
                if (state.proofCaptureStep >= 3)
                {
                    state.proofCaptureStep = -1;
                    state.mode = state.proofRestoreMode;
                }
            }
        }

        ASSERT_HRESULT_SUCCEEDED(renderContext->Present());
    };

    RunLoop(frame);

    ResetSceneInput(hwnd);
    g_previewState = nullptr;
    g_previewWindow = nullptr;
    g_previewInputFocused = false;
    g_blockMouseUntilRelease = false;
    g_keyWasDown.fill(false);

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
