cbuffer SceneConstants : register(b0)
{
    row_major float4x4 gWorld;
    row_major float4x4 gViewProjection;
    float4 gCameraTime;      // xyz = camera, w = elapsed seconds
    float4 gKeyLight;        // xyz = travel direction, w = intensity
    float4 gFillLight;       // xyz = travel direction, w = intensity
    float4 gRimLight;        // xyz = travel direction, w = intensity
    float4 gMaterial;        // x = use albedo, y = flip V, z = exposure, w = ambient
    float4 gSurface;         // x = reserved, y = normal map strength, z = specular, w = roughness bias
    float4 gOptions;         // x = use normal, y = use PGS, zw = reserved
    float4 gCameraRight;     // xyz = camera right, w = aspect * tan(fov / 2)
    float4 gCameraUp;        // xyz = camera up, w = tan(fov / 2)
    float4 gCameraForward;   // xyz = camera forward
    float4 gEnvironment;     // x = use map, y = light intensity, z = background intensity, w = reflection strength
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
    float4 gDebug;             // xyz = reserved, w = shader family
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
    float2 stretchAxis : TEXCOORD2;
    float2 repairUvCenter : TEXCOORD3;
    float2 repairUvScale : TEXCOORD4;
};

struct VSOutput
{
    float4 position    : SV_POSITION;
    float3 worldPos    : TEXCOORD0;
    float3 normal      : TEXCOORD1;
    float2 uv          : TEXCOORD2;
};

VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4 worldPosition = mul(float4(input.position, 1.0), gWorld);
    output.position = mul(worldPosition, gViewProjection);
    output.worldPos = worldPosition.xyz;
    output.normal = normalize(mul(float4(input.normal, 0.0), gWorld).xyz);
    output.uv = input.uv;
    return output;
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
        float2 uv = 0.0;
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

float2 SampleNormalXYGrad(float2 sampleUv, float2 gradientX, float2 gradientY)
{
    const float4 packed = gNormalMap.SampleGrad(gTextureSampler, sampleUv, gradientX, gradientY);
    return float2(
        SampleChannel(packed, gSemanticChannels.x),
        SampleChannel(packed, gSemanticChannels.y)) * 2.0 - 1.0;
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

    float4 materialWeights = NormalizedMaterialWeights(uv);
    float3 materialBase = max(BlendMaterialColour(materialWeights), 0.001);
    float3 materialF0 = clamp(BlendMaterialF0(materialWeights), 0.0, 1.0);
    float materialGloss = max(BlendMaterialGloss(materialWeights), 0.0);

    // RAW SOURCE and NSAMDR FINAL use this exact sampling path. Their bound
    // textures are the only material difference between the two panes.
    float4 sampledAlbedoRgba = gAlbedo.SampleGrad(gTextureSampler, uv, ddx(uv), ddy(uv));
    float3 sampledAlbedo = sampledAlbedoRgba.rgb;
    float paintMask = gAuxTextures.z > 0.5
        ? SampleChannel(gPaintMaskMap.Sample(gTextureSampler, uv), gSemanticChannels2.y) : 0.0;

    // V5 maps are detail resources tinted by the material library. Legacy
    // _d maps are already authored colour textures; applying the SOF colour
    // set again creates the dark, mixed-material Raven regression.
    const int shaderFamily = (int)round(gDebug.w);
    float3 materialTint = shaderFamily == 1
        ? float3(1.0, 1.0, 1.0)
        : lerp(materialBase, float3(1.0, 1.0, 1.0), saturate(paintMask));
    float3 areaAlbedo = gMaterial.x > 0.5 ? sampledAlbedo * materialTint : materialTint;

    float dirtAmount = gAuxTextures.x > 0.5
        ? saturate(SampleChannel(gDirtMap.Sample(gTextureSampler, uv), gSemanticChannels2.z)) : 0.0;
    areaAlbedo *= lerp(1.0, 0.58, dirtAmount * 0.58);
    float aoSample = gAuxTextures.y > 0.5
        ? saturate(SampleChannel(gAoMap.Sample(gTextureSampler, uv), gSemanticChannels2.x)) : 1.0;
    float ao = lerp(0.32, 1.0, aoSample);
    float3 albedo = max(areaAlbedo, 0.0);

    float3 mappedNormal = geometricNormal;
    float2 authoredNormalXY = 0.0;
    if (gOptions.x > 0.5)
    {
        authoredNormalXY = SampleNormalXYGrad(uv, ddx(uv), ddy(uv)) * gSurface.y;
        const float authoredNormalZ = sqrt(saturate(1.0 - dot(authoredNormalXY, authoredNormalXY)));
        mappedNormal = ApplyMappedNormal(
            geometricNormal,
            input.worldPos,
            uv,
            normalize(float3(authoredNormalXY, authoredNormalZ)));
    }

    float3 n = mappedNormal;

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

    float3 shadedAlbedo = albedo;

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

    if (gAreaTextures.w > 0.5)
    {
        const float glowMask = SampleChannel(gGlowMap.Sample(gTextureSampler, uv), gSemanticChannels2.w);
        colour += glowMask * gAreaEffects.rgb * 1.8;
    }

    colour *= gMaterial.z;
    colour = colour / (1.0 + colour);
    colour = pow(saturate(colour), 1.0 / 2.2);

    float textureAlpha = (gAreaTextures.x > 0.5 && (int)round(gDebug.w) == 1) ? sampledAlbedoRgba.a : 1.0;
    float outputAlpha = gAreaTint.w > 0.5 ? saturate(gAreaSurface.z * textureAlpha) : 1.0;
    return float4(colour, outputAlpha);
}
