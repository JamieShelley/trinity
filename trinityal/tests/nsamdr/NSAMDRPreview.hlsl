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
    float4 gOptions;         // x = use normal, y = use PGS, z = force repair mask, w = reserved
    float4 gCameraRight;     // xyz = camera right, w = aspect * tan(fov / 2)
    float4 gCameraUp;        // xyz = camera up, w = tan(fov / 2)
    float4 gCameraForward;   // xyz = camera forward
    float4 gEnvironment;     // x = use map, y = light intensity, z = background intensity, w = reflection strength
    float4 gDiagnostics;     // x = checker scale, y = reserved, z = texture width, w = texture height
    float4 gStructure;       // x = source-detail gain, y = object-space patch frequency, z = damaged-region isolation, w = difference scale
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
    float4 gRepair;            // x = repair method, y = sampling LOD bias, z = source-neighbourhood radius, w = transfer strength
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
    float3 localPos    : TEXCOORD1;
    float3 normal      : TEXCOORD2;
    float2 uv          : TEXCOORD3;
    float stretchHint : TEXCOORD4;
    float2 stretchAxis : TEXCOORD5;
    float2 repairUvCenter : TEXCOORD6;
    float2 repairUvScale : TEXCOORD7;
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
    output.stretchAxis = input.stretchAxis;
    output.repairUvCenter = input.repairUvCenter;
    output.repairUvScale = input.repairUvScale;
    return output;
}


float Hash21(float2 p)
{
    p = frac(p * float2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return frac(p.x * p.y);
}

float2 Hash22(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float2 Rotate2D(float2 value, float angle)
{
    float sineValue;
    float cosineValue;
    sincos(angle, sineValue, cosineValue);
    return float2(
        cosineValue * value.x - sineValue * value.y,
        sineValue * value.x + cosineValue * value.y);
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

float2 StretchCorrectedUv(
    float2 sourceUv,
    float2 repairCenter,
    float2 stretchAxis,
    float2 repairScale,
    float recoveryAmount)
{
    const float2 axis = normalize(stretchAxis + float2(1.0e-5, 0.0));
    const float2 perpendicular = float2(-axis.y, axis.x);
    const float2 delta = sourceUv - repairCenter;
    const float2 effectiveScale = lerp(
        float2(1.0, 1.0),
        max(repairScale, float2(1.0, 1.0)),
        max(recoveryAmount, 0.0));
    return repairCenter +
        axis * dot(delta, axis) * effectiveScale.x +
        perpendicular * dot(delta, perpendicular) * effectiveScale.y;
}

float3 SampleAlbedoHighPass(float2 sampleUv, float blurLod)
{
    const float3 source = gAlbedo.SampleLevel(gTextureSampler, sampleUv, 0.0).rgb;
    const float3 lowFrequency = gAlbedo.SampleLevel(gTextureSampler, sampleUv, blurLod).rgb;
    return source - lowFrequency;
}

float2 SampleNormalXY(float2 sampleUv, float lod)
{
    const float4 packed = gNormalMap.SampleLevel(gTextureSampler, sampleUv, lod);
    return float2(
        SampleChannel(packed, gSemanticChannels.x),
        SampleChannel(packed, gSemanticChannels.y)) * 2.0 - 1.0;
}

float2 SampleNormalHighPass(float2 sampleUv, float blurLod)
{
    return SampleNormalXY(sampleUv, 0.0) - SampleNormalXY(sampleUv, blurLod);
}

float2 SampleNormalXYGrad(float2 sampleUv, float2 gradientX, float2 gradientY)
{
    const float4 packed = gNormalMap.SampleGrad(gTextureSampler, sampleUv, gradientX, gradientY);
    return float2(
        SampleChannel(packed, gSemanticChannels.x),
        SampleChannel(packed, gSemanticChannels.y)) * 2.0 - 1.0;
}

float4 SampleNSAMDRAlbedo(float2 sampleUv)
{
    // Public Mode 3 receives the V4 tile-context material already reconstructed
    // offline. The live pixel shader uses the same anisotropic sampling path as
    // the original-source pane and performs no hidden cleanup or sharpening.
    return gAlbedo.SampleGrad(gTextureSampler, sampleUv, ddx(sampleUv), ddy(sampleUv));
}

float2 SampleNSAMDRNormalXY(float2 sampleUv)
{
    // Semantic textures remain deterministic. Mode 3 samples the authored normal
    // map with the same gradients as the baseline material path.
    return SampleNormalXYGrad(sampleUv, ddx(sampleUv), ddy(sampleUv));
}

float PatchWeight(float2 delta)
{
    // Radial overlap removes the rectangular cell boundaries that made the old
    // procedural repair appear as a grid on the hull.
    const float distanceSquared = dot(delta, delta);
    const float falloff = saturate(1.35 - distanceSquared);
    return falloff * falloff * falloff;
}

float2 OrientedPatchOffset(
    float2 localCoordinate,
    float2 cell,
    float2 sourceAxis,
    float patchRadius,
    float seed)
{
    const float2 randomPair = Hash22(cell + seed);
    const float angle = (randomPair.x - 0.5) * 1.57079633;
    const float2 rotatedLocal = Rotate2D(localCoordinate, angle);
    const float2 axis = normalize(sourceAxis + float2(1.0e-5, 0.0));
    const float2 perpendicular = float2(-axis.y, axis.x);
    const float2 orientedLocal = axis * rotatedLocal.x + perpendicular * rotatedLocal.y;
    const float2 jitter = (randomPair - 0.5) * 1.35;
    return (orientedLocal * 0.46 + jitter) * patchRadius;
}

float3 SourceAlbedoPatchPlane(
    float2 baseUv,
    float2 objectPlane,
    float2 sourceAxis,
    float patchFrequency,
    float patchRadius,
    float blurLod,
    float seed)
{
    const float2 patchCoordinate = objectPlane * patchFrequency;
    const float2 baseCell = floor(patchCoordinate);
    const float2 fractional = frac(patchCoordinate);
    float3 accumulated = 0.0;
    float accumulatedWeight = 0.0;

    [unroll]
    for (int y = 0; y < 2; ++y)
    {
        [unroll]
        for (int x = 0; x < 2; ++x)
        {
            const float2 corner = float2((float)x, (float)y);
            const float2 cell = baseCell + corner;
            const float2 localCoordinate = fractional - corner;
            const float weight = PatchWeight(localCoordinate);
            const float2 patchUv = baseUv + OrientedPatchOffset(
                localCoordinate, cell, sourceAxis, patchRadius, seed);
            accumulated += SampleAlbedoHighPass(patchUv, blurLod) * weight;
            accumulatedWeight += weight;
        }
    }
    return accumulated / max(accumulatedWeight, 1.0e-5);
}

float2 SourceNormalPatchPlane(
    float2 baseUv,
    float2 objectPlane,
    float2 sourceAxis,
    float patchFrequency,
    float patchRadius,
    float blurLod,
    float seed)
{
    const float2 patchCoordinate = objectPlane * patchFrequency;
    const float2 baseCell = floor(patchCoordinate);
    const float2 fractional = frac(patchCoordinate);
    float2 accumulated = 0.0;
    float accumulatedWeight = 0.0;

    [unroll]
    for (int y = 0; y < 2; ++y)
    {
        [unroll]
        for (int x = 0; x < 2; ++x)
        {
            const float2 corner = float2((float)x, (float)y);
            const float2 cell = baseCell + corner;
            const float2 localCoordinate = fractional - corner;
            const float weight = PatchWeight(localCoordinate);
            const float2 patchUv = baseUv + OrientedPatchOffset(
                localCoordinate, cell, sourceAxis, patchRadius, seed);
            accumulated += SampleNormalHighPass(patchUv, blurLod) * weight;
            accumulatedWeight += weight;
        }
    }
    return accumulated / max(accumulatedWeight, 1.0e-5);
}

float3 SourceAlbedoPatchTransfer(
    float2 baseUv,
    float3 objectPosition,
    float3 geometricNormal,
    float2 sourceAxis,
    float patchFrequency,
    float patchRadius,
    float blurLod)
{
    float3 weights = pow(abs(geometricNormal), 8.0);
    weights /= max(weights.x + weights.y + weights.z, 1.0e-5);
    const float3 xy = SourceAlbedoPatchPlane(baseUv, objectPosition.xy, sourceAxis, patchFrequency, patchRadius, blurLod, 11.7);
    const float3 yz = SourceAlbedoPatchPlane(baseUv, objectPosition.yz, sourceAxis, patchFrequency, patchRadius, blurLod, 37.1);
    const float3 zx = SourceAlbedoPatchPlane(baseUv, objectPosition.zx, sourceAxis, patchFrequency, patchRadius, blurLod, 73.9);
    return xy * weights.z + yz * weights.x + zx * weights.y;
}

float2 SourceNormalPatchTransfer(
    float2 baseUv,
    float3 objectPosition,
    float3 geometricNormal,
    float2 sourceAxis,
    float patchFrequency,
    float patchRadius,
    float blurLod)
{
    float3 weights = pow(abs(geometricNormal), 8.0);
    weights /= max(weights.x + weights.y + weights.z, 1.0e-5);
    const float2 xy = SourceNormalPatchPlane(baseUv, objectPosition.xy, sourceAxis, patchFrequency, patchRadius, blurLod, 19.3);
    const float2 yz = SourceNormalPatchPlane(baseUv, objectPosition.yz, sourceAxis, patchFrequency, patchRadius, blurLod, 41.9);
    const float2 zx = SourceNormalPatchPlane(baseUv, objectPosition.zx, sourceAxis, patchFrequency, patchRadius, blurLod, 89.7);
    return xy * weights.z + yz * weights.x + zx * weights.y;
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
    float3 materialBase = max(BlendMaterialColour(materialWeights), 0.001);
    float3 materialF0 = clamp(BlendMaterialF0(materialWeights), 0.0, 1.0);
    float materialGloss = max(BlendMaterialGloss(materialWeights), 0.0);

    float4 sampledAlbedoRgba = mode == 3
        ? SampleNSAMDRAlbedo(uv)
        : gAlbedo.Sample(gTextureSampler, uv);
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
    // Repair eligibility comes from the mesh-space UV Jacobian and therefore
    // remains stable while the camera moves. Screen derivatives remain diagnostic
    // evidence only and no longer cause the repair to crawl across clean surfaces.
    float damage = gOptions.z > 0.5 ? 1.0 : saturate(input.stretchHint);

    float texelSpanX = max(abs(duvdx.x) * gDiagnostics.z, abs(duvdy.x) * gDiagnostics.z);
    float texelSpanY = max(abs(duvdx.y) * gDiagnostics.w, abs(duvdy.y) * gDiagnostics.w);
    float estimatedMip = log2(max(max(texelSpanX, texelSpanY), 1.0));
    float actualMipNormalized = saturate((estimatedMip + 1.0) / 7.0);

    if (mode == 2)
    {
        float3 dirColour = 0.5 + 0.5 * float3(input.stretchAxis.x, input.stretchAxis.y, 1.0 - abs(input.stretchAxis.x));
        float3 mipColour = float3(actualMipNormalized, derivativeDamage, anisotropyDamage);
        float3 colour = lerp(DamageHeatmap(damage), dirColour, 0.35);
        colour = lerp(colour, mipColour, 0.25);
        return float4(saturate(colour), 1.0);
    }

    float3 mappedNormal = geometricNormal;
    float2 authoredNormalXY = 0.0;
    if (gOptions.x > 0.5)
    {
        authoredNormalXY = (mode == 3
            ? SampleNSAMDRNormalXY(uv)
            : SampleNormalXY(uv, 0.0)) * gSurface.y;
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
    if (mode == 3)
    {
        // Normal-variance compensation retains reconstructed detail without the
        // sparkling that would otherwise force us back to a visibly fuzzy mip.
        const float normalVariation = saturate(
            (length(ddx(n)) + length(ddy(n))) * 0.38);
        roughness = clamp(sqrt(roughness * roughness + normalVariation * 0.055), 0.04, 0.98);
    }
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
