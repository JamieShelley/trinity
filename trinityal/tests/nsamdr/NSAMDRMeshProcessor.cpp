#include "StdAfx.h"
#include "NSAMDRMeshProcessor.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
int MeshProcessor::ResolveObjIndex(int rawIndex, size_t count)
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

bool MeshProcessor::ParseIntegerPart(const std::string& text, int& value)
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

bool MeshProcessor::ParseObjCornerToken(
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

float MeshProcessor::ComputeTriangleDensityAndAnisotropy(
    const std::array<XMFLOAT3, 3>& p,
    const std::array<XMFLOAT2, 3>& uv,
    float& anisotropy,
    XMFLOAT2& stretchAxis,
    XMFLOAT2& stretchMagnitudes,
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
        stretchAxis = XMFLOAT2(1.0f, 0.0f);
        stretchMagnitudes = XMFLOAT2(1024.0f, 1024.0f);
        degenerate = true;
        return 1024.0f;
    }

    const float inverse = 1.0f / determinant;
    const XMFLOAT3 dpdu = Multiply3(Subtract3(Multiply3(e1, dv2), Multiply3(e2, dv1)), inverse);
    const XMFLOAT3 dpdv = Multiply3(Add3(Multiply3(e1, -du2), Multiply3(e2, du1)), inverse);

    // Eigen-analysis of J^T J gives both the physical stretch ratio and the
    // stable source-UV axis along which the surface is stretched most strongly.
    const float a = Dot3(dpdu, dpdu);
    const float b = Dot3(dpdu, dpdv);
    const float c = Dot3(dpdv, dpdv);
    const float trace = a + c;
    const float determinantMetric = std::max(a * c - b * b, 0.0f);
    const float discriminant = std::sqrt(std::max(trace * trace * 0.25f - determinantMetric, 0.0f));
    const float lambdaMaximum = std::max(trace * 0.5f + discriminant, 1.0e-12f);
    const float lambdaMinimum = std::max(trace * 0.5f - discriminant, 1.0e-12f);
    const float sigmaMaximum = std::sqrt(lambdaMaximum);
    const float sigmaMinimum = std::sqrt(lambdaMinimum);

    if (std::abs(b) > 1.0e-8f)
    {
        const float axisX = lambdaMaximum - c;
        const float axisY = b;
        const float axisLength = std::sqrt(axisX * axisX + axisY * axisY);
        stretchAxis = axisLength > 1.0e-8f
            ? XMFLOAT2(axisX / axisLength, axisY / axisLength)
            : XMFLOAT2(1.0f, 0.0f);
    }
    else
    {
        stretchAxis = a >= c ? XMFLOAT2(1.0f, 0.0f) : XMFLOAT2(0.0f, 1.0f);
    }

    anisotropy = sigmaMaximum / std::max(sigmaMinimum, 1.0e-8f);
    stretchMagnitudes = XMFLOAT2(sigmaMaximum, sigmaMinimum);
    degenerate = false;
    return std::sqrt(sigmaMaximum * sigmaMinimum);
}

bool MeshProcessor::LoadObjMesh(const std::string& path, ObjMesh& mesh, std::string& error)
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
                triangle.stretchAxis,
                triangle.stretchMagnitudes,
                triangle.degenerateUv);
        }

        triangle.repairUvCenter = XMFLOAT2(
            (triangle.uvs[0].x + triangle.uvs[1].x + triangle.uvs[2].x) / 3.0f,
            (triangle.uvs[0].y + triangle.uvs[1].y + triangle.uvs[2].y) / 3.0f);

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

    std::vector<float> rawStretchScores;
    rawStretchScores.reserve(buildTriangles.size());
    for (const TriangleBuildData& triangle : buildTriangles)
    {
        float rawStretch = 1.0f;
        if (!triangle.degenerateUv)
        {
            const float anisotropyScore = std::log2(std::max(triangle.anisotropy, 1.0f));
            const float densityScore = std::max(
                std::log2(std::max(triangle.density / medianDensity, 1.0e-8f)),
                0.0f);
            rawStretch = ClampFloat(
                std::max(anisotropyScore / 4.0f, densityScore / 3.0f),
                0.0f,
                1.0f);
        }
        rawStretchScores.push_back(rawStretch);
    }

    std::vector<float> sortedStretchScores = rawStretchScores;
    std::sort(sortedStretchScores.begin(), sortedStretchScores.end());
    const auto stretchPercentile = [&sortedStretchScores](float percentile) {
        if (sortedStretchScores.empty()) return 0.0f;
        const float position = ClampFloat(percentile, 0.0f, 1.0f) *
            static_cast<float>(sortedStretchScores.size() - 1U);
        const size_t lowerIndex = static_cast<size_t>(std::floor(position));
        const size_t upperIndex = std::min(lowerIndex + 1U, sortedStretchScores.size() - 1U);
        const float fraction = position - static_cast<float>(lowerIndex);
        return sortedStretchScores[lowerIndex] +
            (sortedStretchScores[upperIndex] - sortedStretchScores[lowerIndex]) * fraction;
    };
    const float stretchCalibrationLow = stretchPercentile(0.50f);
    const float stretchCalibrationHigh = std::max(
        stretchPercentile(0.95f),
        stretchCalibrationLow + 1.0e-4f);

    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.drawRanges.clear();
    mesh.vertices.reserve(buildTriangles.size() * 3U);
    mesh.indices.reserve(buildTriangles.size() * 3U);
    mesh.degenerateUvTriangles = 0;
    mesh.averageStretch = 0.0f;
    mesh.maximumStretch = 0.0f;

    for (size_t triangleIndex = 0; triangleIndex < buildTriangles.size(); ++triangleIndex)
    {
        const TriangleBuildData& triangle = buildTriangles[triangleIndex];
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

        const float rawStretch = rawStretchScores[triangleIndex];
        const float stretch = ClampFloat(
            (rawStretch - stretchCalibrationLow) /
                std::max(stretchCalibrationHigh - stretchCalibrationLow, 1.0e-4f),
            0.0f,
            1.0f);
        if (triangle.degenerateUv)
        {
            ++mesh.degenerateUvTriangles;
        }

        mesh.averageStretch += stretch;
        mesh.maximumStretch = std::max(mesh.maximumStretch, stretch);

        const XMFLOAT2 repairUvScale(
            ClampFloat(triangle.stretchMagnitudes.x / medianDensity, 1.0f, 3.5f),
            ClampFloat(triangle.stretchMagnitudes.y / medianDensity, 1.0f, 3.5f));

        for (size_t cornerIndex = 0; cornerIndex < 3; ++cornerIndex)
        {
            const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back({
                triangle.positions[cornerIndex],
                triangle.normals[cornerIndex],
                triangle.uvs[cornerIndex],
                stretch,
                triangle.stretchAxis,
                triangle.repairUvCenter,
                repairUvScale});
            mesh.indices.push_back(index);
        }
        mesh.drawRanges.back().indexCount += 3U;
    }

    mesh.triangleCount = static_cast<uint32_t>(buildTriangles.size());
    if (mesh.triangleCount > 0)
    {
        mesh.averageStretch /= static_cast<float>(mesh.triangleCount);
    }
    mesh.stretchCalibrationLow = stretchCalibrationLow;
    mesh.stretchCalibrationHigh = stretchCalibrationHigh;
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

} // namespace nsamdr
