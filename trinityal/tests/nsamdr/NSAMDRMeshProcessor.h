#pragma once

#include "NSAMDRPreviewTypes.h"

namespace nsamdr
{
class MeshProcessor final
{
public:
    int ResolveObjIndex(int rawIndex, size_t count);
    bool ParseIntegerPart(const std::string& text, int& value);
    bool ParseObjCornerToken( const std::string& token, size_t positionCount, size_t texcoordCount, size_t normalCount, ObjCorner& corner);
    float ComputeTriangleDensityAndAnisotropy( const std::array<XMFLOAT3, 3>& p, const std::array<XMFLOAT2, 3>& uv, float& anisotropy, XMFLOAT2& stretchAxis, XMFLOAT2& stretchMagnitudes, bool& degenerate);
    bool LoadObjMesh(const std::string& path, ObjMesh& mesh, std::string& error);
};

} // namespace nsamdr
