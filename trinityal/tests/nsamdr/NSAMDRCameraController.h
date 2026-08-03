#pragma once

#include "NSAMDRPreviewTypes.h"

namespace nsamdr
{
class CameraController final
{
public:
    void GetCameraBasis(const PreviewState& state, XMFLOAT3& right, XMFLOAT3& up, XMFLOAT3& forward);
    void BuildCameraMatrices( const PreviewState& state, uint32_t width, uint32_t height, XMMATRIX& world, XMMATRIX& view, XMMATRIX& projection, XMFLOAT3& eye, XMFLOAT3& target);
    bool RayTriangleIntersection( const XMFLOAT3& origin, const XMFLOAT3& direction, const XMFLOAT3& a, const XMFLOAT3& b, const XMFLOAT3& c, float& distance);
    bool PickMeshAtScreenPoint( const PreviewState& state, const ObjMesh& mesh, uint32_t width, uint32_t height, int screenX, int screenY, XMFLOAT3& hitPoint, float& hitDistance);
    bool FocusCameraAtScreenPoint( PreviewState& state, const ObjMesh& mesh, uint32_t width, uint32_t height, int screenX, int screenY);
    void ApplyZoomRequest( PreviewState& state, const ObjMesh& mesh, uint32_t width, uint32_t height);
};

} // namespace nsamdr
