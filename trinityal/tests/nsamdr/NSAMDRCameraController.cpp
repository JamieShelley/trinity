#include "StdAfx.h"
#include "NSAMDRCameraController.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
void CameraController::GetCameraBasis(const PreviewState& state, XMFLOAT3& right, XMFLOAT3& up, XMFLOAT3& forward)
{
    const float cosPitch = std::cos(state.orbitPitch);
    forward = Normalize3(XMFLOAT3(
        -std::sin(state.orbitYaw) * cosPitch,
        -std::sin(state.orbitPitch),
        std::cos(state.orbitYaw) * cosPitch));
    right = Normalize3(Cross3(XMFLOAT3(0.0f, 1.0f, 0.0f), forward));
    up = Normalize3(Cross3(forward, right));
}

void CameraController::BuildCameraMatrices(
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

bool CameraController::RayTriangleIntersection(
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

bool CameraController::PickMeshAtScreenPoint(
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

bool CameraController::FocusCameraAtScreenPoint(
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

void CameraController::ApplyZoomRequest(
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

} // namespace nsamdr
