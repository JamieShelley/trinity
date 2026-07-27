// Copyright © 2023 CCP ehf.

#pragma once
#ifndef BVH_H
#define BVH_H

#include "../Resources/Tr2CmfContent.h"
#include "ITr2DebugRenderer2.h"

namespace BVH
{

const int32_t BVH_MAX_NODE_SIZE = 4; // TODO: intern, play around with this value and profile...
struct BVHNode
{
	CcpMath::AxisAlignedBox aabb;
	//CcpMath::Sphere sphere;
	uint32_t firstChildIndex : 28;
	uint32_t numObj : 3;
	uint32_t leaf : 1;
};

struct BoundingVolumeHierarchy
{
	std::vector<uint32_t> primitives;
	std::vector<BVHNode> nodes;
};

struct IntersectedNode
{
	const BVHNode* node;
	float distance;
};

struct BVHContent
{
	const cmf::Data* data;
	std::vector<BoundingVolumeHierarchy> bvhs;
};

BVHContent CreateBVHContent( Tr2CmfContents& content, int32_t lodIndex );

template <typename GetIndex, typename GetPosition>
bool Intersection(
	const BVHContent& bvhContent,
	const GetIndex& indices,
	const GetPosition& positions,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	int32_t meshIndex,
	int32_t areaIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

template <typename GetIndex, typename GetPosition>
bool Intersection(
	const BVHContent& bvhContent,
	const GetIndex& indices,
	const GetPosition& positions,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	int32_t meshIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

template <typename GetIndex, typename GetPosition>
bool Intersection(
	const BVHContent& bvhContent,
	const std::vector<GetIndex>& indicesPerMesh,
	const std::vector<GetPosition>& positionsPerMesh,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

void Visualize( const BVHContent& bvhContent, Tr2DebugObjectReference owner, const Matrix& transform, ITr2DebugRenderer2& renderer );

}

#endif // BVH_H