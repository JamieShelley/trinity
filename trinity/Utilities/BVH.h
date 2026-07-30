// Copyright © 2023 CCP ehf.

#pragma once
#ifndef BVH_H
#define BVH_H

#include "../Resources/Tr2CmfContent.h"
#include "ITr2DebugRenderer2.h"
#include "GeometryUtils.h"

namespace BVH
{

struct RayCastIndexReader
{
	const uint8_t* data = nullptr;
	bool stride16 = false;

	uint32_t operator()( uint32_t i ) const
	{
		return stride16 ? ( (uint16_t*)data )[i] : ( (uint32_t*)data )[i];
	}
};

struct RayCastPositionReader
{
	const uint8_t* data = nullptr;
	uint32_t vertexSize = 0;
	Tr2VertexDefinition::DataType dataType;

	Vector3 operator()( uint32_t i ) const
	{
		Vector3 v;
		ConvertDataToVector3( dataType, data + i * vertexSize, &v );
		return v;
	}
};

struct RayCastBoneReader
{
	const uint8_t* data = nullptr;
	uint32_t vertexSize = 0;
	Tr2VertexDefinition::DataType dataType;

	int32_t operator()( uint32_t i ) const
	{
		int32_t boneIndex;
		if( !GetBoneIndex( dataType, data + i * vertexSize, boneIndex ) )
		{
			boneIndex = -1;
		}
		return boneIndex;
	}
};

struct RayCaster
{
	std::vector<RayCastIndexReader> m_indices;
	std::vector<RayCastPositionReader> m_positions;
	std::vector<RayCastBoneReader> m_bones;
	std::vector<int32_t> m_lodIndices;
	bool m_prepared = false;
};

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

BVHContent CreateBVHContent( Tr2CmfContents& content, const std::vector<int32_t>& lodIndex );

bool Intersection(
	const BVHContent& bvhContent,
	const RayCaster& rayCaster,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	int32_t meshIndex,
	int32_t areaIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

bool Intersection(
	const BVHContent& bvhContent,
	const RayCaster& rayCaster,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	int32_t meshIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

bool Intersection(
	const BVHContent& bvhContent,
	const RayCaster& rayCaster,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	uint32_t& meshIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

bool Intersection(
	const BVHContent& bvhContent,
	const RayCaster& rayCaster,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	uint32_t areaIndex,
	uint32_t& meshIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance );

void Visualize( const BVHContent& bvhContent, Tr2DebugObjectReference owner, const Matrix& transform, ITr2DebugRenderer2& renderer );

}

#endif // BVH_H