// Copyright © 2023 CCP ehf.

#pragma once
#ifndef BVH_H
#define BVH_H

#include "../Resources/Tr2CmfContent.h"
#include "ITr2DebugRenderer2.h"
#include "GeometryUtils.h"

namespace BVH
{

const int32_t BVH_MAX_NODE_SIZE = 4;

struct alignas( 16 ) BVHNode
{
	Vector3 boundsMin;
	uint32_t firstChildIndex : 26;
	uint32_t numObj : 3;
	uint32_t leaf : 1;
	uint32_t magicalPadding : 2;	// this exists to prevent performance degradation due to simd interpreting its .w component as denormal float
	Vector3 boundsMax;
	uint32_t padding = 0;

	BVHNode() : 
		boundsMin( {} ),
		firstChildIndex( 0 ),
		numObj( 0 ),
		leaf( 0 ),
		magicalPadding( 3 ),
		boundsMax( {} ),
		padding( 0 )
	{
	}
};

struct alignas( 16 ) BVHLeafTriangle
{
	Vector3 vertex0;
	uint32_t element : 30;
	uint32_t magicalPadding : 2;	// this exists to prevent performance degradation due to simd interpreting its .w component as denormal float
	Vector3 edge1;
	uint32_t padding0;
	Vector3 edge2;
	uint32_t padding1;

	BVHLeafTriangle() :
		vertex0( {} ),
		element( 0 ),
		magicalPadding( 3 ),
		edge1( {} ),
		padding0( 0 ),
		edge2( {} ),
		padding1( 0 )
	{
	}
};

// BVHNode and BVHLeafTriangle data layout have been optimized for SIMD. That's why there is padding in those structs, and why we assert it here.
static_assert( sizeof( BVHNode ) == 32 );
static_assert( offsetof( BVHNode, boundsMax ) == 16 );
static_assert( sizeof( BVHLeafTriangle ) == 48 );
static_assert( offsetof( BVHLeafTriangle, edge1 ) == 16 );

struct BoundingVolumeHierarchy
{
	std::vector<BVHLeafTriangle> triangles;
	std::vector<BVHNode> nodes;
};

struct IntersectedNode
{
	const BVHNode* node;
	float distance;
};

struct BVHContent
{
	Tr2CmfContents content;
	std::vector<int32_t> lodIndices;
	std::vector<BoundingVolumeHierarchy> bvhs;
};

cmf::ConstIndexBufferStream GetIndices( BVHContent& self, int meshIndex );
cmf::ConstBufferElementStream<Vector3> GetPositions( BVHContent& self, int meshIndex );
std::optional<cmf::ConstBufferElementStream<std::array<uint32_t, 4>>> GetBones( BVHContent& self, int meshIndex );
std::optional<cmf::ConstBufferElementStream<Vector4>> GetColors( BVHContent& self, int meshIndex );

BVHContent CreateBVHContent( Tr2CmfContents& content, const std::vector<int32_t>& lodIndex );

bool Intersection(
	const BVHContent& bvhContent,
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