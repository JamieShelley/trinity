#include "StdAfx.h"
#include "BVH.h"

namespace BVH
{
/*
const int32_t BVH_MAX_NODE_SIZE = 4; // TODO: intern, play around with this value and profile...
struct BVHNode
{
	CcpMath::AxisAlignedBox aabb;
	//CcpMath::Sphere sphere;
	uint32_t firstChildIndex : 28;
	uint32_t numObj : 3;
	uint32_t leaf : 1;
};
*/
struct Primitive
{
	// TODO: intern, test if storing triangles is faster
	uint32_t element;

	CcpMath::AxisAlignedBox aabb;
	//CcpMath::Sphere sphere;
};
/*
struct BoundingVolumeHierarchy
{
	std::vector<uint32_t> primitives;
	std::vector<BVHNode> nodes;
};
*/
int FindLargestDimension( Vector3 dimensions )
{
	if( dimensions.x > dimensions.y )
	{
		if( dimensions.x > dimensions.z )
		{
			return 0;
		}
		else
		{
			return 2;
		}
	}
	else
	{
		if( dimensions.y > dimensions.z )
		{
			return 1;
		}
		else
		{
			return 2;
		}
	}
}

//int FindSplitIndex( const std::vector<Primitive>& primitives, int leftIndex, int rightIndex, int dimension, const CcpMath::AxisAlignedBox& aabb )
//{
//	float target = ( aabb.m_max[dimension] + aabb.m_min[dimension] );
//
//	// TODO: intern, compare with linear search. binary search requires sorted vector...
//	auto iter = std::upper_bound( primitives.begin() + leftIndex, primitives.begin() + rightIndex, target,
//		[dimension]( float target, const Primitive& p )
//		{
//			return target < ( p.aabb.m_max[dimension] + p.aabb.m_min[dimension] );
//		}
//	);
//	int mid = (int)std::distance( primitives.begin(), iter );
//	if( mid <= leftIndex || mid >= rightIndex )
//	{
//		return ( rightIndex + leftIndex ) >> 1;
//	}
//
//	return mid;
//}

CcpMath::AxisAlignedBox CreateAABB( const std::vector<Primitive>& primitives, int from, int to )
{
	CcpMath::AxisAlignedBox aabb{};
	for( int i = from; i < to; i++ )
	{
		aabb.Include( primitives[i].aabb );
	}
	return aabb;
}

void CreateBVHNodes(
	std::vector<Primitive>& primitives,
	std::vector<BVHNode>& nodes,
	int leftIndex,
	int rightIndex,
	uint32_t nodeIndex )
{
	BVHNode& node = nodes[nodeIndex];
	if( rightIndex - leftIndex > BVH_MAX_NODE_SIZE )
	{
		int dimension = FindLargestDimension( node.aabb.m_max - node.aabb.m_min );

		// TODO: intern, try if linear search and partition might work better than sort and binary search. problem: fallback still needs median...
		//std::sort( primitives.begin() + leftIndex, primitives.begin() + rightIndex,
		//	[dimension]( const Primitive& a, const Primitive& b )
		//	{
		//		return a.aabb.Center()[dimension] < b.aabb.Center()[dimension];
		//	}
		//);
		//int split = FindSplitIndex( primitives, leftIndex, rightIndex, dimension, node.aabb );

		// TODO: intern, experiment with splitting criteria
		float target = ( node.aabb.m_max[dimension] + node.aabb.m_min[dimension] );
		auto partitionedElement = std::partition( primitives.begin() + leftIndex, primitives.begin() + rightIndex, [dimension, target]( const Primitive& p ) {
			return ( p.aabb.m_max[dimension] + p.aabb.m_min[dimension] ) < target;
		} );

		int split = (int)std::distance( primitives.begin(), partitionedElement );
		if( split <= leftIndex || split >= rightIndex )
		{
			split = ( rightIndex + leftIndex ) >> 1;
			std::nth_element( primitives.begin() + leftIndex, primitives.begin() + split, primitives.begin() + rightIndex, [dimension]( const Primitive& a, const Primitive& b ) {
				return ( a.aabb.m_max[dimension] + a.aabb.m_min[dimension] ) < ( b.aabb.m_max[dimension] + b.aabb.m_min[dimension] );
			} );
		}

		node.firstChildIndex = nodes.size();
		node.numObj = 2;
		node.leaf = false;
		BVHNode leftChild;
		leftChild.aabb = CreateAABB( primitives, leftIndex, split );
		BVHNode rightChild;
		rightChild.aabb = CreateAABB( primitives, split, rightIndex );
		nodes.push_back( leftChild );
		nodes.push_back( rightChild );
		uint32_t leftChildIndex = (uint32_t)nodes.size() - 2;
		uint32_t rightChildIndex = (uint32_t)nodes.size() - 1;
		CreateBVHNodes( primitives, nodes, leftIndex, split, leftChildIndex );
		CreateBVHNodes( primitives, nodes, split, rightIndex, rightChildIndex );
	}
	else
	{
		node.firstChildIndex = leftIndex;
		node.numObj = rightIndex - leftIndex;
		node.leaf = true;
	}
}

template <typename GetIndex, typename GetPosition>
BoundingVolumeHierarchy CreateBVH(
	const GetIndex& indices,
	const GetPosition& positions,
	uint32_t firstElement,
	uint32_t elementCount )
{
	BoundingVolumeHierarchy bvh{};

	if( elementCount == 0 )
	{
		return bvh;
	}

	std::vector<Primitive> primitives;
	primitives.reserve( elementCount );
	bvh.primitives.reserve( elementCount );
	bvh.nodes.reserve( 2 * elementCount - 1 );

	for( uint32_t i = firstElement; i < firstElement + elementCount; i++ )
	{
		Primitive primitive;
		int index0 = indices( i * 3 );
		int index1 = indices( i * 3 + 1 );
		int index2 = indices( i * 3 + 2 );

		Vector3 position0 = positions( index0 );
		Vector3 position1 = positions( index1 );
		Vector3 position2 = positions( index2 );

		primitive.aabb.IncludePoint( position0 );
		primitive.aabb.IncludePoint( position1 );
		primitive.aabb.IncludePoint( position2 );

		primitive.element = i;

		primitives.push_back( primitive );
	}

	int leftIndex = 0;
	int rightIndex = elementCount;
	BVHNode root;
	root.aabb = CreateAABB( primitives, 0, (int)primitives.size() );
	bvh.nodes.push_back( root );
	CreateBVHNodes( primitives, bvh.nodes, leftIndex, rightIndex, 0 );
	for( int i = 0; i < primitives.size(); i++ )
	{
		bvh.primitives.push_back( primitives[i].element );
	}

	bvh.nodes.shrink_to_fit();

	return bvh;
}

// modified version of IntersectAxisAlignedBoxRay
bool Intersects( const Vector3& origin, const XMVECTOR& invRayDir, const CcpMath::AxisAlignedBox& aabb, float& distance )
{
	XMVECTOR minA = XMVectorSet( aabb.m_min.x, aabb.m_min.y, aabb.m_min.z, 0.0f );
	XMVECTOR maxA = XMVectorSet( aabb.m_max.x, aabb.m_max.y, aabb.m_max.z, 0.0f );

	XMVECTOR t0 = ( minA - origin ) * invRayDir;
	XMVECTOR t1 = ( maxA - origin ) * invRayDir;

	XMVECTOR smallerIntersection = XMVectorMin( t0, t1 );
	XMVECTOR biggerIntersection = XMVectorMax( t0, t1 );

	float minT = max( XMVectorGetX( smallerIntersection ), max( XMVectorGetY( smallerIntersection ), XMVectorGetZ( smallerIntersection ) ) );
	float maxT = min( XMVectorGetX( biggerIntersection ), min( XMVectorGetY( biggerIntersection ), XMVectorGetZ( biggerIntersection ) ) );

	distance = minT;

	return maxT > 0.f && maxT >= minT;
}
/*
struct IntersectedNode
{
	const BVHNode* node;
	float distance;
};
*/
template <typename GetIndex, typename GetPosition>
bool Intersection(
	const BoundingVolumeHierarchy& bvh,
	const GetIndex& indices,
	const GetPosition& positions,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance )
{
	if( bvh.nodes.empty() )
	{
		return false;
	}

	XMVECTOR invRayDir = XMVectorReciprocal( ray.direction );

	uint32_t hitPrimitive;
	float hitDistance;
	if( !Intersects( ray.origin, invRayDir, bvh.nodes[0].aabb, hitDistance ) || rayLength < hitDistance )
	{
		return false;
	}

	stack.clear();
	bool hit = false;
	const BVHNode* currentNode = &bvh.nodes[0];
	while( true )
	{
		if( !currentNode->leaf )
		{
			const BVHNode* leftChild = &bvh.nodes[currentNode->firstChildIndex];
			const BVHNode* rightChild = &bvh.nodes[currentNode->firstChildIndex + 1];
			float leftDistance;
			float rightDistance;
			bool hitLeft = Intersects( ray.origin, invRayDir, leftChild->aabb, leftDistance ) && leftDistance <= rayLength;
			bool hitRight = Intersects( ray.origin, invRayDir, rightChild->aabb, rightDistance ) && rightDistance <= rayLength;
			if( hitLeft && hitRight )
			{
				if( leftDistance < rightDistance )
				{
					stack.push_back( IntersectedNode{ rightChild, rightDistance } );
					currentNode = leftChild;
				}
				else
				{
					stack.push_back( IntersectedNode{ leftChild, leftDistance } );
					currentNode = rightChild;
				}
				continue;
			}
			else if( hitLeft )
			{
				currentNode = leftChild;
				continue;
			}
			else if( hitRight )
			{
				currentNode = rightChild;
				continue;
			}
		}
		else
		{
			for( uint32_t i = currentNode->firstChildIndex; i < currentNode->firstChildIndex + currentNode->numObj; i++ )
			{
				float hitU, hitV;
				Vector3 vertex0 = positions( indices( bvh.primitives[i] * 3 + 0 ) );
				Vector3 vertex1 = positions( indices( bvh.primitives[i] * 3 + 1 ) );
				Vector3 vertex2 = positions( indices( bvh.primitives[i] * 3 + 2 ) );
				if( IntersectTri( &vertex0, &vertex1, &vertex2, &ray.origin, &ray.direction, &hitU, &hitV, &hitDistance ) )
				{
					if( hitDistance < rayLength )
					{
						rayLength = hitDistance;
						u = hitU;
						v = hitV;
						hitPrimitive = bvh.primitives[i];
						hit = true;
					}
				}
			}
		}
		while( stack.size() > 0 && stack.back().distance > rayLength )
		{
			stack.pop_back();
		}
		if( stack.size() > 0 )
		{
			currentNode = stack.back().node;
			stack.pop_back();
		}
		else
		{
			break;
		}
	}

	if( hit )
	{
		distance = rayLength;
		primitive = hitPrimitive;
	}
	return hit;
}
/*
struct BVHContent
{
	const cmf::Data* data;
	std::vector<BoundingVolumeHierarchy> bvhs;
};
*/
BVHContent CreateBVHContent( Tr2CmfContents& content, const std::vector<int32_t>& lodIndices )
{
	BVHContent bvhContent;
	bvhContent.data = content.GetData();
	for( int32_t i = 0; i < bvhContent.data->meshes.size(); i++ )
	{
		const auto& mesh = bvhContent.data->meshes[i];
		int32_t lodIndex = lodIndices[i];

		auto ib = mesh.lods[lodIndex].ib;
		auto ibSectionData = content.GetSection( ib.index );
		auto indices = cmf::ConstIndexBufferStream( ibSectionData, ib );

		auto element = cmf::FindElement( mesh.decl, cmf::Usage::Position );
		auto vb = mesh.lods[lodIndex].vb;
		uint32_t numVerts = cmf::GetStreamElementCount( vb );
		auto vertices = content.GetViewData( vb );
		cmf::ConstBufferElementStream<Vector3> positions( *element, vertices, numVerts, vb.stride );

		auto getIndex = [&indices]( int32_t i ) { return indices[i]; };
		auto getPositions = [&positions]( int32_t i ) { return positions[i]; };

		for( const auto& area : mesh.lods[lodIndex].areas )
		{
			bvhContent.bvhs.push_back( CreateBVH( getIndex, getPositions, area.firstElement, area.elementCount ) );
		}
	}
	return bvhContent;
}

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
	float& distance )
{
	size_t areasOffset = 0;
	for( int32_t i = 0; i < meshIndex; i++ )
	{
		areasOffset += bvhContent.data->meshes[i].areas.size();
	}

	return Intersection( bvhContent.bvhs[areasOffset + areaIndex], rayCaster.m_indices[meshIndex], rayCaster.m_positions[meshIndex], stack, ray, rayLength, primitive, u, v, distance );
}

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
	float& distance )
{
	size_t areasOffset = 0;
	for( int32_t i = 0; i < meshIndex; i++ )
	{
		areasOffset += bvhContent.data->meshes[i].areas.size();
	}

	bool hit = false;
	for( size_t areaIndex = 0; areaIndex < bvhContent.data->meshes[meshIndex].areas.size(); areaIndex++ )
	{
		const auto& bvh = bvhContent.bvhs[areasOffset + areaIndex];
		uint32_t hitPrimitive;
		if( Intersection( bvh, rayCaster.m_indices[meshIndex], rayCaster.m_positions[meshIndex], stack, ray, rayLength, hitPrimitive, u, v, rayLength ) )
		{
			primitive = hitPrimitive;
			distance = rayLength;
			hit = true;
		}
	}
	return hit;
}

bool Intersection(
	const BVHContent& bvhContent,
	const RayCaster& rayCaster,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	uint32_t& mesh,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance )
{
	bool hit = false;
	size_t areasOffset = 0;
	for( size_t meshIndex = 0; meshIndex < bvhContent.data->meshes.size(); meshIndex++ )
	{
		if( !rayCaster.m_indices[meshIndex].data || !rayCaster.m_positions[meshIndex].data )
		{
			continue;
		}

		for( size_t areaIndex = 0; areaIndex < bvhContent.data->meshes[meshIndex].areas.size(); areaIndex++ )
		{
			const auto& bvh = bvhContent.bvhs[areasOffset + areaIndex];
			uint32_t hitPrimitive;
			if( Intersection( bvh, rayCaster.m_indices[meshIndex], rayCaster.m_positions[meshIndex], stack, ray, rayLength, hitPrimitive, u, v, rayLength ) )
			{
				mesh = (uint32_t)meshIndex;
				primitive = hitPrimitive;
				distance = rayLength;
				hit = true;
			}
		}
		areasOffset += bvhContent.data->meshes[meshIndex].areas.size();
	}
	return hit;
}

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
	float& distance )
{
	bool hit = false;
	size_t areasOffset = 0;
	for( size_t i = 0; i < bvhContent.data->meshes.size(); i++ )
	{
		if( !rayCaster.m_indices[i].data || !rayCaster.m_positions[i].data || areaIndex >= bvhContent.data->meshes[i].areas.size() )
		{
			continue;
		}

		{
			const auto& bvh = bvhContent.bvhs[areasOffset + areaIndex];
			uint32_t hitPrimitive;
			if( Intersection( bvh, rayCaster.m_indices[i], rayCaster.m_positions[i], stack, ray, rayLength, hitPrimitive, u, v, rayLength ) )
			{
				meshIndex = (uint32_t)i;
				primitive = hitPrimitive;
				distance = rayLength;
				hit = true;
			}
		}
		areasOffset += bvhContent.data->meshes[i].areas.size();
	}
	return hit;
}

void Visualize( const BVHNode& node, Tr2DebugObjectReference owner, const Matrix& transform, ITr2DebugRenderer2& renderer )
{
	if( node.leaf )
	{
		renderer.DrawBox( owner, transform, node.aabb.m_min, node.aabb.m_max, ITr2DebugRenderer2::Wireframe, Color( 1.f, 0.f, 0.f, 1.f ) );
	}
	else
	{
		renderer.DrawBox( owner, transform, node.aabb.m_min, node.aabb.m_max, ITr2DebugRenderer2::Wireframe, Color( 1.f, 1.f, 1.f, 1.f ) );
	}
}

void Visualize( const BoundingVolumeHierarchy& bvh, Tr2DebugObjectReference owner, const Matrix& transform, ITr2DebugRenderer2& renderer )
{
	for( const auto& node : bvh.nodes )
	{
		Visualize( node, owner, transform, renderer );
	}
}


void Visualize( const BVHContent& bvhContent, Tr2DebugObjectReference owner, const Matrix& transform, ITr2DebugRenderer2& renderer )
{
	for( const auto& bvh : bvhContent.bvhs )
	{
		Visualize( bvh, owner, transform, renderer );
	}
}

}
