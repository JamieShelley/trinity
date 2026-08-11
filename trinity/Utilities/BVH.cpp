#include "StdAfx.h"
#include "BVH.h"

namespace BVH
{

struct Primitive
{
	uint32_t element;
	CcpMath::AxisAlignedBox aabb;
};

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
		int dimension = FindLargestDimension( node.boundsMax - node.boundsMin );

		// If we were to move the bvh construction into cmf, we could choose more expensive splitting criteria. Right now this has to be fast.
		float target = ( node.boundsMax[dimension] + node.boundsMin[dimension] );
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
		auto leftChildAABB = CreateAABB( primitives, leftIndex, split );
		leftChild.boundsMin = leftChildAABB.m_min;
		leftChild.boundsMax = leftChildAABB.m_max;
		BVHNode rightChild;
		auto rightChildAABB = CreateAABB( primitives, split, rightIndex );
		rightChild.boundsMin = rightChildAABB.m_min;
		rightChild.boundsMax = rightChildAABB.m_max;
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
	bvh.triangles.reserve( elementCount );
	bvh.nodes.reserve( 2 * elementCount - 1 );

	auto triangleVertices = [&indices, &positions]( int32_t i, Vector3 vertices[3] ) {
		int index0 = indices( i * 3 );
		int index1 = indices( i * 3 + 1 );
		int index2 = indices( i * 3 + 2 );

		vertices[0] = positions( index0 );
		vertices[1] = positions( index1 );
		vertices[2] = positions( index2 );
	};

	for( uint32_t i = firstElement; i < firstElement + elementCount; i++ )
	{
		Primitive primitive;

		Vector3 vertices[3];
		triangleVertices( i, vertices );
		primitive.aabb.IncludePoint( vertices[0] );
		primitive.aabb.IncludePoint( vertices[1] );
		primitive.aabb.IncludePoint( vertices[2] );

		primitive.element = i;

		primitives.push_back( primitive );
	}

	int leftIndex = 0;
	int rightIndex = elementCount;
	BVHNode root;
	auto rootAABB = CreateAABB( primitives, 0, (int)primitives.size() );
	root.boundsMin = rootAABB.m_min;
	root.boundsMax = rootAABB.m_max;
	bvh.nodes.push_back( root );
	CreateBVHNodes( primitives, bvh.nodes, leftIndex, rightIndex, 0 );

	for( int i = 0; i < primitives.size(); i++ )
	{
		BVHLeafTriangle triangle;
		Vector3 vertices[3];
		triangleVertices( primitives[i].element, vertices );
		triangle.vertex0 = vertices[0];
		triangle.edge1 = vertices[1] - vertices[0];
		triangle.edge2 = vertices[2] - vertices[0];
		triangle.element = primitives[i].element;
		bvh.triangles.push_back( triangle );
	}

	bvh.nodes.shrink_to_fit();

	return bvh;
}

// modified version of IntersectAxisAlignedBoxRay
bool Intersects( const XMVECTOR& origin, const XMVECTOR& invRayDir, const BVHNode& node, float& distance )
{
	XMVECTOR minA = *(Vector4*)&node.boundsMin;
	XMVECTOR maxA = *(Vector4*)&node.boundsMax;

	XMVECTOR t0 = ( minA - origin ) * invRayDir;
	XMVECTOR t1 = ( maxA - origin ) * invRayDir;

	XMVECTOR smallerIntersection = XMVectorMin( t0, t1 );
	XMVECTOR biggerIntersection = XMVectorMax( t0, t1 );

	float minT = max( XMVectorGetX( smallerIntersection ), max( XMVectorGetY( smallerIntersection ), XMVectorGetZ( smallerIntersection ) ) );
	float maxT = min( XMVectorGetX( biggerIntersection ), min( XMVectorGetY( biggerIntersection ), XMVectorGetZ( biggerIntersection ) ) );

	distance = minT;

	return maxT > 0.f && maxT >= minT;
}

bool Intersection(
	const BoundingVolumeHierarchy& bvh,
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

	XMVECTOR rayOrigin = ray.origin;
	XMVECTOR rayDir = ray.direction;
	XMVECTOR invRayDir = XMVectorReciprocal( ray.direction );
	invRayDir = XMVectorClamp( 
		invRayDir, 
		XMVectorReplicate( -std::numeric_limits<float>::max() ), 
		XMVectorReplicate( std::numeric_limits<float>::max() ) 
	);

	uint32_t hitPrimitive;
	float hitDistance;
	if( !Intersects( rayOrigin, invRayDir, bvh.nodes[0], hitDistance ) || rayLength < hitDistance )
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
			bool hitLeft = Intersects( rayOrigin, invRayDir, *leftChild, leftDistance ) && leftDistance <= rayLength;
			bool hitRight = Intersects( rayOrigin, invRayDir, *rightChild, rightDistance ) && rightDistance <= rayLength;
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
				XMVECTOR vertex0 = *(Vector4*)&bvh.triangles[i].vertex0;
				XMVECTOR edge1 = *(Vector4*)&bvh.triangles[i].edge1;
				XMVECTOR edge2 = *(Vector4*)&bvh.triangles[i].edge2;
				if( IntersectTri( vertex0, edge1, edge2, rayOrigin, rayDir, &hitU, &hitV, &hitDistance ) )
				{
					if( hitDistance < rayLength )
					{
						rayLength = hitDistance;
						u = hitU;
						v = hitV;
						hitPrimitive = bvh.triangles[i].element;
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

cmf::ConstIndexBufferStream GetIndices( BVHContent& self, int meshIndex )
{
	const auto& mesh = self.content.GetData()->meshes[meshIndex];
	int32_t lodIndex = self.lodIndices[meshIndex];
	auto ib = mesh.lods[lodIndex].ib;
	auto ibSectionData = self.content.GetSection( ib.index );
	auto indices = cmf::ConstIndexBufferStream( ibSectionData, ib );
	return indices;
}

cmf::ConstBufferElementStream<Vector3> GetPositions( BVHContent& self, int meshIndex )
{
	const auto& mesh = self.content.GetData()->meshes[meshIndex];
	int32_t lodIndex = self.lodIndices[meshIndex];
	auto positionElement = cmf::FindElement( mesh.decl, cmf::Usage::Position );
	auto vb = mesh.lods[lodIndex].vb;
	uint32_t numVerts = cmf::GetStreamElementCount( vb );
	auto vertices = self.content.GetViewData( vb );
	cmf::ConstBufferElementStream<Vector3> positions( *positionElement, vertices, numVerts, vb.stride );
	return positions;
}

std::optional<cmf::ConstBufferElementStream<std::array<uint32_t, 4>>> GetBones( BVHContent& self, int meshIndex )
{
	const auto& mesh = self.content.GetData()->meshes[meshIndex];
	int32_t lodIndex = self.lodIndices[meshIndex];
	auto boneElement = cmf::FindElement( mesh.decl, cmf::Usage::BoneIndices );
	auto vb = mesh.lods[lodIndex].vb;
	uint32_t numVerts = cmf::GetStreamElementCount( vb );
	auto vertices = self.content.GetViewData( vb );
	std::optional<cmf::ConstBufferElementStream<std::array<uint32_t, 4>>> bones;
	if( boneElement )
	{
		bones.emplace( *boneElement, vertices, numVerts, vb.stride );
	}
	return bones;
}

std::optional<cmf::ConstBufferElementStream<Vector4>> GetColors( BVHContent& self, int meshIndex )
{
	const auto& mesh = self.content.GetData()->meshes[meshIndex];
	int32_t lodIndex = self.lodIndices[meshIndex];
	auto colorElement = cmf::FindElement( mesh.decl, cmf::Usage::Color );
	auto vb = mesh.lods[lodIndex].vb;
	uint32_t numVerts = cmf::GetStreamElementCount( vb );
	auto vertices = self.content.GetViewData( vb );
	std::optional<cmf::ConstBufferElementStream<Vector4>> colors;
	if( colorElement )
	{
		colors.emplace( *colorElement, vertices, numVerts, vb.stride );
	}
	return colors;
}

BVHContent CreateBVHContent( Tr2CmfContents& content, const std::vector<int32_t>& lodIndices )
{
	BVHContent bvhContent;
	bvhContent.content = std::move( content );
	bvhContent.lodIndices = lodIndices;
	for( int32_t i = 0; i < bvhContent.content.GetData()->meshes.size(); i++ )
	{
		const auto& mesh = bvhContent.content.GetData()->meshes[i];

		if( mesh.topology != cmf::MeshTopology::TriangleList )
		{
			continue;
		}

		auto indices = GetIndices( bvhContent, i );
		auto positions = GetPositions( bvhContent, i );

		auto getIndex = [&indices]( int32_t i ) { return indices[i]; };
		auto getPositions = [&positions]( int32_t i ) { return positions[i]; };

		for( const auto& area : mesh.lods[lodIndices[i]].areas )
		{
			bvhContent.bvhs.push_back( CreateBVH( getIndex, getPositions, area.firstElement, area.elementCount ) );
		}
	}
	return bvhContent;
}

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
	float& distance )
{
	if( bvhContent.content.GetData()->meshes[meshIndex].topology != cmf::MeshTopology::TriangleList )
	{
		return false;
	}

	size_t areasOffset = 0;
	for( int32_t i = 0; i < meshIndex; i++ )
	{
		const auto& mesh = bvhContent.content.GetData()->meshes[i];
		if( mesh.topology != cmf::MeshTopology::TriangleList )
		{
			continue;
		}
		areasOffset += mesh.areas.size();
	}

	return Intersection( bvhContent.bvhs[areasOffset + areaIndex], stack, ray, rayLength, primitive, u, v, distance );
}

bool Intersection(
	const BVHContent& bvhContent,
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	int32_t meshIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance )
{
	if( bvhContent.content.GetData()->meshes[meshIndex].topology != cmf::MeshTopology::TriangleList )
	{
		return false;
	}

	size_t areasOffset = 0;
	for( int32_t i = 0; i < meshIndex; i++ )
	{
		const auto& mesh = bvhContent.content.GetData()->meshes[i];
		if( mesh.topology != cmf::MeshTopology::TriangleList )
		{
			continue;
		}
		areasOffset += mesh.areas.size();
	}

	bool hit = false;
	for( size_t areaIndex = 0; areaIndex < bvhContent.content.GetData()->meshes[meshIndex].areas.size(); areaIndex++ )
	{
		const auto& bvh = bvhContent.bvhs[areasOffset + areaIndex];
		uint32_t hitPrimitive;
		if( Intersection( bvh, stack, ray, rayLength, hitPrimitive, u, v, rayLength ) )
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
	std::vector<IntersectedNode>& stack,
	const CcpMath::Ray& ray,
	float rayLength,
	uint32_t& meshIndex,
	uint32_t& primitive,
	float& u,
	float& v,
	float& distance )
{
	bool hit = false;
	size_t areasOffset = 0;
	for( size_t i = 0; i < bvhContent.content.GetData()->meshes.size(); i++ )
	{
		const auto& mesh = bvhContent.content.GetData()->meshes[i];
		if( mesh.topology != cmf::MeshTopology::TriangleList )
		{
			continue;
		}

		for( size_t areaIndex = 0; areaIndex < mesh.areas.size(); areaIndex++ )
		{
			const auto& bvh = bvhContent.bvhs[areasOffset + areaIndex];
			uint32_t hitPrimitive;
			if( Intersection( bvh, stack, ray, rayLength, hitPrimitive, u, v, rayLength ) )
			{
				meshIndex = (uint32_t)i;
				primitive = hitPrimitive;
				distance = rayLength;
				hit = true;
			}
		}
		areasOffset += mesh.areas.size();
	}
	return hit;
}

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
	float& distance )
{
	bool hit = false;
	size_t areasOffset = 0;
	for( size_t i = 0; i < bvhContent.content.GetData()->meshes.size(); i++ )
	{
		const auto& mesh = bvhContent.content.GetData()->meshes[i];
		if( mesh.topology != cmf::MeshTopology::TriangleList )
		{
			continue;
		}

		if( areaIndex < mesh.areas.size() )
		{
			const auto& bvh = bvhContent.bvhs[areasOffset + areaIndex];
			uint32_t hitPrimitive;
			if( Intersection( bvh, stack, ray, rayLength, hitPrimitive, u, v, rayLength ) )
			{
				meshIndex = (uint32_t)i;
				primitive = hitPrimitive;
				distance = rayLength;
				hit = true;
			}
		}
		areasOffset += mesh.areas.size();
	}
	return hit;
}

void Visualize( const BVHNode& node, Tr2DebugObjectReference owner, const Matrix& transform, ITr2DebugRenderer2& renderer )
{
	if( node.leaf )
	{
		renderer.DrawBox( owner, transform, node.boundsMin, node.boundsMax, ITr2DebugRenderer2::Wireframe, Color( 1.f, 0.f, 0.f, 1.f ) );
	}
	else
	{
		renderer.DrawBox( owner, transform, node.boundsMin, node.boundsMax, ITr2DebugRenderer2::Wireframe, Color( 1.f, 1.f, 1.f, 1.f ) );
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
