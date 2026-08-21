// Copyright © 2026 CCP ehf.

#include "StdAfx.h"
#include "EveBoosterUtilities.h"
#include "Resources/TriGeometryRes.h"

namespace
{

template<typename Vertex>
ALResult GetBoxVB( Tr2SuballocatedBuffer::Allocation& vb, Tr2PrimaryRenderContext& renderContext )
{
	const uint32_t vertexCount = 4 * 6;
	Vertex vertices[vertexCount];
	auto p = &vertices[0];
	( p++ )->position = Vector3( -1.0f, -1.0f, 0.0f );
	( p++ )->position = Vector3( 1.0f, -1.0f, 0.0f );
	( p++ )->position = Vector3( 1.0f, 1.0f, 0.0f );
	( p++ )->position = Vector3( -1.0f, 1.0f, 0.0f );

	( p++ )->position = Vector3( -1.0f, -1.0f, -1.0f );
	( p++ )->position = Vector3( -1.0f, 1.0f, -1.0f );
	( p++ )->position = Vector3( 1.0f, 1.0f, -1.0f );
	( p++ )->position = Vector3( 1.0f, -1.0f, -1.0f );

	( p++ )->position = Vector3( -1.0f, -1.0f, 0.0f );
	( p++ )->position = Vector3( -1.0f, 1.0f, 0.0f );
	( p++ )->position = Vector3( -1.0f, 1.0f, -1.0f );
	( p++ )->position = Vector3( -1.0f, -1.0f, -1.0f );

	( p++ )->position = Vector3( 1.0f, -1.0f, 0.0f );
	( p++ )->position = Vector3( 1.0f, -1.0f, -1.0f );
	( p++ )->position = Vector3( 1.0f, 1.0f, -1.0f );
	( p++ )->position = Vector3( 1.0f, 1.0f, 0.0f );

	( p++ )->position = Vector3( -1.0f, -1.0f, 0.0f );
	( p++ )->position = Vector3( -1.0f, -1.0f, -1.0f );
	( p++ )->position = Vector3( 1.0f, -1.0f, -1.0f );
	( p++ )->position = Vector3( 1.0f, -1.0f, 0.0f );

	( p++ )->position = Vector3( -1.0f, 1.0f, 0.0f );
	( p++ )->position = Vector3( 1.0f, 1.0f, 0.0f );
	( p++ )->position = Vector3( 1.0f, 1.0f, -1.0f );
	( p++ )->position = Vector3( -1.0f, 1.0f, -1.0f );

	return g_sharedBuffer.Allocate( sizeof( Vertex ), vertexCount, &vertices[0], renderContext, vb );
}

ALResult GetStarVB( Tr2SuballocatedBuffer::Allocation& vb, Tr2PrimaryRenderContext& renderContext )
{
	const uint32_t vertexCount = 4 * 4;
	EveBoosterVertex vertices[vertexCount];
	auto p = &vertices[0];
	for( unsigned int i = 0; i < vertexCount; i += 4 )
	{
		float t = (float)i * XM_PI / 4.f / 4.f;
		float x = cos( t ) * 0.5f;
		float y = sin( t ) * 0.5f;
		p->position = Vector3( -x, -y, 0.f );
		p->texCoord = Vector2( 1.f, 1.f );
		++p;
		p->position = Vector3( -x, -y, -1.f );
		p->texCoord = Vector2( 1.f, 0.f );
		++p;
		p->position = Vector3( x, y, -1.f );
		p->texCoord = Vector2( 0.f, 0.f );
		++p;
		p->position = Vector3( x, y, 0.0f );
		p->texCoord = Vector2( 0.f, 1.f );
		++p;
	}

	return g_sharedBuffer.Allocate( sizeof( EveBoosterVertex ), vertexCount, &vertices[0], renderContext, vb );
}

}

Tr2ProceduralBuffer MakeChildBoosterBoxBuffer()
{
	return Tr2ProceduralBuffer( BlueSharedString( "ChildBoosterBoxVB" ), GetBoxVB<EveChildBoosterVertex> );
}

Tr2ProceduralBuffer MakeBoosterBoxBuffer()
{
	return Tr2ProceduralBuffer( BlueSharedString( "BoosterBoxVB" ), GetBoxVB<EveBoosterVertex> );
}

Tr2ProceduralBuffer MakeBoosterStarBuffer()
{
	return Tr2ProceduralBuffer( BlueSharedString( "BoosterStarVB" ), GetStarVB );
}
