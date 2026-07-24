// Copyright © 2026 CCP ehf.

#include "StdAfx.h"
#include "EveChildPartData.h"
#include "IEveEffectChildrenOwner.h"
#include "../EveStation2.h"
#include "EveChildInstancedMeshes.h"
#include "EveChildContainer.h"
#include <cmf/transforms.h>


EveChildPartData::EveChildPartData( IRoot* )
{
}

EveSpaceObjectChild::PartTag EveChildPartData::GetUnusedPartID() const
{
	return std::accumulate( m_parts.begin(), m_parts.end(), 1u, []( EveSpaceObjectChild::PartTag maxId, const PartData& part ) {
		return std::max( maxId, part.partId + 1 );
	} );
}



void EveModularObjectModifier::Create( SpaceObjectType* object, EveSOF* sof )
{
	m_object = object;
	m_sof = sof;
	for( auto& child : object->GetEffectChildren() )
	{
		if( EveChildPartDataPtr partData = BlueCastPtr( child ) )
		{
			m_data = partData;
			break;
		}
	}
	for( auto& child : object->GetEffectChildren() )
	{
		if( EveChildInstancedMeshesPtr instancedMeshes = BlueCastPtr( child ) )
		{
			m_instancedMeshes = instancedMeshes;
			break;
		}
	}
}

EveModularObjectModifier::~EveModularObjectModifier()
{
	if ( m_object )
	{
		CcpMath::Sphere bounds;
		for( auto& part : m_data->m_parts )
		{
			bounds.Include( part.boundingSphere );
		}
		m_object->SetBoundingSphereInformation( bounds );
	}
}

EveSpaceObjectChild::PartTag EveModularObjectModifier::AddHull( const char* hullName, const char* factionName, const char* raceName, const Vector3& position, const Quaternion& rotation, const Vector3& scale )
{
	auto id = m_data->GetUnusedPartID();
	auto size = m_object->GetEffectChildren().size();
	auto dna = std::string( hullName ) + ":" + ( factionName[0] ? factionName : m_data->m_faction.c_str() ) + ":" + ( raceName[0] ? raceName : m_data->m_race.c_str() );
	if ( !m_sof->BuildChild( m_object, dna.c_str(), id, TransformationMatrix( scale, rotation, position ) ) )
	{
		// TODO: return error
		return 0;
	}

	if( !m_instancedMeshes )
	{
		for( size_t i = size; i < m_object->GetEffectChildren().size(); ++i )
		{
			if( EveChildInstancedMeshesPtr instancedMesh = BlueCastPtr( m_object->GetEffectChildren()[i] ) )
			{
				m_instancedMeshes = instancedMesh;
				break;
			}
		}
	}

	// SOF will reset the bounding sphere of the object to the one of the part
	// Store the part bounding sphere and recalculate the bounding sphere of the modular object after adding all the parts
	CcpMath::Sphere sphere{ m_object->GetBoundingSphereCenter(), m_object->GetBoundingSphereRadius() };

	auto part = EveChildPartData::PartData{ id, position, rotation, scale, sphere };
	m_data->m_parts.emplace_back( part );
	return id;
}

EveSpaceObjectChild::PartTag EveModularObjectModifier::AddChild( const char* resPath, const Vector3& position, const Quaternion& rotation, const Vector3& scale )
{
	if( auto child = BeResMan->LoadObject<EveSpaceObjectChild>( resPath ) )
	{
		child->Setup( &scale, &rotation, &position, Tr2Lod::TR2_LOD_LOW );
		m_object->AddToEffectChildrenList( child );
		auto id = m_data->GetUnusedPartID();
		m_data->m_parts.emplace_back( EveChildPartData::PartData{ id, position, rotation, scale } );
		return id;
	}
	// TODO: return error
	return 0;
}

BlueStdResult EveModularObjectModifier::Remove( EveSpaceObjectChild::PartTag partId )
{
	auto found = std::find_if( m_data->m_parts.begin(), m_data->m_parts.end(), [partId]( const EveChildPartData::PartData& part ) {
		return part.partId == partId;
	} );
	if( found == m_data->m_parts.end() )
	{
		return BlueStdResultType::BLUE_STD_RESULT_KEY_ERROR;
	}

	for( size_t i = 0; i < m_object->GetEffectChildren().size(); )
	{
		auto child = m_object->GetEffectChildren()[i];
		if( child->GetPartTag() == partId )
		{
			m_object->RemoveFromEffectChildrenList( child );
			continue;
		}
		++i;
	}

	for ( auto& set : m_object->GetLocatorSets() )
	{
		auto& locators = *set->GetLocators();
		auto removed = std::remove_if( locators.begin(), locators.end(), [partId]( const auto& locator ) {
			return locator.partTag == partId;
		} );
		locators.Resize( std::distance( removed, locators.end() ) );
	}

	if( m_instancedMeshes )
	{
		m_instancedMeshes->RemoveInstancesByPartTag( partId );
	}
	m_data->m_parts.erase( found );
	return BlueStdResultType::BLUE_STD_RESULT_OK;
}

BlueStdResult EveModularObjectModifier::SetTransform( EveSpaceObjectChild::PartTag partId, const Vector3& position, const Quaternion& rotation, Vector3 scale )
{
	auto found = std::find_if( m_data->m_parts.begin(), m_data->m_parts.end(), [partId]( const EveChildPartData::PartData& part ) {
		return part.partId == partId;
	} );
	if( found == m_data->m_parts.end() )
	{
		return BlueStdResultType::BLUE_STD_RESULT_KEY_ERROR;
	}

	cmf::Transform oldTransform{ found->position, found->rotation, found->scale };
	cmf::Transform newTransform{ position, rotation, scale };
	auto invOldTransform = cmf::Inverse( oldTransform );

	for( auto& set : m_object->GetLocatorSets() )
	{
		auto& locators = *set->GetLocators();
		for( auto& locator : locators )
		{
			if( locator.partTag == partId )
			{
				locator.scale.x = scale.x / found->scale.x;
				locator.scale.y = scale.y / found->scale.y;
				locator.scale.z = scale.z / found->scale.z;
				locator.direction = invOldTransform.rotation * rotation;
				locator.position = cmf::TransformPoint( cmf::TransformPoint( locator.position, invOldTransform ), newTransform );
			}
		}
	}

	found->position = position;
	found->rotation = rotation;
	found->scale = scale;

	for( auto& child : m_object->GetEffectChildren() )
	{
		if( child->GetPartTag() == partId )
		{
			child->Setup( &scale, &rotation, &position, Tr2Lod::TR2_LOD_LOW );
		}
	}
	return BlueStdResultType::BLUE_STD_RESULT_OK;
}

BlueStdResult EveModularObjectModifier::GetPosition( EveSpaceObjectChild::PartTag partId, Vector3& position ) const
{
	auto found = std::find_if( m_data->m_parts.begin(), m_data->m_parts.end(), [partId]( const EveChildPartData::PartData& part ) {
		return part.partId == partId;
	} );
	if( found == m_data->m_parts.end() )
	{
		return BlueStdResultType::BLUE_STD_RESULT_KEY_ERROR;
	}
	position = found->position;
	return BlueStdResultType::BLUE_STD_RESULT_OK;
}

BlueStdResult EveModularObjectModifier::GetRotation( EveSpaceObjectChild::PartTag partId, Quaternion& rotation ) const
{
	auto found = std::find_if( m_data->m_parts.begin(), m_data->m_parts.end(), [partId]( const EveChildPartData::PartData& part ) {
		return part.partId == partId;
	} );
	if( found == m_data->m_parts.end() )
	{
		return BlueStdResultType::BLUE_STD_RESULT_KEY_ERROR;
	}
	rotation = found->rotation;
	return BlueStdResultType::BLUE_STD_RESULT_OK;
}

BlueStdResult EveModularObjectModifier::GetScale( EveSpaceObjectChild::PartTag partId, Vector3& scale ) const
{
	auto found = std::find_if( m_data->m_parts.begin(), m_data->m_parts.end(), [partId]( const EveChildPartData::PartData& part ) {
		return part.partId == partId;
	} );
	if( found == m_data->m_parts.end() )
	{
		return BlueStdResultType::BLUE_STD_RESULT_KEY_ERROR;
	}
	scale = found->scale;
	return BlueStdResultType::BLUE_STD_RESULT_OK;
}


std::pair<IEveSpaceObject2Ptr, EveModularObjectModifierPtr> CreateModularObject( EveSOF* sof, const char* factionName, const char* raceName )
{
	EveStation2Ptr object;
	object.CreateInstance();
	object->Initialize();

	EveChildPartDataPtr partData;
	partData.CreateInstance();
	partData->m_faction = factionName;
	partData->m_race = raceName;

	object->AddToEffectChildrenList( partData );

	EveModularObjectModifierPtr modifier;
	modifier.CreateInstance();
	modifier->Create( object, sof );
	return { IEveSpaceObject2Ptr( object ), modifier };
}

EveModularObjectModifierPtr ModifyModularObject( EveModularObjectModifier::SpaceObjectType* object, EveSOF* sof )
{
	EveModularObjectModifierPtr modifier;
	modifier.CreateInstance();
	modifier->Create( object, sof );
	return modifier;
}