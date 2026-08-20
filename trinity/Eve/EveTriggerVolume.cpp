// Copyright © 2026 CCP ehf.

#include "StdAfx.h"
#include "EveTriggerVolume.h"
#include "TriDevice.h"
#include "TriUtil.h"

EveTriggerVolume::EveTriggerVolume( IRoot* lockobj ) :
	PARENTLOCK( m_volumes ),
	m_rotation( 0.0f, 0.0f, 0.0f, 1.0f ),
	m_translation( 0.0f, 0.0f, 0.0f ),
	m_worldTransform( IdentityMatrix() ),
	m_enterThreshold( 0.5f ),
	m_isInside( false ),
	m_currentIntensity( 0.0f ),
	m_display( true )
{
}

EveTriggerVolume::~EveTriggerVolume()
{
}

void EveTriggerVolume::RebuildBoundingSphere()
{
	CCP_STATS_ZONE( __FUNCTION__ );

	// Reset to the uninitialized sentinel (radius -1); a zero-radius reset reads as
	// initialized, making the first merge below blend in a phantom sphere at the origin.
	// Not CcpMath::Sphere::IncludeSphere, which fails to grow when the included sphere
	// fully contains the current one.
	m_boundingSphere = CcpMath::Sphere();

	for( auto volume = m_volumes.begin(); volume != m_volumes.end(); ++volume )
	{
		if( !( *volume )->IsEnabled() )
		{
			continue;
		}

		auto volumeSphere = ( *volume )->GetBoundingSphere();

		if( !volumeSphere.IsInitialized() )
		{
			continue;
		}
		// if sphere is not initialized, just copy it
		// also if the sphere we are including in this sphere, then also copy it
		if( !m_boundingSphere.IsInitialized() || volumeSphere.IsSphereInside( m_boundingSphere ) )
		{
			m_boundingSphere = volumeSphere;
			continue;
		}
		// do not update if is inside
		if( m_boundingSphere.IsSphereInside( volumeSphere ) )
		{
			continue;
		}

		// extend sphere
		Vector3 delta = volumeSphere.center - m_boundingSphere.center;
		float deltaLen = Length( delta );

		m_boundingSphere.center += 0.5f * ( 1.f + ( volumeSphere.radius - m_boundingSphere.radius ) / deltaLen ) * delta;
		m_boundingSphere.radius = 0.5f * ( m_boundingSphere.radius + volumeSphere.radius + deltaLen );
	}
}

void EveTriggerVolume::UpdateWorldTransform()
{
	m_worldTransform = RotationMatrix( m_rotation ) * TranslationMatrix( m_translation );
}

/////////////////////////////////////////////////////////////////////////////////////
// IEveSpaceObject2
void EveTriggerVolume::UpdateSyncronous( const EveUpdateContext& updateContext )
{
	CCP_STATS_ZONE( __FUNCTION__ );

	UpdateWorldTransform();

	RebuildBoundingSphere();

	// The tracked position function may thunk into Python, so it must be evaluated on the
	// synchronous update path rather than in UpdateAsyncronous.
	UpdateTriggerState( updateContext );
}

void EveTriggerVolume::UpdateTriggerState( const EveUpdateContext& updateContext )
{
	m_currentIntensity = 0.0f;

	bool inside = false;
	if( m_trackedPosition && m_volumes.size() > 0 )
	{
		Vector3 trackedPosition;
		m_trackedPosition->Update( &trackedPosition, updateContext.GetTime() );

		Matrix inverseWorldTransform = Inverse( m_worldTransform );
		Vector3 positionInObjectSpace = Transform( trackedPosition, inverseWorldTransform ).GetXYZ();

		// check first if the tracked position is within the bounding sphere
		if( m_boundingSphere.IsPointInside( positionInObjectSpace ) )
		{
			// Now find the intensity within the volumes
			for( const auto& volume : m_volumes )
			{
				if( !volume->IsEnabled() )
				{
					continue;
				}
				m_currentIntensity = std::max( m_currentIntensity, volume->GetIntensity( positionInObjectSpace ) );
				if( m_currentIntensity == 1.0f )
				{
					// early exit
					break;
				}
			}
		}

		inside = m_currentIntensity >= m_enterThreshold;
	}

	if( inside != m_isInside )
	{
		m_isInside = inside;
	}
}

void EveTriggerVolume::UpdateAsyncronous( const EveUpdateContext& updateContext )
{
}

void EveTriggerVolume::UpdateVisibility( const EveUpdateContext& updateContext, const Matrix& parentTransform )
{
}

void EveTriggerVolume::GetRenderables( std::vector<ITr2Renderable*>& renderables, Tr2ImpostorManager* impostors )
{
}

bool EveTriggerVolume::GetBoundingSphere( Vector4& sphere, BoundingSphereQuery query ) const
{
	Vector3 worldCenter = Transform( m_boundingSphere.center, m_worldTransform ).GetXYZ();
	sphere = Vector4( worldCenter.x, worldCenter.y, worldCenter.z, std::max( m_boundingSphere.radius, 1.0f ) );
	return true;
}

void EveTriggerVolume::UpdateModelCenterWorldPosition( Vector3& position, Be::Time t )
{
	UpdateWorldTransform();
	GetModelCenterWorldPosition( position );
}

void EveTriggerVolume::GetModelCenterWorldPosition( Vector3& position ) const
{
	position = Transform( m_boundingSphere.center, m_worldTransform ).GetXYZ();
}

bool EveTriggerVolume::GetLocalBoundingBox( Vector3& min, Vector3& max )
{
	// Fall back to a unit box when no volumes are set up yet, so the object stays pickable in Graphite.
	float radius = std::max( m_boundingSphere.radius, 1.0f );
	min = m_boundingSphere.center - Vector3( radius, radius, radius );
	max = m_boundingSphere.center + Vector3( radius, radius, radius );
	return true;
}

void EveTriggerVolume::GetLocalToWorldTransform( Matrix& transform ) const
{
	transform = m_worldTransform;
}

/////////////////////////////////////////////////////////////////////////////////////
// IWorldPosition
Vector3 EveTriggerVolume::GetWorldPosition()
{
	return m_worldTransform.GetTranslation();
}

Quaternion EveTriggerVolume::GetWorldRotation()
{
	return Normalize( RotationQuaternion( m_worldTransform ) );
}

/////////////////////////////////////////////////////////////////////////////////////
// IInitialize
bool EveTriggerVolume::Initialize()
{
	UpdateWorldTransform();
	RebuildBoundingSphere();
	return true;
}
