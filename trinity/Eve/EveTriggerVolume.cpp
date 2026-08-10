// Copyright © 2026 CCP ehf.

#include "StdAfx.h"
#include "EveTriggerVolume.h"
#include "TriDevice.h"
#include "TriUtil.h"

#if BLUE_WITH_PYTHON
namespace
{
void InvokeTriggerCallback( void* context, bool entered )
{
	EveTriggerVolume* triggerVolume = reinterpret_cast<EveTriggerVolume*>( context );

	triggerVolume->InvokeCallback( entered );

	// A reference was added when the callback was queued - release it here.
	triggerVolume->GetRawRoot()->Unlock();
}

void TriggerEnterCallback( void* context )
{
	InvokeTriggerCallback( context, true );
}

void TriggerExitCallback( void* context )
{
	InvokeTriggerCallback( context, false );
}
}
#endif

EveTriggerVolume::EveTriggerVolume( IRoot* lockobj ) :
	PARENTLOCK( m_volumes ),
	PARENTLOCK( m_exclusionVolumes ),
	PARENTLOCK( m_externalParameters ),
	m_rotation( 0.0f, 0.0f, 0.0f, 1.0f ),
	m_translation( 0.0f, 0.0f, 0.0f ),
	m_boundingSphere( Vector3( 0.0, 0.0, 0.0 ), 0.0 ),
	m_worldTransform( IdentityMatrix() ),
	m_enterThreshold( 0.5f ),
	m_forceTriggered( false ),
	m_isInside( false ),
	m_currentIntensity( 0.0f ),
	m_display( true )
#if BLUE_WITH_PYTHON
	,
	m_callable( NULL )
#endif
{
}

EveTriggerVolume::~EveTriggerVolume()
{
#if BLUE_WITH_PYTHON
	Py_XDECREF( m_callable );
#endif
}

void EveTriggerVolume::RebuildBoundingSphere()
{
	CCP_STATS_ZONE( __FUNCTION__ );

	m_boundingSphere.center *= 0.0f;
	m_boundingSphere.radius *= 0.0f;

	for( auto volume = m_volumes.begin(); volume != m_volumes.end(); ++volume )
	{
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

const char* EveTriggerVolume::GetEffectiveName() const
{
	if( !m_name.empty() )
	{
		return m_name.c_str();
	}
	for( auto volume = m_volumes.begin(); volume != m_volumes.end(); ++volume )
	{
		const char* volumeName = ( *volume )->GetName();
		if( volumeName && volumeName[0] != '\0' )
		{
			return volumeName;
		}
	}
	return "";
}

#if BLUE_WITH_PYTHON
void EveTriggerVolume::SetCallback( PyObject* callable )
{
	Py_XDECREF( m_callable );

	if( callable == NULL || callable == Py_None )
	{
		m_callable = NULL;
	}
	else
	{
		m_callable = callable;
		Py_XINCREF( m_callable );
	}
}

void EveTriggerVolume::InvokeCallback( bool entered )
{
	if( !m_callable )
	{
		return;
	}

	PyObject* args = Py_BuildValue( "(sO)", GetEffectiveName(), entered ? Py_True : Py_False );
	if( !args )
	{
		return;
	}

	PyObject* result = PyObject_CallObject( m_callable, args );
	Py_DECREF( args );
	if( result )
	{
		Py_DECREF( result );
	}
	else
	{
		CCP_LOGWARN( "EveTriggerVolume: Callback raised an exception" );
	}
}
#endif

void EveTriggerVolume::QueueCallback( bool entered )
{
#if BLUE_WITH_PYTHON
	if( !m_callable )
	{
		return;
	}

	if( !PyCallable_Check( m_callable ) )
	{
		CCP_LOGWARN( "EveTriggerVolume: Callback is not a callable object" );
		return;
	}

	// Defer the actual Python call to a well defined point on the main thread.
	GetRawRoot()->Lock();
	gTriDev->AddPostUpdateCallback( entered ? TriggerEnterCallback : TriggerExitCallback, reinterpret_cast<void*>( this ) );
#endif
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
	if( m_forceTriggered )
	{
		m_currentIntensity = 1.0f;
		inside = true;
	}
	else if( m_trackedPosition && m_volumes.size() > 0 )
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
				m_currentIntensity = std::max( m_currentIntensity, volume->GetIntensity( positionInObjectSpace ) );
				if( m_currentIntensity == 1.0f )
				{
					// early exit
					break;
				}
			}

			if( m_currentIntensity != 0.0f )
			{
				// check if the tracked position is within an exclusion volume
				float negativeIntensity = 0.0f;
				for( const auto& volume : m_exclusionVolumes )
				{
					negativeIntensity = std::max( negativeIntensity, volume->GetIntensity( positionInObjectSpace ) );
					if( negativeIntensity == 1.0f )
					{
						// early exit
						break;
					}
				}
				m_currentIntensity = std::max( 0.0f, m_currentIntensity - negativeIntensity );
			}
		}

		inside = m_currentIntensity >= m_enterThreshold;
	}

	if( inside != m_isInside )
	{
		m_isInside = inside;
		QueueCallback( inside );
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
	return Normalize( m_rotation );
}

/////////////////////////////////////////////////////////////////////////////////////
// IInitialize
bool EveTriggerVolume::Initialize()
{
	UpdateWorldTransform();
	RebuildBoundingSphere();
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////
// ITr2DebugRenderable
void EveTriggerVolume::GetDebugOptions( Tr2DebugRendererOptions& options )
{
	options.insert( "Trigger Volumes" );
	options.insert( "Trigger Exclusion Volumes" );
	options.insert( "Trigger Bounding Sphere" );
}

void EveTriggerVolume::RenderDebugInfo( ITr2DebugRenderer2& renderer )
{
	if( renderer.HasOption( GetRawRoot(), "Trigger Volumes" ) )
	{
		// green when the tracked position is inside, white otherwise
		Color color = m_isInside ? 0xFF33FF33 : 0xFFFFFFFF;
		for( auto volume = m_volumes.begin(); volume != m_volumes.end(); ++volume )
		{
			( *volume )->RenderDebugInfo( renderer, m_worldTransform, color );
		}
	}

	if( renderer.HasOption( GetRawRoot(), "Trigger Exclusion Volumes" ) )
	{
		for( auto volume = m_exclusionVolumes.begin(); volume != m_exclusionVolumes.end(); ++volume )
		{
			( *volume )->RenderDebugInfo( renderer, m_worldTransform, 0xFFFF3333 );
		}
	}

	if( renderer.HasOption( GetRawRoot(), "Trigger Bounding Sphere" ) )
	{
		renderer.DrawSphere( this, TranslationMatrix( m_boundingSphere.center ) * m_worldTransform, m_boundingSphere.radius, 10, Tr2DebugRenderer::Wireframe, 0xff333333 );
	}
}
