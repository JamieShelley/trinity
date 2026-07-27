// Copyright © 2026 Fenris Creations ehf.

#include "StdAfx.h"
#include "EveChildTurret.h"
#include "Eve/Turret/EveTurretFiringFX.h"
#include "Tr2MeshBase.h"
#include "TriMath.h"
#include "TriObserverLocal.h"

// names of system bones like they are in the granny file
constexpr const char* s_systemBoneSkeletonNames[] = {
	"invalid", // SYSBONE_INVALID
	"Sys_Rotation_Arm", // SYSBONE_ROTATION
	"Sys_Rotation_Arm01", // SYSBONE_ROTATION1
	"Sys_Rotation_Arm02", // SYSBONE_ROTATION2
	"Sys_CounterRotation", // SYSBONE_COUNTER_ROTATION
	"Sys_Pitch_Barrel", // SYSBONE_PITCH
	"Sys_Pitch_Barrel1", // SYSBONE_PITCH1
	"Sys_Pitch_Barrel2", // SYSBONE_PITCH2
	"Sys_Height", // SYSBONE_SCALED_HEIGHT
	"Sys_Pitch_Arm01", // SYSBONE_SCALED_PITCH01
	"Sys_Pitch_Arm02", // SYSBONE_SCALED_PITCH02
	"Sys_Pitch_Arm03", // SYSBONE_SCALED_PITCH03
	"Sys_Pitch_Arm04", // SYSBONE_SCALED_PITCH04
	"Sys_Pitch_Arm05", // SYSBONE_SCALED_PITCH05
	"Sys_Pitch_Arm06", // SYSBONE_SCALED_PITCH06
};

// invalids
constexpr unsigned int INVALID_BONE_INDEX = 0xffffffff;
constexpr unsigned int INVALID_TURRET_INDEX = 0xffffffff;

// some very static timings, no need to confuse artists by exposing them
constexpr float TRACKING_FADE_TIME = 1.f;

EveChildTurret::EveChildTurret( IRoot* lockobj ) :
	EveChildMesh( lockobj )
{
	for( unsigned int i = 0; i < SYSBONE_MAX; ++i )
	{
		m_systemBoneID[i] = INVALID_BONE_INDEX;
	}

	m_target.CreateInstance();

	PrepareResources();
}

EveChildTurret::~EveChildTurret()
{
	if( m_firingEffect )
	{
		m_firingEffect->CleanUp();
	}

	m_cachedGeometryRes = nullptr;
	m_skeleton = nullptr;
	m_skeletonBoneIndices.clear();
	m_boneBounds.clear();

	ReleaseResources( TRISTORAGE_ALL );
}
bool EveChildTurret::Initialize()
{
	// pass down some user-defined data into sub-modules we don't save out
	// TODO: keep or discard
	// m_target->SetBehaviour( m_laserMissBehaviour, m_projectileMissBehaviour, m_impactSize, m_impactBehaviour );

	return EveChildMesh::Initialize();
}
bool EveChildTurret::OnModified( Be::Var* value )
{
	/*
	if( IsMatch( value, m_laserMissBehaviour ) || IsMatch( value, m_projectileMissBehaviour ) || IsMatch( value, m_impactSize ) || IsMatch( value, m_impactBehaviour ) )
	{
		// TODO: keep or discard
		m_target->SetBehaviour( m_laserMissBehaviour, m_projectileMissBehaviour, m_impactSize, m_impactBehaviour );
	}
	*/
	return EveChildMesh::OnModified( value );
}
void EveChildTurret::RegisterComponents()
{
	EveChildMesh::RegisterComponents();
	const auto registry = GetComponentRegistry();
	if( registry && m_display )
	{
		if( EveEntityPtr entity = BlueCastPtr( m_firingEffect ) )
		{
			entity->Register( registry );
		}
	}
}
void EveChildTurret::UnRegisterComponents()
{
	EveChildMesh::UnRegisterComponents();
	const auto registry = this->GetComponentRegistry();
	if( registry )
	{
		if( EveEntityPtr entity = BlueCastPtr( m_firingEffect ) )
		{
			entity->UnRegister( registry );
		}
	}
}
void EveChildTurret::GetDebugOptions( Tr2DebugRendererOptions& options )
{
	EveChildMesh::GetDebugOptions( options );

	if( m_firingEffect )
	{
		m_firingEffect->GetDebugOptions( options );
	}

	if( m_turretMovementObserver )
	{
		m_turretMovementObserver->GetDebugOptions( options );
	}
}
void EveChildTurret::RenderDebugInfo( ITr2DebugRenderer2& renderer )
{
	EveChildMesh::RenderDebugInfo( renderer );

	if( m_firingEffect )
	{
		m_firingEffect->RenderDebugInfo( renderer );
	}

	if( m_turretMovementObserver && m_playMovementSound )
	{
		m_turretMovementObserver->RenderDebugInfo( renderer );
	}
}

void EveChildTurret::UpdateSyncronous( const EveUpdateContext& updateContext, const EveChildUpdateParams& params )
{
	float deltaT = updateContext.GetDeltaT();

	/*
	// TODO: LODs needed?
	// LODing
	if( UpdateLOD( updateContext ) )
	{
		// LOD change, so just call ::InitializeGeometryResource(), takes care of everything
		InitializeGeometryResource();

		// LOD change: toggle source dest effect of the attached firingFX
		if( m_firingEffect )
		{
			switch( m_lodLevel )
			{
			case LOD_DISABLED:
			case LOD_HIGHEST:
				m_firingEffect->SetDisplaySourceObject( true );
				break;
			default:
				m_firingEffect->SetDisplaySourceObject( false );
				break;
			}
		}
	}
	*/

	UpdateCachedGeometryData();

	if( m_sequencer )
	{
		m_sequencer->RemoveFinishedAnimations( Tr2Renderer::GetAnimationTime() );
	}

	// setup and update attached firing effect
	if( m_firingEffect )
	{
		// if the attached firing effect is looping, then we must recheck if active turret is still the best,
		if( m_firingEffect->IsLooping() )
		{
			if( m_state == STATE_FIRING )
			{
				// don't do it every frame, cause this will result in popping
				m_recheckTimeLeft -= deltaT;
				if( m_recheckTimeLeft < 0.f )
				{
					Vector3 source = m_worldTransform.GetTranslation();
					Vector3 position = source;
					if( int closestLocator = m_target->FindClosestLocator( &source, &position ) )
					{
						if( closestLocator != m_target->GetLocator() )
						{
							// Set up the firing states correctly
							SetupFiringState();
						}
					}
					// recheck every 2 seconds
					m_recheckTimeLeft = 2.f;
				}
			}
		}
		m_firingEffect->UpdateSynchronous( updateContext );
	}

	// update the target locator position
	// TODO: probably wrong
	Vector3 position = m_parentData.transform.GetTranslation();
	if( m_firingEffect )
	{
		m_firingEffect->GetStartPosition( position );
	}

	m_target->Update( deltaT, &position );

	if( m_mesh && m_turretMovementObserver != nullptr )
	{
		// TODO: new turret movementObserver prob needed
		// m_turretMovementObserver->Update( m_singleTurrets[0].worldMatrix );
	}
	EveChildMesh::UpdateSyncronous( updateContext, params );
}

void EveChildTurret::UpdateAsyncronous( const EveUpdateContext& updateContext, const EveChildUpdateParams& params )
{

	// TODO: prob some freakyness in here
	float deltaT = updateContext.GetDeltaT();
	// handle fading of turret tracking
	if( m_trackingInfluenceDelta != 0.f )
	{
		m_trackingInfluence += m_trackingInfluenceDelta * deltaT;
		if( m_trackingInfluence > m_maxTrackingTime )
		{
			m_trackingInfluence = m_maxTrackingTime;
			m_trackingInfluenceDelta = 0.f;
		}
		else if( m_trackingInfluence < 0.f )
		{
			m_trackingInfluence = 0.f;
			m_trackingInfluenceDelta = 0.f;
		}
	}

	if( m_delayToFadeOutTracking > 0.f )
	{
		m_delayToFadeOutTracking -= deltaT;
		if( m_delayToFadeOutTracking <= 0.f )
		{
			m_delayToFadeOutTracking = 0.f;
			m_trackingInfluenceDelta = -1.f / TRACKING_FADE_TIME;
		}
	}

	if( m_delayToFadeInTracking > 0.f )
	{
		m_delayToFadeInTracking -= deltaT;
		if( m_delayToFadeInTracking <= 0.f )
		{
			m_delayToFadeInTracking = 0.f;
			m_trackingInfluenceDelta = 1.f / TRACKING_FADE_TIME;
		}
	}

	// TODO: does this have to happen after the timing stuff over there ^
	// Should handle all the mesh data and transforms
	EveChildMesh::UpdateAsyncronous( updateContext, params );

	// setup and update attached firing effect
	if( m_firingEffect )
	{
		// TODO: is this a valid replacement of if( m_activeTurret != INVALID_TURRET_INDEX )
		if( m_mesh )
		{
			// update all muzzle points in the firing effect
			for( unsigned int i = 0; i < m_firingEffect->GetPerMuzzleEffectCount(); ++i )
			{
				// get world transform of this muzzle bone
				Matrix matrix = GetFiringBoneWorldTransform( i );
				// and set it to the muzzle
				m_firingEffect->SetMuzzleTransform( i, &matrix );
			}
			m_firingEffectMuzzlePosSet = true;
		}

		m_firingEffect->SetEndPosition( m_target->GetTargetPosition() );

		// time update (return value tells us if effect is ready to fire!)
		if( m_firingEffect->UpdateAsynchronous( updateContext ) )
		{
			// if we haven't initialised muzzle positions, do it now
			// this can happen, and if we don't do this all effects originate from
			// the player ship until turret geometry is loaded and muzzle positions
			// properly set
			if( !m_firingEffectMuzzlePosSet )
			{
				for( unsigned int i = 0; i < m_firingEffect->GetPerMuzzleEffectCount(); ++i )
				{
					// use something relatively sensible, even absent geometry
					m_firingEffect->SetMuzzleTransform( i, &m_parentData.transform );
				}

				m_firingEffectMuzzlePosSet = true;
			}
			m_firingEffect->SetDisplayDestObject( m_target->ShowDestObject() );
		}
	}
}
void EveChildTurret::UpdateCachedGeometryData()
{
	auto* geometryRes = GetGeometryRes();
	if( geometryRes == m_cachedGeometryRes )
	{
		return;
	}
	ReleaseCachedGeometryData();
	if( geometryRes && geometryRes->IsGood() )
	{
		BuildCachedGeometryData( *geometryRes );
		m_cachedGeometryRes = geometryRes;
	}
}
void EveChildTurret::BuildCachedGeometryData( TriGeometryRes& geometryRes )
{
	// finished loading the turret geometry resource, so grab vertex decl and bounding sphere
	if( geometryRes.GetMeshCount() )
	{
		if( geometryRes.GetMeshData( 0 ) )
		{
			// get a bounding box for visibility detection, if this is not already set in the redfile
			// TODO: might not be needed
			if( m_worldBoundingSphere.radius == 0.f )
			{
				geometryRes.RecalculateBoundingSphere();
				Vector4 boundingSphere;
				geometryRes.GetBoundingSphere( 0, boundingSphere );
				m_worldBoundingSphere = CcpMath::Sphere( boundingSphere );
			}
		}
	}

	if( geometryRes.GetSkeletonCount() )
	{
		if( TriGeometryResSkeletonData* skeletonData = geometryRes.GetSkeletonData( 0 ) )
		{
			for( int i = 0; i < SYSBONE_MAX; ++i )
			{
				// in case we don't find system bone, ::FindJoint() returns 0xffffffff
				m_systemBoneID[i] = skeletonData->FindJoint( s_systemBoneSkeletonNames[i] );
			}

			InitializeFiringEffect();
		}
	}

	InitializeAnimation();

	// TODO: forceXAnimation based on m_state?
	ForceIdleAnimation();
}

void EveChildTurret::ReleaseCachedGeometryData()
{
	// TODO: for now duplicates destructor
	m_cachedGeometryRes = nullptr;
	m_skeleton = nullptr;
	m_skeletonBoneIndices.clear();
	m_boneBounds.clear();
}

void EveChildTurret::EnterStateDeactive()
{
	switch( m_state )
	{
	case STATE_DEACTIVE:
		// do nothing if we are already in this state
		break;
	case STATE_IDLE:
	case STATE_RELOADING:
		// no fadeout of tracking, just play deactive anim and then the deactive loop
		m_trackingInfluence = 0.f;
		PlayAnimation( "Pack", "Inactive" );
		m_delayToFadeOutTracking = 0.f;
		break;
	case STATE_FIRING:
		// stop shooting
		if( m_firingEffect )
		{
			m_firingEffect->StopFiring();
		}
		// DON'T break, just continue with stopping things:
	case STATE_TARGETING:
		// fadeout the tracking, play deactive anim and then the deactive loop
		m_delayToFadeOutTracking = 0.0001f;
		m_target->StopFireAtLocator();

		PlayAnimation( "Pack", "Inactive", TRACKING_FADE_TIME );
		break;

	default:
		break;
	}
	m_state = STATE_DEACTIVE;
}

void EveChildTurret::EnterStateIdle()
{
	// TODO: might want to remove this state
	// if( !m_isOnline )
	// {
	// 	return;
	// }

	switch( m_state )
	{
	case STATE_INVALID:
	case STATE_RELOADING:
		// just play active loop
		PlayAnimation( "", "Active" );
		break;
	case STATE_DEACTIVE:
		// start unpack animation, disable tracking and then into active loop
		PlayAnimation( "Deploy", "Active" );
		m_trackingInfluence = 0.f;
		break;
	case STATE_IDLE:
		// do nothing here
		break;
	case STATE_TARGETING:
	case STATE_FIRING:
		// stop shooting, fadeout tracking, then into active loop
		m_delayToFadeOutTracking = 0.0001f;
		m_target->StopFireAtLocator();
		if( m_firingEffect )
		{
			m_firingEffect->StopFiring();
		}
		PlayAnimation( "", "Active", TRACKING_FADE_TIME );

		if( m_playMovementSound && !m_targetingToIdleMovementAudioEvent.empty() )
		{
			SendEventToAudEmitter( m_turretMovementObserver, m_targetingToIdleMovementAudioEvent );
		}
		break;
	}
	m_state = STATE_IDLE;
}

void EveChildTurret::EnterStateTargeting()
{
	float animLength = 0.f;
	// TODO: might want to remove this state
	// if( !m_isOnline )
	// {
	// 	return;
	// }

	// what state are we in?
	switch( m_state )
	{
	case STATE_DEACTIVE:
		// play deploy anim, then active loop and fade in tracking
		animLength = PlayAnimation( "Deploy", "Active", TRACKING_FADE_TIME );
		// fade in tracking
		m_delayToFadeInTracking = animLength + 0.0001f;
		break;
	case STATE_IDLE:
	case STATE_RELOADING:
		// fadein tracking, play active loop
		m_delayToFadeInTracking = 0.0001f;
		PlayAnimation( "", "Active", TRACKING_FADE_TIME );
		break;
	case STATE_TARGETING:
		break;
	case STATE_FIRING:
		// stop shooting, then into active loop
		m_target->StopFireAtLocator();
		if( m_firingEffect )
		{
			m_firingEffect->StopFiring();
		}
		PlayAnimation( "", "Active", 0.f );
		break;

	default:
		break;
	}
	m_state = STATE_TARGETING;
}

void EveChildTurret::EnterStateFiring()
{
	if( !SetupFiringState() )
	{
		return;
	}

	// only if we are in firing mode, call ::StopFiring() on the effect right before
	// we call ::PrepareFiring(), it'll clean things up in the effect
	if( m_firingEffect && m_state == STATE_FIRING )
	{
		if( m_firingEffect->IsLooping() )
		{
			// We don't want to start and stop the curves when the turret is looping and firing
			m_firingEffect->PrepareFiringEffectMoveObjects();
			return;
		}
		m_firingEffect->StopFiring();
	}

	// We're starting a firing sequence, we need to set up our firing effect time-delays
	if( m_firingEffect )
	{
		// TODO: random firing delay yay or nay
		//if( m_maxCyclingFirePos > 1 )
		//{
		//	m_firingEffect->PrepareFiring( m_randomFiringDelay, m_currentCyclingFiresPos, m_cyclingFireGroupCount );
		//}
		//else
		//{
		float randomFiringDelay = 0.f;
		m_firingEffect->PrepareFiring( randomFiringDelay );
		//}

		if( m_target != nullptr )
		{
			m_firingEffect->SetImpactConfiguration( m_target->GetImpactConfiguration() );
		}
	}

	// finally, we can set state
	m_state = STATE_FIRING;
}

bool EveChildTurret::SetupFiringState()
{
	if( m_state == STATE_DEACTIVE )
	{
		// this state change is forbidden!
		CCP_LOGERR( "EveChildTurret %s wants to fire but is in deactive state.", m_name.c_str() );
		return false;
	}
	int closestLocator = -1;
	{
		Vector3 source = m_worldTransform.GetTranslation();
		Vector3 position = source;
		closestLocator = m_target->FindClosestLocator( &source, &position );
	}

	// TODO: remove or keep?
	// if this turret is set to cycle through the muzzles for firing, do it here
	/*
	if( m_maxCyclingFirePos > 1 )
	{
		m_currentCyclingFiresPos += m_cyclingFireGroupCount;
		if( m_currentCyclingFiresPos >= m_maxCyclingFirePos * m_cyclingFireGroupCount )
		{
			m_currentCyclingFiresPos = 0;
		}
	}
	*/

	// timing: apply a randomized fire delay
	// TODO: remove?
	// CalcRandomDelay();
	float randomFiringDelay = 0.f; // TODO: temp value

	// timing: is the length of the firing effect known?
	float effectTotalTime = m_firingEffect ? m_firingEffect->GetFiringDuration() : 0.f;
	float effectPeakTime = m_firingEffect ? m_firingEffect->GetFiringPeakTime() : 0.f;

	Vector3 source = m_parentData.transform.GetTranslation();

	// what state are we in?
	switch( m_state )
	{
	case STATE_IDLE:
	case STATE_RELOADING:
		// and delay the effect until we are facing target
		randomFiringDelay += m_maxTrackingTime;
		// fadein tracking, play fire anim (only one the firing turret!) and then the active anim
		m_delayToFadeInTracking = 0.0001f;

		PlayAnimation( GetFireAnimationName(), "Active", randomFiringDelay );
		// assign locator and turret
		m_target->StartFireAtLocator( closestLocator, randomFiringDelay + effectPeakTime, effectTotalTime - effectPeakTime, &source );
		break;
	case STATE_FIRING:
	case STATE_TARGETING:
		PlayAnimation( GetFireAnimationName(), "Active", randomFiringDelay );
		m_target->StartFireAtLocator( closestLocator, randomFiringDelay + effectPeakTime, effectTotalTime - effectPeakTime, &source );
		break;

	default:
		break;
	}

	return true;
}

void EveChildTurret::EnterStateReloading()
{
	// what state are we in?
	switch( m_state )
	{
	case STATE_DEACTIVE:
		// ignore this state change: when the turret is inactive, no reload state can be shown!
		break;
	case STATE_INVALID:
	case STATE_IDLE:
	case STATE_RELOADING:
		// just play reloading anim and then loop
		PlayAnimation( "Reload", "Active", 0.f );
		break;
	case STATE_TARGETING:
	case STATE_FIRING:
		// stop shooting, fadeout tracking, then into active loop
		m_delayToFadeOutTracking = 0.0001f;
		m_target->StopFireAtLocator();
		if( m_firingEffect )
		{
			m_firingEffect->StopFiring();
		}

		PlayAnimation( "Reload", "Active", TRACKING_FADE_TIME );
		break;

	default:
		break;
	}
	m_state = STATE_RELOADING;
}

void EveChildTurret::ForceStateDeactive()
{
	// turn it all off
	m_trackingInfluence = 0.f;
	m_delayToFadeOutTracking = 0.f;
	m_target->StopFireAtLocator();
	if( m_firingEffect )
	{
		m_firingEffect->StopFiring();
	}
	// finally, we can set state
	m_state = STATE_DEACTIVE;

	// now force-play the deactive anim for this state
	ForceIdleAnimation();
}

void EveChildTurret::ForceIdleAnimation()
{
	std::string idleAnimName = "";
	// what state?
	switch( m_state )
	{
	case STATE_DEACTIVE:
		idleAnimName = "Inactive";
		break;
	case STATE_IDLE:
	case STATE_TARGETING:
	case STATE_FIRING:
		idleAnimName = "Active";
		break;

	default:
		break;
	}

	// set it to all turrets in this set
	if( idleAnimName.length() > 0 )
	{
		PlayAnimation( "", idleAnimName, 0.f );
	}
}

void EveChildTurret::ForceStateTargeting()
{
	m_trackingInfluence = m_maxTrackingTime;
	m_trackingInfluenceDelta = 0.f;

	m_state = STATE_TARGETING;

	// now force-play the deactive anim for this state
	PlayAnimation( "", "Active", 0.f );
}

Matrix EveChildTurret::GetFiringBoneWorldTransform( unsigned int muzzle ) const
{
	if( !m_mesh )
	{
		return m_parentData.transform;
	}

	Matrix matrix = m_worldTransform;

	// get the boneID for that muzzle from firing effect
	if( !m_firingEffect )
	{
		return matrix;
	}
	unsigned int boneID = m_firingEffect->GetPerMuzzleBoneID( muzzle );
	return GetTurretBoneTransform( boneID );
}

void EveChildTurret::InitializeFiringEffect()
{
	if( !m_firingEffect )
	{
		return;
	}
	m_firingEffect->RegisterWithQuadRenderer( *Tr2QuadRenderer::Instance() );

	auto geometryResource = GetGeometryRes();
	if( geometryResource && geometryResource->GetSkeletonCount() )
	{
		if( TriGeometryResSkeletonData* skeletonData = geometryResource->GetSkeletonData( 0 ) )
		{
			const auto muzzleCount = m_firingEffect->GetPerMuzzleEffectCount();
			if( muzzleCount > EveTurretFiringFX::MUZZLECOUNT_MAX )
			{
				CCP_LOGERR( "Upper limit of firing bones is %d, this turret has %d", EveTurretFiringFX::MUZZLECOUNT_MAX, muzzleCount );
			}

			const unsigned int boneCount = std::min( muzzleCount, static_cast<unsigned int>( EveTurretFiringFX::MUZZLECOUNT_MAX ) );
			// firing bones should always be on the format Pos_FireXX where XX can range form 01 to 99
			for( unsigned int i = 0; i < boneCount; ++i )
			{
				char boneName[32];
				int boneNameIndex = i + 1;
				snprintf( boneName, sizeof boneName, "%s%02u", m_firingEffect->GetFiringBoneName(), boneNameIndex );

				// in case we don't find positional bone, ::FindJoint() returns 0xffffffff
				m_firingEffect->SetMuzzleBoneID( i, skeletonData->FindJoint( boneName ) );
			}
		}
	}
}

void EveChildTurret::InitializeAnimation()
{
	EveChildMesh::InitializeAnimation();
	if( const auto geometryResource = GetGeometryRes() )
	{
		// get a model, a meshbinding and animation stuff from the resource
		const cmf::Data* cmfData = geometryResource->GetCMFData();
		if( cmfData && cmfData->skeletons.size() )
		{
			const auto mesh = std::find_if( cmfData->meshes.begin(), cmfData->meshes.end(), []( const cmf::Mesh& m ) {
				return m.skeleton == 0;
			} );

			if( mesh != cmfData->meshes.end() && mesh->boneBindings.size() )
			{
				if( m_skeletonBoneIndices.empty() )
				{
					m_skeleton = &cmfData->skeletons[0];


					if( !m_sequencer )
					{
						m_sequencer = std::make_unique<cmf::AnimationSequencer>( *m_skeleton );
						cmf::RestPose( m_pose, *m_skeleton );
					}

					m_skeletonBoneIndices = Tr2GrannyAnimationUtils::CreateMapping( *m_skeleton, mesh->boneBindings, static_cast<uint32_t>( mesh->boneBindings.size() ) );
				}
			}
		}
	}
}

void EveChildTurret::ModifySystemBoneTransform( SystemBones bone, const Vector3* target, const Matrix* localTransform, Vector3& position, Quaternion& rotation ) const
{
	switch( bone )
	{
	case SYSBONE_INVALID:
		break;
	case SYSBONE_ROTATION:
	case SYSBONE_ROTATION01:
	case SYSBONE_ROTATION02: {
		// rotation of turret 360 degrees, alpha is between -pi and pi
		float alpha = atan2( target->x, target->z );
		// never forget do apply influence!
		alpha *= m_trackingInfluence;
		// 1st: make quaternion
		Quaternion quat = RotationQuaternion( alpha, 0.f, 0.f );
		// 2nd: apply this quat after the original one
		quat = rotation * quat;
		// TODO: cmf_transform ?
		// 3rd: make granny_transform from quat
		rotation = quat;
	}
	break;
	case SYSBONE_COUNTER_ROTATION: {
		// inverse(!!) rotation of turret 360 degress, alpha is between -pi and pi
		float alpha = -atan2( target->x, target->z );
		// never forget do apply influence!
		alpha *= m_trackingInfluence;
		// 1st: make quaternion
		Quaternion quat = RotationQuaternion( alpha, 0.f, 0.f );
		// 2nd: apply this quat after the original one
		quat = rotation * quat;
		// 3rd: make granny_transform from quat
		rotation = quat;
	}
	break;
	case SYSBONE_PITCH:
	case SYSBONE_PITCH1:
	case SYSBONE_PITCH2: {
		CalcTransformForPitchBone( target, XMConvertToRadians( m_sysBonePitchMin ), XMConvertToRadians( m_sysBonePitchMax ), bone, localTransform, rotation );
	}
	break;
	case SYSBONE_SCALED_HEIGHT: {
		// pitch of barrel 90 degrees
		Vector3 directionNormal = Normalize( *target );
		float height = TriClamp( directionNormal.y, 0.f, 1.f );
		// never forget do apply influence!
		height *= m_trackingInfluence;
		// it's a pos extension with a scale
		Vector3 pos = Vector3( 0.f, height * m_sysBoneHeight, 0.f ) + position;
		position = pos;
	}
	break;
	case SYSBONE_SCALED_PITCH01:
	case SYSBONE_SCALED_PITCH02:
	case SYSBONE_SCALED_PITCH03:
	case SYSBONE_SCALED_PITCH04:
	case SYSBONE_SCALED_PITCH05:
	case SYSBONE_SCALED_PITCH06: {
		CalcTransformForPitchBone( target, 0.f, XMConvertToRadians( m_sysBonePitchMax ), bone, nullptr, rotation );
	}
	break;
	default:
		break;
	}
}

void EveChildTurret::CalcTransformForPitchBone( const Vector3* target, float minPitch, float maxPitch, unsigned int boneIndex, const Matrix* localTransform, Quaternion& rotation ) const
{
	float pitchOffset = GetBonePitchOffset( boneIndex );
	float pitchFactor = GetBonePitchFactor( boneIndex );
	// pitch of barrel 90 degrees
	Vector3 bone_position( 0.f, 0.f, 0.f );

	if( localTransform )
	{
		bone_position = localTransform->GetTranslation();
	}

	Vector3 relTarget = *target - bone_position;
	Vector3 dirNrm = Normalize( relTarget );
	float radians = asinf( dirNrm.y );

	if( localTransform )
	{
		Vector3 bone_direction = Normalize( bone_position );
		float d = Dot( bone_direction, *target );
		if( d < Length( bone_position ) )
		{
			// Assuming up is enough for now to avoid cross products
			radians = TriFloatSign( relTarget.y ) * XM_PI - radians;
		}
	}

	float alpha = TriClamp( radians, minPitch, maxPitch );
	// modify!
	alpha = pitchFactor * alpha + XMConvertToRadians( pitchOffset );
	// never forget do apply influence!
	alpha *= m_trackingInfluence;
	// 1st: make quaternion
	Quaternion quat = RotationQuaternion( 0.f, -alpha, 0.f );
	// 2nd: apply this quat after the original one
	quat = rotation * quat;
	// 2nd: make granny_transform from quat
	rotation = quat;
}

float EveChildTurret::GetBonePitchFactor( unsigned int boneIndex ) const
{
	switch( boneIndex )
	{
	case SYSBONE_PITCH:
	case SYSBONE_PITCH1:
	case SYSBONE_PITCH2:
		return m_sysBonePitchFactor;
	case SYSBONE_SCALED_PITCH01:
		return m_sysBonePitch01Factor;
	case SYSBONE_SCALED_PITCH02:
		return m_sysBonePitch02Factor;
	case SYSBONE_SCALED_PITCH03:
		return m_sysBonePitch03Factor;
	default:
		return 1.0f;
	}
}

float EveChildTurret::GetBonePitchOffset( unsigned int boneIndex ) const
{
	switch( boneIndex )
	{
	case SYSBONE_PITCH:
	case SYSBONE_PITCH1:
	case SYSBONE_PITCH2:
		return m_sysBonePitchOffset;
	case SYSBONE_SCALED_PITCH01:
		return m_sysBonePitch01Offset;
	case SYSBONE_SCALED_PITCH02:
		return m_sysBonePitch02Offset;
	case SYSBONE_SCALED_PITCH03:
		return m_sysBonePitch03Offset;
	default:
		return 0.0f;
	}
}
Matrix EveChildTurret::GetTurretBoneTransform( uint32_t boneID ) const
{
	Matrix matrix = m_worldTransform;


	// TODO: should support lowLodTransform? prob yes
	// return lowLodTransform * matrix;
	if( m_animationUpdater )
	{
		const auto& worldTransforms = m_animationUpdater->GetWorldTransforms();
		if( boneID < worldTransforms.size() )
		{
			return worldTransforms[boneID] * matrix;
		}
	}
	// TODO: port rest of function?
	return matrix;
}

TriGeometryRes* EveChildTurret::GetGeometryRes() const
{
	return m_mesh ? m_mesh->GetGeometryResource() : nullptr;
}

// TODO: heavily refactor once animation ownership is decided (m_animationUpdater vs m_sequencer).
// Known defects for the rewrite:
// - a missing anim name aborts the whole request instead of playing what was found
// - the idle anim starts at `delay` instead of after the one-shot finishes
//   (original sequenced via player SetStartTime/SetStopTime, no equivalent wired here)
// - StopAnimation is a stub, so the "stop all animation" call below does nothing
// - duplicated find_if lookups
float EveChildTurret::PlayAnimation( const std::string& animName, const std::string& animNameIdle, float delay )
{
	auto geometryRes = GetGeometryRes();
	if( !m_animationUpdater || !geometryRes )
	{
		return 0.f;
	}
	float animLength = 0.f;

	if( auto cmfData = geometryRes->GetCMFData() )
	{
		// there can be more animations in one res, so find right one
		size_t animIx = cmfData->animations.size();
		if( !animName.empty() )
		{
			auto animation = std::find_if( cmfData->animations.begin(), cmfData->animations.end(), [&animName]( const cmf::Animation& anim ) {
				return cmf::ToStdStringView( anim.name ) == animName;
			} );
			if( animation == cmfData->animations.end() )
			{
				return 0.f;
			}
			animIx = std::distance( cmfData->animations.begin(), animation );
		}

		size_t idleIx = cmfData->animations.size();
		if( !animNameIdle.empty() )
		{
			auto animation = std::find_if( cmfData->animations.begin(), cmfData->animations.end(), [&animNameIdle]( const cmf::Animation& anim ) {
				return cmf::ToStdStringView( anim.name ) == animNameIdle;
			} );
			if( animation == cmfData->animations.end() )
			{
				return 0.f;
			}
			idleIx = std::distance( cmfData->animations.begin(), animation );
		}

		// stop all animation
		StopAnimation( delay );

		// granny, play first anim once, if provided & found
		if( animIx != cmfData->animations.size() )
		{
			auto& animation = cmfData->animations[animIx];
			animLength = animation.duration;

			// ( const char* animName, bool replace, int loopCount, float delay, float speed, bool clearWhenDone )
			std::string animationName = cmf::ToStdString( animation.name );
			bool replace = false; // TODO: correct?
			int loopCount = 1;
			float speed = 1.f;
			m_animationUpdater->PlayAnimation( animationName.c_str(), replace, loopCount, delay, speed );
			// TODO: set start/stop time seems different from ^
			// player->SetStartTime( delay + Tr2Renderer::GetAnimationTime() );
			// player->SetStopTime( animLength + delay + Tr2Renderer::GetAnimationTime() );
		}

		// then play idle anim on loop (after delay), if provided & found
		if( idleIx != cmfData->animations.size() )
		{
			auto& animation = cmfData->animations[idleIx];
			std::string animationName = cmf::ToStdString( animation.name );
			bool replace = false; // TODO: correct?
			int loopCount = 0;
			float speed = 1.f;

			m_animationUpdater->PlayAnimation( animationName.c_str(), replace, loopCount, delay, speed );
			// TODO: set start time seems different from ^
			// player->SetStartTime( animLength + delay + Tr2Renderer::GetAnimationTime() );
		}
	}

	return animLength;
}

void EveChildTurret::StopAnimation( float delay )
{
	// TODO: prob not needed and can just use animation class function instead of this probably useless middleman
	// if we don't have a geometry, animation is useless and probably unwanted
	if( !GetGeometryRes() )
	{
		return;
	}

	// stop
	if( m_animationUpdater )
	{
		/*
		// TODO: find correct functions
		m_animationUpdater->EnumerateAnimations( [&]( const std::shared_ptr<cmf::AnimationPlayer>& player ) {
			player->SetStopTime( delay + Tr2Renderer::GetAnimationTime() );
		} );

		m_animationUpdater->RemoveFinishedAnimations( Tr2Renderer::GetAnimationTime() );
		*/
	}
}

std::string EveChildTurret::GetFireAnimationName() const
{
	/*
	// TODO: idk if this is useful skipping for now
	// if m_currentCyclingFiresPos is 0, it's just "Fire"
	std::string res = "Fire";
	if( m_currentCyclingFiresPos > 0 )
	{
		res.push_back( '0' );
		res.push_back( '0' + m_currentCyclingFiresPos / m_cyclingFireGroupCount );
	}

	return res;
	*/
	return "";
}

EveTurretFiringFX* EveChildTurret::GetFiringEffect()
{
	return m_firingEffect;
}

void EveChildTurret::SetFiringEffect( EveTurretFiringFX* firingEffect )
{
	auto registry = GetComponentRegistry();
	if( EveEntityPtr entity = BlueCastPtr( m_firingEffect ) )
	{
		entity->UnRegister( registry );
	}
	m_firingEffect = firingEffect;
	if( EveEntityPtr entity = BlueCastPtr( m_firingEffect ) )
	{
		entity->Register( registry );
	}
	InitializeFiringEffect();
}

void EveChildTurret::SetTargetObject( IRoot* target )
{
	if( !target )
	{
		return;
	}
	ITriTargetablePtr oldTargetPtr = m_target->GetTargetable();

	// attach to target
	m_target->SetTargetable( target );

	if( m_playMovementSound && !m_idleToTargetingMovementAudioEvent.empty() )
	{
		// Always trigger movement sounds if coming from IDLE state, otherwise trigger it only if you're targeting a new object.
		if( m_state == STATE_IDLE || !oldTargetPtr.IsEqualObject( m_target->GetTargetable() ) )
		{
			SendEventToAudEmitter( m_turretMovementObserver, m_idleToTargetingMovementAudioEvent );
		}
	}

	// update the firing effect we have one
	SetTargetScale();
}

ITriTargetablePtr EveChildTurret::GetTargetObject()
{
	return m_target->GetTargetable();
}

void EveChildTurret::SetTargetScale()
{
	if( m_firingEffect )
	{
		float radius = m_target->GetRadius();
		m_firingEffect->SetScaleByRadius( radius );
	}
}
