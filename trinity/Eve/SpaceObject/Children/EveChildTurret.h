// Copyright © 2026 Fenris Creations ehf.

#pragma once
#ifndef EveChildTurret_H
#define EveChildTurret_H

#include "EveChildMesh.h"
#include "Eve/Turret/EveTurretTarget.h"

BLUE_DECLARE( EveTurretFiringFX );
BLUE_DECLARE( EveTurretTarget );

BLUE_CLASS( EveChildTurret ) :
	public EveChildMesh
{
public:
	EXPOSE_TO_BLUE();

	EveChildTurret( IRoot* lockobj = nullptr );
	~EveChildTurret();

	bool Initialize() override;
	bool OnModified( Be::Var * value ) override;
	void RegisterComponents() override;
	void UnRegisterComponents() override;

	void GetDebugOptions( Tr2DebugRendererOptions & options ) override;
	void RenderDebugInfo( ITr2DebugRenderer2 & renderer ) override;

	// EveSpaceObjectChild
	void UpdateSyncronous( const EveUpdateContext& updateContext, const EveChildUpdateParams& params ) override;
	void UpdateAsyncronous( const EveUpdateContext& updateContext, const EveChildUpdateParams& params ) override;

	void UpdateCachedGeometryData();
	void BuildCachedGeometryData( TriGeometryRes & geometryRes );
	void ReleaseCachedGeometryData();

	// action
	void EnterStateDeactive();
	void EnterStateIdle();
	void EnterStateTargeting();
	void EnterStateFiring();
	bool SetupFiringState();
	void EnterStateReloading();

	void ForceStateDeactive();
	void ForceIdleAnimation();
	void ForceStateTargeting();

	Matrix GetFiringBoneWorldTransform( unsigned int muzzle ) const;

	// turret set states
	enum State
	{
		STATE_INVALID = 0,
		STATE_DEACTIVE,
		STATE_IDLE,
		STATE_TARGETING,
		STATE_FIRING,
		STATE_RELOADING,
	};

protected:
	// system-controlled bones
	// TODO: needed?
	enum SystemBones
	{
		SYSBONE_INVALID = 0,
		SYSBONE_ROTATION,
		SYSBONE_ROTATION01,
		SYSBONE_ROTATION02,
		SYSBONE_COUNTER_ROTATION,
		SYSBONE_PITCH,
		SYSBONE_PITCH1,
		SYSBONE_PITCH2,
		SYSBONE_SCALED_HEIGHT,
		SYSBONE_SCALED_PITCH01,
		SYSBONE_SCALED_PITCH02,
		SYSBONE_SCALED_PITCH03,
		SYSBONE_SCALED_PITCH04,
		SYSBONE_SCALED_PITCH05,
		SYSBONE_SCALED_PITCH06,
		SYSBONE_MAX,
	};

	// setup the attached firing effect
	void InitializeFiringEffect();

	void InitializeAnimation() override;

	// TODO: Need update LOD ?

	// set transform for tracking
	void ModifySystemBoneTransform( SystemBones bone, const Vector3* target, const Matrix* localTransform, Vector3& position, Quaternion& rotation ) const;

	// Calculates the pitch for a bone based on the parameters
	void CalcTransformForPitchBone( const Vector3* target, float minPitch, float maxPitch, unsigned int boneIndex, const Matrix* localTransform, Quaternion& rotation ) const;

	// Returns the correct pitch factor for a specific bone index
	float GetBonePitchFactor( unsigned int boneIndex ) const;
	// Returns the correct pitch offset for a specific bone index
	float GetBonePitchOffset( unsigned int boneIndex ) const;

	Matrix GetTurretBoneTransform( uint32_t boneID ) const;

	TriGeometryRes* GetGeometryRes() const;

	// animation
	float PlayAnimation( const std::string& animName, const std::string& animNameIdle, float delay = 0.f );
	void StopAnimation( float delay = 0.f );
	std::string GetFireAnimationName() const;

	EveTurretFiringFX* GetFiringEffect();
	void SetFiringEffect( EveTurretFiringFX * firingEffect );

	// TODO: rename GrannyBoneBindingBounds?
	std::vector<GrannyBoneBindingBounds> m_boneBounds;

	// Assign the target object
	void SetTargetObject( IRoot * target );
	ITriTargetablePtr GetTargetObject();
	void SetTargetScale();

	// target (object we are tracking)
	EveTurretTargetPtr m_target;

	bool m_isOnline = true;

	// impacts
	float m_impactSize = 0.f;
	ImpactBehaviour::Type m_impactBehaviour = ImpactBehaviour::DAMAGE_LOCATOR;

	// tracking
	float m_trackingInfluence = 0.f;
	float m_trackingInfluenceDelta = 0.f;
	float m_delayToFadeOutTracking = 0.f;
	float m_delayToFadeInTracking = 0.f;
	float m_maxTrackingTime = 1.f;

	// animation
	const cmf::Skeleton* m_skeleton = nullptr;
	std::vector<int32_t> m_skeletonBoneIndices;
	std::unique_ptr<cmf::AnimationSequencer> m_sequencer;
	cmf::SkeletonPose m_pose;

	uint32_t m_maxCyclingFirePos = 1;
	uint32_t m_cyclingFireGroupCount = 1;
	uint32_t m_currentCyclingFiresPos = 0;

	// system bones
	unsigned int m_systemBoneID[SYSBONE_MAX];
	// specific system bone values
	float m_sysBoneHeight = 1.f;
	float m_sysBonePitchOffset = 0.f;
	float m_sysBonePitchFactor = 1.f;
	float m_sysBonePitchMin = 0.f;
	float m_sysBonePitchMax = 90.f;
	float m_sysBonePitch01Offset = 0.f;
	float m_sysBonePitch01Factor = 1.f;
	float m_sysBonePitch02Offset = 0.f;
	float m_sysBonePitch02Factor = 1.f;
	float m_sysBonePitch03Offset = 0.f;
	float m_sysBonePitch03Factor = 1.f;

	// state of turret set
	State m_state = STATE_IDLE;

	float m_recheckTimeLeft = -1.f;

	// firing effect redfile path
	std::string m_firingEffectResPath;

	// TODO: move firing effect into its own class?
	// firing effect
	EveTurretFiringFXPtr m_firingEffect;
	bool m_firingEffectMuzzlePosSet = false;

	// Audio specific attributes
	bool m_playMovementSound = true;
	TriObserverLocalPtr m_turretMovementObserver;
	std::wstring m_idleToTargetingMovementAudioEvent;
	std::wstring m_targetingToIdleMovementAudioEvent;

	TriGeometryResPtr m_cachedGeometryRes;
};

TYPEDEF_BLUECLASS( EveChildTurret );

#endif
