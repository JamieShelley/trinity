// Copyright © 2026 Fenris Creations ehf.

#pragma once
#ifndef EveChildTurret_H
#define EveChildTurret_H

#include "EveChildMesh.h"
#include "Eve/Turret/EveTurretTarget.h"

BLUE_DECLARE( EveTurretFiringFX );
BLUE_DECLARE( EveTurretTarget );
BLUE_DECLARE( EveTurretTarget );
BLUE_DECLARE( EveChildInstanceContainer );

BLUE_CLASS( EveChildTurret ) :
	public EveChildMesh, public IBlueAsyncResNotifyTarget
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

	// IBlueAsyncResNotifyTarget
	void ReleaseCachedData( BlueAsyncRes * resource ) override;
	void RebuildCachedData( BlueAsyncRes * resource ) override;

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

	// animation
	float PlayAnimation( const std::string& animName, const std::string& animNameIdle, float delay = 0.f );
	void StopAnimation( float delay = 0.f );
	std::string GetFireAnimationName() const;

	EveTurretFiringFXPtr GetFiringEffect();
	void SetFiringEffect( const EveTurretFiringFXPtr& firingEffect );

	// TODO: rename GrannyBoneBindingBounds?
	std::vector<GrannyBoneBindingBounds> m_boneBounds;

	// Assign the target object
	void SetTargetObject( IRoot * target );
	ITriTargetablePtr GetTargetObject();
	void SetTargetScale();

	// target (object we are tracking)
	EveTurretTargetPtr m_target;

	// impacts
	float m_impactSize;
	ImpactBehaviour::Type m_impactBehaviour;

	// tracking
	float m_trackingInfluence;
	float m_trackingInfluenceDelta;
	float m_delayToFadeOutTracking;
	float m_delayToFadeInTracking;
	float m_maxTrackingTime;

	// animation
	// TODO: needed?
	struct AnimationRequest
	{
		std::string animName;
		std::string animNameIdle;
	};
	std::vector<AnimationRequest> m_animationQueue;
	const cmf::Skeleton* m_skeleton;
	std::vector<int32_t> m_skeletonBoneIndices;
	std::unique_ptr<cmf::AnimationSequencer> m_sequencer;
	cmf::SkeletonPose m_pose;

	// system bones
	unsigned int m_systemBoneID[SYSBONE_MAX];
	// specific system bone values
	float m_sysBoneHeight;
	float m_sysBonePitchOffset;
	float m_sysBonePitchFactor;
	float m_sysBonePitchMin;
	float m_sysBonePitchMax;
	float m_sysBonePitch01Offset;
	float m_sysBonePitch01Factor;
	float m_sysBonePitch02Offset;
	float m_sysBonePitch02Factor;
	float m_sysBonePitch03Offset;
	float m_sysBonePitch03Factor;

	// state of turret set
	State m_state;

	float m_recheckTimeLeft;

	// firing effect redfile path
	std::string m_firingEffectResPath;

	// TODO: move firing effect into its own class?
	// firing effect
	EveTurretFiringFXPtr m_firingEffect;
	bool m_firingEffectMuzzlePosSet;

	// Audio specific attributes
	bool m_playMovementSound;
	TriObserverLocalPtr m_turretMovementObserver;
	std::wstring m_idleToTargetingMovementAudioEvent;
	std::wstring m_targetingToIdleMovementAudioEvent;
};

TYPEDEF_BLUECLASS( EveChildTurret );

#endif
