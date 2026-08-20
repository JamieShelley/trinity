// Copyright © 2026 CCP ehf.

#pragma once

#ifndef EveTriggerVolume_h
#define EveTriggerVolume_h

#include "IWorldPosition.h"
#include "IEveSpaceObject2.h"
#include "Tr2DebugRenderer.h"
#include "Eve/Volume/IEveVolume.h"
#include "Utilities/BoundingSphere.h"

#ifdef BLUE_USE_LOCAL_ITr2DebugRenderer2
// This is only needed for py2 as the file now belongs in blue.
// Unfortunatly the blue py2 branch cannot be updated at present due to security vulnerability work.
// The file version in the older blue versions had diverged from this one is incompatible.
#include "Include/ITr2DebugRenderer2.h"
#else
#include <ITr2DebugRenderer2.h>
#endif

#include <ITriFunction.h>

BLUE_DECLARE_INTERFACE( IEveVolume );
BLUE_DECLARE_IVECTOR( IEveVolume );
BLUE_DECLARE( EveTriggerVolume );

/**
 * @class EveTriggerVolume
 * @brief A standalone spatial trigger that tracks whether a tracked position is inside its volumes.
 *
 * The trigger region is defined by a list of IEveVolume shapes (box/sphere/ellipsoid), placed relative
 * to the object's translation/rotation and editable in Graphite like any other top-level scene object.
 * The tracked position (typically the player ship's destiny ball) is attached from Python via the
 * trackedPositionCurve slot. The isInside attribute reflects whether it is past the
 * enterThreshold intensity boundary.
 */
BLUE_CLASS( EveTriggerVolume ) :
	public IWorldPosition,
	public IEveSpaceObject2,
	public IInitialize
{
public:
	EXPOSE_TO_BLUE();

	EveTriggerVolume( IRoot* lockobj = NULL );
	~EveTriggerVolume();

	/**
	 * @brief Recomputes the broad-phase bounding sphere from the volume list.
	 */
	void RebuildBoundingSphere();

	/////////////////////////////////////////////////////////////////////////////////////
	// IEveSpaceObject2
	void UpdateSyncronous( const EveUpdateContext& updateContext );
	void UpdateAsyncronous( const EveUpdateContext& updateContext );
	void UpdateVisibility( const EveUpdateContext& updateContext, const Matrix& parentTransform );
	void GetRenderables( std::vector<ITr2Renderable*> & renderables, Tr2ImpostorManager * impostors );
	bool GetBoundingSphere( Vector4 & sphere, BoundingSphereQuery query = EVE_BOUNDS_NORMAL ) const;
	void UpdateModelCenterWorldPosition( Vector3 & position, Be::Time t );
	void GetModelCenterWorldPosition( Vector3 & position ) const;
	bool GetLocalBoundingBox( Vector3 & min, Vector3 & max );
	void GetLocalToWorldTransform( Matrix & transform ) const;

	/////////////////////////////////////////////////////////////////////////////////////
	// IWorldPosition
	virtual Vector3 GetWorldPosition();
	virtual Quaternion GetWorldRotation();

	/////////////////////////////////////////////////////////////////////////////////////
	// IInitialize
	bool Initialize() override;

	Quaternion m_rotation; ///< Local rotation of the trigger volume, editable in Graphite.
	Vector3 m_translation; ///< Local translation of the trigger volume, editable in Graphite.

private:
	/**
	 * @brief Rebuilds the world transform from the translation/rotation attributes.
	 */
	void UpdateWorldTransform();

	/**
	 * @brief Evaluates whether the tracked position is inside the volumes.
	 */
	void UpdateTriggerState( const EveUpdateContext& updateContext );

	std::string m_name; ///< The name identifier, so one handler can serve many trigger volumes.
	PIEveVolumeVector m_volumes; ///< The volumes defining the trigger region.

	CcpMath::Sphere m_boundingSphere; ///< Broad-phase bounding sphere around all volumes, in local space.

	// TODO: derive the tracked position from the EveSpace scene instead of attaching it from Python.
	ITriVectorFunctionPtr m_trackedPosition; ///< Vector function slot for attaching a destiny ball as the tracked position.

	Matrix m_worldTransform; ///< World transform built from the translation/rotation attributes.

	float m_enterThreshold; ///< Volume intensity at which the tracked position counts as inside (0..1).
	bool m_isInside; ///< Current inside/outside state of the tracked position.
	float m_currentIntensity; ///< Most recent evaluated intensity, for debugging.
	bool m_display; ///< Not really used for trigger volumes, but here for consistency with the EveSpaceObject interface.
};

/**
 * @brief Macro that creates container typedefs for EveTriggerVolume.
 */
TYPEDEF_BLUECLASS( EveTriggerVolume );

#endif
