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
 * @brief A standalone spatial trigger that fires a Python callback when a tracked position enters or exits its volumes.
 *
 * The trigger region is defined by a list of IEveVolume shapes (box/sphere/ellipsoid), placed relative
 * to the object's translation/rotation and editable in Graphite like any other top-level scene object.
 * The tracked position (typically the player ship's destiny ball) is attached from Python via the
 * trackedPositionCurve slot. When the tracked position crosses the enterThreshold intensity boundary,
 * the registered callback is invoked as callback( name, entered ) at a safe point after update.
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

#if BLUE_WITH_PYTHON
	/**
	 * @brief Sets the Python callable invoked on enter/exit transitions.
	 *
	 * The callable is invoked as callback( name, entered ) where entered is True on entry
	 * and False on exit. Pass None to clear the callback.
	 *
	 * @param callable Python callable or None.
	 */
	void SetCallback( PyObject* callable );

	/**
	 * @brief Invokes the stored callback. Called from the post-update callback on the main thread.
	 * @param entered True if the tracked position entered the volume, false if it exited.
	 */
	void InvokeCallback( bool entered );
#endif

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
	 * @brief Evaluates whether the tracked position is inside the volumes and fires the callback on transitions.
	 */
	void UpdateTriggerState( const EveUpdateContext& updateContext );

	/**
	 * @brief Queues the callback for invocation at the post-update point on the main thread.
	 * @param entered True if the tracked position entered the volume, false if it exited.
	 */
	void QueueCallback( bool entered );

	std::string m_name; ///< The name identifier, passed to the callback so one handler can serve many volumes.
	PIEveVolumeVector m_volumes; ///< The volumes defining the trigger region.
	PIEveVolumeVector m_exclusionVolumes; ///< Volumes subtracted from the trigger region.

	CcpMath::Sphere m_boundingSphere; ///< Broad-phase bounding sphere around all volumes, in local space.

	// TODO: derive the tracked position from the EveSpace scene instead of attaching it from Python.
	ITriVectorFunctionPtr m_trackedPosition; ///< Vector function slot for attaching a destiny ball as the tracked position.

	Matrix m_worldTransform; ///< World transform built from the translation/rotation attributes.

	float m_enterThreshold; ///< Volume intensity at which the tracked position counts as inside (0..1).
	bool m_isInside; ///< Current inside/outside state of the tracked position.
	float m_currentIntensity; ///< Most recent evaluated intensity, for debugging.
	bool m_display; ///< Not really used for trigger volumes, but here for consistency with the EveSpaceObject interface.

#if BLUE_WITH_PYTHON

	// TODO: replace the raw PyObject callback with BLUESCRIPTCALLBACK.
	PyObject* m_callable; ///< Python callable invoked on enter/exit transitions.

	// TODO: bind controllers to the trigger volume for VFX.
#endif
};

/**
 * @brief Macro that creates container typedefs for EveTriggerVolume.
 */
TYPEDEF_BLUECLASS( EveTriggerVolume );

#endif
