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
BLUE_DECLARE( Tr2ExternalParameter );
BLUE_DECLARE_VECTOR( Tr2ExternalParameter );
BLUE_DECLARE( EveTriggerVolume );

/**
 * @class EveTriggerVolume
 * @brief A volume that triggers a Python callback when a tracked position enters or exits it.
 *
 */
BLUE_CLASS( EveTriggerVolume ) :
	public IWorldPosition,
	public IEveSpaceObject2,
	public IInitialize,
	public ITr2DebugRenderable
{
public:
	EXPOSE_TO_BLUE();

	EveTriggerVolume( IRoot* lockobj = NULL );
	~EveTriggerVolume();

	/**
	 * @brief Recomputes the broad-phase bounding sphere from the volume list.
	 */
	void RebuildBoundingSphere();

	/**
	 * @brief Sets the callable invoked on enter/exit transitions.
	 *
	 * The callable is invoked as callback( name, entered ) where entered is True on entry
	 * and False on exit. Pass None to clear the callback.
	 *
	 * @param callback Callable or None.
	 */
	void SetCallback( const BlueScriptCallback& callback );

	/**
	 * @brief Invokes the stored callback. Called from the post-update callback on the main thread.
	 * @param entered True if the tracked position entered the volume, false if it exited.
	 */
	void InvokeCallback( bool entered );

	/////////////////////////////////////////////////////////////////////////////////////
	// IEveSpaceObject2
	void UpdateSyncronous( const EveUpdateContext& updateContext ) override;
	void UpdateAsyncronous( const EveUpdateContext& updateContext ) override;
	void UpdateVisibility( const EveUpdateContext& updateContext, const Matrix& parentTransform ) override;
	void GetRenderables( std::vector<ITr2Renderable*> & renderables, Tr2ImpostorManager * impostors ) override;
	bool GetBoundingSphere( Vector4 & sphere, BoundingSphereQuery query = EVE_BOUNDS_NORMAL ) const override;
	void UpdateModelCenterWorldPosition( Vector3 & position, Be::Time t ) override;
	void GetModelCenterWorldPosition( Vector3 & position ) const override;
	bool GetLocalBoundingBox( Vector3 & min, Vector3 & max ) override;
	void GetLocalToWorldTransform( Matrix & transform ) const override;

	/////////////////////////////////////////////////////////////////////////////////////
	// IWorldPosition
	Vector3 GetWorldPosition() override;
	Quaternion GetWorldRotation() override;

	/////////////////////////////////////////////////////////////////////////////////////
	// IInitialize
	bool Initialize() override;

	/////////////////////////////////////////////////////////////////////////////////////
	// ITr2DebugRenderable
	void GetDebugOptions( Tr2DebugRendererOptions & options ) override;
	void RenderDebugInfo( ITr2DebugRenderer2 & renderer ) override;

	Quaternion m_rotation; ///< Local rotation of the trigger volume, editable in Graphite.
	Vector3 m_translation; ///< Local translation of the trigger volume, editable in Graphite.

private:
	/**
	 * @brief Rebuilds the world transform from the position/rotation curves when attached
	 * (e.g. a destiny ball in the client), otherwise from the translation/rotation attributes.
	 */
	void UpdateWorldTransform( Be::Time time );

	/**
	 * @brief Returns the highest intensity any enabled volume in the list gives the position.
	 * @param volumes The volumes to evaluate.
	 * @param position The position to evaluate, in object space.
	 */
	static float GetMaxIntensity( const PIEveVolumeVector& volumes, const Vector3& position );

	/**
	 * @brief Evaluates whether the tracked position is inside the volumes and fires the callback on transitions.
	 */
	void UpdateTriggerState( const EveUpdateContext& updateContext );

	/**
	 * @brief Queues the callback for invocation at the post-update point on the main thread.
	 * @param entered True if the tracked position entered the volume, false if it exited.
	 */
	void QueueCallback( bool entered );

	/**
	 * @brief Returns the name passed to the callback.
	 *
	 * Prefers the first non-empty volume name over the name attribute: external parameters
	 * in a .red file cannot reference the root object, so per-placement names (e.g. dungeon
	 * asset manipulations) are bound to the first volume, and the client overwrites the root
	 * name attribute with the destiny ball ID when adding the object to the scene.
	 */
	const char* GetEffectiveName() const;

	std::string m_name; ///< The name identifier, passed to the callback so one handler can serve many volumes.
	PIEveVolumeVector m_volumes; ///< The volumes defining the trigger region.
	PIEveVolumeVector m_exclusionVolumes; ///< Volumes subtracted from the trigger region.
	PTr2ExternalParameterVector m_externalParameters; ///< External parameters exposing per-placement values, e.g. for dungeon asset manipulations.

	CcpMath::Sphere m_boundingSphere; ///< Broad-phase bounding sphere around all volumes, in local space.

	// TODO: derive the tracked position from the EveSpace scene instead of attaching it from Python.
	ITriVectorFunctionPtr m_trackedPosition; ///< Vector function slot for attaching a destiny ball as the tracked position.

	ITriVectorFunctionPtr m_ballPosition; ///< Position curve slot; the client attaches the object's own destiny ball here.
	ITriQuaternionFunctionPtr m_ballRotation; ///< Rotation curve slot; the client attaches the object's own destiny ball here.

	Matrix m_worldTransform; ///< World transform built from the position/rotation curves or the translation/rotation attributes.

	float m_enterThreshold; ///< Volume intensity at which the tracked position counts as inside (0..1).
	bool m_forceTriggered; ///< Debug: force the trigger into the entered state, e.g. for testing in Graphite.
	bool m_isInside; ///< Current inside/outside state of the tracked position.
	float m_currentIntensity; ///< Most recent evaluated intensity, for debugging.
	bool m_display; ///< Not really used for trigger volumes, but here for consistency with the EveSpaceObject interface.

	BlueScriptCallback m_callback; ///< Script callable invoked on enter/exit transitions.

	// TODO: bind controllers to the trigger volume for VFX.
};

/**
 * @brief Macro that creates container typedefs for EveTriggerVolume.
 */
TYPEDEF_BLUECLASS( EveTriggerVolume );

#endif
