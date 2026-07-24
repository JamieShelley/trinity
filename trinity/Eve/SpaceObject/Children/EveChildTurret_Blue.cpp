// Copyright © 2026 Fenris Creations ehf.

#include "StdAfx.h"
#include "EveChildTurret.h"

BLUE_DEFINE( EveChildTurret );

const Be::ClassInfo* EveChildTurret::ExposeToBlue()
{
	EXPOSURE_BEGIN( EveChildTurret, "" )
		MAP_INTERFACE( EveChildTurret )
	EXPOSURE_CHAINTO( EveChildMesh )
}
