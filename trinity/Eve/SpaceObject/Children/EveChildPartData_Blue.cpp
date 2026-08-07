// Copyright © 2026 CCP ehf.

#include "StdAfx.h"
#include "EveChildPartData.h"


BLUE_DEFINE( EveChildPartData );

const Be::ClassInfo* EveChildPartData::ExposeToBlue()
{
	EXPOSURE_BEGIN( EveChildPartData, "" )
		MAP_INTERFACE( EveSpaceObjectChild );
		MAP_INTERFACE( IEveSpaceObjectChild )
	EXPOSURE_END()
}


BLUE_DEFINE_NONEXPOSED( EveModularObjectModifier );

const Be::ClassInfo* EveModularObjectModifier::ExposeToBlue()
{
	EXPOSURE_BEGIN( EveModularObjectModifier, "" )
		MAP_METHOD_AND_WRAP( "AddHull", AddHull, "" );
		MAP_METHOD_AND_WRAP( "AddChild", AddChild, "" )
		MAP_METHOD_AND_WRAP( "Remove", Remove, "" )
		MAP_METHOD_AND_WRAP( "SetTransform", SetTransform, "" )
		MAP_METHOD_AND_WRAP( "GetPosition", GetPosition, "" )
		MAP_METHOD_AND_WRAP( "GetRotation", GetRotation, "" )
		MAP_METHOD_AND_WRAP( "GetScale", GetScale, "" )

	EXPOSURE_END()
}

MAP_FUNCTION_AND_WRAP( "CreateModularObject", CreateModularObject, "" );
MAP_FUNCTION_AND_WRAP( "ModifyModularObject", ModifyModularObject, "" );

MAP_FUNCTION_AND_WRAP( "GetInvalidPartTag", GetInvalidPartTag, "Gets the NO_PART_TAG constant" );