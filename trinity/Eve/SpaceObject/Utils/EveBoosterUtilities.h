// Copyright © 2026 CCP ehf.

#pragma once
#ifndef EveBoosterUtilities_H
#define EveBoosterUtilities_H

#include "Tr2ProceduralResources.h"

struct EveBoosterVertex
{
	Vector3 position;
	Vector2 texCoord;
};

struct EveChildBoosterVertex
{
	Vector3 position;
};

Tr2ProceduralBuffer MakeChildBoosterBoxBuffer();
Tr2ProceduralBuffer MakeBoosterBoxBuffer();
Tr2ProceduralBuffer MakeBoosterStarBuffer();

#endif
