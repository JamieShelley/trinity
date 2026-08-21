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

Tr2ProceduralBuffer MakeBoosterBoxBuffer();
Tr2ProceduralBuffer MakeBoosterStarBuffer();

#endif