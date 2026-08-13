// Copyright © 2026

#pragma once
#ifndef Tr2TextureDetailReconstructionSettings_H
#define Tr2TextureDetailReconstructionSettings_H

namespace Tr2TextureDetailReconstructionSettings
{
enum Quality
{
    QUALITY_OFF = 0,
    QUALITY_BALANCED = 1,
    QUALITY_HIGH = 2
};

static const char* const SETTING_NAME = "textureDetailReconstructionQuality";

int GetQuality();
void SetQuality(int quality);
bool IsSupported();
uint64_t GetGeneration();
}

#endif // Tr2TextureDetailReconstructionSettings_H
