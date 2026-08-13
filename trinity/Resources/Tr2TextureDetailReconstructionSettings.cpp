// Copyright © 2026

#include "StdAfx.h"
#include "Tr2TextureDetailReconstructionSettings.h"
#include "TriSettingsRegistrar.h"

namespace
{
int s_textureDetailReconstructionQuality = Tr2TextureDetailReconstructionSettings::QUALITY_OFF;
TRI_REGISTER_SETTING(
    Tr2TextureDetailReconstructionSettings::SETTING_NAME,
    s_textureDetailReconstructionQuality );

std::atomic<int> s_observedQuality( Tr2TextureDetailReconstructionSettings::QUALITY_OFF );
std::atomic<uint64_t> s_generation( 1 );

int ClampQuality( int quality )
{
    if( quality < Tr2TextureDetailReconstructionSettings::QUALITY_OFF )
    {
        return Tr2TextureDetailReconstructionSettings::QUALITY_OFF;
    }
    if( quality > Tr2TextureDetailReconstructionSettings::QUALITY_HIGH )
    {
        return Tr2TextureDetailReconstructionSettings::QUALITY_HIGH;
    }
    return quality;
}
}

namespace Tr2TextureDetailReconstructionSettings
{
int GetQuality()
{
    const int quality = ClampQuality( s_textureDetailReconstructionQuality );
    s_textureDetailReconstructionQuality = quality;
    const int previous = s_observedQuality.exchange( quality );
    if( previous != quality )
    {
        s_generation.fetch_add( 1 );
    }
    return quality;
}

void SetQuality( int quality )
{
    s_textureDetailReconstructionQuality = ClampQuality( quality );
    GetQuality();
}

bool IsSupported()
{
#if defined( _WIN32 )
    return true;
#else
    return false;
#endif
}

uint64_t GetGeneration()
{
    GetQuality();
    return s_generation.load();
}
}
