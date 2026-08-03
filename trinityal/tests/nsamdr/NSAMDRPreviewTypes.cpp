#include "StdAfx.h"
#include "NSAMDRPreviewTypes.h"

namespace nsamdr
{
const char* ShaderFamilyName(ShaderFamily family)
{
    switch (family)
    {
        case ShaderFamily::LegacyPgs: return "legacy PGS";
        case ShaderFamily::V5Separate: return "V5 separate";
        case ShaderFamily::V5Packed: return "V5++ packed";
        default: return "unknown";
    }
}

} // namespace nsamdr
