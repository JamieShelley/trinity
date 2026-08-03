#pragma once

#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
class PreviewShaderLibrary final
{
public:
    bool Compile(const char* entryPoint, const char* profile, ComPtr<ID3DBlob>& shaderBlob);
    std::string ResolveShaderPath() const;
};
} // namespace nsamdr
