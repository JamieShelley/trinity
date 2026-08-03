#include "StdAfx.h"
#include "NSAMDRShaderLibrary.h"

namespace nsamdr
{
bool PreviewShaderLibrary::Compile(const char* entryPoint, const char* profile, ComPtr<ID3DBlob>& shaderBlob)
{
    const std::string shaderPath = ResolveShaderPath();
    const std::wstring wideShaderPath = ToWidePath(shaderPath);
    if (wideShaderPath.empty())
    {
        const std::string diagnostic = "NSAMDR could not resolve adjacent shader file: " + shaderPath;
        std::fprintf(stderr, "%s\n", diagnostic.c_str());
        OutputDebugStringA(diagnostic.c_str());
        ADD_FAILURE() << diagnostic;
        return false;
    }

    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
#if defined(_DEBUG)
    flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    ComPtr<ID3DBlob> errorBlob;
    const HRESULT result = D3DCompileFromFile(
        wideShaderPath.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entryPoint,
        profile,
        flags,
        0,
        shaderBlob.ReleaseAndGetAddressOf(),
        errorBlob.ReleaseAndGetAddressOf());

    if (FAILED(result))
    {
        const char* errorText = errorBlob
            ? static_cast<const char*>(errorBlob->GetBufferPointer())
            : "Unknown HLSL compilation error";
        std::ostringstream diagnostic;
        diagnostic << "NSAMDR HLSL compilation failed for " << entryPoint
                   << " (" << profile << ") from " << shaderPath << "\n"
                   << errorText;
        const std::string diagnosticText = diagnostic.str();
        std::fprintf(stderr, "%s\n", diagnosticText.c_str());
        OutputDebugStringA(diagnosticText.c_str());
        ADD_FAILURE() << diagnosticText;
        return false;
    }
    return true;
}

std::string PreviewShaderLibrary::ResolveShaderPath() const
{
#if defined(NSAMDR_PREVIEW_SHADER_PATH)
    return NSAMDR_PREVIEW_SHADER_PATH;
#else
    std::string adjacent = __FILE__;
    const size_t separator = adjacent.find_last_of("/\\");
    if (separator != std::string::npos)
    {
        adjacent.resize(separator + 1U);
    }
    else
    {
        adjacent.clear();
    }
    adjacent += "NSAMDRPreview.hlsl";
    return adjacent;
#endif
}

} // namespace nsamdr
