#include "StdAfx.h"
#include "NSAMDRAssetProcessor.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
AssetProcessor::AssetProcessor(PreviewShaderLibrary& shaderLibrary, MeshProcessor& meshProcessor)
    : m_shaderLibrary(shaderLibrary),
      m_meshProcessor(meshProcessor)
{
}

bool AssetProcessor::CreateFallbackTexture(ID3D11Device* device, PreviewResources& resources)
{
    const uint32_t pixels[4] = {
        0xff251b10U,
        0xff332419U,
        0xff332419U,
        0xff251b10U,
    };

    D3D11_TEXTURE2D_DESC textureDescription{};
    textureDescription.Width = 2;
    textureDescription.Height = 2;
    textureDescription.MipLevels = 1;
    textureDescription.ArraySize = 1;
    textureDescription.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDescription.SampleDesc.Count = 1;
    textureDescription.Usage = D3D11_USAGE_IMMUTABLE;
    textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initialData{};
    initialData.pSysMem = pixels;
    initialData.SysMemPitch = 2U * sizeof(uint32_t);

    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(device->CreateTexture2D(&textureDescription, &initialData, texture.GetAddressOf())) ||
        FAILED(device->CreateShaderResourceView(texture.Get(), nullptr, resources.albedoView.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create fallback NSAMDR texture";
        return false;
    }

    resources.textureWidth = 2;
    resources.textureHeight = 2;
    resources.hasExternalAlbedo = false;
    return true;
}

ShaderFamily AssetProcessor::ParseShaderFamily(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "legacy_pgs") return ShaderFamily::LegacyPgs;
    if (lower == "v5_separate") return ShaderFamily::V5Separate;
    if (lower == "v5_packed") return ShaderFamily::V5Packed;
    return ShaderFamily::Unknown;
}

MaterialPass AssetProcessor::ParseMaterialPass(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "decal") return MaterialPass::Decal;
    if (lower == "transparent") return MaterialPass::Transparent;
    if (lower == "additive") return MaterialPass::Additive;
    return MaterialPass::Opaque;
}

std::vector<std::string> AssetProcessor::SplitTabs(const std::string& line)
{
    std::vector<std::string> values;
    size_t start = 0;
    for (;;)
    {
        const size_t separator = line.find('\t', start);
        values.push_back(line.substr(start, separator == std::string::npos ? std::string::npos : separator - start));
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    return values;
}

bool AssetProcessor::ParseFloatValue(const std::string& text, float& value)
{
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || (end && *end != '\0') || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

bool AssetProcessor::ParseIntValue(const std::string& text, int& value)
{
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || (end && *end != '\0')) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool AssetProcessor::LoadAreaMaterialSources(const std::string& path, std::vector<AreaMaterialSource>& materials, std::string& error)
{
    materials.clear();
    if (path.empty()) return true;
    std::ifstream input(path);
    if (!input)
    {
        error = "Could not open SOF material manifest: " + path;
        return false;
    }

    std::map<std::string, size_t> columns;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> values = SplitTabs(line);
        if (columns.empty())
        {
            for (size_t index = 0; index < values.size(); ++index) columns[values[index]] = index;
            continue;
        }

        auto value = [&](const char* name, const char* alias = nullptr) -> std::string {
            auto it = columns.find(name);
            if (it == columns.end() && alias) it = columns.find(alias);
            return it != columns.end() && it->second < values.size() ? values[it->second] : std::string();
        };
        auto parse = [&](const char* name, float& destination, const char* alias = nullptr) -> bool {
            const std::string text = value(name, alias);
            return text.empty() || ParseFloatValue(text, destination);
        };

        AreaMaterialSource material;
        try
        {
            const std::string group = value("group");
            if (group.empty()) throw std::runtime_error("missing group");
            material.groupIndex = std::stoi(group);
        }
        catch (...)
        {
            error = "Malformed group index in SOF material manifest";
            return false;
        }
        material.pass = ParseMaterialPass(value("pass"));
        material.areaType = value("area_type");
        material.areaName = value("area_name");
        material.shaderPath = value("shader");
        material.shaderFamily = ParseShaderFamily(value("shader_family"));
        material.albedoPath = value("albedo");
        material.normalPath = value("normal");
        material.pgsPath = value("material", "pgs");
        material.glowPath = value("glow");
        material.dirtPath = value("dirt");
        material.aoPath = value("ao");
        material.paintMaskPath = value("paint_mask");
        material.roughnessMapPath = value("roughness_map");

        auto parseChannel = [&](const char* name, int& destination) -> bool {
            const std::string text = value(name);
            return text.empty() || ParseIntValue(text, destination);
        };
        if (!parseChannel("normal_x_channel", material.normalXChannel) ||
            !parseChannel("normal_y_channel", material.normalYChannel) ||
            !parseChannel("roughness_channel", material.roughnessChannel) ||
            !parseChannel("material_channel", material.materialChannel) ||
            !parseChannel("ao_channel", material.aoChannel) ||
            !parseChannel("paint_channel", material.paintChannel) ||
            !parseChannel("dirt_channel", material.dirtChannel) ||
            !parseChannel("glow_channel", material.glowChannel))
        {
            error = "Malformed semantic channel in SOF material manifest";
            return false;
        }

        if (!parse("tint_r", material.tint.x) ||
            !parse("tint_g", material.tint.y) ||
            !parse("tint_b", material.tint.z) ||
            !parse("detail_scale", material.detailScale) ||
            !parse("roughness", material.roughness) ||
            !parse("specular", material.specular) ||
            !parse("alpha", material.alpha) ||
            !parse("glow_r", material.glowColour.x) ||
            !parse("glow_g", material.glowColour.y) ||
            !parse("glow_b", material.glowColour.z))
        {
            error = "Malformed numeric value in SOF material manifest";
            return false;
        }

        const std::array<XMFLOAT3, 4> fallbackColours = {
            material.tint,
            XMFLOAT3(material.tint.x * 1.15f, material.tint.y * 1.15f, material.tint.z * 1.15f),
            XMFLOAT3(material.tint.x * 0.62f, material.tint.y * 0.62f, material.tint.z * 0.62f),
            XMFLOAT3(0.035f, 0.045f, 0.055f),
        };
        for (size_t slotIndex = 0; slotIndex < material.slots.size(); ++slotIndex)
        {
            MaterialSlotSource& slot = material.slots[slotIndex];
            slot.colour = fallbackColours[slotIndex];
            const float legacyF0 = std::max(0.018f, std::min(0.22f, 0.04f * material.specular));
            slot.f0 = XMFLOAT3(legacyF0, legacyF0, legacyF0);
            slot.gloss = std::max(0.0f, 1.0f - material.roughness);

            const std::string prefix = "mtl" + std::to_string(slotIndex + 1U) + "_";
            if (!parse((prefix + "r").c_str(), slot.colour.x) ||
                !parse((prefix + "g").c_str(), slot.colour.y) ||
                !parse((prefix + "b").c_str(), slot.colour.z) ||
                !parse((prefix + "f0_r").c_str(), slot.f0.x) ||
                !parse((prefix + "f0_g").c_str(), slot.f0.y) ||
                !parse((prefix + "f0_b").c_str(), slot.f0.z))
            {
                error = "Malformed material-slot value in SOF material manifest";
                return false;
            }
            const std::string glossText = value((prefix + "gloss").c_str());
            if (!glossText.empty())
            {
                if (!ParseFloatValue(glossText, slot.gloss))
                {
                    error = "Malformed material gloss in SOF material manifest";
                    return false;
                }
            }
            else
            {
                float legacyRoughness = 1.0f - slot.gloss;
                const std::string roughnessText = value((prefix + "roughness").c_str());
                if (!roughnessText.empty() && !ParseFloatValue(roughnessText, legacyRoughness))
                {
                    error = "Malformed legacy material roughness in SOF material manifest";
                    return false;
                }
                slot.gloss = std::max(0.0f, 1.0f - legacyRoughness);
            }
        }
        parse("general_data_x", material.generalDataX);
        const std::string semanticComplete = value("semantic_complete");
        material.semanticComplete = semanticComplete == "1" || semanticComplete == "true" || semanticComplete == "TRUE";
        const std::string baselineComplete = value("baseline_complete");
        material.baselineComplete = baselineComplete == "1" || baselineComplete == "true" || baselineComplete == "TRUE";
        material.unresolvedSemantics = value("unresolved_semantics");
        const std::string unresolvedCount = value("unresolved_count");
        if (!unresolvedCount.empty())
        {
            try { material.unresolvedCount = std::max(0, std::stoi(unresolvedCount)); }
            catch (...) { material.unresolvedCount = 1; }
        }
        materials.push_back(std::move(material));
    }
    if (columns.empty())
    {
        error = "SOF material manifest has no column header";
        return false;
    }
    return true;
}

bool AssetProcessor::LoadWicTexture(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const std::string& path,
    bool srgb,
    ComPtr<ID3D11ShaderResourceView>& view,
    uint32_t& outputWidth,
    uint32_t& outputHeight,
    const char* label)
{
    // -------------------------------------------------------------------------
    // MODE 7 / SECTION 4 — CUSTOM SEMANTIC MIP LOADING
    // The offline generator writes adjacent "name.mipN.png" levels. When those
    // files exist, load them verbatim rather than asking DirectX GenerateMips to
    // average colour, normal, roughness and categorical masks identically.
    // -------------------------------------------------------------------------
    const HRESULT initialiseResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool mustUninitialise = SUCCEEDED(initialiseResult);
    if (FAILED(initialiseResult) && initialiseResult != RPC_E_CHANGED_MODE)
    {
        ADD_FAILURE() << "COM initialisation failed while loading " << label;
        return false;
    }

    struct DecodedMip
    {
        UINT width = 0;
        UINT height = 0;
        std::vector<uint8_t> pixels;
    };

    bool success = false;
    do
    {
        ComPtr<IWICImagingFactory> factory;
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(factory.GetAddressOf()))))
        {
            ADD_FAILURE() << "Could not create Windows Imaging Component factory";
            break;
        }

        auto decode = [&](const std::string& imagePath, DecodedMip& decoded, bool required) -> bool
        {
            const std::wstring widePath = ToWidePath(imagePath);
            if (widePath.empty())
            {
                if (required) ADD_FAILURE() << "Could not convert " << label << " path to a Windows path: " << imagePath;
                return false;
            }
            ComPtr<IWICBitmapDecoder> decoder;
            if (FAILED(factory->CreateDecoderFromFilename(widePath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, decoder.GetAddressOf())))
            {
                if (required) ADD_FAILURE() << "Could not decode " << label << ": " << imagePath;
                return false;
            }
            ComPtr<IWICBitmapFrameDecode> frame;
            if (FAILED(decoder->GetFrame(0, frame.GetAddressOf())) ||
                FAILED(frame->GetSize(&decoded.width, &decoded.height)) ||
                decoded.width == 0 || decoded.height == 0)
            {
                if (required) ADD_FAILURE() << label << " has invalid dimensions: " << imagePath;
                return false;
            }
            ComPtr<IWICFormatConverter> converter;
            if (FAILED(factory->CreateFormatConverter(converter.GetAddressOf())) ||
                FAILED(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)))
            {
                if (required) ADD_FAILURE() << "Could not convert " << label << " to RGBA8: " << imagePath;
                return false;
            }
            const size_t rowPitch = static_cast<size_t>(decoded.width) * 4U;
            const size_t byteCount = rowPitch * static_cast<size_t>(decoded.height);
            if (byteCount > static_cast<size_t>(std::numeric_limits<UINT>::max()))
            {
                if (required) ADD_FAILURE() << label << " is too large: " << imagePath;
                return false;
            }
            decoded.pixels.resize(byteCount);
            if (FAILED(converter->CopyPixels(nullptr, static_cast<UINT>(rowPitch), static_cast<UINT>(byteCount), decoded.pixels.data())))
            {
                if (required) ADD_FAILURE() << "Could not copy " << label << " pixels: " << imagePath;
                return false;
            }
            return true;
        };

        std::vector<DecodedMip> mips;
        mips.emplace_back();
        if (!decode(path, mips.back(), true)) break;

        const size_t extensionPosition = path.find_last_of('.');
        const std::string stem = extensionPosition == std::string::npos ? path : path.substr(0, extensionPosition);
        const std::string extension = extensionPosition == std::string::npos ? std::string(".png") : path.substr(extensionPosition);
        for (int level = 1; level < 20; ++level)
        {
            const std::string mipPath = stem + ".mip" + std::to_string(level) + extension;
            std::ifstream probe(mipPath, std::ios::binary);
            if (!probe.good()) break;
            probe.close();

            DecodedMip decoded;
            if (!decode(mipPath, decoded, true))
            {
                mips.clear();
                break;
            }
            const UINT expectedWidth = std::max<UINT>(1U, mips.back().width / 2U);
            const UINT expectedHeight = std::max<UINT>(1U, mips.back().height / 2U);
            if (decoded.width != expectedWidth || decoded.height != expectedHeight)
            {
                ADD_FAILURE() << "Custom semantic mip has incorrect dimensions: " << mipPath
                              << " expected=" << expectedWidth << "x" << expectedHeight
                              << " actual=" << decoded.width << "x" << decoded.height;
                mips.clear();
                break;
            }
            mips.push_back(std::move(decoded));
            if (mips.back().width == 1U && mips.back().height == 1U) break;
        }
        if (mips.empty()) break;

        ComPtr<ID3D11Texture2D> texture;
        D3D11_TEXTURE2D_DESC textureDescription{};
        textureDescription.Width = mips.front().width;
        textureDescription.Height = mips.front().height;
        textureDescription.ArraySize = 1;
        textureDescription.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        textureDescription.SampleDesc.Count = 1;
        textureDescription.Usage = D3D11_USAGE_DEFAULT;
        textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        if (mips.size() > 1U)
        {
            textureDescription.MipLevels = static_cast<UINT>(mips.size());
            std::vector<D3D11_SUBRESOURCE_DATA> initialData(mips.size());
            for (size_t index = 0; index < mips.size(); ++index)
            {
                initialData[index].pSysMem = mips[index].pixels.data();
                initialData[index].SysMemPitch = mips[index].width * 4U;
            }
            if (FAILED(device->CreateTexture2D(&textureDescription, initialData.data(), texture.GetAddressOf())))
            {
                ADD_FAILURE() << "Could not create DirectX custom-mip " << label << " texture: " << path;
                break;
            }
        }
        else
        {
            // Existing modes retain the original runtime-generated mip path.
            textureDescription.MipLevels = 0;
            textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
            textureDescription.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
            if (FAILED(device->CreateTexture2D(&textureDescription, nullptr, texture.GetAddressOf())))
            {
                ADD_FAILURE() << "Could not create DirectX " << label << " texture: " << path;
                break;
            }
            context->UpdateSubresource(
                texture.Get(),
                0,
                nullptr,
                mips.front().pixels.data(),
                mips.front().width * 4U,
                0);
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC viewDescription{};
        viewDescription.Format = srgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
        viewDescription.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        viewDescription.Texture2D.MostDetailedMip = 0;
        viewDescription.Texture2D.MipLevels = mips.size() > 1U ? static_cast<UINT>(mips.size()) : UINT(-1);
        if (FAILED(device->CreateShaderResourceView(texture.Get(), &viewDescription, view.ReleaseAndGetAddressOf())))
        {
            ADD_FAILURE() << "Could not create DirectX " << label << " texture view: " << path;
            break;
        }
        if (mips.size() == 1U) context->GenerateMips(view.Get());

        outputWidth = mips.front().width;
        outputHeight = mips.front().height;
        if (mips.size() > 1U)
        {
            std::printf("NSAMDR semantic mip chain loaded: %s levels=%zu\n", path.c_str(), mips.size());
        }
        success = true;
    } while (false);

    if (mustUninitialise) CoUninitialize();
    return success;
}

bool AssetProcessor::EnsureSelectedEnvironmentLoaded(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    PreviewResources& resources,
    const PreviewState& state)
{
    if (resources.environments.empty()) return true;
    const size_t selectedIndex = std::min<size_t>(state.environmentIndex, resources.environments.size() - 1U);
    EnvironmentGpu& selected = resources.environments[selectedIndex];
    if (!selected.view)
    {
        // Keep one large equirectangular background resident at a time.
        for (size_t index = 0; index < resources.environments.size(); ++index)
        {
            if (index != selectedIndex) resources.environments[index].view.Reset();
        }
        if (!LoadWicTexture(
                device,
                context,
                selected.path,
                true,
                selected.view,
                selected.width,
                selected.height,
                "EVE environment image"))
        {
            return false;
        }
    }
    resources.environmentView = selected.view;
    resources.environmentWidth = selected.width;
    resources.environmentHeight = selected.height;
    return true;
}

bool AssetProcessor::CreatePreviewTargets(
    ID3D11Device* device,
    IDXGISwapChain* swapChain,
    PreviewResources& resources)
{
    resources.renderTargetView.Reset();
    resources.depthStencilView.Reset();
    resources.depthTexture.Reset();

    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(backBuffer.GetAddressOf()))))
    {
        ADD_FAILURE() << "Failed to retrieve TrinityAL DX11 back buffer";
        return false;
    }

    D3D11_TEXTURE2D_DESC backBufferDescription{};
    backBuffer->GetDesc(&backBufferDescription);
    resources.width = backBufferDescription.Width;
    resources.height = backBufferDescription.Height;

    if (FAILED(device->CreateRenderTargetView(backBuffer.Get(), nullptr, resources.renderTargetView.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview render-target view";
        return false;
    }

    D3D11_TEXTURE2D_DESC depthDescription{};
    depthDescription.Width = resources.width;
    depthDescription.Height = resources.height;
    depthDescription.MipLevels = 1;
    depthDescription.ArraySize = 1;
    depthDescription.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthDescription.SampleDesc.Count = 1;
    depthDescription.Usage = D3D11_USAGE_DEFAULT;
    depthDescription.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    if (FAILED(device->CreateTexture2D(&depthDescription, nullptr, resources.depthTexture.GetAddressOf())) ||
        FAILED(device->CreateDepthStencilView(resources.depthTexture.Get(), nullptr, resources.depthStencilView.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview depth buffer";
        return false;
    }
    return true;
}

bool AssetProcessor::CreatePreviewResources(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    IDXGISwapChain* swapChain,
    const ObjMesh& mesh,
    const std::string& albedoPath,
    const std::string& normalPath,
    const std::string& pgsPath,
    const std::vector<std::string>& environmentPaths,
    const std::string& materialManifestPath,
    PreviewResources& resources)
{
    if (!CreatePreviewTargets(device, swapChain, resources)) return false;

    ComPtr<ID3DBlob> vertexBlob;
    ComPtr<ID3DBlob> pixelBlob;
    ComPtr<ID3DBlob> backgroundVertexBlob;
    ComPtr<ID3DBlob> backgroundPixelBlob;
    if (!m_shaderLibrary.Compile("VSMain", "vs_5_0", vertexBlob) ||
        !m_shaderLibrary.Compile("PSMain", "ps_5_0", pixelBlob) ||
        !m_shaderLibrary.Compile("VSBackground", "vs_5_0", backgroundVertexBlob) ||
        !m_shaderLibrary.Compile("PSBackground", "ps_5_0", backgroundPixelBlob))
    {
        return false;
    }

    if (FAILED(device->CreateVertexShader(vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), nullptr, resources.vertexShader.GetAddressOf())) ||
        FAILED(device->CreatePixelShader(pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), nullptr, resources.pixelShader.GetAddressOf())) ||
        FAILED(device->CreateVertexShader(backgroundVertexBlob->GetBufferPointer(), backgroundVertexBlob->GetBufferSize(), nullptr, resources.backgroundVertexShader.GetAddressOf())) ||
        FAILED(device->CreatePixelShader(backgroundPixelBlob->GetBufferPointer(), backgroundPixelBlob->GetBufferSize(), nullptr, resources.backgroundPixelShader.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview shaders";
        return false;
    }

    const D3D11_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, position)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, normal)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, uv)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, stretchHint)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, stretchAxis)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 3, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, repairUvCenter)), D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 4, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(Vertex, repairUvScale)), D3D11_INPUT_PER_VERTEX_DATA, 0},
    };

    if (FAILED(device->CreateInputLayout(
            inputElements,
            static_cast<UINT>(sizeof(inputElements) / sizeof(inputElements[0])),
            vertexBlob->GetBufferPointer(),
            vertexBlob->GetBufferSize(),
            resources.inputLayout.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR preview input layout";
        return false;
    }

    if (mesh.vertices.empty() || mesh.indices.empty() ||
        mesh.vertices.size() * sizeof(Vertex) > static_cast<size_t>(std::numeric_limits<UINT>::max()) ||
        mesh.indices.size() * sizeof(uint32_t) > static_cast<size_t>(std::numeric_limits<UINT>::max()))
    {
        ADD_FAILURE() << "OBJ mesh is empty or exceeds D3D11 buffer limits";
        return false;
    }

    resources.indexCount = static_cast<uint32_t>(mesh.indices.size());

    D3D11_BUFFER_DESC vertexBufferDescription{};
    vertexBufferDescription.ByteWidth = static_cast<UINT>(mesh.vertices.size() * sizeof(Vertex));
    vertexBufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    vertexBufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vertexData{};
    vertexData.pSysMem = mesh.vertices.data();

    D3D11_BUFFER_DESC indexBufferDescription{};
    indexBufferDescription.ByteWidth = static_cast<UINT>(mesh.indices.size() * sizeof(uint32_t));
    indexBufferDescription.Usage = D3D11_USAGE_IMMUTABLE;
    indexBufferDescription.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA indexData{};
    indexData.pSysMem = mesh.indices.data();

    D3D11_BUFFER_DESC constantBufferDescription{};
    constantBufferDescription.ByteWidth = static_cast<UINT>(sizeof(SceneConstants));
    constantBufferDescription.Usage = D3D11_USAGE_DYNAMIC;
    constantBufferDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constantBufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    if (FAILED(device->CreateBuffer(&vertexBufferDescription, &vertexData, resources.vertexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&indexBufferDescription, &indexData, resources.indexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&constantBufferDescription, nullptr, resources.constantBuffer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create NSAMDR OBJ preview buffers";
        return false;
    }

    D3D11_RASTERIZER_DESC rasterizerDescription{};
    rasterizerDescription.FillMode = D3D11_FILL_SOLID;
    rasterizerDescription.CullMode = D3D11_CULL_NONE;
    rasterizerDescription.DepthClipEnable = TRUE;
    if (FAILED(device->CreateRasterizerState(&rasterizerDescription, resources.solidRasterizer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create solid rasterizer state";
        return false;
    }
    rasterizerDescription.FillMode = D3D11_FILL_WIREFRAME;
    if (FAILED(device->CreateRasterizerState(&rasterizerDescription, resources.wireRasterizer.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create wireframe rasterizer state";
        return false;
    }

    D3D11_BLEND_DESC blendDescription{};
    blendDescription.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device->CreateBlendState(&blendDescription, resources.opaqueBlendState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create opaque blend state";
        return false;
    }
    blendDescription.RenderTarget[0].BlendEnable = TRUE;
    blendDescription.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDescription.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDescription.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDescription.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDescription.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    blendDescription.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    if (FAILED(device->CreateBlendState(&blendDescription, resources.alphaBlendState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create alpha blend state";
        return false;
    }
    blendDescription.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
    blendDescription.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    if (FAILED(device->CreateBlendState(&blendDescription, resources.additiveBlendState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create additive blend state";
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC depthStateDescription{};
    depthStateDescription.DepthEnable = TRUE;
    depthStateDescription.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthStateDescription.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    depthStateDescription.StencilEnable = FALSE;
    if (FAILED(device->CreateDepthStencilState(&depthStateDescription, resources.depthWriteState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create depth-write state";
        return false;
    }
    depthStateDescription.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    if (FAILED(device->CreateDepthStencilState(&depthStateDescription, resources.depthReadState.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create depth-read state";
        return false;
    }

    D3D11_SAMPLER_DESC samplerDescription{};
    samplerDescription.Filter = D3D11_FILTER_ANISOTROPIC;
    samplerDescription.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDescription.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDescription.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDescription.MaxAnisotropy = 16;
    samplerDescription.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDescription.MinLOD = 0.0f;
    samplerDescription.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(device->CreateSamplerState(&samplerDescription, resources.textureSampler.GetAddressOf())))
    {
        ADD_FAILURE() << "Failed to create albedo sampler";
        return false;
    }

    bool success = true;
    if (albedoPath.empty())
    {
        success = CreateFallbackTexture(device, resources);
    }
    else
    {
        success = LoadWicTexture(
            device, context, albedoPath, true, resources.albedoView, resources.textureWidth, resources.textureHeight, "albedo image");
        resources.hasExternalAlbedo = success;
    }
    if (!success) return false;

    if (!normalPath.empty())
    {
        resources.hasNormalMap = LoadWicTexture(
            device, context, normalPath, false, resources.normalView, resources.normalWidth, resources.normalHeight, "normal map");
        if (!resources.hasNormalMap) return false;
    }
    if (!pgsPath.empty())
    {
        resources.hasPgsMap = LoadWicTexture(
            device, context, pgsPath, false, resources.pgsView, resources.pgsWidth, resources.pgsHeight, "PGS material map");
        if (!resources.hasPgsMap) return false;
    }
    for (const std::string& environmentPath : environmentPaths)
    {
        EnvironmentGpu environment;
        environment.path = environmentPath;
        environment.label = FileLabel(environmentPath);
        resources.environments.push_back(std::move(environment));
    }
    resources.hasEnvironment = !resources.environments.empty();

    std::vector<AreaMaterialSource> materialSources;
    std::string materialError;
    if (!LoadAreaMaterialSources(materialManifestPath, materialSources, materialError))
    {
        ADD_FAILURE() << materialError;
        return false;
    }
    for (const AreaMaterialSource& source : materialSources)
    {
        for (const ObjDrawRange& range : mesh.drawRanges)
        {
            if (range.groupIndex != source.groupIndex) continue;
            AreaMaterialGpu material;
            material.source = source;
            material.drawRange = range;
            material.albedoView = resources.albedoView;
            material.normalView = resources.normalView;
            material.pgsView = resources.pgsView;
            material.albedoWidth = resources.textureWidth;
            material.albedoHeight = resources.textureHeight;
            material.hasAlbedo = resources.hasExternalAlbedo;
            material.hasNormal = resources.hasNormalMap;
            material.hasPgs = resources.hasPgsMap;
            if (!source.albedoPath.empty())
            {
                material.hasAlbedo = LoadWicTexture(
                    device, context, source.albedoPath, true, material.albedoView, material.albedoWidth, material.albedoHeight, "SOF area albedo");
                if (!material.hasAlbedo) return false;
            }
            if (!source.normalPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasNormal = LoadWicTexture(
                    device, context, source.normalPath, false, material.normalView, width, height, "SOF area normal map");
                if (!material.hasNormal) return false;
            }
            if (!source.pgsPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasPgs = LoadWicTexture(
                    device, context, source.pgsPath, false, material.pgsView, width, height, "SOF area material map");
                if (!material.hasPgs) return false;
            }
            if (!source.glowPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasGlow = LoadWicTexture(
                    device, context, source.glowPath, true, material.glowView, width, height, "SOF area glow map");
                if (!material.hasGlow) return false;
            }
            if (!source.dirtPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasDirt = LoadWicTexture(
                    device, context, source.dirtPath, false, material.dirtView, width, height, "SOF area dirt map");
                if (!material.hasDirt) return false;
            }
            if (!source.aoPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasAo = LoadWicTexture(
                    device, context, source.aoPath, false, material.aoView, width, height, "SOF area AO map");
                if (!material.hasAo) return false;
            }
            if (!source.paintMaskPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasPaintMask = LoadWicTexture(
                    device, context, source.paintMaskPath, false, material.paintMaskView, width, height, "SOF area paint mask");
                if (!material.hasPaintMask) return false;
            }
            if (!source.roughnessMapPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasRoughnessMap = LoadWicTexture(
                    device, context, source.roughnessMapPath, false, material.roughnessMapView, width, height, "SOF area roughness map");
                if (!material.hasRoughnessMap) return false;
            }
            resources.areaMaterials.push_back(std::move(material));
        }
    }
    std::stable_sort(resources.areaMaterials.begin(), resources.areaMaterials.end(), [](const AreaMaterialGpu& left, const AreaMaterialGpu& right) {
        return static_cast<int>(left.source.pass) < static_cast<int>(right.source.pass);
    });
    resources.baselineComplete = !resources.areaMaterials.empty();
    resources.baselineUnresolvedCount = 0;
    for (const AreaMaterialGpu& material : resources.areaMaterials)
    {
        const bool effectiveSemantics = material.source.semanticComplete && material.hasAlbedo && material.hasNormal && material.hasPgs;
        const bool roughnessResolved = material.hasRoughnessMap || material.source.shaderFamily == ShaderFamily::LegacyPgs;
        const bool coreTextures = effectiveSemantics && roughnessResolved;
        resources.baselineComplete = resources.baselineComplete && material.source.baselineComplete && coreTextures;
        resources.baselineUnresolvedCount += material.source.unresolvedCount + (coreTextures ? 0 : 1);
    }
    return true;
}

bool AssetProcessor::CreateCandidateMeshBuffers(
    ID3D11Device* device,
    const ObjMesh& mesh,
    CandidateAssetGpu& candidate,
    std::string& error)
{
    if (mesh.vertices.empty() || mesh.indices.empty() ||
        mesh.vertices.size() * sizeof(Vertex) > static_cast<size_t>(std::numeric_limits<UINT>::max()) ||
        mesh.indices.size() * sizeof(uint32_t) > static_cast<size_t>(std::numeric_limits<UINT>::max()))
    {
        error = "Candidate OBJ mesh is empty or exceeds D3D11 buffer limits";
        return false;
    }

    D3D11_BUFFER_DESC vertexDescription{};
    vertexDescription.ByteWidth = static_cast<UINT>(mesh.vertices.size() * sizeof(Vertex));
    vertexDescription.Usage = D3D11_USAGE_IMMUTABLE;
    vertexDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vertexData{};
    vertexData.pSysMem = mesh.vertices.data();

    D3D11_BUFFER_DESC indexDescription{};
    indexDescription.ByteWidth = static_cast<UINT>(mesh.indices.size() * sizeof(uint32_t));
    indexDescription.Usage = D3D11_USAGE_IMMUTABLE;
    indexDescription.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA indexData{};
    indexData.pSysMem = mesh.indices.data();

    if (FAILED(device->CreateBuffer(&vertexDescription, &vertexData, candidate.vertexBuffer.GetAddressOf())) ||
        FAILED(device->CreateBuffer(&indexDescription, &indexData, candidate.indexBuffer.GetAddressOf())))
    {
        error = "Failed to create candidate OBJ vertex/index buffers";
        return false;
    }
    candidate.indexCount = static_cast<uint32_t>(mesh.indices.size());
    return true;
}

bool AssetProcessor::LoadCandidateAsset(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const std::string& label,
    const std::string& objPath,
    const std::string& materialManifestPath,
    CandidateAssetGpu& candidate)
{
    candidate = CandidateAssetGpu{};
    candidate.label = label;
    candidate.objPath = objPath;
    candidate.materialManifestPath = materialManifestPath;
    if (objPath.empty() || materialManifestPath.empty())
    {
        candidate.status = "candidate environment paths were not supplied";
        return true;
    }

    std::string meshError;
    if (!m_meshProcessor.LoadObjMesh(objPath, candidate.mesh, meshError))
    {
        candidate.status = "OBJ load failed: " + meshError;
        return true;
    }
    std::string bufferError;
    if (!CreateCandidateMeshBuffers(device, candidate.mesh, candidate, bufferError))
    {
        candidate.status = bufferError;
        return true;
    }

    std::vector<AreaMaterialSource> sources;
    std::string materialError;
    if (!LoadAreaMaterialSources(materialManifestPath, sources, materialError))
    {
        candidate.status = materialError;
        return true;
    }

    // Candidate atlases are 4K and the same physical packed texture can be
    // referenced by several semantic columns and draw groups. Cache by physical
    // path so each packed GPU texture is created only once. The load order is
    // deliberate: AR is first opened through its albedo sRGB view (alpha stays
    // linear), while NO/PMDG are first opened through linear semantic views.
    struct CachedCandidateTexture
    {
        ComPtr<ID3D11ShaderResourceView> view;
        uint32_t width = 0;
        uint32_t height = 0;
    };
    std::map<std::string, CachedCandidateTexture> textureCache;
    auto loadCached = [&](const std::string& path, bool srgb,
                          ComPtr<ID3D11ShaderResourceView>& view,
                          uint32_t& width, uint32_t& height,
                          const char* textureLabel) -> bool
    {
        if (path.empty()) return false;
        const std::string key = path;
        const auto existing = textureCache.find(key);
        if (existing != textureCache.end())
        {
            view = existing->second.view;
            width = existing->second.width;
            height = existing->second.height;
            return true;
        }

        CachedCandidateTexture loaded;
        if (!LoadWicTexture(device, context, path, srgb, loaded.view,
                            loaded.width, loaded.height, textureLabel))
        {
            return false;
        }
        view = loaded.view;
        width = loaded.width;
        height = loaded.height;
        textureCache.emplace(key, std::move(loaded));
        return true;
    };

    for (const AreaMaterialSource& source : sources)
    {
        for (const ObjDrawRange& range : candidate.mesh.drawRanges)
        {
            if (range.groupIndex != source.groupIndex) continue;
            AreaMaterialGpu material;
            material.source = source;
            material.drawRange = range;
            if (!source.albedoPath.empty())
            {
                material.hasAlbedo = loadCached(
                    source.albedoPath, true, material.albedoView,
                    material.albedoWidth, material.albedoHeight,
                    "candidate area albedo");
                if (!material.hasAlbedo)
                {
                    candidate.status = "failed to load candidate albedo: " + source.albedoPath;
                    return true;
                }
            }
            if (!source.normalPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasNormal = loadCached(
                    source.normalPath, false, material.normalView, width, height,
                    "candidate area normal map");
                if (!material.hasNormal)
                {
                    candidate.status = "failed to load candidate normal map: " + source.normalPath;
                    return true;
                }
                candidate.maximumTextureWidth = std::max(candidate.maximumTextureWidth, width);
                candidate.maximumTextureHeight = std::max(candidate.maximumTextureHeight, height);
            }
            if (!source.pgsPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasPgs = loadCached(
                    source.pgsPath, false, material.pgsView, width, height,
                    "candidate area material map");
                if (!material.hasPgs)
                {
                    candidate.status = "failed to load candidate material map: " + source.pgsPath;
                    return true;
                }
                candidate.maximumTextureWidth = std::max(candidate.maximumTextureWidth, width);
                candidate.maximumTextureHeight = std::max(candidate.maximumTextureHeight, height);
            }
            if (!source.glowPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasGlow = loadCached(
                    source.glowPath, true, material.glowView, width, height,
                    "candidate area glow map");
                if (!material.hasGlow)
                {
                    candidate.status = "failed to load candidate glow map: " + source.glowPath;
                    return true;
                }
            }
            if (!source.dirtPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasDirt = loadCached(
                    source.dirtPath, false, material.dirtView, width, height,
                    "candidate area dirt map");
                if (!material.hasDirt)
                {
                    candidate.status = "failed to load candidate dirt map: " + source.dirtPath;
                    return true;
                }
            }
            if (!source.aoPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasAo = loadCached(
                    source.aoPath, false, material.aoView, width, height,
                    "candidate area AO map");
                if (!material.hasAo)
                {
                    candidate.status = "failed to load candidate AO map: " + source.aoPath;
                    return true;
                }
            }
            if (!source.paintMaskPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasPaintMask = loadCached(
                    source.paintMaskPath, false, material.paintMaskView, width, height,
                    "candidate area paint mask");
                if (!material.hasPaintMask)
                {
                    candidate.status = "failed to load candidate paint mask: " + source.paintMaskPath;
                    return true;
                }
            }
            if (!source.roughnessMapPath.empty())
            {
                uint32_t width = 0, height = 0;
                material.hasRoughnessMap = loadCached(
                    source.roughnessMapPath, false, material.roughnessMapView, width, height,
                    "candidate area roughness map");
                if (!material.hasRoughnessMap)
                {
                    candidate.status = "failed to load candidate roughness map: " + source.roughnessMapPath;
                    return true;
                }
            }
            candidate.maximumTextureWidth = std::max(candidate.maximumTextureWidth, material.albedoWidth);
            candidate.maximumTextureHeight = std::max(candidate.maximumTextureHeight, material.albedoHeight);
            candidate.areaMaterials.push_back(std::move(material));
        }
    }

    std::stable_sort(candidate.areaMaterials.begin(), candidate.areaMaterials.end(), [](const AreaMaterialGpu& left, const AreaMaterialGpu& right) {
        return static_cast<int>(left.source.pass) < static_cast<int>(right.source.pass);
    });
    if (candidate.areaMaterials.empty())
    {
        candidate.status = "candidate material manifest did not map to any OBJ draw groups";
        return true;
    }
    candidate.available = true;
    candidate.status = "loaded; " + std::to_string(textureCache.size()) + " unique GPU textures";
    return true;
}

} // namespace nsamdr
