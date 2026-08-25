// Copyright © 2026
// NSAMDR V10.8.5 standalone Granny-free DX11 preview host.
//
// This executable intentionally does not create or link TrinityAL_dx11.  The
// NSAMDR diagnostic renderer already consumes raw ID3D11Device/context/swapchain
// interfaces, so rebuilding the complete TrinityAL library only pulled the
// unrelated legacy Nsight Aftermath integration into the preview path.

#include "StdAfx.h"
#include "NSAMDRPreviewApplication.h"

#include <filesystem>
#include <wincodec.h>
#include <wrl/client.h>

bool g_exitInteractiveOnCharacter = true;

namespace
{
using Microsoft::WRL::ComPtr;

bool g_comInitialized = false;

LRESULT CALLBACK StandaloneWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, message, wParam, lParam);
    }
}

std::wstring WidenUtf8(const std::string& value)
{
    if (value.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    if (count <= 1) return {};
    std::wstring result(static_cast<size_t>(count), L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, result.data(), count) <= 0) return {};
    if (!result.empty() && result.back() == L'\0') result.pop_back();
    return result;
}

bool SaveSwapChainPng(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    IDXGISwapChain* swapChain,
    const std::string& path)
{
    if (!device || !context || !swapChain || path.empty()) return false;

    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(backBuffer.GetAddressOf())))) return false;

    D3D11_TEXTURE2D_DESC sourceDesc{};
    backBuffer->GetDesc(&sourceDesc);

    D3D11_TEXTURE2D_DESC stagingDesc = sourceDesc;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;
    stagingDesc.ArraySize = 1;
    stagingDesc.MipLevels = 1;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.SampleDesc.Quality = 0;

    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&stagingDesc, nullptr, staging.GetAddressOf()))) return false;

    context->CopyResource(staging.Get(), backBuffer.Get());
    context->Flush();

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;

    bool success = false;
    do
    {
        const std::filesystem::path outputPath = std::filesystem::u8path(path);
        std::error_code ec;
        if (outputPath.has_parent_path())
        {
            std::filesystem::create_directories(outputPath.parent_path(), ec);
        }

        const std::wstring widePath = WidenUtf8(path);
        if (widePath.empty()) break;

        ComPtr<IWICImagingFactory> factory;
        if (FAILED(CoCreateInstance(
                CLSID_WICImagingFactory,
                nullptr,
                CLSCTX_INPROC_SERVER,
                IID_PPV_ARGS(factory.GetAddressOf())))) break;

        ComPtr<IWICStream> stream;
        if (FAILED(factory->CreateStream(stream.GetAddressOf()))) break;
        if (FAILED(stream->InitializeFromFilename(widePath.c_str(), GENERIC_WRITE))) break;

        ComPtr<IWICBitmapEncoder> encoder;
        if (FAILED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, encoder.GetAddressOf()))) break;
        if (FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) break;

        ComPtr<IWICBitmapFrameEncode> frame;
        ComPtr<IPropertyBag2> properties;
        if (FAILED(encoder->CreateNewFrame(frame.GetAddressOf(), properties.GetAddressOf()))) break;
        if (FAILED(frame->Initialize(properties.Get()))) break;
        if (FAILED(frame->SetSize(sourceDesc.Width, sourceDesc.Height))) break;

        WICPixelFormatGUID pixelFormat = GUID_WICPixelFormat32bppRGBA;
        if (sourceDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
            sourceDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)
        {
            pixelFormat = GUID_WICPixelFormat32bppBGRA;
        }
        else if (sourceDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
                 sourceDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
        {
            break;
        }

        if (FAILED(frame->SetPixelFormat(&pixelFormat))) break;
        const UINT imageBytes = static_cast<UINT>(mapped.RowPitch * sourceDesc.Height);
        if (FAILED(frame->WritePixels(
                sourceDesc.Height,
                mapped.RowPitch,
                imageBytes,
                static_cast<BYTE*>(mapped.pData)))) break;
        if (FAILED(frame->Commit())) break;
        if (FAILED(encoder->Commit())) break;
        success = true;
    } while (false);

    context->Unmap(staging.Get(), 0);
    return success;
}

struct StandaloneDx11Host
{
    HINSTANCE instance = GetModuleHandleW(nullptr);
    HWND window = nullptr;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapChain;
    ATOM windowClass = 0;

    ~StandaloneDx11Host()
    {
        if (context) context->ClearState();
        if (swapChain) swapChain->SetFullscreenState(FALSE, nullptr);
        swapChain.Reset();
        context.Reset();
        device.Reset();
        if (window && IsWindow(window)) DestroyWindow(window);
        if (windowClass != 0) UnregisterClassW(L"NSAMDRStandalonePreviewWindow", instance);
        if (g_comInitialized)
        {
            CoUninitialize();
            g_comInitialized = false;
        }
    }

    bool Initialize()
    {
        const HRESULT comResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        g_comInitialized = SUCCEEDED(comResult);

        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
        wc.lpfnWndProc = StandaloneWindowProc;
        wc.hInstance = instance;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.lpszClassName = L"NSAMDRStandalonePreviewWindow";
        windowClass = RegisterClassExW(&wc);
        if (windowClass == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;

        RECT rect{0, 0, 1440, 900};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
        window = CreateWindowExW(
            0,
            wc.lpszClassName,
            L"NSAMDR — Standalone DX11 Preview",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            nullptr,
            nullptr,
            instance,
            nullptr);
        if (!window) return false;

        DXGI_SWAP_CHAIN_DESC swapDesc{};
        swapDesc.BufferCount = 1;
        swapDesc.BufferDesc.Width = 1440;
        swapDesc.BufferDesc.Height = 900;
        swapDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapDesc.OutputWindow = window;
        swapDesc.SampleDesc.Count = 1;
        swapDesc.Windowed = TRUE;
        swapDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        const D3D_FEATURE_LEVEL requestedLevel = D3D_FEATURE_LEVEL_11_0;
        D3D_FEATURE_LEVEL createdLevel = D3D_FEATURE_LEVEL_11_0;
        const UINT deviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        const HRESULT result = D3D11CreateDeviceAndSwapChain(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            deviceFlags,
            &requestedLevel,
            1,
            D3D11_SDK_VERSION,
            &swapDesc,
            swapChain.GetAddressOf(),
            device.GetAddressOf(),
            &createdLevel,
            context.GetAddressOf());
        if (FAILED(result)) return false;

        ShowWindow(window, SW_SHOW);
        UpdateWindow(window);
        return true;
    }

    bool Resize(uint32_t width, uint32_t height)
    {
        if (!swapChain || width < 1 || height < 1) return false;
        return SUCCEEDED(swapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0));
    }

    bool Present()
    {
        return swapChain && SUCCEEDED(swapChain->Present(1, 0));
    }

    void RunLoop(const std::function<void()>& frame)
    {
        bool quit = false;
        while (!quit && window && IsWindow(window))
        {
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                if (message.message == WM_QUIT)
                {
                    quit = true;
                    break;
                }
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            if (quit) break;
            if (IsIconic(window))
            {
                WaitMessage();
                continue;
            }
            frame();
        }
    }
};
} // namespace

TEST(NSAMDRRendering, RealObjShipPreview)
{
    StandaloneDx11Host hostContext;
    ASSERT_TRUE(hostContext.Initialize())
        << "Failed to create the standalone D3D11 window/device/swapchain.";

    nsamdr::PreviewHost host;
    host.device = hostContext.device.Get();
    host.context = hostContext.context.Get();
    host.swapChain = hostContext.swapChain.Get();
    host.window = hostContext.window;
    host.resize = [&hostContext](uint32_t width, uint32_t height) {
        return hostContext.Resize(width, height);
    };
    host.present = [&hostContext]() {
        return hostContext.Present();
    };
    host.runLoop = [&hostContext](const std::function<void()>& frame) {
        hostContext.RunLoop(frame);
    };
    host.screenshot = [&hostContext](const std::string& path) {
        if (!SaveSwapChainPng(
                hostContext.device.Get(),
                hostContext.context.Get(),
                hostContext.swapChain.Get(),
                path))
        {
            ADD_FAILURE() << "Failed to save NSAMDR standalone preview screenshot: " << path;
        }
    };

    nsamdr::PreviewApplication application(std::move(host));
    application.Run();
}
