#pragma once

// NSAMDR V10.8.5 standalone preview precompiled-header replacement.
// Deliberately contains no TrinityAL headers and therefore no Nsight Aftermath
// dependency.  The diagnostic viewer uses D3D11 directly.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include <cstdint>
#include <string>

#ifndef TRINITY_DIRECTX11
#define TRINITY_DIRECTX11 2
#endif
#ifndef TRINITY_PLATFORM
#define TRINITY_PLATFORM TRINITY_DIRECTX11
#endif

#include <gtest/gtest.h>
