#pragma once

#include <cstdint>

namespace nsamdr
{
enum class NSAMDRQuality : int
{
    Off = 0,
    Balanced = 1,
    High = 2,
    Ultra = 3
};

inline const char* NSAMDRQualityName(NSAMDRQuality quality)
{
    switch (quality)
    {
    case NSAMDRQuality::Off: return "Off";
    case NSAMDRQuality::Balanced: return "Balanced";
    case NSAMDRQuality::High: return "High";
    case NSAMDRQuality::Ultra: return "Ultra";
    default: return "Off";
    }
}

// Stable client-facing policy. Carbon/EVE UI should select only the quality
// level; Trinity/NSAMDR owns the detailed policy mapping behind that value.
struct NSAMDRGraphicsSettings
{
    NSAMDRQuality quality = NSAMDRQuality::Off;
    bool useDiskCache = true;
    bool reconstructShips = false;
    bool reconstructStructures = false;
    bool preserveMicrodetail = false;
    bool sharedPbrBoundaryAlignment = false;
    uint32_t targetScale = 1U;
    uint64_t cacheBudgetBytes = 0ULL;
    float minimumConfidence = 1.0f;

    bool Enabled() const { return quality != NSAMDRQuality::Off; }
};

inline NSAMDRGraphicsSettings ResolveNSAMDRGraphicsSettings(NSAMDRQuality quality)
{
    NSAMDRGraphicsSettings settings{};
    settings.quality = quality;
    switch (quality)
    {
    case NSAMDRQuality::Off:
        settings.useDiskCache = true;
        settings.reconstructShips = false;
        settings.reconstructStructures = false;
        settings.preserveMicrodetail = false;
        settings.sharedPbrBoundaryAlignment = false;
        settings.targetScale = 1U;
        settings.cacheBudgetBytes = 0ULL;
        settings.minimumConfidence = 1.0f;
        break;
    case NSAMDRQuality::Balanced:
        settings.reconstructShips = true;
        settings.reconstructStructures = false;
        settings.preserveMicrodetail = false;
        settings.sharedPbrBoundaryAlignment = true;
        settings.targetScale = 2U;
        settings.cacheBudgetBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        settings.minimumConfidence = 0.92f;
        break;
    case NSAMDRQuality::High:
        settings.reconstructShips = true;
        settings.reconstructStructures = true;
        settings.preserveMicrodetail = true;
        settings.sharedPbrBoundaryAlignment = true;
        settings.targetScale = 2U;
        settings.cacheBudgetBytes = 12ULL * 1024ULL * 1024ULL * 1024ULL;
        settings.minimumConfidence = 0.88f;
        break;
    case NSAMDRQuality::Ultra:
        settings.reconstructShips = true;
        settings.reconstructStructures = true;
        settings.preserveMicrodetail = true;
        settings.sharedPbrBoundaryAlignment = true;
        settings.targetScale = 4U;
        settings.cacheBudgetBytes = 30ULL * 1024ULL * 1024ULL * 1024ULL;
        settings.minimumConfidence = 0.85f;
        break;
    }
    return settings;
}

enum class NSAMDRBackend : int
{
    None = 0,
    CUDA = 1,
    DirectML = 2
};

enum class NSAMDRRuntimeStatus : int
{
    Disabled = 0,
    Unavailable,
    Ready,
    LoadingModel,
    Reconstructing,
    Cached,
    Throttled,
    Error
};

struct NSAMDRCapabilities
{
    bool available = false;
    NSAMDRQuality maximumQuality = NSAMDRQuality::Off;
    NSAMDRBackend backend = NSAMDRBackend::None;
    uint64_t dedicatedVideoMemoryBytes = 0ULL;
};
}
