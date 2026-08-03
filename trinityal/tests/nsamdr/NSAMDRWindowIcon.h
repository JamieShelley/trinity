#pragma once

#include "NSAMDRPreviewPlatform.h"

namespace nsamdr
{
class WindowIcon final
{
public:
    WindowIcon() = default;
    ~WindowIcon();

    bool Apply(HWND window, std::string& error);

private:
    HWND m_window = nullptr;
    HICON m_large = nullptr;
    HICON m_small = nullptr;
    HICON m_previousLarge = nullptr;
    HICON m_previousSmall = nullptr;
    HICON m_previousClassLarge = nullptr;
    HICON m_previousClassSmall = nullptr;
    bool m_classLargeChanged = false;
    bool m_classSmallChanged = false;
};
} // namespace nsamdr
