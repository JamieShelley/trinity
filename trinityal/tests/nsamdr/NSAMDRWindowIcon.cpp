#include "StdAfx.h"
#include "NSAMDRWindowIcon.h"
#include "NSAMDRPreviewResource.h"

namespace nsamdr
{
namespace
{
std::string Win32Error(const char* operation)
{
    const DWORD code = GetLastError();
    std::ostringstream stream;
    stream << operation << " failed with Win32 error " << code;
    return stream.str();
}
}

WindowIcon::~WindowIcon()
{
    if (m_window && IsWindow(m_window))
    {
        SendMessageW(m_window, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(m_previousLarge));
        SendMessageW(m_window, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(m_previousSmall));
        if (m_classLargeChanged)
        {
            SetClassLongPtrW(m_window, GCLP_HICON, reinterpret_cast<LONG_PTR>(m_previousClassLarge));
        }
        if (m_classSmallChanged)
        {
            SetClassLongPtrW(m_window, GCLP_HICONSM, reinterpret_cast<LONG_PTR>(m_previousClassSmall));
        }
    }
    if (m_large) DestroyIcon(m_large);
    if (m_small) DestroyIcon(m_small);
}

bool WindowIcon::Apply(HWND window, std::string& error)
{
    if (!window)
    {
        error = "NSAMDR window handle is null";
        return false;
    }

    HINSTANCE instance = GetModuleHandleW(nullptr);
    if (!instance)
    {
        error = Win32Error("GetModuleHandleW");
        return false;
    }

    m_large = reinterpret_cast<HICON>(LoadImageW(
        instance,
        MAKEINTRESOURCEW(IDI_NSAMDR_PREVIEW),
        IMAGE_ICON,
        GetSystemMetrics(SM_CXICON),
        GetSystemMetrics(SM_CYICON),
        LR_DEFAULTCOLOR));
    if (!m_large)
    {
        error = Win32Error("LoadImageW large NSAMDR icon");
        return false;
    }

    m_small = reinterpret_cast<HICON>(LoadImageW(
        instance,
        MAKEINTRESOURCEW(IDI_NSAMDR_PREVIEW),
        IMAGE_ICON,
        GetSystemMetrics(SM_CXSMICON),
        GetSystemMetrics(SM_CYSMICON),
        LR_DEFAULTCOLOR));
    if (!m_small)
    {
        error = Win32Error("LoadImageW small NSAMDR icon");
        return false;
    }

    m_window = window;
    m_previousLarge = reinterpret_cast<HICON>(
        SendMessageW(window, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(m_large)));
    m_previousSmall = reinterpret_cast<HICON>(
        SendMessageW(window, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(m_small)));

    // WM_SETICON updates the individual window. Windows may still source the
    // taskbar/title icon from the registered window class, so update both class
    // slots as well and restore them when the preview exits. A zero previous
    // value is valid and does not indicate failure.
    SetLastError(ERROR_SUCCESS);
    m_previousClassLarge = reinterpret_cast<HICON>(
        SetClassLongPtrW(window, GCLP_HICON, reinterpret_cast<LONG_PTR>(m_large)));
    m_classLargeChanged = GetLastError() == ERROR_SUCCESS;

    SetLastError(ERROR_SUCCESS);
    m_previousClassSmall = reinterpret_cast<HICON>(
        SetClassLongPtrW(window, GCLP_HICONSM, reinterpret_cast<LONG_PTR>(m_small)));
    m_classSmallChanged = GetLastError() == ERROR_SUCCESS;

    SendMessageW(window, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(m_large));
    SendMessageW(window, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(m_small));
    SetWindowPos(
        window, nullptr, 0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    RedrawWindow(window, nullptr, nullptr, RDW_INVALIDATE | RDW_FRAME | RDW_UPDATENOW);
    return true;
}
} // namespace nsamdr
