#include "StdAfx.h"
#include "NSAMDRInputController.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{

InputController* InputController::s_active = nullptr;

InputController::InputController(CameraController& cameraController)
    : m_cameraController(cameraController)
{
}

InputController::~InputController()
{
    Detach();
}

bool InputController::Attach(HWND window, PreviewState& state)
{
    if (s_active != nullptr && s_active != this) return false;
    m_window = window;
    m_state = &state;
    m_inputFocused = false;
    m_blockMouseUntilRelease = false;
    m_keyWasDown.fill(false);
    g_exitInteractiveOnCharacter = false;
    s_active = this;
    m_previousWindowProc = reinterpret_cast<WNDPROC>(
        SetWindowLongPtr(window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&InputController::WindowProcThunk)));
    if (m_previousWindowProc == nullptr)
    {
        s_active = nullptr;
        m_window = nullptr;
        m_state = nullptr;
        g_exitInteractiveOnCharacter = true;
        return false;
    }
    RefreshPreviewInputFocus(window);
    return true;
}

void InputController::Detach()
{
    ResetSceneInput(m_window);
    if (m_window != nullptr && IsWindow(m_window) && m_previousWindowProc != nullptr)
    {
        SetWindowLongPtr(m_window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(m_previousWindowProc));
    }
    m_previousWindowProc = nullptr;
    m_state = nullptr;
    m_window = nullptr;
    m_inputFocused = false;
    m_blockMouseUntilRelease = false;
    m_keyWasDown.fill(false);
    if (s_active == this) s_active = nullptr;
    g_exitInteractiveOnCharacter = true;
}

void InputController::RefreshFocus()
{
    if (m_window != nullptr) RefreshPreviewInputFocus(m_window);
}

void InputController::Reset()
{
    ResetSceneInput(m_window);
}

bool InputController::IsFocused() const
{
    return m_inputFocused && IsPreviewWindowFocused(m_window);
}

bool InputController::ConsumePendingResize(uint32_t& width, uint32_t& height)
{
    if (m_pendingResizeWidth < 64U || m_pendingResizeHeight < 64U) return false;
    width = m_pendingResizeWidth;
    height = m_pendingResizeHeight;
    m_pendingResizeWidth = 0U;
    m_pendingResizeHeight = 0U;
    return true;
}

LRESULT CALLBACK InputController::WindowProcThunk(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    return s_active != nullptr
        ? s_active->WindowProc(hwnd, message, wParam, lParam)
        : DefWindowProc(hwnd, message, wParam, lParam);
}
HWND InputController::GetPreviewRootWindow(HWND hwnd) const
{
    HWND root = GetAncestor(hwnd, GA_ROOT);
    return root != nullptr ? root : hwnd;
}

bool InputController::IsPreviewWindowFocused(HWND hwnd) const
{
    if (hwnd == nullptr || !IsWindow(hwnd)) return false;
    return GetForegroundWindow() == GetPreviewRootWindow(hwnd);
}

bool InputController::IsPreviewMouseInputMessage(UINT message)
{
    return message >= WM_MOUSEFIRST && message <= WM_MOUSELAST;
}

bool InputController::IsPreviewKeyboardInputMessage(UINT message)
{
    return (message >= WM_KEYFIRST && message <= WM_KEYLAST) ||
           message == WM_CHAR ||
           message == WM_DEADCHAR ||
           message == WM_SYSCHAR ||
           message == WM_SYSDEADCHAR ||
           message == WM_UNICHAR;
}

bool InputController::IsPreviewInputMessage(UINT message)
{
    return IsPreviewMouseInputMessage(message) ||
           IsPreviewKeyboardInputMessage(message);
}

bool InputController::AnyMouseButtonPhysicallyDown()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_XBUTTON1) & 0x8000) != 0 ||
           (GetAsyncKeyState(VK_XBUTTON2) & 0x8000) != 0;
}

void InputController::ResetSceneInput(HWND hwnd)
{
    m_leftMouseDown = false;
    m_rightMouseDown = false;
    m_middleMouseDown = false;
    m_sceneMouseGesture = false;
    m_orbitDragging = false;
    m_panDragging = false;

    if (GetCapture() == hwnd)
    {
        ReleaseCapture();
    }

    if (m_state != nullptr)
    {
        m_state->requestFocus = false;
        m_state->requestZoom = false;
        m_state->zoomWheelSteps = 0.0f;
    }
}

void InputController::SynchronizeHotkeyState()
{
    for (size_t index = 0; index < m_keyWasDown.size(); ++index)
    {
        m_keyWasDown[index] = (GetAsyncKeyState(static_cast<int>(index)) & 0x8000) != 0;
    }
}

void InputController::SetPreviewInputFocus(HWND hwnd, bool focused)
{
    if (m_inputFocused == focused) return;

    m_inputFocused = focused;
    if (!focused)
    {
        ResetSceneInput(hwnd);
        m_keyWasDown.fill(false);
        m_blockMouseUntilRelease = false;
        return;
    }

    // A key already held while another application was active must not become
    // a fresh preview hotkey when focus returns.
    SynchronizeHotkeyState();

    // WM_MOUSEACTIVATE sets this before focus changes when the activating click
    // lands in the client area. Preserve it so that click only activates the
    // window; it cannot also orbit, pan, zoom, focus, or operate the UI.
    m_blockMouseUntilRelease = m_blockMouseUntilRelease || AnyMouseButtonPhysicallyDown();
}

void InputController::RefreshPreviewInputFocus(HWND hwnd)
{
    SetPreviewInputFocus(hwnd, IsPreviewWindowFocused(hwnd));
}

bool InputController::MouseMessageHasAnyButtonDown(WPARAM buttonFlags)
{
    return (buttonFlags & (MK_LBUTTON | MK_RBUTTON | MK_MBUTTON | MK_XBUTTON1 | MK_XBUTTON2)) != 0 ||
           AnyMouseButtonPhysicallyDown();
}

bool InputController::IsMousePointInScene(HWND hwnd, int x, int y)
{
    if (m_state == nullptr) return false;
    RECT client{};
    if (!GetClientRect(hwnd, &client)) return false;
    return x >= static_cast<int>(m_state->sceneViewportX) &&
           x < client.right && y >= client.top && y < client.bottom;
}

void InputController::RefreshMouseButtonsFromMove(WPARAM buttonFlags)
{
    // WM_MOUSEMOVE carries the authoritative button chord. GetAsyncKeyState is
    // included as a recovery path when a button transition was consumed by the
    // host window procedure before this subclass saw it.
    m_leftMouseDown = (buttonFlags & MK_LBUTTON) != 0 || (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    m_rightMouseDown = (buttonFlags & MK_RBUTTON) != 0 || (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    m_middleMouseDown = (buttonFlags & MK_MBUTTON) != 0 || (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0;
}

void InputController::UpdateMouseDragMode(HWND hwnd, int x, int y)
{
    const bool chordPan = m_sceneMouseGesture && m_leftMouseDown && m_rightMouseDown;
    const bool shouldPan = m_sceneMouseGesture && (m_middleMouseDown || chordPan);
    const bool shouldOrbit = m_sceneMouseGesture && m_rightMouseDown && !shouldPan;
    const bool modeChanged = shouldPan != m_panDragging || shouldOrbit != m_orbitDragging;

    m_panDragging = shouldPan;
    m_orbitDragging = shouldOrbit;
    if (modeChanged)
    {
        // Rebase the cursor when changing between orbit and pan so the second
        // button cannot inject a stale movement delta. Do not rebase every move.
        m_lastMouseX = x;
        m_lastMouseY = y;
    }

    if (shouldPan || shouldOrbit)
    {
        if (GetCapture() != hwnd) SetCapture(hwnd);
    }
    else if (GetCapture() == hwnd)
    {
        ReleaseCapture();
    }
}

LRESULT InputController::WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    const bool activatingClientClick =
        message == WM_MOUSEACTIVATE &&
        !IsPreviewWindowFocused(hwnd) &&
        LOWORD(lParam) == HTCLIENT;

    switch (message)
    {
    case WM_ACTIVATEAPP:
        SetPreviewInputFocus(hwnd, wParam != FALSE);
        break;
    case WM_ACTIVATE:
        SetPreviewInputFocus(hwnd, LOWORD(wParam) != WA_INACTIVE);
        break;
    case WM_SETFOCUS:
        SetPreviewInputFocus(hwnd, true);
        break;
    case WM_KILLFOCUS:
        SetPreviewInputFocus(hwnd, false);
        break;
    default:
        RefreshPreviewInputFocus(hwnd);
        break;
    }

    // A click in an inactive client area should activate the preview only. The
    // same physical click must not also manipulate the camera or ImGui controls.
    if (activatingClientClick)
    {
        m_blockMouseUntilRelease = true;
    }

    const bool inputFocused = m_inputFocused && IsPreviewWindowFocused(hwnd);
    const bool mouseInput = IsPreviewMouseInputMessage(message);
    const bool previewInput = IsPreviewInputMessage(message);

    // Consume the complete mouse gesture that activated the window. Clearing the
    // block is deferred until every mouse button has been released.
    const bool mouseWasBlocked = mouseInput && inputFocused && m_blockMouseUntilRelease;
    if (mouseWasBlocked)
    {
        if ((message == WM_MOUSEMOVE ||
             message == WM_LBUTTONUP ||
             message == WM_RBUTTONUP ||
             message == WM_MBUTTONUP ||
             message == WM_XBUTTONUP) &&
            !MouseMessageHasAnyButtonDown(wParam))
        {
            m_blockMouseUntilRelease = false;
        }
    }

    LRESULT imguiResult = 0;
    if (ImGui::GetCurrentContext() != nullptr &&
        (!previewInput || (inputFocused && !mouseWasBlocked)))
    {
        imguiResult = ImGui_ImplWin32_WndProcHandler(hwnd, message, wParam, lParam);
    }
    const bool imguiWantsMouse =
        ImGui::GetCurrentContext() != nullptr &&
        ImGui::GetIO().WantCaptureMouse;

    // GetAsyncKeyState and mouse capture can otherwise make the preview react to
    // input intended for another application. Focus and activation messages are
    // still passed through, but render/UI input is discarded while inactive.
    if (previewInput && (!inputFocused || mouseWasBlocked))
    {
        return 0;
    }

    if (m_state != nullptr)
    {
        switch (message)
        {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED)
            {
                m_pendingResizeWidth = static_cast<uint32_t>(LOWORD(lParam));
                m_pendingResizeHeight = static_cast<uint32_t>(HIWORD(lParam));
            }
            break;
        case WM_LBUTTONDBLCLK:
            if (!imguiWantsMouse)
            {
                m_state->focusMouseX = GET_X_LPARAM(lParam);
                m_state->focusMouseY = GET_Y_LPARAM(lParam);
                m_state->requestFocus = true;
                return 0;
            }
            break;
        case WM_LBUTTONDOWN:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            if (m_sceneMouseGesture || IsMousePointInScene(hwnd, x, y))
            {
                m_sceneMouseGesture = true;
                m_leftMouseDown = true;
                UpdateMouseDragMode(hwnd, x, y);
                // Keep scene clicks away from the host procedure. Passing the first
                // button through allowed it to steal capture before the chord formed.
                return 0;
            }
            break;
        }
        case WM_RBUTTONDOWN:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            if (m_sceneMouseGesture || IsMousePointInScene(hwnd, x, y))
            {
                m_sceneMouseGesture = true;
                m_rightMouseDown = true;
                UpdateMouseDragMode(hwnd, x, y);
                return 0;
            }
            break;
        }
        case WM_MBUTTONDOWN:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            if (m_sceneMouseGesture || IsMousePointInScene(hwnd, x, y))
            {
                m_sceneMouseGesture = true;
                m_middleMouseDown = true;
                UpdateMouseDragMode(hwnd, x, y);
                return 0;
            }
            break;
        }
        case WM_LBUTTONUP:
            if (m_sceneMouseGesture)
            {
                m_leftMouseDown = false;
                UpdateMouseDragMode(hwnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
                if (!m_rightMouseDown && !m_middleMouseDown) m_sceneMouseGesture = false;
                return 0;
            }
            break;
        case WM_RBUTTONUP:
            if (m_sceneMouseGesture)
            {
                m_rightMouseDown = false;
                UpdateMouseDragMode(hwnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
                if (!m_leftMouseDown && !m_middleMouseDown) m_sceneMouseGesture = false;
                return 0;
            }
            break;
        case WM_MBUTTONUP:
            if (m_sceneMouseGesture)
            {
                m_middleMouseDown = false;
                UpdateMouseDragMode(hwnd, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
                if (!m_leftMouseDown && !m_rightMouseDown) m_sceneMouseGesture = false;
                return 0;
            }
            break;
        case WM_MOUSEMOVE:
        {
            const int x = GET_X_LPARAM(lParam);
            const int y = GET_Y_LPARAM(lParam);
            RefreshMouseButtonsFromMove(wParam);

            // Recover a chord even when one of its button-down messages was routed
            // through ImGui or the host procedure. Physical button state on motion
            // is sufficient to arm the scene gesture, but only while this preview
            // is the foreground window.
            const bool requestedDrag = m_middleMouseDown || m_rightMouseDown;
            if (!m_sceneMouseGesture && requestedDrag && IsMousePointInScene(hwnd, x, y))
            {
                m_sceneMouseGesture = true;
            }
            UpdateMouseDragMode(hwnd, x, y);

            if (m_orbitDragging || m_panDragging)
            {
                const int deltaX = x - m_lastMouseX;
                const int deltaY = y - m_lastMouseY;
                m_lastMouseX = x;
                m_lastMouseY = y;

                if (m_orbitDragging)
                {
                    const float sensitivity = 0.006f * m_state->orbitSensitivity;
                    m_state->orbitYaw += static_cast<float>(deltaX) * sensitivity;
                    m_state->orbitPitch = ClampFloat(
                        m_state->orbitPitch + static_cast<float>(deltaY) * sensitivity,
                        -1.45f,
                        1.45f);
                }
                if (m_panDragging)
                {
                    XMFLOAT3 right, up, forward;
                    m_cameraController.GetCameraBasis(*m_state, right, up, forward);
                    const float fine = (GetKeyState(VK_SHIFT) & 0x8000) != 0 ? 0.22f : 1.0f;
                    const float scale = m_state->cameraDistance * 0.0016f * m_state->panSpeed * fine;
                    const XMFLOAT3 movement = Add3(
                        Multiply3(right, -static_cast<float>(deltaX) * scale),
                        Multiply3(up, static_cast<float>(deltaY) * scale));
                    m_state->targetX += movement.x;
                    m_state->targetY += movement.y;
                    m_state->targetZ += movement.z;
                }
                return 0;
            }

            if (m_sceneMouseGesture && !m_leftMouseDown && !m_rightMouseDown && !m_middleMouseDown)
            {
                m_sceneMouseGesture = false;
            }
            break;
        }
        case WM_MOUSEWHEEL:
            if (!imguiWantsMouse)
            {
                POINT cursor{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
                ScreenToClient(hwnd, &cursor);
                const float wheelSteps =
                    static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) /
                    static_cast<float>(WHEEL_DELTA);
                const float fine = (GetKeyState(VK_CONTROL) & 0x8000) != 0 ? 0.22f : 1.0f;
                m_state->requestZoom = true;
                m_state->zoomWheelSteps += wheelSteps * fine;
                m_state->zoomMouseX = cursor.x;
                m_state->zoomMouseY = cursor.y;
                return 0;
            }
            break;
        case WM_CAPTURECHANGED:
        case WM_CANCELMODE:
            ResetSceneInput(hwnd);
            break;
        default:
            break;
        }
    }

    if (message == WM_CLOSE)
    {
        DestroyWindow(hwnd);
        return 0;
    }
    if (message == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    if (imguiResult != 0)
    {
        return imguiResult;
    }

    return m_previousWindowProc != nullptr
        ? CallWindowProc(m_previousWindowProc, hwnd, message, wParam, lParam)
        : DefWindowProc(hwnd, message, wParam, lParam);
}

bool InputController::KeyPressed(int virtualKey)
{
    if (m_window == nullptr ||
        !m_inputFocused ||
        !IsPreviewWindowFocused(m_window) ||
        virtualKey < 0 ||
        virtualKey >= static_cast<int>(m_keyWasDown.size()))
    {
        return false;
    }

    const bool down = (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
    const bool pressed = down && !m_keyWasDown[static_cast<size_t>(virtualKey)];
    m_keyWasDown[static_cast<size_t>(virtualKey)] = down;
    return pressed;
}

} // namespace nsamdr
