#pragma once

#include "NSAMDRCameraController.h"

namespace nsamdr
{
class InputController final
{
public:
    explicit InputController(CameraController& cameraController);
    ~InputController();

    InputController(const InputController&) = delete;
    InputController& operator=(const InputController&) = delete;

    bool Attach(HWND window, PreviewState& state);
    void Detach();
    void RefreshFocus();
    void Reset();
    bool IsFocused() const;
    bool KeyPressed(int virtualKey);
    bool ConsumePendingResize(uint32_t& width, uint32_t& height);

private:
    static LRESULT CALLBACK WindowProcThunk(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);

    HWND GetPreviewRootWindow(HWND hwnd) const;
    bool IsPreviewWindowFocused(HWND hwnd) const;
    bool IsPreviewMouseInputMessage(UINT message);
    bool IsPreviewKeyboardInputMessage(UINT message);
    bool IsPreviewInputMessage(UINT message);
    bool AnyMouseButtonPhysicallyDown();
    void ResetSceneInput(HWND hwnd);
    void SynchronizeHotkeyState();
    void SetPreviewInputFocus(HWND hwnd, bool focused);
    void RefreshPreviewInputFocus(HWND hwnd);
    bool MouseMessageHasAnyButtonDown(WPARAM buttonFlags);
    bool IsMousePointInScene(HWND hwnd, int x, int y);
    void RefreshMouseButtonsFromMove(WPARAM buttonFlags);
    void UpdateMouseDragMode(HWND hwnd, int x, int y);

    static InputController* s_active;

    CameraController& m_cameraController;
    WNDPROC m_previousWindowProc = nullptr;
    HWND m_window = nullptr;
    PreviewState* m_state = nullptr;
    bool m_inputFocused = false;
    bool m_blockMouseUntilRelease = false;
    bool m_orbitDragging = false;
    bool m_panDragging = false;
    bool m_leftMouseDown = false;
    bool m_rightMouseDown = false;
    bool m_middleMouseDown = false;
    bool m_sceneMouseGesture = false;
    std::array<bool, 256> m_keyWasDown{};
    int m_lastMouseX = 0;
    int m_lastMouseY = 0;
    uint32_t m_pendingResizeWidth = 0U;
    uint32_t m_pendingResizeHeight = 0U;
};
} // namespace nsamdr
