@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "GUI=%SCRIPT_DIR%build_gui.py"

if not exist "%GUI%" (
    echo ERROR: Could not find:
    echo   "%GUI%"
    pause
    exit /b 2
)

where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 "%GUI%"
    set "RESULT=%ERRORLEVEL%"
    if not "%RESULT%"=="0" pause
    exit /b %RESULT%
)

where python.exe >nul 2>nul
if not errorlevel 1 (
    python "%GUI%"
    set "RESULT=%ERRORLEVEL%"
    if not "%RESULT%"=="0" pause
    exit /b %RESULT%
)

echo ERROR: Python 3 was not found.
echo Install Python 3 with Tkinter, then run this file again.
pause
exit /b 3
