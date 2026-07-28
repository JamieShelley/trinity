@echo off
setlocal EnableExtensions

rem Usage:
rem   _run_tests.bat <config> <dx11|dx12> [filter] [mode]
rem mode: normal | interactive | screenshots | compare

set "CONFIG_KEY=%~1"
set "BACKEND=%~2"
set "FILTER=%~3"
set "MODE=%~4"

if not defined CONFIG_KEY goto :usage
if not defined BACKEND goto :usage
if not defined FILTER set "FILTER=*"
if not defined MODE set "MODE=normal"

if /I "%BACKEND%"=="dx11" (
    set "TEST_TARGET=TrinityALTest_dx11"
) else if /I "%BACKEND%"=="dx12" (
    set "TEST_TARGET=TrinityALTest_dx12"
) else (
    echo ERROR: Tests require dx11 or dx12.
    goto :usage
)

call "%~dp0_build_config.bat" "%CONFIG_KEY%" "%BACKEND%" "%TEST_TARGET%" build
if errorlevel 1 exit /b %ERRORLEVEL%

if /I "%CONFIG_KEY%"=="debug" (
    set "PRESET=x64-windows-debug"
) else if /I "%CONFIG_KEY%"=="release" (
    set "PRESET=x64-windows-release"
) else if /I "%CONFIG_KEY%"=="internal" (
    set "PRESET=x64-windows-internal"
) else if /I "%CONFIG_KEY%"=="trinitydev" (
    set "PRESET=x64-windows-trinitydev"
) else (
    goto :usage
)

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\.cmake-build-%PRESET%-%BACKEND%"
set "TEST_EXE="

for /R "%BUILD_DIR%" %%F in (%TEST_TARGET%.exe) do (
    if not defined TEST_EXE set "TEST_EXE=%%~fF"
)

if not defined TEST_EXE (
    echo ERROR: %TEST_TARGET%.exe was not found under:
    echo   "%BUILD_DIR%"
    exit /b 20
)

set "EXTRA_ARGS="
if /I "%MODE%"=="interactive" (
    set "EXTRA_ARGS=--interactive"
) else if /I "%MODE%"=="screenshots" (
    set "SCREENSHOT_DIR=%ROOT%\artifacts\screenshots"
    if not exist "%SCREENSHOT_DIR%" mkdir "%SCREENSHOT_DIR%"
    set "EXTRA_ARGS=--screenshots --screenshotdir "%SCREENSHOT_DIR%""
) else if /I "%MODE%"=="compare" (
    set "SCREENSHOT_DIR=%ROOT%\artifacts\screenshots"
    set "EXTRA_ARGS=--compare --screenshotdir "%SCREENSHOT_DIR%""
) else if /I not "%MODE%"=="normal" (
    echo ERROR: Unsupported test mode "%MODE%".
    goto :usage
)

echo.
echo Running:
echo   "%TEST_EXE%" --gtest_filter=%FILTER% %EXTRA_ARGS%
echo.

"%TEST_EXE%" "--gtest_filter=%FILTER%" %EXTRA_ARGS%
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   %~nx0 ^<debug^|release^|internal^|trinitydev^> ^<dx11^|dx12^> [gtest-filter] [normal^|interactive^|screenshots^|compare]
exit /b 1
