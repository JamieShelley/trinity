@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\..\.."

set "CONFIG=tools\nsamdr\neural\default_training_config.json"
set "WAIT_PID="
set "SOURCE_ROOT="
set "DEVICE="
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--config" (
    set "CONFIG=%~2"
    shift
    shift
    goto parse
)
if /I "%~1"=="--wait-pid" (
    set "WAIT_PID=%~2"
    shift
    shift
    goto parse
)
if /I "%~1"=="--device" (
    set "DEVICE=%~2"
    shift
    shift
    goto parse
)
if /I "%~1"=="--source-root" (
    set "SOURCE_ROOT=%~2"
    shift
    shift
    goto parse
)
echo ERROR: unknown argument %~1
exit /b 2
:parsed

if not exist scripts\build\verify_and_clean_nsamdr_layout.bat (
    echo ERROR: scripts\build\verify_and_clean_nsamdr_layout.bat is missing.
    echo The NSAMDR override was not fully extracted into the repository root.
    exit /b 2
)
if not exist scripts\build\train_nsamdr.bat (
    echo ERROR: scripts\build\train_nsamdr.bat is missing.
    echo The NSAMDR override was not fully extracted into the repository root.
    exit /b 2
)
if not exist scripts\build\test_nsamdr_real_eve_asset.bat (
    echo ERROR: scripts\build\test_nsamdr_real_eve_asset.bat is missing.
    echo Restore the existing NSAMDR branch launcher before running the combined workflow.
    exit /b 2
)

if defined WAIT_PID (
    echo Waiting for preview process %WAIT_PID% to close before rebuilding...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Wait-Process -Id %WAIT_PID% -ErrorAction SilentlyContinue"
)

call scripts\build\verify_and_clean_nsamdr_layout.bat
if errorlevel 1 goto failed

call scripts\build\cleanup_nsamdr_intermediate_modes.bat
if errorlevel 1 goto failed

set "DEVICE_ARGS="
if defined DEVICE set "DEVICE_ARGS=--device !DEVICE!"
if defined SOURCE_ROOT (
    call scripts\build\train_nsamdr.bat --config "%CONFIG%" --source-root "%SOURCE_ROOT%" !DEVICE_ARGS!
) else (
    call scripts\build\train_nsamdr.bat --config "%CONFIG%" !DEVICE_ARGS!
)
if errorlevel 1 goto failed

call scripts\build\test_nsamdr.bat --config "%CONFIG%"
if errorlevel 1 goto failed

echo Regenerating candidates, rebuilding TrinityALTest_dx11, and reopening the preview...
call scripts\build\test_nsamdr_real_eve_asset.bat
if errorlevel 1 goto failed
exit /b 0

:failed
set "RESULT=%errorlevel%"
if "%RESULT%"=="0" set "RESULT=1"
echo.
echo NSAMDR retrain/build/preview pipeline failed with exit code %RESULT%.
pause
exit /b %RESULT%
