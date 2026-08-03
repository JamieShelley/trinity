@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

if not exist "%ROOT%\CMakePresets.json" (
    echo ERROR: This script must be under scripts\build in the Trinity repository.
    exit /b 2
)

rem Permanent NSAMDR constraint: the real-asset preview is SDK-free.
rem Remove the superseded native real-ship path and its incomplete build cache.
call :RemoveFile "%ROOT%\scripts\build\run_nsamdr_realship_dx11.bat"
call :RemoveFile "%ROOT%\scripts\build\run_nsamdr_preview_dx11.bat"
call :RemoveFile "%ROOT%\scripts\build\build_nsamdr_realship_dx11.bat"
call :RemoveFile "%ROOT%\scripts\build\nsamdr\NSAMDRProjectInclude.cmake"
call :RemoveFile "%ROOT%\scripts\build\nsamdr\FindEveSharedCache.ps1"
call :RemoveFile "%ROOT%\scripts\build\nsamdr\PrepareNSAMDRVcpkgManifest.ps1"
call :RemoveFile "%ROOT%\trinity\tools\nsamdr\NSAMDRRealShipViewer.cpp"
call :RemoveFile "%ROOT%\trinity\tools\nsamdr\NSAMDRRealShipViewer.h"
call :RemoveFile "%ROOT%\trinity\tools\nsamdr\NSAMDRRealShipViewerMain.cpp"
call :RemoveFile "%ROOT%\trinity\tools\nsamdr\README.md"
if exist "%ROOT%\trinity\tools\nsamdr" rmdir "%ROOT%\trinity\tools\nsamdr" >nul 2>nul
if exist "%ROOT%\.cmake-build-x64-windows-trinitydev-nsamdr-realship-dx11" (
    echo Removing superseded native-viewer build cache...
    rmdir /s /q "%ROOT%\.cmake-build-x64-windows-trinitydev-nsamdr-realship-dx11"
)

call "%~dp0verify_and_clean_nsamdr_layout.bat"
if errorlevel 1 exit /b !ERRORLEVEL!

call "%~dp0run_nsamdr_eve_asset_dx11.bat" %*
exit /b !ERRORLEVEL!

:RemoveFile
if exist "%~1" (
    del /f /q "%~1" >nul 2>nul
    if exist "%~1" (
        echo ERROR: Could not remove superseded file:
        echo   %~1
        exit /b 80
    )
)
exit /b 0
