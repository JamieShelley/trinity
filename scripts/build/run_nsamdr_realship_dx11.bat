@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\.cmake-build-x64-windows-trinitydev-nsamdr-realship-dx11"
set "SETUP_DRIVER=%~dp0setup_dependencies.bat"
set "PROJECT_INCLUDE=%~dp0nsamdr\NSAMDRProjectInclude.cmake"
set "SOURCE_STATE_HELPER=%~dp0nsamdr\SourceBuildState.ps1"
set "VCPKG_DIR=%ROOT%\vendor\github.com\microsoft\vcpkg"
set "REGISTRY_DIR=%ROOT%\vendor\github.com\carbonengine\vcpkg-registry"
set "OVERLAY_PORTS=%~dp0nsamdr\vcpkg-overlay-ports"
set "MANIFEST_LOG=%BUILD_DIR%\vcpkg-manifest-install.log"
set "SOURCE_CONTEXT=viewer=NSAMDRRealShipViewer-v2;config=TrinityDev;dx11=ON;dx12=OFF;tests=OFF;shader=OFF;granny=ON"
set "VIEWER_EXE="
set "SHARED_CACHE=%~1"
set "BUILD_ONLY=0"
if /I "%NSAMDR_BUILD_ONLY%"=="1" set "BUILD_ONLY=1"

if not exist "%ROOT%\CMakePresets.json" (
    echo ERROR: This script must be under scripts\build in the Trinity repository.
    exit /b 2
)
if not exist "%ROOT%\trinity\tools\nsamdr\NSAMDRRealShipViewer.cpp" (
    echo ERROR: The real-ship viewer source is missing.
    echo   "%ROOT%\trinity\tools\nsamdr\NSAMDRRealShipViewer.cpp"
    echo The obsolete TrinityAL proxy will not be built as a fallback.
    exit /b 3
)
if not exist "%PROJECT_INCLUDE%" (
    echo ERROR: Missing real-ship CMake injection:
    echo   "%PROJECT_INCLUDE%"
    exit /b 4
)
if not exist "%SOURCE_STATE_HELPER%" (
    echo ERROR: Missing source-state helper:
    echo   "%SOURCE_STATE_HELPER%"
    exit /b 5
)

if not exist "%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake" goto :setup_dependencies
if not exist "%REGISTRY_DIR%\triplets" goto :setup_dependencies
goto :dependencies_ready

:setup_dependencies
if not exist "%SETUP_DRIVER%" (
    echo ERROR: Missing dependency setup script:
    echo   "%SETUP_DRIVER%"
    exit /b 6
)
echo Repository dependencies are missing. Running setup_dependencies.bat...
call "%SETUP_DRIVER%"
if errorlevel 1 exit /b !ERRORLEVEL!

:dependencies_ready
if defined VCPKG_OVERLAY_PORTS (
    set "VCPKG_OVERLAY_PORTS=%OVERLAY_PORTS%;%VCPKG_OVERLAY_PORTS%"
) else (
    set "VCPKG_OVERLAY_PORTS=%OVERLAY_PORTS%"
)

where cmake.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: cmake.exe was not found in PATH.
    exit /b 7
)
where git.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: git.exe was not found in PATH.
    exit /b 8
)
where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: powershell.exe was not found.
    exit /b 9
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%SOURCE_STATE_HELPER%" ^
    -Mode Prepare ^
    -RepoRoot "%ROOT%" ^
    -BuildDir "%BUILD_DIR%" ^
    -Context "%SOURCE_CONTEXT%"
if errorlevel 1 (
    echo ERROR: Could not determine or prepare the real-viewer source/build state.
    exit /b 10
)

set "IMPORT_PATH=%BUILD_DIR%\vcpkg_installed\x64-windows-trinitydev"

echo ============================================================
echo NSAMDR REAL EVE SHIP VIEWER - FULL TRINITY DX11
echo ============================================================
echo This does NOT build or launch TrinityALTest_dx11.
echo Repository : %ROOT%
echo Build dir  : %BUILD_DIR%
echo Target     : NSAMDRRealShipViewer
echo Granny     : ON
if "%BUILD_ONLY%"=="1" echo Mode       : BUILD ONLY
if not "%BUILD_ONLY%"=="1" echo Mode       : BUILD AND RUN
echo ============================================================
echo.

pushd "%ROOT%" || exit /b 11
cmake --preset x64-windows-trinitydev ^
    -B "%BUILD_DIR%" ^
    -DBUILD_DX11=ON ^
    -DBUILD_DX12=OFF ^
    -DBUILD_TESTING=OFF ^
    -DBUILD_SHADER_COMPILER=OFF ^
    -DWITH_GRANNY=ON ^
    -DCMAKE_PROJECT_INCLUDE:FILEPATH="%PROJECT_INCLUDE%"
if errorlevel 1 goto :configure_failed

cmake --build "%BUILD_DIR%" --config TrinityDev --target NSAMDRRealShipViewer --parallel
if errorlevel 1 goto :build_failed
popd

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%SOURCE_STATE_HELPER%" ^
    -Mode Commit ^
    -RepoRoot "%ROOT%" ^
    -BuildDir "%BUILD_DIR%" ^
    -Context "%SOURCE_CONTEXT%"
if errorlevel 1 (
    echo ERROR: Build succeeded, but its source-state stamp could not be recorded.
    exit /b 12
)

set "EXPECTED_EXE=%BUILD_DIR%\carbon\autobuild\NSAMDRRealShipViewer\Windows\x64\v141\NSAMDRRealShipViewer_trinitydev.exe"
if exist "!EXPECTED_EXE!" set "VIEWER_EXE=!EXPECTED_EXE!"
if not defined VIEWER_EXE for /r "%BUILD_DIR%" %%F in (NSAMDRRealShipViewer*.exe) do if not defined VIEWER_EXE set "VIEWER_EXE=%%~fF"

if not defined VIEWER_EXE (
    echo ERROR: The real-ship executable was not produced.
    echo The obsolete TrinityAL proxy will not be launched as a fallback.
    exit /b 20
)

echo.
echo REAL-SHIP BUILD COMPLETED:
echo   !VIEWER_EXE!
echo.
if "%BUILD_ONLY%"=="1" exit /b 0

if not defined SHARED_CACHE if defined EVE_SHARED_CACHE set "SHARED_CACHE=%EVE_SHARED_CACHE%"
if not defined SHARED_CACHE if exist "C:\EVE\SharedCache\ResFiles" set "SHARED_CACHE=C:\EVE\SharedCache"
if not defined SHARED_CACHE if exist "%LOCALAPPDATA%\CCP\EVE\SharedCache\ResFiles" set "SHARED_CACHE=%LOCALAPPDATA%\CCP\EVE\SharedCache"
if not defined SHARED_CACHE if exist "%PROGRAMDATA%\CCP\EVE\SharedCache\ResFiles" set "SHARED_CACHE=%PROGRAMDATA%\CCP\EVE\SharedCache"

if not defined SHARED_CACHE (
    echo ERROR: EVE SharedCache was not auto-detected.
    echo Run:
    echo   %~nx0 "D:\path\to\EVE\SharedCache"
    echo The folder must contain ResFiles.
    exit /b 21
)
for %%I in ("%SHARED_CACHE%") do set "SHARED_CACHE=%%~fI"
if not exist "!SHARED_CACHE!\ResFiles" (
    echo ERROR: EVE SharedCache does not contain ResFiles:
    echo   !SHARED_CACHE!
    exit /b 22
)

set "RES_INDEX="
for /r "!SHARED_CACHE!" %%F in (resfileindex.txt) do if not defined RES_INDEX set "RES_INDEX=%%~fF"
if not defined RES_INDEX (
    echo ERROR: resfileindex.txt was not found below:
    echo   !SHARED_CACHE!
    exit /b 23
)

set "PYTHON_STDLIB="
for %%P in (
    "!IMPORT_PATH!\tools\python3\Lib"
    "!IMPORT_PATH!\tools\python3\lib"
    "!IMPORT_PATH!\bin\python\Lib"
) do if not defined PYTHON_STDLIB if exist "%%~fP\encodings" set "PYTHON_STDLIB=%%~fP"
if not defined PYTHON_STDLIB for /d %%P in ("!IMPORT_PATH!\tools\python*\Lib") do if exist "%%~fP\encodings" set "PYTHON_STDLIB=%%~fP"

set "NSAMDR_EVE_SHARED_CACHE=!SHARED_CACHE!"
set "NSAMDR_EVE_RES_INDEX=!RES_INDEX!"
set "NSAMDR_IMPORT_PATH=!IMPORT_PATH!"
if defined PYTHON_STDLIB set "NSAMDR_PYTHON_STDLIB=!PYTHON_STDLIB!"

echo Launching the full-Trinity viewer:
echo   !VIEWER_EXE!
echo SharedCache:
echo   !SHARED_CACHE!
echo.
pushd "%ROOT%" || exit /b 24
"!VIEWER_EXE!" --shared-cache "!SHARED_CACHE!" --res-index "!RES_INDEX!"
set "RESULT=!ERRORLEVEL!"
popd
exit /b !RESULT!

:configure_failed
set "RESULT=!ERRORLEVEL!"
popd
echo.
echo ERROR: Full-Trinity real-ship CMake configuration failed with exit code !RESULT!.
goto :show_manifest

:build_failed
set "RESULT=!ERRORLEVEL!"
popd
echo.
echo ERROR: NSAMDRRealShipViewer build failed with exit code !RESULT!.

:show_manifest
if exist "%MANIFEST_LOG%" (
    echo.
    echo ================= vcpkg manifest log =================
    type "%MANIFEST_LOG%"
    echo ================= end manifest log ===================
)
exit /b !RESULT!
