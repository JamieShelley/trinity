@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\.cmake-build-x64-windows-trinitydev-nsamdr-obj-dx11"
set "PROJECT_INCLUDE=%~dp0nsamdr\NSAMDROBJProjectInclude.cmake"
set "VCPKG_DIR=%ROOT%\vendor\github.com\microsoft\vcpkg"
set "REGISTRY_DIR=%ROOT%\vendor\github.com\carbonengine\vcpkg-registry"
set "PREVIEW_EXE="
set "BUILD_ONLY=0"
if /I "%NSAMDR_BUILD_ONLY%"=="1" set "BUILD_ONLY=1"
set "REUSE_EXISTING_VIEWER=0"
set "STALE_VIEWER=0"
if /I "%NSAMDR_PREVIEW_REUSE_EXISTING_VIEWER%"=="1" set "REUSE_EXISTING_VIEWER=1"

if not exist "%ROOT%\CMakeLists.txt" (
    echo ERROR: This script must be under scripts\build in the Carbon Trinity repository.
    exit /b 2
)
if not exist "%ROOT%\trinityal\tests\nsamdr\standalone\NSAMDRStandalonePreview.cpp" (
    echo ERROR: Missing standalone Granny-free DX11 preview host:
    echo   "%ROOT%\trinityal\tests\nsamdr\standalone\NSAMDRStandalonePreview.cpp"
    exit /b 3
)
if not exist "%PROJECT_INCLUDE%" (
    echo ERROR: Missing OBJ preview CMake injection:
    echo   "%PROJECT_INCLUDE%"
    exit /b 4
)

rem The standalone viewer does not repair, build or link TrinityAL.
rem It is intentionally isolated from the unrelated Nsight Aftermath dependency.

if not exist "%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake" (
    echo ERROR: The repository vcpkg checkout is missing:
    echo   "%VCPKG_DIR%"
    echo Initialise the repository dependencies before building the preview.
    exit /b 6
)
if not exist "%REGISTRY_DIR%\triplets" (
    echo ERROR: The Carbon vcpkg registry is missing:
    echo   "%REGISTRY_DIR%"
    echo Initialise the repository dependencies before building the preview.
    exit /b 6
)
rem NSAMDR does not require a private vcpkg overlay.  Earlier packages
rem exported a path to an empty directory that ZIP extraction could not
rem preserve, causing vcpkg configuration to fail before the build began.
rem Clear the process-local value and the CMake cache entry below.
set "VCPKG_OVERLAY_PORTS="

where cmake.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: cmake.exe was not found in PATH.
    exit /b 7
)
call :resolve_fxc
if errorlevel 1 exit /b !ERRORLEVEL!

if "%BUILD_ONLY%"=="0" call :resolve_model "%~1"
if errorlevel 1 exit /b !ERRORLEVEL!
if "%BUILD_ONLY%"=="0" call :resolve_albedo "%~2"
if errorlevel 1 exit /b !ERRORLEVEL!
if "%BUILD_ONLY%"=="0" call :resolve_optional_texture NSAMDR_NORMAL "Normal map" "%~3"
if errorlevel 1 exit /b !ERRORLEVEL!
if "%BUILD_ONLY%"=="0" call :resolve_optional_texture NSAMDR_PGS "PGS material map" "%~4"
if errorlevel 1 exit /b !ERRORLEVEL!
if "%BUILD_ONLY%"=="0" call :resolve_optional_texture NSAMDR_ENVIRONMENT "EVE nebula environment" "%~5"
if errorlevel 1 exit /b !ERRORLEVEL!
if "%BUILD_ONLY%"=="0" call :resolve_optional_file NSAMDR_MATERIALS "SOF material manifest" "%~6"
if errorlevel 1 exit /b !ERRORLEVEL!

set "IMPORT_PATH=%BUILD_DIR%\vcpkg_installed\x64-windows-trinitydev"

rem Diagnostic previews should not be held hostage by an unrelated CMake
rem reconfigure when a previously built viewer executable already exists.
if "%BUILD_ONLY%"=="0" call :locate_preview_exe
if "%BUILD_ONLY%"=="0" if "%REUSE_EXISTING_VIEWER%"=="1" if defined PREVIEW_EXE (
    call :viewer_binary_matches_sources
    if errorlevel 1 (
        set "STALE_VIEWER=1"
        echo Existing DX11 viewer is older than the NSAMDR preview sources.
        echo It will NOT be reused; rebuilding the standalone viewer.
    ) else (
        echo Reusing source-current DX11 preview executable without CMake configure/build:
        echo   !PREVIEW_EXE!
        goto :launch_preview
    )
)

echo ============================================================
echo NSAMDR REAL OBJ SHIP PREVIEW - GRANNY FREE DX11
echo ============================================================
echo Repository : %ROOT%
echo Build dir  : %BUILD_DIR%
echo Target     : NSAMDRPreview_dx11
echo Granny     : OFF
if "%BUILD_ONLY%"=="0" echo Model      : !NSAMDR_OBJ!
if "%BUILD_ONLY%"=="0" if defined NSAMDR_ALBEDO echo Albedo     : !NSAMDR_ALBEDO!
if "%BUILD_ONLY%"=="0" if not defined NSAMDR_ALBEDO echo Albedo     : neutral fallback
if "%BUILD_ONLY%"=="0" if defined NSAMDR_NORMAL echo Normal     : !NSAMDR_NORMAL!
if "%BUILD_ONLY%"=="0" if defined NSAMDR_PGS echo PGS map    : !NSAMDR_PGS!
if "%BUILD_ONLY%"=="0" if defined NSAMDR_ENVIRONMENT echo Environment: !NSAMDR_ENVIRONMENT!
if "%BUILD_ONLY%"=="0" if not defined NSAMDR_ENVIRONMENT echo Environment: procedural fallback
if "%BUILD_ONLY%"=="0" if defined NSAMDR_MATERIALS echo Materials  : !NSAMDR_MATERIALS!
if "%BUILD_ONLY%"=="0" if not defined NSAMDR_MATERIALS echo Materials  : global texture fallback
if "%BUILD_ONLY%"=="1" echo Action     : build only
echo ============================================================

pushd "%ROOT%" || exit /b 11

echo CMake will perform its normal incremental configure/build.
echo CMake source directory: %ROOT%

call :validate_cmake_cache
set "CACHE_RESULT=!ERRORLEVEL!"
if not "!CACHE_RESULT!"=="0" (
    popd
    exit /b !CACHE_RESULT!
)

call :configure_preview
set "CONFIGURE_RESULT=!ERRORLEVEL!"
if not "!CONFIGURE_RESULT!"=="0" (
    call :report_failed_cmake_identity
    echo WARNING: Initial CMake configuration failed with exit code !CONFIGURE_RESULT!.
    echo Resetting stale CMake metadata and retrying once. Installed vcpkg packages are preserved.
    call :reset_cmake_metadata
    if errorlevel 1 (
        popd
        exit /b !ERRORLEVEL!
    )
    call :configure_preview
    set "CONFIGURE_RESULT=!ERRORLEVEL!"
)
if not "!CONFIGURE_RESULT!"=="0" (
    call :report_failed_cmake_identity
    if "%BUILD_ONLY%"=="0" (
        call :locate_preview_exe
        if defined PREVIEW_EXE if "!STALE_VIEWER!"=="0" (
            echo WARNING: CMake configuration failed after a clean metadata retry.
            echo WARNING: Reusing the source-current previously built DX11 viewer for diagnostic preview:
            echo   !PREVIEW_EXE!
            popd
            goto :launch_preview
        )
        if "!STALE_VIEWER!"=="1" (
            echo ERROR: CMake configuration failed and the only existing viewer is stale.
            echo ERROR: Refusing to show an old TrinityAL-hosted viewer as the current result.
        )
    )
    echo ERROR: CMake configuration failed after a clean metadata retry and no reusable preview executable exists.
    popd
    exit /b 20
)

rem Build only the standalone NSAMDR viewer.
rem It consumes raw D3D11 interfaces and deliberately does not build/link
rem TrinityAL_dx11, keeping Nsight Aftermath out of this preview path.
cmake --build "%BUILD_DIR%" ^
    --config TrinityDev ^
    --target NSAMDRPreview_dx11 ^
    --parallel
set "BUILD_RESULT=!ERRORLEVEL!"
popd

if not "!BUILD_RESULT!"=="0" (
    echo ERROR: OBJ preview build failed with exit code !BUILD_RESULT!.
    exit /b !BUILD_RESULT!
)

if "%BUILD_ONLY%"=="1" (
    echo Build completed successfully.
    exit /b 0
)

rem Locate the freshly built viewer (or an existing viewer if the build did not move it).
call :locate_preview_exe
if not defined PREVIEW_EXE (
    echo ERROR: NSAMDRPreview_dx11 executable was not found under:
    echo   "%BUILD_DIR%"
    echo Expected the normal Carbon output at:
    echo   "%BUILD_DIR%\nsamdr-preview\NSAMDRPreview_dx11.exe"
    exit /b 30
)

:launch_preview
if not defined PREVIEW_EXE call :locate_preview_exe
if not defined PREVIEW_EXE (
    echo ERROR: No reusable or freshly built standalone NSAMDR DX11 preview executable was found.
    exit /b 30
)
echo Launching: !PREVIEW_EXE!
pushd "%ROOT%" || exit /b 31
if /I "!NSAMDR_PREVIEW_VISIBLE_LAUNCH!"=="1" (
    rem Interactive diagnostic viewer: launch visibly and detach.
    rem The workflow stage is complete when Windows accepts the launch; the
    rem operator can then inspect/close the comparison window independently.
    echo Launch mode: visible detached diagnostic viewer
    start "" "!PREVIEW_EXE!" --gtest_filter=NSAMDRRendering.RealObjShipPreview --interactive --gtest_color=yes
    set "RUN_RESULT=!ERRORLEVEL!"
    if "!RUN_RESULT!"=="0" echo Viewer process launched successfully.
) else (
    "!PREVIEW_EXE!" --gtest_filter=NSAMDRRendering.RealObjShipPreview --interactive --gtest_color=yes
    set "RUN_RESULT=!ERRORLEVEL!"
)
popd
exit /b !RUN_RESULT!


:locate_preview_exe
set "PREVIEW_EXE="
set "EXPECTED_PREVIEW_EXE=%BUILD_DIR%\nsamdr-preview\NSAMDRPreview_dx11.exe"
if exist "!EXPECTED_PREVIEW_EXE!" set "PREVIEW_EXE=!EXPECTED_PREVIEW_EXE!"
if not defined PREVIEW_EXE (
    for /f "usebackq delims=" %%F in (`where.exe /r "%BUILD_DIR%" NSAMDRPreview_dx11.exe 2^>nul`) do (
        if not defined PREVIEW_EXE if exist "%%~fF" set "PREVIEW_EXE=%%~fF"
    )
)
exit /b 0


:viewer_binary_matches_sources
if not defined PREVIEW_EXE exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$exe=(Get-Item -LiteralPath '!PREVIEW_EXE!').LastWriteTimeUtc; " ^
  "$src=@(); " ^
  "$src += Get-ChildItem -LiteralPath '%ROOT%\trinityal\tests\nsamdr' -File -Recurse -Include '*.cpp','*.h','*.hlsl','*.ico'; " ^
  "$src += Get-Item -LiteralPath '%PROJECT_INCLUDE%'; " ^
  "if($src | Where-Object { $_.LastWriteTimeUtc -gt $exe }) { exit 1 } else { exit 0 }"
exit /b !ERRORLEVEL!

:configure_preview
cmake --preset x64-windows-trinitydev ^
    -S "%ROOT%" ^
    -B "%BUILD_DIR%" ^
    -DBUILD_DX11=ON ^
    -DBUILD_DX12=OFF ^
    -DBUILD_TESTING=OFF ^
    -DBUILD_SHADER_COMPILER=OFF ^
    -DWITH_GRANNY=OFF ^
    -DVCPKG_MANIFEST_INSTALL=OFF ^
    -DVCPKG_OVERLAY_PORTS= ^
    -DFXC_TOOL:FILEPATH="!FXC_TOOL!" ^
    -DCMAKE_PROJECT_INCLUDE="%PROJECT_INCLUDE%"
exit /b !ERRORLEVEL!

:report_failed_cmake_identity
set "NSAMDR_FAILED_CACHE_HOME="
set "NSAMDR_FAILED_CACHE_PROJECT="
if exist "%BUILD_DIR%\CMakeCache.txt" (
    for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"CMAKE_HOME_DIRECTORY:INTERNAL=" "%BUILD_DIR%\CMakeCache.txt" 2^>nul') do set "NSAMDR_FAILED_CACHE_HOME=%%B"
    for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"CMAKE_PROJECT_NAME:STATIC=" "%BUILD_DIR%\CMakeCache.txt" 2^>nul') do set "NSAMDR_FAILED_CACHE_PROJECT=%%B"
)
if defined NSAMDR_FAILED_CACHE_HOME (
    echo Failed CMake source identity:
    echo   CMAKE_HOME_DIRECTORY=!NSAMDR_FAILED_CACHE_HOME!
) else (
    echo Failed CMake source identity: cache did not record CMAKE_HOME_DIRECTORY
)
if defined NSAMDR_FAILED_CACHE_PROJECT (
    echo   CMAKE_PROJECT_NAME=!NSAMDR_FAILED_CACHE_PROJECT!
)
exit /b 0

:validate_cmake_cache
if not exist "%BUILD_DIR%\CMakeCache.txt" exit /b 0
set "NSAMDR_CACHE_HOME="
set "NSAMDR_CACHE_PROJECT="
for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"CMAKE_HOME_DIRECTORY:INTERNAL=" "%BUILD_DIR%\CMakeCache.txt" 2^>nul') do set "NSAMDR_CACHE_HOME=%%B"
for /f "tokens=1,* delims==" %%A in ('findstr /B /C:"CMAKE_PROJECT_NAME:STATIC=" "%BUILD_DIR%\CMakeCache.txt" 2^>nul') do set "NSAMDR_CACHE_PROJECT=%%B"
if not defined NSAMDR_CACHE_HOME (
    echo WARNING: Existing CMake cache has no CMAKE_HOME_DIRECTORY entry.
    call :reset_cmake_metadata
    exit /b !ERRORLEVEL!
)
for %%I in ("!NSAMDR_CACHE_HOME!") do set "NSAMDR_CACHE_HOME_FULL=%%~fI"
echo Existing CMake cache source: !NSAMDR_CACHE_HOME_FULL!
if /I not "!NSAMDR_CACHE_HOME_FULL!"=="%ROOT%" (
    echo WARNING: CMake cache source does not match the Trinity repository root.
    call :reset_cmake_metadata
    exit /b !ERRORLEVEL!
)
if not defined NSAMDR_CACHE_PROJECT (
    echo WARNING: Existing CMake cache has no CMAKE_PROJECT_NAME entry.
    call :reset_cmake_metadata
    exit /b !ERRORLEVEL!
)
if /I not "!NSAMDR_CACHE_PROJECT!"=="carbon-trinity" (
    echo WARNING: CMake cache top-level project is "!NSAMDR_CACHE_PROJECT!" instead of "carbon-trinity".
    call :reset_cmake_metadata
    exit /b !ERRORLEVEL!
)
echo Existing CMake cache project: !NSAMDR_CACHE_PROJECT!
exit /b 0

:reset_cmake_metadata
if not exist "%BUILD_DIR%" exit /b 0
echo Removing stale CMake configuration metadata from:
echo   "%BUILD_DIR%"
if exist "%BUILD_DIR%\CMakeCache.txt" del /f /q "%BUILD_DIR%\CMakeCache.txt" >nul 2>nul
if exist "%BUILD_DIR%\CMakeFiles" rmdir /s /q "%BUILD_DIR%\CMakeFiles"
if exist "%BUILD_DIR%\Testing" rmdir /s /q "%BUILD_DIR%\Testing"
for %%F in (
    cmake_install.cmake
    CTestTestfile.cmake
    Makefile
    build.ninja
    rules.ninja
    .ninja_deps
    .ninja_log
) do (
    if exist "%BUILD_DIR%\%%F" del /f /q "%BUILD_DIR%\%%F" >nul 2>nul
)
del /f /q "%BUILD_DIR%\*.sln" >nul 2>nul
del /f /q "%BUILD_DIR%\*.vcxproj" >nul 2>nul
del /f /q "%BUILD_DIR%\*.vcxproj.filters" >nul 2>nul
rem Keep vcpkg_installed and compiled outputs; regenerate only CMake identity.
exit /b 0

:resolve_fxc
set "FXC_TOOL="

rem The preview is an x64 target. Never accept the ARM64 or x86 SDK tools.
if defined WindowsSdkVerBinPath if exist "!WindowsSdkVerBinPath!x64\fxc.exe" set "FXC_TOOL=!WindowsSdkVerBinPath!x64\fxc.exe"
if not defined FXC_TOOL if defined WindowsSdkDir if defined WindowsSDKVersion if exist "!WindowsSdkDir!bin\!WindowsSDKVersion!x64\fxc.exe" set "FXC_TOOL=!WindowsSdkDir!bin\!WindowsSDKVersion!x64\fxc.exe"
if not defined FXC_TOOL if exist "!ProgramFiles(x86)!\Windows Kits\10\bin\x64\fxc.exe" set "FXC_TOOL=!ProgramFiles(x86)!\Windows Kits\10\bin\x64\fxc.exe"
if not defined FXC_TOOL if exist "!ProgramFiles(x86)!\Windows Kits\10\bin" (
    for /f "usebackq delims=" %%D in (`dir /b /ad /o-n "!ProgramFiles(x86)!\Windows Kits\10\bin" 2^>nul`) do (
        if not defined FXC_TOOL if exist "!ProgramFiles(x86)!\Windows Kits\10\bin\%%D\x64\fxc.exe" set "FXC_TOOL=!ProgramFiles(x86)!\Windows Kits\10\bin\%%D\x64\fxc.exe"
    )
)
if not defined FXC_TOOL (
    for /f "usebackq delims=" %%F in (`where.exe fxc.exe 2^>nul`) do (
        echo %%~fF| findstr /i /r "\\x64\\fxc\.exe$" >nul
        if not errorlevel 1 if not defined FXC_TOOL set "FXC_TOOL=%%~fF"
    )
)
if not defined FXC_TOOL (
    echo ERROR: Windows SDK x64 fxc.exe was not found.
    echo Install the Windows 10 SDK x64 tools in Visual Studio Installer.
    echo NSAMDR will not download CCP's private fxc package.
    exit /b 10
)
echo !FXC_TOOL!| findstr /i "\\x64\\fxc.exe" >nul
if errorlevel 1 (
    echo ERROR: Refusing non-x64 FXC compiler:
    echo   !FXC_TOOL!
    exit /b 10
)
echo FXC compiler ^(x64^): !FXC_TOOL!
echo Vcpkg manifest installation: OFF ^(using the dependencies already installed in this build directory^)
exit /b 0

:resolve_model
set "MODEL_INPUT=%~1"
if not defined MODEL_INPUT (
    echo.
    set /p "MODEL_INPUT=Paste an OBJ or GR2 file path: "
)
if not defined MODEL_INPUT (
    echo ERROR: No model path was supplied.
    echo Usage: %~nx0 ^<ship.obj^|ship.gr2^> [albedo.png] [normal.png] [pgs.png] [environment.png] [ship.materials.tsv]
    exit /b 40
)
for %%I in ("!MODEL_INPUT!") do (
    set "MODEL_INPUT=%%~fI"
    set "MODEL_EXTENSION=%%~xI"
    set "MODEL_BASENAME=%%~nI"
)
if not exist "!MODEL_INPUT!" (
    echo ERROR: Model file does not exist:
    echo   "!MODEL_INPUT!"
    exit /b 41
)

if /I "!MODEL_EXTENSION!"==".obj" (
    set "NSAMDR_OBJ=!MODEL_INPUT!"
    exit /b 0
)
if /I not "!MODEL_EXTENSION!"==".gr2" (
    echo ERROR: Unsupported model extension "!MODEL_EXTENSION!".
    echo Supply a Wavefront OBJ or an EVE GR2 file.
    exit /b 42
)

where node.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js 18 or newer is required for the Granny-free CarbonEngineJS GR2 converter.
    exit /b 43
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
    echo ERROR: npm is required to install the open-source CarbonEngineJS converter dependency.
    exit /b 44
)
set "CONVERTER_DIR=%ROOT%\tools\nsamdr\gr2_converter"
set "CONVERTER_SCRIPT=!CONVERTER_DIR!\convert_eve_asset.mjs"
set "CONVERTER_PACKAGE=!CONVERTER_DIR!\package.json"
if not exist "!CONVERTER_SCRIPT!" (
    echo ERROR: Missing Granny-free CarbonEngineJS converter:
    echo   "!CONVERTER_SCRIPT!"
    exit /b 45
)
if not exist "!CONVERTER_PACKAGE!" (
    echo ERROR: Missing CarbonEngineJS converter package manifest:
    echo   "!CONVERTER_PACKAGE!"
    exit /b 46
)
pushd "!CONVERTER_DIR!" || exit /b 47
node.exe --input-type=module --eval "await import('@carbonenginejs/format-gr2'); await import('@carbonenginejs/runtime-resource/formats/dds');" >nul 2>nul
set "MODULE_PROBE_RESULT=!ERRORLEVEL!"
popd
if "!MODULE_PROBE_RESULT!"=="0" goto :converter_dependencies_ready
goto :install_converter_dependencies

:install_converter_dependencies
echo Installing open-source CarbonEngineJS readers from public GitHub source archives...
if exist "!CONVERTER_DIR!\package-lock.json" del /f /q "!CONVERTER_DIR!\package-lock.json" >nul 2>nul
if exist "!CONVERTER_DIR!\node_modules" rmdir /s /q "!CONVERTER_DIR!\node_modules"
pushd "!CONVERTER_DIR!" || exit /b 47
call npm.cmd install --no-audit --no-fund --omit=dev --package-lock=false
set "NPM_RESULT=!ERRORLEVEL!"
popd
if not "!NPM_RESULT!"=="0" (
    echo ERROR: CarbonEngineJS GitHub-source dependency installation failed.
    echo Check HTTPS access to github.com, then rerun.
    exit /b !NPM_RESULT!
)
pushd "!CONVERTER_DIR!" || exit /b 47
node.exe --input-type=module --eval "await import('@carbonenginejs/format-gr2'); await import('@carbonenginejs/runtime-resource/formats/dds');"
set "MODULE_PROBE_RESULT=!ERRORLEVEL!"
popd
if not "!MODULE_PROBE_RESULT!"=="0" (
    echo ERROR: npm finished, but Node could not import the converter entry points.
    echo The module-resolution error above identifies the unresolved package.
    exit /b 47
)
goto :converter_dependencies_ready

:converter_dependencies_ready
set "CONVERT_DIR=%ROOT%\artifacts\nsamdr\converted"
if not exist "!CONVERT_DIR!" mkdir "!CONVERT_DIR!"
set "CONVERTED_OBJ=!CONVERT_DIR!\!MODEL_BASENAME!.obj"
set "CONVERSION_SUMMARY=!CONVERT_DIR!\!MODEL_BASENAME!.conversion.json"

echo Converting highest-detail GR2 mesh to OBJ without Granny...
echo   Source: !MODEL_INPUT!
echo   Output: !CONVERTED_OBJ!
node.exe "!CONVERTER_SCRIPT!" gr2-to-obj "!MODEL_INPUT!" "!CONVERTED_OBJ!" "!CONVERSION_SUMMARY!"
if errorlevel 1 (
    echo ERROR: CarbonEngineJS GR2 conversion failed.
    exit /b 48
)
if not exist "!CONVERTED_OBJ!" (
    echo ERROR: Converter returned success but did not create:
    echo   "!CONVERTED_OBJ!"
    exit /b 49
)
set "NSAMDR_OBJ=!CONVERTED_OBJ!"
exit /b 0

:resolve_albedo
set "ALBEDO_INPUT=%~1"
set "NSAMDR_ALBEDO="
if not defined ALBEDO_INPUT exit /b 0
for %%I in ("!ALBEDO_INPUT!") do (
    set "ALBEDO_INPUT=%%~fI"
    set "ALBEDO_EXTENSION=%%~xI"
)
if not exist "!ALBEDO_INPUT!" (
    echo ERROR: Albedo image does not exist:
    echo   "!ALBEDO_INPUT!"
    exit /b 50
)
if /I "!ALBEDO_EXTENSION!"==".dds" (
    echo ERROR: The Granny-free preview uses Windows Imaging Component and does not load DDS directly.
    echo Convert it first, for example:
    echo   texconv -ft png "!ALBEDO_INPUT!"
    exit /b 51
)
set "NSAMDR_ALBEDO=!ALBEDO_INPUT!"
exit /b 0

:resolve_optional_texture
set "OPTIONAL_VARIABLE=%~1"
set "OPTIONAL_LABEL=%~2"
set "OPTIONAL_INPUT=%~3"
set "%OPTIONAL_VARIABLE%="
if not defined OPTIONAL_INPUT exit /b 0
for %%I in ("!OPTIONAL_INPUT!") do (
    set "OPTIONAL_INPUT=%%~fI"
    set "OPTIONAL_EXTENSION=%%~xI"
)
if not exist "!OPTIONAL_INPUT!" (
    echo ERROR: !OPTIONAL_LABEL! does not exist:
    echo   "!OPTIONAL_INPUT!"
    exit /b 52
)
if /I "!OPTIONAL_EXTENSION!"==".dds" (
    echo ERROR: !OPTIONAL_LABEL! must be converted to PNG for the WIC viewer path.
    exit /b 53
)
set "%OPTIONAL_VARIABLE%=!OPTIONAL_INPUT!"
exit /b 0

:resolve_optional_file
set "OPTIONAL_VARIABLE=%~1"
set "OPTIONAL_LABEL=%~2"
set "OPTIONAL_INPUT=%~3"
set "%OPTIONAL_VARIABLE%="
if not defined OPTIONAL_INPUT exit /b 0
for %%I in ("!OPTIONAL_INPUT!") do set "OPTIONAL_INPUT=%%~fI"
if not exist "!OPTIONAL_INPUT!" (
    echo ERROR: !OPTIONAL_LABEL! does not exist:
    echo   "!OPTIONAL_INPUT!"
    exit /b 54
)
set "%OPTIONAL_VARIABLE%=!OPTIONAL_INPUT!"
exit /b 0
