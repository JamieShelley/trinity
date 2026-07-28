@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\.cmake-build-x64-windows-trinitydev-nsamdr-obj-dx11"
set "SETUP_DRIVER=%~dp0setup_dependencies.bat"
set "PROJECT_INCLUDE=%~dp0nsamdr\NSAMDROBJProjectInclude.cmake"
set "SOURCE_STATE_HELPER=%~dp0nsamdr\SourceBuildState.ps1"
set "VCPKG_DIR=%ROOT%\vendor\github.com\microsoft\vcpkg"
set "REGISTRY_DIR=%ROOT%\vendor\github.com\carbonengine\vcpkg-registry"
set "OVERLAY_PORTS=%~dp0nsamdr\vcpkg-overlay-ports"
set "SOURCE_CONTEXT=viewer=NSAMDRRealObjPreview-v3;config=TrinityDev;dx11=ON;dx12=OFF;tests=ON;shader=OFF;granny=OFF"
set "PREVIEW_EXE="
set "BUILD_ONLY=0"
if /I "%NSAMDR_BUILD_ONLY%"=="1" set "BUILD_ONLY=1"

if not exist "%ROOT%\CMakePresets.json" (
    echo ERROR: This script must be under scripts\build in the Carbon Trinity repository.
    exit /b 2
)
if not exist "%ROOT%\trinityal\tests\nsamdr\NSAMDRShipPreview.cpp" (
    echo ERROR: Missing Granny-free OBJ preview source:
    echo   "%ROOT%\trinityal\tests\nsamdr\NSAMDRShipPreview.cpp"
    exit /b 3
)
if not exist "%PROJECT_INCLUDE%" (
    echo ERROR: Missing OBJ preview CMake injection:
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

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%SOURCE_STATE_HELPER%" ^
    -Mode Prepare ^
    -RepoRoot "%ROOT%" ^
    -BuildDir "%BUILD_DIR%" ^
    -Context "%SOURCE_CONTEXT%"
if errorlevel 1 (
    echo ERROR: Could not determine or prepare the OBJ-preview source/build state.
    exit /b 10
)

set "IMPORT_PATH=%BUILD_DIR%\vcpkg_installed\x64-windows-trinitydev"

echo ============================================================
echo NSAMDR REAL OBJ SHIP PREVIEW - GRANNY FREE DX11
echo ============================================================
echo Repository : %ROOT%
echo Build dir  : %BUILD_DIR%
echo Target     : TrinityALTest_dx11
echo Granny     : OFF
if "%BUILD_ONLY%"=="0" echo Model      : !NSAMDR_OBJ!
if "%BUILD_ONLY%"=="0" if defined NSAMDR_ALBEDO echo Albedo     : !NSAMDR_ALBEDO!
if "%BUILD_ONLY%"=="0" if not defined NSAMDR_ALBEDO echo Albedo     : neutral fallback
if "%BUILD_ONLY%"=="0" if defined NSAMDR_NORMAL echo Normal     : !NSAMDR_NORMAL!
if "%BUILD_ONLY%"=="0" if defined NSAMDR_PGS echo PGS map    : !NSAMDR_PGS!
if "%BUILD_ONLY%"=="0" if defined NSAMDR_ENVIRONMENT echo Environment: !NSAMDR_ENVIRONMENT!
if "%BUILD_ONLY%"=="0" if not defined NSAMDR_ENVIRONMENT echo Environment: procedural fallback
if "%BUILD_ONLY%"=="1" echo Action     : build only
echo ============================================================

pushd "%ROOT%" || exit /b 11

cmake --preset x64-windows-trinitydev ^
    -S "%ROOT%" ^
    -B "%BUILD_DIR%" ^
    -DBUILD_DX11=ON ^
    -DBUILD_DX12=OFF ^
    -DBUILD_TESTING=ON ^
    -DBUILD_SHADER_COMPILER=OFF ^
    -DWITH_GRANNY=OFF ^
    -DCMAKE_PROJECT_INCLUDE="%PROJECT_INCLUDE%"
if errorlevel 1 (
    echo ERROR: CMake configuration failed.
    popd
    exit /b 20
)

cmake --build "%BUILD_DIR%" ^
    --config TrinityDev ^
    --target TrinityALTest_dx11 ^
    --parallel
set "BUILD_RESULT=!ERRORLEVEL!"
popd

if not "!BUILD_RESULT!"=="0" (
    echo ERROR: OBJ preview build failed with exit code !BUILD_RESULT!.
    exit /b !BUILD_RESULT!
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%SOURCE_STATE_HELPER%" ^
    -Mode Commit ^
    -RepoRoot "%ROOT%" ^
    -BuildDir "%BUILD_DIR%" ^
    -Context "%SOURCE_CONTEXT%"
if errorlevel 1 (
    echo ERROR: Build succeeded, but the source-state stamp could not be recorded.
    exit /b 21
)

if "%BUILD_ONLY%"=="1" (
    echo Build completed successfully.
    exit /b 0
)

rem The Carbon output is normally placed below carbon\autobuild rather than
rem directly in BUILD_DIR.  FOR /R with a literal filename can yield a candidate
rem path even when that file does not exist, so only accept verified files.
set "EXPECTED_PREVIEW_EXE=%BUILD_DIR%\carbon\autobuild\TrinityALTest\Windows\x64\v141\TrinityALTest_dx11_trinitydev.exe"
if exist "!EXPECTED_PREVIEW_EXE!" set "PREVIEW_EXE=!EXPECTED_PREVIEW_EXE!"

if not defined PREVIEW_EXE (
    for /f "usebackq delims=" %%F in (`where.exe /r "%BUILD_DIR%" TrinityALTest_dx11_trinitydev.exe 2^>nul`) do (
        if not defined PREVIEW_EXE if exist "%%~fF" set "PREVIEW_EXE=%%~fF"
    )
)
if not defined PREVIEW_EXE (
    for /f "usebackq delims=" %%F in (`where.exe /r "%BUILD_DIR%" TrinityALTest_dx11.exe 2^>nul`) do (
        if not defined PREVIEW_EXE if exist "%%~fF" set "PREVIEW_EXE=%%~fF"
    )
)
if not defined PREVIEW_EXE (
    echo ERROR: TrinityALTest_dx11 executable was not found under:
    echo   "%BUILD_DIR%"
    echo Expected the normal Carbon output at:
    echo   "!EXPECTED_PREVIEW_EXE!"
    exit /b 30
)

echo Launching: !PREVIEW_EXE!
pushd "%ROOT%" || exit /b 31
"!PREVIEW_EXE!" --gtest_filter=NSAMDRRendering.RealObjShipPreview --interactive --gtest_color=yes
set "RUN_RESULT=!ERRORLEVEL!"
popd
exit /b !RUN_RESULT!

:resolve_model
set "MODEL_INPUT=%~1"
if not defined MODEL_INPUT (
    echo.
    set /p "MODEL_INPUT=Paste an OBJ or GR2 file path: "
)
if not defined MODEL_INPUT (
    echo ERROR: No model path was supplied.
    echo Usage: %~nx0 ^<ship.obj^|ship.gr2^> [albedo.png]
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

set "CONVERTER=%EVEGR2TOOBJ_EXE%"
if not defined CONVERTER set "CONVERTER=%ROOT%\tools\nsamdr\evegr2toobj\evegr2toobj.exe"
if not exist "!CONVERTER!" (
    echo ERROR: A GR2 file was supplied, but evegr2toobj.exe was not found.
    echo.
    echo Place a locally obtained converter here:
    echo   "%ROOT%\tools\nsamdr\evegr2toobj\evegr2toobj.exe"
    echo.
    echo The converter and Granny runtime are not redistributed by this overlay.
    echo Alternatively convert the GR2 separately and pass the resulting OBJ.
    exit /b 43
)
for %%I in ("!CONVERTER!") do set "CONVERTER_DIR=%%~dpI"
if not exist "!CONVERTER_DIR!granny2.dll" (
    echo ERROR: granny2.dll was not found beside the local converter:
    echo   "!CONVERTER_DIR!granny2.dll"
    echo Use a legally obtained local runtime compatible with the converter.
    exit /b 44
)

set "CONVERT_DIR=%ROOT%\artifacts\nsamdr\converted"
if not exist "!CONVERT_DIR!" mkdir "!CONVERT_DIR!"
set "CONVERTED_OBJ=!CONVERT_DIR!\!MODEL_BASENAME!.obj"

echo Converting GR2 to OBJ...
echo   Source: !MODEL_INPUT!
echo   Output: !CONVERTED_OBJ!
"!CONVERTER!" "!MODEL_INPUT!" "!CONVERTED_OBJ!"
if errorlevel 1 (
    echo ERROR: evegr2toobj failed.
    exit /b 45
)
if not exist "!CONVERTED_OBJ!" (
    echo ERROR: Converter returned success but did not create:
    echo   "!CONVERTED_OBJ!"
    exit /b 46
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
