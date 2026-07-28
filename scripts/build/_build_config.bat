@echo off
setlocal EnableExtensions

rem Internal build driver.
rem Usage:
rem   _build_config.bat <config> <backend> [target] [action] [shader]
rem
rem config:  debug | release | internal | trinitydev
rem backend: dx11 | dx12 | both | stub
rem target:  ALL or a CMake target name
rem action:  build | configure | rebuild
rem shader:  shader to enable BUILD_SHADER_COMPILER

set "CONFIG_KEY=%~1"
set "BACKEND=%~2"
set "TARGET=%~3"
set "ACTION=%~4"
set "SHADER_OPTION=%~5"

if not defined CONFIG_KEY goto :usage
if not defined BACKEND goto :usage
if not defined TARGET set "TARGET=ALL"
if not defined ACTION set "ACTION=build"

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
if not exist "%ROOT%\CMakePresets.json" (
    echo ERROR: CMakePresets.json was not found at:
    echo   "%ROOT%"
    echo Copy the scripts\build directory into the Carbon Trinity repository.
    exit /b 2
)

where cmake.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: cmake.exe was not found in PATH.
    echo Carbon Trinity requires CMake 3.31 or newer.
    exit /b 3
)

where git.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: git.exe was not found in PATH.
    exit /b 4
)

if /I "%CONFIG_KEY%"=="debug" (
    set "PRESET=x64-windows-debug"
    set "CMAKE_CONFIG=Debug"
) else if /I "%CONFIG_KEY%"=="release" (
    set "PRESET=x64-windows-release"
    set "CMAKE_CONFIG=Release"
) else if /I "%CONFIG_KEY%"=="internal" (
    set "PRESET=x64-windows-internal"
    set "CMAKE_CONFIG=Internal"
) else if /I "%CONFIG_KEY%"=="trinitydev" (
    set "PRESET=x64-windows-trinitydev"
    set "CMAKE_CONFIG=TrinityDev"
) else (
    echo ERROR: Unsupported configuration "%CONFIG_KEY%".
    goto :usage
)

if /I "%BACKEND%"=="dx11" (
    set "BUILD_DX11=ON"
    set "BUILD_DX12=OFF"
) else if /I "%BACKEND%"=="dx12" (
    set "BUILD_DX11=OFF"
    set "BUILD_DX12=ON"
) else if /I "%BACKEND%"=="both" (
    set "BUILD_DX11=ON"
    set "BUILD_DX12=ON"
) else if /I "%BACKEND%"=="stub" (
    set "BUILD_DX11=OFF"
    set "BUILD_DX12=OFF"
) else (
    echo ERROR: Unsupported backend "%BACKEND%".
    goto :usage
)

set "BUILD_SHADER_COMPILER=OFF"
if /I "%SHADER_OPTION%"=="shader" set "BUILD_SHADER_COMPILER=ON"

set "BUILD_DIR=%ROOT%\.cmake-build-%PRESET%-%BACKEND%"
set "VCPKG_DIR=%ROOT%\vendor\github.com\microsoft\vcpkg"
set "REGISTRY_DIR=%ROOT%\vendor\github.com\carbonengine\vcpkg-registry"

if not exist "%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake" (
    echo Dependencies are missing. Running setup_dependencies.bat...
    call "%~dp0setup_dependencies.bat"
    if errorlevel 1 exit /b %ERRORLEVEL%
)

if not exist "%REGISTRY_DIR%" (
    echo Dependencies are missing. Running setup_dependencies.bat...
    call "%~dp0setup_dependencies.bat"
    if errorlevel 1 exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo Carbon Trinity build
echo Configuration : %CMAKE_CONFIG%
echo Backend       : %BACKEND%
echo DX11          : %BUILD_DX11%
echo DX12          : %BUILD_DX12%
echo Tests         : ON
echo ShaderCompiler: %BUILD_SHADER_COMPILER%
echo Target        : %TARGET%
echo Action        : %ACTION%
echo Build folder  : %BUILD_DIR%
echo ============================================================
echo.

pushd "%ROOT%" || exit /b 5

cmake --preset "%PRESET%" ^
    -B "%BUILD_DIR%" ^
    -DBUILD_DX11=%BUILD_DX11% ^
    -DBUILD_DX12=%BUILD_DX12% ^
    -DBUILD_TESTING=ON ^
    -DBUILD_SHADER_COMPILER=%BUILD_SHADER_COMPILER%

if errorlevel 1 (
    echo ERROR: CMake configuration failed.
    popd
    exit /b 10
)

if /I "%ACTION%"=="configure" (
    echo Configuration completed.
    popd
    exit /b 0
)

if /I "%ACTION%"=="rebuild" (
    echo Cleaning configured targets...
    cmake --build "%BUILD_DIR%" --config "%CMAKE_CONFIG%" --target clean
    if errorlevel 1 (
        echo ERROR: Clean failed.
        popd
        exit /b 11
    )
) else if /I not "%ACTION%"=="build" (
    echo ERROR: Unsupported action "%ACTION%".
    popd
    goto :usage_after_popd
)

if /I "%TARGET%"=="ALL" (
    cmake --build "%BUILD_DIR%" --config "%CMAKE_CONFIG%" --parallel
) else (
    cmake --build "%BUILD_DIR%" --config "%CMAKE_CONFIG%" --target "%TARGET%" --parallel
)

set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" (
    echo ERROR: Build failed with exit code %RESULT%.
    exit /b %RESULT%
)

echo.
echo Build completed successfully.
exit /b 0

:usage
echo Usage:
echo   %~nx0 ^<debug^|release^|internal^|trinitydev^> ^<dx11^|dx12^|both^|stub^> [target] [build^|configure^|rebuild] [shader]
echo.
echo Examples:
echo   %~nx0 trinitydev dx11 ALL build
echo   %~nx0 debug dx12 TrinityALTest_dx12 rebuild
echo   %~nx0 trinitydev stub ShaderCompiler build shader
exit /b 1

:usage_after_popd
echo Usage:
echo   %~nx0 ^<debug^|release^|internal^|trinitydev^> ^<dx11^|dx12^|both^|stub^> [target] [build^|configure^|rebuild] [shader]
exit /b 1
