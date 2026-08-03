@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "CHECK_ONLY=0"
if /I "%~1"=="--check-only" set "CHECK_ONLY=1"
if not "%~1"=="" if /I not "%~1"=="--check-only" (
    echo ERROR: unknown argument %~1
    echo Usage: verify_and_clean_nsamdr_layout.bat [--check-only]
    exit /b 2
)

if not exist "%ROOT%\CMakePresets.json" (
    echo ERROR: The script is not installed under scripts\build in the Trinity repository.
    echo Expected repository marker: "%ROOT%\CMakePresets.json"
    exit /b 2
)
set "TRINITYAL_REPAIR=%ROOT%\scripts\build\nsamdr\RepairMissingTrinityALMarker.ps1"
if not exist "%TRINITYAL_REPAIR%" (
    echo ERROR: Missing TrinityAL source-tree repair script:
    echo   "%TRINITYAL_REPAIR%"
    exit /b 2
)

rem Previous NSAMDR verifiers incorrectly treated legitimate upstream
rem trinityal\scripts, trinityal\tools and trinityal\trinityal content as
rem duplicate override paths. Repair every tracked TrinityAL file that is absent,
rem without overwriting any existing or modified file.
if "%CHECK_ONLY%"=="1" (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
        -File "%TRINITYAL_REPAIR%" ^
        -RepoRoot "%ROOT%" ^
        -CheckOnly
) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
        -File "%TRINITYAL_REPAIR%" ^
        -RepoRoot "%ROOT%"
)
if errorlevel 1 (
    echo ERROR: TrinityAL tracked source-tree verification or repair failed.
    exit /b 2
)

for %%F in (
    "%ROOT%\trinityal\CMakeLists.txt"
    "%ROOT%\trinityal\ALLog.h"
    "%ROOT%\trinityal\ALResult.cpp"
    "%ROOT%\trinityal\tests\CMakeLists.txt"
    "%ROOT%\trinityal\tests\ALResultTest.cpp"
    "%ROOT%\trinityal\tests\TrinityALTest.cpp"
) do if not exist "%%~F" (
    echo ERROR: Required TrinityAL source remains missing after repair: "%%~F"
    exit /b 2
)

pushd "%ROOT%" || exit /b 2
set /a REMOVED=0
set /a FAILURES=0

echo ============================================================
echo NSAMDR SOURCE LAYOUT VERIFICATION
echo Repository: %ROOT%
if "%CHECK_ONLY%"=="1" (echo Action    : check only) else (echo Action    : clean invalid NSAMDR paths, then verify)
echo ============================================================

if "%CHECK_ONLY%"=="0" (
    rem Preserve all upstream TrinityAL directories. Only the dedicated
    rem trinityal\tests\nsamdr directory is owned by this override.

    call :RemoveFile "README_NSAMDR.md"
    call :RemoveFile "NSAMDR_MODE7_NEURAL_RUNTIME_V5_NOTES.md"
    call :RemoveFile "NSAMDR_THREE_MODE_PIPELINE_V5_2_NOTES.md"
    call :RemoveFile "scripts\build\nsamdr\SourceBuildState.ps1"
    call :RemoveFile "NSAMDR_MODE7_CONTOUR_CLEAN_V4_NOTES.md"
    call :RemoveFile "NSAMDR_MODE7_COVERAGE_AWARE_FIX_NOTES.md"
    call :RemoveFile "NSAMDR_MODE7_SUBPIXEL_BOUNDARY_FIX_NOTES.md"

    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRMode6Pipeline.inl"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRMode7Pipeline.inl"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDR_MODE7_NEURAL_RUNTIME_V5_NOTES.md"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRNativeEveRenderer.cpp"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRNativeEveRenderer.h"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRNeuralRuntime.h"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRNeuralRuntime.cpp"
    call :RemoveFile "trinityal\tests\nsamdr\NSAMDRNeuralWeights.hlsli"
    for %%F in ("trinityal\tests\nsamdr\*.inl") do if exist "%%~F" call :RemoveFile "%%~F"
    call :RemoveFile "scripts\build\train_nsamdr_mode7.bat"
    call :RemoveFile "scripts\build\test_nsamdr_mode7.bat"
    call :RemoveFile "scripts\build\retrain_nsamdr_mode7_and_preview.bat"
    call :RemoveFile "tools\nsamdr\neural\train_mode7_kernel.py"
    call :RemoveFile "tools\nsamdr\neural\test_mode7_kernel.py"
    call :RemoveFile "tools\nsamdr\strategy_pipeline\mode6.py"
    call :RemoveFile "tools\nsamdr\strategy_pipeline\mode7.py"
    call :RemoveFile "tools\nsamdr\strategy_pipeline\builders.py"

    call :RemoveDirectory "tools\nsamdr\__pycache__"
    call :RemoveDirectory "tools\nsamdr\strategy_pipeline\__pycache__"
    call :RemoveDirectory "tools\nsamdr\neural\__pycache__"
    call :RemoveDirectory "tools\nsamdr\tests\__pycache__"

    rem The NSAMDR source directory is owned by this override. Remove every
    rem stale file or subdirectory that is not in the current source manifest.
    call :CleanNSAMDRSourceDirectory
)

for %%F in (
    "trinityal\CMakeLists.txt"
    "trinityal\ALLog.h"
    "trinityal\ALResult.cpp"
    "trinityal\tests\CMakeLists.txt"
    "trinityal\tests\ALResultTest.cpp"
    "trinityal\tests\TrinityALTest.cpp"
    "scripts\build\verify_and_clean_nsamdr_layout.bat"
    "scripts\build\cleanup_nsamdr_intermediate_modes.bat"
    "scripts\build\setup_nsamdr_cuda.bat"
    "scripts\build\setup_nsamdr_cpu.bat"
    "scripts\build\train_nsamdr.bat"
    "scripts\build\test_nsamdr.bat"
    "scripts\build\retrain_nsamdr_and_preview.bat"
    "scripts\build\test_nsamdr_real_eve_asset.bat"
    "scripts\build\run_nsamdr_eve_asset_dx11.bat"
    "scripts\build\run_nsamdr_obj_preview_dx11.bat"
    "scripts\build\build_nsamdr_obj_preview_dx11.bat"
    "scripts\build\generate_nsamdr_strategy_candidates.bat"
    "scripts\build\nsamdr\NSAMDROBJProjectInclude.cmake"
    "scripts\build\nsamdr\RepairMissingTrinityALMarker.ps1"
    "tools\nsamdr\eve_asset_test.py"
    "tools\nsamdr\generate_strategy_candidates.py"
    "tools\nsamdr\gr2_converter\README.md"
    "tools\nsamdr\gr2_converter\package.json"
    "tools\nsamdr\gr2_converter\convert_eve_asset.mjs"
    "tools\nsamdr\gr2_converter\vendor\core-math-compat\package.json"
    "tools\nsamdr\gr2_converter\vendor\core-math-compat\mesh.js"
    "tools\nsamdr\gr2_converter\vendor\core-math-compat\num.js"
    "tools\nsamdr\gr2_converter\vendor\core-math-compat\tangent.js"
    "tools\nsamdr\gr2_converter\vendor\core-math-compat\vec3.js"
    "tools\nsamdr\neural\default_training_config.json"
    "tools\nsamdr\neural\requirements.txt"
    "tools\nsamdr\neural\train_nsamdr_kernel.py"
    "tools\nsamdr\neural\test_nsamdr_kernel.py"
    "tools\nsamdr\neural\verify_cuda.py"
    "tools\nsamdr\strategy_pipeline\model.py"
    "tools\nsamdr\tests\test_strategy_candidates.py"
    "trinityal\tests\nsamdr\README.md"
    "trinityal\tests\nsamdr\NSAMDRShipPreview.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreview.hlsl"
    "trinityal\tests\nsamdr\NSAMDRPreviewPlatform.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewTypes.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewTypes.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreviewUtilities.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewUtilities.cpp"
    "trinityal\tests\nsamdr\NSAMDRCameraController.h"
    "trinityal\tests\nsamdr\NSAMDRCameraController.cpp"
    "trinityal\tests\nsamdr\NSAMDRMeshProcessor.h"
    "trinityal\tests\nsamdr\NSAMDRMeshProcessor.cpp"
    "trinityal\tests\nsamdr\NSAMDRInputController.h"
    "trinityal\tests\nsamdr\NSAMDRInputController.cpp"
    "trinityal\tests\nsamdr\NSAMDRShaderLibrary.h"
    "trinityal\tests\nsamdr\NSAMDRShaderLibrary.cpp"
    "trinityal\tests\nsamdr\NSAMDRStrategyModes.h"
    "trinityal\tests\nsamdr\NSAMDRStrategyModes.cpp"
    "trinityal\tests\nsamdr\NSAMDRMode3Pipeline.h"
    "trinityal\tests\nsamdr\NSAMDRMode3Pipeline.cpp"
    "trinityal\tests\nsamdr\NSAMDRAssetProcessor.h"
    "trinityal\tests\nsamdr\NSAMDRAssetProcessor.cpp"
    "trinityal\tests\nsamdr\NSAMDRTrainingController.h"
    "trinityal\tests\nsamdr\NSAMDRTrainingController.cpp"
    "trinityal\tests\nsamdr\NSAMDRSceneController.h"
    "trinityal\tests\nsamdr\NSAMDRSceneController.cpp"
    "trinityal\tests\nsamdr\NSAMDRRenderPipeline.h"
    "trinityal\tests\nsamdr\NSAMDRRenderPipeline.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreviewRenderer.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewRenderer.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreviewProcessing.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewProcessing.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreviewPanel.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewPanel.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreviewApplication.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewApplication.cpp"
    "trinityal\tests\nsamdr\NSAMDRWindowIcon.h"
    "trinityal\tests\nsamdr\NSAMDRWindowIcon.cpp"
    "trinityal\tests\nsamdr\NSAMDRPreviewResource.h"
    "trinityal\tests\nsamdr\NSAMDRPreviewIcon.ico"
    "trinityal\tests\nsamdr\NSAMDRPreviewIcon.png"
) do call :RequireFile "%%~F"

for %%P in (
    "scripts\build\train_nsamdr_mode7.bat"
    "scripts\build\test_nsamdr_mode7.bat"
    "scripts\build\retrain_nsamdr_mode7_and_preview.bat"
    "tools\nsamdr\strategy_pipeline\mode6.py"
    "tools\nsamdr\strategy_pipeline\mode7.py"
    "trinityal\tests\nsamdr\NSAMDRMode6Pipeline.inl"
    "trinityal\tests\nsamdr\NSAMDRMode7Pipeline.inl"
    "trinityal\tests\nsamdr\NSAMDRNeuralRuntime.h"
    "trinityal\tests\nsamdr\NSAMDRNeuralRuntime.cpp"
    "trinityal\tests\nsamdr\NSAMDRNeuralWeights.hlsli"
) do call :RequireAbsent "%%~P"

for %%F in ("trinityal\tests\nsamdr\*.inl") do if exist "%%~F" (
    echo ERROR: .inl implementation remains: %%~F
    set /a FAILURES+=1
)

if !FAILURES! NEQ 0 (
    echo.
    echo NSAMDR layout verification FAILED: !FAILURES! problem^(s^).
    echo Re-extract the complete override into "%ROOT%" and rerun this script.
    popd
    exit /b 1
)

echo.
if "%CHECK_ONLY%"=="0" echo Removed invalid NSAMDR entries: !REMOVED!
echo NSAMDR layout verified successfully.
echo Correct roots:
echo   scripts\build
echo   tools\nsamdr
echo   trinityal\tests\nsamdr
popd
exit /b 0

:CleanNSAMDRSourceDirectory
for %%N in (
    "README.md"
    "NSAMDRShipPreview.cpp"
    "NSAMDRPreview.hlsl"
    "NSAMDRPreviewPlatform.h"
    "NSAMDRPreviewTypes.h"
    "NSAMDRPreviewTypes.cpp"
    "NSAMDRPreviewUtilities.h"
    "NSAMDRPreviewUtilities.cpp"
    "NSAMDRCameraController.h"
    "NSAMDRCameraController.cpp"
    "NSAMDRMeshProcessor.h"
    "NSAMDRMeshProcessor.cpp"
    "NSAMDRInputController.h"
    "NSAMDRInputController.cpp"
    "NSAMDRShaderLibrary.h"
    "NSAMDRShaderLibrary.cpp"
    "NSAMDRStrategyModes.h"
    "NSAMDRStrategyModes.cpp"
    "NSAMDRMode3Pipeline.h"
    "NSAMDRMode3Pipeline.cpp"
    "NSAMDRAssetProcessor.h"
    "NSAMDRAssetProcessor.cpp"
    "NSAMDRTrainingController.h"
    "NSAMDRTrainingController.cpp"
    "NSAMDRSceneController.h"
    "NSAMDRSceneController.cpp"
    "NSAMDRRenderPipeline.h"
    "NSAMDRRenderPipeline.cpp"
    "NSAMDRPreviewRenderer.h"
    "NSAMDRPreviewRenderer.cpp"
    "NSAMDRPreviewProcessing.h"
    "NSAMDRPreviewProcessing.cpp"
    "NSAMDRPreviewPanel.h"
    "NSAMDRPreviewPanel.cpp"
    "NSAMDRPreviewApplication.h"
    "NSAMDRPreviewApplication.cpp"
    "NSAMDRWindowIcon.h"
    "NSAMDRWindowIcon.cpp"
    "NSAMDRPreviewResource.h"
    "NSAMDRPreviewIcon.ico"
    "NSAMDRPreviewIcon.png"
) do set "NSAMDR_VALID_%%~N=1"

for %%F in ("trinityal\tests\nsamdr\*") do (
    if exist "%%~fF\" (
        if not defined NSAMDR_VALID_%%~nxF call :RemoveDirectory "%%~F"
    ) else (
        if not defined NSAMDR_VALID_%%~nxF call :RemoveFile "%%~F"
    )
)

for %%N in (
    "README.md"
    "NSAMDRShipPreview.cpp"
    "NSAMDRPreview.hlsl"
    "NSAMDRPreviewPlatform.h"
    "NSAMDRPreviewTypes.h"
    "NSAMDRPreviewTypes.cpp"
    "NSAMDRPreviewUtilities.h"
    "NSAMDRPreviewUtilities.cpp"
    "NSAMDRCameraController.h"
    "NSAMDRCameraController.cpp"
    "NSAMDRMeshProcessor.h"
    "NSAMDRMeshProcessor.cpp"
    "NSAMDRInputController.h"
    "NSAMDRInputController.cpp"
    "NSAMDRShaderLibrary.h"
    "NSAMDRShaderLibrary.cpp"
    "NSAMDRStrategyModes.h"
    "NSAMDRStrategyModes.cpp"
    "NSAMDRMode3Pipeline.h"
    "NSAMDRMode3Pipeline.cpp"
    "NSAMDRAssetProcessor.h"
    "NSAMDRAssetProcessor.cpp"
    "NSAMDRTrainingController.h"
    "NSAMDRTrainingController.cpp"
    "NSAMDRSceneController.h"
    "NSAMDRSceneController.cpp"
    "NSAMDRRenderPipeline.h"
    "NSAMDRRenderPipeline.cpp"
    "NSAMDRPreviewRenderer.h"
    "NSAMDRPreviewRenderer.cpp"
    "NSAMDRPreviewProcessing.h"
    "NSAMDRPreviewProcessing.cpp"
    "NSAMDRPreviewPanel.h"
    "NSAMDRPreviewPanel.cpp"
    "NSAMDRPreviewApplication.h"
    "NSAMDRPreviewApplication.cpp"
    "NSAMDRWindowIcon.h"
    "NSAMDRWindowIcon.cpp"
    "NSAMDRPreviewResource.h"
    "NSAMDRPreviewIcon.ico"
    "NSAMDRPreviewIcon.png"
) do set "NSAMDR_VALID_%%~N="
exit /b 0

:RemoveDirectory
if exist "%~1\" (
    echo Removing invalid directory: %~1
    rmdir /s /q "%~1"
    if exist "%~1\" (
        echo ERROR: Could not remove directory: %~1
        set /a FAILURES+=1
    ) else (
        set /a REMOVED+=1
    )
)
exit /b 0

:RemoveFile
if exist "%~1" (
    echo Removing obsolete file: %~1
    del /f /q "%~1"
    if exist "%~1" (
        echo ERROR: Could not remove file: %~1
        set /a FAILURES+=1
    ) else (
        set /a REMOVED+=1
    )
)
exit /b 0

:RequireFile
if not exist "%~1" (
    echo ERROR: Missing required file: %~1
    set /a FAILURES+=1
)
exit /b 0

:RequireAbsent
if exist "%~1" (
    echo ERROR: Invalid or obsolete path remains: %~1
    set /a FAILURES+=1
)
exit /b 0
