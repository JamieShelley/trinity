@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "ASSET_ID=%~1"
if not defined ASSET_ID set "ASSET_ID=cb1_t1"
set "SIZE=%~2"
if not defined SIZE set "SIZE=4096"
set "ASSET_DIR=%ROOT%\artifacts\nsamdr\eve_assets\%ASSET_ID%"
set "OBJ=%ASSET_DIR%\%ASSET_ID%.obj"
set "MATERIALS=%ASSET_DIR%\ship.materials.tsv"
set "OUTPUT=%ASSET_DIR%\strategy_candidates_%SIZE%"
set "PYTHON_CMD="
if defined NSAMDR_PYTHON_EXE if exist "%NSAMDR_PYTHON_EXE%" set "PYTHON_CMD=%NSAMDR_PYTHON_EXE%"
if not defined PYTHON_CMD if exist "%ROOT%\artifacts\nsamdr\python-env\Scripts\python.exe" set "PYTHON_CMD=%ROOT%\artifacts\nsamdr\python-env\Scripts\python.exe"
if not defined PYTHON_CMD if exist "%ROOT%\artifacts\nsamdr\python-env-cpu\Scripts\python.exe" set "PYTHON_CMD=%ROOT%\artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
if not defined PYTHON_CMD (
    echo ERROR: NSAMDR PyTorch environment was not found.
    echo Run scripts\build\setup_nsamdr_cuda.bat or setup_nsamdr_cpu.bat.
    exit /b 2
)
if not exist "%OBJ%" (
    echo ERROR: Missing OBJ: %OBJ%
    exit /b 3
)
if not exist "%MATERIALS%" (
    echo ERROR: Missing material manifest: %MATERIALS%
    exit /b 4
)

"%PYTHON_CMD%" "%ROOT%\tools\nsamdr\generate_strategy_candidates.py" ^
    --obj "%OBJ%" ^
    --materials "%MATERIALS%" ^
    --asset-manifest "%ASSET_DIR%\asset_manifest.json" ^
    --output-root "%OUTPUT%" ^
    --target-size "%SIZE%" ^
    --super-resolution-backend auto ^
    --inference-device auto ^
    --install-dependencies %3
exit /b %ERRORLEVEL%
