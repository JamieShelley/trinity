@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%" || exit /b 2

set "FORCE=0"
if /I "%~1"=="--force" set "FORCE=1"
if not "%~1"=="" if /I not "%~1"=="--force" (
    echo ERROR: unknown argument %~1
    echo Usage: setup_nsamdr_cuda.bat [--force]
    exit /b 2
)

set "BASE_PYTHON=python"
if defined NSAMDR_BOOTSTRAP_PYTHON set "BASE_PYTHON=%NSAMDR_BOOTSTRAP_PYTHON%"
set "ENV_DIR=%ROOT%\artifacts\nsamdr\python-env"
set "ENV_PYTHON=%ENV_DIR%\Scripts\python.exe"

if "%FORCE%"=="1" if exist "%ENV_DIR%\" (
    echo Removing existing NSAMDR CUDA environment...
    rmdir /s /q "%ENV_DIR%"
    if exist "%ENV_DIR%\" (
        echo ERROR: Could not remove "%ENV_DIR%".
        exit /b 2
    )
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo ERROR: nvidia-smi was not found. Install or repair the NVIDIA display driver first.
    exit /b 2
)

echo NVIDIA driver / GPU:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if errorlevel 1 exit /b %ERRORLEVEL%

if exist "%ENV_PYTHON%" (
    "%ENV_PYTHON%" -c "import numpy, PIL, torch" >nul 2>nul
    if not errorlevel 1 (
        "%ENV_PYTHON%" tools\nsamdr\neural\verify_cuda.py --device-index 0 --require-arch sm_120 >nul 2>nul
        if not errorlevel 1 goto ready
    )
) else (
    echo Creating dedicated NSAMDR Python environment:
    echo   %ENV_DIR%
    "%BASE_PYTHON%" -m venv "%ENV_DIR%"
    if errorlevel 1 exit /b %ERRORLEVEL%
)

"%ENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b %ERRORLEVEL%

"%ENV_PYTHON%" -m pip install -r tools\nsamdr\neural\requirements.txt
if errorlevel 1 exit /b %ERRORLEVEL%

echo Installing PyTorch 2.11.0 with the CUDA 12.8 runtime and Blackwell sm_120 support...
"%ENV_PYTHON%" -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b %ERRORLEVEL%

goto ready

:ready
"%ENV_PYTHON%" tools\nsamdr\neural\verify_cuda.py --device-index 0 --require-arch sm_120
if errorlevel 1 (
    echo.
    echo ERROR: NSAMDR CUDA setup did not pass verification.
    echo Ensure the NVIDIA driver supports the RTX 5080, then rerun this script.
    exit /b %ERRORLEVEL%
)
echo.
echo NSAMDR CUDA environment is ready.
echo Training scripts will automatically use:
echo   %ENV_PYTHON%
exit /b 0
