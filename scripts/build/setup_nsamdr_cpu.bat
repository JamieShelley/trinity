@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%" || exit /b 2

set "FORCE=0"
if /I "%~1"=="--force" set "FORCE=1"
if not "%~1"=="" if /I not "%~1"=="--force" (
    echo ERROR: unknown argument %~1
    echo Usage: setup_nsamdr_cpu.bat [--force]
    exit /b 2
)

set "BASE_PYTHON=python"
if defined NSAMDR_BOOTSTRAP_PYTHON set "BASE_PYTHON=%NSAMDR_BOOTSTRAP_PYTHON%"
set "ENV_DIR=%ROOT%\artifacts\nsamdr\python-env-cpu"
set "ENV_PYTHON=%ENV_DIR%\Scripts\python.exe"

if "%FORCE%"=="1" if exist "%ENV_DIR%\" (
    echo Removing existing NSAMDR CPU environment...
    rmdir /s /q "%ENV_DIR%"
    if exist "%ENV_DIR%\" (
        echo ERROR: Could not remove "%ENV_DIR%".
        exit /b 2
    )
)

if exist "%ENV_PYTHON%" (
    "%ENV_PYTHON%" -c "import numpy, PIL, torch; x=torch.randn((128,128)); y=x@x; print('CPU matrix test passed:', float(y.mean()))"
    if not errorlevel 1 goto ready
)

echo Creating dedicated NSAMDR CPU Python environment:
echo   %ENV_DIR%
"%BASE_PYTHON%" -m venv "%ENV_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create the Python virtual environment.
    echo Install a working 64-bit Python distribution or set NSAMDR_BOOTSTRAP_PYTHON.
    exit /b %ERRORLEVEL%
)

"%ENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b %ERRORLEVEL%

"%ENV_PYTHON%" -m pip install -r tools\nsamdr\neural\requirements.txt
if errorlevel 1 exit /b %ERRORLEVEL%

echo Installing the pinned CPU-only PyTorch wheel...
"%ENV_PYTHON%" -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b %ERRORLEVEL%

:ready
"%ENV_PYTHON%" -c "import platform, torch; x=torch.randn((512,512)); y=x@x; print('Python:', platform.python_version()); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda or 'none'); print('CPU matrix test passed:', float(y.mean()))"
if errorlevel 1 (
    echo ERROR: NSAMDR CPU setup did not pass verification.
    exit /b %ERRORLEVEL%
)
echo.
echo NSAMDR CPU environment is ready.
echo Training scripts will use:
echo   %ENV_PYTHON%
exit /b 0
