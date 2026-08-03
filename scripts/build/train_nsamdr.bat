@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\..\.."

call "%~dp0verify_and_clean_nsamdr_layout.bat" --check-only
if errorlevel 1 exit /b %ERRORLEVEL%

set "CONFIG=tools\nsamdr\neural\default_training_config.json"
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
echo Usage: train_nsamdr.bat [--config path] [--source-root path] [--device cuda^|cpu^|auto]
exit /b 2
:parsed

if not defined SOURCE_ROOT if defined NSAMDR_TRAINING_SOURCE_ROOT set "SOURCE_ROOT=%NSAMDR_TRAINING_SOURCE_ROOT%"
if not defined SOURCE_ROOT if defined NSAMDR_EVE_CACHE set "SOURCE_ROOT=%NSAMDR_EVE_CACHE%"
if not defined DEVICE if defined NSAMDR_TRAINING_DEVICE set "DEVICE=%NSAMDR_TRAINING_DEVICE%"

if defined SOURCE_ROOT if not exist "%SOURCE_ROOT%\" (
    echo ERROR: training source root does not exist:
    echo   %SOURCE_ROOT%
    echo Replace the README placeholder with the real EVE SharedCache or extracted texture directory.
    exit /b 2
)

if not defined DEVICE call :ChooseDevice
if errorlevel 1 exit /b %ERRORLEVEL%

if /I "%DEVICE%"=="gpu" set "DEVICE=cuda"
if /I not "%DEVICE%"=="cuda" if /I not "%DEVICE%"=="cpu" if /I not "%DEVICE%"=="auto" (
    echo ERROR: invalid training device "%DEVICE%". Use cuda, cpu, or auto.
    exit /b 2
)

if not exist "%CONFIG%" (
    echo ERROR: training config not found: %CONFIG%
    exit /b 2
)

call :ResolvePython
if errorlevel 1 exit /b %ERRORLEVEL%

echo.
echo ============================================================
echo NSAMDR TRAINING
echo Device : %DEVICE%
echo Python : %PYTHON%
if defined SOURCE_ROOT echo Corpus : %SOURCE_ROOT%
if not defined SOURCE_ROOT echo Corpus : synthetic training only
echo Config : %CONFIG%
echo ============================================================

"%PYTHON%" -c "import numpy, PIL, torch" >nul 2>nul
if errorlevel 1 (
    echo ERROR: neural training dependencies are missing from:
    echo   %PYTHON%
    if /I "%DEVICE%"=="cuda" echo Run: call scripts\build\setup_nsamdr_cuda.bat
    if /I "%DEVICE%"=="cpu" echo Run: call scripts\build\setup_nsamdr_cpu.bat
    exit /b 3
)

if /I "%DEVICE%"=="cuda" (
    "%PYTHON%" tools\nsamdr\neural\verify_cuda.py --device-index 0 --require-arch sm_120
    if errorlevel 1 (
        echo ERROR: CUDA verification failed. GPU training has not started.
        exit /b 4
    )
)

set "DEVICE_ARGS=--device %DEVICE%"
if defined SOURCE_ROOT (
    "%PYTHON%" tools\nsamdr\neural\train_nsamdr_kernel.py --repo-root "%CD%" --config "%CONFIG%" --source-root "%SOURCE_ROOT%" %DEVICE_ARGS%
) else (
    "%PYTHON%" tools\nsamdr\neural\train_nsamdr_kernel.py --repo-root "%CD%" --config "%CONFIG%" %DEVICE_ARGS%
)
if errorlevel 1 exit /b %ERRORLEVEL%

echo NSAMDR training complete.
exit /b 0

:ChooseDevice
echo.
echo ============================================================
echo SELECT NSAMDR TRAINING DEVICE
echo ============================================================
echo [1] NVIDIA GPU / CUDA  - recommended
echo     Requirements:
echo       - NVIDIA GPU with a working current driver and nvidia-smi
echo       - Internet access for the first environment setup
echo       - RTX 50-series uses the pinned CUDA 12.8 PyTorch wheel
echo       - The wheel must expose Blackwell target sm_120
echo       - Environment: artifacts\nsamdr\python-env
echo.
echo [2] CPU
echo     Requirements:
echo       - Working 64-bit Python installation
echo       - Internet access for the first environment setup
echo       - No NVIDIA driver or CUDA requirement
echo       - Much slower than GPU training
echo       - Environment: artifacts\nsamdr\python-env-cpu
echo.
set "DEVICE_CHOICE="
set /p "DEVICE_CHOICE=Select 1 or 2 [1]: "
if not defined DEVICE_CHOICE set "DEVICE_CHOICE=1"
if /I "%DEVICE_CHOICE%"=="1" (
    set "DEVICE=cuda"
    exit /b 0
)
if /I "%DEVICE_CHOICE%"=="gpu" (
    set "DEVICE=cuda"
    exit /b 0
)
if /I "%DEVICE_CHOICE%"=="cuda" (
    set "DEVICE=cuda"
    exit /b 0
)
if /I "%DEVICE_CHOICE%"=="2" (
    set "DEVICE=cpu"
    exit /b 0
)
if /I "%DEVICE_CHOICE%"=="cpu" (
    set "DEVICE=cpu"
    exit /b 0
)
echo ERROR: select 1 for GPU or 2 for CPU.
exit /b 2

:ResolvePython
if defined NSAMDR_PYTHON_EXE (
    set "PYTHON=%NSAMDR_PYTHON_EXE%"
    exit /b 0
)

if /I "%DEVICE%"=="cuda" (
    set "PYTHON=artifacts\nsamdr\python-env\Scripts\python.exe"
    if not exist "!PYTHON!" (
        echo.
        echo CUDA environment is missing. Running the required setup now...
        call scripts\build\setup_nsamdr_cuda.bat
        if errorlevel 1 exit /b !ERRORLEVEL!
    )
    "!PYTHON!" tools\nsamdr\neural\verify_cuda.py --device-index 0 --require-arch sm_120 >nul 2>nul
    if errorlevel 1 (
        echo.
        echo CUDA environment exists but failed verification. Repairing it now...
        call scripts\build\setup_nsamdr_cuda.bat --force
        if errorlevel 1 exit /b !ERRORLEVEL!
    )
    exit /b 0
)

if /I "%DEVICE%"=="cpu" (
    set "PYTHON=artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
    if not exist "!PYTHON!" (
        echo.
        echo CPU environment is missing. Running the required setup now...
        call scripts\build\setup_nsamdr_cpu.bat
        if errorlevel 1 exit /b !ERRORLEVEL!
    )
    "!PYTHON!" -c "import numpy, PIL, torch; x=torch.randn((64,64)); y=x@x; print(float(y.mean()))" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo CPU environment exists but failed verification. Repairing it now...
        call scripts\build\setup_nsamdr_cpu.bat --force
        if errorlevel 1 exit /b !ERRORLEVEL!
    )
    exit /b 0
)

rem device=auto: prefer a verified CUDA environment, otherwise use/create CPU.
set "PYTHON=artifacts\nsamdr\python-env\Scripts\python.exe"
if exist "!PYTHON!" (
    "!PYTHON!" tools\nsamdr\neural\verify_cuda.py --device-index 0 --require-arch sm_120 >nul 2>nul
    if not errorlevel 1 exit /b 0
)
set "DEVICE=cpu"
set "PYTHON=artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
if not exist "!PYTHON!" (
    call scripts\build\setup_nsamdr_cpu.bat
    if errorlevel 1 exit /b !ERRORLEVEL!
)
exit /b 0
