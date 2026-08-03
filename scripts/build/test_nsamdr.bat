@echo off
setlocal EnableExtensions
cd /d "%~dp0\..\.."

call "%~dp0verify_and_clean_nsamdr_layout.bat" --check-only
if errorlevel 1 exit /b %ERRORLEVEL%

set "METADATA=artifacts\nsamdr\neural\nsamdr_tile_context.json"
:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--metadata" (
    set "METADATA=%~2"
    shift
    shift
    goto parse
)
if /I "%~1"=="--config" (
    rem Accepted so the renderer workflow can pass one profile to every step.
    shift
    shift
    goto parse
)
echo ERROR: unknown argument %~1
exit /b 2
:parsed

if defined NSAMDR_PYTHON_EXE (
    set "PYTHON=%NSAMDR_PYTHON_EXE%"
) else if exist "artifacts\nsamdr\python-env\Scripts\python.exe" (
    set "PYTHON=artifacts\nsamdr\python-env\Scripts\python.exe"
) else if exist "artifacts\nsamdr\python-env-cpu\Scripts\python.exe" (
    set "PYTHON=artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if not exist "%METADATA%" (
    echo ERROR: trained model metadata not found: %METADATA%
    exit /b 3
)

"%PYTHON%" tools\nsamdr\neural\test_nsamdr_kernel.py --repo-root "%CD%" --metadata "%METADATA%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m unittest tools.nsamdr.tests.test_strategy_candidates
if errorlevel 1 exit /b %errorlevel%

echo NSAMDR neural tests passed.
exit /b 0
