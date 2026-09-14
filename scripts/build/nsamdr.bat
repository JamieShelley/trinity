@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "PYTHON=%NSAMDR_PYTHON_EXE%"
if defined PYTHON if not exist "%PYTHON%" if exist "%ROOT%\%PYTHON%" set "PYTHON=%ROOT%\%PYTHON%"
if not defined PYTHON set "PYTHON=%ROOT%\artifacts\nsamdr\python-env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%ROOT%\artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
if /I "%~1"=="gui" (
    "%PYTHON%" -u "%ROOT%\tools\nsamdr\gui\nsamdr_v16_multifamily_workflow_gui.py"
    exit /b %ERRORLEVEL%
)
if /I "%~1"=="diversity" (
    shift
    "%PYTHON%" -u "%ROOT%\tools\nsamdr\neural\discover_nsamdr_v16_raven_diversity.py" --repo-root "%ROOT%" %*
    exit /b %ERRORLEVEL%
)
"%PYTHON%" -u "%ROOT%\tools\nsamdr\nsamdr_cli.py" %*
exit /b %ERRORLEVEL%
