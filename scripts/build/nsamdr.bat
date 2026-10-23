@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "PYTHON=%NSAMDR_PYTHON_EXE%"
if defined PYTHON if not exist "%PYTHON%" if exist "%ROOT%\%PYTHON%" set "PYTHON=%ROOT%\%PYTHON%"
if not defined PYTHON set "PYTHON=%ROOT%\artifacts\nsamdr\python-env\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%ROOT%\artifacts\nsamdr\python-env-cpu\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" -u "%ROOT%\tools\nsamdr\nsamdr_cli.py" %*
exit /b %ERRORLEVEL%
