@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "REPORT=%~1"
if not defined REPORT set "REPORT=%ROOT%\artifacts\nsamdr\eve_assets\ship.materials.report.json"
if not exist "%REPORT%" (
    echo ERROR: Baseline report not found:
    echo   %REPORT%
    echo Pass the generated ship.materials.report.json path explicitly.
    exit /b 2
)
where python.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: python.exe was not found in PATH.
    exit /b 3
)
python.exe "%ROOT%\tools\nsamdr\validate_baseline.py" "%REPORT%"
exit /b %ERRORLEVEL%
