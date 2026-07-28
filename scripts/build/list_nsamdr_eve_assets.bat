@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "SHARED_CACHE=%~1"
set "QUERY=%~2"
if not defined QUERY set "QUERY=cb1"
where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 "%ROOT%\tools\nsamdr\eve_asset_test.py" list --shared-cache "%SHARED_CACHE%" --query "%QUERY%" --limit 200
    exit /b %ERRORLEVEL%
)
python "%ROOT%\tools\nsamdr\eve_asset_test.py" list --shared-cache "%SHARED_CACHE%" --query "%QUERY%" --limit 200
exit /b %ERRORLEVEL%
