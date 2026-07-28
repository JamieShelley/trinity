@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "SHARED_CACHE=%~1"
set "ASSET_QUERY=%~2"
if not defined ASSET_QUERY set "ASSET_QUERY=res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"
set "HELPER=%ROOT%\tools\nsamdr\eve_asset_test.py"
set "LAUNCHER=%~dp0run_nsamdr_obj_preview_dx11.bat"
set "PYTHON_CMD="

if not exist "%HELPER%" (
    echo ERROR: Missing EVE asset helper:
    echo   "%HELPER%"
    exit /b 2
)
if not exist "%LAUNCHER%" (
    echo ERROR: Missing NSAMDR OBJ preview launcher:
    echo   "%LAUNCHER%"
    exit /b 3
)

where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    where python.exe >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo ERROR: Python 3 was not found.
    exit /b 4
)

where node.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js 18 or newer is required for the open-source GR2/DDS conversion path.
    echo Install Node.js, reopen the terminal, and retry.
    exit /b 5
)

node -e "const m=Number(process.versions.node.split('.')[0]); process.exit(m>=18?0:1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js 18 or newer is required.
    node --version
    exit /b 6
)

echo ============================================================
echo NSAMDR REAL EVE ASSET TEST - GRANNY SDK FREE
echo ============================================================
echo Repository   : %ROOT%
if defined SHARED_CACHE (echo SharedCache  : %SHARED_CACHE%) else (echo SharedCache  : auto-detect)
echo Asset query : %ASSET_QUERY%
echo Geometry    : EVE GR2 via CarbonEngineJS pure-JS reader
echo Texture     : EVE DDS via CarbonEngineJS software decoder
echo Viewer      : TrinityAL DX11 OBJ preview
echo ============================================================

%PYTHON_CMD% "%HELPER%" prepare-run ^
    --repo-root "%ROOT%" ^
    --shared-cache "%SHARED_CACHE%" ^
    --query "%ASSET_QUERY%" ^
    --launcher "%LAUNCHER%"
exit /b %ERRORLEVEL%
