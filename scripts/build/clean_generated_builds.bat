@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

echo This removes only generated directories matching:
echo   "%ROOT%\.cmake-build-x64-windows-*"
echo.
set /P "CONFIRM=Type CLEAN to continue: "
if /I not "%CONFIRM%"=="CLEAN" (
    echo Cancelled.
    exit /b 1
)

for /D %%D in ("%ROOT%\.cmake-build-x64-windows-*") do (
    echo Removing "%%~fD"
    rmdir /S /Q "%%~fD"
    if errorlevel 1 (
        echo ERROR: Could not remove "%%~fD".
        exit /b 2
    )
)

echo Generated build directories removed.
exit /b 0
