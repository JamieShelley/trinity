@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%" || exit /b 2

call "%~dp0verify_and_clean_nsamdr_layout.bat"
if errorlevel 1 exit /b %ERRORLEVEL%

echo Removing obsolete NSAMDR generated intermediate-mode candidates...
if exist "artifacts\nsamdr\eve_assets" (
    for /d /r "artifacts\nsamdr\eve_assets" %%D in (
        mode4_*
        mode5_*
        mode6_*
        mode7_*
    ) do (
        if exist "%%~fD" rmdir /s /q "%%~fD"
    )
)

echo NSAMDR cleanup complete.
exit /b 0
