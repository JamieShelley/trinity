@echo off
setlocal EnableExtensions

call "%~dp0_build_config.bat" trinitydev dx11 ALL configure
if errorlevel 1 exit /b %ERRORLEVEL%

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cmake --build "%ROOT%\.cmake-build-x64-windows-trinitydev-dx11" --target help
exit /b %ERRORLEVEL%
