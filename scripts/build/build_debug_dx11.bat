@echo off
call "%~dp0_build_config.bat" debug dx11 ALL build
exit /b %ERRORLEVEL%
