@echo off
call "%~dp0_build_config.bat" debug dx12 ALL build
exit /b %ERRORLEVEL%
