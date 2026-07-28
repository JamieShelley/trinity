@echo off
call "%~dp0_build_config.bat" internal dx11 ALL build
exit /b %ERRORLEVEL%
