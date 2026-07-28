@echo off
call "%~dp0_build_config.bat" internal dx12 ALL build
exit /b %ERRORLEVEL%
