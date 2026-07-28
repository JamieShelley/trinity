@echo off
call "%~dp0_build_config.bat" trinitydev dx11 ALL build
exit /b %ERRORLEVEL%
