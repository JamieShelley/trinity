@echo off
call "%~dp0_build_config.bat" trinitydev dx12 ALL build
exit /b %ERRORLEVEL%
