@echo off
call "%~dp0_build_config.bat" trinitydev dx11 ALL rebuild
exit /b %ERRORLEVEL%
