@echo off
call "%~dp0_build_config.bat" trinitydev dx12 ALL rebuild
exit /b %ERRORLEVEL%
