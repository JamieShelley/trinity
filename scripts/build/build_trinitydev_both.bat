@echo off
call "%~dp0_build_config.bat" trinitydev both ALL build
exit /b %ERRORLEVEL%
