@echo off
call "%~dp0_build_config.bat" release dx12 ALL build
exit /b %ERRORLEVEL%
