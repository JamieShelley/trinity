@echo off
call "%~dp0_build_config.bat" release dx11 ALL build
exit /b %ERRORLEVEL%
