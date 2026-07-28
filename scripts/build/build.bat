@echo off
call "%~dp0_build_config.bat" %*
exit /b %ERRORLEVEL%
