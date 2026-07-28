@echo off
call "%~dp0_build_config.bat" trinitydev stub trinity_stub build
exit /b %ERRORLEVEL%
